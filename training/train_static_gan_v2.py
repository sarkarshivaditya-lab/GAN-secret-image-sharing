import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from models.decoder import ShareDecoder
from models.encoder import ShareEncoder
from models.static_discriminator import StaticDiscriminator
from project_utils import (
    NUM_SHARES,
    build_cifar10_loaders,
    get_device,
    psnr_from_mse,
    reconstruct,
    save_csv,
    save_json,
    seed_everything,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a learned image decoder with four mathematically noise-like shares."
    )
    parser.add_argument("--train-images", type=int, default=10000)
    parser.add_argument("--validation-images", type=int, default=1000)
    parser.add_argument("--test-images", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--generator-lr", type=float, default=1e-4)
    parser.add_argument("--discriminator-lr", type=float, default=2e-4)
    parser.add_argument("--static-weight", type=float, default=0.001)
    parser.add_argument("--l1-weight", type=float, default=0.10)
    parser.add_argument("--discriminator-steps", type=int, default=1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--checkpoint-dir", default="checkpoints/static_gan")
    return parser.parse_args()


def hinge_discriminator_loss(real_logits, fake_logits):
    return F.relu(1.0 - real_logits).mean() + F.relu(1.0 + fake_logits).mean()


def train_static_discriminators(discriminators, optimizers, shares, steps):
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for discriminator, optimizer, share in zip(discriminators, optimizers, shares):
        for _ in range(steps):
            discriminator.train()
            real_static = torch.rand_like(share)
            real_logits = discriminator(real_static)
            fake_logits = discriminator(share.detach())
            loss = hinge_discriminator_loss(real_logits, fake_logits)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_correct += (real_logits >= 0).sum().item()
            total_correct += (fake_logits < 0).sum().item()
            total_examples += 2 * share.shape[0]

    divisor = NUM_SHARES * steps
    return total_loss / divisor, total_correct / total_examples


def generator_static_loss(discriminators, shares):
    return sum(-discriminator(share).mean() for discriminator, share in zip(discriminators, shares)) / NUM_SHARES


def reconstruction_loss(reconstruction, images, l1_weight):
    mse = F.mse_loss(reconstruction, images)
    l1 = F.l1_loss(reconstruction, images)
    return mse + l1_weight * l1, mse, l1


def static_statistics(shares):
    target_std = 1.0 / (12.0 ** 0.5)
    metrics = []
    for share in shares:
        mean = share.mean().item()
        std = share.flatten(1).std(unbiased=False).mean().item()
        horizontal = (share[:, :, :, 1:] - share[:, :, :, :-1]).pow(2).mean().item()
        vertical = (share[:, :, 1:, :] - share[:, :, :-1, :]).pow(2).mean().item()
        metrics.append(
            {
                "mean": mean,
                "std": std,
                "mean_error": abs(mean - 0.5),
                "std_error": abs(std - target_std),
                "horizontal_difference_mse": horizontal,
                "vertical_difference_mse": vertical,
            }
        )
    return metrics


def evaluate(encoder, decoder, loader, device):
    encoder.eval()
    decoder.eval()
    total_mse = 0.0
    total_count = 0
    static_totals = [
        {"mean": 0.0, "std": 0.0, "horizontal_difference_mse": 0.0, "vertical_difference_mse": 0.0}
        for _ in range(NUM_SHARES)
    ]

    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            shares = encoder(images)
            reconstruction = reconstruct(decoder, shares)
            batch = images.shape[0]
            total_mse += F.mse_loss(reconstruction, images).item() * batch
            total_count += batch
            for index, metrics in enumerate(static_statistics(shares)):
                for key in static_totals[index]:
                    static_totals[index][key] += metrics[key] * batch

    mse = total_mse / total_count
    for metrics in static_totals:
        for key in metrics:
            metrics[key] /= total_count
    return mse, psnr_from_mse(mse), static_totals


def save_best(encoder, decoder, discriminators, epoch, mse, psnr, static_metrics, args, device):
    directory = Path(args.checkpoint_dir)
    directory.mkdir(parents=True, exist_ok=True)
    torch.save(encoder.state_dict(), directory / "encoder_best.pth")
    torch.save(decoder.state_dict(), directory / "decoder_best.pth")
    for index, discriminator in enumerate(discriminators, start=1):
        torch.save(discriminator.state_dict(), directory / f"static_discriminator_{index}_best.pth")

    save_json(
        {
            "experiment": "static_gan_v2",
            "share_scheme": "three_independent_uniform_masks_plus_modular_payload_share",
            "best_epoch": epoch,
            "validation_mse": mse,
            "validation_psnr_db": psnr,
            "validation_static_metrics": static_metrics,
            "train_images": args.train_images,
            "validation_images": args.validation_images,
            "test_images": args.test_images,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "image_size": args.image_size,
            "generator_lr": args.generator_lr,
            "discriminator_lr": args.discriminator_lr,
            "static_weight": args.static_weight,
            "l1_weight": args.l1_weight,
            "discriminator_steps": args.discriminator_steps,
            "grad_clip": args.grad_clip,
            "seed": args.seed,
            "device": str(device),
        },
        directory / "best_info.json",
    )


def main():
    args = parse_args()
    if args.epochs < 1 or args.discriminator_steps < 1:
        raise ValueError("epochs and discriminator-steps must be positive")

    seed_everything(args.seed)
    device = get_device()
    print(f"Device: {device}")
    print("TV-Static Share GAN v2")
    print(f"Train images: {args.train_images}")
    print(f"Validation images: {args.validation_images}")
    print(f"Test images: {args.test_images}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Static GAN weight: {args.static_weight}")
    print(f"L1 weight: {args.l1_weight}")
    print(f"Discriminator steps: {args.discriminator_steps}")
    print("Share construction: three independent uniform masks plus one modular payload share")
    print("Reconstruction resolution: native CIFAR-10 32x32")
    print("Initialization: scratch")
    print()

    train_loader, validation_loader, _ = build_cifar10_loaders(
        data_dir=args.data_dir,
        train_images=args.train_images,
        validation_images=args.validation_images,
        test_images=args.test_images,
        batch_size=args.batch_size,
        image_size=args.image_size,
        seed=args.seed,
    )

    encoder = ShareEncoder().to(device)
    decoder = ShareDecoder().to(device)
    discriminators = [StaticDiscriminator().to(device) for _ in range(NUM_SHARES)]

    generator_parameters = list(encoder.parameters()) + list(decoder.parameters())
    generator_optimizer = torch.optim.Adam(
        generator_parameters,
        lr=args.generator_lr,
        betas=(0.5, 0.999),
    )
    discriminator_optimizers = [
        torch.optim.Adam(
            discriminator.parameters(),
            lr=args.discriminator_lr,
            betas=(0.5, 0.999),
        )
        for discriminator in discriminators
    ]

    best_psnr = float("-inf")
    history = []

    for epoch in range(1, args.epochs + 1):
        encoder.train()
        decoder.train()
        totals = {
            "reconstruction": 0.0,
            "mse": 0.0,
            "l1": 0.0,
            "static": 0.0,
            "discriminator": 0.0,
            "discriminator_accuracy": 0.0,
            "generator": 0.0,
        }
        batches = 0

        for batch_index, (images, _) in enumerate(train_loader, start=1):
            images = images.to(device)

            with torch.no_grad():
                discriminator_shares = encoder(images)
            discriminator_loss, discriminator_accuracy = train_static_discriminators(
                discriminators,
                discriminator_optimizers,
                discriminator_shares,
                args.discriminator_steps,
            )

            shares = encoder(images)
            reconstruction = reconstruct(decoder, shares)
            total_recon, mse_loss, l1_loss = reconstruction_loss(
                reconstruction,
                images,
                args.l1_weight,
            )

            for discriminator in discriminators:
                for parameter in discriminator.parameters():
                    parameter.requires_grad = False
                discriminator.eval()

            static_loss = generator_static_loss(discriminators, shares)
            generator_loss = total_recon + args.static_weight * static_loss

            generator_optimizer.zero_grad(set_to_none=True)
            generator_loss.backward()
            torch.nn.utils.clip_grad_norm_(generator_parameters, args.grad_clip)
            generator_optimizer.step()

            for discriminator in discriminators:
                for parameter in discriminator.parameters():
                    parameter.requires_grad = True

            totals["reconstruction"] += total_recon.item()
            totals["mse"] += mse_loss.item()
            totals["l1"] += l1_loss.item()
            totals["static"] += static_loss.item()
            totals["discriminator"] += discriminator_loss
            totals["discriminator_accuracy"] += discriminator_accuracy
            totals["generator"] += generator_loss.item()
            batches += 1

            if batch_index == 1 or batch_index % 100 == 0:
                print(
                    f"Epoch {epoch}/{args.epochs} | Batch {batch_index}/{len(train_loader)} | "
                    f"Recon {mse_loss.item():.6f} | L1 {l1_loss.item():.6f} | "
                    f"Static {static_loss.item():.6f} | "
                    f"D-acc {discriminator_accuracy * 100:.2f}%"
                )

        validation_mse, validation_psnr, validation_static = evaluate(
            encoder,
            decoder,
            validation_loader,
            device,
        )
        row = {
            "epoch": epoch,
            "train_reconstruction_loss": totals["reconstruction"] / batches,
            "train_mse": totals["mse"] / batches,
            "train_l1": totals["l1"] / batches,
            "train_static_loss": totals["static"] / batches,
            "train_discriminator_loss": totals["discriminator"] / batches,
            "train_discriminator_accuracy": totals["discriminator_accuracy"] / batches,
            "train_generator_loss": totals["generator"] / batches,
            "validation_mse": validation_mse,
            "validation_psnr_db": validation_psnr,
        }
        history.append(row)

        print(
            f"Epoch {epoch}/{args.epochs} | Validation MSE {validation_mse:.6f} | "
            f"Validation PSNR {validation_psnr:.3f} dB"
        )
        for index, metrics in enumerate(validation_static, start=1):
            print(
                f"Share {index} | mean {metrics['mean']:.4f} | std {metrics['std']:.4f} | "
                f"H-diff {metrics['horizontal_difference_mse']:.4f} | "
                f"V-diff {metrics['vertical_difference_mse']:.4f}"
            )

        if validation_psnr > best_psnr:
            best_psnr = validation_psnr
            save_best(
                encoder,
                decoder,
                discriminators,
                epoch,
                validation_mse,
                validation_psnr,
                validation_static,
                args,
                device,
            )
            print(f"Saved new best checkpoint: {best_psnr:.3f} dB")

    directory = Path(args.checkpoint_dir)
    save_csv(history, directory / "training_log.csv")
    save_json(
        {
            "experiment": "static_gan_v2",
            "share_scheme": "three_independent_uniform_masks_plus_modular_payload_share",
            "best_validation_psnr_db": best_psnr,
            "history": history,
        },
        directory / "training_history.json",
    )
    print("Training complete.")
    print(f"Best validation PSNR: {best_psnr:.3f} dB")
    print(f"Checkpoints: {args.checkpoint_dir}")


if __name__ == "__main__":
    main()
