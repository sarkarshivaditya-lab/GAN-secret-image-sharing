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
        description="Train a four-share GAN whose shares resemble TV static."
    )
    parser.add_argument("--train-images", type=int, default=10000)
    parser.add_argument("--test-images", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--generator-lr", type=float, default=1e-4)
    parser.add_argument("--discriminator-lr", type=float, default=2e-4)
    parser.add_argument("--static-weight", type=float, default=0.025)
    parser.add_argument("--statistics-weight", type=float, default=0.05)
    parser.add_argument("--contribution-weight", type=float, default=0.05)
    parser.add_argument("--discriminator-steps", type=int, default=2)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--checkpoint-dir", default="checkpoints/static_gan")
    parser.add_argument("--init-checkpoint-dir", default="")
    return parser.parse_args()


def load_state(path, device):
    return torch.load(path, map_location=device, weights_only=False)


def load_initialization(encoder, decoder, args, device):
    if not args.init_checkpoint_dir:
        print("Initialized from scratch")
        return

    directory = Path(args.init_checkpoint_dir)
    encoder_path = directory / "encoder_best.pth"
    decoder_path = directory / "decoder_best.pth"
    if not encoder_path.exists() or not decoder_path.exists():
        raise FileNotFoundError(
            "Initialization directory must contain encoder_best.pth and decoder_best.pth."
        )

    encoder.load_state_dict(load_state(encoder_path, device))
    decoder.load_state_dict(load_state(decoder_path, device))
    print(f"Initialized encoder from: {encoder_path}")
    print(f"Initialized decoder from: {decoder_path}")


def static_statistics_loss(shares):
    target_mean = 0.5
    target_std = 1.0 / (12.0 ** 0.5)
    total = 0.0

    for share in shares:
        mean = share.mean(dim=(1, 2, 3))
        std = share.flatten(1).std(dim=1, unbiased=False)
        horizontal = (share[:, :, :, 1:] - share[:, :, :, :-1]).pow(2).mean(dim=(1, 2, 3))
        vertical = (share[:, :, 1:, :] - share[:, :, :-1, :]).pow(2).mean(dim=(1, 2, 3))

        total = total + F.mse_loss(mean, torch.full_like(mean, target_mean))
        total = total + F.mse_loss(std, torch.full_like(std, target_std))
        total = total + 0.25 * F.mse_loss(
            horizontal,
            torch.full_like(horizontal, 1.0 / 6.0),
        )
        total = total + 0.25 * F.mse_loss(
            vertical,
            torch.full_like(vertical, 1.0 / 6.0),
        )

    return total / NUM_SHARES


def share_contribution_loss(decoder, shares):
    """Prevent the decoder from silently ignoring one or more share channels."""
    with torch.no_grad():
        combined = decoder(*shares).detach()

    contribution_targets = []
    for index in range(NUM_SHARES):
        masked = list(shares)
        masked[index] = torch.full_like(masked[index], 0.5)
        missing_reconstruction = decoder(*masked)
        contribution = F.mse_loss(missing_reconstruction, combined)
        contribution_targets.append(contribution)

    values = torch.stack(contribution_targets)
    target = values.mean().detach()
    return F.mse_loss(values, torch.full_like(values, target))


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
            real_targets = torch.ones_like(real_logits)
            fake_targets = torch.zeros_like(fake_logits)

            loss = 0.5 * (
                F.binary_cross_entropy_with_logits(real_logits, real_targets)
                + F.binary_cross_entropy_with_logits(fake_logits, fake_targets)
            )

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
    total = 0.0
    for discriminator, share in zip(discriminators, shares):
        logits = discriminator(share)
        total += F.binary_cross_entropy_with_logits(logits, torch.ones_like(logits))
    return total / NUM_SHARES


def evaluate(encoder, decoder, loader, device):
    encoder.eval()
    decoder.eval()
    total_mse = 0.0
    total_count = 0

    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            reconstruction = reconstruct(decoder, encoder(images))
            batch = images.shape[0]
            total_mse += F.mse_loss(reconstruction, images).item() * batch
            total_count += batch

    mse = total_mse / total_count
    return mse, psnr_from_mse(mse)


def save_best(encoder, decoder, discriminators, epoch, mse, psnr, args, device):
    directory = Path(args.checkpoint_dir)
    directory.mkdir(parents=True, exist_ok=True)
    torch.save(encoder.state_dict(), directory / "encoder_best.pth")
    torch.save(decoder.state_dict(), directory / "decoder_best.pth")
    for index, discriminator in enumerate(discriminators, start=1):
        torch.save(discriminator.state_dict(), directory / f"static_discriminator_{index}_best.pth")

    save_json(
        {
            "experiment": "static_gan",
            "best_epoch": epoch,
            "validation_mse": mse,
            "validation_psnr_db": psnr,
            "train_images": args.train_images,
            "test_images": args.test_images,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "image_size": args.image_size,
            "generator_lr": args.generator_lr,
            "discriminator_lr": args.discriminator_lr,
            "static_weight": args.static_weight,
            "statistics_weight": args.statistics_weight,
            "contribution_weight": args.contribution_weight,
            "discriminator_steps": args.discriminator_steps,
            "seed": args.seed,
            "init_checkpoint_dir": args.init_checkpoint_dir,
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
    print("TV-Static Share GAN")
    print(f"Train images: {args.train_images}")
    print(f"Test images: {args.test_images}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Static GAN weight: {args.static_weight}")
    print(f"Statistics weight: {args.statistics_weight}")
    print(f"Contribution weight: {args.contribution_weight}")
    print(f"Discriminator steps: {args.discriminator_steps}")
    print()

    train_loader, test_loader = build_cifar10_loaders(
        data_dir=args.data_dir,
        train_images=args.train_images,
        test_images=args.test_images,
        batch_size=args.batch_size,
        image_size=args.image_size,
        seed=args.seed,
    )

    encoder = ShareEncoder().to(device)
    decoder = ShareDecoder().to(device)
    discriminators = [StaticDiscriminator().to(device) for _ in range(NUM_SHARES)]

    load_initialization(encoder, decoder, args, device)

    generator_parameters = list(encoder.parameters()) + list(decoder.parameters())
    generator_optimizer = torch.optim.Adam(generator_parameters, lr=args.generator_lr, betas=(0.5, 0.999))
    discriminator_optimizers = [
        torch.optim.Adam(discriminator.parameters(), lr=args.discriminator_lr, betas=(0.5, 0.999))
        for discriminator in discriminators
    ]

    best_psnr = float("-inf")
    history = []

    for epoch in range(1, args.epochs + 1):
        encoder.train()
        decoder.train()
        totals = {
            "reconstruction": 0.0,
            "static": 0.0,
            "statistics": 0.0,
            "contribution": 0.0,
            "discriminator": 0.0,
            "discriminator_accuracy": 0.0,
            "generator": 0.0,
        }
        batches = 0

        for batch_index, (images, _) in enumerate(train_loader, start=1):
            images = images.to(device)
            shares = encoder(images)

            discriminator_loss, discriminator_accuracy = train_static_discriminators(
                discriminators,
                discriminator_optimizers,
                shares,
                args.discriminator_steps,
            )

            shares = encoder(images)
            reconstruction = reconstruct(decoder, shares)
            reconstruction_loss = F.mse_loss(reconstruction, images)
            static_loss = generator_static_loss(discriminators, shares)
            statistics_loss = static_statistics_loss(shares)
            contribution_loss = share_contribution_loss(decoder, shares)

            for discriminator in discriminators:
                for parameter in discriminator.parameters():
                    parameter.requires_grad = False
                discriminator.eval()

            static_loss = generator_static_loss(discriminators, shares)
            generator_loss = (
                reconstruction_loss
                + args.static_weight * static_loss
                + args.statistics_weight * statistics_loss
                + args.contribution_weight * contribution_loss
            )

            generator_optimizer.zero_grad(set_to_none=True)
            generator_loss.backward()
            torch.nn.utils.clip_grad_norm_(generator_parameters, args.grad_clip)
            generator_optimizer.step()

            for discriminator in discriminators:
                for parameter in discriminator.parameters():
                    parameter.requires_grad = True

            totals["reconstruction"] += reconstruction_loss.item()
            totals["static"] += static_loss.item()
            totals["statistics"] += statistics_loss.item()
            totals["contribution"] += contribution_loss.item()
            totals["discriminator"] += discriminator_loss
            totals["discriminator_accuracy"] += discriminator_accuracy
            totals["generator"] += generator_loss.item()
            batches += 1

            if batch_index == 1 or batch_index % 100 == 0:
                print(
                    f"Epoch {epoch}/{args.epochs} | Batch {batch_index}/{len(train_loader)} | "
                    f"Recon {reconstruction_loss.item():.6f} | "
                    f"Static {static_loss.item():.6f} | "
                    f"Stats {statistics_loss.item():.6f} | "
                    f"Contrib {contribution_loss.item():.6f} | "
                    f"D-acc {discriminator_accuracy * 100:.2f}%"
                )

        validation_mse, validation_psnr = evaluate(encoder, decoder, test_loader, device)
        row = {
            "epoch": epoch,
            "train_reconstruction_loss": totals["reconstruction"] / batches,
            "train_static_loss": totals["static"] / batches,
            "train_statistics_loss": totals["statistics"] / batches,
            "train_contribution_loss": totals["contribution"] / batches,
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

        if validation_psnr > best_psnr:
            best_psnr = validation_psnr
            save_best(encoder, decoder, discriminators, epoch, validation_mse, validation_psnr, args, device)
            print(f"Saved new best checkpoint: {best_psnr:.3f} dB")

    save_csv(history, Path(args.checkpoint_dir) / "training_log.csv")
    save_json(
        {"experiment": "static_gan", "best_validation_psnr_db": best_psnr, "history": history},
        Path(args.checkpoint_dir) / "training_history.json",
    )
    print("Training complete.")
    print(f"Best validation PSNR: {best_psnr:.3f} dB")
    print(f"Checkpoints: {args.checkpoint_dir}")


if __name__ == "__main__":
    main()
