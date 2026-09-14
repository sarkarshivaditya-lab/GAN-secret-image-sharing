import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from models.attacker import ShareAttacker
from models.decoder import ShareDecoder
from models.encoder import ShareEncoder
from models.privacy_discriminator import PrivacyDiscriminator
from project_utils import (
    NUM_SHARES,
    build_cifar10_loaders,
    checkpoint_paths,
    model_parameter_summary,
    psnr_from_mse,
    reconstruct,
    save_csv,
    save_json,
    seed_everything,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the privacy-aware four-share image-sharing model."
    )
    parser.add_argument("--train-images", type=int, default=10000)
    parser.add_argument("--test-images", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--generator-lr", type=float, default=1e-4)
    parser.add_argument("--attacker-lr", type=float, default=2e-4)
    parser.add_argument("--discriminator-lr", type=float, default=2e-4)
    parser.add_argument("--privacy-weight", type=float, default=0.10)
    parser.add_argument("--privacy-warmup-epochs", type=int, default=5)
    parser.add_argument("--gan-weight", type=float, default=0.01)
    parser.add_argument("--attacker-steps", type=int, default=2)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--checkpoint-dir", default="checkpoints/privacy_gan")
    parser.add_argument(
        "--init-checkpoint-dir",
        default="checkpoints/privacy_sweep/lambda_0.10",
    )
    return parser.parse_args()


def set_requires_grad(model, enabled):
    for parameter in model.parameters():
        parameter.requires_grad = enabled


def load_checkpoint_state(path, device):
    return torch.load(path, map_location=device, weights_only=False)


def load_optional_initialization(encoder, decoder, args, device):
    if not args.init_checkpoint_dir:
        return False

    directory = Path(args.init_checkpoint_dir)
    encoder_candidates = [directory / "encoder_best.pth", directory / "encoder.pth"]
    decoder_candidates = [directory / "decoder_best.pth", directory / "decoder.pth"]

    encoder_path = next((path for path in encoder_candidates if path.exists()), None)
    decoder_path = next((path for path in decoder_candidates if path.exists()), None)

    if encoder_path is None or decoder_path is None:
        raise FileNotFoundError(
            "Initial checkpoint directory must contain encoder_best.pth/decoder_best.pth "
            "or the legacy encoder.pth/decoder.pth pair."
        )

    encoder.load_state_dict(load_checkpoint_state(encoder_path, device))
    decoder.load_state_dict(load_checkpoint_state(decoder_path, device))
    print(f"Loaded encoder: {encoder_path}")
    print(f"Loaded decoder: {decoder_path}")
    return True


def create_attackers(device, lr):
    attackers = [ShareAttacker().to(device) for _ in range(NUM_SHARES)]
    optimizers = [torch.optim.Adam(attacker.parameters(), lr=lr) for attacker in attackers]
    return attackers, optimizers


def train_attackers(encoder, attackers, optimizers, images, steps):
    encoder.eval()
    with torch.no_grad():
        shares = encoder(images)

    total_loss = 0.0
    for share_index, attacker in enumerate(attackers):
        attacker.train()
        for _ in range(steps):
            attacked = attacker(shares[share_index])
            loss = F.mse_loss(attacked, images)
            optimizers[share_index].zero_grad(set_to_none=True)
            loss.backward()
            optimizers[share_index].step()
            total_loss += loss.item()
    return total_loss / (NUM_SHARES * steps)


def train_privacy_discriminator(discriminator, optimizer, images, shares):
    discriminator.train()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for share in shares:
        batch_size = images.shape[0]
        permutation = torch.randperm(batch_size, device=images.device)
        mismatched_images = images[permutation]

        positive_logits = discriminator(images, share.detach())
        negative_logits = discriminator(mismatched_images, share.detach())
        positive_targets = torch.ones_like(positive_logits)
        negative_targets = torch.zeros_like(negative_logits)

        positive_loss = F.binary_cross_entropy_with_logits(positive_logits, positive_targets)
        negative_loss = F.binary_cross_entropy_with_logits(negative_logits, negative_targets)
        loss = 0.5 * (positive_loss + negative_loss)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_correct += (positive_logits >= 0).sum().item()
        total_correct += (negative_logits < 0).sum().item()
        total_examples += 2 * batch_size

    return total_loss / NUM_SHARES, total_correct / total_examples


def calculate_generator_privacy_loss(discriminator, images, shares):
    total_loss = 0.0
    for share in shares:
        logits = discriminator(images, share)
        total_loss += F.binary_cross_entropy_with_logits(logits, torch.zeros_like(logits))
    return total_loss / NUM_SHARES


def evaluate_reconstruction(encoder, decoder, loader, device):
    encoder.eval()
    decoder.eval()
    total_mse = 0.0
    total_images = 0

    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            reconstructed = reconstruct(decoder, encoder(images))
            batch_size = images.shape[0]
            total_mse += F.mse_loss(reconstructed, images).item() * batch_size
            total_images += batch_size

    mse = total_mse / total_images
    return mse, psnr_from_mse(mse)


def effective_weight(target_weight, epoch, warmup_epochs):
    if target_weight <= 0.0 or warmup_epochs <= 0:
        return target_weight
    return target_weight * min(epoch / warmup_epochs, 1.0)


def save_best(
    encoder,
    decoder,
    discriminator,
    epoch,
    validation_mse,
    validation_psnr,
    args,
    device,
    effective_privacy_weight,
):
    directory = Path(args.checkpoint_dir)
    directory.mkdir(parents=True, exist_ok=True)
    paths = checkpoint_paths(directory)

    torch.save(encoder.state_dict(), paths["encoder"])
    torch.save(decoder.state_dict(), paths["decoder"])
    torch.save(discriminator.state_dict(), paths["discriminator"])

    save_json(
        {
            "experiment": "privacy_gan",
            "best_epoch": epoch,
            "validation_mse": validation_mse,
            "validation_psnr_db": validation_psnr,
            "device": str(device),
            "train_images": args.train_images,
            "test_images": args.test_images,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "image_size": args.image_size,
            "generator_lr": args.generator_lr,
            "attacker_lr": args.attacker_lr,
            "discriminator_lr": args.discriminator_lr,
            "privacy_weight": args.privacy_weight,
            "privacy_warmup_epochs": args.privacy_warmup_epochs,
            "effective_privacy_weight_at_best_epoch": effective_privacy_weight,
            "gan_weight": args.gan_weight,
            "attacker_steps": args.attacker_steps,
            "seed": args.seed,
            "init_checkpoint_dir": args.init_checkpoint_dir,
            "parameter_counts": model_parameter_summary(
                encoder,
                decoder,
                ShareAttacker().to(device),
                discriminator,
            ),
        },
        paths["metadata"],
    )


def main():
    args = parse_args()
    if args.epochs < 1 or args.attacker_steps < 1:
        raise ValueError("epochs and attacker-steps must be positive.")
    if args.privacy_warmup_epochs < 0:
        raise ValueError("privacy-warmup-epochs cannot be negative.")

    seed_everything(args.seed)
    device = (
        torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    print(f"Device: {device}")
    print("Privacy-GAN training")
    print(f"Train images: {args.train_images}")
    print(f"Test images: {args.test_images}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Target privacy weight: {args.privacy_weight}")
    print(f"Privacy warmup epochs: {args.privacy_warmup_epochs}")
    print(f"GAN weight: {args.gan_weight}")
    print(f"Initialization: {args.init_checkpoint_dir or 'scratch'}")
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
    discriminator = PrivacyDiscriminator().to(device)

    initialized = load_optional_initialization(encoder, decoder, args, device)
    if initialized:
        print(f"Initialized from: {args.init_checkpoint_dir}")
    else:
        print("Initialized from scratch")

    attackers, attacker_optimizers = create_attackers(device, args.attacker_lr)
    generator_optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=args.generator_lr,
    )
    discriminator_optimizer = torch.optim.Adam(
        discriminator.parameters(),
        lr=args.discriminator_lr,
    )

    best_psnr = float("-inf")
    history = []

    for epoch in range(1, args.epochs + 1):
        encoder.train()
        decoder.train()
        current_privacy_weight = effective_weight(
            args.privacy_weight,
            epoch,
            args.privacy_warmup_epochs,
        )

        totals = {
            "reconstruction": 0.0,
            "attack": 0.0,
            "privacy": 0.0,
            "discriminator": 0.0,
            "discriminator_accuracy": 0.0,
            "generator": 0.0,
        }
        batches = 0

        for batch_index, (images, _) in enumerate(train_loader, start=1):
            images = images.to(device)

            train_attackers(
                encoder,
                attackers,
                attacker_optimizers,
                images,
                args.attacker_steps,
            )

            shares = encoder(images)
            discriminator_loss, discriminator_accuracy = train_privacy_discriminator(
                discriminator,
                discriminator_optimizer,
                images,
                shares,
            )

            shares = encoder(images)
            reconstructed = reconstruct(decoder, shares)
            reconstruction_loss = F.mse_loss(reconstructed, images)

            attack_loss = 0.0
            for share_index, attacker in enumerate(attackers):
                attacker.eval()
                attack_loss += F.mse_loss(attacker(shares[share_index]), images)
            attack_loss = attack_loss / NUM_SHARES

            set_requires_grad(discriminator, False)
            discriminator.eval()
            for attacker in attackers:
                set_requires_grad(attacker, False)
                attacker.eval()

            privacy_loss = calculate_generator_privacy_loss(
                discriminator,
                images,
                shares,
            )
            generator_loss = (
                reconstruction_loss
                - current_privacy_weight * attack_loss
                + args.gan_weight * privacy_loss
            )

            generator_optimizer.zero_grad(set_to_none=True)
            generator_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(decoder.parameters()),
                args.grad_clip,
            )
            generator_optimizer.step()

            set_requires_grad(discriminator, True)
            for attacker in attackers:
                set_requires_grad(attacker, True)

            totals["reconstruction"] += reconstruction_loss.item()
            totals["attack"] += attack_loss.item()
            totals["privacy"] += privacy_loss.item()
            totals["discriminator"] += discriminator_loss
            totals["discriminator_accuracy"] += discriminator_accuracy
            totals["generator"] += generator_loss.item()
            batches += 1

            if batch_index == 1 or batch_index % 100 == 0:
                print(
                    f"Epoch {epoch}/{args.epochs} | "
                    f"Batch {batch_index}/{len(train_loader)} | "
                    f"Recon {reconstruction_loss.item():.6f} | "
                    f"Attack {attack_loss.item():.6f} | "
                    f"Privacy {privacy_loss.item():.6f} | "
                    f"Priv-w {current_privacy_weight:.4f} | "
                    f"D {discriminator_loss:.6f} | "
                    f"D-acc {discriminator_accuracy * 100:.2f}%"
                )

        validation_mse, validation_psnr = evaluate_reconstruction(
            encoder,
            decoder,
            test_loader,
            device,
        )
        row = {
            "epoch": epoch,
            "effective_privacy_weight": current_privacy_weight,
            "train_reconstruction_loss": totals["reconstruction"] / batches,
            "train_attack_loss": totals["attack"] / batches,
            "train_privacy_loss": totals["privacy"] / batches,
            "train_discriminator_loss": totals["discriminator"] / batches,
            "train_discriminator_accuracy": totals["discriminator_accuracy"] / batches,
            "train_generator_loss": totals["generator"] / batches,
            "validation_mse": validation_mse,
            "validation_psnr_db": validation_psnr,
        }
        history.append(row)

        print()
        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"Privacy weight {current_privacy_weight:.4f} | "
            f"Validation MSE {validation_mse:.6f} | "
            f"Validation PSNR {validation_psnr:.3f} dB"
        )

        if validation_psnr > best_psnr:
            best_psnr = validation_psnr
            save_best(
                encoder,
                decoder,
                discriminator,
                epoch,
                validation_mse,
                validation_psnr,
                args,
                device,
                current_privacy_weight,
            )
            print(f"Saved new best checkpoint: {best_psnr:.3f} dB")

    save_csv(history, Path(args.checkpoint_dir) / "training_log.csv")
    save_json(
        {
            "experiment": "privacy_gan",
            "best_validation_psnr_db": best_psnr,
            "history": history,
        },
        Path(args.checkpoint_dir) / "training_history.json",
    )

    print("Training complete.")
    print(f"Best validation PSNR: {best_psnr:.3f} dB")
    print(f"Checkpoints: {args.checkpoint_dir}")


if __name__ == "__main__":
    main()
