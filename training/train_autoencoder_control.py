import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from models.decoder import ShareDecoder
from models.encoder import ShareEncoder
from project_utils import (
    build_cifar10_loaders,
    get_device,
    psnr_from_mse,
    save_csv,
    save_json,
    seed_everything,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the same payload encoder and decoder without the secret-sharing path."
    )
    parser.add_argument("--train-images", type=int, default=10000)
    parser.add_argument("--validation-images", type=int, default=1000)
    parser.add_argument("--test-images", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--l1-weight", type=float, default=0.10)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--checkpoint-dir", default="checkpoints/autoencoder_control")
    return parser.parse_args()


def reconstruction_loss(reconstruction, images, l1_weight):
    mse = F.mse_loss(reconstruction, images)
    l1 = F.l1_loss(reconstruction, images)
    return mse + l1_weight * l1, mse, l1


def evaluate(encoder, decoder, loader, device):
    encoder.eval()
    decoder.eval()
    total_mse = 0.0
    total_count = 0

    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            payload = encoder.encode_payload(images)
            reconstruction = decoder.decode_payload(payload)
            batch = images.shape[0]
            total_mse += F.mse_loss(reconstruction, images).item() * batch
            total_count += batch

    mse = total_mse / total_count
    return mse, psnr_from_mse(mse)


def save_checkpoint(encoder, decoder, directory, prefix):
    torch.save(encoder.state_dict(), directory / f"encoder_{prefix}.pth")
    torch.save(decoder.state_dict(), directory / f"decoder_{prefix}.pth")


def save_best(encoder, decoder, epoch, mse, psnr, args, device, prefix):
    directory = Path(args.checkpoint_dir)
    directory.mkdir(parents=True, exist_ok=True)
    save_checkpoint(encoder, decoder, directory, prefix)
    save_json(
        {
            "experiment": "autoencoder_control",
            "purpose": "direct_payload_encoder_decoder_control_without_secret_sharing_or_discriminators",
            "best_epoch": epoch,
            f"{prefix}_mse": mse,
            f"{prefix}_psnr_db": psnr,
            "train_images": args.train_images,
            "validation_images": args.validation_images,
            "test_images": args.test_images,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "image_size": args.image_size,
            "learning_rate": args.learning_rate,
            "l1_weight": args.l1_weight,
            "grad_clip": args.grad_clip,
            "seed": args.seed,
            "device": str(device),
        },
        directory / f"{prefix}_info.json",
    )


def main():
    args = parse_args()
    if args.epochs < 1:
        raise ValueError("epochs must be positive")

    seed_everything(args.seed)
    device = get_device()
    print(f"Device: {device}")
    print("Direct Autoencoder Control")
    print(f"Train images: {args.train_images}")
    print(f"Validation images: {args.validation_images}")
    print(f"Test images: {args.test_images}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"L1 weight: {args.l1_weight}")
    print("Path: image -> learned payload -> reconstruction")
    print("Secret-sharing masks: disabled")
    print("Discriminators: disabled")
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
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=args.learning_rate,
        betas=(0.5, 0.999),
    )

    best_train_psnr = float("-inf")
    best_validation_psnr = float("-inf")
    history = []

    for epoch in range(1, args.epochs + 1):
        encoder.train()
        decoder.train()
        totals = {"loss": 0.0, "mse": 0.0, "l1": 0.0}
        batches = 0

        for images, _ in train_loader:
            images = images.to(device)
            payload = encoder.encode_payload(images)
            reconstruction = decoder.decode_payload(payload)
            total_loss, mse_loss, l1_loss = reconstruction_loss(
                reconstruction,
                images,
                args.l1_weight,
            )

            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    list(encoder.parameters()) + list(decoder.parameters()),
                    args.grad_clip,
                )
            optimizer.step()

            totals["loss"] += total_loss.item()
            totals["mse"] += mse_loss.item()
            totals["l1"] += l1_loss.item()
            batches += 1

        train_mse, train_psnr = evaluate(encoder, decoder, train_loader, device)
        validation_mse, validation_psnr = evaluate(
            encoder,
            decoder,
            validation_loader,
            device,
        )

        row = {
            "epoch": epoch,
            "train_loss": totals["loss"] / batches,
            "train_batch_mse": totals["mse"] / batches,
            "train_batch_l1": totals["l1"] / batches,
            "train_mse": train_mse,
            "train_psnr_db": train_psnr,
            "validation_mse": validation_mse,
            "validation_psnr_db": validation_psnr,
        }
        history.append(row)

        if train_psnr > best_train_psnr:
            best_train_psnr = train_psnr
            save_best(
                encoder,
                decoder,
                epoch,
                train_mse,
                train_psnr,
                args,
                device,
                "train_best",
            )

        if validation_psnr > best_validation_psnr:
            best_validation_psnr = validation_psnr
            save_best(
                encoder,
                decoder,
                epoch,
                validation_mse,
                validation_psnr,
                args,
                device,
                "validation_best",
            )

        print(
            f"Epoch {epoch:03d}/{args.epochs:03d} | "
            f"loss={row['train_loss']:.6f} | "
            f"train PSNR={train_psnr:.3f} dB | "
            f"validation PSNR={validation_psnr:.3f} dB"
        )

    directory = Path(args.checkpoint_dir)
    directory.mkdir(parents=True, exist_ok=True)
    save_csv(history, directory / "training_history.csv")
    save_json(
        {
            "experiment": "autoencoder_control",
            "best_train_psnr_db": best_train_psnr,
            "best_validation_psnr_db": best_validation_psnr,
            "train_images": args.train_images,
            "validation_images": args.validation_images,
            "test_images": args.test_images,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "image_size": args.image_size,
            "learning_rate": args.learning_rate,
            "l1_weight": args.l1_weight,
            "grad_clip": args.grad_clip,
            "seed": args.seed,
            "device": str(device),
        },
        directory / "experiment_summary.json",
    )

    print()
    print("Training complete.")
    print(f"Best training PSNR: {best_train_psnr:.3f} dB")
    print(f"Best validation PSNR: {best_validation_psnr:.3f} dB")
    print(f"Checkpoints: {args.checkpoint_dir}")


if __name__ == "__main__":
    main()
