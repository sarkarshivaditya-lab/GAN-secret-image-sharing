import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image

from models.decoder import ShareDecoder
from models.encoder import ShareEncoder
from project_utils import build_cifar10_loaders, get_device, reconstruct, seed_everything


NUM_SHARES = 4


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a TV-static share GAN.")
    parser.add_argument("--checkpoint-dir", default="checkpoints/static_gan")
    parser.add_argument("--split", choices=("train", "validation", "test"), default="test")
    parser.add_argument("--train-images", type=int, default=10000)
    parser.add_argument("--validation-images", type=int, default=1000)
    parser.add_argument("--test-images", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="outputs/static_gan")
    parser.add_argument("--samples", type=int, default=8)
    return parser.parse_args()


def load(model, path, device):
    model.load_state_dict(torch.load(path, map_location=device, weights_only=False))
    model.eval()
    return model


def static_metrics(share):
    flat = share.flatten(1)
    mean = flat.mean(dim=1)
    std = flat.std(dim=1, unbiased=False)
    horizontal = (share[:, :, :, 1:] - share[:, :, :, :-1]).pow(2).mean(dim=(1, 2, 3))
    vertical = (share[:, :, 1:, :] - share[:, :, :-1, :]).pow(2).mean(dim=(1, 2, 3))
    return {
        "mean": mean.mean().item(),
        "std": std.mean().item(),
        "mean_abs_deviation_from_half": (mean - 0.5).abs().mean().item(),
        "std_abs_deviation_from_uniform": (std - (1.0 / (12.0 ** 0.5))).abs().mean().item(),
        "horizontal_difference_mse": horizontal.mean().item(),
        "vertical_difference_mse": vertical.mean().item(),
        "target_horizontal_difference_mse": 1.0 / 6.0,
        "target_vertical_difference_mse": 1.0 / 6.0,
    }


def masked_reconstruction(decoder, shares, missing_index):
    masked = list(shares)
    masked[missing_index] = torch.zeros_like(masked[missing_index])
    return decoder(*masked)


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = get_device()
    checkpoint_dir = Path(args.checkpoint_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    encoder = load(ShareEncoder().to(device), checkpoint_dir / "encoder_best.pth", device)
    decoder = load(ShareDecoder().to(device), checkpoint_dir / "decoder_best.pth", device)

    train_loader, validation_loader, test_loader = build_cifar10_loaders(
        data_dir=args.data_dir,
        train_images=args.train_images,
        validation_images=args.validation_images,
        test_images=args.test_images,
        batch_size=args.batch_size,
        image_size=args.image_size,
        seed=args.seed,
    )

    loaders = {
        "train": train_loader,
        "validation": validation_loader,
        "test": test_loader,
    }
    counts = {
        "train": args.train_images,
        "validation": args.validation_images,
        "test": args.test_images,
    }
    loader = loaders[args.split]
    sample_count = counts[args.split]

    if sample_count < 1:
        raise ValueError("The selected split must contain at least one image")
    if args.samples < 1:
        raise ValueError("samples must be positive")

    total_mse = 0.0
    total_count = 0
    first_images = None
    first_shares = None
    first_reconstruction = None
    missing_share_mse = [0.0] * NUM_SHARES
    missing_share_count = 0
    aggregate_static = [[] for _ in range(NUM_SHARES)]

    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            shares = encoder(images)
            reconstruction = reconstruct(decoder, shares)
            batch = images.shape[0]
            total_mse += F.mse_loss(reconstruction, images).item() * batch
            total_count += batch

            for index in range(NUM_SHARES):
                masked = masked_reconstruction(decoder, shares, index)
                missing_share_mse[index] += F.mse_loss(masked, images).item() * batch
                aggregate_static[index].append(static_metrics(shares[index]))
            missing_share_count += batch

            if first_images is None:
                display_count = min(args.samples, batch)
                first_images = images[:display_count]
                first_shares = [share[:display_count] for share in shares]
                first_reconstruction = reconstruction[:display_count]

    if total_count == 0:
        raise RuntimeError(f"No images were loaded from the {args.split} split")

    mse = total_mse / total_count
    psnr = float("inf") if mse <= 0 else 10.0 * torch.log10(torch.tensor(1.0 / mse)).item()

    share_metrics = {}
    for index in range(NUM_SHARES):
        batches = aggregate_static[index]
        keys = batches[0].keys()
        share_metrics[f"share_{index + 1}"] = {
            key: sum(item[key] for item in batches) / len(batches)
            for key in keys
        }

    masked_results = {}
    for index in range(NUM_SHARES):
        masked_mse = missing_share_mse[index] / missing_share_count
        masked_psnr = float("inf") if masked_mse <= 0 else 10.0 * torch.log10(
            torch.tensor(1.0 / masked_mse)
        ).item()
        masked_results[f"without_share_{index + 1}"] = {
            "mse": masked_mse,
            "psnr_db": masked_psnr,
        }

    share_display = [
        F.interpolate(share, size=(args.image_size, args.image_size), mode="nearest")
        for share in first_shares
    ]
    rows = [first_images, *share_display, first_reconstruction]
    grid = make_grid(
        torch.cat(rows, dim=0).cpu(),
        nrow=first_images.shape[0],
        padding=2,
    )
    save_image(grid.clamp(0, 1), output_dir / "static_reconstruction_grid.png")

    noise_grid = make_grid(
        first_shares[0].cpu(),
        nrow=first_shares[0].shape[0],
        padding=2,
    )
    save_image(noise_grid.clamp(0, 1), output_dir / "reference_uniform_noise.png")

    results = {
        "experiment": "static_gan",
        "checkpoint_dir": str(checkpoint_dir),
        "device": str(device),
        "split": args.split,
        "train_images": args.train_images,
        "validation_images": args.validation_images,
        "test_images": args.test_images,
        "evaluated_images": sample_count,
        "reconstruction_mse": mse,
        "reconstruction_psnr_db": psnr,
        "share_static_metrics": share_metrics,
        "leave_one_share_out": masked_results,
        "visuals": {
            "static_reconstruction_grid": str(output_dir / "static_reconstruction_grid.png"),
            "reference_uniform_noise": str(output_dir / "reference_uniform_noise.png"),
        },
    }
    with (output_dir / "results.json").open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    print(f"Device: {device}")
    print(f"Evaluation split: {args.split}")
    print(f"Evaluated images: {sample_count}")
    print(f"Reconstruction PSNR: {psnr:.3f} dB")
    for name, metrics in share_metrics.items():
        print(
            f"{name}: mean={metrics['mean']:.4f}, std={metrics['std']:.4f}, "
            f"H-diff={metrics['horizontal_difference_mse']:.4f}, "
            f"V-diff={metrics['vertical_difference_mse']:.4f}"
        )
    for name, metrics in masked_results.items():
        print(f"{name}: PSNR={metrics['psnr_db']:.3f} dB")
    print(f"Saved: {output_dir / 'results.json'}")
    print(f"Saved: {output_dir / 'static_reconstruction_grid.png'}")
    print(f"Saved: {output_dir / 'reference_uniform_noise.png'}")


if __name__ == "__main__":
    main()
