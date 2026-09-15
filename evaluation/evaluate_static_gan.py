import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image

from models.decoder import ShareDecoder
from models.encoder import ShareEncoder
from project_utils import build_cifar10_loaders, get_device, reconstruct, seed_everything


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a TV-static share GAN.")
    parser.add_argument("--checkpoint-dir", default="checkpoints/static_gan")
    parser.add_argument("--test-images", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--seed", type=int, default=123)
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
    horizontal = (share[:, :, :, 1:] - share[:, :, :, :-1]).pow(2).mean().item()
    vertical = (share[:, :, 1:, :] - share[:, :, :-1, :]).pow(2).mean().item()
    return {
        "mean": mean.mean().item(),
        "std": std.mean().item(),
        "horizontal_difference_mse": horizontal,
        "vertical_difference_mse": vertical,
    }


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = get_device()
    checkpoint_dir = Path(args.checkpoint_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    encoder = load(ShareEncoder().to(device), checkpoint_dir / "encoder_best.pth", device)
    decoder = load(ShareDecoder().to(device), checkpoint_dir / "decoder_best.pth", device)

    _, test_loader = build_cifar10_loaders(
        data_dir=args.data_dir,
        train_images=1,
        test_images=args.test_images,
        batch_size=args.batch_size,
        image_size=args.image_size,
        seed=args.seed,
    )

    total_mse = 0.0
    total_count = 0
    first_images = None
    first_shares = None
    first_reconstruction = None

    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)
            shares = encoder(images)
            reconstruction = reconstruct(decoder, shares)
            batch = images.shape[0]
            total_mse += F.mse_loss(reconstruction, images).item() * batch
            total_count += batch

            if first_images is None:
                first_images = images[:args.samples]
                first_shares = [share[:args.samples] for share in shares]
                first_reconstruction = reconstruction[:args.samples]

    mse = total_mse / total_count
    psnr = float("inf") if mse <= 0 else 10.0 * torch.log10(torch.tensor(1.0 / mse)).item()

    share_metrics = {
        f"share_{index + 1}": static_metrics(share)
        for index, share in enumerate(first_shares)
    }

    share_display = [
        F.interpolate(share, size=(args.image_size, args.image_size), mode="nearest")
        for share in first_shares
    ]
    rows = [first_images, *share_display, first_reconstruction]
    grid = make_grid(torch.cat(rows, dim=0).cpu(), nrow=first_images.shape[0], padding=2)
    save_image(grid.clamp(0, 1), output_dir / "static_reconstruction_grid.png")

    results = {
        "experiment": "static_gan",
        "checkpoint_dir": str(checkpoint_dir),
        "device": str(device),
        "test_images": args.test_images,
        "reconstruction_mse": mse,
        "reconstruction_psnr_db": psnr,
        "share_static_metrics": share_metrics,
        "visual": str(output_dir / "static_reconstruction_grid.png"),
    }
    with (output_dir / "results.json").open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    print(f"Device: {device}")
    print(f"Legitimate reconstruction PSNR: {psnr:.3f} dB")
    for name, metrics in share_metrics.items():
        print(
            f"{name}: mean={metrics['mean']:.4f}, "
            f"std={metrics['std']:.4f}, "
            f"H-diff={metrics['horizontal_difference_mse']:.4f}, "
            f"V-diff={metrics['vertical_difference_mse']:.4f}"
        )
    print(f"Saved: {output_dir / 'results.json'}")
    print(f"Saved: {output_dir / 'static_reconstruction_grid.png'}")


if __name__ == "__main__":
    main()
