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
    parser = argparse.ArgumentParser(description="Evaluate the v2 TV-static image-sharing system.")
    parser.add_argument("--checkpoint-dir", default="checkpoints/static_gan")
    parser.add_argument("--test-images", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output-dir", default="outputs/static_gan_v2")
    parser.add_argument("--samples", type=int, default=8)
    return parser.parse_args()


def load(model, path, device):
    model.load_state_dict(torch.load(path, map_location=device, weights_only=False))
    model.eval()
    return model


def psnr_from_mse(mse):
    if mse <= 0.0:
        return float("inf")
    return 10.0 * torch.log10(torch.tensor(1.0 / mse)).item()


def static_metrics(share):
    flat = share.flatten(1)
    per_sample_mean = flat.mean(dim=1)
    per_sample_std = flat.std(dim=1, unbiased=False)
    horizontal = (share[:, :, :, 1:] - share[:, :, :, :-1]).pow(2).mean(dim=(1, 2, 3))
    vertical = (share[:, :, 1:, :] - share[:, :, :-1, :]).pow(2).mean(dim=(1, 2, 3))
    return {
        "mean": per_sample_mean.mean().item(),
        "std": per_sample_std.mean().item(),
        "mean_abs_error": (per_sample_mean - 0.5).abs().mean().item(),
        "std_abs_error": (per_sample_std - (1.0 / (12.0 ** 0.5))).abs().mean().item(),
        "horizontal_difference_mse": horizontal.mean().item(),
        "vertical_difference_mse": vertical.mean().item(),
    }


def recover_payload(shares):
    return torch.remainder(sum(shares), 1.0)


def leave_one_out_payload(shares, missing_index):
    masked = list(shares)
    masked[missing_index] = torch.zeros_like(masked[missing_index])
    return recover_payload(masked)


def grid_save(images, shares, reconstruction, output_path, image_size, sample_count):
    visible_shares = [
        F.interpolate(share[:sample_count], size=(image_size, image_size), mode="nearest")
        for share in shares
    ]
    rows = [images[:sample_count], *visible_shares, reconstruction[:sample_count]]
    grid = make_grid(torch.cat(rows, dim=0).cpu(), nrow=sample_count, padding=2)
    save_image(grid.clamp(0, 1), str(output_path))


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
    missing_share_mse = [0.0] * NUM_SHARES
    missing_payload_distance = [0.0] * NUM_SHARES
    static_totals = [
        {"mean": 0.0, "std": 0.0, "mean_abs_error": 0.0, "std_abs_error": 0.0,
         "horizontal_difference_mse": 0.0, "vertical_difference_mse": 0.0}
        for _ in range(NUM_SHARES)
    ]
    first_images = None
    first_shares = None
    first_reconstruction = None

    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)
            shares = list(encoder(images))
            reconstruction = reconstruct(decoder, shares)
            payload = recover_payload(shares)
            direct_payload_error = (payload - encoder.payload(encoder.features(images))).abs().max().item()

            batch = images.shape[0]
            total_mse += F.mse_loss(reconstruction, images).item() * batch
            total_count += batch

            for index, share in enumerate(shares):
                metrics = static_metrics(share)
                for key in static_totals[index]:
                    static_totals[index][key] += metrics[key] * batch

                missing_payload = leave_one_out_payload(shares, index)
                missing_reconstruction = decoder(*[
                    torch.zeros_like(shares[index]) if current == index else shares[current]
                    for current in range(NUM_SHARES)
                ])
                missing_mse = F.mse_loss(missing_reconstruction, images).item()
                missing_share_mse[index] += missing_mse * batch
                missing_payload_distance[index] += (missing_payload - payload).abs().mean().item() * batch

            if first_images is None:
                sample_count = min(args.samples, batch)
                first_images = images[:sample_count].detach().cpu()
                first_shares = [share[:sample_count].detach().cpu() for share in shares]
                first_reconstruction = reconstruction[:sample_count].detach().cpu()
                first_direct_payload_error = direct_payload_error

    mse = total_mse / total_count
    psnr = psnr_from_mse(mse)

    share_metrics = {}
    for index in range(NUM_SHARES):
        share_metrics[f"share_{index + 1}"] = {
            key: value / total_count for key, value in static_totals[index].items()
        }

    leave_one_out = {}
    for index in range(NUM_SHARES):
        masked_mse = missing_share_mse[index] / total_count
        leave_one_out[f"without_share_{index + 1}"] = {
            "reconstruction_mse": masked_mse,
            "reconstruction_psnr_db": psnr_from_mse(masked_mse),
            "mean_payload_change": missing_payload_distance[index] / total_count,
        }

    grid_path = output_dir / "static_reconstruction_grid.png"
    reference_path = output_dir / "reference_uniform_noise.png"
    if first_images is not None:
        grid_save(first_images, first_shares, first_reconstruction, grid_path, args.image_size, first_images.shape[0])
        reference = torch.rand_like(first_shares[0])
        reference = F.interpolate(reference, size=(args.image_size, args.image_size), mode="nearest")
        reference_grid = make_grid(reference.cpu(), nrow=first_images.shape[0], padding=2)
        save_image(reference_grid.clamp(0, 1), str(reference_path))

    results = {
        "experiment": "static_gan_v2",
        "device": str(device),
        "checkpoint_dir": str(checkpoint_dir),
        "test_images": args.test_images,
        "reconstruction_mse": mse,
        "reconstruction_psnr_db": psnr,
        "max_payload_recovery_error_first_batch": first_direct_payload_error,
        "share_static_metrics": share_metrics,
        "leave_one_share_out": leave_one_out,
        "visuals": {
            "static_reconstruction_grid": str(grid_path),
            "reference_uniform_noise": str(reference_path),
        },
    }

    with (output_dir / "results.json").open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    print(f"Device: {device}")
    print(f"Legitimate reconstruction PSNR: {psnr:.3f} dB")
    print(f"Max first-batch payload recovery error: {first_direct_payload_error:.8f}")
    for name, metrics in share_metrics.items():
        print(
            f"{name}: mean={metrics['mean']:.4f}, std={metrics['std']:.4f}, "
            f"H-diff={metrics['horizontal_difference_mse']:.4f}, "
            f"V-diff={metrics['vertical_difference_mse']:.4f}"
        )
    for name, metrics in leave_one_out.items():
        print(
            f"{name}: PSNR={metrics['reconstruction_psnr_db']:.3f} dB | "
            f"payload-change={metrics['mean_payload_change']:.6f}"
        )
    print(f"Saved: {output_dir / 'results.json'}")
    print(f"Saved: {grid_path}")
    print(f"Saved: {reference_path}")


if __name__ == "__main__":
    main()
