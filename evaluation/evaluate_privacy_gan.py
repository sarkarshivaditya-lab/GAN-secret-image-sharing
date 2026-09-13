import argparse
import math
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from models.attacker import ShareAttacker
from models.decoder import ShareDecoder
from models.encoder import ShareEncoder
from project_utils import (
    NUM_SHARES,
    build_cifar10_loaders,
    load_state_dict_compat,
    reconstruct,
    save_comparison_grid,
    save_json,
    save_tensor_image,
    seed_everything,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate Privacy-GAN reconstruction and single-share privacy."
    )
    parser.add_argument("--test-images", type=int, default=1000)
    parser.add_argument("--attacker-train-images", type=int, default=5000)
    parser.add_argument("--attacker-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--attacker-lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--checkpoint-dir", default="checkpoints/privacy_gan")
    parser.add_argument("--output-dir", default="outputs/privacy_gan")
    parser.add_argument("--visual-samples", type=int, default=8)
    return parser.parse_args()


def psnr_from_mse(mse):
    if mse <= 0.0:
        return float("inf")
    return 10.0 * math.log10(1.0 / mse)


def load_models(checkpoint_dir, device):
    directory = Path(checkpoint_dir)
    encoder_path = directory / "encoder_best.pth"
    decoder_path = directory / "decoder_best.pth"

    if not encoder_path.exists() or not decoder_path.exists():
        raise FileNotFoundError(
            f"Missing Privacy-GAN checkpoint in {directory}. "
            "Train the model before running evaluation."
        )

    encoder = ShareEncoder().to(device)
    decoder = ShareDecoder().to(device)

    load_state_dict_compat(encoder, encoder_path, device)
    load_state_dict_compat(decoder, decoder_path, device)

    encoder.eval()
    decoder.eval()
    return encoder, decoder


def train_fresh_attacker(encoder, loader, share_index, device, epochs, lr):
    attacker = ShareAttacker().to(device)
    optimizer = torch.optim.Adam(attacker.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        attacker.train()
        total_loss = 0.0
        total_images = 0

        for images, _ in loader:
            images = images.to(device)
            with torch.no_grad():
                share = encoder(images)[share_index]

            reconstructed = attacker(share)
            loss = F.mse_loss(reconstructed, images)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            batch_size = images.shape[0]
            total_loss += loss.item() * batch_size
            total_images += batch_size

        if epoch == 1 or epoch == epochs or epoch % 5 == 0:
            print(
                f"Share {share_index + 1} attacker | "
                f"Epoch {epoch}/{epochs} | "
                f"Train MSE {total_loss / total_images:.6f}"
            )

    return attacker


def evaluate_attacker(encoder, attacker, loader, share_index, device):
    encoder.eval()
    attacker.eval()
    total_mse = 0.0
    total_images = 0
    first_attack = None

    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            shares = encoder(images)
            attacked = attacker(shares[share_index])

            if first_attack is None:
                first_attack = attacked.detach().cpu()

            batch_size = images.shape[0]
            total_mse += F.mse_loss(attacked, images).item() * batch_size
            total_images += batch_size

    mse = total_mse / total_images
    return mse, psnr_from_mse(mse), first_attack


def evaluate_discriminator(discriminator_path, encoder, loader, device):
    from models.privacy_discriminator import PrivacyDiscriminator

    if not discriminator_path.exists():
        return None

    discriminator = PrivacyDiscriminator().to(device)
    load_state_dict_compat(discriminator, discriminator_path, device)
    discriminator.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            shares = encoder(images)
            for share in shares:
                permutation = torch.randperm(images.shape[0], device=device)
                mismatched = images[permutation]
                positive = discriminator(images, share)
                negative = discriminator(mismatched, share)
                correct += (positive >= 0).sum().item()
                correct += (negative < 0).sum().item()
                total += 2 * images.shape[0]

    return {"accuracy": correct / total}


def share_entropy(share):
    values = share.detach().float().cpu().numpy().reshape(-1)
    if values.size == 0 or float(values.max()) - float(values.min()) < 1e-12:
        return 0.0
    normalized = (values - values.min()) / (values.max() - values.min())
    bins = np.clip((normalized * 255).astype(np.int32), 0, 255)
    counts = np.bincount(bins, minlength=256).astype(np.float64)
    probabilities = counts / counts.sum()
    probabilities = probabilities[probabilities > 0]
    return float(-np.sum(probabilities * np.log2(probabilities)))


def pearson(a, b):
    a = a.detach().float().reshape(-1)
    b = b.detach().float().reshape(-1)
    a = a - a.mean()
    b = b - b.mean()
    denominator = torch.sqrt(torch.sum(a ** 2) * torch.sum(b ** 2))
    if denominator.item() < 1e-12:
        return 0.0
    return float((torch.sum(a * b) / denominator).item())


def evaluate_full_and_partial(encoder, decoder, loader, device, visual_samples):
    total_mse = 0.0
    total_images = 0
    partial_mse = {count: 0.0 for count in range(1, NUM_SHARES + 1)}
    first_batch = None

    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            shares = list(encoder(images))
            reconstructed = reconstruct(decoder, shares)
            batch_size = images.shape[0]
            total_mse += F.mse_loss(reconstructed, images).item() * batch_size
            total_images += batch_size

            if first_batch is None:
                first_batch = (
                    images.detach().cpu(),
                    [share.detach().cpu() for share in shares],
                    reconstructed.detach().cpu(),
                )

            for count in range(1, NUM_SHARES + 1):
                partial = []
                for index, share in enumerate(shares):
                    if index < count:
                        partial.append(share)
                    else:
                        partial.append(torch.zeros_like(share))
                partial_reconstruction = reconstruct(decoder, partial)
                partial_mse[count] += (
                    F.mse_loss(partial_reconstruction, images).item()
                    * batch_size
                )

    result = {
        "mse": total_mse / total_images,
        "psnr_db": psnr_from_mse(total_mse / total_images),
        "partial_zero_filled": {},
    }
    for count in range(1, NUM_SHARES + 1):
        mse = partial_mse[count] / total_images
        result["partial_zero_filled"][str(count)] = {
            "mse": mse,
            "psnr_db": psnr_from_mse(mse),
        }

    if first_batch is not None:
        images, shares, reconstructed = first_batch
        sample_count = min(visual_samples, images.shape[0])
        save_comparison_grid(
            images[:sample_count],
            [share[:sample_count] for share in shares],
            reconstructed[:sample_count],
            "outputs/privacy_gan/visuals/full_reconstruction_grid.png",
        )

    return result


def main():
    args = parse_args()
    seed_everything(args.seed)

    device = (
        torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    output_dir = Path(args.output_dir)
    visual_dir = output_dir / "visuals"
    output_dir.mkdir(parents=True, exist_ok=True)
    visual_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print("Privacy-GAN evaluation")
    print(f"Checkpoint directory: {args.checkpoint_dir}")
    print()

    _, test_loader = build_cifar10_loaders(
        data_dir=args.data_dir,
        train_images=args.attacker_train_images,
        test_images=args.test_images,
        batch_size=args.batch_size,
        image_size=args.image_size,
        seed=args.seed,
    )
    train_loader, _ = build_cifar10_loaders(
        data_dir=args.data_dir,
        train_images=args.attacker_train_images,
        test_images=1,
        batch_size=args.batch_size,
        image_size=args.image_size,
        seed=args.seed,
    )

    encoder, decoder = load_models(args.checkpoint_dir, device)

    legitimate = evaluate_full_and_partial(
        encoder,
        decoder,
        test_loader,
        device,
        args.visual_samples,
    )

    print(f"Legitimate reconstruction PSNR: {legitimate['psnr_db']:.3f} dB")

    attackers = []
    attack_results = {}
    attack_outputs = []

    for share_index in range(NUM_SHARES):
        attacker = train_fresh_attacker(
            encoder,
            train_loader,
            share_index,
            device,
            args.attacker_epochs,
            args.attacker_lr,
        )
        mse, psnr, first_attack = evaluate_attacker(
            encoder,
            attacker,
            test_loader,
            share_index,
            device,
        )
        attackers.append(attacker)
        attack_results[f"share_{share_index + 1}"] = {
            "mse": mse,
            "psnr_db": psnr,
        }
        attack_outputs.append(first_attack)
        print(f"Share {share_index + 1} fresh-attacker PSNR: {psnr:.3f} dB")

    if attack_outputs:
        sample_count = min(args.visual_samples, attack_outputs[0].shape[0])
        attacker_grid = torch.cat(
            [output[:sample_count] for output in attack_outputs],
            dim=0,
        )
        save_tensor_image(
            torch.cat(
                [
                    attacker_grid[i:i + sample_count]
                    for i in range(0, attacker_grid.shape[0], sample_count)
                ],
                dim=0,
            ),
            visual_dir / "fresh_attacker_grid.png",
        )

    share_stats = {}
    cross_correlations = {}

    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)
            shares = list(encoder(images))
            for index, share in enumerate(shares):
                key = f"share_{index + 1}"
                share_stats.setdefault(key, {"entropy_sum": 0.0, "batches": 0})
                share_stats[key]["entropy_sum"] += share_entropy(share)
                share_stats[key]["batches"] += 1
            for i, j in combinations(range(NUM_SHARES), 2):
                key = f"share_{i + 1}_vs_share_{j + 1}"
                cross_correlations.setdefault(key, {"sum": 0.0, "count": 0})
                cross_correlations[key]["sum"] += pearson(shares[i], shares[j])
                cross_correlations[key]["count"] += 1

    share_summary = {}
    for key, values in share_stats.items():
        share_summary[key] = {
            "mean_entropy_bits": values["entropy_sum"] / values["batches"]
        }

    correlation_summary = {
        key: values["sum"] / values["count"]
        for key, values in cross_correlations.items()
    }

    discriminator_result = evaluate_discriminator(
        Path(args.checkpoint_dir) / "discriminator_best.pth",
        encoder,
        test_loader,
        device,
    )

    results = {
        "experiment": "privacy_gan",
        "device": str(device),
        "checkpoint_dir": args.checkpoint_dir,
        "test_images": args.test_images,
        "attacker_train_images": args.attacker_train_images,
        "attacker_epochs": args.attacker_epochs,
        "batch_size": args.batch_size,
        "image_size": args.image_size,
        "seed": args.seed,
        "legitimate_reconstruction": legitimate,
        "fresh_single_share_attack": attack_results,
        "share_statistics": share_summary,
        "cross_share_correlation": correlation_summary,
        "privacy_discriminator": discriminator_result,
        "visual_outputs": {
            "full_reconstruction_grid": str(visual_dir / "full_reconstruction_grid.png"),
            "fresh_attacker_grid": str(visual_dir / "fresh_attacker_grid.png"),
        },
    }

    save_json(results, output_dir / "results.json")

    with (output_dir / "results.txt").open("w", encoding="utf-8") as file:
        file.write("Privacy-GAN evaluation\n\n")
        file.write(
            f"Legitimate reconstruction PSNR: "
            f"{legitimate['psnr_db']:.6f} dB\n"
        )
        for key, result in attack_results.items():
            file.write(f"{key}: {result['psnr_db']:.6f} dB\n")
        file.write("\nPartial zero-filled diagnostic:\n")
        for count, result in legitimate["partial_zero_filled"].items():
            file.write(
                f"{count} available shares: "
                f"{result['psnr_db']:.6f} dB\n"
            )

    print()
    print(f"Results saved to: {output_dir / 'results.json'}")
    print(f"Visuals saved to: {visual_dir}")
    print("Evaluation complete.")


if __name__ == "__main__":
    main()
