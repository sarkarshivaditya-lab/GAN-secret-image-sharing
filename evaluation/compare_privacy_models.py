import argparse
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from models.attacker import ShareAttacker
from models.decoder import ShareDecoder
from models.encoder import ShareEncoder
from project_utils import (
    NUM_SHARES,
    build_cifar10_loaders,
    load_state_dict_compat,
    save_json,
    seed_everything,
)


MODELS = {
    "Baseline": (
        "checkpoints/encoder_baseline.pth",
        "checkpoints/decoder_baseline.pth",
    ),
    "Privacy sweep lambda 0.10": (
        "checkpoints/privacy_sweep/lambda_0.10/encoder.pth",
        "checkpoints/privacy_sweep/lambda_0.10/decoder.pth",
    ),
    "Privacy-GAN": (
        "checkpoints/privacy_gan/encoder_best.pth",
        "checkpoints/privacy_gan/decoder_best.pth",
    ),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare main image-sharing model variants under the same attacker protocol."
    )
    parser.add_argument("--train-images", type=int, default=5000)
    parser.add_argument("--test-images", type=int, default=1000)
    parser.add_argument("--attacker-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--attacker-lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=321)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output", default="outputs/privacy_model_comparison/results.json")
    return parser.parse_args()


def psnr_from_mse(mse):
    if mse <= 0.0:
        return float("inf")
    return 10.0 * math.log10(1.0 / mse)


def load_models(encoder_path, decoder_path, device):
    if not Path(encoder_path).exists() or not Path(decoder_path).exists():
        return None

    encoder = ShareEncoder().to(device)
    decoder = ShareDecoder().to(device)
    load_state_dict_compat(encoder, encoder_path, device)
    load_state_dict_compat(decoder, decoder_path, device)
    encoder.eval()
    decoder.eval()
    return encoder, decoder


def reconstruction_metrics(encoder, decoder, loader, device):
    total_mse = 0.0
    total_images = 0
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            reconstructed = decoder(*encoder(images))
            batch_size = images.shape[0]
            total_mse += F.mse_loss(reconstructed, images).item() * batch_size
            total_images += batch_size
    mse = total_mse / total_images
    return {"mse": mse, "psnr_db": psnr_from_mse(mse)}


def train_attacker(encoder, loader, share_index, device, epochs, lr):
    attacker = ShareAttacker().to(device)
    optimizer = torch.optim.Adam(attacker.parameters(), lr=lr)

    for _ in range(epochs):
        attacker.train()
        for images, _ in loader:
            images = images.to(device)
            with torch.no_grad():
                share = encoder(images)[share_index]
            reconstructed = attacker(share)
            loss = F.mse_loss(reconstructed, images)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    return attacker


def attacker_metrics(encoder, attacker, loader, share_index, device):
    total_mse = 0.0
    total_images = 0
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            share = encoder(images)[share_index]
            reconstructed = attacker(share)
            batch_size = images.shape[0]
            total_mse += F.mse_loss(reconstructed, images).item() * batch_size
            total_images += batch_size
    mse = total_mse / total_images
    return {"mse": mse, "psnr_db": psnr_from_mse(mse)}


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

    train_loader, test_loader = build_cifar10_loaders(
        data_dir=args.data_dir,
        train_images=args.train_images,
        test_images=args.test_images,
        batch_size=args.batch_size,
        image_size=args.image_size,
        seed=args.seed,
    )

    comparison = {
        "device": str(device),
        "train_images": args.train_images,
        "test_images": args.test_images,
        "attacker_epochs": args.attacker_epochs,
        "models": {},
    }

    for name, (encoder_path, decoder_path) in MODELS.items():
        print(f"\nEvaluating {name}")
        models = load_models(encoder_path, decoder_path, device)
        if models is None:
            print("Checkpoint not available; skipping.")
            continue

        encoder, decoder = models
        result = {
            "encoder": encoder_path,
            "decoder": decoder_path,
            "legitimate_reconstruction": reconstruction_metrics(
                encoder,
                decoder,
                test_loader,
                device,
            ),
            "fresh_single_share_attack": {},
        }

        for share_index in range(NUM_SHARES):
            attacker = train_attacker(
                encoder,
                train_loader,
                share_index,
                device,
                args.attacker_epochs,
                args.attacker_lr,
            )
            result["fresh_single_share_attack"][
                f"share_{share_index + 1}"
            ] = attacker_metrics(
                encoder,
                attacker,
                test_loader,
                share_index,
                device,
            )
            del attacker

        comparison["models"][name] = result
        print(
            f"Legitimate PSNR: "
            f"{result['legitimate_reconstruction']['psnr_db']:.3f} dB"
        )
        for key, metrics in result["fresh_single_share_attack"].items():
            print(f"{key}: {metrics['psnr_db']:.3f} dB")

    save_json(comparison, args.output)
    print(f"\nComparison saved to {args.output}")


if __name__ == "__main__":
    main()
