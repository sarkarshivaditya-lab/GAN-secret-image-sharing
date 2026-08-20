import math
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from models.encoder import ShareEncoder
from models.decoder import ShareDecoder
from models.attacker import ShareAttacker


TRAIN_IMAGES = 5000
TEST_IMAGES = 1000

BATCH_SIZE = 16
ATTACKER_EPOCHS = 10
ATTACKER_LEARNING_RATE = 0.0002

DEVICE = (
    torch.device("mps")
    if torch.backends.mps.is_available()
    else torch.device("cpu")
)


MODELS = {
    "Baseline": {
        "encoder": "checkpoints/encoder_baseline.pth",
        "decoder": "checkpoints/decoder_baseline.pth"
    },
    "Lambda 0.10": {
        "encoder": "checkpoints/privacy_sweep/"
        "lambda_0.10/encoder.pth",
        "decoder": "checkpoints/privacy_sweep/"
        "lambda_0.10/decoder.pth"
    },
    "Privacy-GAN": {
        "encoder": "checkpoints/privacy_gan/"
        "encoder_best.pth",
        "decoder": "checkpoints/privacy_gan/"
        "decoder_best.pth"
    }
}


def calculate_psnr(
    original,
    reconstructed
):
    mse = torch.mean(
        (original - reconstructed) ** 2
    )

    if mse.item() == 0:
        return float("inf")

    return 10 * math.log10(
        1.0 / mse.item()
    )


def load_model(
    encoder_path,
    decoder_path
):
    encoder = ShareEncoder().to(
        DEVICE
    )

    decoder = ShareDecoder().to(
        DEVICE
    )

    encoder.load_state_dict(
        torch.load(
            encoder_path,
            map_location=DEVICE
        )
    )

    decoder.load_state_dict(
        torch.load(
            decoder_path,
            map_location=DEVICE
        )
    )

    encoder.eval()
    decoder.eval()

    for parameter in encoder.parameters():
        parameter.requires_grad = False

    for parameter in decoder.parameters():
        parameter.requires_grad = False

    return encoder, decoder


def evaluate_reconstruction(
    encoder,
    decoder,
    test_loader
):
    total_loss = 0.0
    total_psnr = 0.0
    batches = 0

    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(
                DEVICE
            )

            shares = encoder(
                images
            )

            reconstructed = decoder(
                *shares
            )

            loss = F.mse_loss(
                reconstructed,
                images
            )

            psnr = calculate_psnr(
                images,
                reconstructed
            )

            total_loss += loss.item()
            total_psnr += psnr
            batches += 1

    return (
        total_loss / batches,
        total_psnr / batches
    )


def train_attacker(
    encoder,
    train_loader,
    share_index
):
    attacker = ShareAttacker().to(
        DEVICE
    )

    optimizer = torch.optim.Adam(
        attacker.parameters(),
        lr=ATTACKER_LEARNING_RATE
    )

    for epoch in range(
        ATTACKER_EPOCHS
    ):
        attacker.train()

        total_loss = 0.0
        batches = 0

        for images, _ in train_loader:
            images = images.to(
                DEVICE
            )

            with torch.no_grad():
                shares = encoder(
                    images
                )

            share = shares[
                share_index
            ]

            reconstructed = attacker(
                share
            )

            loss = F.mse_loss(
                reconstructed,
                images
            )

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            total_loss += loss.item()
            batches += 1

    return attacker


def evaluate_attacker(
    encoder,
    attacker,
    test_loader,
    share_index
):
    encoder.eval()
    attacker.eval()

    total_loss = 0.0
    total_psnr = 0.0
    batches = 0

    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(
                DEVICE
            )

            shares = encoder(
                images
            )

            share = shares[
                share_index
            ]

            reconstructed = attacker(
                share
            )

            loss = F.mse_loss(
                reconstructed,
                images
            )

            psnr = calculate_psnr(
                images,
                reconstructed
            )

            total_loss += loss.item()
            total_psnr += psnr
            batches += 1

    return (
        total_loss / batches,
        total_psnr / batches
    )


def evaluate_model(
    model_name,
    model_paths,
    train_loader,
    test_loader
):
    print()
    print("=" * 60)
    print(
        f"Evaluating: {model_name}"
    )
    print("=" * 60)

    encoder, decoder = load_model(
        model_paths["encoder"],
        model_paths["decoder"]
    )

    reconstruction_loss, reconstruction_psnr = (
        evaluate_reconstruction(
            encoder,
            decoder,
            test_loader
        )
    )

    print(
        f"Reconstruction PSNR: "
        f"{reconstruction_psnr:.2f} dB"
    )

    attack_psnrs = []

    for share_index in range(4):
        print()
        print(
            f"Training fresh attacker "
            f"for Share {share_index + 1}"
        )

        attacker = train_attacker(
            encoder,
            train_loader,
            share_index
        )

        attack_loss, attack_psnr = (
            evaluate_attacker(
                encoder,
                attacker,
                test_loader,
                share_index
            )
        )

        attack_psnrs.append(
            attack_psnr
        )

        print(
            f"Share {share_index + 1} "
            f"Attack PSNR: "
            f"{attack_psnr:.2f} dB"
        )

        del attacker

        if DEVICE.type == "mps":
            torch.mps.empty_cache()

    average_attack_psnr = (
        sum(attack_psnrs) / 4.0
    )

    worst_case_attack_psnr = (
        max(attack_psnrs)
    )

    print()
    print(
        f"{model_name} average attack PSNR: "
        f"{average_attack_psnr:.2f} dB"
    )

    print(
        f"{model_name} worst-case attack PSNR: "
        f"{worst_case_attack_psnr:.2f} dB"
    )

    del encoder
    del decoder

    if DEVICE.type == "mps":
        torch.mps.empty_cache()

    return {
        "reconstruction_loss":
            reconstruction_loss,
        "reconstruction_psnr":
            reconstruction_psnr,
        "attack_psnrs":
            attack_psnrs,
        "average_attack_psnr":
            average_attack_psnr,
        "worst_case_attack_psnr":
            worst_case_attack_psnr
    }


def main():
    print("Device:", DEVICE)
    print()
    print(
        "Controlled privacy model benchmark"
    )
    print(
        "Attacker training images:",
        TRAIN_IMAGES
    )
    print(
        "Attacker test images:",
        TEST_IMAGES
    )
    print(
        "Attacker epochs:",
        ATTACKER_EPOCHS
    )
    print(
        "Attacker batch size:",
        BATCH_SIZE
    )
    print()

    transform = transforms.Compose([
        transforms.Resize(
            (256, 256)
        ),
        transforms.ToTensor()
    ])

    train_dataset = datasets.CIFAR10(
        root="data",
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.CIFAR10(
        root="data",
        train=False,
        download=True,
        transform=transform
    )

    train_dataset = Subset(
        train_dataset,
        range(TRAIN_IMAGES)
    )

    test_dataset = Subset(
        test_dataset,
        range(TEST_IMAGES)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    results = {}

    for model_name, model_paths in MODELS.items():
        results[model_name] = (
            evaluate_model(
                model_name,
                model_paths,
                train_loader,
                test_loader
            )
        )

    print()
    print()
    print("=" * 90)
    print(
        "FINAL PRIVACY BENCHMARK"
    )
    print("=" * 90)

    print(
        f"{'Model':<18}"
        f"{'Recon':>10}"
        f"{'Share 1':>11}"
        f"{'Share 2':>11}"
        f"{'Share 3':>11}"
        f"{'Share 4':>11}"
        f"{'Average':>11}"
        f"{'Worst':>11}"
    )

    print("-" * 90)

    for model_name, result in results.items():
        attacks = result[
            "attack_psnrs"
        ]

        print(
            f"{model_name:<18}"
            f"{result['reconstruction_psnr']:>10.2f}"
            f"{attacks[0]:>11.2f}"
            f"{attacks[1]:>11.2f}"
            f"{attacks[2]:>11.2f}"
            f"{attacks[3]:>11.2f}"
            f"{result['average_attack_psnr']:>11.2f}"
            f"{result['worst_case_attack_psnr']:>11.2f}"
        )

    print("=" * 90)

    os.makedirs(
        "outputs/privacy_benchmark",
        exist_ok=True
    )

    results_path = (
        "outputs/privacy_benchmark/"
        "benchmark_results.txt"
    )

    with open(
        results_path,
        "w"
    ) as file:
        file.write(
            "Controlled privacy model benchmark\n"
        )

        file.write(
            f"Training images: "
            f"{TRAIN_IMAGES}\n"
        )

        file.write(
            f"Test images: "
            f"{TEST_IMAGES}\n"
        )

        file.write(
            f"Attacker epochs: "
            f"{ATTACKER_EPOCHS}\n"
        )

        file.write(
            f"Batch size: "
            f"{BATCH_SIZE}\n"
        )

        file.write("\n")

        for model_name, result in results.items():
            file.write(
                f"{model_name}\n"
            )

            file.write(
                f"Reconstruction PSNR: "
                f"{result['reconstruction_psnr']:.6f} dB\n"
            )

            for index, psnr in enumerate(
                result["attack_psnrs"]
            ):
                file.write(
                    f"Share {index + 1}: "
                    f"{psnr:.6f} dB\n"
                )

            file.write(
                f"Average attack PSNR: "
                f"{result['average_attack_psnr']:.6f} dB\n"
            )

            file.write(
                f"Worst-case attack PSNR: "
                f"{result['worst_case_attack_psnr']:.6f} dB\n"
            )

            file.write("\n")

    print()
    print(
        f"Benchmark saved to: "
        f"{results_path}"
    )

    print()
    print(
        "Controlled benchmark complete."
    )


if __name__ == "__main__":
    main()