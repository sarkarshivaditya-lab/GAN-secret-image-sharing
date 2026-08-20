import json
import math
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from models.encoder import ShareEncoder
from models.decoder import ShareDecoder
from models.attacker import ShareAttacker


DEVICE = (
    torch.device("mps")
    if torch.backends.mps.is_available()
    else torch.device("cpu")
)

CANDIDATES = [
    0.003,
    0.005
]

TRAIN_IMAGES = 5000
TEST_IMAGES = 1000

ATTACKER_EPOCHS = 10
ATTACKER_BATCH_SIZE = 16
ATTACKER_LR = 0.0002

OUTPUT_DIR = "outputs/gan_weight_candidates"
CHECKPOINT_DIR = "checkpoints/gan_weight_sweep"


def calculate_psnr(
    original,
    reconstructed
):
    mse = torch.mean(
        (original - reconstructed) ** 2
    )

    if mse.item() <= 0:
        return float("inf")

    return 10.0 * math.log10(
        1.0 / mse.item()
    )


def load_model(
    gan_weight
):
    weight_name = f"{gan_weight:.3f}"

    directory = os.path.join(
        CHECKPOINT_DIR,
        f"gan_{weight_name}"
    )

    encoder = ShareEncoder().to(
        DEVICE
    )

    decoder = ShareDecoder().to(
        DEVICE
    )

    encoder.load_state_dict(
        torch.load(
            os.path.join(
                directory,
                "encoder.pth"
            ),
            map_location=DEVICE
        )
    )

    decoder.load_state_dict(
        torch.load(
            os.path.join(
                directory,
                "decoder.pth"
            ),
            map_location=DEVICE
        )
    )

    encoder.eval()
    decoder.eval()

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
    attacker,
    share_index,
    train_loader
):
    optimizer = torch.optim.Adam(
        attacker.parameters(),
        lr=ATTACKER_LR
    )

    encoder.eval()
    attacker.train()

    for epoch in range(
        ATTACKER_EPOCHS
    ):
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

        average_loss = (
            total_loss / batches
        )

        print(
            f"Share {share_index + 1} | "
            f"Epoch [{epoch + 1}/"
            f"{ATTACKER_EPOCHS}] | "
            f"Train Loss: "
            f"{average_loss:.6f}"
        )


def evaluate_attacker(
    encoder,
    attacker,
    share_index,
    test_loader
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


def evaluate_candidate(
    gan_weight,
    train_loader,
    test_loader
):
    print()
    print("=" * 60)
    print(
        f"Evaluating GAN weight: "
        f"{gan_weight:.3f}"
    )
    print("=" * 60)

    encoder, decoder = load_model(
        gan_weight
    )

    reconstruction_loss, reconstruction_psnr = (
        evaluate_reconstruction(
            encoder,
            decoder,
            test_loader
        )
    )

    print(
        f"Legitimate Reconstruction Loss: "
        f"{reconstruction_loss:.6f}"
    )

    print(
        f"Legitimate Reconstruction PSNR: "
        f"{reconstruction_psnr:.2f} dB"
    )

    attack_results = []

    for share_index in range(4):
        print()
        print(
            f"Training fresh independent "
            f"attacker for Share "
            f"{share_index + 1}"
        )

        attacker = ShareAttacker().to(
            DEVICE
        )

        train_attacker(
            encoder,
            attacker,
            share_index,
            train_loader
        )

        attack_loss, attack_psnr = (
            evaluate_attacker(
                encoder,
                attacker,
                share_index,
                test_loader
            )
        )

        print(
            f"Share {share_index + 1} "
            f"Attack PSNR: "
            f"{attack_psnr:.2f} dB"
        )

        attack_results.append(
            {
                "share": share_index + 1,
                "loss": attack_loss,
                "psnr": attack_psnr
            }
        )

        del attacker

        if DEVICE.type == "mps":
            torch.mps.empty_cache()

    attack_psnrs = [
        result["psnr"]
        for result in attack_results
    ]

    average_attack_psnr = (
        sum(attack_psnrs)
        /
        len(attack_psnrs)
    )

    worst_case_attack_psnr = max(
        attack_psnrs
    )

    print()
    print(
        f"GAN {gan_weight:.3f} "
        f"average attack PSNR: "
        f"{average_attack_psnr:.2f} dB"
    )

    print(
        f"GAN {gan_weight:.3f} "
        f"worst-case attack PSNR: "
        f"{worst_case_attack_psnr:.2f} dB"
    )

    result = {
        "gan_weight": gan_weight,
        "reconstruction_loss": reconstruction_loss,
        "reconstruction_psnr": reconstruction_psnr,
        "shares": attack_results,
        "average_attack_psnr": average_attack_psnr,
        "worst_case_attack_psnr": worst_case_attack_psnr
    }

    del encoder
    del decoder

    if DEVICE.type == "mps":
        torch.mps.empty_cache()

    return result


def save_results(
    results
):
    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    json_path = os.path.join(
        OUTPUT_DIR,
        "results.json"
    )

    txt_path = os.path.join(
        OUTPUT_DIR,
        "results.txt"
    )

    with open(
        json_path,
        "w"
    ) as file:
        json.dump(
            results,
            file,
            indent=4
        )

    with open(
        txt_path,
        "w"
    ) as file:
        file.write(
            "GAN Weight Candidate Evaluation\n"
        )

        file.write(
            "=" * 80 + "\n\n"
        )

        for result in results:
            file.write(
                f"GAN Weight: "
                f"{result['gan_weight']:.3f}\n"
            )

            file.write(
                f"Reconstruction PSNR: "
                f"{result['reconstruction_psnr']:.2f} dB\n"
            )

            for share in result["shares"]:
                file.write(
                    f"Share {share['share']} Attack PSNR: "
                    f"{share['psnr']:.2f} dB\n"
                )

            file.write(
                f"Average Attack PSNR: "
                f"{result['average_attack_psnr']:.2f} dB\n"
            )

            file.write(
                f"Worst-case Attack PSNR: "
                f"{result['worst_case_attack_psnr']:.2f} dB\n"
            )

            file.write(
                "\n"
            )

    return json_path, txt_path


def main():
    print(
        "Device:",
        DEVICE
    )

    print()
    print(
        "GAN-weight independent attacker evaluation"
    )

    print(
        "Candidates:",
        CANDIDATES
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
        ATTACKER_BATCH_SIZE
    )

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
        batch_size=ATTACKER_BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=ATTACKER_BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    results = []

    for gan_weight in CANDIDATES:
        result = evaluate_candidate(
            gan_weight,
            train_loader,
            test_loader
        )

        results.append(
            result
        )

    json_path, txt_path = save_results(
        results
    )

    print()
    print("=" * 90)
    print(
        "GAN WEIGHT CANDIDATE RESULTS"
    )
    print("=" * 90)

    print(
        f"{'GAN':<10}"
        f"{'Recon':<12}"
        f"{'Share 1':<12}"
        f"{'Share 2':<12}"
        f"{'Share 3':<12}"
        f"{'Share 4':<12}"
        f"{'Average':<12}"
        f"{'Worst':<12}"
    )

    print("-" * 90)

    for result in results:
        share_psnrs = [
            share["psnr"]
            for share in result["shares"]
        ]

        print(
            f"{result['gan_weight']:<10.3f}"
            f"{result['reconstruction_psnr']:<12.2f}"
            f"{share_psnrs[0]:<12.2f}"
            f"{share_psnrs[1]:<12.2f}"
            f"{share_psnrs[2]:<12.2f}"
            f"{share_psnrs[3]:<12.2f}"
            f"{result['average_attack_psnr']:<12.2f}"
            f"{result['worst_case_attack_psnr']:<12.2f}"
        )

    print("=" * 90)

    print(
        f"JSON results: {json_path}"
    )

    print(
        f"Text results: {txt_path}"
    )

    print()
    print(
        "Candidate evaluation complete."
    )


if __name__ == "__main__":
    main()