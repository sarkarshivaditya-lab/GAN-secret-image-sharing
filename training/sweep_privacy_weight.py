import os
import math
import json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from models.encoder import ShareEncoder
from models.decoder import ShareDecoder
from models.attacker import ShareAttacker


TRAIN_IMAGES = 5000
TEST_IMAGES = 1000
EPOCHS = 5
BATCH_SIZE = 8

LEARNING_RATE_ENCODER = 0.0001
LEARNING_RATE_ATTACKER = 0.0002

ATTACKER_STEPS = 2

PRIVACY_WEIGHTS = [
    0.05,
    0.10,
    0.20,
    0.30,
    0.50
]

DEVICE = (
    torch.device("mps")
    if torch.backends.mps.is_available()
    else torch.device("cpu")
)


def calculate_psnr(original, reconstructed):
    mse = torch.mean((original - reconstructed) ** 2)

    if mse.item() == 0:
        return float("inf")

    return 10 * math.log10(1.0 / mse.item())


def create_attackers(device):
    return [
        ShareAttacker().to(device)
        for _ in range(4)
    ]


def create_attacker_optimizers(attackers):
    return [
        torch.optim.Adam(
            attacker.parameters(),
            lr=LEARNING_RATE_ATTACKER
        )
        for attacker in attackers
    ]


def train_attackers(
    encoder,
    attackers,
    optimizers,
    images
):
    encoder.eval()

    with torch.no_grad():
        shares = encoder(images)

    total_loss = 0.0

    for share_index in range(4):
        attacker = attackers[share_index]
        optimizer = optimizers[share_index]

        for _ in range(ATTACKER_STEPS):
            share = shares[share_index]

            reconstructed = attacker(share)

            loss = F.mse_loss(
                reconstructed,
                images
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

    return total_loss / (4 * ATTACKER_STEPS)


def freeze_attackers(attackers):
    for attacker in attackers:
        for parameter in attacker.parameters():
            parameter.requires_grad = False


def unfreeze_attackers(attackers):
    for attacker in attackers:
        for parameter in attacker.parameters():
            parameter.requires_grad = True


def evaluate(
    encoder,
    decoder,
    attackers,
    test_loader
):
    encoder.eval()
    decoder.eval()

    for attacker in attackers:
        attacker.eval()

    reconstruction_loss_total = 0.0
    reconstruction_psnr_total = 0.0

    attack_losses = [
        0.0,
        0.0,
        0.0,
        0.0
    ]

    attack_psnrs = [
        0.0,
        0.0,
        0.0,
        0.0
    ]

    batches = 0

    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(DEVICE)

            shares = encoder(images)

            reconstructed = decoder(*shares)

            reconstruction_loss = F.mse_loss(
                reconstructed,
                images
            )

            reconstruction_psnr = calculate_psnr(
                images,
                reconstructed
            )

            reconstruction_loss_total += (
                reconstruction_loss.item()
            )

            reconstruction_psnr_total += (
                reconstruction_psnr
            )

            for share_index in range(4):
                attacked = attackers[share_index](
                    shares[share_index]
                )

                attack_loss = F.mse_loss(
                    attacked,
                    images
                )

                attack_psnr = calculate_psnr(
                    images,
                    attacked
                )

                attack_losses[share_index] += (
                    attack_loss.item()
                )

                attack_psnrs[share_index] += (
                    attack_psnr
                )

            batches += 1

    return {
        "reconstruction_loss":
            reconstruction_loss_total / batches,
        "reconstruction_psnr":
            reconstruction_psnr_total / batches,
        "attack_losses": [
            loss / batches
            for loss in attack_losses
        ],
        "attack_psnrs": [
            psnr / batches
            for psnr in attack_psnrs
        ]
    }


def train_one_weight(
    privacy_weight,
    train_loader,
    test_loader
):
    print()
    print("=" * 60)
    print(
        f"PRIVACY WEIGHT: {privacy_weight}"
    )
    print("=" * 60)

    encoder = ShareEncoder().to(DEVICE)
    decoder = ShareDecoder().to(DEVICE)

    encoder.load_state_dict(
        torch.load(
            "checkpoints/encoder_baseline.pth",
            map_location=DEVICE
        )
    )

    decoder.load_state_dict(
        torch.load(
            "checkpoints/decoder_baseline.pth",
            map_location=DEVICE
        )
    )

    attackers = create_attackers(DEVICE)

    attacker_optimizers = create_attacker_optimizers(
        attackers
    )

    encoder_optimizer = torch.optim.Adam(
        list(encoder.parameters()) +
        list(decoder.parameters()),
        lr=LEARNING_RATE_ENCODER
    )

    best_reconstruction_psnr = float("-inf")
    best_epoch = 0
    best_result = None

    output_dir = (
        f"checkpoints/privacy_sweep/"
        f"lambda_{privacy_weight:.2f}"
    )

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    for epoch in range(EPOCHS):
        encoder.train()
        decoder.train()

        for attacker in attackers:
            attacker.train()

        reconstruction_total = 0.0
        attack_total = 0.0
        batches = 0

        for batch_index, (images, _) in enumerate(
            train_loader
        ):
            images = images.to(DEVICE)

            attack_loss = train_attackers(
                encoder,
                attackers,
                attacker_optimizers,
                images
            )

            shares = encoder(images)

            reconstructed = decoder(*shares)

            reconstruction_loss = F.mse_loss(
                reconstructed,
                images
            )

            total_attack_loss = 0.0

            for share_index in range(4):
                attacked = attackers[share_index](
                    shares[share_index]
                )

                share_attack_loss = F.mse_loss(
                    attacked,
                    images
                )

                total_attack_loss += (
                    share_attack_loss
                )

            average_attack_loss = (
                total_attack_loss / 4.0
            )

            encoder_loss = (
                reconstruction_loss
                -
                privacy_weight * average_attack_loss
            )

            encoder_optimizer.zero_grad()

            for optimizer in attacker_optimizers:
                optimizer.zero_grad()

            freeze_attackers(attackers)

            encoder_loss.backward()

            encoder_optimizer.step()

            unfreeze_attackers(attackers)

            reconstruction_total += (
                reconstruction_loss.item()
            )

            attack_total += (
                average_attack_loss.item()
            )

            batches += 1

            if batch_index % 200 == 0:
                print(
                    f"Epoch [{epoch + 1}/{EPOCHS}] "
                    f"Batch [{batch_index}/{len(train_loader)}] "
                    f"Recon Loss: "
                    f"{reconstruction_loss.item():.6f} "
                    f"Attack Loss: "
                    f"{average_attack_loss.item():.6f}"
                )

        average_reconstruction = (
            reconstruction_total / batches
        )

        average_attack = (
            attack_total / batches
        )

        result = evaluate(
            encoder,
            decoder,
            attackers,
            test_loader
        )

        print()
        print(
            f"Epoch [{epoch + 1}/{EPOCHS}]"
        )
        print(
            f"Train Reconstruction Loss: "
            f"{average_reconstruction:.6f}"
        )
        print(
            f"Train Average Attack Loss: "
            f"{average_attack:.6f}"
        )
        print(
            f"Validation Reconstruction PSNR: "
            f"{result['reconstruction_psnr']:.2f} dB"
        )

        for share_index in range(4):
            print(
                f"Share {share_index + 1} "
                f"Attack PSNR: "
                f"{result['attack_psnrs'][share_index]:.2f} dB"
            )

        if (
            result["reconstruction_psnr"]
            >
            best_reconstruction_psnr
        ):
            best_reconstruction_psnr = (
                result["reconstruction_psnr"]
            )

            best_epoch = epoch + 1
            best_result = result

            torch.save(
                encoder.state_dict(),
                f"{output_dir}/encoder.pth"
            )

            torch.save(
                decoder.state_dict(),
                f"{output_dir}/decoder.pth"
            )

            for share_index in range(4):
                torch.save(
                    attackers[share_index].state_dict(),
                    f"{output_dir}/"
                    f"attacker_share_{share_index + 1}.pth"
                )

            print(
                "New best reconstruction checkpoint saved."
            )

    summary = {
        "privacy_weight": privacy_weight,
        "best_epoch": best_epoch,
        "reconstruction_psnr":
            best_result["reconstruction_psnr"],
        "reconstruction_loss":
            best_result["reconstruction_loss"],
        "attack_psnrs":
            best_result["attack_psnrs"],
        "attack_losses":
            best_result["attack_losses"]
    }

    with open(
        f"{output_dir}/results.json",
        "w"
    ) as file:
        json.dump(
            summary,
            file,
            indent=4
        )

    print()
    print(
        f"Best λ={privacy_weight:.2f} result:"
    )
    print(
        f"Reconstruction: "
        f"{best_result['reconstruction_psnr']:.2f} dB"
    )

    for share_index in range(4):
        print(
            f"Share {share_index + 1}: "
            f"{best_result['attack_psnrs'][share_index]:.2f} dB"
        )

    return summary


def main():
    print("Device:", DEVICE)
    print()
    print("Privacy-weight sweep")
    print(
        "Weights:",
        PRIVACY_WEIGHTS
    )
    print(
        "Training images:",
        TRAIN_IMAGES
    )
    print(
        "Test images:",
        TEST_IMAGES
    )
    print(
        "Epochs:",
        EPOCHS
    )

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
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

    results = []

    for privacy_weight in PRIVACY_WEIGHTS:
        result = train_one_weight(
            privacy_weight,
            train_loader,
            test_loader
        )

        results.append(result)

        if DEVICE.type == "mps":
            torch.mps.empty_cache()

    with open(
        "checkpoints/privacy_sweep/all_results.json",
        "w"
    ) as file:
        json.dump(
            results,
            file,
            indent=4
        )

    print()
    print("=" * 60)
    print("PRIVACY WEIGHT SWEEP COMPLETE")
    print("=" * 60)

    print()
    print(
        "λ      Reconstruction    S1       S2       S3       S4"
    )

    for result in results:
        attacks = result["attack_psnrs"]

        print(
            f"{result['privacy_weight']:.2f}    "
            f"{result['reconstruction_psnr']:.2f} dB       "
            f"{attacks[0]:.2f}    "
            f"{attacks[1]:.2f}    "
            f"{attacks[2]:.2f}    "
            f"{attacks[3]:.2f}"
        )


if __name__ == "__main__":
    main()