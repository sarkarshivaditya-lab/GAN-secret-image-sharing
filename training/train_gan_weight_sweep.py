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
from models.privacy_discriminator import PrivacyDiscriminator


TRAIN_IMAGES = 5000
TEST_IMAGES = 1000

EPOCHS = 3
BATCH_SIZE = 8

GENERATOR_LR = 0.0001
ATTACKER_LR = 0.0002
DISCRIMINATOR_LR = 0.0002

PRIVACY_WEIGHT = 0.10

GAN_WEIGHTS = [
    0.001,
    0.003,
    0.005,
    0.010,
    0.020
]

ATTACKER_STEPS = 2

OUTPUT_DIR = "checkpoints/gan_weight_sweep"

DEVICE = (
    torch.device("mps")
    if torch.backends.mps.is_available()
    else torch.device("cpu")
)


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


def create_attackers():
    return [
        ShareAttacker().to(DEVICE)
        for _ in range(4)
    ]


def create_attacker_optimizers(
    attackers
):
    return [
        torch.optim.Adam(
            attacker.parameters(),
            lr=ATTACKER_LR
        )
        for attacker in attackers
    ]


def freeze_attackers(
    attackers
):
    for attacker in attackers:
        for parameter in attacker.parameters():
            parameter.requires_grad = False


def unfreeze_attackers(
    attackers
):
    for attacker in attackers:
        for parameter in attacker.parameters():
            parameter.requires_grad = True


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
            reconstructed = attacker(
                shares[share_index]
            )

            loss = F.mse_loss(
                reconstructed,
                images
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

    return total_loss / (
        4 * ATTACKER_STEPS
    )


def train_discriminator(
    discriminator,
    optimizer,
    images,
    shares
):
    discriminator.train()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for share in shares:
        batch_size = images.shape[0]

        permutation = torch.randperm(
            batch_size,
            device=DEVICE
        )

        mismatched_images = images[
            permutation
        ]

        positive_logits = discriminator(
            images,
            share.detach()
        )

        negative_logits = discriminator(
            mismatched_images,
            share.detach()
        )

        positive_targets = torch.ones_like(
            positive_logits
        )

        negative_targets = torch.zeros_like(
            negative_logits
        )

        positive_loss = (
            F.binary_cross_entropy_with_logits(
                positive_logits,
                positive_targets
            )
        )

        negative_loss = (
            F.binary_cross_entropy_with_logits(
                negative_logits,
                negative_targets
            )
        )

        loss = (
            positive_loss +
            negative_loss
        ) * 0.5

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        positive_predictions = (
            torch.sigmoid(
                positive_logits
            ) >= 0.5
        )

        negative_predictions = (
            torch.sigmoid(
                negative_logits
            ) < 0.5
        )

        total_correct += (
            positive_predictions.sum().item()
            +
            negative_predictions.sum().item()
        )

        total_examples += (
            2 * batch_size
        )

    return (
        total_loss / 4.0,
        total_correct / total_examples
    )


def calculate_privacy_gan_loss(
    discriminator,
    images,
    shares
):
    total_loss = 0.0

    for share in shares:
        logits = discriminator(
            images,
            share
        )

        targets = torch.zeros_like(
            logits
        )

        total_loss += (
            F.binary_cross_entropy_with_logits(
                logits,
                targets
            )
        )

    return total_loss / 4.0


def evaluate_reconstruction(
    encoder,
    decoder,
    test_loader
):
    encoder.eval()
    decoder.eval()

    total_loss = 0.0
    total_psnr = 0.0
    batches = 0

    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(DEVICE)

            shares = encoder(images)

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


def save_checkpoint(
    encoder,
    decoder,
    discriminator,
    gan_weight,
    epoch,
    psnr
):
    weight_name = (
        f"{gan_weight:.3f}"
    )

    directory = os.path.join(
        OUTPUT_DIR,
        f"gan_{weight_name}"
    )

    os.makedirs(
        directory,
        exist_ok=True
    )

    torch.save(
        encoder.state_dict(),
        os.path.join(
            directory,
            "encoder.pth"
        )
    )

    torch.save(
        decoder.state_dict(),
        os.path.join(
            directory,
            "decoder.pth"
        )
    )

    torch.save(
        discriminator.state_dict(),
        os.path.join(
            directory,
            "discriminator.pth"
        )
    )

    with open(
        os.path.join(
            directory,
            "info.json"
        ),
        "w"
    ) as file:
        json.dump(
            {
                "gan_weight": gan_weight,
                "privacy_weight": PRIVACY_WEIGHT,
                "best_epoch": epoch,
                "best_reconstruction_psnr": psnr
            },
            file,
            indent=4
        )


def run_experiment(
    gan_weight,
    train_loader,
    test_loader
):
    print()
    print(
        f"Starting GAN weight: "
        f"{gan_weight}"
    )
    print(
        f"Privacy weight: "
        f"{PRIVACY_WEIGHT}"
    )

    encoder = ShareEncoder().to(
        DEVICE
    )

    decoder = ShareDecoder().to(
        DEVICE
    )

    attackers = create_attackers()

    attacker_optimizers = (
        create_attacker_optimizers(
            attackers
        )
    )

    discriminator = (
        PrivacyDiscriminator()
        .to(DEVICE)
    )

    generator_optimizer = torch.optim.Adam(
        list(encoder.parameters())
        +
        list(decoder.parameters()),
        lr=GENERATOR_LR
    )

    discriminator_optimizer = (
        torch.optim.Adam(
            discriminator.parameters(),
            lr=DISCRIMINATOR_LR
        )
    )

    best_psnr = float("-inf")
    best_epoch = 0

    for epoch in range(EPOCHS):
        encoder.train()
        decoder.train()

        reconstruction_total = 0.0
        attack_total = 0.0
        privacy_total = 0.0
        discriminator_total = 0.0

        batches = 0

        for batch_index, (
            images,
            _
        ) in enumerate(train_loader):

            images = images.to(
                DEVICE
            )

            train_attackers(
                encoder,
                attackers,
                attacker_optimizers,
                images
            )

            shares = encoder(
                images
            )

            discriminator_loss, _ = (
                train_discriminator(
                    discriminator,
                    discriminator_optimizer,
                    images,
                    shares
                )
            )

            shares = encoder(
                images
            )

            reconstructed = decoder(
                *shares
            )

            reconstruction_loss = (
                F.mse_loss(
                    reconstructed,
                    images
                )
            )

            total_attack_loss = 0.0

            for share_index in range(4):
                attacked = attackers[
                    share_index
                ](
                    shares[share_index]
                )

                total_attack_loss += (
                    F.mse_loss(
                        attacked,
                        images
                    )
                )

            average_attack_loss = (
                total_attack_loss / 4.0
            )

            freeze_attackers(
                attackers
            )

            privacy_gan_loss = (
                calculate_privacy_gan_loss(
                    discriminator,
                    images,
                    shares
                )
            )

            generator_loss = (
                reconstruction_loss
                -
                PRIVACY_WEIGHT *
                average_attack_loss
                +
                gan_weight *
                privacy_gan_loss
            )

            generator_optimizer.zero_grad()

            generator_loss.backward()

            generator_optimizer.step()

            unfreeze_attackers(
                attackers
            )

            reconstruction_total += (
                reconstruction_loss.item()
            )

            attack_total += (
                average_attack_loss.item()
            )

            privacy_total += (
                privacy_gan_loss.item()
            )

            discriminator_total += (
                discriminator_loss
            )

            batches += 1

            if batch_index % 200 == 0:
                print(
                    f"GAN {gan_weight:.3f} | "
                    f"Epoch [{epoch + 1}/{EPOCHS}] "
                    f"Batch "
                    f"[{batch_index}/{len(train_loader)}] "
                    f"Recon: "
                    f"{reconstruction_loss.item():.6f} "
                    f"Attack: "
                    f"{average_attack_loss.item():.6f} "
                    f"GAN: "
                    f"{privacy_gan_loss.item():.6f}"
                )

        validation_loss, validation_psnr = (
            evaluate_reconstruction(
                encoder,
                decoder,
                test_loader
            )
        )

        average_reconstruction = (
            reconstruction_total / batches
        )

        average_attack = (
            attack_total / batches
        )

        average_privacy_gan = (
            privacy_total / batches
        )

        average_discriminator_loss = (
            discriminator_total / batches
        )

        print()
        print(
            f"GAN weight {gan_weight:.3f} | "
            f"Epoch [{epoch + 1}/{EPOCHS}]"
        )

        print(
            f"Train Reconstruction Loss: "
            f"{average_reconstruction:.6f}"
        )

        print(
            f"Train Attack Loss: "
            f"{average_attack:.6f}"
        )

        print(
            f"Train Privacy GAN Loss: "
            f"{average_privacy_gan:.6f}"
        )

        print(
            f"Train Discriminator Loss: "
            f"{average_discriminator_loss:.6f}"
        )

        print(
            f"Validation Reconstruction Loss: "
            f"{validation_loss:.6f}"
        )

        print(
            f"Validation Reconstruction PSNR: "
            f"{validation_psnr:.2f} dB"
        )

        if validation_psnr > best_psnr:
            best_psnr = validation_psnr
            best_epoch = epoch + 1

            save_checkpoint(
                encoder,
                decoder,
                discriminator,
                gan_weight,
                best_epoch,
                best_psnr
            )

            print(
                f"New best checkpoint saved."
            )

    result = {
        "gan_weight": gan_weight,
        "privacy_weight": PRIVACY_WEIGHT,
        "best_epoch": best_epoch,
        "best_reconstruction_psnr": best_psnr
    }

    del encoder
    del decoder
    del discriminator

    for attacker in attackers:
        del attacker

    if DEVICE.type == "mps":
        torch.mps.empty_cache()

    return result


def main():
    print("Device:", DEVICE)
    print()
    print(
        "GAN weight sweep"
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
    print(
        "Batch size:",
        BATCH_SIZE
    )
    print(
        "Privacy weight:",
        PRIVACY_WEIGHT
    )
    print(
        "GAN weights:",
        GAN_WEIGHTS
    )
    print(
        "Attacker steps:",
        ATTACKER_STEPS
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

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    results = []

    for gan_weight in GAN_WEIGHTS:
        result = run_experiment(
            gan_weight,
            train_loader,
            test_loader
        )

        results.append(
            result
        )

    results_path = os.path.join(
        OUTPUT_DIR,
        "results.json"
    )

    with open(
        results_path,
        "w"
    ) as file:
        json.dump(
            results,
            file,
            indent=4
        )

    print()
    print("=" * 70)
    print(
        "GAN WEIGHT SWEEP COMPLETE"
    )
    print("=" * 70)

    print(
        f"{'GAN Weight':<15}"
        f"{'Best Epoch':<15}"
        f"{'Recon PSNR':<15}"
    )

    print("-" * 45)

    for result in results:
        print(
            f"{result['gan_weight']:<15.3f}"
            f"{result['best_epoch']:<15}"
            f"{result['best_reconstruction_psnr']:<15.2f}"
        )

    print("=" * 70)

    print(
        f"Results saved to: "
        f"{results_path}"
    )


if __name__ == "__main__":
    main()