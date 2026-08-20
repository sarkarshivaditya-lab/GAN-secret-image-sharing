import os
import math
import random
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from models.encoder import ShareEncoder
from models.decoder import ShareDecoder
from models.attacker import ShareAttacker


QUICK_TEST = True

TRAIN_IMAGES = 5000
TEST_IMAGES = 1000
EPOCHS = 3
BATCH_SIZE = 8

LEARNING_RATE_ENCODER = 0.0001
LEARNING_RATE_ATTACKER = 0.0002

PRIVACY_WEIGHT = 0.5


def calculate_psnr(original, reconstructed):
    mse = torch.mean((original - reconstructed) ** 2)

    if mse.item() == 0:
        return float("inf")

    return 10 * math.log10(1.0 / mse.item())


def evaluate(
    encoder,
    decoder,
    attacker,
    test_loader,
    device
):
    encoder.eval()
    decoder.eval()
    attacker.eval()

    reconstruction_loss_total = 0.0
    attack_loss_total = 0.0

    reconstruction_psnr_total = 0.0
    attack_psnr_total = 0.0

    batches = 0

    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)

            shares = encoder(images)

            reconstructed = decoder(*shares)

            share_index = random.randint(0, 3)
            selected_share = shares[share_index]

            attacked = attacker(selected_share)

            reconstruction_loss = F.mse_loss(
                reconstructed,
                images
            )

            attack_loss = F.mse_loss(
                attacked,
                images
            )

            reconstruction_psnr = calculate_psnr(
                images,
                reconstructed
            )

            attack_psnr = calculate_psnr(
                images,
                attacked
            )

            reconstruction_loss_total += reconstruction_loss.item()
            attack_loss_total += attack_loss.item()

            reconstruction_psnr_total += reconstruction_psnr
            attack_psnr_total += attack_psnr

            batches += 1

    return (
        reconstruction_loss_total / batches,
        attack_loss_total / batches,
        reconstruction_psnr_total / batches,
        attack_psnr_total / batches
    )


def main():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print("Device:", device)

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

    if QUICK_TEST:
        train_dataset = Subset(
            train_dataset,
            range(TRAIN_IMAGES)
        )

        test_dataset = Subset(
            test_dataset,
            range(TEST_IMAGES)
        )

        print()
        print("QUICK ADVERSARIAL TEST")
        print("Training images:", TRAIN_IMAGES)
        print("Test images:", TEST_IMAGES)
        print("Epochs:", EPOCHS)
        print("Batch size:", BATCH_SIZE)
        print("Privacy weight:", PRIVACY_WEIGHT)
        print()

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

    encoder = ShareEncoder().to(device)
    decoder = ShareDecoder().to(device)
    attacker = ShareAttacker().to(device)

    encoder.load_state_dict(
        torch.load(
            "checkpoints/encoder_baseline.pth",
            map_location=device
        )
    )

    decoder.load_state_dict(
        torch.load(
            "checkpoints/decoder_baseline.pth",
            map_location=device
        )
    )

    attacker.load_state_dict(
        torch.load(
            "checkpoints/attackers/attacker_share_1.pth",
            map_location=device
        )
    )

    encoder.train()
    decoder.train()
    attacker.train()

    encoder_optimizer = torch.optim.Adam(
        list(encoder.parameters()) +
        list(decoder.parameters()),
        lr=LEARNING_RATE_ENCODER
    )

    attacker_optimizer = torch.optim.Adam(
        attacker.parameters(),
        lr=LEARNING_RATE_ATTACKER
    )

    os.makedirs(
        "checkpoints/adversarial",
        exist_ok=True
    )

    best_reconstruction_psnr = float("-inf")

    for epoch in range(EPOCHS):
        encoder.train()
        decoder.train()
        attacker.train()

        reconstruction_total = 0.0
        attack_total = 0.0
        batches = 0

        for batch_index, (images, _) in enumerate(train_loader):
            images = images.to(device)

            with torch.no_grad():
                shares = encoder(images)

            share_index = random.randint(0, 3)
            selected_share = shares[share_index]

            attacked = attacker(selected_share)

            attacker_loss = F.mse_loss(
                attacked,
                images
            )

            attacker_optimizer.zero_grad()

            attacker_loss.backward()

            attacker_optimizer.step()

            shares = encoder(images)

            reconstructed = decoder(*shares)

            share_index = random.randint(0, 3)
            selected_share = shares[share_index]

            attacked = attacker(selected_share)

            reconstruction_loss = F.mse_loss(
                reconstructed,
                images
            )

            attack_loss = F.mse_loss(
                attacked,
                images
            )

            encoder_loss = (
                reconstruction_loss
                - PRIVACY_WEIGHT * attack_loss
            )

            encoder_optimizer.zero_grad()

            encoder_loss.backward()

            encoder_optimizer.step()

            reconstruction_total += reconstruction_loss.item()
            attack_total += attack_loss.item()

            batches += 1

            if batch_index % 100 == 0:
                print(
                    f"Epoch [{epoch + 1}/{EPOCHS}] "
                    f"Batch [{batch_index}/{len(train_loader)}] "
                    f"Recon Loss: {reconstruction_loss.item():.6f} "
                    f"Attack Loss: {attack_loss.item():.6f}"
                )

        average_reconstruction = (
            reconstruction_total / batches
        )

        average_attack = (
            attack_total / batches
        )

        (
            validation_reconstruction,
            validation_attack,
            reconstruction_psnr,
            attack_psnr
        ) = evaluate(
            encoder,
            decoder,
            attacker,
            test_loader,
            device
        )

        print()
        print("=" * 60)
        print(f"Epoch [{epoch + 1}/{EPOCHS}]")
        print(
            f"Train Reconstruction Loss: "
            f"{average_reconstruction:.6f}"
        )
        print(
            f"Train Attack Loss: "
            f"{average_attack:.6f}"
        )
        print(
            f"Validation Reconstruction Loss: "
            f"{validation_reconstruction:.6f}"
        )
        print(
            f"Validation Attack Loss: "
            f"{validation_attack:.6f}"
        )
        print(
            f"Reconstruction PSNR: "
            f"{reconstruction_psnr:.2f} dB"
        )
        print(
            f"Single-share Attack PSNR: "
            f"{attack_psnr:.2f} dB"
        )
        print("=" * 60)
        print()

        if reconstruction_psnr > best_reconstruction_psnr:
            best_reconstruction_psnr = reconstruction_psnr

            torch.save(
                encoder.state_dict(),
                "checkpoints/adversarial/encoder_adversarial.pth"
            )

            torch.save(
                decoder.state_dict(),
                "checkpoints/adversarial/decoder_adversarial.pth"
            )

            torch.save(
                attacker.state_dict(),
                "checkpoints/adversarial/attacker_adversarial.pth"
            )

            print("New best adversarial model saved.")
            print()

    print()
    print("ADVERSARIAL TRAINING COMPLETE")
    print(
        "Best reconstruction PSNR:",
        f"{best_reconstruction_psnr:.2f} dB"
    )


if __name__ == "__main__":
    main()