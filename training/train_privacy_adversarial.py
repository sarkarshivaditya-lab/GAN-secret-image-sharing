import os
import math
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

ATTACKER_STEPS = 2


def calculate_psnr(original, reconstructed):
    mse = torch.mean((original - reconstructed) ** 2)

    if mse.item() == 0:
        return float("inf")

    return 10 * math.log10(1.0 / mse.item())


def create_attacker(device):
    return ShareAttacker().to(device)


def evaluate(
    encoder,
    decoder,
    attackers,
    test_loader,
    device
):
    encoder.eval()
    decoder.eval()

    for attacker in attackers:
        attacker.eval()

    reconstruction_loss_total = 0.0
    reconstruction_psnr_total = 0.0

    attack_losses = [0.0, 0.0, 0.0, 0.0]
    attack_psnrs = [0.0, 0.0, 0.0, 0.0]

    batches = 0

    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)

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
                share = shares[share_index]

                attacked = attackers[share_index](share)

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

    average_reconstruction_loss = (
        reconstruction_loss_total / batches
    )

    average_reconstruction_psnr = (
        reconstruction_psnr_total / batches
    )

    average_attack_losses = [
        loss / batches
        for loss in attack_losses
    ]

    average_attack_psnrs = [
        psnr / batches
        for psnr in attack_psnrs
    ]

    return (
        average_reconstruction_loss,
        average_reconstruction_psnr,
        average_attack_losses,
        average_attack_psnrs
    )


def train_attackers(
    encoder,
    attackers,
    attacker_optimizers,
    images,
    device
):
    encoder.eval()

    with torch.no_grad():
        shares = encoder(images)

    total_attack_loss = 0.0

    for share_index in range(4):
        attacker = attackers[share_index]
        optimizer = attacker_optimizers[share_index]

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

            total_attack_loss += loss.item()

    return total_attack_loss / (
        4 * ATTACKER_STEPS
    )


def freeze_attacker_parameters(attackers):
    for attacker in attackers:
        for parameter in attacker.parameters():
            parameter.requires_grad = False


def unfreeze_attacker_parameters(attackers):
    for attacker in attackers:
        for parameter in attacker.parameters():
            parameter.requires_grad = True


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
        print("QUICK PRIVACY ADVERSARIAL TEST")
        print("Training images:", TRAIN_IMAGES)
        print("Test images:", TEST_IMAGES)
        print("Epochs:", EPOCHS)
        print("Batch size:", BATCH_SIZE)
        print("Privacy weight:", PRIVACY_WEIGHT)
        print("Attacker steps:", ATTACKER_STEPS)
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

    attackers = [
        create_attacker(device)
        for _ in range(4)
    ]

    encoder.train()
    decoder.train()

    for attacker in attackers:
        attacker.train()

    attacker_optimizers = [
        torch.optim.Adam(
            attacker.parameters(),
            lr=LEARNING_RATE_ATTACKER
        )
        for attacker in attackers
    ]

    encoder_optimizer = torch.optim.Adam(
        list(encoder.parameters()) +
        list(decoder.parameters()),
        lr=LEARNING_RATE_ENCODER
    )

    os.makedirs(
        "checkpoints/privacy_adversarial",
        exist_ok=True
    )

    best_reconstruction_psnr = float("-inf")

    for epoch in range(EPOCHS):
        encoder.train()
        decoder.train()

        for attacker in attackers:
            attacker.train()

        reconstruction_total = 0.0
        attack_total = 0.0
        batches = 0

        for batch_index, (images, _) in enumerate(train_loader):
            images = images.to(device)

            attack_loss = train_attackers(
                encoder,
                attackers,
                attacker_optimizers,
                images,
                device
            )

            unfreeze_attacker_parameters(
                attackers
            )

            shares = encoder(images)

            reconstructed = decoder(*shares)

            reconstruction_loss = F.mse_loss(
                reconstructed,
                images
            )

            total_attack_loss = 0.0

            for share_index in range(4):
                share = shares[share_index]

                attacked = attackers[share_index](share)

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
                PRIVACY_WEIGHT * average_attack_loss
            )

            encoder_optimizer.zero_grad()

            for optimizer in attacker_optimizers:
                optimizer.zero_grad()

            freeze_attacker_parameters(
                attackers
            )

            encoder_loss.backward()

            encoder_optimizer.step()

            unfreeze_attacker_parameters(
                attackers
            )

            reconstruction_total += (
                reconstruction_loss.item()
            )

            attack_total += (
                average_attack_loss.item()
            )

            batches += 1

            if batch_index % 100 == 0:
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

        (
            validation_reconstruction_loss,
            reconstruction_psnr,
            validation_attack_losses,
            validation_attack_psnrs
        ) = evaluate(
            encoder,
            decoder,
            attackers,
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
            f"Train Average Attack Loss: "
            f"{average_attack:.6f}"
        )
        print(
            f"Validation Reconstruction Loss: "
            f"{validation_reconstruction_loss:.6f}"
        )
        print(
            f"Reconstruction PSNR: "
            f"{reconstruction_psnr:.2f} dB"
        )

        for share_index in range(4):
            print(
                f"Share {share_index + 1} "
                f"Attack Loss: "
                f"{validation_attack_losses[share_index]:.6f}"
            )

            print(
                f"Share {share_index + 1} "
                f"Attack PSNR: "
                f"{validation_attack_psnrs[share_index]:.2f} dB"
            )

        print("=" * 60)
        print()

        if reconstruction_psnr > best_reconstruction_psnr:
            best_reconstruction_psnr = reconstruction_psnr

            torch.save(
                encoder.state_dict(),
                "checkpoints/privacy_adversarial/"
                "encoder_privacy_adversarial.pth"
            )

            torch.save(
                decoder.state_dict(),
                "checkpoints/privacy_adversarial/"
                "decoder_privacy_adversarial.pth"
            )

            for share_index in range(4):
                torch.save(
                    attackers[share_index].state_dict(),
                    "checkpoints/privacy_adversarial/"
                    f"attacker_share_{share_index + 1}.pth"
                )

            print("New best privacy-adversarial model saved.")
            print()

    print()
    print("PRIVACY ADVERSARIAL TRAINING COMPLETE")
    print(
        "Best reconstruction PSNR:",
        f"{best_reconstruction_psnr:.2f} dB"
    )


if __name__ == "__main__":
    main()