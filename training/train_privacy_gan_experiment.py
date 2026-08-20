import os
import math
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

LEARNING_RATE_GENERATOR = 0.0001
LEARNING_RATE_ATTACKER = 0.0002
LEARNING_RATE_DISCRIMINATOR = 0.0002

PRIVACY_WEIGHT = 0.10
GAN_WEIGHT = 0.01

ATTACKER_STEPS = 2

CHECKPOINT_DIR = "checkpoints/privacy_gan"

DEVICE = (
    torch.device("mps")
    if torch.backends.mps.is_available()
    else torch.device("cpu")
)


def calculate_psnr(original, reconstructed):
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


def create_attacker_optimizers(attackers):
    return [
        torch.optim.Adam(
            attacker.parameters(),
            lr=LEARNING_RATE_ATTACKER
        )
        for attacker in attackers
    ]


def freeze_attackers(attackers):
    for attacker in attackers:
        for parameter in attacker.parameters():
            parameter.requires_grad = False


def unfreeze_attackers(attackers):
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
            attacked = attacker(
                shares[share_index]
            )

            loss = F.mse_loss(
                attacked,
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

    average_loss = total_loss / 4.0
    accuracy = total_correct / total_examples

    return average_loss, accuracy


def calculate_generator_privacy_loss(
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
    epoch,
    psnr
):
    os.makedirs(
        CHECKPOINT_DIR,
        exist_ok=True
    )

    torch.save(
        encoder.state_dict(),
        f"{CHECKPOINT_DIR}/encoder_best.pth"
    )

    torch.save(
        decoder.state_dict(),
        f"{CHECKPOINT_DIR}/decoder_best.pth"
    )

    torch.save(
        discriminator.state_dict(),
        f"{CHECKPOINT_DIR}/discriminator_best.pth"
    )

    with open(
        f"{CHECKPOINT_DIR}/best_info.txt",
        "w"
    ) as file:
        file.write(
            f"Best epoch: {epoch}\n"
        )

        file.write(
            f"Best reconstruction PSNR: "
            f"{psnr:.6f} dB\n"
        )

        file.write(
            f"Privacy weight: "
            f"{PRIVACY_WEIGHT}\n"
        )

        file.write(
            f"GAN weight: "
            f"{GAN_WEIGHT}\n"
        )


def main():
    print("Device:", DEVICE)
    print()
    print("Privacy GAN experiment")
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
        "GAN weight:",
        GAN_WEIGHT
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

    encoder = ShareEncoder().to(
        DEVICE
    )

    decoder = ShareDecoder().to(
        DEVICE
    )

    encoder.load_state_dict(
        torch.load(
            "checkpoints/privacy_sweep/"
            "lambda_0.10/encoder.pth",
            map_location=DEVICE
        )
    )

    decoder.load_state_dict(
        torch.load(
            "checkpoints/privacy_sweep/"
            "lambda_0.10/decoder.pth",
            map_location=DEVICE
        )
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
        lr=LEARNING_RATE_GENERATOR
    )

    discriminator_optimizer = (
        torch.optim.Adam(
            discriminator.parameters(),
            lr=LEARNING_RATE_DISCRIMINATOR
        )
    )

    best_psnr = float("-inf")
    best_epoch = 0

    results = []

    for epoch in range(EPOCHS):
        encoder.train()
        decoder.train()

        reconstruction_total = 0.0
        attack_total = 0.0
        privacy_total = 0.0
        discriminator_total = 0.0
        discriminator_accuracy_total = 0.0

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

            discriminator_loss_value, discriminator_accuracy = (
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

            reconstruction_loss = F.mse_loss(
                reconstructed,
                images
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

            privacy_loss = (
                calculate_generator_privacy_loss(
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
                GAN_WEIGHT *
                privacy_loss
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
                privacy_loss.item()
            )

            discriminator_total += (
                discriminator_loss_value
            )

            discriminator_accuracy_total += (
                discriminator_accuracy
            )

            batches += 1

            if batch_index % 100 == 0:
                print(
                    f"Epoch [{epoch + 1}/{EPOCHS}] "
                    f"Batch "
                    f"[{batch_index}/{len(train_loader)}] "
                    f"Recon Loss: "
                    f"{reconstruction_loss.item():.6f} "
                    f"Attack Loss: "
                    f"{average_attack_loss.item():.6f} "
                    f"GAN Loss: "
                    f"{privacy_loss.item():.6f} "
                    f"D Loss: "
                    f"{discriminator_loss_value:.6f}"
                )

        validation_loss, validation_psnr = (
            evaluate_reconstruction(
                encoder,
                decoder,
                test_loader
            )
        )

        train_reconstruction = (
            reconstruction_total / batches
        )

        train_attack = (
            attack_total / batches
        )

        train_privacy = (
            privacy_total / batches
        )

        average_discriminator_loss = (
            discriminator_total / batches
        )

        train_discriminator_accuracy = (
            discriminator_accuracy_total
            / batches
        )

        results.append({
            "epoch": epoch + 1,
            "train_reconstruction":
                train_reconstruction,
            "train_attack":
                train_attack,
            "train_privacy":
                train_privacy,
            "train_discriminator":
                average_discriminator_loss,
            "discriminator_accuracy":
                train_discriminator_accuracy,
            "validation_loss":
                validation_loss,
            "validation_psnr":
                validation_psnr
        })

        print()
        print("=" * 60)

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}]"
        )

        print(
            f"Train Reconstruction Loss: "
            f"{train_reconstruction:.6f}"
        )

        print(
            f"Train Attack Loss: "
            f"{train_attack:.6f}"
        )

        print(
            f"Train Privacy GAN Loss: "
            f"{train_privacy:.6f}"
        )

        print(
            f"Train Discriminator Loss: "
            f"{average_discriminator_loss:.6f}"
        )

        print(
            f"Train Discriminator Accuracy: "
            f"{train_discriminator_accuracy * 100:.2f}%"
        )

        print(
            f"Validation Reconstruction Loss: "
            f"{validation_loss:.6f}"
        )

        print(
            f"Validation Reconstruction PSNR: "
            f"{validation_psnr:.2f} dB"
        )

        print("=" * 60)

        if validation_psnr > best_psnr:
            best_psnr = validation_psnr
            best_epoch = epoch + 1

            save_checkpoint(
                encoder,
                decoder,
                discriminator,
                best_epoch,
                best_psnr
            )

            print()
            print(
                f"New best model saved "
                f"(PSNR: {best_psnr:.2f} dB)"
            )

    os.makedirs(
        CHECKPOINT_DIR,
        exist_ok=True
    )

    with open(
        f"{CHECKPOINT_DIR}/training_log.txt",
        "w"
    ) as file:
        file.write(
            "Privacy GAN experiment\n"
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
            f"Epochs: {EPOCHS}\n"
        )

        file.write(
            f"Privacy weight: "
            f"{PRIVACY_WEIGHT}\n"
        )

        file.write(
            f"GAN weight: "
            f"{GAN_WEIGHT}\n"
        )

        file.write(
            f"Attacker steps: "
            f"{ATTACKER_STEPS}\n"
        )

        file.write("\n")

        for result in results:
            file.write(
                f"Epoch {result['epoch']}\n"
            )

            file.write(
                f"Train Reconstruction Loss: "
                f"{result['train_reconstruction']:.6f}\n"
            )

            file.write(
                f"Train Attack Loss: "
                f"{result['train_attack']:.6f}\n"
            )

            file.write(
                f"Train Privacy GAN Loss: "
                f"{result['train_privacy']:.6f}\n"
            )

            file.write(
                f"Train Discriminator Loss: "
                f"{result['train_discriminator']:.6f}\n"
            )

            file.write(
                f"Discriminator Accuracy: "
                f"{result['discriminator_accuracy'] * 100:.2f}%\n"
            )

            file.write(
                f"Validation Reconstruction Loss: "
                f"{result['validation_loss']:.6f}\n"
            )

            file.write(
                f"Validation Reconstruction PSNR: "
                f"{result['validation_psnr']:.4f} dB\n"
            )

            file.write("\n")

        file.write(
            f"Best epoch: {best_epoch}\n"
        )

        file.write(
            f"Best reconstruction PSNR: "
            f"{best_psnr:.4f} dB\n"
        )

    print()
    print("=" * 60)
    print("PRIVACY GAN EXPERIMENT COMPLETE")
    print("=" * 60)
    print(
        f"Best epoch: {best_epoch}"
    )
    print(
        f"Best reconstruction PSNR: "
        f"{best_psnr:.2f} dB"
    )
    print(
        f"Saved to: {CHECKPOINT_DIR}"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()