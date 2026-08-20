import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from models.encoder import ShareEncoder
from models.decoder import ShareDecoder
from models.attacker import ShareAttacker
from models.discriminator import ShareDiscriminator

from training.gan_losses import (
    make_real_share_distribution,
    discriminator_loss,
    generator_gan_loss
)


TRAIN_IMAGES = 500
TEST_IMAGES = 100

EPOCHS = 1
BATCH_SIZE = 8

LEARNING_RATE_GENERATOR = 0.0001
LEARNING_RATE_ATTACKER = 0.0002
LEARNING_RATE_DISCRIMINATOR = 0.0002

PRIVACY_WEIGHT = 0.10
GAN_WEIGHT = 0.01

ATTACKER_STEPS = 2

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

    return 10 * torch.log10(
        torch.tensor(
            1.0,
            device=DEVICE
        ) / mse
    ).item()


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
            share = shares[share_index]

            attacked = attacker(share)

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
    discriminator_optimizer,
    shares
):
    discriminator.train()

    total_loss = 0.0

    for share in shares:
        batch_size = share.shape[0]

        real = make_real_share_distribution(
            batch_size,
            3,
            32,
            32,
            DEVICE
        )

        real_logits = discriminator(
            real
        )

        fake_logits = discriminator(
            share.detach()
        )

        loss = discriminator_loss(
            real_logits,
            fake_logits
        )

        discriminator_optimizer.zero_grad()

        loss.backward()

        discriminator_optimizer.step()

        total_loss += loss.item()

    return total_loss / 4.0


def main():
    print("Device:", DEVICE)
    print()
    print("GAN integration sanity test")
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

    encoder = ShareEncoder().to(DEVICE)
    decoder = ShareDecoder().to(DEVICE)

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
        ShareDiscriminator()
        .to(DEVICE)
    )

    generator_optimizer = torch.optim.Adam(
        list(encoder.parameters()) +
        list(decoder.parameters()),
        lr=LEARNING_RATE_GENERATOR
    )

    discriminator_optimizer = (
        torch.optim.Adam(
            discriminator.parameters(),
            lr=LEARNING_RATE_DISCRIMINATOR
        )
    )

    encoder.train()
    decoder.train()

    os.makedirs(
        "checkpoints/gan",
        exist_ok=True
    )

    for epoch in range(EPOCHS):
        reconstruction_total = 0.0
        attack_total = 0.0
        gan_total = 0.0
        discriminator_total = 0.0

        batches = 0

        for batch_index, (
            images,
            _
        ) in enumerate(train_loader):

            images = images.to(DEVICE)

            attack_loss = train_attackers(
                encoder,
                attackers,
                attacker_optimizers,
                images
            )

            shares = encoder(images)

            discriminator_loss_value = (
                train_discriminator(
                    discriminator,
                    discriminator_optimizer,
                    shares
                )
            )

            shares = encoder(images)

            reconstructed = decoder(*shares)

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

            total_fake_logits = 0.0

            for share in shares:
                fake_logits = discriminator(
                    share
                )

                total_fake_logits += (
                    generator_gan_loss(
                        fake_logits
                    )
                )

            average_gan_loss = (
                total_fake_logits / 4.0
            )

            generator_loss = (
                reconstruction_loss
                -
                PRIVACY_WEIGHT *
                average_attack_loss
                +
                GAN_WEIGHT *
                average_gan_loss
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

            gan_total += (
                average_gan_loss.item()
            )

            discriminator_total += (
                discriminator_loss_value
            )

            batches += 1

            if batch_index % 25 == 0:
                print()
                print(
                    f"Epoch [{epoch + 1}/{EPOCHS}] "
                    f"Batch "
                    f"[{batch_index}/{len(train_loader)}]"
                )

                print(
                    f"Reconstruction Loss: "
                    f"{reconstruction_loss.item():.6f}"
                )

                print(
                    f"Attack Loss: "
                    f"{average_attack_loss.item():.6f}"
                )

                print(
                    f"Generator GAN Loss: "
                    f"{average_gan_loss.item():.6f}"
                )

                print(
                    f"Discriminator Loss: "
                    f"{discriminator_loss_value:.6f}"
                )

        average_reconstruction = (
            reconstruction_total / batches
        )

        average_attack = (
            attack_total / batches
        )

        average_gan = (
            gan_total / batches
        )

        average_discriminator = (
            discriminator_total / batches
        )

        encoder.eval()
        decoder.eval()

        validation_reconstruction = 0.0
        validation_psnr = 0.0
        validation_batches = 0

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

                validation_reconstruction += (
                    loss.item()
                )

                validation_psnr += psnr

                validation_batches += 1

        validation_reconstruction /= (
            validation_batches
        )

        validation_psnr /= (
            validation_batches
        )

        print()
        print(
            "=" * 60
        )

        print(
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
            f"Train Generator GAN Loss: "
            f"{average_gan:.6f}"
        )

        print(
            f"Train Discriminator Loss: "
            f"{average_discriminator:.6f}"
        )

        print(
            f"Validation Reconstruction Loss: "
            f"{validation_reconstruction:.6f}"
        )

        print(
            f"Validation Reconstruction PSNR: "
            f"{validation_psnr:.2f} dB"
        )

        print(
            "=" * 60
        )

    torch.save(
        encoder.state_dict(),
        "checkpoints/gan/encoder_gan_test.pth"
    )

    torch.save(
        decoder.state_dict(),
        "checkpoints/gan/decoder_gan_test.pth"
    )

    torch.save(
        discriminator.state_dict(),
        "checkpoints/gan/discriminator_test.pth"
    )

    print()
    print(
        "GAN integration sanity test complete."
    )

    print(
        "Checkpoints saved in:"
    )

    print(
        "checkpoints/gan/"
    )


if __name__ == "__main__":
    main()