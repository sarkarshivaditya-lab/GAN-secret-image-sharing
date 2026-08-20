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

BATCH_SIZE = 8
EPOCHS = 3

LEARNING_RATE = 0.0002
ATTACKER_LEARNING_RATE = 0.0002

PRIVACY_WEIGHT = 0.1
GAN_WEIGHT = 0.005

ATTACKER_STEPS = 2

WORST_SHARE_WEIGHT = 0.75
AVERAGE_SHARE_WEIGHT = 0.25

CHECKPOINT_DIR = "checkpoints/worst_share_privacy"

DEVICE = (
    torch.device("mps")
    if torch.backends.mps.is_available()
    else torch.device("cpu")
)


def calculate_psnr(original, reconstructed):
    mse = torch.mean(
        (original - reconstructed) ** 2
    )

    if mse.item() <= 0:
        return float("inf")

    return 10 * math.log10(
        1.0 / mse.item()
    )


def load_base_models():
    encoder = ShareEncoder().to(DEVICE)
    decoder = ShareDecoder().to(DEVICE)

    encoder.load_state_dict(
        torch.load(
            "checkpoints/privacy_gan/encoder_best.pth",
            map_location=DEVICE
        )
    )

    decoder.load_state_dict(
        torch.load(
            "checkpoints/privacy_gan/decoder_best.pth",
            map_location=DEVICE
        )
    )

    return encoder, decoder


def train_attackers(
    encoder,
    images,
    attackers,
    attacker_optimizers
):
    encoder.eval()

    with torch.no_grad():
        shares = encoder(images)

    attack_losses = []

    for share_index in range(4):
        attacker = attackers[share_index]
        optimizer = attacker_optimizers[share_index]

        attacker.train()

        share = shares[share_index].detach()

        for _ in range(ATTACKER_STEPS):
            reconstructed = attacker(share)

            loss = F.mse_loss(
                reconstructed,
                images
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            reconstructed = attacker(share)

            attack_loss = F.mse_loss(
                reconstructed,
                images
            )

        attack_losses.append(
            attack_loss
        )

    return shares, attack_losses


def calculate_privacy_loss(
    encoder,
    images,
    attackers
):
    shares = encoder(images)

    attack_losses = []

    for share_index in range(4):
        attacker = attackers[share_index]

        attacker.eval()

        reconstructed = attacker(
            shares[share_index]
        )

        attack_loss = F.mse_loss(
            reconstructed,
            images
        )

        attack_losses.append(
            attack_loss
        )

    stacked = torch.stack(
        attack_losses
    )

    average_attack_loss = torch.mean(
        stacked
    )

    worst_attack_loss = torch.max(
        stacked
    )

    privacy_loss = (
        AVERAGE_SHARE_WEIGHT * average_attack_loss
        + WORST_SHARE_WEIGHT * worst_attack_loss
    )

    return (
        privacy_loss,
        average_attack_loss,
        worst_attack_loss,
        attack_losses,
        shares
    )


def train_discriminator(
    shares,
    discriminator,
    optimizer
):
    discriminator.train()

    real = torch.randn_like(
        shares[0]
    )

    fake = shares[0].detach()

    real_logits = discriminator(
        real
    )

    fake_logits = discriminator(
        fake
    )

    real_targets = torch.ones_like(
        real_logits
    )

    fake_targets = torch.zeros_like(
        fake_logits
    )

    real_loss = F.binary_cross_entropy_with_logits(
        real_logits,
        real_targets
    )

    fake_loss = F.binary_cross_entropy_with_logits(
        fake_logits,
        fake_targets
    )

    discriminator_loss = (
        real_loss + fake_loss
    ) / 2.0

    optimizer.zero_grad()
    discriminator_loss.backward()
    optimizer.step()

    with torch.no_grad():
        real_predictions = (
            torch.sigmoid(real_logits) >= 0.5
        ).float()

        fake_predictions = (
            torch.sigmoid(fake_logits) < 0.5
        ).float()

        correct = (
            real_predictions.sum()
            + fake_predictions.sum()
        )

        total = (
            real_predictions.numel()
            + fake_predictions.numel()
        )

        accuracy = (
            correct / total
        ).item()

    return (
        discriminator_loss.item(),
        accuracy
    )


def generator_gan_loss(
    shares,
    discriminator
):
    losses = []

    for share in shares:
        logits = discriminator(
            share
        )

        targets = torch.ones_like(
            logits
        )

        loss = F.binary_cross_entropy_with_logits(
            logits,
            targets
        )

        losses.append(loss)

    return torch.stack(
        losses
    ).mean()


def build_discriminator():
    try:
        from models.discriminator import ShareDiscriminator

        return ShareDiscriminator().to(
            DEVICE
        )

    except ImportError:
        try:
            from models.privacy_discriminator import PrivacyDiscriminator

            return PrivacyDiscriminator().to(
                DEVICE
            )

        except ImportError:
            return None


def evaluate_model(
    encoder,
    decoder,
    attackers,
    test_loader
):
    encoder.eval()
    decoder.eval()

    for attacker in attackers:
        attacker.eval()

    total_reconstruction_loss = 0.0
    total_reconstruction_psnr = 0.0

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
            images = images.to(
                DEVICE
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

            reconstruction_psnr = calculate_psnr(
                images,
                reconstructed
            )

            total_reconstruction_loss += (
                reconstruction_loss.item()
            )

            total_reconstruction_psnr += (
                reconstruction_psnr
            )

            for share_index in range(4):
                attack_reconstruction = attackers[
                    share_index
                ](
                    shares[share_index]
                )

                attack_loss = F.mse_loss(
                    attack_reconstruction,
                    images
                )

                attack_psnr = calculate_psnr(
                    images,
                    attack_reconstruction
                )

                attack_losses[
                    share_index
                ] += attack_loss.item()

                attack_psnrs[
                    share_index
                ] += attack_psnr

            batches += 1

    attack_losses = [
        value / batches
        for value in attack_losses
    ]

    attack_psnrs = [
        value / batches
        for value in attack_psnrs
    ]

    return {
        "reconstruction_loss":
            total_reconstruction_loss / batches,
        "reconstruction_psnr":
            total_reconstruction_psnr / batches,
        "attack_losses":
            attack_losses,
        "attack_psnrs":
            attack_psnrs,
        "average_attack_psnr":
            sum(attack_psnrs) / 4.0,
        "worst_attack_psnr":
            max(attack_psnrs)
    }


def save_checkpoint(
    encoder,
    decoder,
    attackers,
    epoch,
    metrics
):
    os.makedirs(
        CHECKPOINT_DIR,
        exist_ok=True
    )

    torch.save(
        encoder.state_dict(),
        os.path.join(
            CHECKPOINT_DIR,
            "encoder_best.pth"
        )
    )

    torch.save(
        decoder.state_dict(),
        os.path.join(
            CHECKPOINT_DIR,
            "decoder_best.pth"
        )
    )

    for share_index, attacker in enumerate(
        attackers
    ):
        torch.save(
            attacker.state_dict(),
            os.path.join(
                CHECKPOINT_DIR,
                f"attacker_share_{share_index + 1}.pth"
            )
        )

    with open(
        os.path.join(
            CHECKPOINT_DIR,
            "best_info.txt"
        ),
        "w"
    ) as file:
        file.write(
            f"Best epoch: {epoch}\n"
        )

        file.write(
            f"Reconstruction PSNR: "
            f"{metrics['reconstruction_psnr']:.4f}\n"
        )

        file.write(
            f"Average attack PSNR: "
            f"{metrics['average_attack_psnr']:.4f}\n"
        )

        file.write(
            f"Worst attack PSNR: "
            f"{metrics['worst_attack_psnr']:.4f}\n"
        )

        for index, psnr in enumerate(
            metrics["attack_psnrs"]
        ):
            file.write(
                f"Share {index + 1} attack PSNR: "
                f"{psnr:.4f}\n"
            )


def main():
    print("Device:", DEVICE)
    print()
    print("Worst-share privacy experiment")
    print("Training images:", TRAIN_IMAGES)
    print("Test images:", TEST_IMAGES)
    print("Epochs:", EPOCHS)
    print("Batch size:", BATCH_SIZE)
    print("Privacy weight:", PRIVACY_WEIGHT)
    print("GAN weight:", GAN_WEIGHT)
    print("Attacker steps:", ATTACKER_STEPS)
    print("Worst-share weight:", WORST_SHARE_WEIGHT)
    print("Average-share weight:", AVERAGE_SHARE_WEIGHT)
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

    encoder, decoder = load_base_models()

    attackers = []

    for _ in range(4):
        attackers.append(
            ShareAttacker().to(
                DEVICE
            )
        )

    attacker_optimizers = []

    for attacker in attackers:
        attacker_optimizers.append(
            torch.optim.Adam(
                attacker.parameters(),
                lr=ATTACKER_LEARNING_RATE
            )
        )

    encoder_optimizer = torch.optim.Adam(
        encoder.parameters(),
        lr=LEARNING_RATE
    )

    decoder_optimizer = torch.optim.Adam(
        decoder.parameters(),
        lr=LEARNING_RATE
    )

    discriminator = build_discriminator()

    discriminator_optimizer = None

    if discriminator is not None:
        discriminator_optimizer = torch.optim.Adam(
            discriminator.parameters(),
            lr=LEARNING_RATE
        )

        print(
            "GAN discriminator: enabled"
        )
    else:
        print(
            "GAN discriminator: not found"
        )
        print(
            "Continuing with worst-share privacy loss."
        )

    best_metrics = None
    best_epoch = 0

    for epoch in range(
        EPOCHS
    ):
        encoder.train()
        decoder.train()

        for attacker in attackers:
            attacker.train()

        total_reconstruction_loss = 0.0
        total_privacy_loss = 0.0
        total_average_attack_loss = 0.0
        total_worst_attack_loss = 0.0
        total_gan_loss = 0.0
        total_discriminator_loss = 0.0

        batches = 0

        for batch_index, (images, _) in enumerate(
            train_loader
        ):
            images = images.to(
                DEVICE
            )

            with torch.no_grad():
                shares_for_attack = encoder(
                    images
                )

            for share_index in range(4):
                attacker = attackers[
                    share_index
                ]

                optimizer = attacker_optimizers[
                    share_index
                ]

                share = shares_for_attack[
                    share_index
                ].detach()

                for _ in range(
                    ATTACKER_STEPS
                ):
                    attacker.train()

                    attacker_output = attacker(
                        share
                    )

                    attacker_loss = F.mse_loss(
                        attacker_output,
                        images
                    )

                    optimizer.zero_grad()
                    attacker_loss.backward()
                    optimizer.step()

            encoder.train()
            decoder.train()

            (
                privacy_loss,
                average_attack_loss,
                worst_attack_loss,
                _,
                shares
            ) = calculate_privacy_loss(
                encoder,
                images,
                attackers
            )

            reconstructed = decoder(
                *shares
            )

            reconstruction_loss = F.mse_loss(
                reconstructed,
                images
            )

            gan_loss = torch.tensor(
                0.0,
                device=DEVICE
            )

            discriminator_loss_value = 0.0

            if discriminator is not None:
                discriminator_loss_value, _ = (
                    train_discriminator(
                        shares,
                        discriminator,
                        discriminator_optimizer
                    )
                )

                gan_loss = generator_gan_loss(
                    shares,
                    discriminator
                )

            total_loss = (
                reconstruction_loss
                - PRIVACY_WEIGHT * privacy_loss
                + GAN_WEIGHT * gan_loss
            )

            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()

            total_loss.backward()

            torch.nn.utils.clip_grad_norm_(
                encoder.parameters(),
                1.0
            )

            torch.nn.utils.clip_grad_norm_(
                decoder.parameters(),
                1.0
            )

            encoder_optimizer.step()
            decoder_optimizer.step()

            total_reconstruction_loss += (
                reconstruction_loss.item()
            )

            total_privacy_loss += (
                privacy_loss.item()
            )

            total_average_attack_loss += (
                average_attack_loss.item()
            )

            total_worst_attack_loss += (
                worst_attack_loss.item()
            )

            total_gan_loss += (
                gan_loss.item()
            )

            total_discriminator_loss += (
                discriminator_loss_value
            )

            batches += 1

            if batch_index % 100 == 0:
                print(
                    f"Epoch [{epoch + 1}/{EPOCHS}] "
                    f"Batch [{batch_index}/{len(train_loader)}] "
                    f"Recon: {reconstruction_loss.item():.6f} "
                    f"Privacy: {privacy_loss.item():.6f} "
                    f"AvgAttack: {average_attack_loss.item():.6f} "
                    f"WorstAttack: {worst_attack_loss.item():.6f} "
                    f"GAN: {gan_loss.item():.6f}"
                )

        train_reconstruction_loss = (
            total_reconstruction_loss / batches
        )

        train_privacy_loss = (
            total_privacy_loss / batches
        )

        train_average_attack_loss = (
            total_average_attack_loss / batches
        )

        train_worst_attack_loss = (
            total_worst_attack_loss / batches
        )

        train_gan_loss = (
            total_gan_loss / batches
        )

        train_discriminator_loss = (
            total_discriminator_loss / batches
        )

        metrics = evaluate_model(
            encoder,
            decoder,
            attackers,
            test_loader
        )

        print()
        print("=" * 60)
        print(
            f"Epoch [{epoch + 1}/{EPOCHS}]"
        )
        print(
            f"Train Reconstruction Loss: "
            f"{train_reconstruction_loss:.6f}"
        )
        print(
            f"Train Privacy Loss: "
            f"{train_privacy_loss:.6f}"
        )
        print(
            f"Train Average Attack Loss: "
            f"{train_average_attack_loss:.6f}"
        )
        print(
            f"Train Worst Attack Loss: "
            f"{train_worst_attack_loss:.6f}"
        )
        print(
            f"Train GAN Loss: "
            f"{train_gan_loss:.6f}"
        )
        print(
            f"Train Discriminator Loss: "
            f"{train_discriminator_loss:.6f}"
        )
        print(
            f"Validation Reconstruction Loss: "
            f"{metrics['reconstruction_loss']:.6f}"
        )
        print(
            f"Validation Reconstruction PSNR: "
            f"{metrics['reconstruction_psnr']:.2f} dB"
        )

        for share_index, psnr in enumerate(
            metrics["attack_psnrs"]
        ):
            print(
                f"Share {share_index + 1} Attack PSNR: "
                f"{psnr:.2f} dB"
            )

        print(
            f"Average Attack PSNR: "
            f"{metrics['average_attack_psnr']:.2f} dB"
        )

        print(
            f"Worst-case Attack PSNR: "
            f"{metrics['worst_attack_psnr']:.2f} dB"
        )
        print("=" * 60)
        print()

        if (
            best_metrics is None
            or metrics["worst_attack_psnr"]
            < best_metrics["worst_attack_psnr"]
        ):
            best_metrics = metrics
            best_epoch = epoch + 1

            save_checkpoint(
                encoder,
                decoder,
                attackers,
                best_epoch,
                metrics
            )

            print(
                "New best worst-share privacy checkpoint saved."
            )
            print()

    print()
    print("=" * 70)
    print("WORST-SHARE PRIVACY EXPERIMENT COMPLETE")
    print("=" * 70)
    print(
        f"Best epoch: {best_epoch}"
    )
    print(
        f"Best reconstruction PSNR: "
        f"{best_metrics['reconstruction_psnr']:.2f} dB"
    )
    print(
        f"Best average attack PSNR: "
        f"{best_metrics['average_attack_psnr']:.2f} dB"
    )
    print(
        f"Best worst-case attack PSNR: "
        f"{best_metrics['worst_attack_psnr']:.2f} dB"
    )

    for share_index, psnr in enumerate(
        best_metrics["attack_psnrs"]
    ):
        print(
            f"Share {share_index + 1}: "
            f"{psnr:.2f} dB"
        )

    print(
        f"Saved to: {CHECKPOINT_DIR}"
    )
    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()