import math
import json
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.utils import save_image

from models.encoder import ShareEncoder
from models.decoder import ShareDecoder
from models.attacker import ShareAttacker


TRAIN_IMAGES = 5000
TEST_IMAGES = 1000

BATCH_SIZE = 8
EPOCHS = 5

LEARNING_RATE = 0.0002
ATTACKER_LEARNING_RATE = 0.0002
DISCRIMINATOR_LEARNING_RATE = 0.0001

PRIVACY_WEIGHT = 0.1
GAN_WEIGHT = 0.002

ATTACKER_STEPS = 2

WORST_SHARE_WEIGHT = 0.75
AVERAGE_SHARE_WEIGHT = 0.25

CHECKPOINT_DIR = "checkpoints/static_share_gan"
OUTPUT_DIR = "outputs/static_share_gan"

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

    return 10.0 * math.log10(
        1.0 / mse.item()
    )


def load_models():
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


class StaticDiscriminator(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.network = torch.nn.Sequential(
            torch.nn.Conv2d(
                3,
                32,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            torch.nn.LeakyReLU(
                0.2,
                inplace=True
            ),
            torch.nn.Conv2d(
                32,
                64,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            torch.nn.BatchNorm2d(64),
            torch.nn.LeakyReLU(
                0.2,
                inplace=True
            ),
            torch.nn.Conv2d(
                64,
                128,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            torch.nn.BatchNorm2d(128),
            torch.nn.LeakyReLU(
                0.2,
                inplace=True
            ),
            torch.nn.Conv2d(
                128,
                256,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            torch.nn.BatchNorm2d(256),
            torch.nn.LeakyReLU(
                0.2,
                inplace=True
            ),
            torch.nn.AdaptiveAvgPool2d(
                (1, 1)
            ),
            torch.nn.Flatten(),
            torch.nn.Linear(
                256,
                1
            )
        )

    def forward(self, x):
        return self.network(x)


def make_static_reference(
    shape,
    device
):
    noise = torch.randn(
        shape,
        device=device
    )

    noise = noise / (
        noise.std(
            dim=(1, 2, 3),
            keepdim=True
        ) + 1e-8
    )

    noise = noise * 0.25

    noise = torch.clamp(
        noise,
        -1.0,
        1.0
    )

    return noise


def calculate_noise_statistics(shares):
    statistics = []

    for share in shares:
        mean = share.mean().item()
        std = share.std().item()

        centered = share - share.mean(
            dim=(2, 3),
            keepdim=True
        )

        horizontal_corr = (
            centered[:, :, :, :-1]
            * centered[:, :, :, 1:]
        ).mean().item()

        vertical_corr = (
            centered[:, :, :-1, :]
            * centered[:, :, 1:, :]
        ).mean().item()

        fft = torch.fft.rfft2(
            share,
            dim=(-2, -1)
        )

        magnitude = torch.abs(
            fft
        )

        low_h = max(
            1,
            magnitude.shape[-2] // 8
        )

        low_w = max(
            1,
            magnitude.shape[-1] // 8
        )

        low_frequency_energy = magnitude[
            :, :, :low_h, :low_w
        ].mean().item()

        high_frequency_energy = magnitude[
            :, :, low_h:, low_w:
        ].mean().item()

        statistics.append({
            "mean": mean,
            "std": std,
            "horizontal_correlation": horizontal_corr,
            "vertical_correlation": vertical_corr,
            "low_frequency_energy": low_frequency_energy,
            "high_frequency_energy": high_frequency_energy
        })

    return statistics


def train_attackers(
    encoder,
    images,
    attackers,
    optimizers
):
    encoder.eval()

    with torch.no_grad():
        shares = encoder(
            images
        )

    for share_index in range(4):
        attacker = attackers[
            share_index
        ]

        optimizer = optimizers[
            share_index
        ]

        share = shares[
            share_index
        ].detach()

        for _ in range(
            ATTACKER_STEPS
        ):
            attacker.train()

            output = attacker(
                share
            )

            loss = F.mse_loss(
                output,
                images
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def calculate_privacy_loss(
    encoder,
    images,
    attackers
):
    shares = encoder(
        images
    )

    attack_losses = []

    for share_index in range(4):
        attacker = attackers[
            share_index
        ]

        attacker.eval()

        output = attacker(
            shares[share_index]
        )

        loss = F.mse_loss(
            output,
            images
        )

        attack_losses.append(
            loss
        )

    stacked = torch.stack(
        attack_losses
    )

    average_attack_loss = stacked.mean()
    worst_attack_loss = stacked.max()

    privacy_loss = (
        AVERAGE_SHARE_WEIGHT
        * average_attack_loss
        + WORST_SHARE_WEIGHT
        * worst_attack_loss
    )

    return (
        privacy_loss,
        average_attack_loss,
        worst_attack_loss,
        attack_losses,
        shares
    )


def train_static_discriminator(
    discriminator,
    optimizer,
    shares
):
    discriminator.train()

    fake = torch.cat(
        [
            share.detach()
            for share in shares
        ],
        dim=0
    )

    real = make_static_reference(
        fake.shape,
        DEVICE
    )

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

    loss = (
        real_loss + fake_loss
    ) * 0.5

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    with torch.no_grad():
        real_probability = torch.sigmoid(
            real_logits
        )

        fake_probability = torch.sigmoid(
            fake_logits
        )

        real_correct = (
            real_probability >= 0.5
        ).float().mean()

        fake_correct = (
            fake_probability < 0.5
        ).float().mean()

        accuracy = (
            real_correct + fake_correct
        ) * 0.5

    return (
        loss.item(),
        accuracy.item()
    )


def generator_static_loss(
    discriminator,
    shares
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

        losses.append(
            loss
        )

    return torch.stack(
        losses
    ).mean()


def save_visual_results(
    encoder,
    decoder,
    loader,
    epoch
):
    encoder.eval()
    decoder.eval()

    images, _ = next(
        iter(loader)
    )

    images = images.to(
        DEVICE
    )

    with torch.no_grad():
        shares = encoder(
            images
        )

        reconstructed = decoder(
            *shares
        )

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    save_image(
        images[:8],
        os.path.join(
            OUTPUT_DIR,
            f"epoch_{epoch}_original.png"
        ),
        nrow=4
    )

    for index in range(4):
        share = shares[
            index
        ][:8]

        share = share.clone()

        minimum = share.amin(
            dim=(1, 2, 3),
            keepdim=True
        )

        maximum = share.amax(
            dim=(1, 2, 3),
            keepdim=True
        )

        share = (
            share - minimum
        ) / (
            maximum - minimum + 1e-8
        )

        save_image(
            share,
            os.path.join(
                OUTPUT_DIR,
                f"epoch_{epoch}_share_{index + 1}.png"
            ),
            nrow=4
        )

    reconstructed = torch.clamp(
        reconstructed[:8],
        0.0,
        1.0
    )

    save_image(
        reconstructed,
        os.path.join(
            OUTPUT_DIR,
            f"epoch_{epoch}_reconstructed.png"
        ),
        nrow=4
    )


def evaluate(
    encoder,
    decoder,
    attackers,
    loader
):
    encoder.eval()
    decoder.eval()

    for attacker in attackers:
        attacker.eval()

    reconstruction_losses = []
    reconstruction_psnrs = []

    attack_losses = [
        [],
        [],
        [],
        []
    ]

    attack_psnrs = [
        [],
        [],
        [],
        []
    ]

    all_statistics = []

    with torch.no_grad():
        for images, _ in loader:
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

            reconstruction_losses.append(
                reconstruction_loss.item()
            )

            reconstruction_psnrs.append(
                calculate_psnr(
                    images,
                    reconstructed
                )
            )

            all_statistics.append(
                calculate_noise_statistics(
                    shares
                )
            )

            for share_index in range(4):
                attack_output = attackers[
                    share_index
                ](
                    shares[share_index]
                )

                loss = F.mse_loss(
                    attack_output,
                    images
                )

                attack_losses[
                    share_index
                ].append(
                    loss.item()
                )

                attack_psnrs[
                    share_index
                ].append(
                    calculate_psnr(
                        images,
                        attack_output
                    )
                )

    averaged_statistics = []

    for share_index in range(4):
        values = [
            batch[share_index]
            for batch in all_statistics
        ]

        averaged_statistics.append({
            key: sum(
                item[key]
                for item in values
            ) / len(values)
            for key in values[0]
        })

    averaged_attack_psnrs = [
        sum(values) / len(values)
        for values in attack_psnrs
    ]

    return {
        "reconstruction_loss":
            sum(reconstruction_losses)
            / len(reconstruction_losses),
        "reconstruction_psnr":
            sum(reconstruction_psnrs)
            / len(reconstruction_psnrs),
        "attack_losses": [
            sum(values) / len(values)
            for values in attack_losses
        ],
        "attack_psnrs":
            averaged_attack_psnrs,
        "average_attack_psnr":
            sum(averaged_attack_psnrs) / 4.0,
        "worst_attack_psnr":
            max(averaged_attack_psnrs),
        "noise_statistics":
            averaged_statistics
    }


def save_checkpoint(
    encoder,
    decoder,
    attackers,
    discriminator,
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

    torch.save(
        discriminator.state_dict(),
        os.path.join(
            CHECKPOINT_DIR,
            "discriminator_best.pth"
        )
    )

    for index, attacker in enumerate(
        attackers
    ):
        torch.save(
            attacker.state_dict(),
            os.path.join(
                CHECKPOINT_DIR,
                f"attacker_share_{index + 1}.pth"
            )
        )

    with open(
        os.path.join(
            CHECKPOINT_DIR,
            "best_metrics.json"
        ),
        "w"
    ) as file:
        json.dump(
            {
                "epoch": epoch,
                **metrics
            },
            file,
            indent=2
        )


def main():
    print("Device:", DEVICE)
    print()
    print("Static-share GAN training")
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

    encoder, decoder = load_models()

    attackers = []

    for _ in range(4):
        attackers.append(
            ShareAttacker().to(
                DEVICE
            )
        )

    attacker_optimizers = [
        torch.optim.Adam(
            attacker.parameters(),
            lr=ATTACKER_LEARNING_RATE
        )
        for attacker in attackers
    ]

    encoder_optimizer = torch.optim.Adam(
        encoder.parameters(),
        lr=LEARNING_RATE
    )

    decoder_optimizer = torch.optim.Adam(
        decoder.parameters(),
        lr=LEARNING_RATE
    )

    discriminator = StaticDiscriminator().to(
        DEVICE
    )

    discriminator_optimizer = torch.optim.Adam(
        discriminator.parameters(),
        lr=DISCRIMINATOR_LEARNING_RATE
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

        total_reconstruction = 0.0
        total_privacy = 0.0
        total_average_attack = 0.0
        total_worst_attack = 0.0
        total_gan = 0.0
        total_discriminator = 0.0
        total_discriminator_accuracy = 0.0

        batches = 0

        for batch_index, (images, _) in enumerate(
            train_loader
        ):
            images = images.to(
                DEVICE
            )

            train_attackers(
                encoder,
                images,
                attackers,
                attacker_optimizers
            )

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

            (
                discriminator_loss,
                discriminator_accuracy
            ) = train_static_discriminator(
                discriminator,
                discriminator_optimizer,
                shares
            )

            gan_loss = generator_static_loss(
                discriminator,
                shares
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

            total_reconstruction += (
                reconstruction_loss.item()
            )

            total_privacy += (
                privacy_loss.item()
            )

            total_average_attack += (
                average_attack_loss.item()
            )

            total_worst_attack += (
                worst_attack_loss.item()
            )

            total_gan += (
                gan_loss.item()
            )

            total_discriminator += (
                discriminator_loss
            )

            total_discriminator_accuracy += (
                discriminator_accuracy
            )

            batches += 1

            if batch_index % 100 == 0:
                print(
                    f"Epoch [{epoch + 1}/{EPOCHS}] "
                    f"Batch [{batch_index}/{len(train_loader)}] "
                    f"Recon: {reconstruction_loss.item():.6f} "
                    f"Privacy: {privacy_loss.item():.6f} "
                    f"WorstAttack: {worst_attack_loss.item():.6f} "
                    f"GAN: {gan_loss.item():.6f} "
                    f"D: {discriminator_accuracy:.2%}"
                )

        metrics = evaluate(
            encoder,
            decoder,
            attackers,
            test_loader
        )

        average_discriminator_loss = (
            total_discriminator / batches
        )

        average_discriminator_accuracy = (
            total_discriminator_accuracy
            / batches
        )

        print()
        print("=" * 70)
        print(
            f"Epoch [{epoch + 1}/{EPOCHS}]"
        )
        print(
            f"Train Reconstruction Loss: "
            f"{total_reconstruction / batches:.6f}"
        )
        print(
            f"Train Privacy Loss: "
            f"{total_privacy / batches:.6f}"
        )
        print(
            f"Train Average Attack Loss: "
            f"{total_average_attack / batches:.6f}"
        )
        print(
            f"Train Worst Attack Loss: "
            f"{total_worst_attack / batches:.6f}"
        )
        print(
            f"Train GAN Loss: "
            f"{total_gan / batches:.6f}"
        )
        print(
            f"Train Discriminator Loss: "
            f"{average_discriminator_loss:.6f}"
        )
        print(
            f"Train Discriminator Accuracy: "
            f"{average_discriminator_accuracy:.2%}"
        )
        print(
            f"Validation Reconstruction Loss: "
            f"{metrics['reconstruction_loss']:.6f}"
        )
        print(
            f"Validation Reconstruction PSNR: "
            f"{metrics['reconstruction_psnr']:.2f} dB"
        )

        for index, psnr in enumerate(
            metrics["attack_psnrs"]
        ):
            print(
                f"Share {index + 1} Attack PSNR: "
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

        print()
        print("Share statistics:")

        for index, stats in enumerate(
            metrics["noise_statistics"]
        ):
            print(
                f"Share {index + 1} | "
                f"Mean: {stats['mean']:.5f} | "
                f"Std: {stats['std']:.5f} | "
                f"H-Corr: {stats['horizontal_correlation']:.6f} | "
                f"V-Corr: {stats['vertical_correlation']:.6f}"
            )

        print("=" * 70)
        print()

        save_visual_results(
            encoder,
            decoder,
            test_loader,
            epoch + 1
        )

        if (
            best_metrics is None
            or (
                metrics["worst_attack_psnr"]
                < best_metrics["worst_attack_psnr"]
                and metrics["reconstruction_psnr"]
                > 20.0
            )
        ):
            best_metrics = metrics
            best_epoch = epoch + 1

            save_checkpoint(
                encoder,
                decoder,
                attackers,
                discriminator,
                best_epoch,
                metrics
            )

            print(
                "New best static-share checkpoint saved."
            )
            print()

    print()
    print("=" * 70)
    print("STATIC-SHARE GAN TRAINING COMPLETE")
    print("=" * 70)

    if best_metrics is not None:
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
    else:
        print(
            "No checkpoint satisfied the reconstruction constraint."
        )

    print()
    print(
        f"Checkpoints: {CHECKPOINT_DIR}"
    )
    print(
        f"Visual results: {OUTPUT_DIR}"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()