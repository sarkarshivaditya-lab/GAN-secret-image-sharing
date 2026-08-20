import os
import json
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models.encoder import ShareEncoder
from models.decoder import ShareDecoder
from models.attacker import ShareAttacker


DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

TRAIN_IMAGES = 5000
TEST_IMAGES = 1000
EPOCHS = 5
BATCH_SIZE = 8

PRIVACY_WEIGHT = 0.03
WORST_PRIVACY_WEIGHT = 0.10
GAN_WEIGHT = 0.002
STATIC_WEIGHT = 0.03

ATTACKER_STEPS = 2

LR = 1e-4
ATTACKER_LR = 1e-4
DISCRIMINATOR_LR = 2e-5

NUM_SHARES = 4
IMAGE_SIZE = 32


def psnr(mse):
    if mse <= 0:
        return 99.0

    return 10.0 * math.log10(1.0 / mse)


class StaticDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 1, 3, 1, 1),

            nn.AdaptiveAvgPool2d(1)
        )

    def forward(self, x):
        return self.net(x).view(-1, 1)


def static_statistics_loss(shares):
    """
    Encourages every share toward a static-like distribution.

    Components:
    1. Mean matching
    2. Standard deviation matching
    3. Horizontal correlation suppression
    4. Vertical correlation suppression
    """

    loss = torch.tensor(
        0.0,
        device=DEVICE
    )

    for share in shares:

        mean = share.mean()
        std = share.std()

        mean_loss = mean.pow(2)
        std_loss = (std - 1.0).pow(2)

        horizontal = (
            share[:, :, :, 1:]
            * share[:, :, :, :-1]
        )

        vertical = (
            share[:, :, 1:, :]
            * share[:, :, :-1, :]
        )

        horizontal_corr = horizontal.mean().abs()
        vertical_corr = vertical.mean().abs()

        loss = (
            loss
            + mean_loss
            + std_loss
            + horizontal_corr
            + vertical_corr
        )

    return loss / len(shares)


def normalize_share_for_discriminator(share):
    mean = share.mean(
        dim=(1, 2, 3),
        keepdim=True
    )

    std = share.std(
        dim=(1, 2, 3),
        keepdim=True
    ) + 1e-6

    return (share - mean) / std


def split_shares(encoded):
    """
    Supports encoder outputs in either form:

    [B, 12, H, W]

    or

    [B, 4, 3, H, W]

    or a tuple/list containing four shares.
    """

    if isinstance(encoded, (tuple, list)):
        if len(encoded) != NUM_SHARES:
            raise ValueError(
                f"Expected {NUM_SHARES} shares, "
                f"got {len(encoded)}"
            )

        return list(encoded)

    if encoded.dim() == 5:

        if encoded.size(1) != NUM_SHARES:
            raise ValueError(
                f"Expected {NUM_SHARES} shares, "
                f"got {encoded.size(1)}"
            )

        return [
            encoded[:, i]
            for i in range(encoded.size(1))
        ]

    if encoded.dim() == 4:

        channels = encoded.size(1)

        if channels != NUM_SHARES * 3:
            raise ValueError(
                f"Expected {NUM_SHARES * 3} channels, "
                f"got {channels}"
            )

        return list(
            torch.chunk(
                encoded,
                NUM_SHARES,
                dim=1
            )
        )

    raise ValueError(
        f"Unsupported encoder output shape: "
        f"{encoded.shape}"
    )


def reconstruct(decoder, shares):
    """
    The ShareDecoder expects the four shares as
    separate arguments.

    DO NOT concatenate the shares before calling
    the decoder.
    """

    if len(shares) != NUM_SHARES:
        raise ValueError(
            f"Expected {NUM_SHARES} shares, "
            f"got {len(shares)}"
        )

    return decoder(*shares)


def make_loader(train=True):

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    dataset = datasets.CIFAR10(
        root="./data",
        train=train,
        download=True,
        transform=transform
    )

    if train:

        dataset = torch.utils.data.Subset(
            dataset,
            range(
                min(
                    TRAIN_IMAGES,
                    len(dataset)
                )
            )
        )

    else:

        dataset = torch.utils.data.Subset(
            dataset,
            range(
                min(
                    TEST_IMAGES,
                    len(dataset)
                )
            )
        )

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=train,
        num_workers=0
    )


def train_attacker_step(
    attacker,
    attacker_optimizer,
    share,
    original
):

    attacker.train()

    prediction = attacker(share)

    loss = nn.functional.mse_loss(
        prediction,
        original
    )

    attacker_optimizer.zero_grad()

    loss.backward()

    attacker_optimizer.step()

    return loss.detach()


def train_all_attackers(
    attackers,
    attacker_optimizers,
    shares,
    original
):

    losses = []

    for i in range(NUM_SHARES):

        for _ in range(ATTACKER_STEPS):

            loss = train_attacker_step(
                attackers[i],
                attacker_optimizers[i],
                shares[i].detach(),
                original
            )

        losses.append(loss)

    return losses


def attacker_privacy_losses(
    attackers,
    shares,
    original
):

    """
    Returns the reconstruction loss of every
    independent attacker.

    Higher attacker loss means:
        worse reconstruction by attacker
        =
        better privacy.
    """

    losses = []

    for i in range(NUM_SHARES):

        prediction = attackers[i](
            shares[i]
        )

        loss = nn.functional.mse_loss(
            prediction,
            original
        )

        losses.append(loss)

    return losses


def evaluate_attackers(
    attackers,
    encoder,
    decoder,
    loader
):

    encoder.eval()
    decoder.eval()

    for attacker in attackers:
        attacker.eval()

    total_recon = 0.0

    count = 0

    share_mse = [
        0.0
        for _ in range(NUM_SHARES)
    ]

    with torch.no_grad():

        for images, _ in loader:

            images = images.to(DEVICE)

            encoded = encoder(images)

            shares = split_shares(
                encoded
            )

            reconstructed = reconstruct(
                decoder,
                shares
            )

            recon_loss = nn.functional.mse_loss(
                reconstructed,
                images
            )

            batch_size = images.size(0)

            total_recon += (
                recon_loss.item()
                * batch_size
            )

            for i in range(NUM_SHARES):

                prediction = attackers[i](
                    shares[i]
                )

                mse = nn.functional.mse_loss(
                    prediction,
                    images
                )

                share_mse[i] += (
                    mse.item()
                    * batch_size
                )

            count += batch_size

    recon_mse = total_recon / count

    attack_mse = [
        value / count
        for value in share_mse
    ]

    attack_psnr = [
        psnr(value)
        for value in attack_mse
    ]

    return (
        recon_mse,
        psnr(recon_mse),
        attack_psnr
    )


def save_checkpoint(
    encoder,
    decoder,
    discriminator,
    epoch,
    recon_psnr,
    attack_psnr
):

    directory = (
        "checkpoints/"
        "balanced_static_gan"
    )

    os.makedirs(
        directory,
        exist_ok=True
    )

    torch.save(
        encoder.state_dict(),
        os.path.join(
            directory,
            "encoder_best.pth"
        )
    )

    torch.save(
        decoder.state_dict(),
        os.path.join(
            directory,
            "decoder_best.pth"
        )
    )

    torch.save(
        discriminator.state_dict(),
        os.path.join(
            directory,
            "discriminator_best.pth"
        )
    )

    info = {
        "epoch": epoch,
        "reconstruction_psnr": recon_psnr,
        "attack_psnr": attack_psnr,
        "average_attack_psnr": (
            sum(attack_psnr)
            / len(attack_psnr)
        ),
        "worst_case_attack_psnr": max(
            attack_psnr
        )
    }

    with open(
        os.path.join(
            directory,
            "best_info.json"
        ),
        "w"
    ) as f:

        json.dump(
            info,
            f,
            indent=4
        )


def main():

    print(
        f"Device: {DEVICE}"
    )

    print()

    print(
        "Balanced Static-Share GAN training"
    )

    print(
        f"Training images: {TRAIN_IMAGES}"
    )

    print(
        f"Test images: {TEST_IMAGES}"
    )

    print(
        f"Epochs: {EPOCHS}"
    )

    print(
        f"Batch size: {BATCH_SIZE}"
    )

    print(
        f"Privacy weight: {PRIVACY_WEIGHT}"
    )

    print(
        f"Worst privacy weight: "
        f"{WORST_PRIVACY_WEIGHT}"
    )

    print(
        f"GAN weight: {GAN_WEIGHT}"
    )

    print(
        f"Static weight: {STATIC_WEIGHT}"
    )

    print(
        f"Attacker steps: {ATTACKER_STEPS}"
    )

    print()

    train_loader = make_loader(
        True
    )

    test_loader = make_loader(
        False
    )

    # =========================================================
    # MODEL INITIALIZATION
    # =========================================================

    encoder = ShareEncoder().to(
        DEVICE
    )

    decoder = ShareDecoder().to(
        DEVICE
    )

    discriminator = StaticDiscriminator().to(
        DEVICE
    )

    attackers = [
        ShareAttacker().to(DEVICE)
        for _ in range(NUM_SHARES)
    ]

    # =========================================================
    # OPTIMIZERS
    # =========================================================

    encoder_optimizer = optim.Adam(
        encoder.parameters(),
        lr=LR
    )

    decoder_optimizer = optim.Adam(
        decoder.parameters(),
        lr=LR
    )

    discriminator_optimizer = optim.Adam(
        discriminator.parameters(),
        lr=DISCRIMINATOR_LR
    )

    attacker_optimizers = [
        optim.Adam(
            attacker.parameters(),
            lr=ATTACKER_LR
        )
        for attacker in attackers
    ]

    reconstruction_criterion = nn.MSELoss()

    gan_criterion = (
        nn.BCEWithLogitsLoss()
    )

    best_score = float("inf")

    best_epoch = -1

    # =========================================================
    # TRAINING
    # =========================================================

    for epoch in range(EPOCHS):

        encoder.train()

        decoder.train()

        discriminator.train()

        for attacker in attackers:
            attacker.train()

        total_recon = 0.0
        total_privacy = 0.0
        total_worst = 0.0
        total_static = 0.0
        total_gan = 0.0
        total_d = 0.0

        total_batches = 0

        for batch_idx, (
            images,
            _
        ) in enumerate(train_loader):

            images = images.to(
                DEVICE
            )

            # =================================================
            # GENERATE SHARES
            # =================================================

            encoded = encoder(
                images
            )

            shares = split_shares(
                encoded
            )

            # =================================================
            # TRAIN INDEPENDENT ATTACKERS
            # =================================================

            train_all_attackers(
                attackers,
                attacker_optimizers,
                shares,
                images
            )

            # =================================================
            # TRAIN STATIC DISCRIMINATOR
            # =================================================

            discriminator_optimizer.zero_grad()

            real = torch.randn_like(
                shares[0]
            )

            real = normalize_share_for_discriminator(
                real
            )

            fake_parts = []

            for share in shares:

                fake_parts.append(
                    normalize_share_for_discriminator(
                        share.detach()
                    )
                )

            fake = torch.cat(
                fake_parts,
                dim=0
            )

            real_output = discriminator(
                real
            )

            fake_output = discriminator(
                fake
            )

            real_labels = torch.ones_like(
                real_output
            )

            fake_labels = torch.zeros_like(
                fake_output
            )

            real_loss = gan_criterion(
                real_output,
                real_labels
            )

            fake_loss = gan_criterion(
                fake_output,
                fake_labels
            )

            discriminator_loss = (
                real_loss
                + fake_loss
            ) / 2.0

            discriminator_loss.backward()

            discriminator_optimizer.step()

            # =================================================
            # TRAIN ENCODER / DECODER
            # =================================================

            encoder_optimizer.zero_grad()

            decoder_optimizer.zero_grad()

            encoded = encoder(
                images
            )

            shares = split_shares(
                encoded
            )

            # =================================================
            # LEGITIMATE RECONSTRUCTION
            # =================================================

            reconstructed = reconstruct(
                decoder,
                shares
            )

            reconstruction_loss = (
                reconstruction_criterion(
                    reconstructed,
                    images
                )
            )

            # =================================================
            # ATTACKER PRIVACY
            # =================================================

            attack_losses = (
                attacker_privacy_losses(
                    attackers,
                    shares,
                    images
                )
            )

            average_attack_loss = (
                torch.stack(
                    attack_losses
                ).mean()
            )

            worst_attack_loss = (
                torch.stack(
                    attack_losses
                ).max()
            )

            # =================================================
            # STATIC STATISTICS
            # =================================================

            static_loss = (
                static_statistics_loss(
                    shares
                )
            )

            # =================================================
            # GENERATOR GAN LOSS
            # =================================================

            fake_for_gan = torch.cat(
                [
                    normalize_share_for_discriminator(
                        share
                    )
                    for share in shares
                ],
                dim=0
            )

            generator_output = discriminator(
                fake_for_gan
            )

            generator_labels = (
                torch.ones_like(
                    generator_output
                )
            )

            generator_gan_loss = (
                gan_criterion(
                    generator_output,
                    generator_labels
                )
            )

            # =================================================
            # TOTAL LOSS
            # =================================================

            total_loss = (
                reconstruction_loss
                - PRIVACY_WEIGHT
                * average_attack_loss
                - WORST_PRIVACY_WEIGHT
                * worst_attack_loss
                + GAN_WEIGHT
                * generator_gan_loss
                + STATIC_WEIGHT
                * static_loss
            )

            total_loss.backward()

            torch.nn.utils.clip_grad_norm_(
                encoder.parameters(),
                5.0
            )

            torch.nn.utils.clip_grad_norm_(
                decoder.parameters(),
                5.0
            )

            encoder_optimizer.step()

            decoder_optimizer.step()

            # =================================================
            # STATISTICS
            # =================================================

            total_recon += (
                reconstruction_loss.item()
            )

            total_privacy += (
                average_attack_loss.item()
            )

            total_worst += (
                worst_attack_loss.item()
            )

            total_static += (
                static_loss.item()
            )

            total_gan += (
                generator_gan_loss.item()
            )

            total_d += (
                discriminator_loss.item()
            )

            total_batches += 1

            if batch_idx % 100 == 0:

                print(
                    f"Epoch [{epoch + 1}/{EPOCHS}] "
                    f"Batch [{batch_idx}/"
                    f"{len(train_loader)}] "
                    f"Recon: "
                    f"{reconstruction_loss.item():.6f} "
                    f"Privacy: "
                    f"{average_attack_loss.item():.6f} "
                    f"WorstAttack: "
                    f"{worst_attack_loss.item():.6f} "
                    f"Static: "
                    f"{static_loss.item():.6f} "
                    f"GAN: "
                    f"{generator_gan_loss.item():.6f} "
                    f"D: "
                    f"{discriminator_loss.item():.6f}"
                )

        # =====================================================
        # EPOCH SUMMARY
        # =====================================================

        avg_recon = (
            total_recon
            / total_batches
        )

        avg_privacy = (
            total_privacy
            / total_batches
        )

        avg_worst = (
            total_worst
            / total_batches
        )

        avg_static = (
            total_static
            / total_batches
        )

        avg_gan = (
            total_gan
            / total_batches
        )

        avg_d = (
            total_d
            / total_batches
        )

        (
            recon_mse,
            recon_psnr,
            attack_psnr
        ) = evaluate_attackers(
            attackers,
            encoder,
            decoder,
            test_loader
        )

        average_attack_psnr = (
            sum(attack_psnr)
            / len(attack_psnr)
        )

        worst_attack_psnr = max(
            attack_psnr
        )

        print()

        print(
            "=" * 70
        )

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}]"
        )

        print(
            f"Train Reconstruction Loss: "
            f"{avg_recon:.6f}"
        )

        print(
            f"Train Average Privacy Loss: "
            f"{avg_privacy:.6f}"
        )

        print(
            f"Train Worst Attack Loss: "
            f"{avg_worst:.6f}"
        )

        print(
            f"Train Static Loss: "
            f"{avg_static:.6f}"
        )

        print(
            f"Train GAN Loss: "
            f"{avg_gan:.6f}"
        )

        print(
            f"Train Discriminator Loss: "
            f"{avg_d:.6f}"
        )

        print(
            f"Validation Reconstruction Loss: "
            f"{recon_mse:.6f}"
        )

        print(
            f"Validation Reconstruction PSNR: "
            f"{recon_psnr:.2f} dB"
        )

        for i, value in enumerate(
            attack_psnr
        ):

            print(
                f"Share {i + 1} Attack PSNR: "
                f"{value:.2f} dB"
            )

        print(
            f"Average Attack PSNR: "
            f"{average_attack_psnr:.2f} dB"
        )

        print(
            f"Worst-case Attack PSNR: "
            f"{worst_attack_psnr:.2f} dB"
        )

        print(
            "=" * 70
        )

        # =====================================================
        # BALANCED CHECKPOINT SCORE
        # =====================================================

        privacy_penalty = max(
            0.0,
            worst_attack_psnr - 18.0
        )

        reconstruction_penalty = max(
            0.0,
            30.0 - recon_psnr
        )

        score = (
            privacy_penalty
            + 0.5
            * reconstruction_penalty
        )

        if score < best_score:

            best_score = score

            best_epoch = epoch + 1

            save_checkpoint(
                encoder,
                decoder,
                discriminator,
                best_epoch,
                recon_psnr,
                attack_psnr
            )

            print()

            print(
                "New best balanced "
                "static-share checkpoint saved."
            )

            print(
                f"Score: {score:.4f}"
            )

            print(
                f"Reconstruction: "
                f"{recon_psnr:.2f} dB"
            )

            print(
                f"Worst attack: "
                f"{worst_attack_psnr:.2f} dB"
            )

            print()

    # =========================================================
    # COMPLETE
    # =========================================================

    print()

    print(
        "=" * 70
    )

    print(
        "BALANCED STATIC-SHARE GAN "
        "TRAINING COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        f"Best epoch: {best_epoch}"
    )

    info_path = (
        "checkpoints/"
        "balanced_static_gan/"
        "best_info.json"
    )

    if os.path.exists(
        info_path
    ):

        with open(
            info_path
        ) as f:

            info = json.load(f)

        print(
            f"Best reconstruction PSNR: "
            f"{info['reconstruction_psnr']:.2f} dB"
        )

        print(
            f"Best average attack PSNR: "
            f"{info['average_attack_psnr']:.2f} dB"
        )

        print(
            f"Best worst-case attack PSNR: "
            f"{info['worst_case_attack_psnr']:.2f} dB"
        )

    print()

    print(
        "Checkpoints: "
        "checkpoints/balanced_static_gan"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()