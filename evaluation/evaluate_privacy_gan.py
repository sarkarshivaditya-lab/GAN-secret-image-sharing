import math
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from models.encoder import ShareEncoder
from models.decoder import ShareDecoder
from models.attacker import ShareAttacker


TRAIN_IMAGES = 5000
TEST_IMAGES = 1000

BATCH_SIZE = 16
ATTACKER_EPOCHS = 10
ATTACKER_LEARNING_RATE = 0.0002

ENCODER_PATH = (
    "checkpoints/privacy_gan/"
    "encoder_best.pth"
)

DECODER_PATH = (
    "checkpoints/privacy_gan/"
    "decoder_best.pth"
)

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


def load_models():
    encoder = ShareEncoder().to(
        DEVICE
    )

    decoder = ShareDecoder().to(
        DEVICE
    )

    encoder.load_state_dict(
        torch.load(
            ENCODER_PATH,
            map_location=DEVICE
        )
    )

    decoder.load_state_dict(
        torch.load(
            DECODER_PATH,
            map_location=DEVICE
        )
    )

    encoder.eval()
    decoder.eval()

    for parameter in encoder.parameters():
        parameter.requires_grad = False

    for parameter in decoder.parameters():
        parameter.requires_grad = False

    return encoder, decoder


def evaluate_legitimate_reconstruction(
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


def train_fresh_attacker(
    encoder,
    train_loader,
    share_index
):
    attacker = ShareAttacker().to(
        DEVICE
    )

    optimizer = torch.optim.Adam(
        attacker.parameters(),
        lr=ATTACKER_LEARNING_RATE
    )

    print()
    print(
        f"Training fresh attacker "
        f"for Share {share_index + 1}"
    )

    for epoch in range(
        ATTACKER_EPOCHS
    ):
        attacker.train()

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

    return attacker


def evaluate_attacker(
    encoder,
    attacker,
    test_loader,
    share_index
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


def main():
    print("Device:", DEVICE)
    print()
    print(
        "Privacy-GAN independent attacker evaluation"
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

    reconstruction_loss, reconstruction_psnr = (
        evaluate_legitimate_reconstruction(
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

    results = []

    print()
    print(
        "Training fresh independent attackers..."
    )

    for share_index in range(4):
        attacker = train_fresh_attacker(
            encoder,
            train_loader,
            share_index
        )

        attack_loss, attack_psnr = (
            evaluate_attacker(
                encoder,
                attacker,
                test_loader,
                share_index
            )
        )

        results.append({
            "share": share_index + 1,
            "loss": attack_loss,
            "psnr": attack_psnr
        })

        print(
            f"Share {share_index + 1} "
            f"Fresh Attack PSNR: "
            f"{attack_psnr:.2f} dB"
        )

        del attacker

        if DEVICE.type == "mps":
            torch.mps.empty_cache()

    print()
    print("=" * 60)
    print(
        "PRIVACY-GAN INDEPENDENT ATTACKER RESULTS"
    )
    print("=" * 60)

    for result in results:
        print(
            f"Share {result['share']}: "
            f"{result['psnr']:.2f} dB"
        )

    print()
    print(
        f"Legitimate Reconstruction: "
        f"{reconstruction_psnr:.2f} dB"
    )

    output_dir = (
        "outputs/privacy_gan"
    )

    import os

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    with open(
        f"{output_dir}/results.txt",
        "w"
    ) as file:
        file.write(
            "Privacy-GAN independent attacker evaluation\n"
        )

        file.write(
            f"Legitimate Reconstruction PSNR: "
            f"{reconstruction_psnr:.6f} dB\n"
        )

        file.write("\n")

        for result in results:
            file.write(
                f"Share {result['share']}: "
                f"{result['psnr']:.6f} dB\n"
            )

    print()
    print(
        f"Results saved to "
        f"{output_dir}/results.txt"
    )

    print()
    print(
        "Independent Privacy-GAN evaluation complete."
    )


if __name__ == "__main__":
    main()