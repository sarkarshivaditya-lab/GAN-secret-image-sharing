import os
import math
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.utils import save_image

from models.encoder import ShareEncoder
from models.decoder import ShareDecoder
from models.attacker import ShareAttacker


TEST_IMAGES = 1000
BATCH_SIZE = 16

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


def load_models(device):
    encoder = ShareEncoder().to(device)
    decoder = ShareDecoder().to(device)

    encoder.load_state_dict(
        torch.load(
            "checkpoints/adversarial/encoder_adversarial.pth",
            map_location=device
        )
    )

    decoder.load_state_dict(
        torch.load(
            "checkpoints/adversarial/decoder_adversarial.pth",
            map_location=device
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
    test_loader,
    device
):
    total_loss = 0.0
    total_psnr = 0.0
    batches = 0

    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)

            shares = encoder(images)

            reconstructed = decoder(*shares)

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


def train_independent_attacker(
    encoder,
    train_loader,
    share_index,
    device
):
    attacker = ShareAttacker().to(device)

    optimizer = torch.optim.Adam(
        attacker.parameters(),
        lr=0.0002
    )

    epochs = 3

    for epoch in range(epochs):
        attacker.train()

        total_loss = 0.0
        batches = 0

        for images, _ in train_loader:
            images = images.to(device)

            with torch.no_grad():
                shares = encoder(images)
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
            batches += 1

        average_loss = total_loss / batches

        print(
            f"Share {share_index + 1} | "
            f"Epoch [{epoch + 1}/{epochs}] | "
            f"Attack Train Loss: {average_loss:.6f}"
        )

    return attacker


def evaluate_independent_attacker(
    encoder,
    attacker,
    test_loader,
    share_index,
    device
):
    encoder.eval()
    attacker.eval()

    total_loss = 0.0
    total_psnr = 0.0
    batches = 0

    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)

            shares = encoder(images)

            share = shares[share_index]

            reconstructed = attacker(share)

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


def save_visual_results(
    encoder,
    decoder,
    test_dataset,
    device
):
    output_dir = "outputs/adversarial"

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    image, label = test_dataset[0]

    image = image.unsqueeze(0).to(device)

    with torch.no_grad():
        shares = encoder(image)

        reconstructed = decoder(*shares)

    save_image(
        image.cpu(),
        f"{output_dir}/original.png"
    )

    for index, share in enumerate(shares, start=1):
        save_image(
            share.cpu(),
            f"{output_dir}/share_{index}.png"
        )

    save_image(
        reconstructed.cpu(),
        f"{output_dir}/reconstructed.png"
    )

    print()
    print("Visual results saved:")
    print(f"{output_dir}/original.png")

    for index in range(1, 5):
        print(
            f"{output_dir}/share_{index}.png"
        )

    print(
        f"{output_dir}/reconstructed.png"
    )

    print("Test image label:", label)


def main():
    print("Device:", DEVICE)
    print()

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
        range(5000)
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

    encoder, decoder = load_models(
        DEVICE
    )

    reconstruction_loss, reconstruction_psnr = (
        evaluate_legitimate_reconstruction(
            encoder,
            decoder,
            test_loader,
            DEVICE
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

    print()
    print("Training fresh independent attackers...")
    print()

    attacker_results = []

    for share_index in range(4):
        print()
        print(
            f"Training independent attacker "
            f"for Share {share_index + 1}"
        )

        attacker = train_independent_attacker(
            encoder,
            train_loader,
            share_index,
            DEVICE
        )

        test_loss, test_psnr = (
            evaluate_independent_attacker(
                encoder,
                attacker,
                test_loader,
                share_index,
                DEVICE
            )
        )

        attacker_results.append(
            (test_loss, test_psnr)
        )

        print(
            f"Share {share_index + 1} "
            f"Independent Attack PSNR: "
            f"{test_psnr:.2f} dB"
        )

        del attacker

        if DEVICE.type == "mps":
            torch.mps.empty_cache()

    print()
    print("=" * 60)
    print("INDEPENDENT ATTACKER RESULTS")
    print("=" * 60)

    for index, (_, psnr) in enumerate(
        attacker_results,
        start=1
    ):
        print(
            f"Share {index}: "
            f"{psnr:.2f} dB"
        )

    print()
    print(
        f"Legitimate reconstruction: "
        f"{reconstruction_psnr:.2f} dB"
    )

    save_visual_results(
        encoder,
        decoder,
        test_dataset,
        DEVICE
    )

    print()
    print("Independent evaluation complete.")


if __name__ == "__main__":
    main()