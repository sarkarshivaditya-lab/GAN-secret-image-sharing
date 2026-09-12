import os
import math
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models.encoder import ShareEncoder
from models.attacker import ShareAttacker


QUICK_TEST = True


def calculate_psnr(original, reconstructed):
    mse = torch.mean((original - reconstructed) ** 2)

    if mse.item() == 0:
        return float("inf")

    return 10 * math.log10(1.0 / mse.item())


def evaluate_attacker(
    attacker,
    encoder,
    test_loader,
    share_index,
    device
):
    attacker.eval()
    encoder.eval()

    total_loss = 0.0
    total_psnr = 0.0
    batches = 0

    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)

            shares = encoder(images)

            share = shares[share_index - 1]

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

    average_loss = total_loss / batches
    average_psnr = total_psnr / batches

    return average_loss, average_psnr


def train_attacker(share_index, device):
    print()
    print("=" * 60)
    print(f"Training attacker for Share {share_index}")
    print("=" * 60)

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
        train_dataset = torch.utils.data.Subset(
            train_dataset,
            range(5000)
        )

        test_dataset = torch.utils.data.Subset(
            test_dataset,
            range(1000)
        )

        print()
        print("QUICK TEST MODE")
        print("Training images: 5,000")
        print("Test images: 1,000")
        print("Epochs: 3")
        print("Shares tested: 4")
        print()

    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0
    )

    encoder = ShareEncoder().to(device)

    encoder.load_state_dict(
        torch.load(
            "checkpoints/encoder_baseline.pth",
            map_location=device
        )
    )

    encoder.eval()

    for parameter in encoder.parameters():
        parameter.requires_grad = False

    attacker = ShareAttacker().to(device)

    optimizer = torch.optim.Adam(
        attacker.parameters(),
        lr=0.0002
    )

    if QUICK_TEST:
        epochs = 3
    else:
        epochs = 10

    best_psnr = float("-inf")

    os.makedirs(
        "checkpoints/attackers",
        exist_ok=True
    )

    best_path = (
        f"checkpoints/attackers/"
        f"attacker_share_{share_index}.pth"
    )

    for epoch in range(epochs):
        attacker.train()

        total_loss = 0.0
        batches = 0

        for batch_index, (images, _) in enumerate(train_loader):
            images = images.to(device)

            with torch.no_grad():
                shares = encoder(images)

                share = shares[share_index - 1]

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

            if batch_index % 100 == 0:
                print(
                    f"Share {share_index} | "
                    f"Epoch [{epoch + 1}/{epochs}] | "
                    f"Batch [{batch_index}/{len(train_loader)}] | "
                    f"Loss: {loss.item():.6f}"
                )

        train_loss = total_loss / batches

        validation_loss, validation_psnr = evaluate_attacker(
            attacker,
            encoder,
            test_loader,
            share_index,
            device
        )

        print()
        print(
            f"Share {share_index} | "
            f"Epoch [{epoch + 1}/{epochs}] | "
            f"Train Loss: {train_loss:.6f} | "
            f"Validation Loss: {validation_loss:.6f} | "
            f"Validation PSNR: {validation_psnr:.2f} dB"
        )

        if validation_psnr > best_psnr:
            best_psnr = validation_psnr

            torch.save(
                attacker.state_dict(),
                best_path
            )

            print(
                f"New best attacker saved "
                f"(PSNR: {best_psnr:.2f} dB)"
            )

    print()
    print("=" * 60)
    print(f"Best Share {share_index} attacker PSNR:")
    print(f"{best_psnr:.2f} dB")
    print("=" * 60)
    print(f"Saved: {best_path}")


def main():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print("Device:", device)

    if QUICK_TEST:
        share_indices = range(1, 5)
    else:
        share_indices = range(1, 5)

    for share_index in share_indices:
        train_attacker(
            share_index,
            device
        )

    print()
    print("=" * 60)
    print("Attacker experiment complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()