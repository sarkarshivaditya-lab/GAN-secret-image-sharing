import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from models.encoder import ShareEncoder
from models.privacy_discriminator import PrivacyDiscriminator


TRAIN_IMAGES = 500
BATCH_SIZE = 8
EPOCHS = 2
LEARNING_RATE = 0.0002

DEVICE = (
    torch.device("mps")
    if torch.backends.mps.is_available()
    else torch.device("cpu")
)


def main():
    print("Device:", DEVICE)
    print()
    print("Privacy discriminator learning test")
    print("Training images:", TRAIN_IMAGES)
    print("Batch size:", BATCH_SIZE)
    print("Epochs:", EPOCHS)

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    dataset = datasets.CIFAR10(
        root="data",
        train=True,
        download=True,
        transform=transform
    )

    dataset = Subset(
        dataset,
        range(TRAIN_IMAGES)
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    encoder = ShareEncoder().to(DEVICE)

    encoder.load_state_dict(
        torch.load(
            "checkpoints/privacy_sweep/"
            "lambda_0.10/encoder.pth",
            map_location=DEVICE
        )
    )

    encoder.eval()

    for parameter in encoder.parameters():
        parameter.requires_grad = False

    discriminator = (
        PrivacyDiscriminator()
        .to(DEVICE)
    )

    optimizer = torch.optim.Adam(
        discriminator.parameters(),
        lr=LEARNING_RATE
    )

    for epoch in range(EPOCHS):
        discriminator.train()

        total_loss = 0.0
        correct = 0
        total = 0

        for images, _ in loader:
            images = images.to(DEVICE)

            with torch.no_grad():
                shares = encoder(images)

            share = shares[0]

            batch_size = images.shape[0]

            permutation = torch.randperm(
                batch_size,
                device=DEVICE
            )

            mismatched_images = (
                images[permutation]
            )

            matching_logits = discriminator(
                images,
                share
            )

            mismatching_logits = discriminator(
                mismatched_images,
                share
            )

            matching_targets = torch.ones_like(
                matching_logits
            )

            mismatching_targets = torch.zeros_like(
                mismatching_logits
            )

            matching_loss = (
                F.binary_cross_entropy_with_logits(
                    matching_logits,
                    matching_targets
                )
            )

            mismatching_loss = (
                F.binary_cross_entropy_with_logits(
                    mismatching_logits,
                    mismatching_targets
                )
            )

            loss = (
                matching_loss +
                mismatching_loss
            ) * 0.5

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

            matching_predictions = (
                torch.sigmoid(
                    matching_logits
                ) >= 0.5
            )

            mismatching_predictions = (
                torch.sigmoid(
                    mismatching_logits
                ) < 0.5
            )

            correct += (
                matching_predictions.sum().item()
                +
                mismatching_predictions.sum().item()
            )

            total += (
                2 * batch_size
            )

        average_loss = (
            total_loss / len(loader)
        )

        accuracy = (
            100.0 * correct / total
        )

        print()
        print(
            f"Epoch [{epoch + 1}/{EPOCHS}]"
        )

        print(
            f"Loss: {average_loss:.6f}"
        )

        print(
            f"Accuracy: {accuracy:.2f}%"
        )

    print()
    print(
        "Privacy discriminator learning test complete."
    )


if __name__ == "__main__":
    main()