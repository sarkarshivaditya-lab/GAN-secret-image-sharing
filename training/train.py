import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models.encoder import ShareEncoder
from models.decoder import ShareDecoder
from training.losses import reconstruction_loss

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

    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=0
    )

    encoder = ShareEncoder().to(device)
    decoder = ShareDecoder().to(device)

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) +
        list(decoder.parameters()),
        lr=0.0002
    )

    epochs = 10

    for epoch in range(epochs):
        encoder.train()
        decoder.train()

        total_loss = 0.0

        for batch_index, (images, _) in enumerate(train_loader):
            images = images.to(device)

            optimizer.zero_grad()

            shares = encoder(images)

            reconstructed = decoder(*shares)

            loss = reconstruction_loss(
                images,
                reconstructed
            )

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

            if batch_index % 100 == 0:
                print(
                    f"Epoch [{epoch + 1}/{epochs}] "
                    f"Batch [{batch_index}/{len(train_loader)}] "
                    f"Loss: {loss.item():.6f}"
                )

        average_loss = total_loss / len(train_loader)

        print(
            f"\nEpoch [{epoch + 1}/{epochs}] "
            f"Average Loss: {average_loss:.6f}\n"
        )

    torch.save(
        encoder.state_dict(),
        "checkpoints/encoder_baseline.pth"
    )

    torch.save(
        decoder.state_dict(),
        "checkpoints/decoder_baseline.pth"
    )

    print("Training complete.")
    print("Models saved to checkpoints/")

if __name__ == "__main__":
    main()