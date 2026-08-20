import os
import math
import torch
from torchvision import datasets, transforms
from torchvision.utils import save_image

from models.encoder import ShareEncoder
from models.decoder import ShareDecoder


def calculate_psnr(original, reconstructed):
    mse = torch.mean((original - reconstructed) ** 2)

    if mse.item() == 0:
        return float("inf")

    return 10 * math.log10(1.0 / mse.item())


def main():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print("Device:", device)

    output_dir = "outputs/baseline"
    os.makedirs(output_dir, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    test_dataset = datasets.CIFAR10(
        root="data",
        train=False,
        download=True,
        transform=transform
    )

    encoder = ShareEncoder().to(device)
    decoder = ShareDecoder().to(device)

    encoder.load_state_dict(
        torch.load(
            "checkpoints/encoder_baseline.pth",
            map_location=device
        )
    )

    decoder.load_state_dict(
        torch.load(
            "checkpoints/decoder_baseline.pth",
            map_location=device
        )
    )

    encoder.eval()
    decoder.eval()

    image, label = test_dataset[0]

    image = image.unsqueeze(0).to(device)

    with torch.no_grad():
        shares = encoder(image)

        reconstructed = decoder(*shares)

    save_image(
        image.cpu(),
        f"{output_dir}/original.png"
    )

    for i, share in enumerate(shares, start=1):
        save_image(
            share.cpu(),
            f"{output_dir}/share_{i}.png"
        )

    save_image(
        reconstructed.cpu(),
        f"{output_dir}/reconstructed.png"
    )

    psnr = calculate_psnr(
        image,
        reconstructed
    )

    print()
    print("Evaluation complete.")
    print("Original label:", label)
    print("PSNR:", f"{psnr:.2f} dB")
    print()
    print("Generated files:")

    print(f"{output_dir}/original.png")

    for i in range(1, 5):
        print(f"{output_dir}/share_{i}.png")

    print(f"{output_dir}/reconstructed.png")


if __name__ == "__main__":
    main()