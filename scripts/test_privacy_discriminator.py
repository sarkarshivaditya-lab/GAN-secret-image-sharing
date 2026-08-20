import torch

from models.privacy_discriminator import (
    PrivacyDiscriminator
)


def main():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print("Device:", device)

    discriminator = (
        PrivacyDiscriminator()
        .to(device)
    )

    image = torch.randn(
        2,
        3,
        256,
        256,
        device=device
    )

    share = torch.randn(
        2,
        3,
        32,
        32,
        device=device
    )

    output = discriminator(
        image,
        share
    )

    print(
        "Image:",
        image.shape
    )

    print(
        "Share:",
        share.shape
    )

    print(
        "Discriminator output:",
        output.shape
    )

    if output.shape == (2, 1):
        print()
        print(
            "Privacy discriminator "
            "tensor test: PASSED"
        )
    else:
        print()
        print(
            "Privacy discriminator "
            "tensor test: FAILED"
        )


if __name__ == "__main__":
    main()