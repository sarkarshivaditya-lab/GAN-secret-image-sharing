import torch

from models.discriminator import ShareDiscriminator


def main():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print("Device:", device)

    discriminator = ShareDiscriminator().to(device)

    test_input = torch.randn(
        2,
        3,
        32,
        32,
        device=device
    )

    output = discriminator(test_input)

    print(
        "Input:",
        test_input.shape
    )

    print(
        "Discriminator output:",
        output.shape
    )

    if output.shape == (2, 1):
        print()
        print(
            "Discriminator tensor test: PASSED"
        )
    else:
        print()
        print(
            "Discriminator tensor test: FAILED"
        )


if __name__ == "__main__":
    main()