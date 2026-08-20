import torch

from models.encoder import ShareEncoder
from models.attacker import ShareAttacker


def main():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print("Device:", device)

    encoder = ShareEncoder().to(device)
    attacker = ShareAttacker().to(device)

    test_image = torch.rand(
        2,
        3,
        256,
        256,
        device=device
    )

    with torch.no_grad():
        shares = encoder(test_image)

    share = shares[0]

    reconstructed = attacker(share)

    print()
    print("Original:", test_image.shape)
    print("Share:", share.shape)
    print("Attacker output:", reconstructed.shape)

    if reconstructed.shape == test_image.shape:
        print()
        print("Attacker tensor test: PASSED")
    else:
        print()
        print("Attacker tensor test: FAILED")


if __name__ == "__main__":
    main()