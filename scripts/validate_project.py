import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from models.attacker import ShareAttacker
from models.decoder import ShareDecoder
from models.encoder import ShareEncoder
from models.privacy_discriminator import PrivacyDiscriminator


def main():
    device = (
        torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    print(f"Python: {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Device: {device}")

    image = torch.rand(2, 3, 256, 256, device=device)

    encoder = ShareEncoder().to(device).eval()
    decoder = ShareDecoder().to(device).eval()
    attacker = ShareAttacker().to(device).eval()
    discriminator = PrivacyDiscriminator().to(device).eval()

    with torch.no_grad():
        shares = encoder(image)

        if len(shares) != 4:
            raise RuntimeError("Encoder did not produce four shares.")

        for index, share in enumerate(shares, start=1):
            if share.shape != (2, 3, 32, 32):
                raise RuntimeError(
                    f"Share {index} has unexpected shape: {tuple(share.shape)}"
                )

        reconstruction = decoder(*shares)
        attack_reconstruction = attacker(shares[0])
        discriminator_output = discriminator(image, shares[0])

    expected = (2, 3, 256, 256)
    if reconstruction.shape != expected:
        raise RuntimeError(
            f"Decoder output shape is {tuple(reconstruction.shape)}, expected {expected}."
        )
    if attack_reconstruction.shape != expected:
        raise RuntimeError("Attacker output shape is incorrect.")
    if discriminator_output.shape != (2, 1):
        raise RuntimeError("Privacy discriminator output shape is incorrect.")

    print("Model smoke test passed.")


if __name__ == "__main__":
    main()
