import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from models.decoder import ShareDecoder
from models.encoder import ShareEncoder
from models.static_discriminator import StaticDiscriminator


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
    discriminator = StaticDiscriminator().to(device).eval()

    with torch.no_grad():
        shares = encoder(image)

        if len(shares) != 4:
            raise RuntimeError("Encoder did not produce four shares.")

        for index, share in enumerate(shares, start=1):
            if share.shape != (2, 3, 32, 32):
                raise RuntimeError(
                    f"Share {index} has unexpected shape: {tuple(share.shape)}"
                )
            if share.min().item() < 0.0 or share.max().item() >= 1.0:
                raise RuntimeError(f"Share {index} is outside the expected [0, 1) range.")

        payload = torch.remainder(sum(shares), 1.0)
        reconstruction = decoder(*shares)
        discriminator_output = discriminator(shares[0])

        masked = list(shares)
        masked[0] = torch.zeros_like(masked[0])
        missing_share_payload = torch.remainder(sum(masked), 1.0)

    expected = (2, 3, 256, 256)
    if reconstruction.shape != expected:
        raise RuntimeError(
            f"Decoder output shape is {tuple(reconstruction.shape)}, expected {expected}."
        )
    if discriminator_output.shape != (2, 1):
        raise RuntimeError("Static discriminator output shape is incorrect.")
    if not torch.isfinite(payload).all():
        raise RuntimeError("Recovered payload contains non-finite values.")
    if torch.equal(payload, missing_share_payload):
        raise RuntimeError("Removing a share did not change the recovered payload.")

    print("TV-static share construction smoke test passed.")


if __name__ == "__main__":
    main()
