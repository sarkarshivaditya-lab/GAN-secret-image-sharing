import torch
from models.encoder import ShareEncoder
from models.decoder import ShareDecoder

def main():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print("Device:", device)

    encoder = ShareEncoder().to(device)
    decoder = ShareDecoder().to(device)

    test_image = torch.rand(
        1,
        3,
        256,
        256,
        device=device
    )

    shares = encoder(test_image)

    print("\nInput:")
    print(test_image.shape)

    for i, share in enumerate(shares, start=1):
        print(f"Share {i}:")
        print(share.shape)

    reconstructed = decoder(*shares)

    print("\nReconstructed:")
    print(reconstructed.shape)

    print("\nModel test completed successfully.")

if __name__ == "__main__":
    main()