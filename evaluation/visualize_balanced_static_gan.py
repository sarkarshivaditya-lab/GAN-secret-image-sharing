import os
import json
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from models.encoder import ShareEncoder
from models.decoder import ShareDecoder


DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

CHECKPOINT_DIR = "checkpoints/balanced_static_gan"
OUTPUT_DIR = "outputs/balanced_static_gan_visualization"

ENCODER_PATH = os.path.join(
    CHECKPOINT_DIR,
    "encoder_best.pth"
)

DECODER_PATH = os.path.join(
    CHECKPOINT_DIR,
    "decoder_best.pth"
)

NUM_SHARES = 4
NUM_IMAGES = 8
BATCH_SIZE = 8


def load_checkpoint(model, path):
    checkpoint = torch.load(
        path,
        map_location=DEVICE,
        weights_only=False
    )

    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        elif "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        else:
            model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint)

    return model


def normalize_for_display(x):
    x = x.detach().cpu()

    x_min = x.amin(dim=(-3, -2, -1), keepdim=True)
    x_max = x.amax(dim=(-3, -2, -1), keepdim=True)

    x = (x - x_min) / (x_max - x_min + 1e-8)

    return x.clamp(0, 1)


def correlation_horizontal(x):
    x1 = x[:, :, :, :-1].flatten()
    x2 = x[:, :, :, 1:].flatten()

    x1 = x1 - x1.mean()
    x2 = x2 - x2.mean()

    denominator = torch.sqrt(
        (x1 ** 2).sum() *
        (x2 ** 2).sum()
    ) + 1e-8

    return ((x1 * x2).sum() / denominator).item()


def correlation_vertical(x):
    x1 = x[:, :, :-1, :].flatten()
    x2 = x[:, :, 1:, :].flatten()

    x1 = x1 - x1.mean()
    x2 = x2 - x2.mean()

    denominator = torch.sqrt(
        (x1 ** 2).sum() *
        (x2 ** 2).sum()
    ) + 1e-8

    return ((x1 * x2).sum() / denominator).item()


def calculate_psnr(original, reconstructed):
    mse = torch.mean(
        (original - reconstructed) ** 2
    ).item()

    if mse <= 1e-12:
        return float("inf")

    return 10.0 * torch.log10(
        torch.tensor(1.0 / mse)
    ).item()


def extract_shares(encoder_output):
    if isinstance(encoder_output, tuple):
        return encoder_output

    if isinstance(encoder_output, list):
        return encoder_output

    if torch.is_tensor(encoder_output):

        if encoder_output.dim() == 5:
            return tuple(
                encoder_output[:, i]
                for i in range(encoder_output.shape[1])
            )

        if encoder_output.dim() == 4:
            return tuple(
                encoder_output
                for _ in range(NUM_SHARES)
            )

    raise RuntimeError(
        "Unable to determine share format returned by ShareEncoder."
    )


def combine_shares_for_decoder(shares):
    try:
        return torch.cat(shares, dim=1)
    except Exception:
        return shares


def run_visualization():
    print("Device:", DEVICE)
    print()
    print("Loading balanced static-share checkpoint...")
    print("Encoder:", ENCODER_PATH)
    print("Decoder:", DECODER_PATH)
    print()

    encoder = ShareEncoder().to(DEVICE)
    decoder = ShareDecoder().to(DEVICE)

    encoder = load_checkpoint(
        encoder,
        ENCODER_PATH
    )

    decoder = load_checkpoint(
        decoder,
        DECODER_PATH
    )

    encoder.eval()
    decoder.eval()

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    test_dataset = datasets.CIFAR10(
        root="data",
        train=False,
        download=True,
        transform=transform
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    total_psnr = 0.0
    total_mse = 0.0
    count = 0

    share_statistics = [
        {
            "mean": [],
            "std": [],
            "h_corr": [],
            "v_corr": []
        }
        for _ in range(NUM_SHARES)
    ]

    with torch.no_grad():

        for images, _ in test_loader:

            images = images.to(DEVICE)

            encoded = encoder(images)

            shares = extract_shares(encoded)

            if len(shares) != NUM_SHARES:
                raise RuntimeError(
                    f"Expected {NUM_SHARES} shares, "
                    f"but encoder returned {len(shares)}."
                )

            decoder_input = combine_shares_for_decoder(
                shares
            )

            reconstructed = decoder(
    shares[0],
    shares[1],
    shares[2],
    shares[3]
)

            reconstructed = reconstructed.clamp(
                0.0,
                1.0
            )

            batch_mse = torch.mean(
                (images - reconstructed) ** 2
            ).item()

            batch_psnr = calculate_psnr(
                images,
                reconstructed
            )

            total_mse += batch_mse
            total_psnr += batch_psnr
            count += 1

            for i, share in enumerate(shares):

                share_statistics[i]["mean"].append(
                    share.mean().item()
                )

                share_statistics[i]["std"].append(
                    share.std().item()
                )

                share_statistics[i]["h_corr"].append(
                    correlation_horizontal(share)
                )

                share_statistics[i]["v_corr"].append(
                    correlation_vertical(share)
                )

            if count == 1:

                original_display = normalize_for_display(
                    images
                )

                reconstructed_display = normalize_for_display(
                    reconstructed
                )

                display_shares = [
                    normalize_for_display(share)
                    for share in shares
                ]

                num_show = min(
                    NUM_IMAGES,
                    images.shape[0]
                )

                fig, axes = plt.subplots(
                    NUM_SHARES + 2,
                    num_show,
                    figsize=(3 * num_show, 3 * (NUM_SHARES + 2))
                )

                if num_show == 1:
                    axes = axes.reshape(
                        NUM_SHARES + 2,
                        1
                    )

                for j in range(num_show):

                    axes[0, j].imshow(
                        original_display[j].permute(1, 2, 0)
                    )

                    axes[0, j].set_title(
                        "Original"
                    )

                    axes[0, j].axis("off")

                    for i in range(NUM_SHARES):

                        axes[i + 1, j].imshow(
                            display_shares[i][j].permute(
                                1,
                                2,
                                0
                            )
                        )

                        axes[i + 1, j].set_title(
                            f"Share {i + 1}"
                        )

                        axes[i + 1, j].axis("off")

                    axes[NUM_SHARES + 1, j].imshow(
                        reconstructed_display[j].permute(
                            1,
                            2,
                            0
                        )
                    )

                    axes[NUM_SHARES + 1, j].set_title(
                        "Reconstructed"
                    )

                    axes[NUM_SHARES + 1, j].axis("off")

                plt.tight_layout()

                visualization_path = os.path.join(
                    OUTPUT_DIR,
                    "share_visualization.png"
                )

                plt.savefig(
                    visualization_path,
                    dpi=150,
                    bbox_inches="tight"
                )

                plt.close()

                print(
                    "Visualization saved to:",
                    visualization_path
                )

            if count >= 125:
                break

    average_psnr = total_psnr / count
    average_mse = total_mse / count

    print()
    print("=" * 70)
    print("BALANCED STATIC-SHARE VISUAL DIAGNOSTIC")
    print("=" * 70)

    print(
        f"Average reconstruction MSE: "
        f"{average_mse:.6f}"
    )

    print(
        f"Average reconstruction PSNR: "
        f"{average_psnr:.2f} dB"
    )

    print()

    statistics_json = {}

    for i in range(NUM_SHARES):

        mean_value = sum(
            share_statistics[i]["mean"]
        ) / len(
            share_statistics[i]["mean"]
        )

        std_value = sum(
            share_statistics[i]["std"]
        ) / len(
            share_statistics[i]["std"]
        )

        h_corr = sum(
            share_statistics[i]["h_corr"]
        ) / len(
            share_statistics[i]["h_corr"]
        )

        v_corr = sum(
            share_statistics[i]["v_corr"]
        ) / len(
            share_statistics[i]["v_corr"]
        )

        print(
            f"Share {i + 1}"
        )

        print(
            f"  Mean: {mean_value:.6f}"
        )

        print(
            f"  Std:  {std_value:.6f}"
        )

        print(
            f"  Horizontal correlation: "
            f"{h_corr:.6f}"
        )

        print(
            f"  Vertical correlation: "
            f"{v_corr:.6f}"
        )

        print()

        statistics_json[f"share_{i + 1}"] = {
            "mean": mean_value,
            "std": std_value,
            "horizontal_correlation": h_corr,
            "vertical_correlation": v_corr
        }

    results = {
        "checkpoint": CHECKPOINT_DIR,
        "images_evaluated": count * BATCH_SIZE,
        "reconstruction_mse": average_mse,
        "reconstruction_psnr": average_psnr,
        "shares": statistics_json
    }

    results_path = os.path.join(
        OUTPUT_DIR,
        "diagnostic_results.json"
    )

    with open(
        results_path,
        "w"
    ) as f:
        json.dump(
            results,
            f,
            indent=4
        )

    print(
        "Results saved to:",
        results_path
    )

    print(
        "Visualization saved to:",
        os.path.join(
            OUTPUT_DIR,
            "share_visualization.png"
        )
    )

    print("=" * 70)


if __name__ == "__main__":
    run_visualization()