import os
import json
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models.encoder import ShareEncoder
from models.decoder import ShareDecoder


DEVICE = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

NUM_SHARES = 4
BATCH_SIZE = 8
TEST_IMAGES = 1000

CHECKPOINT_DIR = "checkpoints/balanced_static_gan"
ENCODER_PATH = os.path.join(
    CHECKPOINT_DIR,
    "encoder_best.pth"
)
DECODER_PATH = os.path.join(
    CHECKPOINT_DIR,
    "decoder_best.pth"
)

OUTPUT_DIR = "outputs/secret_sharing_evaluation"
RESULTS_JSON = os.path.join(
    OUTPUT_DIR,
    "results.json"
)
RESULTS_TXT = os.path.join(
    OUTPUT_DIR,
    "results.txt"
)


def psnr_from_mse(mse):
    if mse <= 0:
        return float("inf")

    return 10.0 * math.log10(1.0 / mse)


def load_checkpoint(model, path):
    checkpoint = torch.load(
        path,
        map_location=DEVICE,
        weights_only=False
    )

    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        elif "model_state_dict" in checkpoint:
            checkpoint = checkpoint["model_state_dict"]

    model.load_state_dict(checkpoint)
    model.to(DEVICE)
    model.eval()


def load_models():
    print()
    print("Loading balanced static-share checkpoint...")
    print(f"Encoder: {ENCODER_PATH}")
    print(f"Decoder: {DECODER_PATH}")

    encoder = ShareEncoder().to(DEVICE)
    decoder = ShareDecoder().to(DEVICE)

    load_checkpoint(
        encoder,
        ENCODER_PATH
    )

    load_checkpoint(
        decoder,
        DECODER_PATH
    )

    return encoder, decoder


def get_test_loader():
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    dataset = datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=transform
    )

    if TEST_IMAGES < len(dataset):
        indices = list(range(TEST_IMAGES))
        dataset = torch.utils.data.Subset(
            dataset,
            indices
        )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    return loader


def normalize_share_output(share):
    if not torch.is_tensor(share):
        raise TypeError(
            f"Expected tensor share, received {type(share)}"
        )

    return share


def get_shares(encoder, images):
    output = encoder(images)

    if isinstance(output, (tuple, list)):
        shares = list(output)
    else:
        raise TypeError(
            "ShareEncoder must return four share tensors."
        )

    if len(shares) != NUM_SHARES:
        raise ValueError(
            f"Expected {NUM_SHARES} shares, got {len(shares)}."
        )

    shares = [
        normalize_share_output(share)
        for share in shares
    ]

    return shares


def combine_shares(decoder, shares):
    if len(shares) != NUM_SHARES:
        raise ValueError(
            f"Decoder requires {NUM_SHARES} shares."
        )

    return decoder(
        shares[0],
        shares[1],
        shares[2],
        shares[3]
    )


def combine_partial_shares(decoder, shares, count):
    """
    Evaluate whether partial subsets can reconstruct the image.

    This is intentionally done by zero-filling missing shares.
    It answers an important research question:

        How much information is available when only
        1, 2, or 3 of the 4 shares are present?

    The decoder architecture still receives four inputs.
    Missing shares are replaced by zeros.
    """

    if count < 1 or count > NUM_SHARES:
        raise ValueError(
            "count must be between 1 and 4"
        )

    partial = []

    for i in range(NUM_SHARES):
        if i < count:
            partial.append(shares[i])
        else:
            partial.append(
                torch.zeros_like(shares[i])
            )

    return combine_shares(
        decoder,
        partial
    )


def calculate_entropy(tensor):
    """
    Estimate Shannon entropy of a tensor after quantizing
    values into 256 bins.
    """

    values = (
        tensor
        .detach()
        .float()
        .cpu()
        .numpy()
        .reshape(-1)
    )

    if values.size == 0:
        return 0.0

    min_value = float(values.min())
    max_value = float(values.max())

    if max_value - min_value < 1e-12:
        return 0.0

    normalized = (
        values - min_value
    ) / (
        max_value - min_value
    )

    bins = np.clip(
        (normalized * 255).astype(
            np.int32
        ),
        0,
        255
    )

    counts = np.bincount(
        bins,
        minlength=256
    ).astype(np.float64)

    probabilities = (
        counts / counts.sum()
    )

    probabilities = probabilities[
        probabilities > 0
    ]

    entropy = -np.sum(
        probabilities
        * np.log2(probabilities)
    )

    return float(entropy)


def spatial_correlation(share):
    """
    Calculate horizontal and vertical pixel correlation.

    Values close to zero indicate weak local spatial dependence.

    IMPORTANT:
    All MPS calculations remain float32.
    MPS does not support float64 tensors.
    """

    x = share.detach().float()

    if x.ndim != 4:
        raise ValueError(
            f"Expected [B,C,H,W], got {tuple(x.shape)}"
        )

    horizontal_a = (
        x[:, :, :, :-1]
        .reshape(-1)
    )

    horizontal_b = (
        x[:, :, :, 1:]
        .reshape(-1)
    )

    vertical_a = (
        x[:, :, :-1, :]
        .reshape(-1)
    )

    vertical_b = (
        x[:, :, 1:, :]
        .reshape(-1)
    )

    def correlation(a, b):
        # MPS does not support float64.
        # Keep the entire calculation in float32.
        a = a.float()
        b = b.float()

        a_mean = a.mean()
        b_mean = b.mean()

        a_centered = a - a_mean
        b_centered = b - b_mean

        denominator = torch.sqrt(
            torch.sum(
                a_centered ** 2
            )
            * torch.sum(
                b_centered ** 2
            )
        )

        if (
            denominator.item()
            < 1e-12
        ):
            return 0.0

        value = (
            torch.sum(
                a_centered
                * b_centered
            )
            / denominator
        )

        return float(
            value.item()
        )

    h_corr = correlation(
        horizontal_a,
        horizontal_b
    )

    v_corr = correlation(
        vertical_a,
        vertical_b
    )

    return h_corr, v_corr


def cross_share_correlation(
    share_a,
    share_b
):
    """
    Pearson correlation between two different shares.

    IMPORTANT:
    Keep calculations in float32 because
    Apple MPS does not support float64 tensors.
    """

    a = (
        share_a
        .detach()
        .float()
        .reshape(-1)
    )

    b = (
        share_b
        .detach()
        .float()
        .reshape(-1)
    )

    # DO NOT use .double() here.
    # MPS does not support float64.
    a = a - a.mean()
    b = b - b.mean()

    denominator = torch.sqrt(
        torch.sum(a ** 2)
        * torch.sum(b ** 2)
    )

    if denominator.item() < 1e-12:
        return 0.0

    correlation = (
        torch.sum(a * b)
        / denominator
    )

    return float(
        correlation.item()
    )


def evaluate():
    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    print(
        f"Device: {DEVICE}"
    )

    print()
    print("=" * 70)
    print("SECRET-SHARING DIAGNOSTIC")
    print("=" * 70)
    print()

    print(
        f"Test images: {TEST_IMAGES}"
    )

    print(
        f"Batch size: {BATCH_SIZE}"
    )

    print(
        f"Number of shares: {NUM_SHARES}"
    )

    print()

    encoder, decoder = load_models()

    loader = get_test_loader()

    total_full_mse = 0.0

    total_partial_mse = {
        1: 0.0,
        2: 0.0,
        3: 0.0,
        4: 0.0
    }

    total_images = 0

    share_sum = [
        0.0
    ] * NUM_SHARES

    share_squared_sum = [
        0.0
    ] * NUM_SHARES

    share_count = [
        0
    ] * NUM_SHARES

    share_entropy_sum = [
        0.0
    ] * NUM_SHARES

    horizontal_corr_sum = [
        0.0
    ] * NUM_SHARES

    vertical_corr_sum = [
        0.0
    ] * NUM_SHARES

    cross_corr_sum = {}
    cross_corr_count = {}

    for i in range(NUM_SHARES):
        for j in range(
            i + 1,
            NUM_SHARES
        ):
            key = (
                f"share_{i + 1}"
                f"_vs_share_{j + 1}"
            )

            cross_corr_sum[key] = 0.0
            cross_corr_count[key] = 0

    with torch.no_grad():

        for batch_index, (
            images,
            _
        ) in enumerate(loader):

            images = images.to(
                DEVICE
            )

            shares = get_shares(
                encoder,
                images
            )

            reconstructed = combine_shares(
                decoder,
                shares
            )

            full_mse = F.mse_loss(
                reconstructed,
                images
            )

            batch_size = images.size(0)

            total_full_mse += (
                full_mse.item()
                * batch_size
            )

            for count in range(
                1,
                NUM_SHARES + 1
            ):

                partial_reconstruction = (
                    combine_partial_shares(
                        decoder,
                        shares,
                        count
                    )
                )

                partial_mse = F.mse_loss(
                    partial_reconstruction,
                    images
                )

                total_partial_mse[
                    count
                ] += (
                    partial_mse.item()
                    * batch_size
                )

            total_images += batch_size

            for share_index, share in enumerate(
                shares
            ):

                share_cpu = (
                    share
                    .detach()
                    .float()
                )

                pixels = (
                    share_cpu.numel()
                )

                share_sum[
                    share_index
                ] += (
                    share_cpu.sum().item()
                )

                share_squared_sum[
                    share_index
                ] += (
                    torch.sum(
                        share_cpu ** 2
                    ).item()
                )

                share_count[
                    share_index
                ] += pixels

                share_entropy_sum[
                    share_index
                ] += calculate_entropy(
                    share
                )

                h_corr, v_corr = (
                    spatial_correlation(
                        share
                    )
                )

                horizontal_corr_sum[
                    share_index
                ] += h_corr

                vertical_corr_sum[
                    share_index
                ] += v_corr

            for i in range(NUM_SHARES):

                for j in range(
                    i + 1,
                    NUM_SHARES
                ):

                    key = (
                        f"share_{i + 1}"
                        f"_vs_share_{j + 1}"
                    )

                    corr = (
                        cross_share_correlation(
                            shares[i],
                            shares[j]
                        )
                    )

                    cross_corr_sum[
                        key
                    ] += corr

                    cross_corr_count[
                        key
                    ] += 1

            if batch_index % 50 == 0:

                print(
                    f"Processed batch "
                    f"{batch_index + 1}"
                    f"/{len(loader)}"
                )

    full_mse = (
        total_full_mse
        / total_images
    )

    full_psnr = psnr_from_mse(
        full_mse
    )

    partial_results = {}

    for count in range(
        1,
        NUM_SHARES + 1
    ):

        mse = (
            total_partial_mse[count]
            / total_images
        )

        psnr = psnr_from_mse(
            mse
        )

        partial_results[
            str(count)
        ] = {
            "mse": float(mse),
            "psnr_db": float(psnr)
        }

    share_results = {}

    for i in range(NUM_SHARES):

        mean = (
            share_sum[i]
            / share_count[i]
        )

        mean_square = (
            share_squared_sum[i]
            / share_count[i]
        )

        variance = max(
            0.0,
            mean_square
            - mean ** 2
        )

        std = math.sqrt(
            variance
        )

        average_entropy = (
            share_entropy_sum[i]
            / len(loader)
        )

        average_h_corr = (
            horizontal_corr_sum[i]
            / len(loader)
        )

        average_v_corr = (
            vertical_corr_sum[i]
            / len(loader)
        )

        share_results[
            f"share_{i + 1}"
        ] = {
            "mean": float(mean),
            "std": float(std),
            "entropy_bits": float(
                average_entropy
            ),
            "horizontal_correlation": float(
                average_h_corr
            ),
            "vertical_correlation": float(
                average_v_corr
            )
        }

    cross_results = {}

    for key in cross_corr_sum:

        count = cross_corr_count[key]

        if count > 0:
            value = (
                cross_corr_sum[key]
                / count
            )
        else:
            value = 0.0

        cross_results[key] = float(
            value
        )

    results = {
        "device": str(DEVICE),
        "test_images": TEST_IMAGES,
        "batch_size": BATCH_SIZE,
        "num_shares": NUM_SHARES,
        "full_reconstruction": {
            "mse": float(full_mse),
            "psnr_db": float(full_psnr)
        },
        "partial_reconstruction":
            partial_results,
        "shares": share_results,
        "cross_share_correlation":
            cross_results
    }

    with open(
        RESULTS_JSON,
        "w"
    ) as f:

        json.dump(
            results,
            f,
            indent=4
        )

    write_text_results(
        results
    )

    print()
    print("=" * 70)
    print("SECRET-SHARING DIAGNOSTIC RESULTS")
    print("=" * 70)

    print()
    print("RECONSTRUCTION")
    print("-" * 70)

    for count, result in (
        partial_results.items()
    ):

        if int(count) == NUM_SHARES:
            label = "ALL 4 SHARES"
        else:
            label = (
                f"{count} SHARE(S)"
            )

        print(
            f"{label:<15} "
            f"MSE: {result['mse']:.6f}    "
            f"PSNR: {result['psnr_db']:.2f} dB"
        )

    print()
    print("SHARE STATISTICS")
    print("-" * 70)

    for i in range(NUM_SHARES):

        result = share_results[
            f"share_{i + 1}"
        ]

        print(
            f"Share {i + 1}"
        )

        print(
            f"  Mean: "
            f"{result['mean']:.6f}"
        )

        print(
            f"  Std:  "
            f"{result['std']:.6f}"
        )

        print(
            f"  Entropy: "
            f"{result['entropy_bits']:.4f} bits"
        )

        print(
            f"  Horizontal correlation: "
            f"{result['horizontal_correlation']:.6f}"
        )

        print(
            f"  Vertical correlation: "
            f"{result['vertical_correlation']:.6f}"
        )

    print()
    print("CROSS-SHARE CORRELATION")
    print("-" * 70)

    for key, value in (
        cross_results.items()
    ):

        print(
            f"{key}: {value:.6f}"
        )

    print()
    print("=" * 70)

    print(
        f"JSON results: {RESULTS_JSON}"
    )

    print(
        f"Text results: {RESULTS_TXT}"
    )

    print("=" * 70)


def write_text_results(results):

    lines = []

    lines.append(
        "=" * 70
    )

    lines.append(
        "SECRET-SHARING DIAGNOSTIC"
    )

    lines.append(
        "=" * 70
    )

    lines.append("")

    lines.append(
        f"Device: {results['device']}"
    )

    lines.append(
        f"Test images: "
        f"{results['test_images']}"
    )

    lines.append("")

    lines.append(
        "RECONSTRUCTION RESULTS"
    )

    lines.append(
        "-" * 70
    )

    for count, result in (
        results[
            "partial_reconstruction"
        ].items()
    ):

        lines.append(
            f"{count} share(s) | "
            f"MSE: {result['mse']:.6f} | "
            f"PSNR: {result['psnr_db']:.2f} dB"
        )

    lines.append("")

    lines.append(
        "SHARE STATISTICS"
    )

    lines.append(
        "-" * 70
    )

    for name, result in (
        results["shares"].items()
    ):

        lines.append(
            f"{name}"
        )

        lines.append(
            f"  Mean: "
            f"{result['mean']:.6f}"
        )

        lines.append(
            f"  Std: "
            f"{result['std']:.6f}"
        )

        lines.append(
            f"  Entropy: "
            f"{result['entropy_bits']:.4f} bits"
        )

        lines.append(
            f"  Horizontal correlation: "
            f"{result['horizontal_correlation']:.6f}"
        )

        lines.append(
            f"  Vertical correlation: "
            f"{result['vertical_correlation']:.6f}"
        )

    lines.append("")

    lines.append(
        "CROSS-SHARE CORRELATION"
    )

    lines.append(
        "-" * 70
    )

    for name, value in (
        results[
            "cross_share_correlation"
        ].items()
    ):

        lines.append(
            f"{name}: {value:.6f}"
        )

    lines.append("")

    lines.append(
        "=" * 70
    )

    with open(
        RESULTS_TXT,
        "w"
    ) as f:

        f.write(
            "\n".join(lines)
        )


if __name__ == "__main__":
    evaluate()