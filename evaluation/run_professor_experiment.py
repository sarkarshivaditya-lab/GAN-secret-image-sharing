import os
import json
import math
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt

from models.encoder import ShareEncoder
from models.decoder import ShareDecoder
from models.attacker import ShareAttacker


NUM_SHARES = 4
IMAGE_SIZE = 256

DEFAULT_ENCODER_PATH = os.path.join(
    "checkpoints",
    "encoder_baseline.pth"
)

DEFAULT_DECODER_PATH = os.path.join(
    "checkpoints",
    "decoder_baseline.pth"
)

DEFAULT_OUTPUT_ROOT = "outputs/professor_experiment_baseline"

DEVICE = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


def load_checkpoint(model, path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Checkpoint not found: {path}"
        )

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

    return model


def load_models(encoder_path, decoder_path):
    print()
    print("Loading trained model...")
    print(f"Device: {DEVICE}")
    print(f"Encoder: {encoder_path}")
    print(f"Decoder: {decoder_path}")

    encoder = ShareEncoder()
    decoder = ShareDecoder()

    encoder = load_checkpoint(
        encoder,
        encoder_path
    )

    decoder = load_checkpoint(
        decoder,
        decoder_path
    )

    print("Models loaded successfully.")

    return encoder, decoder


def load_image(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Image not found: {path}"
        )

    image = Image.open(path).convert("RGB")

    original_size = image.size

    image = image.resize(
        (IMAGE_SIZE, IMAGE_SIZE),
        Image.Resampling.LANCZOS
    )

    array = np.asarray(
        image,
        dtype=np.float32
    ) / 255.0

    tensor = torch.from_numpy(
        array
    ).permute(
        2,
        0,
        1
    )

    tensor = tensor.unsqueeze(0)

    return tensor, original_size


def tensor_to_image(tensor):
    tensor = tensor.detach().cpu()

    tensor = tensor.squeeze(0)

    tensor = tensor.permute(
        1,
        2,
        0
    )

    tensor = tensor.clamp(
        0.0,
        1.0
    )

    return tensor.numpy()


def save_image(tensor, path):
    image = tensor_to_image(tensor)

    image_uint8 = (
        image * 255.0
    ).round().astype(
        np.uint8
    )

    Image.fromarray(
        image_uint8
    ).save(path)


def calculate_mse(original, reconstructed):
    error = (
        original
        - reconstructed
    )

    return float(
        torch.mean(
            error ** 2
        ).item()
    )


def calculate_rmse(original, reconstructed):
    mse = calculate_mse(
        original,
        reconstructed
    )

    return float(
        math.sqrt(mse)
    )


def calculate_mae(original, reconstructed):
    error = torch.abs(
        original
        - reconstructed
    )

    return float(
        torch.mean(
            error
        ).item()
    )


def calculate_psnr(original, reconstructed):
    mse = calculate_mse(
        original,
        reconstructed
    )

    if mse <= 1e-12:
        return float("inf")

    return float(
        10.0
        * math.log10(
            1.0 / mse
        )
    )


def calculate_max_error(original, reconstructed):
    error = torch.abs(
        original
        - reconstructed
    )

    return float(
        torch.max(
            error
        ).item()
    )


def calculate_ssim(original, reconstructed):
    """
    Calculate a multi-channel SSIM approximation.

    Images are expected to be in [0, 1].
    The calculation is performed independently
    for each RGB channel and then averaged.
    """

    window_size = 11
    sigma = 1.5

    coordinates = torch.arange(
        window_size,
        device=original.device,
        dtype=torch.float32
    )

    coordinates = (
        coordinates
        - window_size // 2
    )

    gaussian = torch.exp(
        -(
            coordinates ** 2
        )
        / (
            2.0 * sigma ** 2
        )
    )

    gaussian = (
        gaussian
        / gaussian.sum()
    )

    window_2d = (
        gaussian[:, None]
        * gaussian[None, :]
    )

    window = (
        window_2d
        .unsqueeze(0)
        .unsqueeze(0)
    )

    window = window.repeat(
        original.shape[1],
        1,
        1,
        1
    )

    padding = window_size // 2

    mu_x = F.conv2d(
        original,
        window,
        padding=padding,
        groups=original.shape[1]
    )

    mu_y = F.conv2d(
        reconstructed,
        window,
        padding=padding,
        groups=reconstructed.shape[1]
    )

    mu_x_sq = mu_x ** 2
    mu_y_sq = mu_y ** 2
    mu_xy = mu_x * mu_y

    sigma_x_sq = (
        F.conv2d(
            original ** 2,
            window,
            padding=padding,
            groups=original.shape[1]
        )
        - mu_x_sq
    )

    sigma_y_sq = (
        F.conv2d(
            reconstructed ** 2,
            window,
            padding=padding,
            groups=reconstructed.shape[1]
        )
        - mu_y_sq
    )

    sigma_xy = (
        F.conv2d(
            original * reconstructed,
            window,
            padding=padding,
            groups=original.shape[1]
        )
        - mu_xy
    )

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    numerator = (
        (2.0 * mu_xy + c1)
        * (2.0 * sigma_xy + c2)
    )

    denominator = (
        (mu_x_sq + mu_y_sq + c1)
        * (sigma_x_sq + sigma_y_sq + c2)
    )

    ssim_map = (
        numerator
        / (denominator + 1e-12)
    )

    return float(
        ssim_map.mean().item()
    )


def calculate_error_matrix(
    original,
    reconstructed
):
    error = torch.abs(
        original
        - reconstructed
    )

    return error.squeeze(
        0
    ).detach().cpu().numpy()


def save_error_matrix(
    error_matrix,
    output_path
):
    np.save(
        output_path,
        error_matrix
    )


def save_error_heatmap(
    error_matrix,
    output_path
):
    error_map = (
        error_matrix
        .mean(axis=0)
    )

    plt.figure(
        figsize=(7, 7)
    )

    plt.imshow(
        error_map,
        cmap="viridis",
        vmin=0.0,
        vmax=max(
            float(error_map.max()),
            1e-6
        )
    )

    plt.colorbar(
        label="Absolute pixel error"
    )

    plt.title(
        "Reconstruction Error Map"
    )

    plt.axis("off")

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()


def save_share(
    share,
    output_path
):
    share_image = (
        share
        .detach()
        .cpu()
        .squeeze(0)
        .permute(1, 2, 0)
        .clamp(0.0, 1.0)
        .numpy()
    )

    share_uint8 = (
        share_image
        * 255.0
    ).round().astype(
        np.uint8
    )

    Image.fromarray(
        share_uint8
    ).save(output_path)


def calculate_share_statistics(share):
    tensor = share.detach().float()

    mean = float(
        tensor.mean().item()
    )

    std = float(
        tensor.std().item()
    )

    minimum = float(
        tensor.min().item()
    )

    maximum = float(
        tensor.max().item()
    )

    return {
        "mean": mean,
        "std": std,
        "minimum": minimum,
        "maximum": maximum
    }


def save_comparison_figure(
    original,
    shares,
    reconstructed,
    output_path
):
    original_image = tensor_to_image(
        original
    )

    reconstructed_image = tensor_to_image(
        reconstructed
    )

    share_images = []

    for share in shares:
        share_images.append(
            tensor_to_image(
                share
            )
        )

    figure, axes = plt.subplots(
        2,
        3,
        figsize=(14, 9)
    )

    axes[0, 0].imshow(
        original_image
    )

    axes[0, 0].set_title(
        "Original Image"
    )

    axes[0, 0].axis("off")

    for i in range(NUM_SHARES):
        row = 0 if i < 2 else 1

        column = (
            i + 1
            if i < 2
            else i - 2
        )

        axes[row, column].imshow(
            share_images[i]
        )

        axes[row, column].set_title(
            f"Share {i + 1}"
        )

        axes[row, column].axis("off")

    axes[1, 2].imshow(
        reconstructed_image
    )

    axes[1, 2].set_title(
        "Reconstructed Image"
    )

    axes[1, 2].axis("off")

    figure.suptitle(
        "GAN Secret Image Sharing Experiment",
        fontsize=16
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()


def find_attacker_checkpoint(
    share_index,
    attacker_dir
):
    if attacker_dir is None:
        return None

    path = os.path.join(
        attacker_dir,
        f"attacker_share_{share_index}.pth"
    )

    if os.path.exists(path):
        return path

    return None


def evaluate_attacker(
    share,
    original,
    share_index,
    attacker_dir
):
    checkpoint_path = find_attacker_checkpoint(
        share_index,
        attacker_dir
    )

    if checkpoint_path is None:
        if attacker_dir is None:
            return {
                "available": False,
                "reason": (
                    "Attacker evaluation disabled. "
                    "No --attacker-dir was supplied."
                )
            }

        return {
            "available": False,
            "reason": (
                f"No attacker checkpoint found for "
                f"Share {share_index} in {attacker_dir}."
            )
        }

    attacker = ShareAttacker()

    try:
        attacker = load_checkpoint(
            attacker,
            checkpoint_path
        )
    except Exception as error:
        return {
            "available": False,
            "reason": str(error),
            "checkpoint": checkpoint_path
        }

    with torch.no_grad():
        attacker_reconstruction = attacker(
            share
        )

    attacker_mse = calculate_mse(
        original,
        attacker_reconstruction
    )

    attacker_psnr = calculate_psnr(
        original,
        attacker_reconstruction
    )

    attacker_mae = calculate_mae(
        original,
        attacker_reconstruction
    )

    return {
        "available": True,
        "checkpoint": checkpoint_path,
        "mse": attacker_mse,
        "rmse": math.sqrt(attacker_mse),
        "mae": attacker_mae,
        "psnr_db": attacker_psnr
    }


def run_single_experiment(
    image_path,
    image_number,
    encoder,
    decoder,
    output_root,
    attacker_dir
):
    experiment_dir = os.path.join(
        output_root,
        f"image_{image_number}"
    )

    os.makedirs(
        experiment_dir,
        exist_ok=True
    )

    print()
    print(
        f"Processing Image {image_number}"
    )

    print(
        f"Input: {image_path}"
    )

    original, original_size = load_image(
        image_path
    )

    original = original.to(
        DEVICE
    )

    print(
        f"Original dimensions: "
        f"{original_size[0]} x "
        f"{original_size[1]}"
    )

    print(
        f"Model input dimensions: "
        f"{IMAGE_SIZE} x {IMAGE_SIZE}"
    )

    with torch.no_grad():
        shares = list(
            encoder(original)
        )

        if len(shares) != NUM_SHARES:
            raise RuntimeError(
                f"Expected {NUM_SHARES} shares, "
                f"received {len(shares)}."
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

    original_path = os.path.join(
        experiment_dir,
        "original_256x256.png"
    )

    reconstructed_path = os.path.join(
        experiment_dir,
        "reconstructed_256x256.png"
    )

    save_image(
        original,
        original_path
    )

    save_image(
        reconstructed,
        reconstructed_path
    )

    share_paths = []

    for i, share in enumerate(
        shares,
        start=1
    ):
        share_path = os.path.join(
            experiment_dir,
            f"share_{i}.png"
        )

        save_share(
            share,
            share_path
        )

        share_paths.append(
            share_path
        )

    mse = calculate_mse(
        original,
        reconstructed
    )

    rmse = calculate_rmse(
        original,
        reconstructed
    )

    mae = calculate_mae(
        original,
        reconstructed
    )

    psnr = calculate_psnr(
        original,
        reconstructed
    )

    ssim = calculate_ssim(
        original,
        reconstructed
    )

    max_error = calculate_max_error(
        original,
        reconstructed
    )

    error_matrix = calculate_error_matrix(
        original,
        reconstructed
    )

    error_matrix_path = os.path.join(
        experiment_dir,
        "absolute_error_matrix.npy"
    )

    save_error_matrix(
        error_matrix,
        error_matrix_path
    )

    heatmap_path = os.path.join(
        experiment_dir,
        "error_heatmap.png"
    )

    save_error_heatmap(
        error_matrix,
        heatmap_path
    )

    comparison_path = os.path.join(
        experiment_dir,
        "comparison.png"
    )

    save_comparison_figure(
        original,
        shares,
        reconstructed,
        comparison_path
    )

    share_statistics = {}

    for i, share in enumerate(
        shares,
        start=1
    ):
        share_statistics[
            f"share_{i}"
        ] = calculate_share_statistics(
            share
        )

    attacker_results = {}

    for i, share in enumerate(
        shares,
        start=1
    ):
        attacker_results[
            f"share_{i}"
        ] = evaluate_attacker(
            share,
            original,
            i,
            attacker_dir
        )

    metrics = {
        "image_number": image_number,
        "input_path": image_path,
        "original_dimensions": {
            "width": original_size[0],
            "height": original_size[1]
        },
        "model_dimensions": {
            "width": IMAGE_SIZE,
            "height": IMAGE_SIZE
        },
        "number_of_shares": NUM_SHARES,
        "reconstruction": {
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "psnr_db": psnr,
            "ssim": ssim,
            "max_absolute_error": max_error
        },
        "matrix": {
            "definition": (
                "E(x,y,c) = "
                "|I(x,y,c) - I_hat(x,y,c)|"
            ),
            "shape": list(
                error_matrix.shape
            ),
            "mean_absolute_error": float(
                error_matrix.mean()
            ),
            "maximum_absolute_error": float(
                error_matrix.max()
            ),
            "minimum_absolute_error": float(
                error_matrix.min()
            )
        },
        "shares": share_statistics,
        "individual_share_attack_results": attacker_results,
        "files": {
            "original": original_path,
            "reconstructed": reconstructed_path,
            "shares": share_paths,
            "error_matrix": error_matrix_path,
            "error_heatmap": heatmap_path,
            "comparison": comparison_path
        }
    }

    results_path = os.path.join(
        experiment_dir,
        "results.json"
    )

    with open(
        results_path,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            metrics,
            file,
            indent=4
        )

    report_path = os.path.join(
        experiment_dir,
        "report.txt"
    )

    write_text_report(
        metrics,
        report_path
    )

    print()
    print(
        f"Image {image_number} results"
    )

    print(
        f"MSE:       {mse:.8f}"
    )

    print(
        f"RMSE:      {rmse:.8f}"
    )

    print(
        f"MAE:       {mae:.8f}"
    )

    print(
        f"PSNR:      {psnr:.4f} dB"
    )

    print(
        f"SSIM:      {ssim:.6f}"
    )

    print(
        f"Max error: {max_error:.8f}"
    )

    print()
    print(
        f"Results saved to: {experiment_dir}"
    )

    return metrics


def write_text_report(
    metrics,
    output_path
):
    reconstruction = metrics[
        "reconstruction"
    ]

    lines = []

    lines.append(
        "GAN SECRET IMAGE SHARING"
    )

    lines.append(
        "Professor Experiment Report"
    )

    lines.append(
        ""
    )

    lines.append(
        f"Input image: "
        f"{metrics['input_path']}"
    )

    dimensions = metrics[
        "original_dimensions"
    ]

    lines.append(
        f"Original dimensions: "
        f"{dimensions['width']} x "
        f"{dimensions['height']}"
    )

    lines.append(
        f"Model dimensions: "
        f"{IMAGE_SIZE} x {IMAGE_SIZE}"
    )

    lines.append(
        ""
    )

    lines.append(
        "Reconstruction metrics"
    )

    lines.append(
        f"MSE: "
        f"{reconstruction['mse']:.8f}"
    )

    lines.append(
        f"RMSE: "
        f"{reconstruction['rmse']:.8f}"
    )

    lines.append(
        f"MAE: "
        f"{reconstruction['mae']:.8f}"
    )

    lines.append(
        f"PSNR: "
        f"{reconstruction['psnr_db']:.4f} dB"
    )

    lines.append(
        f"SSIM: "
        f"{reconstruction['ssim']:.6f}"
    )

    lines.append(
        f"Maximum absolute pixel error: "
        f"{reconstruction['max_absolute_error']:.8f}"
    )

    lines.append(
        ""
    )

    lines.append(
        "Matrix comparison"
    )

    lines.append(
        "For every RGB pixel:"
    )

    lines.append(
        "E(x,y,c) = |I(x,y,c) - I_hat(x,y,c)|"
    )

    matrix = metrics[
        "matrix"
    ]

    lines.append(
        f"Error matrix shape: "
        f"{matrix['shape']}"
    )

    lines.append(
        f"Mean matrix error: "
        f"{matrix['mean_absolute_error']:.8f}"
    )

    lines.append(
        f"Maximum matrix error: "
        f"{matrix['maximum_absolute_error']:.8f}"
    )

    lines.append(
        ""
    )

    lines.append(
        "Individual-share attacker results"
    )

    for share_name, result in metrics[
        "individual_share_attack_results"
    ].items():

        lines.append(
            f"{share_name}:"
        )

        if not result.get(
            "available",
            False
        ):
            lines.append(
                f"  Attacker unavailable: "
                f"{result.get('reason', 'Unknown reason')}"
            )
            continue

        lines.append(
            f"  Checkpoint: "
            f"{result['checkpoint']}"
        )

        lines.append(
            f"  MSE: "
            f"{result['mse']:.8f}"
        )

        lines.append(
            f"  RMSE: "
            f"{result['rmse']:.8f}"
        )

        lines.append(
            f"  MAE: "
            f"{result['mae']:.8f}"
        )

        lines.append(
            f"  PSNR: "
            f"{result['psnr_db']:.4f} dB"
        )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:
        file.write(
            "\n".join(lines)
        )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run the professor's two-image "
            "GAN secret-sharing experiment."
        )
    )

    parser.add_argument(
        "--image1",
        required=True,
        help="Path to first professor image."
    )

    parser.add_argument(
        "--image2",
        required=True,
        help="Path to second professor image."
    )

    parser.add_argument(
        "--encoder",
        default=DEFAULT_ENCODER_PATH,
        help=(
            "Path to the trained encoder checkpoint. "
            "Default: checkpoints/encoder_baseline.pth"
        )
    )

    parser.add_argument(
        "--decoder",
        default=DEFAULT_DECODER_PATH,
        help=(
            "Path to the trained decoder checkpoint. "
            "Default: checkpoints/decoder_baseline.pth"
        )
    )

    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_ROOT,
        help=(
            "Directory where experiment results "
            "will be saved."
        )
    )

    parser.add_argument(
        "--attacker-dir",
        default=None,
        help=(
            "Optional directory containing matching "
            "attacker_share_1.pth through "
            "attacker_share_4.pth. "
            "If omitted, attacker evaluation is disabled."
        )
    )

    args = parser.parse_args()

    os.makedirs(
        args.output,
        exist_ok=True
    )

    print()
    print(
        "GAN SECRET IMAGE SHARING"
    )

    print(
        "Professor Two-Image Experiment"
    )

    print()

    print(
        f"Encoder checkpoint: {args.encoder}"
    )

    print(
        f"Decoder checkpoint: {args.decoder}"
    )

    if args.attacker_dir is None:
        print(
            "Attacker evaluation: disabled"
        )
    else:
        print(
            f"Attacker directory: {args.attacker_dir}"
        )

    encoder, decoder = load_models(
        args.encoder,
        args.decoder
    )

    image1_results = run_single_experiment(
        args.image1,
        1,
        encoder,
        decoder,
        args.output,
        args.attacker_dir
    )

    image2_results = run_single_experiment(
        args.image2,
        2,
        encoder,
        decoder,
        args.output,
        args.attacker_dir
    )

    summary = {
        "device": str(DEVICE),
        "encoder_checkpoint": args.encoder,
        "decoder_checkpoint": args.decoder,
        "attacker_directory": args.attacker_dir,
        "images": {
            "image_1": image1_results,
            "image_2": image2_results
        }
    }

    summary_path = os.path.join(
        args.output,
        "summary.json"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            summary,
            file,
            indent=4
        )

    print()
    print(
        "Experiment completed."
    )

    print()
    print(
        f"All results: {args.output}"
    )

    print(
        f"Summary: {summary_path}"
    )


if __name__ == "__main__":
    main()