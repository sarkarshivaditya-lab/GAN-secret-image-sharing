import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot Privacy-GAN training history."
    )
    parser.add_argument(
        "--history",
        default="checkpoints/privacy_gan/training_log.csv"
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/privacy_gan/training_plots"
    )
    return parser.parse_args()


def read_history(path):
    with Path(path).open("r", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError("Training history is empty.")
    return rows


def plot_series(epochs, values, title, ylabel, output_path):
    plt.figure(figsize=(9, 5))
    plt.plot(epochs, values)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def main():
    args = parse_args()
    rows = read_history(args.history)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    epochs = [int(row["epoch"]) for row in rows]

    metrics = {
        "validation_psnr_db": (
            "Validation reconstruction PSNR",
            "PSNR (dB)",
            "validation_psnr.png",
        ),
        "train_reconstruction_loss": (
            "Reconstruction loss",
            "MSE",
            "reconstruction_loss.png",
        ),
        "train_attack_loss": (
            "Single-share attacker loss",
            "MSE",
            "attacker_loss.png",
        ),
        "train_privacy_loss": (
            "Privacy discriminator generator loss",
            "BCE loss",
            "privacy_loss.png",
        ),
        "train_discriminator_accuracy": (
            "Privacy discriminator accuracy",
            "Accuracy",
            "discriminator_accuracy.png",
        ),
        "train_generator_loss": (
            "Generator objective",
            "Loss",
            "generator_loss.png",
        ),
    }

    for key, (title, ylabel, filename) in metrics.items():
        values = [float(row[key]) for row in rows]
        plot_series(
            epochs,
            values,
            title,
            ylabel,
            output_dir / filename,
        )

    print(f"Plots saved to {output_dir}")


if __name__ == "__main__":
    main()
