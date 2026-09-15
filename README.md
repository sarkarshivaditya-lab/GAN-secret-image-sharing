# GAN-Based Secret Image Sharing with TV-Static Shares

A practical learned image-sharing system that encodes an image into four small learned shares designed to resemble television static while allowing a decoder to reconstruct the original image from all four shares.

This repository is an experimental machine-learning implementation, not a cryptographic secret-sharing scheme. It does not provide a formal secrecy, indistinguishability, or `(k,n)` threshold guarantee.

## Current goal

The active direction is straightforward:

`source image → encoder → four static-like shares → decoder → reconstructed image`

The desired behavior is that an individual encoded share looks as close as practical to random TV static, while the four shares together retain enough information for accurate reconstruction.

## Architecture

`models/encoder.py` maps a 3-channel 256 × 256 image to four 3-channel 32 × 32 shares. Each share has its own output head.

`models/decoder.py` concatenates the four shares and reconstructs a 3-channel 256 × 256 image.

`models/static_discriminator.py` provides a separate discriminator for each share. Each discriminator learns to distinguish learned shares from uniform random noise, giving the encoder an adversarial signal toward TV-static appearance.

## TV-Static GAN

The maintained training entry point for the new direction is:

```bash
python -m training.train_static_gan \
  --train-images 10000 \
  --test-images 1000 \
  --epochs 30 \
  --batch-size 8
```

The generator is trained with three signals:

- Reconstruction loss, so the four shares remain useful together.
- Static GAN loss, so generated shares approach random-noise appearance.
- Static-statistics loss, which encourages a mean near 0.5, standard deviation near the uniform-noise target, and strong pixel-to-pixel variation.

The default checkpoint directory is `checkpoints/static_gan/`.

If an existing trained encoder/decoder should be used as the starting point:

```bash
python -m training.train_static_gan \
  --init-checkpoint-dir checkpoints/privacy_gan
```

## Evaluation

Run:

```bash
python -m evaluation.evaluate_static_gan \
  --checkpoint-dir checkpoints/static_gan \
  --test-images 1000
```

The evaluator reports reconstruction PSNR and basic staticness statistics for each share. It also creates:

`outputs/static_gan/static_reconstruction_grid.png`

`outputs/static_gan/results.json`

The visual grid contains the original image, enlarged encoded shares, and the reconstructed image.

## Existing Privacy-GAN baseline

The previous Privacy-GAN implementation remains in the repository under `training/train_privacy_gan.py` and `checkpoints/privacy_gan/`. It is preserved as a baseline and is not overwritten by the TV-static implementation.

Its latest preserved experiment reached approximately 36.4 dB legitimate reconstruction, but the shares were highly unbalanced. One share remained strongly reconstructable while another collapsed toward a nearly constant image. The new TV-static trainer is intended to address the practical requirement that the encoded outputs themselves look like noise rather than recognizable image content.

## Dataset

The current implementation uses CIFAR-10 through TorchVision. Images are resized from 32 × 32 to 256 × 256 and converted to tensors in `[0, 1]`.

CIFAR-10 is being used as the initial training environment. The model should not be assumed to generalize to arbitrary high-resolution imagery without further training and validation.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The repository uses Git LFS for `.pth` checkpoints:

```bash
brew install git-lfs
git lfs install
git lfs pull
```

Validate the installation with:

```bash
python -m scripts.validate_project
```

## Repository structure

```text
GAN-secret-image-sharing/
├── models/
│   ├── encoder.py
│   ├── decoder.py
│   ├── attacker.py
│   ├── discriminator.py
│   ├── privacy_discriminator.py
│   └── static_discriminator.py
├── training/
│   ├── train_privacy_gan.py
│   └── train_static_gan.py
├── evaluation/
│   ├── evaluate_privacy_gan.py
│   ├── evaluate_static_gan.py
│   ├── compare_privacy_models.py
│   └── plot_privacy_gan_history.py
├── scripts/
│   └── validate_project.py
├── checkpoints/
├── outputs/
├── data/
├── project_utils.py
├── requirements.txt
└── README.md
```

## Practical workflow

1. Pull the latest repository and Git LFS checkpoints.
2. Run `python -m scripts.validate_project`.
3. Train `training.train_static_gan`.
4. Inspect `outputs/static_gan/static_reconstruction_grid.png`.
5. Run `evaluation.evaluate_static_gan`.
6. If the shares are sufficiently static but reconstruction is weak, reduce the static weights. If reconstruction is strong but shares still contain visible structure, increase the static weights gradually.
7. Preserve successful checkpoints before changing the training configuration.
