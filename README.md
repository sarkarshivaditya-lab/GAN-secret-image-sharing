# GAN-Based Secret Image Sharing with TV-Static Shares

A practical learned image-sharing system that learns a compact image representation and wraps it in four noise-like shares. The decoder recovers the learned representation from all four shares and reconstructs the source image.

This repository is an experimental machine-learning implementation. The active system uses a mathematically noise-masking share layer; it should not be presented as a formal cryptographic security proof without a separate cryptographic analysis.

## Current goal

The active pipeline is:

`source image → learned payload → four TV-static-like shares → payload recovery → learned decoder → reconstructed image`

The practical objective is that the saved shares look like random TV static while the complete set remains decodable.

## Architecture

`models/encoder.py` first compresses the 256 × 256 source image into a learned 3 × 32 × 32 payload.

The payload is then wrapped into four shares:

- Share 1 is an independent uniform random mask.
- Share 2 is an independent uniform random mask.
- Share 3 is an independent uniform random mask.
- Share 4 is the payload minus the first three masks, reduced modulo 1.

Consequently, the four-share sum modulo 1 recovers the payload exactly (up to floating-point precision). Because independent uniform masks are used, the individual shares are designed to have the statistics of uniform random noise rather than carrying ordinary visible image structure.

`models/decoder.py` performs the modular share fusion and then learns to reconstruct the original 256 × 256 image from the recovered payload.

`models/static_discriminator.py` remains as an auxiliary GAN component. It distinguishes the generated shares from fresh uniform noise during training. The masking construction, rather than the discriminator alone, is what makes the shares reliably noise-like.

## Training

The active training entry point is:

```bash
python -m training.train_static_gan_v2 \
  --train-images 10000 \
  --test-images 1000 \
  --epochs 30 \
  --batch-size 8
```

Training starts from scratch. Do not initialize this model from the previous Privacy-GAN checkpoints.

The training objective combines reconstruction loss with a low-weight static GAN loss. The static appearance is primarily enforced by the explicit random-mask share construction, so the GAN does not have to fight the reconstruction objective to make the shares random.

Checkpoints are written to `checkpoints/static_gan/`.

## Evaluation

Run:

```bash
python -m evaluation.evaluate_static_gan \
  --checkpoint-dir checkpoints/static_gan \
  --test-images 1000
```

The evaluator reports reconstruction PSNR, per-share noise statistics, and leave-one-share-out reconstruction. It creates:

`outputs/static_gan/static_reconstruction_grid.png`

`outputs/static_gan/reference_uniform_noise.png`

`outputs/static_gan/results.json`

The reconstruction grid contains the original image, enlarged shares, and the reconstruction. The reference image provides a direct visual comparison against fresh uniform noise.

## Existing Privacy-GAN baseline

The previous Privacy-GAN implementation remains under `training/train_privacy_gan.py` and `checkpoints/privacy_gan/`. It is preserved as a historical baseline. Its learned share heads were able to reconstruct well collectively but produced highly unbalanced shares, including one nearly constant share and another strongly reconstructable share.

The active TV-static system changes the representation mechanism rather than trying to force the previous architecture into a contradictory GAN objective.

## Dataset

The current implementation uses CIFAR-10 through TorchVision. Images are resized from 32 × 32 to 256 × 256 and converted to tensors in `[0, 1]`.

CIFAR-10 is the current training environment. Good performance on this dataset does not imply equivalent performance on arbitrary high-resolution images.

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
│   ├── train_static_gan.py
│   └── train_static_gan_v2.py
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

1. Pull the latest repository and Git LFS files.
2. Run `python -m scripts.validate_project`.
3. Run the small one-epoch smoke training command before a long run.
4. Run `training.train_static_gan_v2` for the full training pass.
5. Run `evaluation.evaluate_static_gan`.
6. Inspect the reconstruction grid against the reference uniform-noise image.
7. Preserve successful checkpoints before changing the configuration.
