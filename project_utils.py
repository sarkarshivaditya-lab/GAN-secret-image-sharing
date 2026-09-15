import csv
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image


NUM_SHARES = 4
IMAGE_SIZE = 256


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_cifar10_loaders(
    data_dir="data",
    train_images=10000,
    validation_images=1000,
    test_images=1000,
    batch_size=8,
    image_size=IMAGE_SIZE,
    seed=42,
):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

    train_dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=transform,
    )

    test_dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=transform,
    )

    requested_train = max(0, int(train_images))
    requested_validation = max(0, int(validation_images))
    requested_test = max(0, int(test_images))

    if requested_train < 1 or requested_validation < 1 or requested_test < 1:
        raise ValueError("train_images, validation_images, and test_images must be positive.")
    if requested_train + requested_validation > len(train_dataset):
        raise ValueError(
            "train_images + validation_images must not exceed the CIFAR-10 training set size."
        )
    if requested_test > len(test_dataset):
        raise ValueError("test_images must not exceed the CIFAR-10 test set size.")

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_indices = torch.randperm(len(train_dataset), generator=generator).tolist()
    train_split_indices = train_indices[:requested_train]
    validation_split_indices = train_indices[requested_train:requested_train + requested_validation]
    test_indices = list(range(requested_test))

    train_subset = torch.utils.data.Subset(train_dataset, train_split_indices)
    validation_subset = torch.utils.data.Subset(train_dataset, validation_split_indices)
    test_subset = torch.utils.data.Subset(test_dataset, test_indices)

    train_loader = torch.utils.data.DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        generator=generator,
    )

    validation_loader = torch.utils.data.DataLoader(
        validation_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    test_loader = torch.utils.data.DataLoader(
        test_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    return train_loader, validation_loader, test_loader


def psnr_from_mse(mse):
    mse = float(mse)
    if mse <= 0.0:
        return float("inf")
    return 10.0 * math.log10(1.0 / mse)


def batch_mse_sum(original, reconstructed):
    batch_size = original.shape[0]
    mse = torch.mean((original - reconstructed) ** 2)
    return mse.item() * batch_size


def validate_shares(shares):
    if not isinstance(shares, (tuple, list)):
        raise TypeError("Encoder must return a tuple or list of shares.")
    if len(shares) != NUM_SHARES:
        raise ValueError(
            f"Expected {NUM_SHARES} shares, received {len(shares)}."
        )
    for index, share in enumerate(shares, start=1):
        if not torch.is_tensor(share):
            raise TypeError(f"Share {index} is not a tensor.")
        if share.ndim != 4:
            raise ValueError(
                f"Share {index} must have shape [B,C,H,W], got {tuple(share.shape)}."
            )
    return list(shares)


def reconstruct(decoder, shares):
    shares = validate_shares(shares)
    return decoder(*shares)


def save_tensor_image(tensor, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_image(tensor.detach().cpu().clamp(0.0, 1.0), str(path))


def save_comparison_grid(images, shares, reconstruction, path):
    rows = [images]
    rows.extend(shares)
    rows.append(reconstruction)
    grid = make_grid(
        torch.cat(rows, dim=0),
        nrow=images.shape[0],
        padding=2,
    )
    save_tensor_image(grid, path)


def load_state_dict_compat(model, path, device):
    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=False,
    )

    if isinstance(checkpoint, dict):
        for key in (
            "model_state_dict",
            "state_dict",
            "encoder_state_dict",
            "decoder_state_dict",
        ):
            if key in checkpoint:
                checkpoint = checkpoint[key]
                break

    model.load_state_dict(checkpoint)
    return model


def save_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def save_csv(rows, path):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def checkpoint_paths(directory):
    directory = Path(directory)
    return {
        "encoder": directory / "encoder_best.pth",
        "decoder": directory / "decoder_best.pth",
        "discriminator": directory / "discriminator_best.pth",
        "metadata": directory / "best_info.json",
        "training_log": directory / "training_log.csv",
    }


def count_parameters(model):
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def model_parameter_summary(encoder, decoder, attacker, discriminator):
    return {
        "encoder_trainable_parameters": count_parameters(encoder),
        "decoder_trainable_parameters": count_parameters(decoder),
        "attacker_trainable_parameters": count_parameters(attacker),
        "discriminator_trainable_parameters": count_parameters(discriminator),
    }
