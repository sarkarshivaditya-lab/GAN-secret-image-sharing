import torch

from project_utils import build_cifar10_loaders


def subset_indices(loader):
    return list(loader.dataset.indices)


def test_train_validation_splits_are_disjoint_and_deterministic():
    train_loader_1, validation_loader_1, test_loader_1 = build_cifar10_loaders(
        data_dir="data",
        train_images=100,
        validation_images=25,
        test_images=50,
        batch_size=8,
        image_size=32,
        seed=42,
    )
    train_loader_2, validation_loader_2, test_loader_2 = build_cifar10_loaders(
        data_dir="data",
        train_images=100,
        validation_images=25,
        test_images=50,
        batch_size=8,
        image_size=32,
        seed=42,
    )

    train_1 = set(subset_indices(train_loader_1))
    validation_1 = set(subset_indices(validation_loader_1))
    test_1 = set(test_loader_1.dataset.indices)

    assert train_1.isdisjoint(validation_1)
    assert train_1.isdisjoint(test_1)
    assert validation_1.isdisjoint(test_1)

    assert subset_indices(train_loader_1) == subset_indices(train_loader_2)
    assert subset_indices(validation_loader_1) == subset_indices(validation_loader_2)
    assert list(test_loader_1.dataset.indices) == list(test_loader_2.dataset.indices)

    assert len(train_1) == 100
    assert len(validation_1) == 25
    assert len(test_1) == 50
