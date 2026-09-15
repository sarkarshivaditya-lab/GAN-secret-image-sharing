from project_utils import build_cifar10_loaders


def subset_indices(loader):
    return list(loader.dataset.indices)


def test_cifar10_splits_are_disjoint_and_deterministic(tmp_path):
    train_loader, validation_loader, test_loader = build_cifar10_loaders(
        data_dir=tmp_path,
        train_images=100,
        validation_images=50,
        test_images=50,
        batch_size=8,
        image_size=32,
        seed=42,
    )

    train_indices = subset_indices(train_loader)
    validation_indices = subset_indices(validation_loader)
    test_indices = subset_indices(test_loader)

    assert len(train_indices) == 100
    assert len(validation_indices) == 50
    assert len(test_indices) == 50
    assert set(train_indices).isdisjoint(validation_indices)
    assert len(set(train_indices).intersection(validation_indices)) == 0

    second_train, second_validation, second_test = build_cifar10_loaders(
        data_dir=tmp_path,
        train_images=100,
        validation_images=50,
        test_images=50,
        batch_size=8,
        image_size=32,
        seed=42,
    )

    assert train_indices == subset_indices(second_train)
    assert validation_indices == subset_indices(second_validation)
    assert test_indices == subset_indices(second_test)
    assert set(train_indices).isdisjoint(set(test_indices))
    assert set(validation_indices).isdisjoint(set(test_indices))
