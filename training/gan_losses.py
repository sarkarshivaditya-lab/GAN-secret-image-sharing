import torch
import torch.nn.functional as F


def make_real_share_distribution(
    batch_size,
    channels,
    height,
    width,
    device
):
    return torch.randn(
        batch_size,
        channels,
        height,
        width,
        device=device
    )


def discriminator_loss(
    real_logits,
    fake_logits
):
    real_targets = torch.ones_like(
        real_logits
    )

    fake_targets = torch.zeros_like(
        fake_logits
    )

    real_loss = F.binary_cross_entropy_with_logits(
        real_logits,
        real_targets
    )

    fake_loss = F.binary_cross_entropy_with_logits(
        fake_logits,
        fake_targets
    )

    return (
        real_loss + fake_loss
    ) * 0.5


def generator_gan_loss(
    fake_logits
):
    real_targets = torch.ones_like(
        fake_logits
    )

    return F.binary_cross_entropy_with_logits(
        fake_logits,
        real_targets
    )


def test_gan_losses():
    device = (
        torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cpu")
    )

    batch_size = 4

    real = make_real_share_distribution(
        batch_size,
        3,
        32,
        32,
        device
    )

    fake = torch.randn(
        batch_size,
        3,
        32,
        32,
        device=device
    )

    real_logits = torch.randn(
        batch_size,
        1,
        device=device
    )

    fake_logits = torch.randn(
        batch_size,
        1,
        device=device
    )

    d_loss = discriminator_loss(
        real_logits,
        fake_logits
    )

    g_loss = generator_gan_loss(
        fake_logits
    )

    print("Device:", device)
    print()
    print("Real distribution:")
    print(real.shape)

    print()
    print("Fake distribution:")
    print(fake.shape)

    print()
    print(
        "Discriminator loss:",
        d_loss.item()
    )

    print(
        "Generator GAN loss:",
        g_loss.item()
    )

    if (
        torch.isfinite(d_loss)
        and torch.isfinite(g_loss)
    ):
        print()
        print(
            "GAN loss test: PASSED"
        )
    else:
        print()
        print(
            "GAN loss test: FAILED"
        )


if __name__ == "__main__":
    test_gan_losses()