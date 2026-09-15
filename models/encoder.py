import torch
import torch.nn as nn


class ShareEncoder(nn.Module):
    """Learn a compact latent payload and wrap it in four noise-like shares."""

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.InstanceNorm2d(256, affine=True),
            nn.ReLU(inplace=True),
        )

        self.latent = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 3, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        payload = self.latent(self.features(x))

        share1 = torch.rand_like(payload)
        share2 = torch.rand_like(payload)
        share3 = torch.rand_like(payload)
        share4 = torch.remainder(payload - share1 - share2 - share3, 1.0)

        return share1, share2, share3, share4
