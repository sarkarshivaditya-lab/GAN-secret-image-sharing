import torch
import torch.nn as nn


class ShareEncoder(nn.Module):
    """Learn a compact payload and wrap it in four TV-static-like shares."""

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.payload = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 3, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        payload = self.payload(self.features(x))

        share1 = torch.rand_like(payload)
        share2 = torch.rand_like(payload)
        share3 = torch.rand_like(payload)
        share4 = torch.remainder(payload - share1 - share2 - share3, 1.0)

        return share1, share2, share3, share4
