import torch
import torch.nn as nn


class ShareEncoder(nn.Module):
    """Encode a native 32x32 RGB image into a 3x32x32 payload and four static shares."""

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.payload = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 3, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        payload = self.payload(self.features(x))

        share1 = torch.rand_like(payload)
        share2 = torch.rand_like(payload)
        share3 = torch.rand_like(payload)
        share4 = torch.remainder(payload - share1 - share2 - share3, 1.0)

        return share1, share2, share3, share4
