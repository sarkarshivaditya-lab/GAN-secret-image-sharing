import torch
import torch.nn as nn


class ShareDecoder(nn.Module):
    """Recover the learned payload and reconstruct a native 32x32 RGB image."""

    def __init__(self):
        super().__init__()

        self.decoder = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, share1, share2, share3, share4):
        payload = torch.remainder(share1 + share2 + share3 + share4, 1.0)
        return self.decoder(payload)
