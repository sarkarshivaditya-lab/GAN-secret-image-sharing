import torch
import torch.nn as nn


class ShareEncoder(nn.Module):
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
        )

        self.share_generator = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(256, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(256, affine=True),
            nn.ReLU(inplace=True),
        )

        self.share_heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(256, 64, kernel_size=3, padding=1),
                nn.InstanceNorm2d(64, affine=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 32, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 3, kernel_size=3, padding=1),
                nn.Sigmoid(),
            )
            for _ in range(4)
        ])

    def forward(self, x):
        features = self.share_generator(self.features(x))
        return tuple(head(features) for head in self.share_heads)
