import torch
import torch.nn as nn


class StaticDiscriminator(nn.Module):
    """Distinguishes learned shares from uniform random TV-static noise."""

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(512, 1)

    def forward(self, share):
        features = self.network(share).flatten(start_dim=1)
        return self.classifier(features)
