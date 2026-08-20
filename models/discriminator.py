import torch
import torch.nn as nn


class ShareDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(
                3,
                64,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.LeakyReLU(
                0.2,
                inplace=True
            ),

            nn.Conv2d(
                64,
                128,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(
                0.2,
                inplace=True
            ),

            nn.Conv2d(
                128,
                256,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(
                0.2,
                inplace=True
            ),

            nn.Conv2d(
                256,
                512,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(
                0.2,
                inplace=True
            )
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),

            nn.Linear(
                512 * 2 * 2,
                256
            ),

            nn.LeakyReLU(
                0.2,
                inplace=True
            ),

            nn.Linear(
                256,
                1
            )
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)

        return x