import torch
import torch.nn as nn


class PrivacyDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.image_encoder = nn.Sequential(
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
            ),

            nn.AdaptiveAvgPool2d(
                (1, 1)
            )
        )

        self.share_encoder = nn.Sequential(
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

            nn.AdaptiveAvgPool2d(
                (1, 1)
            )
        )

        self.classifier = nn.Sequential(
            nn.Linear(
                512 + 256,
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

    def forward(
        self,
        image,
        share
    ):
        image_features = self.image_encoder(
            image
        )

        share_features = self.share_encoder(
            share
        )

        image_features = image_features.flatten(
            start_dim=1
        )

        share_features = share_features.flatten(
            start_dim=1
        )

        combined = torch.cat(
            [
                image_features,
                share_features
            ],
            dim=1
        )

        return self.classifier(
            combined
        )