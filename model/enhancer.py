from pathlib import Path

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNetEnhancer(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, features=(32, 64, 128)):
        super().__init__()
        self.down_blocks = nn.ModuleList()
        self.up_transpose = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        channels = in_channels
        for feature in features:
            self.down_blocks.append(ConvBlock(channels, feature))
            channels = feature

        self.bottleneck = ConvBlock(features[-1], features[-1] * 2)

        current_channels = features[-1] * 2
        for feature in reversed(features):
            self.up_transpose.append(
                nn.ConvTranspose2d(current_channels, feature, kernel_size=2, stride=2)
            )
            self.up_blocks.append(ConvBlock(feature * 2, feature))
            current_channels = feature

        self.output_head = nn.Sequential(nn.Conv2d(features[0], out_channels, kernel_size=1), nn.Sigmoid())

    def forward(self, x):
        skips = []
        for down_block in self.down_blocks:
            x = down_block(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skips = list(reversed(skips))

        for transpose, block, skip in zip(self.up_transpose, self.up_blocks, skips):
            x = transpose(x)
            if x.shape[-2:] != skip.shape[-2:]:
                x = torch.nn.functional.interpolate(
                    x, size=skip.shape[-2:], mode="bilinear", align_corners=False
                )
            x = torch.cat([skip, x], dim=1)
            x = block(x)

        return self.output_head(x)


def load_enhancer(model_path, device):
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Enhancer weights not found at {model_path}")

    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    model = UNetEnhancer().to(device)
    model.load_state_dict(checkpoint)
    model.eval()
    return model
