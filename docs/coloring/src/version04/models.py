"""軽量 U-Net ベースのジェネレーターとディスクリミネーター（96x96対応）。"""

import torch
from torch import nn


class UNetGenerator(nn.Module):
    """Skip connection を持つ少し深めの U-Net 風ジェネレーター（96x96用）。"""

    def __init__(self):
        super().__init__()
        # Encoder (96 -> 48 -> 24 -> 12 -> 6)
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 48, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(48, 96, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(96),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(96, 192, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(192),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.enc4 = nn.Sequential(
            nn.Conv2d(192, 192, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(192),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Decoder with skip connections (6 -> 12 -> 24 -> 48 -> 96)
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(192, 192, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(384, 96, kernel_size=4, stride=2, padding=1),  # cat with enc3
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
        )
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(192, 48, kernel_size=4, stride=2, padding=1),  # cat with enc2
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )
        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(96, 3, kernel_size=4, stride=2, padding=1),  # cat with enc1
            nn.Tanh(),
        )

    def forward(self, gray):
        # Encoder
        e1 = self.enc1(gray)   # 48 x 48 x 48
        e2 = self.enc2(e1)     # 96 x 24 x 24
        e3 = self.enc3(e2)     # 192 x 12 x 12
        e4 = self.enc4(e3)     # 192 x 6 x 6

        # Decoder with skip connections
        d1 = self.dec1(e4)                        # 192 x 12 x 12
        d2 = self.dec2(torch.cat([d1, e3], 1))    # 96 x 24 x 24
        d3 = self.dec3(torch.cat([d2, e2], 1))    # 48 x 48 x 48
        d4 = self.dec4(torch.cat([d3, e1], 1))    # 3 x 96 x 96

        return d4


class PatchDiscriminator(nn.Module):
    """軽量 PatchGAN ディスクリミネーター（96x96用）。"""

    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(4, 48, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(48, 96, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(96),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(96, 192, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(192),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(192, 1, kernel_size=4, stride=1, padding=1),
        )

    def forward(self, gray, color):
        return self.model(torch.cat([gray, color], dim=1))


def initialize_weights(module):
    """正規分布初期化を適用。"""
    classname = module.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(module.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(module.weight.data, 1.0, 0.02)
        nn.init.constant_(module.bias.data, 0)


def build_models():
    generator = UNetGenerator()
    discriminator = PatchDiscriminator()
    generator.apply(initialize_weights)
    discriminator.apply(initialize_weights)
    return generator, discriminator
