"""最もシンプルなGANモデル。

グレースケール画像をカラー画像に変換するGenerator と、
本物/偽物を判定する Discriminator の2つで構成。
"""

import torch
from torch import nn


class Generator(nn.Module):
    """グレースケール(1ch) → カラー(3ch) に変換するネットワーク。"""

    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            # エンコーダ: 画像を圧縮
            nn.Conv2d(1, 8, kernel_size=4, stride=2, padding=1),  # 96→48
            nn.ReLU(),

            # デコーダ: 画像を復元
            nn.ConvTranspose2d(8, 3, kernel_size=4, stride=2, padding=1),  # 48→96
            nn.Tanh(),  # 出力を [-1, 1] に
        )

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    """画像が本物かAI生成かを判定するネットワーク。"""

    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            # 入力: グレースケール(1ch) + カラー(3ch) = 4ch
            nn.Conv2d(4, 16, kernel_size=4, stride=2, padding=1),  # 96→48
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.LazyLinear(1),  # スコアを1つ出力
        )

    def forward(self, gray, color):
        x = torch.cat([gray, color], dim=1)
        return self.model(x)


def build_models():
    """Generator と Discriminator を作成して返す。"""
    return Generator(), Discriminator()