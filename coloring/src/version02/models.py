"""ダミーのGANモデル。実際の計算は行わず、呼び出しの流れだけを確認する。"""

import torch
from torch import nn


class Generator(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        print("[Dummy Generator] forward called")
        return x  # 何も変えずにそのまま返す


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, gray, color):
        print("[Dummy Discriminator] forward called")
        # ダミーのスコアを返す（形だけ整える）
        batch_size = gray.shape[0]
        return torch.zeros((batch_size, 1), device=gray.device)


def build_models():
    print("[Dummy] build_models called")
    return Generator(), Discriminator()
