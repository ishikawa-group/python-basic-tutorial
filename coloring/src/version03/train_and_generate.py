"""バージョン00: アニメ画像の自動カラー化トレーニングスクリプト。

初心者向けに書かれたシンプルな構成です。
グレースケール画像からカラー画像を生成するGANを学習します。
"""

import os
import sys
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
from torchvision.transforms.functional import to_pil_image
from PIL import Image, ImageDraw, ImageFont

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))
from dataset import HFAFPairDataset, basic_transforms
from models import build_models


# =============================================================================
# データ準備
# =============================================================================

def make_dataloaders(data_dir, batch_size, image_size=96, test_ratio=0.2):
    """データセットを読み込み、学習用とテスト用のDataLoaderを作成する。

    Args:
        data_dir: 画像フォルダのパス
        batch_size: 一度に処理する画像の枚数
        image_size: 画像のサイズ（正方形にリサイズ）
        test_ratio: テスト用に使う割合（0.2 = 20%）

    Returns:
        (train_loader, test_loader): 学習用とテスト用のDataLoader
    """
    # 画像の前処理を取得
    color_transform, gray_transform = basic_transforms(image_size)

    # データセットを作成
    dataset = HFAFPairDataset(
        data_dir,
        transform=color_transform,
        grayscale_transform=gray_transform,
        image_size=image_size
    )

    # データセットを学習用とテスト用に分割
    total = len(dataset)
    test_size = max(1, int(total * test_ratio))
    train_size = total - test_size

    train_dataset, test_dataset = random_split(
        dataset,
        [train_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    # DataLoaderを作成
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


# =============================================================================
# デバイス選択
# =============================================================================

def get_device():
    """利用可能な最速のデバイス（GPU/CPU）を返す。"""
    if torch.cuda.is_available():
        return "cuda"  # NVIDIA GPU
    elif torch.backends.mps.is_available():
        return "mps"   # Apple Silicon GPU
    return "cpu"


# =============================================================================
# 学習
# =============================================================================

def train(train_loader, epochs=1, learning_rate=2e-4, device=None):
    """カラー化モデルを学習する。

    Args:
        train_loader: 学習用データ
        epochs: 学習を繰り返す回数
        learning_rate: 学習率（小さいほど慎重に学習）
        device: 使用するデバイス（None なら自動選択）

    Returns:
        (generator, device): 学習済みモデルと使用デバイス
    """
    # デバイスを決定
    if device is None:
        device = get_device()
    print(f"使用デバイス: {device}")

    # モデルを作成してデバイスに転送
    generator, discriminator = build_models()
    generator.to(device)
    discriminator.to(device)

    # 損失関数を定義
    bce_loss = nn.BCEWithLogitsLoss()   # 本物/偽物の判定用
    l1_loss = nn.L1Loss()               # 色の差を測る用

    # 最適化アルゴリズム（Adam）を設定
    optim_g = Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    optim_d = Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))

    # 学習ループ
    for epoch in range(epochs):
        for step, (gray, real) in enumerate(train_loader):
            # データをデバイスに転送
            gray = gray.to(device)
            real = real.to(device)

            # --- ジェネレーター（生成器）の学習 ---
            optim_g.zero_grad()

            # グレースケール画像からカラー画像を生成
            fake = generator(gray)

            # ディスクリミネーターを騙せるか評価
            pred_fake = discriminator(gray, fake)
            target_real = torch.ones_like(pred_fake)  # 「本物」と判定させたい
            loss_gan = bce_loss(pred_fake, target_real)

            # 生成画像と正解画像の差（色の正確さ）
            loss_l1 = l1_loss(fake, real) * 100

            # 合計損失で更新
            loss_g = loss_gan + loss_l1
            loss_g.backward()
            optim_g.step()

            # --- ディスクリミネーター（判別器）の学習 ---
            optim_d.zero_grad()

            # 本物画像を本物と判定
            pred_real = discriminator(gray, real)
            loss_real = bce_loss(pred_real, torch.ones_like(pred_real))

            # 偽物画像を偽物と判定
            pred_fake = discriminator(gray, fake.detach())
            loss_fake = bce_loss(pred_fake, torch.zeros_like(pred_fake))

            # 平均を取って更新
            loss_d = (loss_real + loss_fake) / 2
            loss_d.backward()
            optim_d.step()

            # 進捗を表示（10ステップごと）
            if step % 10 == 0:
                print(
                    f"エポック {epoch+1}/{epochs} "
                    f"ステップ {step}/{len(train_loader)} "
                    f"D損失: {loss_d.item():.4f} G損失: {loss_g.item():.4f}"
                )

        print(f"エポック {epoch+1} 完了")

    return generator, device


# =============================================================================
# 結果の可視化
# =============================================================================

def generate(generator, test_loader, device, output_dir=None):
    """テスト画像をカラー化して、結果を1枚の画像にまとめて保存する。

    3行構成の画像を生成:
    - 1行目: グレースケール入力
    - 2行目: AIによるカラー化結果
    - 3行目: 正解のカラー画像

    Args:
        generator: 学習済みモデル
        test_loader: テスト用データ
        device: 使用デバイス
        output_dir: 保存先フォルダ（None ならカレントディレクトリ）

    Returns:
        保存先のファイルパス（画像がなければ None）
    """
    if output_dir is None:
        output_dir = os.getcwd()

    max_samples = 10  # 表示するサンプル数

    # 結果を格納するリスト
    gray_images = []
    fake_images = []
    real_images = []

    # 推論モードに切り替え
    generator.eval()

    for gray, real in test_loader:
        if len(gray_images) >= max_samples:
            break

        gray = gray.to(device)
        real = real.to(device)

        # 勾配計算を無効化して推論
        with torch.no_grad():
            fake = generator(gray)

        # 必要な枚数だけ取り出す
        n = min(max_samples - len(gray_images), gray.size(0))

        for i in range(n):
            # 正規化を戻す: [-1, 1] → [0, 1]
            # グレースケールは3チャンネルに複製（表示用）
            gray_images.append((gray[i].repeat(3, 1, 1).cpu() * 0.5 + 0.5))
            fake_images.append((fake[i].cpu() * 0.5 + 0.5))
            real_images.append((real[i].cpu() * 0.5 + 0.5))

    # 画像がなければ終了
    if not gray_images:
        return None

    # 1枚の画像にまとめる
    tile_h = gray_images[0].shape[1]
    tile_w = gray_images[0].shape[2]
    num_cols = len(gray_images)

    canvas = Image.new("RGB", (tile_w * num_cols, tile_h * 3))

    # 各行に画像を配置
    for i, img in enumerate(gray_images):
        canvas.paste(to_pil_image(img), (i * tile_w, 0))
    for i, img in enumerate(fake_images):
        canvas.paste(to_pil_image(img), (i * tile_w, tile_h))
    for i, img in enumerate(real_images):
        canvas.paste(to_pil_image(img), (i * tile_w, tile_h * 2))

    # 見やすいサイズに拡大
    scale = 6
    canvas = canvas.resize(
        (canvas.width * scale, canvas.height * scale),
        Image.Resampling.BILINEAR
    )

    # ラベルを追加
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=50)

    labels = ["Grayscale input", "Colorized output", "Original color"]
    for i, text in enumerate(labels):
        y = tile_h * scale * i + 10
        bbox = draw.textbbox((10, y), text, font=font)
        pad = 8
        rect = (bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad)
        draw.rectangle(rect, fill=(0, 0, 0))
        draw.text((10, y), text, fill=(255, 255, 255), font=font)

    # 保存
    output_path = os.path.join(output_dir, "samples_test.png")
    canvas.save(output_path)
    print(f"結果を保存しました: {output_path}")

    return output_path


# =============================================================================
# メイン処理
# =============================================================================

if __name__ == "__main__":
    """メイン関数: データ読み込み → 学習 → 結果保存 の流れを実行する。"""

    # --- 設定 ---
    data_dir = "../../data/HFAF-small"  # 画像フォルダ
    epochs = 4                          # 学習回数
    batch_size = 16                     # バッチサイズ
    learning_rate = 2e-4                # 学習率
    image_size = 64                     # 画像サイズ

    # --- 実行 ---
    # 1. データを準備
    train_loader, test_loader = make_dataloaders(data_dir, batch_size, image_size)

    # 2. モデルを学習
    generator, device = train(train_loader, epochs=epochs, learning_rate=learning_rate)

    # 3. 結果を保存
    generate(generator, test_loader, device)
