"""HFAF-datasetのanime画像から軽量サブセットを作るユーティリティ。"""

import os
import random
import shutil


def build_subsets(
    source_root="data/HFAF-dataset/anime",
    small_root="data/HFAF-small",
    small_ratio=0.1,
    seed=0,
):
    """HFAFのアニメ画像から小さめのサブセットを作成する。

    Args:
        source_root (str): 元のHFAFアニメ画像ディレクトリ。
        small_root (str): 50%をコピーする出力ディレクトリ。
        small_ratio (float): smallセットに使う割合。
        seed (int): 乱数シード。再現性のために固定値を渡せる。
    """

    if not os.path.isdir(source_root):
        raise FileNotFoundError(f"Source directory not found: {source_root}")

    image_paths = []
    for dirpath, _, filenames in os.walk(source_root):
        for name in filenames:
            ext = os.path.splitext(name)[1].lower()
            if ext in {".png", ".jpg", ".jpeg"}:
                image_paths.append(os.path.join(dirpath, name))
    image_paths.sort()
    if not image_paths:
        raise ValueError(f"No images found in {source_root}")

    rng = random.Random(seed)
    targets = [
        (small_root, small_ratio),
    ]
    for target_root, ratio in targets:
        if ratio <= 0 or ratio > 1:
            raise ValueError(f"Ratio must be in (0, 1], got {ratio}")
        count = max(1, int(len(image_paths) * ratio))
        selected = rng.sample(image_paths, count)
        for src in selected:
            rel = os.path.relpath(src, source_root)
            dst = os.path.join(target_root, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
        print(f"Copied {len(selected)} images to {target_root}")


def main():
    build_subsets()


if __name__ == "__main__":
    main()
