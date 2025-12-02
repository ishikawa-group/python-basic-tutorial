"""もっとも基本的な彩色データセットと変換。"""

import os

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def basic_transforms(image_size=96):
    """HFAFのアニメ画像を指定サイズにリサイズし、正規化する変換セットを返す。

    Args:
        image_size (int): リサイズ後の一辺の画素数。

    Returns:
        tuple[transforms.Compose, transforms.Compose]: カラー用とグレースケール用の変換。
    """

    color = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    gray = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    return color, gray


class HFAFPairDataset(Dataset):
    """HFAF-dataset配下のanime画像を使うグレースケール/カラーのペアデータセット。

    Args:
        root (str): anime画像を含むディレクトリ。
        transform (Callable | None): カラー画像に適用する変換。
        grayscale_transform (Callable | None): グレースケール画像に適用する変換。
        image_size (int): デフォルト変換に使うリサイズ後の一辺の長さ。
    """

    def __init__(self, root, transform=None, grayscale_transform=None, image_size=96):
        self.root = os.path.abspath(root)
        if transform is None or grayscale_transform is None:
            color_tf, gray_tf = basic_transforms(image_size=image_size)
            self.transform = transform or color_tf
            self.grayscale_transform = grayscale_transform or gray_tf
        else:
            self.transform = transform
            self.grayscale_transform = grayscale_transform

        self.images = []
        for dirpath, _, filenames in os.walk(self.root):
            for name in filenames:
                ext = os.path.splitext(name)[1].lower()
                if ext in {".png", ".jpg", ".jpeg"}:
                    self.images.append(os.path.join(dirpath, name))
        self.images.sort()
        if not self.images:
            raise ValueError(f"No images found in {self.root}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        path = self.images[idx]
        image = Image.open(path).convert("RGB")
        color = self.transform(image)
        gray = self.grayscale_transform(image)
        return gray, color
