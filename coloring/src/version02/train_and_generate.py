"""ダミー版: GANカラー化のフローだけ確認するスクリプト。
実際の学習や画像生成は行わず、関数呼び出しをログに出すだけ。
"""

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))

from models import build_models


def make_dataloaders(data_dir, batch_size, image_size=96, test_ratio=0.2):
    print("[Dummy] make_dataloaders called")
    print(f"  data_dir={data_dir}, batch_size={batch_size}, image_size={image_size}, test_ratio={test_ratio}")
    # 本来は:
    # color_transform, gray_transform = basic_transforms(image_size)
    # dataset = HFAFPairDataset(data_dir, transform=color_transform, grayscale_transform=gray_transform, image_size=image_size)
    # train_dataset, test_dataset = random_split(dataset, [...], generator=torch.Generator().manual_seed(42))
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    # return train_loader, test_loader
    return "train_loader_dummy", "test_loader_dummy"


def get_device():
    print("[Dummy] get_device called")
    return "cpu"


def train(train_loader, epochs=1, learning_rate=2e-4, device=None):
    print("[Dummy] train called")
    print(f"  train_loader={train_loader}, epochs={epochs}, learning_rate={learning_rate}, device={device}")
    generator, discriminator = build_models()
    print("[Dummy] generator and discriminator created")
    # 本来は:
    # for epoch in range(epochs):
    #     for gray, real in train_loader:
    #         fake = generator(gray)
    #         pred_fake = discriminator(gray, fake)
    #         ...
    #     save checkpoints, log metrics, etc.
    print("[Dummy] training loop would run here")
    return generator, (device or "cpu")


def generate(generator, test_loader, device, output_dir=None):
    print("[Dummy] generate called")
    print(f"  generator={generator.__class__.__name__}, test_loader={test_loader}, device={device}")
    if output_dir is None:
        output_dir = os.getcwd()
    print(f"  output_dir={output_dir}")
    # 本来は:
    # generator.eval()
    # with torch.no_grad():
    #     for gray, real in test_loader:
    #         fake = generator(gray)
    #         collect and save visualization grid
    print("[Dummy] would save generated image here")
    return os.path.join(output_dir, "dummy_output.png")


def main():
    print("[Dummy] main called")
    data_dir = "../../data/HFAF-small"
    epochs = 1
    batch_size = 2
    learning_rate = 1e-3
    image_size = 64

    train_loader, test_loader = make_dataloaders(data_dir, batch_size, image_size)
    generator, device = train(train_loader, epochs=epochs, learning_rate=learning_rate)
    output_path = generate(generator, test_loader, device)
    print(f"[Dummy] done, output at {output_path}")


if __name__ == "__main__":
    main()
