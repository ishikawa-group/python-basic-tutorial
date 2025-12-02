# GANのしくみとPyTorch基本文法（やさしい解説）

## GANの基本アイデア
- **目的**: ノイズから本物っぽい画像を作る「ジェネレーター」と、それが本物か偽物かを判定する「ディスクリミネーター」を互いに競わせる。
- **ジェネレーター**: ランダムベクトルを入力し、画像を出力するモデル。ディスクリミネーターをだませるように学習する。
- **ディスクリミネーター**: 入力画像が本物か偽物かを2値分類するモデル。ジェネレーターの出力を見抜けるように学習する。
- **学習の流れ（繰り返し）**:
  1. 本物画像とジェネレーターの偽物画像をディスクリミネーターに渡して、識別が上手くなるようにパラメータを更新。
  2. ジェネレーターは、ディスクリミネーターをだませるようにパラメータを更新。
- **ポイント**: 2つのモデルが交互に強くなることで、ジェネレーターが徐々に本物に近い画像を作れるようになる。

## PyTorchでよく使う基本レイヤー
シンプルな文法でレイヤーを定義して、`forward`で組み立てる。

- **Conv2d (畳み込み)**
  - 例: `nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1)`
  - 役割: 画像の特徴を抜き出す。入力チャンネル数、出力チャンネル数、カーネルの大きさなどを指定。

- **Linear (全結合)**
  - 例: `nn.Linear(128, 1)`
  - 役割: ベクトルを別の次元に写す。最後の判定や潜在ベクトルの変換に使う。

- **ReLU (活性化)**
  - 例: `nn.ReLU()`
  - 役割: 0未満を0にし、0以上はそのまま返す。非線形性を加えるシンプルな関数。

- **LeakyReLU**
  - 例: `nn.LeakyReLU(0.2)`
  - 役割: 0未満のときに小さい傾きを残す。勾配が0になりにくくする。

- **Tanh**
  - 例: `nn.Tanh()`
  - 役割: 出力を-1〜1に収める。画像生成の最後に使うと、画素値の範囲を整えやすい。

- **BatchNorm2d**
  - 例: `nn.BatchNorm2d(64)`
  - 役割: 特徴マップを正規化し、学習を安定させる。

## 簡単なネットワーク例
- **ジェネレーターの例（概略）**
  ```python
  import torch
  from torch import nn

  class SimpleGenerator(nn.Module):
      def __init__(self):
          super(SimpleGenerator, self).__init__()
          self.net = nn.Sequential(
              nn.Linear(100, 64 * 8 * 8),
              nn.ReLU(),
              nn.Unflatten(1, (64, 8, 8)),
              nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
              nn.ReLU(),
              nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
              nn.Tanh(),
          )

      def forward(self, x):
          return self.net(x)
  ```
  - ノイズベクトルを線形層で大きくし、`ConvTranspose2d`で画像サイズを広げ、最後は`tanh`で-1〜1に整える。

- **ディスクリミネーターの例（概略）**
  ```python
  import torch
  from torch import nn

  class SimpleDiscriminator(nn.Module):
      def __init__(self):
          super(SimpleDiscriminator, self).__init__()
          self.net = nn.Sequential(
              nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
              nn.LeakyReLU(0.2),
              nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
              nn.LeakyReLU(0.2),
              nn.Flatten(),
              nn.Linear(64 * 16 * 16, 1),
          )

      def forward(self, x):
          return self.net(x)
  ```
  - 画像を畳み込みで小さくしながら特徴を集め、最後に1次元にして本物度を出力する。

## 学習ループの基本ステップ
1. **本物画像**と**ノイズから作った偽物画像**を用意する。
2. **ディスクリミネーター更新**: 本物を1、偽物を0とした損失で学習する。
3. **ジェネレーター更新**: ディスクリミネーターが偽物を1と間違えるように損失を作り、学習する。
4. 上記をミニバッチごとに繰り返す。

シンプルな構成でも、上のレイヤーと手順を組み合わせるだけでGANの基本を体験できる。
