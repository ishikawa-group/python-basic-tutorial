# 条件付きGANによるアニメ彩色

このプロジェクトは、グレースケールのアニメ画像をカラー化するGANを学習するためのPyTorchセットアップを提供します。コードベースは進化段階ごと
に`src/version00pre`（ダミーで流れを確認するプレ版）、`src/version00`（最小構成の入門版）、`src/version01`（シンプルなプロトタイプ）、`src/version02`（機能を拡張した現行版）に分けています。

## バージョン構成の概要
- **version00pre**: まずは関数・クラス・ファイル間の呼び出しをprintだけで確認できるプレ版。ダミーのデータセットとモデルで、学習処理は実行されません。
- **version00**: 64×64リサイズと畳み込み2層の`TinyGenerator`/`TinyDiscriminator`で動かす一番やさしい版。CPUでも試しやすい最小構成です。
- **version01**: 128×128のリサイズと軽量な`SimpleGenerator`/`SimpleDiscriminator`で動作する途中経過版。最小限のデータセットクラスと学習ループだけに絞り、ロスもGAN+L1（重み50）に限定して早い検証を狙います。
- **version02**: 256×256リサイズのU-Net風`Generator`とPatchGAN `Discriminator`を備えた拡張版。CIFAR-10を自動ダウンロードして学習データにできるオプションや、各エポックのチェックポイント保存を含むフル機能パイプラインです。

開発や学習の流れとしては、まずversion00preで呼び出し順や定数の配置を確認し、次にversion00で依存関係やデータの読み込みを試し、version01で軽量な検証を行い、その後version02に切り替えて本番の解像度や機能を試す、という段階的な使い分けを想定しています。

## セットアップ
1. 仮想環境を作成し、依存関係をインストールします。
   ```bash
   pip install -r requirements.txt
   ```

2. RGBのアニメ画像（フレームやイラストなど）を含むデータセットディレクトリを用意してください。`version01`は128×128へ、`version02`は256×256へリ
サイズします。`version02`のみ`--use-cifar`フラグでCIFAR-10（カラー画像）を自動ダウンロードして学習に利用できます。

## 学習
### 使い方の流れ
1. 各バージョンの`train.py`冒頭付近にある定数（`data_dir`や`output_dir`など）を自分の環境に合わせて書き換えます。
2. そのまま実行します。
   - バージョン00pre: `python -m src.version00pre.train`
   - バージョン00: `python -m src.version00.train`
   - バージョン01: `python -m src.version01.train`
   - バージョン02: `python -m src.version02.train`

### バージョン別の特徴
- バージョン00pre: データセット・モデル・学習ループがprintだけで動くダミー構成。ディレクトリ文字列を入れて呼び出し順を確認するためのチュートリアルステップです。
- バージョン00: 64×64へリサイズし、畳み込み2層のジェネレーター/ディスクリミネーターで動作。L1ロスの重みは20で、形の崩れを抑えつつ学習の負荷を最小に。
- バージョン01: 128×128へリサイズする簡易パイプライン。`SimpleGenerator`/`SimpleDiscriminator`による軽量な構成で、L1ロスの重みは50。
- バージョン02: 256×256リサイズのU-Net風`Generator`とPatchGAN `Discriminator`を備えた拡張版。CIFAR-10を自動ダウンロードしたい場合は、`train.py`内の`use_cifar`を`True`に書き換えてから実行します。

ポイント（version02）:
- `Generator`: 1チャネルのグレースケール入力を3チャネルのカラー出力へ変換するU-Net風アーキテクチャ。
- `Discriminator`: グレースケール入力と、実画像または生成画像を条件に判定するPatchGANクリティック。
- ロス: adversarial（BCEWithLogits）+ L1再構成ロスを100倍で加重。
- チェックポイントは各エポック後に出力ディレクトリへ保存されます。

## 推論（概要）
学習後、ジェネレーターを読み込んでグレースケール画像に適用します。
```python
import torch
from PIL import Image
from torchvision import transforms
from src.version02.models import Generator

device = "cuda" if torch.cuda.is_available() else "cpu"
model = Generator().to(device)
model.load_state_dict(torch.load("./checkpoints/colorizer_epoch_5.pth", map_location=device)["generator"])
model.eval()

preprocess = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

img = Image.open("./samples/gray.png").convert("L")
with torch.no_grad():
    gray_tensor = preprocess(img).unsqueeze(0).to(device)
    colorized = model(gray_tensor)
```

## メモ
- GPUがあると現実的な速度で学習できます。
- データセットローダーはRGB画像を想定し、グレースケールはその場で生成します。
- より高解像度で扱う場合は、変換処理やモデルの深さを調整してください。

## オフラインでのサンプル出力
環境の都合でPyTorchやPillowをインストールできなくても、プロジェクトが端から端まで動作し、具体的なサンプル画像を生成できることを確認でき
ます。`src.sample_run`スクリプトは、グレースケールのグラデーションと擬似カラー化した結果を左右に並べたPPMファイルを書き出します。

```bash
python -m src.sample_run
```

スクリプト先頭の`output`や`png_output`の定数を書き換えれば、保存先やPNGの有無を変えられます。PPM出力は外部ライブラリなしで動作し、任意
のPNG出力は表示や共有をしやすくするためにPillowを利用します。これにより、機械学習の依存関係が使えない環境でも実行テストが可能です。
