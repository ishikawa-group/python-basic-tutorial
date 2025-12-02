# version01 / version02 のPython超入門メモ

現行リポジトリのコードを読みやすくするために、最低限の文法だけをまとめました。Pythonを初めて触る人が、`src/version01` / `src/version02` のソースをざっと追うときの手引きです。

## 1. `print` で文字を表示する
- 文字列を画面に出す最も基本的な命令です: `print("Hello")`。
- 変数を一緒に表示したいときは f 文字列が便利です: `print(f"epoch={epoch}, loss={loss}")`。

## 2. 変数と数値・文字列
- 代入は `=` で行います: `count = 0`、`name = "colorizer"`。
- `+=` を使うと「足して代入」を1行で書けます: `count += 1`。

## 3. インデントがブロックを作る
- 関数や条件分岐の中身は **スペース4つ** で下げます。
- インデントの揃いがズレると文法エラーになります。`if` や `for` の次の行をそろえて書きましょう。

## 4. `import` の書き方
- 標準ライブラリ: `import os`、`import os.path` をよく使います。
- 外部ライブラリ: `import torch` / `from torchvision import transforms`。
- 自作モジュール: `from src.version01.dataset import SimplePairDataset` のようにディレクトリをドットでつなぎます。

## 5. 関数定義と呼び出し
- `def` で関数を定義します。戻り値は `return` で返します。
  ```python
  def add(a, b):
      return a + b
  ```
- 呼び出しは `result = add(1, 2)` のように行い、必要に応じてキーワード引数も使えます。

## 6. 条件分岐 `if`
- `if 条件:` のあとをインデントして処理を書きます。`elif` / `else` で分岐を追加します。
  ```python
  if use_cuda:
      device = "cuda"
  else:
      device = "cpu"
  ```

## 7. ループ `for`
- シーケンスを順に処理します。`enumerate` でカウンタも一緒に受け取れます。
  ```python
  for step, (gray, real) in enumerate(dataloader):
      print(step)
  ```

## 8. クラス定義とメソッド
- `class` でデータと処理をまとめます。`__init__` はコンストラクタ、`self` はインスタンス自身を指します。
```python
class SimpleGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        # レイヤー定義

    def forward(self, x):
        return x
```

## 9. with 文でファイルを扱う
- `with open(path) as f:` の形で書くと、自動的にクローズされます。
  ```python
  with open("log.txt", "w", encoding="utf-8") as f:
      f.write("done\n")
  ```

## 10. スクリプトとして実行するときの定型
- ファイル末尾に次のように書くと、直接実行されたときだけ `main()` が動き、インポート時は動きません。
  ```python
  if __name__ == "__main__":
      main()
  ```

### このメモの使い方
まず `print`・`import`・`if/for`・関数/クラス定義の形をつかんでから、実際のコードを開くと、処理の流れを理解しやすくなります。複雑な文法は避け、最小限の書き方だけを意識してください。
