# Train Script Overview (GAN colorization)

`train.py` is a simple end-to-end sample for colorizing grayscale anime images. It loads data, trains a generator/discriminator pair, and saves visualization outputs in one file.

## Libraries and roles
- `os`, `sys`: path handling; extend `sys.path` for local imports.
- `pathlib`: safer object-based paths.
- `torch` (PyTorch): tensors/autograd.
  - `torch.nn`: layers, losses.
  - `torch.optim.Adam`: optimizer.
  - `torch.utils.data.DataLoader`, `random_split`: batching and splitting.
  - `torch.no_grad()`: inference without gradients.
  - `torch.cuda.is_available`, `torch.backends.mps.is_available`: GPU checks.
- `torchvision.transforms.functional.to_pil_image`: tensor → PIL.
- `PIL.Image`, `PIL.ImageDraw`, `PIL.ImageFont`: create/draw/save images.
- Custom `dataset`: `HFAFPairDataset` (color/gray pairs), `basic_transforms` (resize/normalize).
- Custom `models`: `build_models` (colorization generator + discriminator).

## Main functions
### `make_dataloaders(data_dir, batch_size, image_size=96, test_ratio=0.2, seed=42)`
- Builds transforms, dataset, and train/test splits (default 80/20, seeded).
- Returns train/test `DataLoader`s (train shuffled, test fixed).
- Errors early if fewer than 2 images.

### `get_device()`
- Picks CUDA → MPS → CPU automatically.

### `train(train_loader, epochs=1, learning_rate=2e-4, device=None)`
- Builds models, moves to device.
- Losses: `BCEWithLogitsLoss` (real/fake), `L1Loss` ×100 (color difference).
- Optimizer: Adam with the given learning rate.
- Loop: train generator, then discriminator; log every 10 steps. Returns generator and device.

### `generate(generator, test_loader, device, output_dir=None, max_columns=10)`
- Colorizes up to `max_columns` test samples and saves a 3-row grid (grayscale/output/target) as `v00_samples_test.png`. Returns path or `None` if no data.

### `main()`
- Sets paths (project root → `data/HFAF-small`), collects tunable settings, then runs: load data → train → save results.
- `if __name__ == "__main__":` ensures `main()` runs only when executed directly, not when imported.

## Core Python syntax (quick reminders)
- Imports bring in toolboxes; extend `sys.path` if needed.
- Functions: `def ...` / `return`; multiple returns via tuples.
- Classes: blueprints with `__init__`; `self` is the instance.
- `for`: iterate; `enumerate` gives index + item. `if/elif/else`: branch logic.
- `while`: loop while condition is true (make sure it ends).
- `with`: context manager for safe setup/cleanup (files, `torch.no_grad()`).
- `try/except/finally`: handle errors; `raise ValueError("reason")` to signal.
- f-strings for readable output; `print` supports `sep`/`end`.
- Paths: `pathlib.Path("data") / "file.txt"` or `os.path.join` for strings.
- Scope: resolved inner → outer (LEGB); avoid `global` when possible.

## How to run
1) `make_dataloaders` loads/splits data. 2) `train` learns to colorize/discriminate. 3) `generate` saves a grid. 4) Run `python -m coloring.src.version00.train` to execute all steps.

This note mirrors the train script’s intent in plain English so beginners can follow the workflow.
