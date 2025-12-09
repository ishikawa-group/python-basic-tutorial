# GAN for Image Colorization - Code Guide

* This guide explains how to use our GAN code for colorizing grayscale images.

* For basic GAN concepts, see [gan_basic.md](gan_basic.md).

## Our Project: Image Colorization

In this project, we use a conditional GAN to **colorize grayscale images**.

Our Generator:
- **Input**: Grayscale (black & white) anime image
- **Output**: Colorized (RGB) anime image

The Discriminator learns to tell the difference between:
- Real color images (from the dataset)
- Fake color images (created by the Generator)

## Training Flow

Our colorization code follows 3 simple steps:

```
Step 1: Load Data        → make_dataloaders()
Step 2: Train Model      → train()
Step 3: Generate Results → generate()
```

### Step 1: Load Data (`make_dataloaders`)

```python
train_loader, test_loader = make_dataloaders(data_dir, batch_size, image_size)
```

What happens:
- Load all images from the data folder
- For each image, create a pair: (grayscale version, color version)
- Split into training set (80%) and test set (20%)
- Package into DataLoader for batch processing

### Step 2: Train Model (`train`)

```python
generator, device = train(train_loader, epochs=4, learning_rate=2e-4)
```

For each epoch (one pass through all training data):

```
For each batch of images:
    1. Generator colorizes grayscale images → fake color images
    2. Discriminator judges: real color vs fake color
    3. Calculate losses:
       - How well did Discriminator detect fakes?
       - How well did Generator fool Discriminator?
       - How close are fake colors to real colors?
    4. Update weights to improve both networks
```

The training loop alternates between:
- **Training Discriminator**: Get better at detecting fakes
- **Training Generator**: Get better at creating realistic colors

### Step 3: Generate Results (`generate`)

```python
generate(generator, test_loader, device)
```

What happens:
- Take test images (not used during training)
- Generator colorizes each grayscale image
- Save a comparison image showing:
  - Row 1: Grayscale input
  - Row 2: Generator's colorized output
  - Row 3: Original color (ground truth)

## Code Structure

### Main Functions

| Function | Purpose |
|----------|---------|
| `make_dataloaders()` | Load images and create train/test data |
| `get_device()` | Select best device (GPU or CPU) |
| `train()` | Train Generator and Discriminator |
| `generate()` | Colorize test images and save results |

### Model Classes

```python
class Generator(nn.Module):
    # Converts grayscale (1 channel) → color (3 channels)
    # Uses encoder-decoder structure

class Discriminator(nn.Module):
    # Takes grayscale + color, outputs real/fake score
    # Uses convolutional layers
```

### Loss Functions

| Loss | Purpose |
|------|---------|
| `BCEWithLogitsLoss` | Binary classification (real vs fake) |
| `L1Loss` | Pixel-wise difference (color accuracy) |

The Generator's total loss combines both:
- **Adversarial loss**: Fool the Discriminator
- **L1 loss × 100**: Match the actual colors (weighted heavily)

## Train-Test Split in PyTorch

```python
from torch.utils.data import random_split

# Split dataset: 80% train, 20% test
total = len(dataset)
test_size = int(total * 0.2)
train_size = total - test_size

train_dataset, test_dataset = random_split(
    dataset,
    [train_size, test_size],
    generator=torch.Generator().manual_seed(42)  # For reproducibility
)
```

* `random_split` randomly divides the dataset into specified sizes
* Setting a `seed` (e.g., 42) ensures the same split every time you run the code

In our colorization code, `make_dataloaders()` handles this automatically:
* `train_loader`: Contains 80% of images for training
* `test_loader`: Contains 20% of images for final evaluation

## How to Run

```bash
# Run the training script
python train_and_generate.py
```

After training completes, you'll find `result.png` with the colorization results!

## Version Differences

| Version | Description |
|---------|-------------|
| version02 | Dummy code - shows function call flow only |
| version03 | Simple GAN - minimal encoder-decoder |
| version04 | U-Net GAN - skip connections for better quality |


# U-Net Generator Implementation

* The Generator uses U-Net architecture with skip connections implemented using `torch.cat`:

```python
class UNetGenerator(nn.Module):
    def __init__(self):
        # Encoder layers
        self.enc1 = ...  # 96 → 48
        self.enc2 = ...  # 48 → 24
        self.enc3 = ...  # 24 → 12
        self.enc4 = ...  # 12 → 6

        # Decoder layers
        self.dec1 = ...  # 6 → 12
        self.dec2 = ...  # 12 → 24
        self.dec3 = ...  # 24 → 48
        self.dec4 = ...  # 48 → 96

    def forward(self, gray):
        # Encoder - save outputs for skip connections
        e1 = self.enc1(gray)  # 96 → 48
        e2 = self.enc2(e1)    # 48 → 24
        e3 = self.enc3(e2)    # 24 → 12
        e4 = self.enc4(e3)    # 12 → 6

        # Decoder - concatenate with encoder outputs
        d1 = self.dec1(e4)                    # 6 → 12
        d2 = self.dec2(torch.cat([d1, e3]))   # 12 → 24, concat with e3
        d3 = self.dec3(torch.cat([d2, e2]))   # 24 → 48, concat with e2
        d4 = self.dec4(torch.cat([d3, e1]))   # 48 → 96, concat with e1

        return d4
```

### Simple vs U-Net Comparison

| Aspect | version03 (Simple) | version04 (U-Net) |
|--------|-------------------|-------------------|
| Skip connections | No | Yes |
| Detail preservation | Lower | Higher |
| Output sharpness | More blurry | Sharper |
| Model complexity | Simpler | More complex |
| Training time | Faster | Slower |

* For beginners, start with version03 to understand the basics, then move to version04 for better results.