# Generative Adversarial Networks (GAN)

## What is GAN?

GAN (Generative Adversarial Network) is a type of neural network that can **generate new images**. Unlike normal neural networks that classify or predict, GANs create something new!

GAN consists of two neural networks that compete against each other:

1. **Generator** - The "artist" that creates fake images
2. **Discriminator** - The "critic" that judges if an image is real or fake

Think of it like a game:
- The Generator tries to create images good enough to fool the Discriminator
- The Discriminator tries to get better at catching fakes
- As they compete, both improve - and the Generator learns to create realistic images!

<div align=center>
<img src="../fig/gan.png" width=60%>
</div>

## Example: MNIST Handwritten Digits

A classic example of GAN is generating handwritten digit images. The MNIST dataset contains 60,000 images of handwritten digits (0-9).

<div align=center>
<img src="../fig/mnist.png" width=40%>
</div>

**How it works:**
1. The Generator starts with random noise (meaningless numbers)
2. It transforms the noise into a 28x28 pixel image
3. The Discriminator compares it with real MNIST images
4. Based on feedback, the Generator improves
5. After training, the Generator can create realistic digit images from any random input!

## Our Project: Image Colorization

In this tutorial, we use GAN for a different task: **colorizing grayscale images**.

Instead of generating images from random noise, our Generator:
- **Input**: Grayscale (black & white) anime image
- **Output**: Colorized (RGB) anime image

The Discriminator learns to tell the difference between:
- Real color images (from the dataset)
- Fake color images (created by the Generator)

This is called **conditional GAN** because the output depends on an input condition (the grayscale image).

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

## U-Net Architecture (version04)

### What is U-Net?

U-Net is a special type of neural network originally designed for medical image segmentation. It's now widely used for image-to-image tasks like colorization because it preserves fine details very well.

The name "U-Net" comes from its U-shaped structure:

```
Input                                              Output
  │                                                  ▲
  ▼                                                  │
┌─────┐                                          ┌─────┐
│96x96│─────────────────────────────────────────▶│96x96│  Skip Connection
└──┬──┘                                          └──▲──┘
   │                                                │
   ▼                                                │
┌─────┐                                          ┌─────┐
│48x48│─────────────────────────────────────────▶│48x48│  Skip Connection
└──┬──┘                                          └──▲──┘
   │                                                │
   ▼                                                │
┌─────┐                                          ┌─────┐
│24x24│─────────────────────────────────────────▶│24x24│  Skip Connection
└──┬──┘                                          └──▲──┘
   │                                                │
   ▼                                                │
┌─────┐              Bottleneck                  ┌─────┐
│12x12│─────────────────────────────────────────▶│12x12│
└─────┘                                          └─────┘

   ◀─── Encoder (shrink) ───▶  ◀─── Decoder (expand) ───▶
```

### Encoder-Decoder Structure

**Encoder (left side - going down):**
- Shrinks the image step by step: 96→48→24→12→6
- Each step extracts higher-level features
- Uses `Conv2d` with `stride=2` to reduce size by half
- Captures "what" is in the image (shapes, objects)

**Decoder (right side - going up):**
- Expands the image back: 6→12→24→48→96
- Each step reconstructs spatial details
- Uses `ConvTranspose2d` to double the size
- Reconstructs "where" things are

### Skip Connections: The Key Feature

The horizontal arrows in the diagram are **skip connections**. This is what makes U-Net special!

**The Problem without Skip Connections:**
- When the image shrinks to 6x6, fine details (edges, textures) are lost
- The decoder has to "guess" these details when expanding back
- Result: blurry output images

**The Solution with Skip Connections:**
- Copy the encoder's output directly to the decoder at each level
- The decoder receives both:
  - High-level features from below (what to draw)
  - Original details from the encoder (how to draw it)
- Result: sharp output images with preserved details!

### Code Implementation

In version04, the skip connections are implemented using `torch.cat`:

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

### Why U-Net Works Well for Colorization

1. **Preserves Structure**: Skip connections keep the original edges and shapes
2. **Learns Color Mapping**: The bottleneck learns which colors go where
3. **Sharp Output**: Fine details from encoder help decoder create crisp results

### Simple vs U-Net Comparison

| Aspect | version03 (Simple) | version04 (U-Net) |
|--------|-------------------|-------------------|
| Skip connections | No | Yes |
| Detail preservation | Lower | Higher |
| Output sharpness | More blurry | Sharper |
| Model complexity | Simpler | More complex |
| Training time | Faster | Slower |

For beginners, start with version03 to understand the basics, then move to version04 for better results!
