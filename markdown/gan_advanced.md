# GAN advanced topics
* Here some advanced topics about the GAN are explained

# Convolution and Conv2d

* Convolution is the fundamental operation in CNNs (Convolutional Neural Networks)
* It extracts features from images by sliding a small filter (kernel) across the input

### What is Convolution?

A convolution slides a small **kernel** (filter) over the input image:
- At each position, it multiplies the kernel values with the overlapping pixels
- The sum of these products becomes one output pixel
- This process detects patterns like edges, textures, and shapes

```
Input (5x5)              Kernel (3x3)           Output (3x3)
┌───┬───┬───┬───┬───┐    ┌───┬───┬───┐         ┌───┬───┬───┐
│ 1 │ 2 │ 3 │ 0 │ 1 │    │ 1 │ 0 │-1 │         │ ? │ ? │ ? │
├───┼───┼───┼───┼───┤    ├───┼───┼───┤    ──▶  ├───┼───┼───┤
│ 0 │ 1 │ 2 │ 1 │ 0 │    │ 1 │ 0 │-1 │         │ ? │ ? │ ? │
├───┼───┼───┼───┼───┤    ├───┼───┼───┤         ├───┼───┼───┤
│ 1 │ 0 │ 1 │ 2 │ 1 │    │ 1 │ 0 │-1 │         │ ? │ ? │ ? │
├───┼───┼───┼───┼───┤    └───┴───┴───┘         └───┴───┴───┘
│ 2 │ 1 │ 0 │ 1 │ 0 │
├───┼───┼───┼───┼───┤    Kernel slides across input,
│ 0 │ 1 │ 2 │ 0 │ 1 │    computing weighted sums
└───┴───┴───┴───┴───┘
```

### Conv2d in PyTorch

`nn.Conv2d` applies 2D convolution to images:

```python
import torch
from torch import nn

# Create a Conv2d layer
conv = nn.Conv2d(
    in_channels=3,    # Input channels (e.g., RGB=3)
    out_channels=16,  # Number of filters (output channels)
    kernel_size=3,    # Size of the filter (3x3)
    stride=1,         # Step size when sliding
    padding=1,        # Zero-padding around input
)

# Example: process an RGB image
x = torch.randn(1, 3, 64, 64)  # [Batch, Channels, Height, Width]
y = conv(x)
print("input:", x.shape)   # torch.Size([1, 3, 64, 64])
print("output:", y.shape)  # torch.Size([1, 16, 64, 64])
```

### Key Parameters

| Parameter | Description | Effect on Output Size |
|-----------|-------------|----------------------|
| **kernel_size** | Size of the sliding filter | Larger kernel → smaller output |
| **stride** | Step size when sliding | Larger stride → smaller output |
| **padding** | Zero-padding around input | More padding → larger output |

### Common Patterns

**Same size** (stride=1, padding matches kernel):
```python
nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)  # 64x64 → 64x64
```

**Half size** (stride=2 for downsampling):
```python
nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1)  # 64x64 → 32x32
```

### Why Convolution Works for Images

1. **Local patterns**: Kernels detect local features (edges, corners, textures)
2. **Parameter sharing**: Same kernel applied everywhere → fewer parameters
3. **Translation invariance**: Can detect a pattern anywhere in the image
4. **Hierarchical features**: Stack layers to detect simple → complex patterns


# Transposed Convolution (Deconvolution)
* Standard convolution often reduces spatial dimensions.
* *Transposed convolution* increases spatial dimensions.
* This is essential for image generation, where we upsample from a small feature map to a full-size image.

### What is it (in plain words)?
* `Conv2d` takes an image/feature-map and usually makes it **smaller** (downsampling).
* `ConvTranspose2d` does the opposite: it makes a feature-map **bigger** (upsampling).
* It is not a true "inverse" of convolution (so "deconvolution" is a confusing nickname).

### Why GAN generators use it
* A generator often starts from a small tensor (like `4x4` or `7x7`).
* Then it repeatedly upsamples until it reaches the final image size.
* `ConvTranspose2d` learns how to upsample in a trainable way (not just copying pixels).

### Primitive example (just shapes)
* This example shows a common GAN setting: `kernel_size=4, stride=2, padding=1` doubles H and W.

```python
import torch
from torch import nn

# Batch=1, Channels=16, Height=8, Width=8
x = torch.randn(1, 16, 8, 8)

up = nn.ConvTranspose2d(
    in_channels=16,
    out_channels=8,
    kernel_size=4,
    stride=2,
    padding=1,
)

y = up(x)
print("input:", x.shape)   # torch.Size([1, 16, 8, 8])
print("output:", y.shape)  # torch.Size([1, 8, 16, 16])
```


# U-Net Architecture
* U-Net is a popular neural network architecture for image-to-image tasks

### Encoder-Decoder Structure

* U-Net has a U-shaped structure with two parts:

**Encoder (downsampling):**
- Shrinks the image step by step: e.g., 96x96 → 48x48 → 24x24 → 12x12 → 6x6
- Each step extracts higher-level features
- Captures "what" is in the image (shapes, objects)

**Decoder (upsampling):**
- Expands the image back: 6x6 → 12x12 → 24x24 → 48x48 → 96x96
- Each step reconstructs spatial details
- Reconstructs "where" things are

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

### Skip Connections

* The horizontal arrows in the diagram are **skip connections**. This is what makes U-Net special.

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

### Why U-Net Works Well for Image-to-Image Tasks

1. **Preserves Structure**: Skip connections keep the original edges and shapes
2. **Learns Transformation**: The bottleneck learns how to transform features
3. **Sharp Output**: Fine details from encoder help decoder create crisp results


# PatchGAN Discriminator
* PatchGAN is a discriminator architecture commonly used in image-to-image GANs like pix2pix.
* Instead of outputting a single "real/fake" score, it outputs a grid of scores.

### Standard Discriminator vs PatchGAN

**Standard Discriminator**
- Flattens the entire image into a 1D vector
- Outputs a single scalar: "Is this image real or fake?"
- Problem: Looks at the whole image at once, may miss local details

**PatchGAN Discriminator**
- Uses only convolutional layers (no flattening)
- Outputs a grid (e.g., 4x4) of real/fake scores
- Each cell in the grid judges a local "patch" of the input image

```
Input Image (64x64)          PatchGAN Output (4x4)
┌────────────────────┐       ┌─────┬─────┬─────┬─────┐
│                    │       │0.80 │0.90 │0.70 │0.85 │
│    Full Image      │  ──▶  ├─────┼─────┼─────┼─────┤
│                    │       │0.75 │0.95 │0.88 │0.82 │
│                    │       ├─────┼─────┼─────┼─────┤
│                    │       │0.92 │0.78 │0.91 │0.86 │
│                    │       ├─────┼─────┼─────┼─────┤
│                    │       │0.83 │0.87 │0.79 │0.90 │
└────────────────────┘       └─────┴─────┴─────┴─────┘
                              Each cell judges a 16x16 patch
```

### Why PatchGAN Works Well

1. **Focuses on Local Texture**: Each patch judges local details like edges and textures
2. **Fewer Parameters**: No huge fully-connected layers needed
3. **Works on Any Image Size**: Convolutional-only architecture is flexible
4. **Better for Style/Texture**: Particularly good at enforcing realistic textures

### Example Architecture

```python
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            # Input: 4 channels (1 gray + 3 color for conditional GAN)
            nn.Conv2d(4, 64, kernel_size=4, stride=2, padding=1),   # 64->32
            nn.ReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # 32->16
            nn.BatchNorm2d(128),
            nn.ReLU(0.2),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1), # 16->8
            nn.BatchNorm2d(256),
            nn.ReLU(0.2),

            nn.Conv2d(256, 1, kernel_size=4, stride=2, padding=1),  # 8->4
            nn.Sigmoid(),
        )

    def forward(self, gray, color):
        x = torch.cat([gray, color], dim=1)  # [B, 4, 64, 64]
        return self.model(x)  # [B, 1, 4, 4] patch grid
```

### Training with PatchGAN

* The loss function (BCELoss) works the same way
* `torch.ones_like(output)` creates a grid of 1s (all patches should be "real")
* `torch.zeros_like(output)` creates a grid of 0s (all patches should be "fake")

```python
# Loss is computed over all patches
loss_real = bce_loss(discriminator(gray, real), torch.ones_like(pred))
loss_fake = bce_loss(discriminator(gray, fake), torch.zeros_like(pred))
```


# Loss Functions: L1 Loss vs BCE Loss

* GANs use different loss functions for different purposes
* Understanding when to use each is key to training GANs effectively

### L1 Loss (Reconstruction Loss)

L1 loss measures pixel-wise difference between generated and target images:

```python
l1_loss = nn.L1Loss()
loss = l1_loss(fake, real)  # How different are the pixels?
```

**Purpose**: Make the output look like the target
**Used for**: Generator (reconstruction quality)

### BCE Loss (Adversarial Loss)

BCE (Binary Cross Entropy) loss measures how well the discriminator is fooled:

```python
bce_loss = nn.BCELoss()
loss = bce_loss(prediction, target)  # Real (1) or Fake (0)?
```

**Purpose**: Make the output look realistic
**Used for**: Both Generator and Discriminator

### Comparison

| Aspect | L1 Loss | BCE Loss |
|--------|---------|----------|
| **What it measures** | Pixel difference | Real vs Fake probability |
| **Formula** | mean(\|fake - real\|) | -[y·log(p) + (1-y)·log(1-p)] |
| **Used by** | Generator only | Generator + Discriminator |
| **Encourages** | Similarity to target | Fooling the discriminator |
| **Without it** | Unrealistic output | Wrong colors/structure |

### Why Use Both?

```python
# Generator uses BOTH losses
loss_bce = bce_loss(discriminator(fake), torch.ones_like(pred))  # Fool D
loss_l1 = l1_loss(fake, real) * 100  # Match target pixels
loss_g = loss_bce + loss_l1  # Combined

# Discriminator uses only BCE
loss_real = bce_loss(discriminator(real), torch.ones_like(pred))  # Real→1
loss_fake = bce_loss(discriminator(fake), torch.zeros_like(pred)) # Fake→0
loss_d = (loss_real + loss_fake) / 2
```

**BCE alone**: Generator might create realistic but wrong images
**L1 alone**: Generator creates correct but blurry images
**Combined**: Correct AND realistic images
