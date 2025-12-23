# GAN advanced topics
* Here some advanced topics about the GAN are explained.

# U-Net Architecture
* U-Net is a popular neural network architecture for image-to-image tasks.
* It's widely used as the Generator in conditional GANs.

### Encoder-Decoder Structure

* U-Net has a U-shaped structure with two parts:

**Encoder (downsampling):**
- Shrinks the image step by step: 96→48→24→12→6
- Each step extracts higher-level features
- Captures "what" is in the image (shapes, objects)

**Decoder (upsampling):**
- Expands the image back: 6→12→24→48→96
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

**Standard Discriminator (MLP):**
- Flattens the entire image into a 1D vector
- Outputs a single scalar: "Is this image real or fake?"
- Problem: Looks at the whole image at once, may miss local details

**PatchGAN Discriminator:**
- Uses only convolutional layers (no flattening)
- Outputs a grid (e.g., 4x4) of real/fake scores
- Each cell in the grid judges a local "patch" of the input image

```
Input Image (64x64)          PatchGAN Output (4x4)
┌────────────────────┐       ┌─────┬─────┬─────┬─────┐
│                    │       │0.8  │0.9  │0.7  │0.85 │
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
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # 32->16
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1), # 16->8
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

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


# Conv2d Output Size Calculation
* For convolutional neural networks (used in advanced GANs like DCGAN), the output size formula is:

```
Output Size = floor((Input + 2×Padding - Kernel) / Stride) + 1
```

| Parameter | Description | Common Values |
|-----------|-------------|---------------|
| **Kernel** | Size of the filter window | 3, 4, 5 |
| **Stride** | Step size when sliding the filter | 1, 2 |
| **Padding** | Zero-padding added to input edges | 0, 1, 2 |

**Examples:**
```
Input=28, Kernel=3, Stride=1, Padding=1 → Output = (28+2-3)/1+1 = 28 (same size)
Input=28, Kernel=4, Stride=2, Padding=1 → Output = (28+2-4)/2+1 = 14 (half size)
Input=14, Kernel=4, Stride=2, Padding=1 → Output = (14+2-4)/2+1 = 7  (half size)
```

**Quick Rules:**
- Kernel=3, Stride=1, Padding=1 → **Same size** (commonly used)
- Kernel=4, Stride=2, Padding=1 → **Half size** (for downsampling)
- Kernel=4, Stride=2, Padding=1 (in TransposeConv2d) → **Double size** (for upsampling)


# Matplotlib `imshow` Warning (Clipping)
Sometimes you may see a message like this when you plot generated images:

```
Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
Got range [0.044..1.0000001].
```

### Why it happens
* `matplotlib.pyplot.imshow()` expects RGB float images to be in the range **0 to 1**.
* During training/visualization we often "unnormalize" tensors (example: `x * 0.5 + 0.5`).
* Because of floating-point rounding, you can get tiny values like `1.0000001`.

### Simple fix
* Clamp the tensor right before plotting:

```python
from torchvision.transforms.functional import to_pil_image

ax.imshow(to_pil_image(img.clamp(0, 1)))
```
