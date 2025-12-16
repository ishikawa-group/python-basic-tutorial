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
