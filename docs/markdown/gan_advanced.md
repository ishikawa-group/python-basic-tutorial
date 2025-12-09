# GAN advanced topics
* Here some advanced topics about the GAN are explained.
 
# Conditional GAN
* In a standard GAN, the Generator creates images from random noise alone. In a **conditional GAN (cGAN)**, the 
Generator receives additional input that guides what to generate.

| Type | Input | Output | Example |
|------|-------|--------|---------|
| Standard GAN | Random noise | Generated image | Random face generation |
| Conditional GAN | Noise + condition | Guided image | Grayscale → Color |

* Examples of conditional GAN tasks:
  - **Image colorization**: Grayscale image → Color image
  - **Image super-resolution**: Low-res → High-res
  - **Style transfer**: Photo → Painting style
  - **Semantic segmentation**: Label map → Realistic image
 

# U-Net Architecture
* U-Net is a popular neural network architecture for image-to-image tasks. It's widely used as the Generator in conditional GANs.

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
