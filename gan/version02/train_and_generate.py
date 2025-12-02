"""Training script for grayscale-to-color GAN (v03).

This version works in Google Colab.
All code is in one file for easy copy-paste.
"""

import os
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
from PIL import Image, ImageDraw, ImageFont


# =============================================================================
# Dataset
# =============================================================================

def basic_transforms(image_size=96):
    """Create transforms for color and grayscale images."""
    color = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    gray = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    return color, gray


class HFAFPairDataset(Dataset):
    """Dataset that returns (grayscale, color) image pairs."""

    def __init__(self, root, transform=None, grayscale_transform=None, image_size=96):
        self.root = os.path.abspath(root)
        if transform is None or grayscale_transform is None:
            color_tf, gray_tf = basic_transforms(image_size=image_size)
            self.transform = transform or color_tf
            self.grayscale_transform = grayscale_transform or gray_tf
        else:
            self.transform = transform
            self.grayscale_transform = grayscale_transform

        # Find all images
        self.images = []
        for dirpath, _, filenames in os.walk(self.root):
            for name in filenames:
                ext = os.path.splitext(name)[1].lower()
                if ext in {".png", ".jpg", ".jpeg"}:
                    self.images.append(os.path.join(dirpath, name))
        self.images.sort()

        if not self.images:
            raise ValueError(f"No images found in {self.root}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        path = self.images[idx]
        image = Image.open(path).convert("RGB")
        color = self.transform(image)
        gray = self.grayscale_transform(image)
        return gray, color


# =============================================================================
# Models
# =============================================================================

class Generator(nn.Module):
    """Convert 1-channel grayscale to 3-channel color."""

    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            # Encoder
            nn.Conv2d(1, 8, kernel_size=4, stride=2, padding=1),  # 96->48
            nn.ReLU(),
            # Decoder
            nn.ConvTranspose2d(8, 3, kernel_size=4, stride=2, padding=1),  # 48->96
            nn.Tanh(),
        )

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    """Judge whether the image pair is real or generated."""

    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=4, stride=2, padding=1),  # 96->48
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.LazyLinear(1),
        )

    def forward(self, gray, color):
        x = torch.cat([gray, color], dim=1)
        return self.model(x)


def build_models():
    """Create generator and discriminator."""
    return Generator(), Discriminator()


# =============================================================================
# Data preparation
# =============================================================================

def make_dataloaders(data_dir, batch_size, image_size=96, test_ratio=0.2):
    """Load dataset and create train/test dataloaders."""
    color_transform, gray_transform = basic_transforms(image_size)

    dataset = HFAFPairDataset(
        data_dir,
        transform=color_transform,
        grayscale_transform=gray_transform,
        image_size=image_size
    )

    # Split into train/test
    total = len(dataset)
    test_size = max(1, int(total * test_ratio))
    train_size = total - test_size

    train_dataset, test_dataset = random_split(
        dataset,
        [train_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


# =============================================================================
# Device selection
# =============================================================================

def get_device():
    """Return the fastest available device."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# =============================================================================
# Training
# =============================================================================

def train(train_loader, epochs=1, learning_rate=2e-4, device=None):
    """Train the colorization GAN."""
    if device is None:
        device = get_device()
    print(f"Device: {device}")

    # Build models
    generator, discriminator = build_models()
    generator.to(device)
    discriminator.to(device)

    # Loss functions
    bce_loss = nn.BCEWithLogitsLoss()
    l1_loss = nn.L1Loss()

    # Optimizers
    optim_g = Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    optim_d = Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))

    # Training loop
    for epoch in range(epochs):
        for step, (gray, real) in enumerate(train_loader):
            gray = gray.to(device)
            real = real.to(device)

            # Train generator
            optim_g.zero_grad()
            fake = generator(gray)
            pred_fake = discriminator(gray, fake)
            loss_gan = bce_loss(pred_fake, torch.ones_like(pred_fake))
            loss_l1 = l1_loss(fake, real) * 100
            loss_g = loss_gan + loss_l1
            loss_g.backward()
            optim_g.step()

            # Train discriminator
            optim_d.zero_grad()
            loss_real = bce_loss(discriminator(gray, real), torch.ones_like(pred_fake))
            loss_fake = bce_loss(discriminator(gray, fake.detach()), torch.zeros_like(pred_fake))
            loss_d = (loss_real + loss_fake) / 2
            loss_d.backward()
            optim_d.step()

            if step % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} Step {step}/{len(train_loader)} "
                      f"D: {loss_d.item():.4f} G: {loss_g.item():.4f}")

        print(f"Epoch {epoch+1} done")

    return generator, device


# =============================================================================
# Visualization
# =============================================================================

def generate(generator, test_loader, device):
    """Colorize test images and save a grid."""
    max_samples = 10
    gray_images, fake_images, real_images = [], [], []

    generator.eval()
    for gray, real in test_loader:
        if len(gray_images) >= max_samples:
            break

        gray = gray.to(device)
        real = real.to(device)

        with torch.no_grad():
            fake = generator(gray)

        n = min(max_samples - len(gray_images), gray.size(0))
        for i in range(n):
            gray_images.append((gray[i].repeat(3, 1, 1).cpu() * 0.5 + 0.5))
            fake_images.append((fake[i].cpu() * 0.5 + 0.5))
            real_images.append((real[i].cpu() * 0.5 + 0.5))

    if not gray_images:
        return None

    # Create canvas
    tile_h = gray_images[0].shape[1]
    tile_w = gray_images[0].shape[2]
    num_cols = len(gray_images)

    canvas = Image.new("RGB", (tile_w * num_cols, tile_h * 3))

    for i, img in enumerate(gray_images):
        canvas.paste(to_pil_image(img), (i * tile_w, 0))
    for i, img in enumerate(fake_images):
        canvas.paste(to_pil_image(img), (i * tile_w, tile_h))
    for i, img in enumerate(real_images):
        canvas.paste(to_pil_image(img), (i * tile_w, tile_h * 2))

    # Scale up
    scale = 6
    canvas = canvas.resize((canvas.width * scale, canvas.height * scale), Image.Resampling.BILINEAR)

    # Add labels with black background
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=50)

    labels = ["Grayscale input", "Colorized output", "Original color"]
    for i, text in enumerate(labels):
        x, y = 10, tile_h * scale * i + 10
        bbox = draw.textbbox((x, y), text, font=font)
        draw.rectangle(bbox, fill=(0, 0, 0))
        draw.text((x, y), text, fill=(255, 255, 255), font=font)

    # Save
    canvas.save("result.png")
    print("Saved to result.png")

    return "result.png"


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    # Settings (change data_dir for your environment)
    data_dir = "../../data/HFAF-small"  # In Colab: "/content/data/HFAF-small"
    epochs = 4
    batch_size = 16
    learning_rate = 2e-4
    image_size = 64

    # Run
    train_loader, test_loader = make_dataloaders(data_dir, batch_size, image_size)
    generator, device = train(train_loader, epochs=epochs, learning_rate=learning_rate)
    generate(generator, test_loader, device)