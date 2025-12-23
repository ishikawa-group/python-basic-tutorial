"""Training script for grayscale-to-color GAN (v02).

This version works in Google Colab.
"""

import os
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import make_grid


# =============================================================================
# Dataset
# =============================================================================


def basic_transforms(image_size=96):
    """Create transforms for color and grayscale images."""
    color = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    gray = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.Grayscale(num_output_channels=1),
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

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert("RGB")
        image = transforms.functional.to_tensor(image)
        return self.grayscale_transform(image), self.transform(image)


# =============================================================================
# Models
# =============================================================================

class Generator(nn.Module):
    """Convert 1-channel grayscale to 3-channel color."""

    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            # Encoder
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),

            # Decoder
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    """Judge whether the image pair is real or generated."""

    def __init__(self, image_size=64, hidden_dim=256):
        super().__init__()
        input_dim = (1 + 3) * image_size * image_size  # gray (1ch) + color (3ch)
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, gray, color):
        x = torch.cat([gray, color], dim=1)
        return self.model(x)


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
# Training
# =============================================================================

def train(train_loader, image_size, epochs=1, learning_rate=2e-4):
    """Train the colorization GAN."""

    # Build models
    generator = Generator()
    discriminator = Discriminator(image_size=image_size)
    generator.to(device)
    discriminator.to(device)

    # Loss functions
    bce_loss = nn.BCELoss()

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
            loss_g = loss_gan
            loss_g.backward()
            optim_g.step()

            # Train discriminator
            optim_d.zero_grad()
            loss_real = bce_loss(discriminator(gray, real), torch.ones_like(pred_fake))
            loss_fake = bce_loss(discriminator(gray, fake.detach()), torch.zeros_like(pred_fake))
            loss_d = (loss_real + loss_fake) / 2
            loss_d.backward()
            optim_d.step()

        print(f"Epoch {epoch+1}/{epochs} done")

    return generator


# =============================================================================
# Visualization
# =============================================================================

def generate(generator, test_loader):
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

    # Beginner-friendly display using torchvision:
    # Put images into a 3 x N grid (rows: gray / fake / real) and show it once.

    num_cols = len(gray_images)
    grid_images = gray_images + fake_images + real_images  # first row, second row, third row
    grid = make_grid(grid_images, nrow=num_cols, padding=2)

    plt.figure(figsize=(2.2 * num_cols, 6))
    plt.title("Top: grayscale input  |  Middle: colorized  |  Bottom: original")
    plt.imshow(to_pil_image(grid.clamp(0, 1)))
    plt.axis("off")
    plt.tight_layout()
    plt.show()
    return None


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Device: {device}")

    data_dir = "../data/HFAF-small"  # In Colab: "/content/data/HFAF-small"
    epochs = 10
    batch_size = 16
    learning_rate = 2e-4
    image_size = 64

    # Run
    train_loader, test_loader = make_dataloaders(data_dir, batch_size, image_size)
    generator = train(train_loader, image_size=image_size, epochs=epochs, learning_rate=learning_rate)
    generate(generator, test_loader)
