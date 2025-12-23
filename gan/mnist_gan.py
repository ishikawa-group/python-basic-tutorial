import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# For Google Colab inline plotting
# %matplotlib inline

# ============================================
# 1. Define Hyperparameters
# ============================================

latent_dim = 64      # Size of random noise input to Generator
hidden_dim = 256     # Number of neurons in hidden layers
image_dim = 28 * 28  # MNIST image size (784 pixels)
batch_size = 64
epochs = 30
learning_rate = 2e-4

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ============================================
# 2. Define Generator Network
# ============================================


class Generator(nn.Module):
    """
    Generator network for MNIST GAN.
      noise (latent_dim) -> hidden layers -> flattened image (image_dim).
    """
    # Architecture: noise → hidden layers → image
    # Input:  random noise vector (size: latent_dim)
    # Output: fake image (size: image_dim)

    def __init__(self):
        super(Generator, self).__init__()
        # Layer 1: latent_dim → hidden_dim, then ReLU activation
        # Layer 2: hidden_dim → hidden_dim, then ReLU activation
        # Layer 3: hidden_dim → image_dim, then Sigmoid activation
        # Sigmoid outputs values in range [0, 1]
        self.model = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, image_dim),
            nn.Sigmoid(),
        )

    def forward(self, z):
        return self.model(z)

# ============================================
# 3. Define Discriminator Network
# ============================================


class Discriminator(nn.Module):
    """
    Discriminator network for MNIST GAN.
    """

    # Architecture: image → hidden layers → real/fake score
    # Input:  image (size: image_dim)
    # Output: single value (probability of being real)

    def __init__(self):
        super(Discriminator, self).__init__()
        # Layer 1: image_dim → hidden_dim, then ReLU activation
        # Layer 2: hidden_dim → hidden_dim, then ReLU activation
        # Layer 3: hidden_dim → 1 (single output), then Sigmoid activation
        # Sigmoid outputs values in range [0, 1] (probability)
        self.model = nn.Sequential(
            nn.Linear(image_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: input image (real or fake)
        # returns: score indicating real (1) or fake (0)
        return self.model(x)

# ============================================
# 4. Initialize Models and Optimizers
# ============================================


generator = Generator().to(device)
discriminator = Discriminator().to(device)

# Loss function: Binary Cross Entropy (BCE)
# Note: Since the Discriminator uses Sigmoid, we use BCELoss (not BCEWithLogitsLoss).
criterion = nn.BCELoss()

# Optimizers: Adam optimizer for both G and D
optimizer_g = optim.Adam(generator.parameters(), lr=learning_rate)
optimizer_d = optim.Adam(discriminator.parameters(), lr=learning_rate)


# ============================================
# 5. Load MNIST Dataset
# ============================================
# Load 60,000 training images of handwritten digits
# Pixel values in range [0, 1]
# Create batches of size batch_size
transform = transforms.ToTensor()  # Pixel values in [0, 1]
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# ============================================
# 6. Training Loop
# ============================================
for epoch in range(epochs):
    for real_images, _ in dataloader:
        # Note: the dataloader in MNIST returns (images, labels), and the latter is not needed now.

        batch_size_current = real_images.size(0)
        real_images = real_images.view(batch_size_current, -1).to(device)

        # ----- Step A: Train Discriminator -----
        # Goal: D should output 1 for real, 0 for fake

        # A1. Feed real images to D
        d_output_real = discriminator(real_images)
        labels_real = torch.ones(batch_size_current, 1).to(device)
        loss_real = criterion(d_output_real, labels_real)  # Should be 1

        # A2. Generate fake images and feed to D
        noise = torch.randn(batch_size_current, latent_dim).to(device)
        fake_images = generator(noise)
        d_output_fake = discriminator(fake_images.detach())
        labels_fake = torch.zeros(batch_size_current, 1).to(device)
        loss_fake = criterion(d_output_fake, labels_fake)  # Should be 0

        # A3. Update D weights
        loss_d = loss_real + loss_fake
        optimizer_d.zero_grad()
        loss_d.backward()
        optimizer_d.step()

        # ----- Step B: Train Generator -----
        # Goal: G should fool D into outputting 1 for fake images

        # B1. Generate new fake images
        noise = torch.randn(batch_size_current, latent_dim).to(device)
        fake_images = generator(noise)

        # B2. Feed fake images to D
        d_output = discriminator(fake_images)
        labels_real = torch.ones(batch_size_current, 1).to(device)
        loss_g = criterion(d_output, labels_real)  # G wants D to say 1 (real)

        # B3. Update G weights
        optimizer_g.zero_grad()
        loss_g.backward()
        optimizer_g.step()

    print(f"Now doing for epoch {epoch+1}/{epochs}")

# ============================================
# 7. Generate New Images (After Training)
# ============================================
# noise = random_normal(size=latent_dim)
# new_image = generator(noise)
# The generated image should look like a handwritten digit!

# Generate final images
generator.eval()
with torch.no_grad():
    noise = torch.randn(16, latent_dim).to(device)
    generated_images = generator(noise).cpu().view(-1, 28, 28)

    # random noize
    noise = torch.randn(16, latent_dim).to(device)

    # get generated images from generator
    generated_images = generator(noise)

    # move device from GPU to CPU, needed to do plotting
    generated_images = generated_images.cpu()

    # "view" is done to resize the figure to 28x28
    generated_images = generated_images.view(-1, 28, 28)

# Do plot
plt.figure(figsize=(5, 5))
for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.imshow(generated_images[i], cmap="gray")
    plt.axis("off")

plt.tight_layout()
plt.show()
