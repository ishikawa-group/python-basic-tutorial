"""Dummy script: confirms the flow of function calls without real training.

Run this to understand the GAN training workflow:
1. make_dataloaders() - Load and prepare data
2. train() - Train the model
3. generate() - Create colorized images
"""


def make_dataloaders(data_dir, batch_size, image_size=64):
    """Load dataset and create train/test dataloaders."""
    print(f"make_dataloaders(data_dir={data_dir}, batch_size={batch_size}, image_size={image_size})")
    print("  -> Would load images and split into train/test")
    return "train_loader", "test_loader"


def get_device():
    """Return the fastest available device."""
    print("get_device()")
    print("  -> Would check for cuda/mps/cpu")
    return "cpu"


def train(train_loader, epochs=1, learning_rate=2e-4):
    """Train the colorization GAN."""
    print(f"train(train_loader, epochs={epochs}, learning_rate={learning_rate})")
    print("  -> Would build Generator and Discriminator")
    print("  -> Would run training loop:")
    for epoch in range(epochs):
        print(f"     Epoch {epoch + 1}/{epochs}")
        print("       - Generate fake images from grayscale")
        print("       - Train Discriminator (real vs fake)")
        print("       - Train Generator (fool Discriminator)")
    print("  -> Training complete")
    return "generator", get_device()


def generate(generator, test_loader, device):
    """Colorize test images and save result."""
    print(f"generate(generator, test_loader, device={device})")
    print("  -> Would colorize test images")
    print("  -> Would save result.png")
    return "result.png"


if __name__ == "__main__":
    print("=" * 50)
    print("GAN Training Flow (Dummy Version)")
    print("=" * 50)
    print()

    # Settings
    data_dir = "./data/HFAF-small"
    epochs = 2
    batch_size = 16
    image_size = 64

    # Run
    print("Step 1: Load Data")
    print("-" * 30)
    train_loader, test_loader = make_dataloaders(data_dir, batch_size, image_size)
    print()

    print("Step 2: Train Model")
    print("-" * 30)
    generator, device = train(train_loader, epochs=epochs)
    print()

    print("Step 3: Generate Results")
    print("-" * 30)
    output_path = generate(generator, test_loader, device)
    print()

    print("=" * 50)
    print(f"Done! Output would be saved to: {output_path}")
    print("=" * 50)
