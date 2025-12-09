# PyTorch Basics

PyTorch is a deep learning library developed by Meta (Facebook). It provides tensors (like NumPy arrays but with GPU support) and automatic differentiation for building neural networks.

## Import

```python
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
```

## Tensor

Tensors are multi-dimensional arrays, similar to NumPy arrays.

```python
# Create tensors
x = torch.zeros(3, 4)       # 3x4 tensor of zeros
x = torch.ones(3, 4)        # 3x4 tensor of ones
x = torch.randn(3, 4)       # 3x4 tensor of random numbers

# Convert from/to NumPy
import numpy as np
a = np.array([1, 2, 3])
t = torch.from_numpy(a)     # NumPy → Tensor
a2 = t.numpy()              # Tensor → NumPy

# Check min/max
torch.min(x), torch.max(x)
```

## Device (CPU / GPU)

PyTorch can run on CPU or GPU. GPU is much faster for training.

```python
# Check available device
if torch.cuda.is_available():
    device = "cuda"           # NVIDIA GPU
elif torch.backends.mps.is_available():
    device = "mps"            # Apple Silicon GPU
else:
    device = "cpu"

# Move tensor/model to device
x = x.to(device)
model = model.to(device)
```

## nn.Module (defining models)

Neural networks are defined as classes that inherit from `nn.Module`.

```python
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(10, 5)
        self.layer2 = nn.Linear(5, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = torch.relu(x)
        x = self.layer2(x)
        return x

model = SimpleModel()
```

## nn.Sequential

For simple networks, you can use `nn.Sequential` to stack layers.

```python
model = nn.Sequential(
    nn.Linear(10, 5),
    nn.ReLU(),
    nn.Linear(5, 1),
)
```

## Common Layers

| Layer | Example | Role |
|-------|---------|------|
| `nn.Linear` | `nn.Linear(128, 64)` | Fully connected layer |
| `nn.Conv2d` | `nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1)` | 2D convolution (extract features) |
| `nn.ConvTranspose2d` | `nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1)` | Upsampling (reverse of Conv2d) |
| `nn.Flatten` | `nn.Flatten()` | Flatten tensor to 1D |
| `nn.LazyLinear` | `nn.LazyLinear(1)` | Linear layer (auto-detects input size) |
| `nn.BatchNorm2d` | `nn.BatchNorm2d(64)` | Normalize feature maps (stabilize training) |

## Activation Functions

| Function | Example | Output Range | Use Case |
|----------|---------|--------------|----------|
| `nn.ReLU` | `nn.ReLU()` | [0, ∞) | General purpose |
| `nn.LeakyReLU` | `nn.LeakyReLU(0.2)` | (-∞, ∞) | Avoid dead gradients |
| `nn.Tanh` | `nn.Tanh()` | [-1, 1] | Image output |
| `nn.Sigmoid` | `nn.Sigmoid()` | [0, 1] | Binary classification |

## Loss Functions

```python
# For binary classification (real/fake)
criterion = nn.BCEWithLogitsLoss()

# For regression (pixel-wise difference)
l1_loss = nn.L1Loss()

# Usage
loss = criterion(prediction, target)
```

## Optimizer

Optimizers update model weights based on gradients.

```python
optimizer = Adam(model.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Training step
optimizer.zero_grad()   # Clear gradients
loss.backward()         # Compute gradients
optimizer.step()        # Update weights
```

## DataLoader

DataLoader loads data in batches for training.

```python
from torch.utils.data import DataLoader, random_split

# Split dataset into train/test
train_size = int(len(dataset) * 0.8)
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Create DataLoader
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Iterate over batches
for batch_data, batch_labels in train_loader:
    # training code here
    pass
```

## torch.no_grad()

Disable gradient computation during inference (saves memory).

```python
model.eval()  # Set model to evaluation mode

with torch.no_grad():
    output = model(input)
```

## Basic Training Loop

```python
model = SimpleModel().to(device)
optimizer = Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

for epoch in range(num_epochs):
    for data, target in train_loader:
        data = data.to(device)
        target = target.to(device)

        optimizer.zero_grad()       # 1. Clear gradients
        output = model(data)        # 2. Forward pass
        loss = criterion(output, target)  # 3. Compute loss
        loss.backward()             # 4. Backward pass
        optimizer.step()            # 5. Update weights

    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
```

## torchvision.transforms

`torchvision.transforms` provides image preprocessing pipelines.

```python
from torchvision import transforms

# Create a transform pipeline
transform = transforms.Compose([
    transforms.Resize((96, 96)),           # Resize to 96x96
    transforms.ToTensor(),                  # Convert PIL Image to Tensor [0, 1]
    transforms.Normalize((0.5,), (0.5,)),   # Normalize to [-1, 1]
])

# Apply transform to an image
from PIL import Image
img = Image.open("photo.jpg")
tensor = transform(img)
```

Common transforms:

| Transform | Example | Description |
|-----------|---------|-------------|
| `Resize` | `transforms.Resize((96, 96))` | Resize image |
| `ToTensor` | `transforms.ToTensor()` | PIL Image → Tensor [0, 1] |
| `Normalize` | `transforms.Normalize((0.5,), (0.5,))` | Normalize values |
| `Grayscale` | `transforms.Grayscale(num_output_channels=1)` | Convert to grayscale |
| `Compose` | `transforms.Compose([...])` | Chain multiple transforms |

## torchvision.transforms.functional

```python
from torchvision.transforms.functional import to_pil_image

# Convert tensor to PIL Image
img = to_pil_image(tensor)
img.save("output.png")
```

## Custom Dataset

To load your own data, create a class that inherits from `torch.utils.data.Dataset`.

```python
from torch.utils.data import Dataset
from PIL import Image
import os

class MyDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        # Find all image files
        self.images = []
        for filename in os.listdir(root_dir):
            if filename.endswith((".png", ".jpg")):
                self.images.append(os.path.join(root_dir, filename))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image

# Usage
dataset = MyDataset("data/images", transform=transform)
print(len(dataset))     # Number of images
img = dataset[0]        # Get first image
```

The `__len__` and `__getitem__` methods are required for DataLoader to work.
