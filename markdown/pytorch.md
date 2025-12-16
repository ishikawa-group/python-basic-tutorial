# PyTorch Basics
* PyTorch is a deep learning library developed by Meta (Facebook).
* It provides tensors (like NumPy arrays but with GPU support).
* It also supports automatic differentiation for building neural networks.

## Import

```python
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
```

## Tensor
* A tensor is a multi-dimensional array (like NumPy), used for data and model weights.
* Two basics you will see all the time: `shape` (size) and `dtype` (type).
* Very small example:

```python
import torch

x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])  # shape: (2, 2)
print(x.shape, x.dtype)
print(x[0, 1])   # indexing -> 2.0
print(x + 10)    # add 10 to every element
```

## Device (CPU / GPU)
* PyTorch can run on CPU or GPU. GPU is much faster for training.

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
* Neural networks are defined as classes that inherit from `nn.Module`.

```python
import torch
from torch import nn

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

x = torch.randn(4, 10)  # 4 samples, 10 features each
y = model(x)
print(y.shape)          # -> torch.Size([4, 1])
```

## nn.Sequential
* For simple networks, you can use `nn.Sequential` to stack layers.

```python
model = nn.Sequential(
    nn.Linear(10, 5),
    nn.ReLU(),
    nn.Linear(5, 1),
)
```

# Common Layers

| Layer | Example | Role |
|-------|---------|------|
| `nn.Linear` | `nn.Linear(128, 64)` | Fully connected layer |
| `nn.Conv2d` | `nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1)` | 2D convolution (extract features) |
| `nn.ConvTranspose2d` | `nn.ConvTranspose2d(64, 3, 4, 2, 1)` | Upsampling (reverse of Conv2d) |
| `nn.Flatten` | `nn.Flatten()` | Flatten tensor to 1D |
| `nn.LazyLinear` | `nn.LazyLinear(1)` | Linear layer (auto-detects input size) |
| `nn.BatchNorm2d` | `nn.BatchNorm2d(64)` | Normalize feature maps (stabilize training) |


# Activation Functions

| Function | Example | Output Range | Use Case |
|----------|---------|--------------|----------|
| `nn.ReLU` | `nn.ReLU()` | [0, ∞) | General purpose |
| `nn.LeakyReLU` | `nn.LeakyReLU(0.2)` | (-∞, ∞) | Avoid dead gradients |
| `nn.Tanh` | `nn.Tanh()` | [-1, 1] | Image output |
| `nn.Sigmoid` | `nn.Sigmoid()` | [0, 1] | Binary classification |


# Loss Functions

```python
# For binary classification (real/fake)
criterion = nn.BCEWithLogitsLoss()

# For regression (pixel-wise difference)
l1_loss = nn.L1Loss()

# Usage
loss = criterion(prediction, target)
```


# Optimizer
* An optimizer updates weights to reduce the loss.
* Typical steps:
    1. `zero_grad()`: clear old gradients from the previous step
    2. `backward()`: compute gradients (how loss changes w.r.t. each weight)
    3. `step()`: update weights using those gradients

Very small example (one update step):

```python
import torch
from torch import nn
from torch.optim import Adam

model = nn.Linear(1, 1)
optimizer = Adam(model.parameters(), lr=0.1)
criterion = nn.MSELoss()

x = torch.tensor([[1.0], [2.0], [3.0]])
y = torch.tensor([[2.0], [4.0], [6.0]])  # target: y = 2x

pred = model(x)
loss = criterion(pred, y)

optimizer.zero_grad()
loss.backward()
optimizer.step()

print(float(loss))
```

# DataLoader
* `DataLoader` turns a `Dataset` into an iterator of **mini-batches**.
* Use `batch_size` to control batch size and `shuffle=True` for training.

* Very small example:

```python
import torch
from torch.utils.data import DataLoader, TensorDataset

# 4 samples: x -> y
x = torch.tensor([[0.0], [1.0], [2.0], [3.0]])
y = torch.tensor([0, 0, 1, 1])

dataset = TensorDataset(x, y)
loader = DataLoader(dataset, batch_size=2, shuffle=True)

for batch_x, batch_y in loader:
    print(batch_x, batch_y)
```
* Common options: `batch_size`, `shuffle`, `num_workers`, `drop_last`.


# torch.no_grad()
* Disable gradient computation during inference (saves memory).

```python
model.eval()  # Set model to evaluation mode

with torch.no_grad():
    output = model(input)
```

# torchvision.transforms
* `torchvision.transforms` helps you preprocess images before feeding them to a model.
* `Compose([...])` applies transforms in order.

* Very small example

```python
from torchvision import transforms
from PIL import Image

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
])

img = Image.new("L", (28, 28), color=128)  # dummy grayscale image
tensor = transform(img)
print(tensor.shape, float(tensor.min()), float(tensor.max()))
```

* Common transforms:

| Transform | Example | Description |
|-----------|---------|-------------|
| `Resize` | `transforms.Resize((96, 96))` | Resize image |
| `ToTensor` | `transforms.ToTensor()` | PIL Image → Tensor [0, 1] |
| `Normalize` | `transforms.Normalize((0.5,), (0.5,))` | Normalize values |
| `Grayscale` | `transforms.Grayscale(num_output_channels=1)` | Convert to grayscale |
| `Compose` | `transforms.Compose([...])` | Chain multiple transforms |


# Custom Dataset

* Create a custom `Dataset` when your data is not already in a built-in dataset class.
* You must implement `__len__()` and `__getitem__()`.

* Very small example:

```python
import torch
from torch.utils.data import Dataset

class TinyDataset(Dataset):
    def __init__(self):
        self.x = torch.tensor([[0.0], [1.0], [2.0], [3.0]])
        self.y = torch.tensor([0, 0, 1, 1])

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

dataset = TinyDataset()
print(len(dataset))
print(dataset[0])
```

* The `__len__` and `__getitem__` methods are what `DataLoader` uses internally.
