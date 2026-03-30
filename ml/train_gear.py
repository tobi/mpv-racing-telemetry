"""Train gear digit recognition model on tobil/racing-gears dataset.

Input: 32x32 grayscale images normalized to [-1, 1]
Output: 10 classes (digits 0-9)
Architecture: Small CNN exported to ONNX for use in mpv via LuaJIT FFI.

Usage:
    cd ml && uv run python train_gear.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from datasets import load_dataset
from PIL import Image

INPUT_SIZE = 32
NUM_CLASSES = 10  # digits 0-9


class HFDataset(Dataset):
    """Wraps a HuggingFace dataset split for PyTorch."""
    def __init__(self, hf_split, transform=None):
        self.data = hf_split
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ex = self.data[idx]
        img = ex['image'].convert('L').resize((INPUT_SIZE, INPUT_SIZE))
        x = transforms.ToTensor()(img)  # [0, 1]
        x = x * 2 - 1  # [-1, 1]
        if self.transform:
            x = self.transform(x)
        return x, ex['label']


class DigitCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, NUM_CLASSES),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def train():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    print("Loading tobil/racing-gears...")
    ds = load_dataset("tobil/racing-gears")

    train_transform = transforms.Compose([
        transforms.RandomAffine(degrees=3, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.GaussianBlur(3, sigma=(0.1, 0.5)),
    ])

    train_set = HFDataset(ds['train'], transform=train_transform)
    val_set = HFDataset(ds['validation'])

    train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=64, shuffle=False, num_workers=0)

    print(f"Train: {len(train_set)}, Val: {len(val_set)}")

    model = DigitCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    best_val_acc = 0
    for epoch in range(80):
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)
            train_correct += (out.argmax(1) == y).sum().item()
            train_total += x.size(0)

        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                val_correct += (out.argmax(1) == y).sum().item()
                val_total += x.size(0)

        train_acc = train_correct / train_total
        val_acc = val_correct / val_total if val_total > 0 else 0
        scheduler.step(1 - val_acc)

        if (epoch + 1) % 10 == 0 or val_acc > best_val_acc:
            print(f"  Epoch {epoch+1:3d}: loss={train_loss/train_total:.4f} "
                  f"train={train_acc:.3f} val={val_acc:.3f}"
                  f"{'  ★' if val_acc > best_val_acc else ''}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "digit_model_best.pth")

    print(f"\nBest val: {best_val_acc:.3f}")

    # Export to ONNX
    model.load_state_dict(torch.load("digit_model_best.pth", weights_only=True))
    model.eval().cpu()
    dummy = torch.randn(1, 1, INPUT_SIZE, INPUT_SIZE)
    torch.onnx.export(
        model, dummy, "digit_model_v4.onnx",
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
    )
    print("Exported: digit_model_v4.onnx")


if __name__ == "__main__":
    train()
