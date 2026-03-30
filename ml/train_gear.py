"""Train gear digit recognition model on real video frame crops.

Input: 32x32 grayscale images normalized to [-1, 1]
Output: 7 classes (1-6 + unknown/0)
Architecture: Small CNN matching the existing digit_model format.

Usage:
    cd ml && uv run python train_gear.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from PIL import Image
from pathlib import Path
import random

INPUT_SIZE = 32
# Classes: 0=unknown, 1-6=gears  (keep compatible with original 0-9+N model)
CLASSES = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "N"]
NUM_CLASSES = len(CLASSES)

class GearDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.transform = transform
        root = Path(root_dir)
        for gear_dir in root.iterdir():
            if not gear_dir.is_dir():
                continue
            gear = gear_dir.name
            if gear not in CLASSES:
                continue
            label = CLASSES.index(gear)
            for img_path in gear_dir.iterdir():
                if img_path.suffix in ('.png', '.jpg'):
                    self.samples.append((str(img_path), label))
        random.shuffle(self.samples)
        print(f"Loaded {len(self.samples)} images from {root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('L').resize((INPUT_SIZE, INPUT_SIZE))
        x = transforms.ToTensor()(img)  # [0, 1]
        x = x * 2 - 1  # [-1, 1]
        if self.transform:
            x = self.transform(x)
        return x, label


class DigitCNN(nn.Module):
    """Same architecture as the existing digit_model for compatibility."""
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
    print(f"Using device: {device}")

    # Data augmentation for training
    train_transform = transforms.Compose([
        transforms.RandomAffine(degrees=3, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.GaussianBlur(3, sigma=(0.1, 0.5)),
    ])

    dataset = GearDataset("data/gear_real", transform=None)

    # 80/20 split
    n_train = int(0.8 * len(dataset))
    n_val = len(dataset) - n_train
    train_set, val_set = random_split(dataset, [n_train, n_val],
                                      generator=torch.Generator().manual_seed(42))

    # Apply augmentation only to training
    class AugDataset(Dataset):
        def __init__(self, subset, transform):
            self.subset = subset
            self.transform = transform
        def __len__(self):
            return len(self.subset)
        def __getitem__(self, idx):
            x, y = self.subset[idx]
            if self.transform:
                x = self.transform(x)
            return x, y

    train_loader = DataLoader(AugDataset(train_set, train_transform),
                              batch_size=32, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=64, shuffle=False)

    model = DigitCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    best_val_acc = 0
    for epoch in range(100):
        # Train
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

        # Validate
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
            print(f"Epoch {epoch+1:3d}: train_loss={train_loss/train_total:.4f} "
                  f"train_acc={train_acc:.3f} val_acc={val_acc:.3f}"
                  f"{' ★' if val_acc > best_val_acc else ''}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "digit_model_best.pth")

    print(f"\nBest val accuracy: {best_val_acc:.3f}")

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
    print(f"Exported to digit_model_v4.onnx")


if __name__ == "__main__":
    train()
