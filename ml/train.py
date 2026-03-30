"""
Train a tiny CNN for digit recognition (gear + lap numbers).
Exports to ONNX for browser inference via onnxruntime-web.

Architecture: Simple 3-conv + 2-fc, ~50KB model.
Input: 32x32 grayscale
Output: 11 classes (0-9, N)

Usage: uv run train.py
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import datasets, transforms
from pathlib import Path
import json

INPUT_SIZE = 32
CLASSES = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "N"]
BATCH_SIZE = 32
EPOCHS = 30
LR = 0.001


class DigitCNN(nn.Module):
    """Tiny CNN: 3 conv layers + 2 FC. ~50KB when exported."""

    def __init__(self, num_classes=11):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),  # 32x32 -> 32x32
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),                  # -> 16x16

            nn.Conv2d(16, 32, 3, padding=1),  # -> 16x16
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),                  # -> 8x8

            nn.Conv2d(32, 64, 3, padding=1),  # -> 8x8
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),          # -> 4x4
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def get_transforms(train=True):
    t = [
        transforms.Grayscale(),
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    ]
    if train:
        t.extend([
            transforms.RandomAffine(degrees=5, translate=(0.1, 0.1), scale=(0.85, 1.15)),
            transforms.ColorJitter(brightness=0.3, contrast=0.3),
        ])
    t.extend([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    return transforms.Compose(t)


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    data_dir = Path("data")

    # Load gear and lap number datasets
    gear_train = gear_val = lap_train = lap_val = None

    # Map class folder names to indices
    # Both gear and lapnum use folders named 0-9, N
    class_to_idx = {c: i for i, c in enumerate(CLASSES)}

    datasets_train = []
    datasets_val = []

    for task in ["gear", "lapnum"]:
        train_dir = data_dir / task / "train"
        val_dir = data_dir / task / "val"

        if train_dir.exists() and any(train_dir.iterdir()):
            # Skip if too few real samples (only dummies)
            real_count = sum(1 for _ in train_dir.rglob("*.png") if _.name != "dummy.png")
            if real_count < 5:
                print(f"  {task} train: skipped ({real_count} real samples)")
                continue
            ds = datasets.ImageFolder(str(train_dir), transform=get_transforms(train=True))
            # Remap class indices to our unified CLASSES
            folder_to_class = {}
            for folder_name, folder_idx in ds.class_to_idx.items():
                if folder_name in class_to_idx:
                    folder_to_class[folder_idx] = class_to_idx[folder_name]
            # Override targets
            ds.targets = [folder_to_class.get(t, t) for t in ds.targets]
            ds.samples = [(p, folder_to_class.get(t, t)) for p, t in ds.samples]
            datasets_train.append(ds)
            print(f"  {task} train: {len(ds)} samples")

        if val_dir.exists() and any(val_dir.iterdir()):
            real_count = sum(1 for _ in val_dir.rglob("*.png") if _.name != "dummy.png")
            if real_count < 2:
                print(f"  {task} val: skipped ({real_count} real samples)")
                continue
            ds = datasets.ImageFolder(str(val_dir), transform=get_transforms(train=False))
            folder_to_class = {}
            for folder_name, folder_idx in ds.class_to_idx.items():
                if folder_name in class_to_idx:
                    folder_to_class[folder_idx] = class_to_idx[folder_name]
            ds.targets = [folder_to_class.get(t, t) for t in ds.targets]
            ds.samples = [(p, folder_to_class.get(t, t)) for p, t in ds.samples]
            datasets_val.append(ds)
            print(f"  {task} val: {len(ds)} samples")

    if not datasets_train:
        print("No training data found! Run extract_training_data.py first.")
        return

    train_ds = ConcatDataset(datasets_train) if len(datasets_train) > 1 else datasets_train[0]
    val_ds = ConcatDataset(datasets_val) if len(datasets_val) > 1 else datasets_val[0]

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"\nTotal: {len(train_ds)} train, {len(val_ds)} val samples")

    # Build model
    model = DigitCNN(num_classes=len(CLASSES)).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model params: {param_count:,} ({param_count * 4 / 1024:.0f} KB float32)")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS)

    # Train
    best_val_acc = 0
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

        scheduler.step()

        # Validate
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        train_acc = train_correct / train_total * 100
        val_acc = val_correct / val_total * 100 if val_total > 0 else 0
        avg_loss = train_loss / train_total

        print(f"  Epoch {epoch+1:2d}/{EPOCHS}: loss={avg_loss:.3f} train={train_acc:.1f}% val={val_acc:.1f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "digit_model_best.pth")

    print(f"\nBest val accuracy: {best_val_acc:.1f}%")

    # Load best model
    model.load_state_dict(torch.load("digit_model_best.pth", weights_only=True))
    model.eval()

    # Export to ONNX
    dummy_input = torch.randn(1, 1, INPUT_SIZE, INPUT_SIZE)
    onnx_path = "digit_model.onnx"
    torch.onnx.export(
        model.cpu(), dummy_input, onnx_path,
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}},
        opset_version=13,
    )
    onnx_size = Path(onnx_path).stat().st_size / 1024
    print(f"Exported ONNX: {onnx_path} ({onnx_size:.0f} KB)")

    # Also save class mapping
    with open("digit_classes.json", "w") as f:
        json.dump(CLASSES, f)

    # Copy to src for browser use
    import shutil
    shutil.copy(onnx_path, "../src/digit_model.onnx")
    shutil.copy("digit_classes.json", "../src/digit_classes.json")
    print(f"Copied model to ../src/")


if __name__ == "__main__":
    main()
