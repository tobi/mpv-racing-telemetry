# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "torch>=2.0",
#     "torchvision>=0.15",
#     "datasets>=3.0",
#     "pillow>=10,<12",
#     "trackio>=0.20",
#     "onnxscript>=0.1",
# ]
# ///
"""Train gear digit recognition model on tobil/racing-gears dataset.

Input: 32x32 grayscale images normalized to [-1, 1]
Output: 10 classes (digits 0-9)
Architecture: Small CNN exported to ONNX for use in mpv via LuaJIT FFI.

Usage:
    uv run ml/train.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from datasets import load_dataset

try:
    import trackio
    HAS_TRACKIO = True
except Exception:
    HAS_TRACKIO = False

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


def compute_f1_macro(preds, targets, num_classes=NUM_CLASSES):
    """Compute macro F1 score using tensor ops."""
    p = torch.tensor(preds)
    t = torch.tensor(targets)
    f1s = []
    for c in range(num_classes):
        tp = ((p == c) & (t == c)).sum().item()
        fp = ((p == c) & (t != c)).sum().item()
        fn = ((p != c) & (t == c)).sum().item()
        if tp + fn == 0:
            continue
        prec = tp / (tp + fp) if tp + fp > 0 else 0
        rec = tp / (tp + fn) if tp + fn > 0 else 0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec > 0 else 0)
    return sum(f1s) / len(f1s) if f1s else 0


def train():
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Device: {device}")

    global HAS_TRACKIO
    if HAS_TRACKIO:
        try:
            run = trackio.init(
                project="racing-gears",
                config={"device": device, "epochs": 80, "lr": 1e-3, "batch_size": 64},
            )
            print(f"Trackio run: {run.name}")
            print('View: trackio show --project "racing-gears"')
        except Exception as e:
            print(f"trackio init failed ({e}), continuing without tracking")
            HAS_TRACKIO = False

    print("Loading tobil/racing-gears...")
    ds = load_dataset("tobil/racing-gears")

    train_transform = transforms.Compose([
        transforms.RandomAffine(degrees=3, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.GaussianBlur(3, sigma=(0.1, 0.5)),
    ])

    train_set = HFDataset(ds['train'], transform=train_transform)
    val_set = HFDataset(ds['validation'])

    workers = 4 if device != "mps" else 0
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=workers)
    val_loader = DataLoader(val_set, batch_size=64, shuffle=False, num_workers=workers)

    print(f"Train: {len(train_set)}, Val: {len(val_set)}")

    model = DigitCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    best_val_acc = 0
    best_val_f1 = 0
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
        val_preds, val_targets = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                preds = out.argmax(1)
                val_correct += (preds == y).sum().item()
                val_total += x.size(0)
                val_preds.extend(preds.cpu().tolist())
                val_targets.extend(y.cpu().tolist())

        train_acc = train_correct / train_total
        val_acc = val_correct / val_total if val_total > 0 else 0
        val_f1 = compute_f1_macro(val_preds, val_targets)
        avg_loss = train_loss / train_total
        scheduler.step(1 - val_f1)

        if HAS_TRACKIO:
            trackio.log({
                "train/loss": avg_loss,
                "train/acc": train_acc,
                "val/acc": val_acc,
                "val/f1_macro": val_f1,
                "lr": optimizer.param_groups[0]["lr"],
            }, step=epoch + 1)

        is_best = val_f1 > best_val_f1
        if (epoch + 1) % 10 == 0 or is_best:
            print(f"  Epoch {epoch+1:3d}: loss={avg_loss:.4f} "
                  f"train={train_acc:.3f} val_acc={val_acc:.3f} "
                  f"val_f1={val_f1:.3f}{'  ★' if is_best else ''}")

        if is_best:
            best_val_acc = val_acc
            best_val_f1 = val_f1
            torch.save(model.state_dict(), "digit_model_best.pth")

    print(f"\nBest val: acc={best_val_acc:.3f} f1={best_val_f1:.3f}")

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
    if HAS_TRACKIO:
        trackio.log({"best_val_acc": best_val_acc, "best_val_f1": best_val_f1})
        trackio.finish()


if __name__ == "__main__":
    train()
