# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "torch>=2.0",
#     "torchvision>=0.15",
#     "datasets>=3.0",
#     "pillow>=10,<12",
#     "trackio>=0.20",
#     "onnx>=1.14",
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

import time
import ctypes
import os
from pathlib import Path

# Preload CUDA driver if available (needed in some Nix environments)
for _cuda_path in ['/usr/lib/libcuda.so.1', '/usr/lib/x86_64-linux-gnu/libcuda.so.1']:
    if os.path.exists(_cuda_path):
        try:
            ctypes.CDLL(_cuda_path)
        except OSError:
            pass
        break

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from datasets import load_dataset
from PIL import Image

try:
    import trackio
    HAS_TRACKIO = True
except Exception:
    HAS_TRACKIO = False

INPUT_SIZE = 32
NUM_CLASSES = 10  # digits 0-9
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
LOCAL_DATA_DIR = SCRIPT_DIR / "local_data"
LOCAL_REPEAT = 20  # upweight fresh hard examples without changing HF dataset


def image_to_tensor(img):
    to_tensor = transforms.ToTensor()
    img = img.convert('L').resize((INPUT_SIZE, INPUT_SIZE))
    return to_tensor(img) * 2 - 1


def precompute_tensors(hf_split, exclude_source=None):
    xs, ys = [], []
    for ex in hf_split:
        if exclude_source and ex.get('source') == exclude_source:
            continue
        xs.append(image_to_tensor(ex['image']))
        ys.append(ex['label'])
    return torch.stack(xs), torch.tensor(ys, dtype=torch.long)


def precompute_local_tensors(root=LOCAL_DATA_DIR, repeat=LOCAL_REPEAT):
    xs, ys = [], []
    if not root.exists():
        return None, None

    for label_dir in sorted(root.iterdir()):
        if not label_dir.is_dir() or not label_dir.name.isdigit():
            continue
        label = int(label_dir.name)
        if not 0 <= label < NUM_CLASSES:
            continue
        for path in sorted(label_dir.iterdir()):
            if path.suffix.lower() not in {'.png', '.jpg', '.jpeg', '.webp'}:
                continue
            with Image.open(path) as img:
                x = image_to_tensor(img)
            for _ in range(repeat):
                xs.append(x.clone())
                ys.append(label)

    if not xs:
        return None, None
    return torch.stack(xs), torch.tensor(ys, dtype=torch.long)


class DigitCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, NUM_CLASSES),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def compute_f1_macro(preds, targets, num_classes=NUM_CLASSES):
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


def evaluate(model, val_x, val_y, device):
    model.eval()
    with torch.no_grad():
        out = model(val_x.to(device))
        preds = out.argmax(1).cpu()
    acc = (preds == val_y).float().mean().item()
    f1 = compute_f1_macro(preds.tolist(), val_y.tolist())
    return acc, f1


def train():
    torch.set_num_threads(16)

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    print(f"Device: {device}")

    global HAS_TRACKIO
    if HAS_TRACKIO:
        try:
            run = trackio.init(project="racing-gears", config={"device": device})
            print(f"Trackio run: {run.name}")
        except Exception as e:
            print(f"trackio init failed ({e}), continuing without tracking")
            HAS_TRACKIO = False

    print("Loading tobil/racing-gears...")
    ds = load_dataset("tobil/racing-gears")

    print("Precomputing tensors...")
    t0 = time.time()
    # Train on all data, validate only on racing images
    train_x, train_y = precompute_tensors(ds['train'])
    local_x, local_y = precompute_local_tensors()
    if local_x is not None:
        train_x = torch.cat([train_x, local_x], dim=0)
        train_y = torch.cat([train_y, local_y], dim=0)
        counts = torch.bincount(local_y, minlength=NUM_CLASSES).tolist()
        print(f"Added local hard examples: {len(local_y)} augmented samples; labels={counts}")
    val_x, val_y = precompute_tensors(ds['validation'], exclude_source='mnist')
    print(f"Precomputed in {time.time()-t0:.1f}s. Train: {len(train_y)}, Val (racing): {len(val_y)}")

    train_transform = transforms.Compose([
        transforms.RandomAffine(degrees=12, translate=(0.15, 0.15), scale=(0.8, 1.2)),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.15)),
    ])

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    sample_weights = 1.0 / torch.bincount(train_y, minlength=NUM_CLASSES).float()[train_y]

    model = DigitCNN().to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model params: {param_count:,}")

    batch_size = 64
    num_epochs = 300
    steps_per_epoch = (len(train_y) + batch_size - 1) // batch_size
    optimizer = optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=3e-3, epochs=num_epochs, steps_per_epoch=steps_per_epoch,
        pct_start=0.05, div_factor=10, final_div_factor=1000
    )

    best_val_f1 = 0
    best_val_acc = 0
    for epoch in range(num_epochs):
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        sampled_idx = torch.multinomial(sample_weights, len(train_y), replacement=True)
        for i in range(0, len(sampled_idx), batch_size):
            idx = sampled_idx[i:i+batch_size]
            x = train_transform(train_x[idx]).to(device)
            y = train_y[idx].to(device)
            out = model(x)
            loss = criterion(out, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss += loss.item() * x.size(0)
            train_correct += (out.argmax(1) == y).sum().item()
            train_total += x.size(0)

        val_acc, val_f1 = evaluate(model, val_x, val_y, device)
        avg_loss = train_loss / train_total

        is_best = val_f1 > best_val_f1
        if (epoch + 1) % 50 == 0 or is_best:
            print(f"  Epoch {epoch+1:3d}: loss={avg_loss:.4f} "
                  f"val_acc={val_acc:.3f} val_f1={val_f1:.3f}{'  ★' if is_best else ''}")

        if is_best:
            best_val_f1 = val_f1
            best_val_acc = val_acc
            torch.save(model.state_dict(), REPO_ROOT / "digit_model_best.pth")

    # Print per-class errors
    model.load_state_dict(torch.load(REPO_ROOT / "digit_model_best.pth", weights_only=True))
    model.eval()
    with torch.no_grad():
        preds = model(val_x.to(device)).argmax(1).cpu()
        for c in range(NUM_CLASSES):
            mask = val_y == c
            if mask.sum() == 0:
                continue
            correct = (preds[mask] == c).sum().item()
            total = mask.sum().item()
            wrong = preds[mask][preds[mask] != c].tolist()
            print(f"  Class {c}: {correct}/{total} ({100*correct/total:.0f}%) errors→{wrong}")

    print(f"\nBest val (racing): acc={best_val_acc:.3f} f1={best_val_f1:.3f}")

    # Export to ONNX
    model.eval().cpu()
    dummy = torch.randn(1, 1, INPUT_SIZE, INPUT_SIZE)
    onnx_path = REPO_ROOT / "digit_model_v4.onnx"
    torch.onnx.export(
        model, dummy, onnx_path,
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
    )
    # Ensure all weights are inlined (no external .data file)
    import onnx
    onnx_model = onnx.load(onnx_path)
    onnx.save(onnx_model, onnx_path, save_as_external_data=False)
    # Clean up any stale external data file
    data_path = str(onnx_path) + ".data"
    if os.path.exists(data_path):
        os.remove(data_path)
    loaded_model_path = SCRIPT_DIR / "digit_model_v4.onnx"
    if loaded_model_path != onnx_path:
        loaded_model_path.write_bytes(onnx_path.read_bytes())
    onnx_size = onnx_path.stat().st_size
    print(f"Exported: {onnx_path} ({onnx_size / 1024:.0f} KB)")
    print(f"Copied loaded model: {loaded_model_path}")
    print(f"Model params: {param_count:,}")
    if HAS_TRACKIO:
        trackio.log({"best_val_acc": best_val_acc, "best_val_f1": best_val_f1})
        trackio.finish()


if __name__ == "__main__":
    train()
