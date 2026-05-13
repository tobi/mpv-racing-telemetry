# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "mlx>=0.29",
#     "torch>=2.0",
#     "datasets>=3.0",
#     "pillow>=10,<12",
#     "trackio>=0.20",
#     "onnx>=1.14",
#     "onnxscript>=0.1",
#     "tqdm>=4.66",
# ]
# ///
"""Train gear digit recognition model on tobil/racing-gears.

Input: 32x32 grayscale images normalized to [-1, 1]
Output: 10 classes (digits 0-9)
Architecture: Small CNN exported to ONNX for use in mpv via LuaJIT FFI.

The training loop uses Apple MLX. If ./dataset exists, it is used as a local
Hugging Face dataset checkout; otherwise the script loads tobil/racing-gears
from the Hub.

Usage:
    uv run ml/train.py
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as mlx_nn
import mlx.optimizers as mlx_optim
import numpy as np
import torch
import torch.nn as torch_nn
from datasets import load_dataset
from PIL import Image
from tqdm.auto import tqdm

try:
    import trackio
    HAS_TRACKIO = True
except Exception:
    HAS_TRACKIO = False

INPUT_SIZE = 32
NUM_CLASSES = 10
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DATASET_DIR = REPO_ROOT / "dataset"


def image_to_array(img: Image.Image) -> np.ndarray:
    img = img.convert("L").resize((INPUT_SIZE, INPUT_SIZE))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = arr * 2.0 - 1.0
    return arr[..., None]  # MLX Conv2d expects NHWC


def precompute_arrays(hf_split, exclude_source: str | None = None):
    xs, ys = [], []
    for ex in hf_split:
        if exclude_source and ex.get("source") == exclude_source:
            continue
        xs.append(image_to_array(ex["image"]))
        ys.append(ex["label"])
    return np.stack(xs).astype(np.float32), np.array(ys, dtype=np.int64)


def load_training_dataset():
    if DATASET_DIR.exists():
        print(f"Loading local HF dataset checkout: {DATASET_DIR}", flush=True)
        return load_dataset(str(DATASET_DIR))

    print("Loading tobil/racing-gears from Hugging Face Hub...", flush=True)
    return load_dataset("tobil/racing-gears")


class DigitCNN(mlx_nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = mlx_nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = mlx_nn.BatchNorm(32)
        self.conv2 = mlx_nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = mlx_nn.BatchNorm(64)
        self.conv3 = mlx_nn.Conv2d(64, 64, 3, padding=1)
        self.bn3 = mlx_nn.BatchNorm(64)
        self.pool = mlx_nn.MaxPool2d(2)
        self.avg = mlx_nn.AvgPool2d(2)  # 8x8 -> 4x4, same as AdaptiveAvgPool2d(4)
        self.fc1 = mlx_nn.Linear(64 * 4 * 4, 128)
        self.drop = mlx_nn.Dropout(0.4)
        self.fc2 = mlx_nn.Linear(128, NUM_CLASSES)

    def __call__(self, x):
        x = self.pool(mlx_nn.relu(self.bn1(self.conv1(x))))
        x = self.pool(mlx_nn.relu(self.bn2(self.conv2(x))))
        x = mlx_nn.relu(self.bn3(self.conv3(x)))
        x = self.avg(x)
        x = x.reshape((x.shape[0], -1))
        x = mlx_nn.relu(self.fc1(x))
        x = self.drop(x)
        return self.fc2(x)


class TorchDigitCNN(torch_nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch_nn.Sequential(
            torch_nn.Conv2d(1, 32, 3, padding=1),
            torch_nn.BatchNorm2d(32),
            torch_nn.ReLU(),
            torch_nn.MaxPool2d(2),
            torch_nn.Conv2d(32, 64, 3, padding=1),
            torch_nn.BatchNorm2d(64),
            torch_nn.ReLU(),
            torch_nn.MaxPool2d(2),
            torch_nn.Conv2d(64, 64, 3, padding=1),
            torch_nn.BatchNorm2d(64),
            torch_nn.ReLU(),
            torch_nn.AdaptiveAvgPool2d(4),
        )
        self.classifier = torch_nn.Sequential(
            torch_nn.Flatten(),
            torch_nn.Linear(64 * 4 * 4, 128),
            torch_nn.ReLU(),
            torch_nn.Dropout(0.4),
            torch_nn.Linear(128, NUM_CLASSES),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def to_numpy_tree(tree: Any):
    if isinstance(tree, dict):
        return {k: to_numpy_tree(v) for k, v in tree.items()}
    return np.array(tree)


def to_mx_tree(tree: Any):
    if isinstance(tree, dict):
        return {k: to_mx_tree(v) for k, v in tree.items()}
    return mx.array(tree)


def compute_f1_macro(preds, targets, num_classes=NUM_CLASSES):
    f1s = []
    for c in range(num_classes):
        tp = int(((preds == c) & (targets == c)).sum())
        fp = int(((preds == c) & (targets != c)).sum())
        fn = int(((preds != c) & (targets == c)).sum())
        if tp + fn == 0:
            continue
        prec = tp / (tp + fp) if tp + fp > 0 else 0
        rec = tp / (tp + fn) if tp + fn > 0 else 0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec > 0 else 0)
    return sum(f1s) / len(f1s) if f1s else 0


def evaluate(model: DigitCNN, val_x: np.ndarray, val_y: np.ndarray, batch_size=512):
    model.eval()
    preds = []
    for i in range(0, len(val_y), batch_size):
        logits = model(mx.array(val_x[i:i + batch_size]))
        preds.append(np.array(mx.argmax(logits, axis=1)))
    preds = np.concatenate(preds)
    acc = float((preds == val_y).mean())
    f1 = compute_f1_macro(preds, val_y)
    return acc, f1, preds


def augment_batch(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    # Lightweight MLX-friendly augmentation: small translations plus random erasing.
    x = x.copy()
    for i in range(x.shape[0]):
        if rng.random() < 0.7:
            dy = int(rng.integers(-4, 5))
            dx = int(rng.integers(-4, 5))
            shifted = np.full_like(x[i], -1.0)
            src_y0, src_y1 = max(0, -dy), min(INPUT_SIZE, INPUT_SIZE - dy)
            dst_y0, dst_y1 = max(0, dy), min(INPUT_SIZE, INPUT_SIZE + dy)
            src_x0, src_x1 = max(0, -dx), min(INPUT_SIZE, INPUT_SIZE - dx)
            dst_x0, dst_x1 = max(0, dx), min(INPUT_SIZE, INPUT_SIZE + dx)
            shifted[dst_y0:dst_y1, dst_x0:dst_x1, :] = x[i, src_y0:src_y1, src_x0:src_x1, :]
            x[i] = shifted
        if rng.random() < 0.3:
            h = int(rng.integers(2, 9))
            w = int(rng.integers(2, 9))
            y = int(rng.integers(0, INPUT_SIZE - h + 1))
            x0 = int(rng.integers(0, INPUT_SIZE - w + 1))
            x[i, y:y + h, x0:x0 + w, :] = -1.0
    return x


def mlx_to_torch_model(model: DigitCNN) -> TorchDigitCNN:
    p = to_numpy_tree(model.parameters())
    torch_model = TorchDigitCNN()
    sd = torch_model.state_dict()

    def conv(name, torch_prefix):
        sd[f"{torch_prefix}.weight"] = torch.tensor(p[name]["weight"].transpose(0, 3, 1, 2))
        sd[f"{torch_prefix}.bias"] = torch.tensor(p[name]["bias"])

    def bn(name, torch_prefix):
        sd[f"{torch_prefix}.weight"] = torch.tensor(p[name]["weight"])
        sd[f"{torch_prefix}.bias"] = torch.tensor(p[name]["bias"])
        sd[f"{torch_prefix}.running_mean"] = torch.tensor(p[name]["running_mean"])
        sd[f"{torch_prefix}.running_var"] = torch.tensor(p[name]["running_var"])
        sd[f"{torch_prefix}.num_batches_tracked"] = torch.tensor(0, dtype=torch.long)

    def linear(name, torch_prefix):
        weight = p[name]["weight"]
        # MLX Conv2d uses NHWC activations, so the first linear layer was
        # trained on a 4x4x64 flatten order. PyTorch flattens NCHW from the
        # exported conv stack, so permute fc1 columns to 64x4x4 order.
        if name == "fc1":
            weight = weight.reshape(128, 4, 4, 64).transpose(0, 3, 1, 2).reshape(128, 64 * 4 * 4)
        sd[f"{torch_prefix}.weight"] = torch.tensor(weight)
        sd[f"{torch_prefix}.bias"] = torch.tensor(p[name]["bias"])

    conv("conv1", "features.0")
    bn("bn1", "features.1")
    conv("conv2", "features.4")
    bn("bn2", "features.5")
    conv("conv3", "features.8")
    bn("bn3", "features.9")
    linear("fc1", "classifier.1")
    linear("fc2", "classifier.4")

    torch_model.load_state_dict(sd)
    torch_model.eval()
    return torch_model


def export_onnx(model: DigitCNN, best_val_acc: float, best_val_f1: float, param_count: int):
    torch_model = mlx_to_torch_model(model)
    torch.save(torch_model.state_dict(), REPO_ROOT / "digit_model_best.pth")

    onnx_path = REPO_ROOT / "digit_model_v4.onnx"
    dummy = torch.randn(1, 1, INPUT_SIZE, INPUT_SIZE)
    if hasattr(torch, "compile"):
        try:
            compiled_model = torch.compile(torch_model, mode="reduce-overhead")
            with torch.no_grad():
                compiled_model(dummy)
            print("torch.compile warmup: ok", flush=True)
        except Exception as e:
            print(f"torch.compile warmup failed ({e}); exporting eager model", flush=True)
    torch.onnx.export(
        torch_model, dummy, onnx_path,
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
    )

    import onnx
    onnx_model = onnx.load(onnx_path)
    onnx.save(onnx_model, onnx_path, save_as_external_data=False)
    data_path = str(onnx_path) + ".data"
    if os.path.exists(data_path):
        os.remove(data_path)

    loaded_model_path = SCRIPT_DIR / "digit_model_v4.onnx"
    if loaded_model_path != onnx_path:
        loaded_model_path.write_bytes(onnx_path.read_bytes())

    onnx_size = onnx_path.stat().st_size
    print(f"Exported: {onnx_path} ({onnx_size / 1024:.0f} KB)", flush=True)
    print(f"Copied loaded model: {loaded_model_path}", flush=True)
    print(f"Model params: {param_count:,}", flush=True)
    if HAS_TRACKIO:
        trackio.log({"best_val_acc": best_val_acc, "best_val_f1": best_val_f1})
        trackio.finish()


def train():
    print(f"MLX default device: {mx.default_device()}", flush=True)
    if "gpu" not in str(mx.default_device().type).lower():
        raise RuntimeError("MLX is not using the Apple GPU; aborting training")

    global HAS_TRACKIO
    if HAS_TRACKIO:
        try:
            run = trackio.init(project="racing-gears", config={"device": str(mx.default_device()), "trainer": "mlx"})
            print(f"Trackio run: {run.name}", flush=True)
        except Exception as e:
            print(f"trackio init failed ({e}), continuing without tracking", flush=True)
            HAS_TRACKIO = False

    ds = load_training_dataset()

    print("Precomputing arrays...", flush=True)
    t0 = time.time()
    train_x, train_y = precompute_arrays(ds["train"])
    val_x, val_y = precompute_arrays(ds["validation"], exclude_source="mnist")
    print(f"Precomputed in {time.time() - t0:.1f}s. Train: {len(train_y)}, Val (racing): {len(val_y)}", flush=True)

    rng = np.random.default_rng(42)
    counts = np.bincount(train_y, minlength=NUM_CLASSES).astype(np.float32)
    sample_weights = (1.0 / np.maximum(counts[train_y], 1.0))
    sample_weights = sample_weights / sample_weights.sum()

    model = DigitCNN()
    param_count = sum(int(np.prod(v.shape)) for group in to_numpy_tree(model.parameters()).values() for v in group.values())
    print(f"Model params: {param_count:,}", flush=True)

    batch_size = 64
    num_epochs = 300
    steps_per_epoch = (len(train_y) + batch_size - 1) // batch_size
    optimizer = mlx_optim.AdamW(learning_rate=3e-3, weight_decay=1e-2)

    def loss_fn(model, x, y):
        logits = model(x)
        return mlx_nn.losses.cross_entropy(logits, y, label_smoothing=0.1, reduction="mean")

    loss_and_grad = mlx_nn.value_and_grad(model, loss_fn)

    best_val_f1 = 0.0
    best_val_acc = 0.0
    best_params = None
    final_loss = 0.0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        sampled_idx = rng.choice(len(train_y), size=len(train_y), replace=True, p=sample_weights)

        pbar = tqdm(
            range(0, len(sampled_idx), batch_size),
            total=steps_per_epoch,
            desc=f"Epoch {epoch + 1}/{num_epochs}",
            dynamic_ncols=True,
            leave=False,
        )
        for i in pbar:
            idx = sampled_idx[i:i + batch_size]
            xb_np = augment_batch(train_x[idx], rng)
            yb_np = train_y[idx]
            xb = mx.array(xb_np)
            yb = mx.array(yb_np)

            loss, grads = loss_and_grad(model, xb, yb)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)

            logits = model(xb)
            pred = np.array(mx.argmax(logits, axis=1))
            loss_val = float(np.array(loss))
            train_loss += loss_val * len(idx)
            train_correct += int((pred == yb_np).sum())
            train_total += len(idx)
            pbar.set_postfix(
                loss=f"{train_loss / train_total:.4f}",
                batch_loss=f"{loss_val:.4f}",
                train_acc=f"{train_correct / train_total:.3f}",
                best_f1=f"{best_val_f1:.3f}",
            )

        avg_loss = train_loss / train_total
        final_loss = avg_loss
        val_acc, val_f1, _ = evaluate(model, val_x, val_y)

        is_best = val_f1 > best_val_f1
        if (epoch + 1) % 50 == 0 or is_best:
            print(f"  Epoch {epoch + 1:3d}: loss={avg_loss:.4f} "
                  f"train_acc={train_correct / train_total:.3f} "
                  f"val_acc={val_acc:.3f} val_f1={val_f1:.3f}{'  ★' if is_best else ''}",
                  flush=True)

        if is_best:
            best_val_f1 = val_f1
            best_val_acc = val_acc
            best_params = to_numpy_tree(model.parameters())

    if best_params is not None:
        model.update(to_mx_tree(best_params))
        mx.eval(model.parameters())

    val_acc, val_f1, preds = evaluate(model, val_x, val_y)
    for c in range(NUM_CLASSES):
        mask = val_y == c
        if not mask.any():
            continue
        correct = int((preds[mask] == c).sum())
        total = int(mask.sum())
        wrong = preds[mask][preds[mask] != c].tolist()
        print(f"  Class {c}: {correct}/{total} ({100 * correct / total:.0f}%) errors→{wrong}", flush=True)

    print(f"\nFinal train loss: {final_loss:.4f}", flush=True)
    print(f"Best val (racing): acc={best_val_acc:.3f} f1={best_val_f1:.3f}", flush=True)
    print(f"Restored best val: acc={val_acc:.3f} f1={val_f1:.3f}", flush=True)

    export_onnx(model, best_val_acc, best_val_f1, param_count)


if __name__ == "__main__":
    train()
