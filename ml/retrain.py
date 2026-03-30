"""Retrain with extra crops from multiple videos. Run: cd ml && uv run retrain.py"""
import json, subprocess, shutil
from pathlib import Path
from PIL import Image
import numpy as np

DATA = Path("data")
EXTRA = DATA / "gear_extra"
LABELS_FILE = EXTRA / "labels.json"

# Load auto-labels  
with open(LABELS_FILE) as f:
    labeled = json.load(f)

print(f"Loaded {len(labeled)} auto-labeled standard crops")

# Also label the variant crops by matching timestamps
# The _tight_, _wide_, _shiftL_, _shiftR_ variants share the same gear as _std_ at same timestamp
labels_by_ts = {}
for item in labeled:
    # Extract video+timestamp from filename like "T7COWXv0HEU_std_30.png"
    parts = item["file"].replace(".png", "").split("_")
    vid = parts[0]
    ts = parts[-1]
    labels_by_ts[f"{vid}_{ts}"] = item["gear"]

all_samples = []
for png in sorted(EXTRA.glob("*.png")):
    name = png.stem
    parts = name.split("_")
    if name.startswith("era_"):
        # ERA-style samples - labeled as 5 from the screenshot
        all_samples.append((str(png), "5"))
        continue
    vid = parts[0]
    ts = parts[-1]
    key = f"{vid}_{ts}"
    if key in labels_by_ts:
        all_samples.append((str(png), labels_by_ts[key]))

print(f"Total samples with labels: {len(all_samples)}")

# Distribution
dist = {}
for _, g in all_samples:
    dist[g] = dist.get(g, 0) + 1
print(f"Distribution: {dist}")

# Create augmented training set
def augment(img, label):
    results = [(img, label)]
    arr = np.array(img)
    h, w = arr.shape[:2]
    # Brightness
    for f in [0.7, 0.85, 1.15, 1.3]:
        results.append((Image.fromarray(np.clip(arr * f, 0, 255).astype(np.uint8)), label))
    # Small rotation
    for a in [-3, 3]:
        results.append((img.rotate(a, fillcolor=0), label))
    # Invert (for dark-on-light vs light-on-dark)
    inv = 255 - arr
    results.append((Image.fromarray(inv), label))
    return results

CLASSES = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "N"]
for split in ["train", "val"]:
    for c in CLASSES:
        (DATA / "gear" / split / c).mkdir(parents=True, exist_ok=True)

idx = 0
counts = {"train": 0, "val": 0}
for path, gear in all_samples:
    try:
        img = Image.open(path).convert("L")
    except:
        continue
    augmented = augment(img, gear)
    for aug_img, aug_label in augmented:
        split = "val" if idx % 10 == 0 else "train"
        save_path = DATA / "gear" / split / aug_label / f"extra_{idx}.png"
        aug_img.save(save_path)
        counts[split] += 1
        idx += 1

print(f"Saved: {counts}")
print("Now run: uv run train.py")
