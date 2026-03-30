"""
Extract gear and lap number training data from the video.
Produces labeled crops with augmentation for training a digit recognizer.

Usage: uv run extract_training_data.py
"""
import subprocess
import json
from pathlib import Path
from PIL import Image
import numpy as np

VIDEO = "../downloads/T7COWXv0HEU.mp4"
OUT_DIR = Path("data")

# Crop regions in 1280x720 video
GEAR_CROP = {"x": 1062, "y": 612, "w": 64, "h": 58}
LAP_CROP = {"x": 275, "y": 5, "w": 40, "h": 22}

# Known ground truth from visual inspection
GEAR_LABELS = {
    935: "1", 950: "2", 955: "2", 980: "2", 1030: "2", 1060: "2",
    940: "3", 975: "3", 1000: "3", 1050: "3", 1090: "3",
    985: "4", 1035: "4",
    945: "5", 960: "5", 970: "5", 1020: "5", 1080: "5",
    990: "6", 995: "6", 1010: "6", 1100: "6", 1120: "6", 1150: "6",
}

# Lap numbers change over time - extract at 1fps and auto-label from pixel signatures
# We know lap 9 starts around t=930, lap 10 around t=1060, lap 11 around t=1180


def extract_frame_crop(video_path, timestamp, crop, output_path):
    """Extract a cropped region from a video frame."""
    cmd = [
        "ffmpeg", "-y", "-ss", str(timestamp),
        "-i", video_path, "-vframes", "1", "-update", "1",
        "-vf", f"crop={crop['w']}:{crop['h']}:{crop['x']}:{crop['y']}",
        str(output_path)
    ]
    subprocess.run(cmd, capture_output=True)


def augment_image(img: Image.Image, label: str) -> list[tuple[Image.Image, str]]:
    """Generate augmented versions of an image."""
    results = [(img, label)]
    arr = np.array(img)
    h, w = arr.shape[:2]

    # Shift variations
    for dx, dy in [(-2, 0), (2, 0), (0, -2), (0, 2), (-2, -2), (2, 2)]:
        shifted = np.zeros_like(arr)
        sx = max(0, dx); sy = max(0, dy)
        ex = min(w, w + dx); ey = min(h, h + dy)
        ssx = max(0, -dx); ssy = max(0, -dy)
        eex = ssx + (ex - sx); eey = ssy + (ey - sy)
        shifted[ssy:eey, ssx:eex] = arr[sy:ey, sx:ex]
        results.append((Image.fromarray(shifted), label))

    # Scale variations
    for scale in [0.85, 0.9, 1.1, 1.15]:
        nw, nh = int(w * scale), int(h * scale)
        scaled = img.resize((nw, nh), Image.BILINEAR)
        # Center crop/pad back to original size
        canvas = Image.new(img.mode, (w, h), 0)
        px = (w - nw) // 2; py = (h - nh) // 2
        canvas.paste(scaled, (max(0, px), max(0, py)))
        results.append((canvas, label))

    # Brightness
    for factor in [0.7, 0.85, 1.15, 1.3]:
        bright = np.clip(arr.astype(float) * factor, 0, 255).astype(np.uint8)
        results.append((Image.fromarray(bright), label))

    # Small rotation
    for angle in [-3, -2, 2, 3]:
        rotated = img.rotate(angle, fillcolor=0)
        results.append((Image.fromarray(np.array(rotated)), label))

    return results


def main():
    # Create output dirs
    for split in ["train", "val"]:
        for cls in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "N"]:
            (OUT_DIR / "gear" / split / cls).mkdir(parents=True, exist_ok=True)
            (OUT_DIR / "lapnum" / split / cls).mkdir(parents=True, exist_ok=True)

    # ── Extract gear training data ──
    print("Extracting gear crops...")
    gear_counts = {}
    idx = 0

    for ts, label in GEAR_LABELS.items():
        crop_path = OUT_DIR / f"gear_raw_{ts}.png"
        extract_frame_crop(VIDEO, ts, GEAR_CROP, crop_path)
        img = Image.open(crop_path).convert("L")  # grayscale

        augmented = augment_image(img, label)
        for aug_img, aug_label in augmented:
            split = "val" if idx % 10 == 0 else "train"
            save_path = OUT_DIR / "gear" / split / aug_label / f"gear_{ts}_{idx}.png"
            aug_img.save(save_path)
            idx += 1
            gear_counts[aug_label] = gear_counts.get(aug_label, 0) + 1

        crop_path.unlink()

    # Also extract at 2fps across the full video for more variety
    print("Extracting bulk gear crops at 2fps...")
    bulk_dir = OUT_DIR / "gear_bulk"
    bulk_dir.mkdir(exist_ok=True)
    subprocess.run([
        "ffmpeg", "-y", "-i", VIDEO,
        "-vf", f"crop={GEAR_CROP['w']}:{GEAR_CROP['h']}:{GEAR_CROP['x']}:{GEAR_CROP['y']},fps=2",
        "-vsync", "vfr", str(bulk_dir / "frame_%06d.png")
    ], capture_output=True)

    # Auto-label bulk frames using the pixel-signature detector
    # Import the JS detector via subprocess
    bulk_frames = sorted(bulk_dir.glob("*.png"))
    print(f"Auto-labeling {len(bulk_frames)} bulk frames...")

    # Quick pixel-based labeling: threshold the center of the image
    # and match against known patterns (simplified from gear-detect.ts)
    for frame_path in bulk_frames:
        img = Image.open(frame_path).convert("L")
        arr = np.array(img)
        # Simple heuristic: look at brightness pattern
        # The digit is white on dark bg, centered roughly at rows 25-50, cols 15-45
        center = arr[25:50, 15:45]
        bright_pct = np.mean(center > 150)

        # Very rough classification by pixel density
        # Skip ambiguous frames
        if bright_pct < 0.05 or bright_pct > 0.6:
            continue

        # Use the existing labeled data to match
        # For now just add as unlabeled for semi-supervised learning
        # (we'll rely on the labeled augmented data)

    # Clean up bulk
    import shutil
    shutil.rmtree(bulk_dir, ignore_errors=True)

    print(f"Gear data: {gear_counts}")

    # ── Extract lap number training data ──
    print("\nExtracting lap number crops...")
    lap_counts = {}
    idx = 0

    # We need to know which lap number is showing at each timestamp
    # Lap transitions (approximate from video analysis):
    # t=0-180: lap 1-5 (race start), t=930: lap 9, t=1060: lap 10, t=1180: lap 11
    # Extract at many timestamps and label based on the visible number
    lap_timestamps = {}
    # Extract one frame per second across the video
    for ts in range(100, 2400, 2):
        crop_path = OUT_DIR / f"lap_raw_{ts}.png"
        extract_frame_crop(VIDEO, ts, LAP_CROP, crop_path)
        if crop_path.exists():
            img = Image.open(crop_path).convert("L")
            arr = np.array(img)
            # Simple digit presence check: enough white pixels?
            if np.mean(arr > 150) > 0.05:
                lap_timestamps[ts] = (img, crop_path)
            else:
                crop_path.unlink()

    print(f"Got {len(lap_timestamps)} lap number crops")

    # Known lap number ground truth (from video analysis)
    LAP_GROUND_TRUTH = {
        # Each entry: (start_time, end_time, lap_number)
        (100, 250): "1", (250, 400): "2", (400, 530): "3",
        (530, 650): "4", (650, 770): "5", (770, 870): "6",
        (870, 940): "7", (940, 950): "8", (950, 1060): "9",
        (1060, 1190): "10", (1190, 1320): "11", (1320, 1440): "12",
        (1440, 1570): "13", (1570, 1700): "14", (1700, 1830): "15",
    }

    for ts, (img, crop_path) in lap_timestamps.items():
        label = None
        for (start, end), lap in LAP_GROUND_TRUTH.items():
            if start <= ts < end:
                label = lap
                break

        if label is None:
            crop_path.unlink()
            continue

        # For multi-digit numbers, we need individual digit labels
        # For now, treat as single-character classification
        # Split multi-digit: each digit gets its own crop
        arr = np.array(img)
        h, w = arr.shape

        if len(label) == 1:
            # Single digit - use whole crop
            augmented = augment_image(img, label)
            for aug_img, aug_label in augmented:
                split = "val" if idx % 10 == 0 else "train"
                save_path = OUT_DIR / "lapnum" / split / aug_label / f"lap_{ts}_{idx}.png"
                aug_img.save(save_path)
                idx += 1
                lap_counts[aug_label] = lap_counts.get(aug_label, 0) + 1
        elif len(label) == 2:
            # Two digits - split horizontally
            mid = w // 2
            left_img = img.crop((0, 0, mid, h))
            right_img = img.crop((mid, 0, w, h))
            for digit_img, digit_label in [(left_img, label[0]), (right_img, label[1])]:
                augmented = augment_image(digit_img, digit_label)
                for aug_img, aug_label in augmented:
                    split = "val" if idx % 10 == 0 else "train"
                    save_path = OUT_DIR / "lapnum" / split / aug_label / f"lap_{ts}_{idx}.png"
                    aug_img.save(save_path)
                    idx += 1
                    lap_counts[aug_label] = lap_counts.get(aug_label, 0) + 1

        crop_path.unlink()

    print(f"Lap number data: {lap_counts}")
    print("\nDone! Training data in", OUT_DIR)


if __name__ == "__main__":
    main()
