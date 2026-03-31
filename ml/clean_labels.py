# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "torch>=2.0",
#     "torchvision>=0.15",
#     "transformers>=4.51",
#     "datasets>=3.0",
#     "pillow>=10,<12",
#     "accelerate>=0.26",
#     "qwen-vl-utils>=0.0.2",
#     "huggingface_hub>=0.20",
# ]
# ///
"""Clean dataset labels using Qwen3-VL-4B vision-language model.

For each image, asks the VLM what digit is shown. When VLM disagrees with
the label, flags it. Then pushes a corrected dataset back to HuggingFace.

Usage:
    uv run ml/clean_labels.py
"""

import ctypes, os, json
for p in ['/usr/lib/libcuda.so.1']:
    if os.path.exists(p):
        try: ctypes.CDLL(p)
        except: pass

import torch
from collections import Counter, defaultdict
from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoModelForImageTextToText, AutoProcessor
from qwen_vl_utils import process_vision_info


def predict_digit(model, processor, img):
    """Ask VLM what digit is in the image."""
    # Upscale small images
    w, h = img.size
    if w < 128 or h < 128:
        img = img.resize((max(w * 4, 128), max(h * 4, 128)), resample=3)

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": (
                "This is a cropped image from a racing game showing a single gear number digit. "
                "What digit (0-9) is displayed? Look carefully at the shape of the number. "
                "Reply with ONLY the single digit, nothing else."
            )},
        ],
    }]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=5, do_sample=False)
    response = processor.batch_decode(
        out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True
    )[0].strip()

    try:
        return int(response[0]) if response and response[0].isdigit() else -1
    except:
        return -1


def main():
    print("Loading Qwen3-VL-4B-Instruct...")
    model = AutoModelForImageTextToText.from_pretrained(
        "Qwen/Qwen3-VL-4B-Instruct",
        torch_dtype=torch.float16,
        device_map="cuda:0",
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-4B-Instruct")

    print("Loading dataset (force fresh)...")
    ds = load_dataset("tobil/racing-gears", download_mode="force_redownload")

    all_corrections = {}

    for split in ["validation", "train"]:
        data = ds[split]
        print(f"\n{'='*60}")
        print(f"Verifying {split}: {len(data)} samples")
        print(f"{'='*60}")

        corrections = {}
        source_stats = defaultdict(lambda: {"total": 0, "mismatches": 0})

        for i in range(len(data)):
            ex = data[i]
            img = ex['image'].convert('RGB')
            label = ex['label']
            source = ex.get('source', 'unknown')

            predicted = predict_digit(model, processor, img)

            source_stats[source]["total"] += 1

            if predicted != label and predicted >= 0:
                source_stats[source]["mismatches"] += 1
                corrections[i] = predicted
                if len(corrections) <= 30:
                    print(f"  [{i:5d}] label={label} → vlm={predicted} source={source}")

            if (i + 1) % 200 == 0:
                print(f"  ... checked {i+1}/{len(data)}, corrections: {len(corrections)}")

        all_corrections[split] = corrections

        print(f"\n--- {split} Summary ---")
        print(f"Total: {len(data)}, Corrections: {len(corrections)} ({100*len(corrections)/len(data):.1f}%)")
        for src, stats in sorted(source_stats.items()):
            pct = 100 * stats["mismatches"] / stats["total"] if stats["total"] > 0 else 0
            print(f"  {src}: {stats['total']} samples, {stats['mismatches']} corrections ({pct:.1f}%)")

    # Save corrections
    with open("ml/label_corrections.json", "w") as f:
        json.dump({s: {str(k): v for k, v in c.items()} for s, c in all_corrections.items()}, f, indent=2)
    print(f"\nSaved corrections to ml/label_corrections.json")

    # Apply corrections
    print("\nApplying corrections...")
    new_splits = {}
    for split in ["train", "validation"]:
        data = ds[split]
        corrections = all_corrections[split]
        images, labels, sources = [], [], []
        for i in range(len(data)):
            ex = data[i]
            label = corrections.get(i, ex['label'])
            if label < 0:
                continue
            images.append(ex['image'])
            labels.append(label)
            sources.append(ex.get('source', 'unknown'))
        new_splits[split] = Dataset.from_dict({"image": images, "label": labels, "source": sources})
        print(f"  {split}: {len(data)} → {len(labels)} ({len(corrections)} relabeled)")
        print(f"    Labels: {sorted(Counter(labels).items())}")

    cleaned = DatasetDict(new_splits)
    print("\nPushing cleaned dataset to tobil/racing-gears...")
    cleaned.push_to_hub("tobil/racing-gears", commit_message="Clean labels using Qwen3-VL-4B VLM verification")
    print("Done!")


if __name__ == "__main__":
    main()
