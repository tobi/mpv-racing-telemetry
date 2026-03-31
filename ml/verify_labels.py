# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "torch>=2.0",
#     "transformers>=4.45",
#     "datasets>=3.0",
#     "pillow>=10,<12",
#     "accelerate>=0.26",
#     "torchvision>=0.15",
#     "qwen-vl-utils>=0.0.2",
# ]
# ///
"""Verify dataset labels using Qwen2.5-VL-3B vision-language model."""

import ctypes, os
for p in ['/usr/lib/libcuda.so.1']:
    if os.path.exists(p):
        try: ctypes.CDLL(p)
        except: pass

import json
import torch
from datasets import load_dataset
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

def main():
    print("Loading model...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-3B-Instruct",
        torch_dtype=torch.float16,
        device_map="cuda:0",
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")

    print("Loading dataset...")
    ds = load_dataset("tobil/racing-gears")

    results = {"train": [], "validation": []}

    for split in ["validation", "train"]:
        print(f"\nVerifying {split} ({len(ds[split])} samples)...")
        mismatches = []
        for i, ex in enumerate(ds[split]):
            img = ex['image'].convert('RGB')
            label = ex['label']
            source = ex['source']

            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": "What single digit (0-9) is shown in this image? Reply with ONLY the digit, nothing else."},
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
            response = processor.batch_decode(out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0].strip()

            try:
                predicted = int(response[0]) if response and response[0].isdigit() else -1
            except:
                predicted = -1

            if predicted != label:
                mismatches.append({
                    "index": i, "label": label, "predicted": predicted,
                    "response": response, "source": source
                })
                print(f"  MISMATCH [{split}][{i}]: label={label} vlm={predicted} ({response}) source={source}")

            if (i + 1) % 100 == 0:
                print(f"  Checked {i+1}/{len(ds[split])}, mismatches so far: {len(mismatches)}")

        results[split] = mismatches
        print(f"{split}: {len(mismatches)} mismatches out of {len(ds[split])}")

    with open("ml/label_verification.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to ml/label_verification.json")


if __name__ == "__main__":
    main()
