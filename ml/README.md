# Digit Recognition Model

Small CNN that recognizes gear numbers (0–9) from 32×32 grayscale crops of racing telemetry overlays. Runs inside mpv via ONNX Runtime + LuaJIT FFI.

## Dataset

Training data is on Hugging Face: **[tobil/racing-gears](https://huggingface.co/datasets/tobil/racing-gears)**

Images are cropped from onboard racing videos — the gear indicator region of the telemetry HUD. Each image is labeled with the digit shown (0–9).

Local validation data lives in `data/val/` and `data/gear_real/`, organized by digit in subdirectories.

## Architecture

`DigitCNN` — a small 3-layer conv net:

```
Conv2d(1→32, 3×3) → BN → ReLU → MaxPool
Conv2d(32→64, 3×3) → BN → ReLU → MaxPool
Conv2d(64→64, 3×3) → BN → ReLU → AdaptiveAvgPool(4)
Flatten → Linear(1024→128) → ReLU → Dropout(0.4) → Linear(128→10)
```

Input: 32×32 grayscale, normalized to [-1, 1].  
Output: 10 classes (digits 0–9).

## Training

```bash
uv run ml/train.py
```

Trains for 300 epochs with OneCycleLR (max lr 3e-3). Exports the best model to ONNX:

- `digit_model_v4.onnx` — production model loaded by mpv
- `digit_model_best.pth` — PyTorch checkpoint

The training script pulls the dataset directly from Hugging Face and excludes synthetic samples from validation.

## Label Cleaning

Two scripts use vision-language models (Qwen) to verify and correct dataset labels:

- `clean_labels.py` — runs Qwen3-VL-4B over the dataset, flags disagreements, pushes corrections back to HF
- `verify_labels.py` — lighter verification pass with Qwen2.5-VL-3B

Manual corrections are tracked in `label_corrections.json`.

## Inference

The ONNX model is loaded in mpv by `digit_ocr.lua` via ONNX Runtime's C API (LuaJIT FFI). See the top-level `digit_ocr.lua` for the inference pipeline.
