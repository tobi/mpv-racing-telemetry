# Agent Notes

## Gear OCR training data workflow

When asked to add examples to the gear digit training data:

1. Work in the Hugging Face dataset checkout, not ad-hoc local folders.
   - `dataset/` is ignored by this repo and is the local checkout of
     `tobil/racing-gears`.
   - If it is missing or was created with `hf download` instead of git, replace it
     with a real dataset repo checkout:
     ```bash
     rm -rf dataset
     git lfs install
     git clone https://huggingface.co/datasets/tobil/racing-gears dataset
     ```

2. Add new images to the dataset repo in the appropriate raw/source structure.
   - For one-off hard examples, prefer adding them under an existing curated source
     in `dataset/raw/...` or the dataset's documented source folders, then update
     labels / manifests as needed.
   - Follow `dataset/AGENTS.md` and `dataset/README.md`; they are the source of
     truth for extraction, labels, and `scripts/build_dataset.py`.
   - Do not resurrect `ml/local_data/`; local hard examples should be upstreamed
     to the HF dataset so training is reproducible.

3. Rebuild and validate the dataset inside `dataset/`.
   ```bash
   cd dataset
   uv run python scripts/build_dataset.py
   ```
   Confirm the generated `data/train-*.parquet` and
   `data/validation-*.parquet` include the new examples.

4. Commit and push the dataset change from inside `dataset/`.
   ```bash
   cd dataset
   git status
   git add .
   git commit -m "add <brief description> gear crops"
   git push
   ```
   If the checkout is not a git repo, use `HfApi.upload_folder(...)` only as a
   fallback, and say so explicitly.

5. Retrain from this repo using the local dataset checkout.
   ```bash
   uv run ml/train.py
   ```
   `ml/train.py` should detect `./dataset` and train from it; otherwise it falls
   back to `tobil/racing-gears` on the Hub. Training is expected to use MLX on the
   Apple GPU and to print `MLX default device: Device(gpu, 0)`.

6. Verify the result.
   - Test the specific supplied crop(s) with `digit_ocr.lua`/mpv if applicable.
   - Run:
     ```bash
     just check
     ```
   - Report final train loss, best validation accuracy/F1, and any problematic
     class errors.

7. Commit model/code changes in this repo separately from the HF dataset commit.
   - Track updated model artifacts that the plugin loads:
     - `digit_model_v4.onnx`
     - `ml/digit_model_v4.onnx`
   - Keep `dataset/`, `ml/data/`, and `ml/local_data/` out of this repo.
