# Racing Telemetry — mpv plugin

# Run mpv with telemetry on a video file
test file="":
    ./test.sh {{file}}

# Run all tests
check:
    cd test && lua test_core.lua && lua test_real_frames.lua

# Build the ONNX C bridge
build:
    cc -shared -O2 -o digit_ocr_bridge.dylib digit_ocr_bridge.c \
        -I/opt/homebrew/include/onnxruntime \
        -L/opt/homebrew/lib -lonnxruntime \
        -Wl,-rpath,/opt/homebrew/lib

# Install mpv scripts globally
install:
    mkdir -p ~/.config/mpv/scripts
    ln -sf {{justfile_directory()}}/telemetry.lua ~/.config/mpv/scripts/telemetry.lua
    ln -sf {{justfile_directory()}}/telemetry_core.lua ~/.config/mpv/scripts/telemetry_core.lua
    ln -sf {{justfile_directory()}}/digit_ocr.lua ~/.config/mpv/scripts/digit_ocr.lua
    ln -sf {{justfile_directory()}}/digit_ocr_bridge.dylib ~/.config/mpv/scripts/digit_ocr_bridge.dylib
    ln -sf {{justfile_directory()}}/ml/digit_model_v3.onnx ~/.config/mpv/scripts/digit_model_v3.onnx

# ML: generate training data
ml-data:
    cd ml && uv run python extract_training_data.py

# ML: train the digit recognition model
ml-train:
    cd ml && uv run python train.py

# ML: retrain with additional data
ml-retrain:
    cd ml && uv run python retrain.py

# ML: setup python environment
ml-setup:
    cd ml && uv sync
