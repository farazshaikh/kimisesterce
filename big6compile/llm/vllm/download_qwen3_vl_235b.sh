#!/bin/bash
# Download Qwen3-VL-235B-A22B-Instruct-FP8 model from Hugging Face
# This is a 235B parameter vision-language model with FP8 quantization

set -euo pipefail

MODEL_NAME="Qwen/Qwen3-VL-235B-A22B-Instruct-FP8"
TARGET_DIR="/compile/llm/models/vllm/Qwen3-VL-235B-A22B-Instruct-FP8"

echo "Downloading Qwen3-VL-235B-A22B-Instruct-FP8..."
echo "Model: $MODEL_NAME"
echo "Target directory: $TARGET_DIR"
echo ""
echo "Note: This is a 235B parameter model with FP8 quantization."
echo "The download size will be approximately 120-150GB."
echo ""

# Create target directory if it doesn't exist
mkdir -p "$TARGET_DIR"

# Activate venv if available
if [ -f "/compile/llm/vllm/venv/bin/activate" ]; then
    source /compile/llm/vllm/venv/bin/activate
fi

# Download using huggingface-cli (recommended) or huggingface_hub
if command -v huggingface-cli &> /dev/null; then
    echo "Using huggingface-cli to download..."
    huggingface-cli download "$MODEL_NAME" \
        --local-dir "$TARGET_DIR" \
        --local-dir-use-symlinks False
else
    echo "Using Python to download..."
    python3 << EOF
from huggingface_hub import snapshot_download
import os

print("Starting download...")
snapshot_download(
    repo_id="$MODEL_NAME",
    local_dir="$TARGET_DIR",
    local_dir_use_symlinks=False
)
print("Download complete!")
EOF
fi

echo ""
echo "Download completed successfully!"
echo "Model saved to: $TARGET_DIR"
echo ""
echo "You can now run the model using: ./run_vllm_qwen3_vl.sh"

