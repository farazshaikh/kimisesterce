#!/bin/bash
# Run llama-server locally with Qwen3-30B model

set -euo pipefail

MODEL_PATH="/wrk/llm/models/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf"
LLAMA_SERVER_BIN="/wrk/llm/llama.cpp/build/bin/llama-server"
LIB_PATH="/wrk/llm/llama.cpp/build/bin"

# Check if model exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "Model not found at: $MODEL_PATH"
    echo "Please download the model to: $MODEL_PATH"
    exit 1
fi

# Check if llama-server exists
if [ ! -f "$LLAMA_SERVER_BIN" ]; then
    echo "llama-server not found at: $LLAMA_SERVER_BIN"
    echo "Please build llama.cpp first"
    exit 1
fi

echo "Starting llama-server with model: $MODEL_PATH"
echo "Server will be available at: http://localhost:8085"

# Set library path for shared libraries
export LD_LIBRARY_PATH="$LIB_PATH:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES=0
exec "$LLAMA_SERVER_BIN" \
    -m "$MODEL_PATH" \
    --port 8085 \
    --mlock \
    --host 0.0.0.0 \
    -c 32768 \
    -fa on \
    --jinja \
    --reasoning-format none \
    --n-gpu-layers 999
