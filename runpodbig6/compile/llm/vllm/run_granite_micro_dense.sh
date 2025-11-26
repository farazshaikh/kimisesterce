#!/bin/bash
# Run vLLM server with IBM Granite 4.0 Micro (DENSE/TRANSFORMER version)
# This is the pure transformer model - should be MUCH faster than Mamba2 hybrid

set -euo pipefail

MODEL_NAME="ibm-granite/granite-4.0-micro"
PORT=8080
HOST="0.0.0.0"

# Activate venv
source /compile/llm/vllm/venv/bin/activate

echo "Starting vLLM server with model: $MODEL_NAME"
echo "Server will be available at: http://localhost:$PORT"
echo "OpenAI-compatible API at: http://localhost:$PORT/v1"
echo ""
echo "Configuration: PURE TRANSFORMER (FAST!)"
echo "  - 40 attention layers (no Mamba2)"
echo "  - BF16/FP16 precision (auto-detected)"
echo "  - GPU 4 (most available memory)"
echo "  - Should be 2-3x faster than Mamba2 hybrid version"
echo ""

# Use GPU 4 which has the most free memory
export CUDA_VISIBLE_DEVICES=4

# Run vLLM with optimized settings for H200
# Pure transformer should be MUCH faster in vLLM
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_NAME" \
    --host "$HOST" \
    --port "$PORT" \
    --dtype auto \
    --gpu-memory-utilization 0.4 \
    --disable-log-requests \
    --trust-remote-code \
    --served-model-name granite-micro-dense \
    --max-model-len 8192 \
    --enable-chunked-prefill \
    --max-num-batched-tokens 8192 \
    --tensor-parallel-size 1 

