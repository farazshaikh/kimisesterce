#!/bin/bash
# Run vLLM server with IBM Granite 4.0 H Micro model
# Optimized for maximum speed on RunPod

set -euo pipefail

MODEL_NAME="ibm-granite/granite-4.0-h-micro"
PORT=8080
HOST="0.0.0.0"

# Activate venv
source /compile/llm/vllm/venv/bin/activate

echo "Starting vLLM server with model: $MODEL_NAME"
echo "Server will be available at: http://localhost:$PORT"
echo "OpenAI-compatible API at: http://localhost:$PORT/v1"
echo ""
echo "Configuration: ULTRA FAST SETTINGS (H200 GPU)"
echo "  - BF16/FP16 precision (auto-detected)"
echo "  - GPU 4 (most available memory)"
echo "  - Aggressive memory utilization (0.95)"
echo "  - Optimized for low latency single requests"
echo ""

# Use GPU 4 which has the most free memory
export CUDA_VISIBLE_DEVICES=4

# Run vLLM with optimized settings for H200
# Keep it simple - vLLM defaults are usually good
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_NAME" \
    --host "$HOST" \
    --port "$PORT" \
    --dtype auto \
    --gpu-memory-utilization 0.3 \
    --disable-log-requests \
    --trust-remote-code \
    --served-model-name granite-micro

