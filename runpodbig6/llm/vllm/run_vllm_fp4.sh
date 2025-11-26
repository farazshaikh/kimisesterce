#!/bin/bash
# Run vLLM server with Qwen3-30B-A3B-Instruct-2507-FP4 model
# This is a FP4 quantized model from NVIDIA - much smaller and faster than Q4_K_M

set -euo pipefail

MODEL_PATH="/wrk/llm/models/vllm/Qwen3-30B-A3B-Instruct-2507-FP4"
PORT=8085
HOST="0.0.0.0"

# Check if model exists
if [ ! -d "$MODEL_PATH" ]; then
    echo "Model not found at: $MODEL_PATH"
    echo "Please download the model first"
    exit 1
fi

# Activate venv
source /wrk/llm/vllm/venv/bin/activate

echo "Starting vLLM server with FP4 model: $MODEL_PATH"
echo "Model size: ~18GB (FP4 quantized)"
echo "Server will be available at: http://$HOST:$PORT"
echo "OpenAI-compatible endpoint: http://$HOST:$PORT/v1/chat/completions"
echo ""

# Start vLLM server with OpenAI-compatible API
# FP4 model should load faster and use less GPU memory than GGUF Q4_K_M
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --host "$HOST" \
    --port "$PORT" \
    --gpu-memory-utilization 0.85 \
    --max-model-len 32768 \
    --dtype auto \
    --quantization fp4 \
    --served-model-name "Qwen3-30B-A3B-Instruct-FP4"
