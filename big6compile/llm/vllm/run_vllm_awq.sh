#!/bin/bash
# Run vLLM server with Qwen3-30B-A3B-Instruct-2507-AWQ-4bit model
# AWQ is a well-supported quantization format in vLLM

set -euo pipefail

MODEL_PATH="/wrk/llm/models/vllm/Qwen3-30B-A3B-Instruct-2507-AWQ"
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

echo "Starting vLLM server with AWQ 4-bit model: $MODEL_PATH"
echo "Model size: ~18GB (AWQ 4-bit quantized)"
echo "Server will be available at: http://$HOST:$PORT"
echo "OpenAI-compatible endpoint: http://$HOST:$PORT/v1/chat/completions"
echo ""
echo "AWQ quantization is production-ready and well-supported by vLLM!"
echo ""

# Start vLLM server with OpenAI-compatible API
# AWQ models now use compressed-tensors format - let vLLM auto-detect
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --host "$HOST" \
    --port "$PORT" \
    --gpu-memory-utilization 0.9 \
    --max-model-len 16384 \
    --dtype auto \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --served-model-name "Qwen3-30B-A3B-Instruct-AWQ"
