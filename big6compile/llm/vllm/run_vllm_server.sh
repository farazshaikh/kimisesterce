#!/bin/bash
# Run vLLM server with Qwen3-30B model (OpenAI-compatible API)

set -euo pipefail

MODEL_PATH="/wrk/llm/models/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf"
PORT=8085
HOST="0.0.0.0"

# Check if model exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "Model not found at: $MODEL_PATH"
    echo "Please download the model to: $MODEL_PATH"
    exit 1
fi

# Activate venv
source /wrk/llm/vllm/venv/bin/activate

echo "Starting vLLM server with model: $MODEL_PATH"
echo "Server will be available at: http://$HOST:$PORT"
echo "OpenAI-compatible endpoint: http://$HOST:$PORT/v1/chat/completions"
echo ""

# Start vLLM server with OpenAI-compatible API
# Note: vLLM can load GGUF files directly!
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --host "$HOST" \
    --port "$PORT" \
    --gpu-memory-utilization 0.92 \
    --max-model-len 32768 \
    --dtype auto \
    --tokenizer "Qwen/Qwen3-30B-A3B-Instruct-2507" \
    --served-model-name "Qwen3-30B-A3B-Instruct"
