#!/bin/bash
# Run vLLM server with Qwen3-VL-235B-A22B-Instruct-FP8 model
# Qwen3-VL-235B-A22B-Instruct-FP8 is a 235B parameter vision-language model with FP8 quantization

set -euo pipefail

MODEL_PATH="/compile/llm/models/vllm/Qwen3-VL-235B-A22B-Instruct-FP8"
PORT=8083
HOST="0.0.0.0"

# Check if model exists
if [ ! -d "$MODEL_PATH" ]; then
    echo "Model not found at: $MODEL_PATH"
    echo "Please download the model first using:"
    echo "  ./download_qwen3_vl.sh"
    exit 1
fi

# Activate venv
source /compile/llm/vllm/venv/bin/activate

echo "Starting vLLM server with Qwen3-VL-235B-A22B-Instruct-FP8 model: $MODEL_PATH"
echo "Model size: ~120-150GB (235B parameters, FP8 quantized)"
echo "Context window: 256K tokens (native)"
echo "Server will be available at: http://$HOST:$PORT"
echo "OpenAI-compatible endpoint: http://$HOST:$PORT/v1/chat/completions"
echo ""
echo "Qwen3-VL-235B features:"
echo "  - 235B parameters with FP8 quantization"
echo "  - 256K token context window (expandable to 1M)"
echo "  - Vision-language capabilities (image and video understanding)"
echo "  - Visual agent capabilities"
echo "  - Advanced spatial perception and reasoning"
echo "  - OCR support for 32 languages"
echo "  - Hermes tool calling support"
echo ""

# Start vLLM server with OpenAI-compatible API
# Qwen3-VL uses FP8 quantization - let vLLM auto-detect
# Use all 4 GPUs with tensor parallelism for this 235B parameter vision-language model
# Note: adjust gpu-memory-utilization and max-model-len based on your GPU memory
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --host "$HOST" \
    --port "$PORT" \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.92 \
    --max-model-len 8192 \
    --dtype auto \
    --trust-remote-code \
    --enable-chunked-prefill \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --served-model-name "Qwen3-VL-235B-A22B-Instruct-FP8"

