#!/bin/bash
# Run vLLM server with GLM-4.6-FP8 model
# GLM-4.6-FP8 is a 353B parameter model with FP8 quantization

set -euo pipefail

#MODEL_PATH="/workspace/.cache/huggingface/hub/models--zai-org--GLM-4.6-FP8/snapshots/dd30e1e9e5a3ac9bd16164f969b6f066c652a7e1"
MODEL_PATH="/compile/llm/models/vllm/GLM-4.6-FP8"
PORT=8083
HOST="0.0.0.0"

# Check if model exists
if [ ! -d "$MODEL_PATH" ]; then
    echo "Model not found at: $MODEL_PATH"
    echo "Please download the model first using:"
    echo "  python -c \"from huggingface_hub import snapshot_download; snapshot_download('zai-org/GLM-4.6-FP8')\""
    exit 1
fi

# Activate venv
source /compile/llm/vllm/venv/bin/activate

echo "Starting vLLM server with GLM-4.6-FP8 model: $MODEL_PATH"
echo "Model size: ~200-300GB (353B parameters, FP8 quantized)"
echo "Context window: 200K tokens"
echo "Server will be available at: http://$HOST:$PORT"
echo "OpenAI-compatible endpoint: http://$HOST:$PORT/v1/chat/completions"
echo ""
echo "GLM-4.6-FP8 features:"
echo "  - 353B parameters with FP8 quantization"
echo "  - 200K token context window"
echo "  - Superior coding performance"
echo "  - Advanced reasoning capabilities"
echo "  - Tool use support"
echo ""

# Start vLLM server with OpenAI-compatible API
# GLM-4.6-FP8 uses FP8 quantization - let vLLM auto-detect
# Use all 4 GPUs with tensor parallelism for this massive 353B parameter model
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --host "$HOST" \
    --port "$PORT" \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 16384 \
    --dtype auto \
    --enable-chunked-prefill \
    --enable-auto-tool-choice \
    --tool-call-parser glm45 \
    --served-model-name "GLM-4.6-FP8" \
    --enable-log-requests
