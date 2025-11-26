#!/bin/bash
# Run SGLang server with GLM-4.6-FP8 model
# GLM-4.6-FP8 is a 353B parameter model with FP8 quantization

set -euo pipefail

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

# Check if SGLang venv exists, if not create it
if [ ! -d "/compile/llm/sglang/venv" ]; then
    echo "SGLang venv not found. Creating virtual environment..."
    python3 -m venv /compile/llm/sglang/venv
    source /compile/llm/sglang/venv/bin/activate
    
    echo "Installing latest SGLang (including pre-releases)..."
    pip install --upgrade pip
    pip install sglang --pre
    
    echo "SGLang installation complete."
else
    # Activate existing venv
    source /compile/llm/sglang/venv/bin/activate
fi

echo "Starting SGLang server with GLM-4.6-FP8 model: $MODEL_PATH"
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
echo "  - Native tool use support"
echo ""
echo "SGLang optimizations:"
echo "  - RadixAttention for automatic KV cache reuse"
echo "  - Faster prefix caching for coding workflows"
echo "  - Multi-turn conversation optimization"
echo ""

# Start SGLang server with OpenAI-compatible API
# SGLang handles GLM-4.6-FP8's tool calling natively
# Use all 4 GPUs with tensor parallelism for this massive 353B parameter model
python -m sglang.launch_server \
    --model-path "$MODEL_PATH" \
    --host "$HOST" \
    --port "$PORT" \
    --tp-size 4 \
    --mem-fraction-static 0.93 \
    --context-length 131072 \
    --trust-remote-code \
    --dtype auto \
    \
    # Chunked prefill (equivalent to vLLM's optimization)
    --enable-dp-attention \
    --chunked-prefill-size 32768 \
    \
    # Tool calling
    --tool-call-parser qwen \
    --served-model-name "Qwen3-235B-A22B-Instruct-FP8"
