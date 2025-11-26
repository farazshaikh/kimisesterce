#!/bin/bash
# Run SGLang server with Qwen3-235B-A22B-Instruct-FP8 model
# Qwen3-235B-A22B-Instruct-FP8 is a 235B parameter text-only model with FP8 quantization

set -euo pipefail

MODEL_PATH="/compile/llm/models/vllm/Qwen3-235B-A22B-Instruct-2507-FP8"
PORT=8083
HOST="0.0.0.0"

# Check if model exists
if [ ! -d "$MODEL_PATH" ]; then
    echo "Model not found at: $MODEL_PATH"
    echo "Please download the model first using:"
    echo "  huggingface-cli download Qwen/Qwen3-235B-A22B-Instruct-FP8 --local-dir $MODEL_PATH"
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

echo "Starting SGLang server with Qwen3-235B-A22B-Instruct-FP8 model: $MODEL_PATH"
echo "Model size: ~120-150GB (235B parameters, FP8 quantized)"
echo "Context window: 256K tokens (native)"
echo "Server will be available at: http://$HOST:$PORT"
echo "OpenAI-compatible endpoint: http://$HOST:$PORT/v1/chat/completions"
echo ""
echo "Qwen3-235B features:"
echo "  - 235B parameters with FP8 quantization"
echo "  - 256K token context window (expandable to 1M)"
echo "  - Text-only instruction following"
echo "  - Advanced reasoning and analysis"
echo "  - Multilingual support"
echo "  - Native tool calling support"
echo ""
echo "SGLang optimizations:"
echo "  - RadixAttention for automatic KV cache reuse"
echo "  - LPM scheduling: optimized for shared system prompts"
echo "  - FlashInfer attention backend (Flash Attention 2)"
echo "  - Chunked prefill and batching enabled"
echo "  - Max context: 131K tokens, batch: 32K tokens"
echo ""

# Start SGLang server with OpenAI-compatible API
# SGLang uses different flags compared to vLLM:
# - --tp-size instead of --tensor-parallel-size
# - --mem-fraction-static instead of --gpu-memory-utilization
# - --context-length instead of --max-model-len (matched: 131K)
# - --enable-dp-attention for chunked prefill
# - --schedule-policy lpm for optimal cache reuse with shared system prompts
# - --max-total-tokens instead of --max-num-batched-tokens (matched: 32K)
# - --attention-backend flashinfer for Flash Attention 2 (FA3 requires separate install)
# - Native tool calling support without additional parsers
python -m sglang.launch_server \
    --model-path "$MODEL_PATH" \
    --host "$HOST" \
    --port "$PORT" \
    --tp-size 4 \
    --mem-fraction-static 0.92 \
    --context-length 131072 \
    --trust-remote-code \
    --enable-dp-attention \
    --schedule-policy lpm \
    --max-running-requests 256 \
    --max-total-tokens 32768 \
    --tool-call-parser qwen \
    --served-model-name "Qwen3-235B-A22B-Instruct-FP8" \
    --log-requests

