#!/bin/bash
# Run vLLM server with Qwen3-235B-A22B-Instruct-FP8 model
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

# Activate venv
source /compile/llm/vllm/venv/bin/activate

# Set library path for DeepGEMM to find PyTorch libraries
export LD_LIBRARY_PATH="/compile/llm/vllm/venv/lib/python3.12/site-packages/torch/lib:$LD_LIBRARY_PATH"

echo "Starting vLLM server with Qwen3-235B-A22B-Instruct-FP8 model: $MODEL_PATH"
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
echo "  - Hermes tool calling support"
echo ""

# Enable detailed logging for performance analysis (commented out to avoid startup issues)
# export VLLM_LOGGING_CONFIG_PATH="/compile/llm/vllm/vllm_logging_config.json"

echo "Default vLLM logging enabled (shows throughput metrics every 10s)"
echo "Logs will show:"
echo "  - Aggregate throughput metrics every 10s"
echo "  - Prefix cache hit rates"
echo "  - Running/waiting requests"
echo ""

# Start vLLM server with OpenAI-compatible API
# vLLM's detailed logging is enabled for performance analysis
# Qwen3 uses FP8 quantization - let vLLM auto-detect
# Use all 4 GPUs with tensor parallelism for this 235B parameter model
# Note: adjust gpu-memory-utilization and max-model-len based on your GPU memory
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --host "$HOST" \
    --port "$PORT" \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.90 \
    --max-model-len 131072 \
    --dtype auto \
    --quantization fp8 \
    --kv-cache-dtype fp8 \
    --trust-remote-code \
    --enable-chunked-prefill \
    --enable-prefix-caching \
    --max-num-batched-tokens 32768 \
    --max-num-seqs 128 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --disable-custom-all-reduce \
    --served-model-name "Qwen3-235B-A22B-Instruct-FP8"


