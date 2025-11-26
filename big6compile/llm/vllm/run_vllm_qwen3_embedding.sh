#!/bin/bash
# Run vLLM server with Qwen3-Embedding-0.6B model
# This provides OpenAI-compatible embeddings API

set -euo pipefail

MODEL_PATH="Qwen/Qwen3-Embedding-0.6B"
PORT=8007
HOST="0.0.0.0"
GPU_DEVICE=4  # Using GPU 4 which has ~99GB free (embedding model needs only 2-3GB)

# Activate venv
source /compile/llm/vllm/venv/bin/activate

echo "Starting vLLM embedding server with Qwen3-Embedding-0.6B model"
echo "Model: $MODEL_PATH"
echo "GPU: $GPU_DEVICE"
echo "Memory usage: ~2-3GB (0.6B parameters)"
echo "Context window: 8192 tokens"
echo "Embedding dimensions: 768"
echo "Server will be available at: http://$HOST:$PORT"
echo "OpenAI-compatible endpoint: http://$HOST:$PORT/v1/embeddings"
echo ""
echo "Example usage with OpenAI client:"
echo "  from openai import OpenAI"
echo "  client = OpenAI(base_url='http://localhost:$PORT/v1', api_key='dummy')"
echo "  response = client.embeddings.create(model='$MODEL_PATH', input='Hello world')"
echo ""

# Start vLLM server with OpenAI-compatible API for embeddings
# Use --task embed for embedding models
# --pooling-type last uses last token pooling (similar to --pooling last in llama-server)
export CUDA_VISIBLE_DEVICES=$GPU_DEVICE

python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --host "$HOST" \
    --port "$PORT" \
    --task embed \
    --max-model-len 8192 \
    --runner "pooling" \
    --dtype auto \
    --gpu-memory-utilization 0.1 \
    --trust-remote-code \
    --served-model-name "Qwen3-Embedding-0.6B"

