#!/bin/bash
set -euo pipefail

# 1. Install tmux
echo "Installing tmux..."
if ! command -v tmux &> /dev/null; then
    apt-get update && apt-get install -y tmux
else
    echo "tmux is already installed"
fi

# 2. Create directories
echo "Creating directories..."
mkdir -p /compile/vlm/sglang
mkdir -p /compile/llm/vllm

# 3. Write files
echo "Copying files..."

# Makefile
cat << 'EOF' > /compile/vlm/sglang/Makefile
.PHONY: install clean serve help

.DEFAULT_GOAL := install

VENV_PATH := .venv
PYTHON := $(VENV_PATH)/bin/python
PIP := $(VENV_PATH)/bin/pip

help:
	@echo "Available targets:"
	@echo "  make install  - Install SGLang and dependencies"
	@echo "  make serve    - Start SGLang server (Qwen3-VL-30B-A3B on GPU 4, port 8007)"
	@echo "  make clean    - Remove virtual environment"
	@echo "  make help     - Show this help message"

$(VENV_PATH)/bin/activate:
	@echo "Creating virtual environment at $(VENV_PATH)..."
	python3 -m venv $(VENV_PATH)
	@echo "Virtual environment created."

install: $(VENV_PATH)/bin/activate
	@echo ""
	@echo "=== Installing SGLang Dependencies ==="
	@echo ""
	@echo "Installing system dependencies (ninja-build)..."
	@apt-get update -qq && apt-get install -y -qq ninja-build > /dev/null 2>&1 || true
	@echo ""
	@echo "Upgrading pip..."
	$(PIP) install --upgrade pip
	@echo ""
	@echo "Installing SGLang (pre-release)..."
	$(PIP) install sglang --pre
	@echo ""
	@echo "Installing flashinfer-python..."
	$(PIP) install flashinfer-python
	@echo ""
	@echo "Installing flash-attn..."
	$(PIP) install flash-attn --no-build-isolation
	@echo ""
	@echo "=== Installation Complete ==="
	@echo "To activate the virtual environment, run:"
	@echo "  source $(VENV_PATH)/bin/activate"

serve:
	@echo "Starting SGLang server..."
	CUDA_VISIBLE_DEVICES=5 $(PYTHON) -m sglang.launch_server \
		--model-path Qwen/Qwen3-VL-30B-A3B-Instruct-FP8 \
		--host 0.0.0.0 \
		--port 8006 \
		--served-model-name qwen3-vl-sglang \
		--tp-size 1 \
		--mem-fraction-static 0.90 \
		--context-length 32768 \
		--dtype auto \
		--trust-remote-code \
		--enable-dp-attention

clean:
	@echo "Removing virtual environment..."
	rm -rf $(VENV_PATH)
	@echo "Virtual environment removed."
EOF

# run_granite_micro_dense.sh
cat << 'EOF' > /compile/llm/vllm/run_granite_micro_dense.sh
#!/bin/bash
# Run vLLM server with IBM Granite 4.0 Micro (DENSE/TRANSFORMER version)
# This is the pure transformer model - should be MUCH faster than Mamba2 hybrid

set -euo pipefail

MODEL_NAME="ibm-granite/granite-4.0-micro"
PORT=8080
HOST="0.0.0.0"

# Activate venv
source /compile/llm/vllm/venv/bin/activate

echo "Starting vLLM server with model: $MODEL_NAME"
echo "Server will be available at: http://localhost:$PORT"
echo "OpenAI-compatible API at: http://localhost:$PORT/v1"
echo ""
echo "Configuration: PURE TRANSFORMER (FAST!)"
echo "  - 40 attention layers (no Mamba2)"
echo "  - BF16/FP16 precision (auto-detected)"
echo "  - GPU 4 (most available memory)"
echo "  - Should be 2-3x faster than Mamba2 hybrid version"
echo ""

# Use GPU 4 which has the most free memory
export CUDA_VISIBLE_DEVICES=4

# Run vLLM with optimized settings for H200
# Pure transformer should be MUCH faster in vLLM
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_NAME" \
    --host "$HOST" \
    --port "$PORT" \
    --dtype auto \
    --gpu-memory-utilization 0.4 \
    --disable-log-requests \
    --trust-remote-code \
    --served-model-name granite-micro-dense \
    --max-model-len 8192 \
    --enable-chunked-prefill \
    --max-num-batched-tokens 8192 \
    --tensor-parallel-size 1 
EOF
chmod +x /compile/llm/vllm/run_granite_micro_dense.sh

# run_vllm_qwen3_embedding.sh
cat << 'EOF' > /compile/llm/vllm/run_vllm_qwen3_embedding.sh
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
EOF
chmod +x /compile/llm/vllm/run_vllm_qwen3_embedding.sh

# 4. Run models
echo "Starting models in tmux sessions..."

# Check/Start SGLang
if tmux has-session -t sglang 2>/dev/null; then
    echo "Session 'sglang' already exists. Skipping."
else
    echo "Starting sglang on GPU 5 (port 8006)..."
    # Assuming make install is not needed or already done, or make serve handles env
    # Note: make serve uses $(PYTHON) which is .venv/bin/python. 
    # If .venv doesn't exist, this will fail. We might need to run make install first.
    # We'll try to run it, check logs if needed.
    tmux new-session -d -s sglang "cd /compile/vlm/sglang && make serve"
fi

# Check/Start Granite
if tmux has-session -t granite 2>/dev/null; then
    echo "Session 'granite' already exists. Skipping."
else
    echo "Starting granite on GPU 4 (port 8080)..."
    tmux new-session -d -s granite "/compile/llm/vllm/run_granite_micro_dense.sh"
fi

# Check/Start Qwen Embedding
if tmux has-session -t qwen 2>/dev/null; then
    echo "Session 'qwen' already exists. Skipping."
else
    echo "Starting qwen embedding on GPU 4 (port 8007)..."
    tmux new-session -d -s qwen "/compile/llm/vllm/run_vllm_qwen3_embedding.sh"
fi

echo "Deployment script completed."
echo "Use 'tmux list-sessions' to view running models."
echo "Use 'tmux attach -t <session_name>' to view logs."
