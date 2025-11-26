.PHONY: install clean serve bench help

.DEFAULT_GOAL := install

VENV_PATH := .venv
PYTHON := $(VENV_PATH)/bin/python
PIP := $(VENV_PATH)/bin/pip

help:
	@echo "Available targets:"
	@echo "  make install  - Install vLLM and dependencies"
	@echo "  make serve    - Start vLLM server (Qwen3-VL-30B-A3B on GPU 5, port 8006)"
	@echo "  make bench    - Run quick benchmark with image.png"
	@echo "  make clean    - Remove virtual environment"
	@echo "  make help     - Show this help message"

$(VENV_PATH)/bin/activate:
	@echo "Creating virtual environment at $(VENV_PATH)..."
	python3 -m venv $(VENV_PATH)
	@echo "Virtual environment created."

install: $(VENV_PATH)/bin/activate
	@echo ""
	@echo "=== Installing vLLM Dependencies ==="
	@echo ""
	@echo "Installing system dependencies (ninja-build)..."
	@apt-get update -qq && apt-get install -y -qq ninja-build > /dev/null 2>&1 || true
	@echo ""
	@echo "Upgrading pip..."
	$(PIP) install --upgrade pip
	@echo ""
	@echo "Installing vllm, flashinfer-python, and hf-transfer..."
	$(PIP) install vllm flashinfer-python hf-transfer
	@echo ""
	@echo "Installing flash-attn..."
	$(PIP) install flash-attn --no-build-isolation
	@echo ""
	@echo "=== Installation Complete ==="
	@echo "To activate the virtual environment, run:"
	@echo "  source $(VENV_PATH)/bin/activate"

serve:
	@echo "Starting vLLM server on GPU 5 (143 GB free)..."
	CUDA_VISIBLE_DEVICES=5 $(VENV_PATH)/bin/vllm serve Qwen/Qwen3-VL-30B-A3B-Instruct \
		--host 0.0.0.0 \
		--port 8006 \
		--served-model-name qwen3-vl \
		--gpu-memory-utilization 0.85 \
		--max-model-len 16384 \
		--max-num-seqs 32 \
		--enable-chunked-prefill \
		--trust-remote-code \
		--dtype auto

clean:
	@echo "Removing virtual environment..."
	rm -rf $(VENV_PATH)
	@echo "Virtual environment removed."

