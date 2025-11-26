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


