#!/bin/bash

################################################################################
# Kimi K2 Instruct Baremetal Initialization Script
# Based on: https://docs.vllm.ai/projects/recipes/en/latest/moonshotai/Kimi-K2.html
################################################################################

set -e

MODEL_NAME="moonshotai/Kimi-K2-Instruct-0905"
VENV_DIR=".venv"

echo "============================================="
echo "Kimi K2 Instruct - Baremetal Setup"
echo "============================================="

# Function to display usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --setup-only        Only setup vLLM environment (no model serving)"
    echo "  --port              Serving port (default: 8000)"
    echo "  --max-model-len     Maximum model length (default: 65536, max: 131072)"
    echo "  --gpu-mem-util      GPU memory utilization (default: 0.85, can go up to 0.95 on B200)"
    echo "  --help              Display this help message"
    echo ""
    echo "Examples:"
    echo "  # Setup environment (first time only)"
    echo "  $0 --setup-only"
    echo ""
    echo "  # Run with default settings (TP8 across all 8 GPUs)"
    echo "  $0"
    echo ""
    echo "  # Run with custom settings"
    echo "  $0 --max-model-len 131072 --gpu-mem-util 0.95 --port 9000"
    exit 1
}

# Parse arguments
SETUP_ONLY=false
PORT=8000
MAX_MODEL_LEN=65536
GPU_MEM_UTIL=0.85

while [[ $# -gt 0 ]]; do
    case $1 in
        --setup-only)
            SETUP_ONLY=true
            shift
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --max-model-len)
            MAX_MODEL_LEN="$2"
            shift 2
            ;;
        --gpu-mem-util)
            GPU_MEM_UTIL="$2"
            shift 2
            ;;
        --help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

################################################################################
# Step 1: Setup vLLM Environment
################################################################################

echo ""
echo "Step 1: Setting up vLLM environment..."
echo "----------------------------------------"

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "Error: 'uv' is not installed. Please install it first:"
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    uv venv "$VENV_DIR"
else
    echo "Virtual environment already exists at $VENV_DIR"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Install vLLM
echo "Installing vLLM..."
uv pip install -U vllm --torch-backend auto

echo "✓ vLLM installation complete"

if [ "$SETUP_ONLY" = true ]; then
    echo ""
    echo "Setup complete! Virtual environment is ready at: $VENV_DIR"
    echo "To activate it manually, run: source $VENV_DIR/bin/activate"
    exit 0
fi

################################################################################
# Step 2: Run Model - TP8 Mode (Only option for 1TB model on 8x B200)
################################################################################

# Create logs directory if it doesn't exist
LOGS_DIR="logs"
mkdir -p "$LOGS_DIR"

# Generate timestamped log filename
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOGS_DIR/kimi_k2_${TIMESTAMP}.log"

echo ""
echo "Step 2: Starting Kimi K2 Instruct..."
echo "----------------------------------------"
echo "Model: $MODEL_NAME"
echo "Mode: Tensor Parallel across 8 GPUs"
echo "Log File: $LOG_FILE"
echo ""

################################################################################
# Tensor Parallelism Mode - 8 GPUs
# This is the ONLY way to run this 1TB model on 8x B200 GPUs
# The model is too large to fit multiple copies for data parallelism
################################################################################

echo "Configuration:"
echo "  - Tensor Parallel Size: 8"
echo "  - Max Model Length: $MAX_MODEL_LEN"
echo "  - Max Num Batched Tokens: 8192"
echo "  - Max Num Seqs: 64"
echo "  - GPU Memory Utilization: $GPU_MEM_UTIL"
echo "  - Port: $PORT"
echo "  - KV Cache: auto (B200 compatibility)"
echo "  - Custom All-Reduce: disabled (fixes TMA descriptor error)"
echo ""
echo "💾 Logs will be saved to: $LOG_FILE"
echo "💡 Tip: You can monitor logs with: tail -f $LOG_FILE"
echo ""

# B200 Blackwell GPU compatibility settings
# - Disable custom all-reduce to avoid TMA descriptor errors
# - Use auto KV cache dtype instead of fp8
# - Lower batch size to reduce memory pressure
# - Disable V1 engine features that may not be stable on Blackwell

echo "Starting vLLM server..." | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"

# Use 'script' command to properly capture all output including colored logs
# Falls back to simple redirection if script command is not available
if command -v script &> /dev/null; then
    # Using script command for better output capture
    script -q -c "vllm serve \"$MODEL_NAME\" \
        --trust-remote-code \
        --tokenizer-mode auto \
        --tensor-parallel-size 8 \
        --dtype bfloat16 \
        --quantization fp8 \
        --max-model-len \"$MAX_MODEL_LEN\" \
        --max-num-seqs 64 \
        --max-num-batched-tokens 8192 \
        --enable-chunked-prefill \
        --kv-cache-dtype auto \
        --gpu-memory-utilization \"$GPU_MEM_UTIL\" \
        --enable-auto-tool-choice \
        --tool-call-parser kimi_k2 \
        --disable-custom-all-reduce \
        --port \"$PORT\"" "$LOG_FILE"
else
    # Fallback: use tee to redirect both stdout and stderr
    vllm serve "$MODEL_NAME" \
        --trust-remote-code \
        --tokenizer-mode auto \
        --tensor-parallel-size 8 \
        --dtype bfloat16 \
        --quantization fp8 \
        --max-model-len "$MAX_MODEL_LEN" \
        --max-num-seqs 64 \
        --max-num-batched-tokens 8192 \
        --enable-chunked-prefill \
        --kv-cache-dtype auto \
        --gpu-memory-utilization "$GPU_MEM_UTIL" \
        --enable-auto-tool-choice \
        --tool-call-parser kimi_k2 \
        --disable-custom-all-reduce \
        --port "$PORT" 2>&1 | tee -a "$LOG_FILE"
fi

echo ""
echo "============================================="
echo "Model serving stopped"
echo "============================================="
