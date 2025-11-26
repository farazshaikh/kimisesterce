#!/bin/bash
# Install vLLM with DeepGemm and CUTLASS optimized kernels
# This will reinstall vLLM with all optimizations enabled

set -euo pipefail

cd /compile/llm/vllm
source venv/bin/activate

echo "Installing vLLM with DeepGemm and CUTLASS optimizations..."
echo "This may take 10-20 minutes as it compiles CUDA kernels from source"
echo ""

# First install dependencies
pip install --upgrade pip setuptools wheel ninja

# Install build dependencies
pip install packaging torch

# Uninstall current vLLM
pip uninstall -y vllm

# Reinstall vLLM from source with all optimizations
# VLLM_INSTALL_PUNICA_KERNELS=1: Punica kernels for LoRA
# This will automatically compile DeepGemm and CUTLASS kernels
MAX_JOBS=8 pip install vllm --no-build-isolation

echo ""
echo "✓ Installation complete!"
echo "DeepGemm and CUTLASS kernels should now be available"
echo ""
echo "Restart your vLLM server to use the new kernels"

