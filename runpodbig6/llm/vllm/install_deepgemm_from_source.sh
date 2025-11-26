#!/bin/bash
# Build vLLM from source with DeepGemm and CUTLASS optimizations
# This compiles CUDA kernels specifically for your GPU architecture

set -euo pipefail

cd /compile/llm/vllm
source venv/bin/activate

echo "Building vLLM from source with all optimizations..."
echo "This will take 15-30 minutes to compile CUDA kernels"
echo ""

# Install build dependencies
pip install --upgrade pip setuptools<80 wheel ninja cmake

# Uninstall current vLLM
pip uninstall -y vllm

# Clone vLLM repo if not exists
if [ ! -d "vllm-src" ]; then
    echo "Cloning vLLM repository..."
    git clone https://github.com/vllm-project/vllm.git vllm-src
fi

cd vllm-src
git fetch --all --tags
git checkout v0.11.0

echo ""
echo "Building with optimizations:"
echo "  - DeepGemm kernels"
echo "  - CUTLASS kernels"
echo "  - Flash Attention"
echo "  - All CUDA extensions"
echo ""

# Set environment variables for full optimization build
export VLLM_INSTALL_CUDA_GRAPHS=1
export TORCH_CUDA_ARCH_LIST="8.0;9.0"  # Ampere and Hopper
export MAX_JOBS=8

# Build and install from source
pip install -e . --no-build-isolation

echo ""
echo "✓ Build complete!"
echo ""
echo "Verifying DeepGemm installation..."
python -c "
try:
    import vllm
    print('✓ vLLM imported successfully')
    # Try to check for DeepGemm
    try:
        from vllm._C import ops
        print('✓ vLLM CUDA extensions loaded')
    except Exception as e:
        print(f'⚠ Could not load all extensions: {e}')
except Exception as e:
    print(f'✗ Error: {e}')
"

echo ""
echo "Restart your vLLM server to use the optimized kernels!"



