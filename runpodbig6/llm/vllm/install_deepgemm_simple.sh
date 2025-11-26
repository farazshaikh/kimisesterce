#!/bin/bash
# Install DeepGEMM from DeepSeek AI
# Clean and efficient FP8 GEMM kernels for vLLM

set -euo pipefail

cd /compile/llm/vllm
source venv/bin/activate

echo "Installing DeepGEMM from DeepSeek AI..."
echo "Repo: https://github.com/deepseek-ai/DeepGEMM"
echo ""

# Clone the repo
if [ ! -d "DeepGEMM" ]; then
    echo "Cloning DeepGEMM repository..."
    git clone --recursive https://github.com/deepseek-ai/DeepGEMM.git
fi

cd DeepGEMM

echo "Building and installing DeepGEMM..."
echo "This will take 5-10 minutes to compile CUDA kernels"
echo ""

# Run the installation script
chmod +x install.sh
./install.sh

echo ""
echo "✓ DeepGEMM installation complete!"
echo ""
echo "Testing installation..."
python -c "
try:
    import deep_gemm
    print('✓ deep_gemm imported successfully')
    print(f'  Available functions: {dir(deep_gemm)}')
except Exception as e:
    print(f'✗ Import failed: {e}')
"

echo ""
echo "Now restart your vLLM server!"
echo "The 'Failed to import DeepGemm kernels' warning should be gone."



