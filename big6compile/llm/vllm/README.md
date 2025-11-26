# vLLM Setup for Qwen3-30B Models

## 🚀 Installation Steps

### Prerequisites
- **Operating System**: Linux
- **Python Version**: 3.9 to 3.12
- **GPU**: NVIDIA GPU with compute capability 7.0 or higher
- **CUDA Version**: 12.1
- **Memory**: ~18GB VRAM + overhead

### Install vLLM

1. **Create and activate virtual environment:**
```bash
cd /compile/llm/vllm
python3 -m venv venv
source venv/bin/activate
```

2. **Upgrade pip and install vLLM with optimizations:**
```bash
pip install --upgrade pip
pip install vllm flashinfer-python hf-transfer
pip install flash-attn --no-build-isolation
```

3. **Verify installation:**
```bash
python -c "import vllm; print('✅ vLLM installed successfully!')"
python -c "import flashinfer; print('✅ FlashInfer installed successfully!')"
python -c "import flash_attn; print('✅ Flash Attention installed successfully!')"
```

### Package Details
- **vLLM**: High-performance LLM inference engine
- **flashinfer-python**: Optimized inference kernels for attention mechanisms
- **hf-transfer**: Fast model downloads from HuggingFace Hub
- **flash-attn**: Memory-efficient attention implementation

### Quick Activation
```bash
# Activate the virtual environment
source /compile/llm/vllm/activate.sh
```

## 📥 Downloading Models

### **Method 1: Using HuggingFace CLI (Recommended - Fastest)**

The `hf` CLI tool is the fastest way to download models:

```bash
# Download GLM-4.6-FP8 (353B parameters, ~200GB)
hf download zai-org/GLM-4.6-FP8

# Download to specific directory
hf download zai-org/GLM-4.6-FP8 --local-dir /compile/llm/models/vllm/GLM-4.6-FP8

# Download Qwen3-30B AWQ model
hf download cpatonn/Qwen3-30B-A3B-Instruct-2507-AWQ-4bit --local-dir /wrk/llm/models/vllm/Qwen3-30B-A3B-Instruct-2507-AWQ

# Download GGUF model
wget https://huggingface.co/TheBloke/Qwen3-30B-A3B-Instruct-2507-GGUF/resolve/main/qwen3-30b-a3b-instruct-2507-q4_k_m.gguf -O /wrk/llm/models/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf
```

### **Method 2: Using Python (huggingface_hub)**

For programmatic downloads:

```bash
cd /compile/llm/vllm
source venv/bin/activate

python -c "
from huggingface_hub import snapshot_download
snapshot_download('zai-org/GLM-4.6-FP8')
"
```

### **Method 3: Using Git Clone + LFS**

For development/experimentation:

```bash
# Install git-lfs (if not installed)
apt install -y git-lfs
git lfs install

# Clone the model repository
cd /compile/llm/models/vllm
git clone https://huggingface.co/zai-org/GLM-4.6-FP8
cd GLM-4.6-FP8
git lfs pull
```

### **Download Tips:**
- **HuggingFace CLI** (`hf download`) is fastest and most reliable
- Models are cached in `~/.cache/huggingface/hub/` by default
- Use `--local-dir` to specify custom download location
- Large models like GLM-4.6-FP8 can take 1-2 hours to download
- Ensure sufficient disk space (~200GB for GLM-4.6-FP8)

## 📦 Installed Models

### 1. **Qwen3-30B-A3B-Instruct-2507-AWQ** ⭐ (RECOMMENDED)
- **Location**: `/wrk/llm/models/vllm/Qwen3-30B-A3B-Instruct-2507-AWQ/`
- **Size**: ~18GB
- **Format**: Safetensors (AWQ 4-bit quantization)
- **Source**: [HuggingFace - cpatonn/Qwen3-30B-A3B-Instruct-2507-AWQ-4bit](https://huggingface.co/cpatonn/Qwen3-30B-A3B-Instruct-2507-AWQ-4bit)
- **Use Case**: **Production-ready for vLLM** - Best balance of speed and quality
- **Run**: `./run_vllm_awq.sh`

### 2. **Qwen3-30B-A3B-Instruct-2507-FP4** (Experimental - NOT WORKING)
- **Location**: `/wrk/llm/models/vllm/Qwen3-30B-A3B-Instruct-2507-FP4/`
- **Size**: ~18GB
- **Format**: Safetensors (FP4 quantized by NVIDIA ModelOpt)
- **Source**: [HuggingFace - NVFP4/Qwen3-30B-A3B-Instruct-2507-FP4](https://huggingface.co/NVFP4/Qwen3-30B-A3B-Instruct-2507-FP4)
- **Status**: ❌ Experimental format not compatible with vLLM 0.11.0
- **Run**: ~~`./run_vllm_fp4.sh`~~ (fails with weight mismatch)

### 3. **Qwen3-30B-A3B-Instruct-2507-Q4_K_M** (GGUF)
- **Location**: `/wrk/llm/models/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf`
- **Size**: ~19GB
- **Format**: GGUF (Q4_K_M quantization)
- **Use Case**: For llama.cpp compatibility
- **Run**: `/wrk/llm/run_local_llama.sh` (llama.cpp) or `./run_vllm_server.sh` (vLLM)

### 4. **GLM-4.6-FP8** 🚀 (MASSIVE - 353B Parameters)
- **Location**: `/workspace/.cache/huggingface/hub/models--zai-org--GLM-4.6-FP8/snapshots/dd30e1e9e5a3ac9bd16164f969b6f066c652a7e1`
- **Size**: ~200GB (353B parameters, FP8 quantized)
- **Format**: Safetensors (FP8 quantization by Z.ai)
- **Source**: [HuggingFace - zai-org/GLM-4.6-FP8](https://huggingface.co/zai-org/GLM-4.6-FP8)
- **GPUs**: Requires 4 GPUs with tensor parallelism
- **Memory**: ~83GB per GPU
- **Context**: 200K tokens
- **Features**: Superior coding, advanced reasoning, tool use, agent capabilities
- **Load Time**: ~50 minutes (one-time startup)
- **Run**: `./run_vllm_glm_4.6_fp8.sh`
- **Port**: 8083

## 🚀 Quick Start

### Option 1: Run GLM-4.6-FP8 Model (353B Parameters - 4 GPUs) 🚀
```bash
cd /compile/llm/vllm
./run_vllm_glm_4.6_fp8.sh
```
**Note**: This is a massive model that requires 4 GPUs and takes ~50 minutes to load initially.

### Option 2: Run AWQ Model with vLLM ⭐ (Recommended for single GPU)
```bash
cd /wrk/llm/vllm
./run_vllm_awq.sh
```

### Option 3: Run GGUF Model with vLLM
```bash
cd /wrk/llm/vllm
./run_vllm_server.sh
```

### Option 4: Run GGUF Model with llama.cpp
```bash
cd /wrk/llm
./run_local_llama.sh
```

## 📊 Model Comparison

| Feature | GLM-4.6-FP8 | Qwen3-30B AWQ | Q4_K_M (GGUF) |
|---------|-------------|---------------|---------------|
| Parameters | 353B | 30B | 30B |
| Size | ~200 GB | 18.1 GB | 19 GB |
| Format | Safetensors FP8 | Safetensors AWQ | GGUF |
| GPUs Required | 4 | 1 | 1 |
| Memory per GPU | 83 GB | 18 GB | 19 GB |
| Context Window | 200K tokens | 32K tokens | 32K tokens |
| Load Time | ~50 min | ~2-3 min | ~2-3 min |
| Coding | 🏆 Superior | 🎯 Excellent | 🎯 Excellent |
| Reasoning | 🏆 Advanced | ✓ Good | ✓ Good |
| Tool Use | ✅ Yes | ✅ Yes | ❌ No |
| Best For | Production/Research | Production | Development |

## 💡 Why Use FP4 Model?

According to [NVIDIA's ModelOpt](https://arxiv.org/abs/2505.09388), FP4 quantization:
- **Better GPU utilization** with vLLM's PagedAttention
- **Faster inference** through optimized CUDA kernels
- **Lower memory footprint** - fits more context in VRAM
- **Native vLLM support** - no format conversion needed

## 🔧 Integration with OSRS

Both models are **OpenAI API compatible** at `http://localhost:8085/v1/chat/completions`

Your OSRS system will work with either model without code changes:
- `os-assistant` ✅
- `os-intent` ✅
- `os-memory` ✅

Just set: `export LLAMACPP_SERVER_URL=http://localhost:8085`

## 🎮 Testing

### Test GLM-4.6-FP8 Model
```bash
cd /compile/llm/vllm
source venv/bin/activate

# Quick test
curl -X POST http://localhost:8083/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "GLM-4.6-FP8",
    "messages": [{"role": "user", "content": "Write a Python function to calculate fibonacci numbers"}],
    "max_tokens": 200
  }'
```

### Test Qwen3-30B AWQ Model
```bash
cd /wrk/llm/vllm
source venv/bin/activate

# Quick test
curl http://localhost:8085/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3-30B-A3B-Instruct-AWQ",
    "messages": [{"role": "user", "content": "Say hello!"}],
    "max_tokens": 50
  }'
```

## 📝 Notes

- **GPU Required**: All models need NVIDIA GPU with Secure Boot disabled
- **Ports**: 
  - GLM-4.6-FP8: 8083
  - Qwen3-30B models: 8085
- **Context Length**: 
  - GLM-4.6-FP8: 200K tokens
  - Qwen3-30B: 32K tokens
- **Memory Requirements**:
  - GLM-4.6-FP8: 4 GPUs × 83GB = 332GB total VRAM
  - Qwen3-30B: 1 GPU × 18-19GB VRAM
- **Multi-GPU Setup**: GLM-4.6-FP8 uses tensor parallelism across 4 GPUs

## 🔄 Switching Between Models

Just stop one server and start another - they all use the same API endpoint!
