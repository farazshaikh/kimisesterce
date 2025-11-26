# SGLang LLM Server Scripts

This directory contains scripts to run LLM servers using [SGLang](https://github.com/sgl-project/sglang) - a fast serving framework for large language models.

## What is SGLang?

SGLang is an optimized serving framework for LLMs that offers:
- **RadixAttention**: Automatic KV cache reuse for improved efficiency
- **Faster prefix caching**: Better performance for multi-turn conversations
- **Higher throughput**: Up to 5x faster than vLLM in certain workloads
- **Native multi-modal support**: Built-in support for vision and audio models
- **OpenAI-compatible API**: Drop-in replacement for OpenAI API

## Differences from vLLM

| Feature | vLLM | SGLang |
|---------|------|--------|
| KV Cache | Basic caching | RadixAttention (automatic reuse) |
| Prefix Caching | Manual configuration | Automatic |
| Multi-turn Performance | Good | Excellent |
| Tool Calling | Requires parser config | Native support |
| Command | `vllm.entrypoints.openai.api_server` | `sglang.launch_server` |

### Command-line Flag Differences

| vLLM Flag | SGLang Flag | Purpose |
|-----------|-------------|---------|
| `--tensor-parallel-size` | `--tp-size` or `--tp` | Number of GPUs for tensor parallelism |
| `--gpu-memory-utilization` | `--mem-fraction-static` | GPU memory fraction to use |
| `--max-model-len` | `--context-length` | Maximum context length |
| `--enable-chunked-prefill` | `--enable-dp-attention` | Enable chunked attention |
| `--tool-call-parser hermes` | *(built-in)* | Tool calling support |

## Available Scripts

### `run_sglang_qwen3_235b.sh`
Runs the Qwen3-235B-A22B-Instruct-FP8 model with SGLang.

**Features:**
- 235B parameters with FP8 quantization (~120-150GB)
- 256K token context window (native, expandable to 1M)
- 4-GPU tensor parallelism
- OpenAI-compatible API on port 8084

**Usage:**
```bash
./run_sglang_qwen3_235b.sh
```

**First Run:**
The script will automatically:
1. Create a Python virtual environment at `/compile/llm/sglang/venv`
2. Install the latest SGLang with FlashInfer backend
3. Start the server

**API Endpoints:**
- Chat completions: `http://localhost:8084/v1/chat/completions`
- Models list: `http://localhost:8084/v1/models`

## Installation

The scripts automatically handle installation on first run. If you want to manually install SGLang:

```bash
python3 -m venv /compile/llm/sglang/venv
source /compile/llm/sglang/venv/bin/activate
pip install --upgrade pip
pip install sglang --pre
```

## When to Use SGLang vs vLLM

**Use SGLang when:**
- You have multi-turn conversations with repeated prefixes
- You need maximum throughput for chat applications
- You want automatic KV cache optimization
- You're building chatbots or agents with long conversations

**Use vLLM when:**
- You need maximum compatibility with HuggingFace models
- You're doing one-off completions without caching benefits
- You need specific vLLM-only features

## Performance Tips

1. **Context Length**: Start with smaller context lengths (e.g., 16384) for better throughput, increase as needed
2. **Memory Fraction**: Adjust `--mem-fraction-static` based on your GPU memory (0.85-0.95)
3. **Tensor Parallelism**: Use `--tp-size` equal to number of GPUs for large models
4. **Data Parallelism**: For high traffic, consider using `--dp-size` for data parallelism

## Testing the Server

Once running, test with curl:

```bash
curl http://localhost:8084/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3-235B-A22B-Instruct-FP8",
    "messages": [
      {"role": "user", "content": "Hello! How are you?"}
    ],
    "temperature": 0.7
  }'
```

## Resources

- [SGLang GitHub](https://github.com/sgl-project/sglang)
- [SGLang Documentation](https://sgl-project.github.io/)
- [Qwen3 Model Card](https://huggingface.co/Qwen/Qwen3-235B-A22B-Instruct-FP8)

