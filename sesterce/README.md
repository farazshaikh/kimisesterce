# Kimi K2 Instruct on 8x B200 GPUs - Quick Start

## 🚀 Fastest Way to Get Started

```bash
# 1. Setup (first time only)
./init_baremetal.sh --setup-only

# 2. Run the model
./init_baremetal.sh
```

That's it! The API will be available at `http://localhost:8000`

> **⚠️ B200 Note**: The script includes special compatibility settings for Blackwell GPUs (disabled custom all-reduce, auto KV cache) to avoid TMA descriptor errors.

> **💾 Logs**: All output is automatically saved to `logs/kimi_k2_TIMESTAMP.log` for debugging and monitoring.

---

## 📖 Understanding Your Setup

**Hardware:**
- **8x NVIDIA B200 GPUs** with 192GB each = **1.5TB total VRAM**

**Model:**
- **Kimi K2 Instruct** - ~1TB model size
- Uses FP8 quantization for efficiency

**Deployment Mode:**
- **TP8 (Tensor Parallel across 8 GPUs)** - This is the ONLY way to run this model
- The model is too large (~1TB) to fit multiple copies on your hardware
- All 8 GPUs work together to handle the model

---

## ⚙️ Customization Options

### Change Context Window

```bash
# Support up to 128k tokens (max)
./init_baremetal.sh --max-model-len 131072

# Default (65k tokens)
./init_baremetal.sh --max-model-len 65536

# Lower memory (32k tokens)
./init_baremetal.sh --max-model-len 32768
```

### Adjust Batch Size (Throughput vs Memory)

```bash
# Higher throughput (more memory usage)
./init_baremetal.sh --max-batch-tokens 65536

# Default (balanced - 32K tokens)
./init_baremetal.sh --max-batch-tokens 32768

# Lower memory usage
./init_baremetal.sh --max-batch-tokens 16384
```

### Use More GPU Memory

```bash
# B200s can handle 95% utilization safely
./init_baremetal.sh --gpu-mem-util 0.95

# Conservative (default)
./init_baremetal.sh --gpu-mem-util 0.85
```

### Change Port

```bash
./init_baremetal.sh --port 9000
```

### Combine Options

```bash
./init_baremetal.sh \
  --max-model-len 131072 \
  --max-batch-tokens 65536 \
  --gpu-mem-util 0.95 \
  --port 8000
```

---

## 🧪 Test the API

Once running, try:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "moonshotai/Kimi-K2-Instruct-0905",
    "messages": [
      {"role": "user", "content": "Explain quantum computing in simple terms"}
    ],
    "max_tokens": 500
  }'
```

**With Tool Calling:**

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "moonshotai/Kimi-K2-Instruct-0905",
    "messages": [
      {"role": "user", "content": "What is the weather in San Francisco?"}
    ],
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "get_weather",
          "description": "Get weather information",
          "parameters": {
            "type": "object",
            "properties": {
              "location": {"type": "string"}
            }
          }
        }
      }
    ]
  }'
```

---

## 📊 What to Expect (8x B200 GPUs)

| Metric | Value |
|--------|-------|
| Time to First Token | 2-4 seconds |
| Output Speed | 350-450 tokens/sec |
| Context Window | 65k tokens (default) |
| Max Context | 128k tokens (configurable) |
| Concurrent Requests | Good for 1-20 |
| Total Throughput | ~3,000-4,000 tok/s |

*Note: B200 performance is estimated to be ~50-70% better than H800 benchmarks in the docs*

---

## 📊 Monitoring Logs

All vLLM output is automatically logged to timestamped files in the `logs/` directory.

### View logs in real-time:
```bash
# Monitor the latest log file
tail -f logs/kimi_k2_*.log

# Or find and follow the most recent log
tail -f $(ls -t logs/kimi_k2_*.log | head -1)
```

### Search logs for errors:
```bash
# Find CUDA errors
grep -i "cuda error" logs/kimi_k2_*.log

# Find out of memory errors
grep -i "out of memory" logs/kimi_k2_*.log

# Find worker failures
grep -i "worker.*died" logs/kimi_k2_*.log
```

### Log file location:
- **Directory**: `logs/`
- **Format**: `kimi_k2_YYYYMMDD_HHMMSS.log`
- **Example**: `logs/kimi_k2_20251114_014639.log`

### Clean up old logs:
```bash
# Remove logs older than 7 days
find logs/ -name "kimi_k2_*.log" -mtime +7 -delete

# Or remove all logs
rm -rf logs/*.log
```

---

## ❓ Troubleshooting

### "TMA descriptor error" or "illegal memory access"
```bash
# ✅ FIXED in current version!
# The script now includes --disable-custom-all-reduce for B200 compatibility
# If you still see this error:
# 1. Check the logs: tail -f logs/kimi_k2_*.log
# 2. See detailed guide: cat B200_TROUBLESHOOTING.md
```

### "Out of Memory"
```bash
# Reduce context length
./init_baremetal.sh --max-model-len 32768

# Or lower GPU memory utilization
./init_baremetal.sh --gpu-mem-util 0.80
```

### "Model not found"
```bash
# First run downloads the model (~100GB+ for FP8 weights)
# Ensure you have:
# - Internet connection
# - ~200GB free space in ~/.cache/huggingface/
```

### "Connection refused"
```bash
# Model takes 5-7 minutes to load after starting
# Wait for the startup process to complete
# Look for "Application startup complete" in the logs
```

### Check Available GPUs
```bash
nvidia-smi

# Should show 8x NVIDIA B200 GPUs
```

### More Detailed Troubleshooting
```bash
cat B200_TROUBLESHOOTING.md
```

---

## 📁 Files in This Repository

- **`init_baremetal.sh`** - Main script to run everything
- **`README.md`** - This quick start guide
- **`B200_TROUBLESHOOTING.md`** - Detailed B200/Blackwell GPU troubleshooting
- **`logs/`** - Directory containing timestamped log files
- **`.gitignore`** - Git ignore file (excludes logs and cache)

---

## 💡 Why Only One Mode?

**Q: Why can't I use Data Parallelism?**  
A: The Kimi K2 model is ~1TB in size. Even with FP8 quantization and 8x 192GB GPUs (1.5TB total), you can only fit ONE copy of the model across all 8 GPUs. Data parallelism would require multiple copies, which is impossible with this hardware.

**Q: What about Pipeline Parallelism?**  
A: Pipeline parallelism (splitting layers vertically) with TP would require 16 GPUs minimum (TP8 + PP2). You only have 8 GPUs.

**Q: Can I increase throughput?**  
A: Yes! Adjust these parameters:
- `--max-num-batched-tokens` (higher = more throughput, more memory)
- `--max-num-seqs` (max concurrent requests)
- `--gpu-mem-util` (use more VRAM for KV cache)

---

## 🔗 Useful Links

- [vLLM Kimi K2 Documentation](https://docs.vllm.ai/projects/recipes/en/latest/moonshotai/Kimi-K2.html)
- [Kimi K2 Model on HuggingFace](https://huggingface.co/moonshotai/Kimi-K2-Instruct-0905)
- [vLLM Documentation](https://docs.vllm.ai/)

---

## 📝 Need Help?

Run the help command:
```bash
./init_baremetal.sh --help
```

---

## 🎉 Summary

**For 8x B200 GPUs with 1TB Kimi K2 model:**

```bash
# Setup once
./init_baremetal.sh --setup-only

# Run with optimal settings  
./init_baremetal.sh --max-batch-tokens 65536 --gpu-mem-util 0.95

# Access at http://localhost:8000
```

**That's it! You're now running Kimi K2 Instruct on your 8x B200 GPUs!** 🚀

---

## ⚡ Performance Tips

1. **Maximize GPU Memory**: Use `--gpu-mem-util 0.95` for larger KV cache
2. **Adjust Context**: Use `--max-model-len 32768` if you don't need 65k/128k context
3. **Tune Batch Size**: 
   - Default 32K tokens balances throughput and memory
   - Increase to 65K for maximum throughput (more VRAM needed)
   - Decrease to 16K to reduce memory pressure
4. **Monitor GPUs**: Run `watch -n 1 nvidia-smi` to monitor GPU utilization
