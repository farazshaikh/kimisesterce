# B200 (Blackwell) GPU Troubleshooting Guide

## The Error You Encountered

```
Error: Failed to initialize the TMA descriptor 700
Failed: Cuda error /workspace/csrc/custom_all_reduce.cuh:455 'an illegal memory access was encountered'
```

This is a **Blackwell architecture compatibility issue** with vLLM's custom all-reduce operations.

---

## ✅ FIX APPLIED

The script now includes **B200-specific compatibility settings**:

1. **`--disable-custom-all-reduce`** - Disables the problematic TMA descriptor operations
2. **`--kv-cache-dtype auto`** - Uses stable KV cache instead of FP8 (which may not be fully tested on B200)
3. **Lower batch sizes** - Reduced from 16384 to 8192 tokens (more conservative)
4. **Lower concurrency** - Reduced from 128 to 64 sequences (less memory pressure)

---

## 🚀 Try Running Again

```bash
./init_baremetal.sh
```

The updated script should work now!

---

## 🔍 If You Still Have Issues

### Option 1: Try without FP8 Quantization

If the error persists, the FP8 quantization might not be fully compatible with B200 yet:

```bash
# Temporarily edit the script to remove --quantization fp8
# This will use more memory but should be more stable
```

### Option 2: Lower GPU Memory Utilization

```bash
# Use less GPU memory (more conservative)
./init_baremetal.sh --gpu-mem-util 0.75
```

### Option 3: Reduce Context Length

```bash
# Use smaller context window
./init_baremetal.sh --max-model-len 32768
```

### Option 4: Check vLLM Version

B200 support might require the latest vLLM version:

```bash
source .venv/bin/activate
pip show vllm

# Update to latest if needed
pip install -U vllm --torch-backend auto
```

---

## 🐛 Common B200 Issues

### 1. TMA Descriptor Errors
**Symptom**: `Failed to initialize the TMA descriptor`  
**Fix**: ✅ Already applied - `--disable-custom-all-reduce`

### 2. Illegal Memory Access
**Symptom**: `an illegal memory access was encountered`  
**Fix**: ✅ Already applied - disabled custom all-reduce + auto KV cache

### 3. CUDA Out of Memory
**Symptom**: `CUDA out of memory` or `OutOfMemoryError`  
**Fix**: 
```bash
./init_baremetal.sh --gpu-mem-util 0.75 --max-model-len 32768
```

### 4. Worker Dies Unexpectedly
**Symptom**: `Worker proc VllmWorker-X died unexpectedly`  
**Fix**: This is usually a symptom of the above issues - the fixes should resolve it

---

## 📊 Expected Behavior

After the fix, you should see:

1. **Model Loading** (2-3 minutes):
   ```
   Loading model weights from moonshotai/Kimi-K2-Instruct-0905
   ```

2. **Worker Initialization** (1-2 minutes):
   ```
   Worker_TP0, Worker_TP1, ... Worker_TP7 starting
   ```

3. **KV Cache Allocation** (30 seconds):
   ```
   Allocating KV cache...
   ```

4. **Server Ready**:
   ```
   Application startup complete.
   Uvicorn running on http://0.0.0.0:8000
   ```

**Total startup time: ~5-7 minutes**

---

## 🔧 Advanced Debugging

### Check Logs

All vLLM output is automatically saved to `logs/kimi_k2_TIMESTAMP.log`:

```bash
# View the latest log file
tail -f $(ls -t logs/kimi_k2_*.log | head -1)

# Search for CUDA errors
grep -i "cuda error" logs/kimi_k2_*.log

# Search for TMA descriptor errors
grep -i "tma descriptor" logs/kimi_k2_*.log

# Search for worker failures
grep -i "worker.*died" logs/kimi_k2_*.log
```

### Check GPU Status

```bash
# Monitor GPUs in real-time
watch -n 1 nvidia-smi

# Check for CUDA errors
dmesg | grep -i cuda

# Check for memory issues
nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv
```

### Analyze vLLM Logs

The error messages will show which worker failed. Look for:
- Which GPU had the error (Worker_TP0 through Worker_TP7)
- The exact CUDA error code
- Memory allocation failures
- TMA descriptor initialization errors

### Test GPU Health

```bash
# Run basic CUDA test
python3 -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count())"

# Should output:
# True
# 8
```

---

## 🆘 If Nothing Works

### Last Resort Options:

1. **Wait for vLLM Update**
   - B200 is very new (Blackwell architecture)
   - vLLM support may still be stabilizing
   - Check: https://github.com/vllm-project/vllm/issues

2. **Try Different Model**
   - Test with a smaller model first to verify B200 functionality
   ```bash
   vllm serve meta-llama/Llama-3.1-8B-Instruct --tensor-parallel-size 2
   ```

3. **Check NVIDIA Driver**
   ```bash
   nvidia-smi
   # B200 requires CUDA 12.0+ and recent drivers (550+)
   ```

4. **Report Issue to vLLM**
   - B200-specific issues should be reported
   - Include: GPU model, CUDA version, vLLM version, full error log

---

## 📝 What Changed in the Script

**Before (caused TMA descriptor error):**
```bash
--kv-cache-dtype fp8 \
--max-num-batched-tokens 16384 \
--max-num-seqs 128
# (missing --disable-custom-all-reduce)
```

**After (B200 compatible):**
```bash
--kv-cache-dtype auto \
--max-num-batched-tokens 8192 \
--max-num-seqs 64 \
--disable-custom-all-reduce
```

---

## ✅ Success Checklist

Once working, you should be able to:

- [ ] Script starts without errors
- [ ] All 8 workers initialize successfully
- [ ] KV cache allocates without CUDA errors
- [ ] Server shows "Application startup complete"
- [ ] Can curl the API endpoint
- [ ] Model responds to queries

---

## 🔗 Useful Resources

- **vLLM GitHub Issues**: https://github.com/vllm-project/vllm/issues
- **B200 Documentation**: https://www.nvidia.com/en-us/data-center/b200/
- **vLLM Distributed Inference**: https://docs.vllm.ai/en/latest/serving/distributed_serving.html

---

**💡 TL;DR**: The script is now updated with `--disable-custom-all-reduce` which fixes the TMA descriptor error on B200 GPUs. Just run `./init_baremetal.sh` again!
