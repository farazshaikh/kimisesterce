#!/usr/bin/env python3
"""Compare vLLM and SGLang evaluation results."""

import json
import sys
import glob
from pathlib import Path

def load_latest_eval(backend):
    """Load the most recent evaluation file for a given backend."""
    pattern = f"/compile/llm/eval_{backend}_*.json"
    files = sorted(glob.glob(pattern), reverse=True)
    
    if not files:
        return None
    
    with open(files[0], 'r') as f:
        return json.load(f)

def main():
    # Load most recent evaluations
    vllm_data = load_latest_eval("vllm")
    sglang_data = load_latest_eval("sglang")
    
    if not vllm_data and not sglang_data:
        print("❌ No evaluation files found!")
        print("Run test_vllm_only.py or test_sglang_only.py first.")
        sys.exit(1)
    
    print(f"\n{'='*80}")
    print("VLLM vs SGLANG COMPARISON")
    print(f"{'='*80}\n")
    
    # Display vLLM results
    if vllm_data:
        print(f"🔵 vLLM (evaluated at {vllm_data['timestamp']})")
        print(f"   All requests:  {vllm_data['statistics']['all_requests']['avg_tokens_per_second']:.2f} tokens/s")
        print(f"   Warm requests: {vllm_data['statistics']['warm_requests']['avg_tokens_per_second']:.2f} tokens/s")
    else:
        print(f"🔵 vLLM: No data available")
    
    print()
    
    # Display SGLang results
    if sglang_data:
        print(f"🟢 SGLang (evaluated at {sglang_data['timestamp']})")
        print(f"   All requests:  {sglang_data['statistics']['all_requests']['avg_tokens_per_second']:.2f} tokens/s")
        print(f"   Warm requests: {sglang_data['statistics']['warm_requests']['avg_tokens_per_second']:.2f} tokens/s")
    else:
        print(f"🟢 SGLang: No data available")
    
    # Comparison
    if vllm_data and sglang_data:
        print(f"\n{'─'*80}")
        print("DETAILED COMPARISON:")
        print(f"{'─'*80}\n")
        
        vllm_warm_tps = vllm_data['statistics']['warm_requests']['avg_tokens_per_second']
        sglang_warm_tps = sglang_data['statistics']['warm_requests']['avg_tokens_per_second']
        
        print(f"{'Metric':<30} {'vLLM':>15} {'SGLang':>15} {'Diff':>15}")
        print(f"{'─'*30} {'─'*15} {'─'*15} {'─'*15}")
        
        # Warm tokens/second
        diff_warm = sglang_warm_tps - vllm_warm_tps
        pct_warm = (diff_warm / vllm_warm_tps) * 100 if vllm_warm_tps > 0 else 0
        print(f"{'Warm tokens/second':<30} {vllm_warm_tps:>15.2f} {sglang_warm_tps:>15.2f} {f'+{diff_warm:.2f} ({pct_warm:+.1f}%)' if diff_warm >= 0 else f'{diff_warm:.2f} ({pct_warm:.1f}%)':>15}")
        
        # Warm time
        vllm_warm_time = vllm_data['statistics']['warm_requests']['avg_time']
        sglang_warm_time = sglang_data['statistics']['warm_requests']['avg_time']
        diff_time = sglang_warm_time - vllm_warm_time
        pct_time = (diff_time / vllm_warm_time) * 100 if vllm_warm_time > 0 else 0
        print(f"{'Warm avg time (s)':<30} {vllm_warm_time:>15.2f} {sglang_warm_time:>15.2f} {f'+{diff_time:.2f} ({pct_time:+.1f}%)' if diff_time >= 0 else f'{diff_time:.2f} ({pct_time:.1f}%)':>15}")
        
        # Warm TTFT
        vllm_warm_ttft = vllm_data['statistics']['warm_requests'].get('avg_ttft', 0)
        sglang_warm_ttft = sglang_data['statistics']['warm_requests'].get('avg_ttft', 0)
        if vllm_warm_ttft > 0 or sglang_warm_ttft > 0:
            diff_ttft = sglang_warm_ttft - vllm_warm_ttft
            pct_ttft = (diff_ttft / vllm_warm_ttft) * 100 if vllm_warm_ttft > 0 else 0
            print(f"{'Warm TTFT (s)':<30} {vllm_warm_ttft:>15.3f} {sglang_warm_ttft:>15.3f} {f'+{diff_ttft:.3f} ({pct_ttft:+.1f}%)' if diff_ttft >= 0 else f'{diff_ttft:.3f} ({pct_ttft:.1f}%)':>15}")
        
        # Overall winner
        print(f"\n{'─'*80}")
        if sglang_warm_tps > vllm_warm_tps:
            speedup = sglang_warm_tps / vllm_warm_tps
            print(f"🏆 SGLang is {speedup:.2f}x faster than vLLM (warm requests)")
        elif vllm_warm_tps > sglang_warm_tps:
            speedup = vllm_warm_tps / sglang_warm_tps
            print(f"🏆 vLLM is {speedup:.2f}x faster than SGLang (warm requests)")
        else:
            print(f"🤝 Both have identical performance")
        
        print(f"{'─'*80}\n")
        
        # Individual request breakdown
        print(f"{'Request Breakdown':<30} {'vLLM (tokens/s)':>20} {'SGLang (tokens/s)':>20}")
        print(f"{'─'*30} {'─'*20} {'─'*20}")
        
        for i in range(len(vllm_data['results'])):
            label = "🥶 Cold start" if i == 0 else f"🔥 Warm #{i}"
            vllm_tps = vllm_data['results'][i]['tps']
            sglang_tps = sglang_data['results'][i]['tps'] if i < len(sglang_data['results']) else 0
            print(f"{label:<30} {vllm_tps:>20.2f} {sglang_tps:>20.2f}")
    
    print(f"\n{'='*80}\n")

if __name__ == "__main__":
    main()

