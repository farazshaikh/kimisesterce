#!/usr/bin/env python3
"""Compare Granite 4.0 Micro versions: Hybrid Mamba2 vs Pure Transformer"""

import requests
import json
import time

HYBRID_ENDPOINT = "http://localhost:8080/v1"  # H-Micro (Mamba2)
DENSE_ENDPOINT = "http://localhost:8081/v1"   # Micro Dense (Transformer)

def test_speed(endpoint, model_name, version_name):
    """Test generation speed for a specific endpoint"""
    
    print(f"\n{'='*70}")
    print(f"Testing {version_name}")
    print(f"Endpoint: {endpoint}")
    print(f"{'='*70}\n")
    
    payload = {
        "model": model_name,
        "messages": [
            {"role": "user", "content": "Write a detailed explanation of quantum computing. Include multiple paragraphs with technical details."}
        ],
        "max_tokens": 150,
        "temperature": 0.7
    }
    
    try:
        # Check if server is alive
        try:
            models_resp = requests.get(f"{endpoint}/models", timeout=5)
            if models_resp.status_code != 200:
                print(f"❌ Server not responding on {endpoint}")
                return None
        except:
            print(f"❌ Cannot connect to {endpoint}")
            print(f"   Make sure the server is running!")
            return None
        
        # Run speed test
        start = time.time()
        response = requests.post(
            f"{endpoint}/chat/completions",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=60
        )
        elapsed = time.time() - start
        
        result = response.json()
        
        if "choices" in result and len(result["choices"]) > 0:
            content = result["choices"][0]["message"]["content"]
            usage = result.get("usage", {})
            
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            
            print(f"✅ Generation complete!")
            print(f"\n📝 Response preview:")
            print(f"{content[:150]}...\n")
            
            print(f"📊 Performance:")
            print(f"  Total time: {elapsed:.2f}s")
            print(f"  Completion tokens: {completion_tokens}")
            print(f"  Speed: {completion_tokens/elapsed:.1f} tokens/sec")
            print(f"  Time per token: {(elapsed/completion_tokens)*1000:.1f}ms")
            
            return {
                "version": version_name,
                "time": elapsed,
                "tokens": completion_tokens,
                "speed": completion_tokens / elapsed
            }
        else:
            print(f"❌ Error: {result}")
            return None
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

if __name__ == "__main__":
    print("\n" + "="*70)
    print("GRANITE 4.0 MICRO VERSION COMPARISON")
    print("="*70)
    
    results = []
    
    # Test Hybrid Mamba2 version (H-Micro)
    result = test_speed(HYBRID_ENDPOINT, "granite-micro", "H-Micro (4 Attention + 36 Mamba2)")
    if result:
        results.append(result)
    
    # Test Pure Transformer version (Dense)
    result = test_speed(DENSE_ENDPOINT, "granite-micro-dense", "Micro Dense (40 Attention Layers)")
    if result:
        results.append(result)
    
    # Summary
    if len(results) >= 2:
        print("\n" + "="*70)
        print("COMPARISON SUMMARY")
        print("="*70)
        
        for r in results:
            print(f"\n{r['version']}:")
            print(f"  Speed: {r['speed']:.1f} tok/s")
            print(f"  Time: {r['time']:.2f}s")
        
        speedup = results[1]['speed'] / results[0]['speed']
        print(f"\n🚀 Speedup: {speedup:.2f}x faster with pure transformer!")
        print("="*70)
    elif len(results) == 1:
        print(f"\n⚠️  Only tested one version. Start both servers to compare!")
    else:
        print(f"\n❌ No servers running. Start at least one server to test!")
        print(f"\nTo start servers:")
        print(f"  H-Micro (Mamba2):    ./run_granite_micro.sh")
        print(f"  Micro Dense (Trans): ./run_granite_micro_dense.sh")

