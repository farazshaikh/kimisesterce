#!/usr/bin/env python3
"""Benchmark Granite 4.0 H Micro for real token/s performance"""

import requests
import json
import time

ENDPOINT = "http://localhost:8080/v1"

def benchmark(num_tokens=100):
    """Benchmark with longer generation to see sustained speed"""
    
    print(f"🚀 Benchmarking Granite 4.0 H Micro")
    print(f"📊 Generating {num_tokens} tokens...\n")
    
    payload = {
        "model": "granite-micro",
        "messages": [
            {"role": "user", "content": "Write a detailed explanation of how neural networks work. Include multiple paragraphs."}
        ],
        "max_tokens": num_tokens,
        "temperature": 0.7
    }
    
    try:
        start = time.time()
        response = requests.post(
            f"{ENDPOINT}/chat/completions",
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
            total_tokens = usage.get("total_tokens", 0)
            
            print(f"✅ Generation complete!")
            print(f"\n📝 Response preview:")
            print(f"{content[:200]}...\n")
            
            print(f"📊 Stats:")
            print(f"  Prompt tokens: {prompt_tokens}")
            print(f"  Completion tokens: {completion_tokens}")
            print(f"  Total tokens: {total_tokens}")
            print(f"  Time: {elapsed:.2f}s")
            print(f"\n🚀 Speed: {completion_tokens/elapsed:.1f} tokens/sec")
            
            # Calculate time per token
            time_per_token = (elapsed / completion_tokens) * 1000 if completion_tokens > 0 else 0
            print(f"⏱️  Time per token: {time_per_token:.1f}ms")
            
            return completion_tokens / elapsed
        else:
            print(f"❌ Error: {result}")
            return 0
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return 0

if __name__ == "__main__":
    print("=" * 60)
    print("Testing sustained generation speed...")
    print("=" * 60)
    print()
    
    speeds = []
    
    # Test 1: 50 tokens
    print("Test 1: 50 tokens")
    print("-" * 60)
    speed = benchmark(50)
    speeds.append(speed)
    print()
    
    # Test 2: 100 tokens  
    print("Test 2: 100 tokens")
    print("-" * 60)
    speed = benchmark(100)
    speeds.append(speed)
    print()
    
    # Test 3: 200 tokens
    print("Test 3: 200 tokens")
    print("-" * 60)
    speed = benchmark(200)
    speeds.append(speed)
    print()
    
    print("=" * 60)
    print(f"Average speed: {sum(speeds)/len(speeds):.1f} tokens/sec")
    print("=" * 60)

