#!/usr/bin/env python3
"""Quick test script for Granite 4.0 Micro Dense (Pure Transformer)"""

import requests
import json
import time
import sys

# Endpoints
LOCAL_ENDPOINT = "http://localhost:8080/v1"
EXTERNAL_ENDPOINT = "http://38.80.152.249:30446/v1"  # RunPod exposed port

def test_dense(endpoint):
    """Test the pure transformer version with various prompts"""
    
    print("="*70)
    print("TESTING GRANITE 4.0 MICRO DENSE (Pure Transformer)")
    print(f"Endpoint: {endpoint}")
    print("="*70)
    print()
    
    # Check server
    print("🔍 Checking server...")
    try:
        response = requests.get(f"{endpoint}/models", timeout=5)
        models = response.json()
        print(f"✅ Server is alive!")
        print(f"   Model: {models['data'][0]['id']}")
        print(f"   Max tokens: {models['data'][0]['max_model_len']}")
        print()
    except Exception as e:
        print(f"❌ Cannot connect to server: {e}")
        print(f"   Make sure you ran: ./run_granite_micro_dense.sh")
        return
    
    # Test cases
    tests = [
        {
            "name": "Quick Response (10 tokens)",
            "prompt": "What is 2+2? Answer in one word.",
            "max_tokens": 10
        },
        {
            "name": "Short Answer (50 tokens)",
            "prompt": "Explain what a neural network is in 2-3 sentences.",
            "max_tokens": 50
        },
        {
            "name": "Medium Response (100 tokens)",
            "prompt": "Write a brief explanation of how transformers work in AI.",
            "max_tokens": 100
        },
        {
            "name": "Long Response (200 tokens)",
            "prompt": "Explain quantum computing in detail with multiple paragraphs.",
            "max_tokens": 200
        }
    ]
    
    results = []
    
    for i, test in enumerate(tests, 1):
        print(f"{'='*70}")
        print(f"Test {i}/4: {test['name']}")
        print(f"{'='*70}")
        
        payload = {
            "model": "granite-micro-dense",
            "messages": [
                {"role": "user", "content": test['prompt']}
            ],
            "max_tokens": test['max_tokens'],
            "temperature": 0.7
        }
        
        try:
            start = time.time()
            response = requests.post(
                f"{endpoint}/chat/completions",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            elapsed = time.time() - start
            
            result = response.json()
            
            if "choices" in result and len(result["choices"]) > 0:
                content = result["choices"][0]["message"]["content"]
                usage = result.get("usage", {})
                
                completion_tokens = usage.get("completion_tokens", 0)
                speed = completion_tokens / elapsed if elapsed > 0 else 0
                
                print(f"\n📝 Response:")
                print(f"{content[:200]}{'...' if len(content) > 200 else ''}")
                print(f"\n📊 Performance:")
                print(f"  Time: {elapsed:.2f}s")
                print(f"  Tokens: {completion_tokens}")
                print(f"  Speed: {speed:.1f} tok/s")
                print(f"  Latency: {(elapsed/completion_tokens)*1000:.1f}ms per token")
                print()
                
                results.append({
                    "test": test['name'],
                    "tokens": completion_tokens,
                    "time": elapsed,
                    "speed": speed
                })
            else:
                print(f"❌ Error: {result}")
                
        except Exception as e:
            print(f"❌ Error: {e}")
    
    # Summary
    if results:
        print("="*70)
        print("SUMMARY")
        print("="*70)
        print()
        
        avg_speed = sum(r['speed'] for r in results) / len(results)
        max_speed = max(r['speed'] for r in results)
        
        print(f"Tests completed: {len(results)}/4")
        print(f"Average speed: {avg_speed:.1f} tok/s")
        print(f"Peak speed: {max_speed:.1f} tok/s")
        print()
        
        print("Detailed results:")
        for r in results:
            print(f"  {r['test']:<30} {r['speed']:>6.1f} tok/s  ({r['time']:.2f}s)")
        
        print()
        print("="*70)
        
        # Performance assessment
        if avg_speed > 200:
            print("🚀 EXCELLENT! Pure transformer is blazing fast!")
        elif avg_speed > 150:
            print("✅ GOOD! Decent performance on H200.")
        elif avg_speed > 100:
            print("⚠️  OK - Could be better. Check GPU utilization.")
        else:
            print("❌ SLOW - Something might be wrong.")
        
        print("="*70)

if __name__ == "__main__":
    # Check if user wants external endpoint
    if len(sys.argv) > 1 and sys.argv[1] == "external":
        endpoint = EXTERNAL_ENDPOINT
        print("Using EXTERNAL endpoint\n")
    else:
        endpoint = LOCAL_ENDPOINT
        print("Using LOCAL endpoint\n")
    
    test_dense(endpoint)
    
    print("\n💡 To test from external network, run:")
    print("  python3 test_granite_dense.py external")

