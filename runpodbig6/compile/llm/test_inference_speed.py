#!/usr/bin/env python3
"""
Quick inference speed test for LLM servers.
Tests both vLLM and SGLang with the same prompt for comparison.
"""

import time
import requests
import json
from typing import Dict, Any

# Test configuration
TEST_PROMPT = "Explain the concept of quantum entanglement in simple terms, as if teaching a high school student."

ENDPOINTS = {
    "vllm": {
        "url": "http://localhost:8083/v1/chat/completions",
        "model": "Qwen3-235B-A22B-Instruct-FP8"
    },
    "sglang": {
        "url": "http://localhost:8084/v1/chat/completions",
        "model": "Qwen3-235B-A22B-Instruct-FP8"
    }
}


def test_inference(endpoint_name: str, url: str, model: str, max_tokens: int = 256) -> Dict[str, Any]:
    """Test inference speed for a given endpoint."""
    
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": TEST_PROMPT}
        ],
        "max_tokens": max_tokens,
        "temperature": 0.7
    }
    
    print(f"\n{'='*80}")
    print(f"Testing {endpoint_name.upper()}")
    print(f"{'='*80}")
    print(f"Endpoint: {url}")
    print(f"Model: {model}")
    print(f"Prompt: {TEST_PROMPT[:80]}...")
    print(f"Max tokens: {max_tokens}")
    print(f"\nSending request...")
    
    try:
        start_time = time.time()
        response = requests.post(
            url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=120
        )
        end_time = time.time()
        
        if response.status_code != 200:
            print(f"❌ Error: HTTP {response.status_code}")
            print(f"Response: {response.text}")
            return None
        
        result = response.json()
        
        # Extract metrics
        total_time = end_time - start_time
        response_text = result['choices'][0]['message']['content']
        tokens_generated = result['usage']['completion_tokens']
        tokens_per_second = tokens_generated / total_time if total_time > 0 else 0
        
        # Display results
        print(f"\n✅ SUCCESS!")
        print(f"\n{'─'*80}")
        print(f"RESPONSE:")
        print(f"{'─'*80}")
        print(response_text)
        print(f"\n{'─'*80}")
        print(f"PERFORMANCE METRICS:")
        print(f"{'─'*80}")
        print(f"  Total time:        {total_time:.2f}s")
        print(f"  Tokens generated:  {tokens_generated}")
        print(f"  Tokens/second:     {tokens_per_second:.2f}")
        print(f"  Prompt tokens:     {result['usage']['prompt_tokens']}")
        print(f"  Total tokens:      {result['usage']['total_tokens']}")
        
        return {
            "endpoint": endpoint_name,
            "total_time": total_time,
            "tokens_generated": tokens_generated,
            "tokens_per_second": tokens_per_second,
            "prompt_tokens": result['usage']['prompt_tokens'],
            "response": response_text
        }
        
    except requests.exceptions.ConnectionError:
        print(f"❌ Connection Error: Could not connect to {url}")
        print(f"   Make sure the {endpoint_name} server is running!")
        return None
    except requests.exceptions.Timeout:
        print(f"❌ Timeout: Request took longer than 120 seconds")
        return None
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return None


def main():
    """Run inference speed tests."""
    
    print("\n" + "="*80)
    print("LLM INFERENCE SPEED TEST")
    print("="*80)
    print(f"Test prompt: {TEST_PROMPT}")
    
    results = []
    
    # Test vLLM
    vllm_result = test_inference(
        "vllm",
        ENDPOINTS["vllm"]["url"],
        ENDPOINTS["vllm"]["model"]
    )
    if vllm_result:
        results.append(vllm_result)
    
    # Test SGLang (optional - user will run this later)
    print("\n" + "="*80)
    print("SGLang test (run this after starting SGLang server)")
    print("="*80)
    response = input("\nDo you want to test SGLang now? (y/n): ").strip().lower()
    
    if response == 'y':
        sglang_result = test_inference(
            "sglang",
            ENDPOINTS["sglang"]["url"],
            ENDPOINTS["sglang"]["model"]
        )
        if sglang_result:
            results.append(sglang_result)
    
    # Comparison
    if len(results) >= 2:
        print("\n" + "="*80)
        print("COMPARISON")
        print("="*80)
        
        for i, result in enumerate(results, 1):
            print(f"\n{i}. {result['endpoint'].upper()}")
            print(f"   Time: {result['total_time']:.2f}s | "
                  f"Tokens/sec: {result['tokens_per_second']:.2f} | "
                  f"Tokens: {result['tokens_generated']}")
        
        # Speed comparison
        if results[0]['tokens_per_second'] > 0 and results[1]['tokens_per_second'] > 0:
            speedup = results[1]['tokens_per_second'] / results[0]['tokens_per_second']
            faster = results[1]['endpoint'] if speedup > 1 else results[0]['endpoint']
            speedup = max(speedup, 1/speedup)
            print(f"\n🏆 {faster.upper()} is {speedup:.2f}x faster!")
    
    print("\n" + "="*80)
    print("Test complete!")
    print("="*80)


if __name__ == "__main__":
    main()

