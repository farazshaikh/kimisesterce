#!/usr/bin/env python3
"""Quick test script for Granite 4.0 H Micro API"""

import requests
import json
import time
import sys

# Endpoints
LOCAL_ENDPOINT = "http://localhost:8080/v1"
EXTERNAL_ENDPOINT = "http://38.80.152.249:30446/v1"  # RunPod exposed port

# Note: If 30446 is the exposed port, make sure it's mapped to 8080 internally

def test_api(endpoint):
    """Test the Granite API with a simple request"""
    
    print(f"Testing Granite 4.0 H Micro API...")
    print(f"Endpoint: {endpoint}")
    print()
    
    # Test 1: List models
    print("=== Test 1: Models List ===")
    try:
        response = requests.get(f"{endpoint}/models", timeout=10)
        print(json.dumps(response.json(), indent=2))
        print()
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    # Test 2: Chat completion
    print("=== Test 2: Chat Completion ===")
    payload = {
        "model": "granite-micro",
        "messages": [
            {"role": "user", "content": "What is the capital of France? Answer in one sentence."}
        ],
        "max_tokens": 50,
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
        print(json.dumps(result, indent=2))
        print()
        
        # Show timing
        if "choices" in result and len(result["choices"]) > 0:
            content = result["choices"][0]["message"]["content"]
            print(f"✅ Response: {content}")
            print(f"⏱️  Time: {elapsed:.2f}s")
            
            if "usage" in result:
                tokens = result["usage"].get("completion_tokens", 0)
                if tokens > 0:
                    print(f"🚀 Speed: {tokens/elapsed:.1f} tokens/sec")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    # Check if user wants external endpoint
    if len(sys.argv) > 1 and sys.argv[1] == "external":
        endpoint = EXTERNAL_ENDPOINT
        print("Using EXTERNAL endpoint\n")
    else:
        endpoint = LOCAL_ENDPOINT
        print("Using LOCAL endpoint\n")
    
    success = test_api(endpoint)
    
    if success:
        print("\n✅ All tests passed!")
        print("\nTo test from external network, run:")
        print("  python3 test_granite.py external")
    else:
        print("\n❌ Tests failed!")
        sys.exit(1)

