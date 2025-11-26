#!/usr/bin/env python3
"""
Test client to check TTFT and timing metrics from vLLM server.
Usage: python test_timing.py [prompt]
"""

import sys
import time
import requests
import json

PORT = 8083
HOST = "localhost"
URL = f"http://{HOST}:{PORT}/v1/chat/completions"

def test_request(prompt="Tell me a short story about a robot."):
    """Send a test request and measure timing."""
    
    print(f"\n{'='*80}")
    print(f"Testing vLLM server at {URL}")
    print(f"{'='*80}\n")
    
    payload = {
        "model": "Qwen3-235B-A22B-Instruct-FP8",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 100,
        "temperature": 0.7,
        "stream": False
    }
    
    print(f"Prompt: {prompt}")
    print(f"Max tokens: {payload['max_tokens']}")
    print(f"\nSending request...\n")
    
    start_time = time.time()
    
    try:
        response = requests.post(
            URL,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=120
        )
        
        end_time = time.time()
        total_time_ms = (end_time - start_time) * 1000
        
        if response.status_code == 200:
            result = response.json()
            
            # Extract metrics
            usage = result.get('usage', {})
            prompt_tokens = usage.get('prompt_tokens', 0)
            completion_tokens = usage.get('completion_tokens', 0)
            total_tokens = usage.get('total_tokens', 0)
            
            # Get response text
            content = result['choices'][0]['message']['content']
            
            print(f"✓ Request completed successfully!\n")
            print(f"TIMING METRICS:")
            print(f"  Total request time: {total_time_ms:.2f} ms")
            
            # Check for custom headers
            if 'x-response-time' in response.headers:
                print(f"  Server response time: {response.headers['x-response-time']} ms")
            if 'x-request-id' in response.headers:
                print(f"  Request ID: {response.headers['x-request-id']}")
            
            print(f"\nTOKEN USAGE:")
            print(f"  Prompt tokens:     {prompt_tokens}")
            print(f"  Completion tokens: {completion_tokens}")
            print(f"  Total tokens:      {total_tokens}")
            
            if completion_tokens > 0 and total_time_ms > 0:
                tokens_per_second = (completion_tokens / total_time_ms) * 1000
                print(f"\nTHROUGHPUT:")
                print(f"  Generation speed: {tokens_per_second:.2f} tokens/s")
                print(f"  Time per token:   {total_time_ms/completion_tokens:.2f} ms/token")
            
            print(f"\nRESPONSE:")
            print(f"  {content[:200]}{'...' if len(content) > 200 else ''}")
            
            print(f"\n{'='*80}")
            print("Check the vLLM server logs for detailed TTFT and per-request metrics!")
            print(f"{'='*80}\n")
            
        else:
            print(f"✗ Error: {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print(f"✗ Could not connect to vLLM server at {URL}")
        print(f"  Make sure the server is running on port {PORT}")
    except requests.exceptions.Timeout:
        print(f"✗ Request timed out after 120 seconds")
    except Exception as e:
        print(f"✗ Error: {e}")


if __name__ == '__main__':
    prompt = sys.argv[1] if len(sys.argv) > 1 else "Tell me a short story about a robot."
    test_request(prompt)



