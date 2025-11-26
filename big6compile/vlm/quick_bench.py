#!/usr/bin/env python3
"""
Quick VLM benchmark - test image understanding with timing
Usage: python quick_bench.py [--url http://localhost:8006] [--image path/to/image.png]
"""

import argparse
import base64
import time
import requests
from pathlib import Path


def encode_image(image_path: str) -> str:
    """Encode image to base64."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def test_vlm(url: str, image_path: str, prompt: str = "Describe this image in detail."):
    """Run a single VLM test."""
    
    # Encode image
    print(f"📸 Loading image: {image_path}")
    image_b64 = encode_image(image_path)
    
    # Prepare request
    payload = {
        "model": "qwen3-vl",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_b64}"}
                    }
                ]
            }
        ],
        "max_tokens": 200,
        "temperature": 0.0
    }
    
    print(f"\n🚀 Sending request to {url}")
    print(f"   Prompt: {prompt}")
    print(f"   Max tokens: {payload['max_tokens']}")
    
    # Send request and measure time
    start = time.time()
    
    try:
        response = requests.post(
            f"{url}/v1/chat/completions",
            json=payload,
            timeout=60
        )
        
        end = time.time()
        elapsed_ms = (end - start) * 1000
        
        if response.status_code == 200:
            result = response.json()
            
            # Extract metrics
            usage = result.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            content = result["choices"][0]["message"]["content"]
            
            # Calculate throughput
            tokens_per_sec = completion_tokens / (elapsed_ms / 1000) if elapsed_ms > 0 else 0
            
            print(f"\n✅ SUCCESS")
            print(f"{'='*80}")
            print(f"⏱️  Total time: {elapsed_ms:.0f} ms ({elapsed_ms/1000:.2f}s)")
            print(f"📊 Tokens:")
            print(f"   Prompt: {prompt_tokens} tokens")
            print(f"   Completion: {completion_tokens} tokens")
            print(f"   Total: {prompt_tokens + completion_tokens} tokens")
            print(f"🚄 Speed: {tokens_per_sec:.1f} tokens/s")
            print(f"⚡ Time per token: {elapsed_ms/completion_tokens:.0f} ms" if completion_tokens > 0 else "")
            print(f"\n💬 Response:")
            print(f"{'='*80}")
            print(content)
            print(f"{'='*80}\n")
            
            return {
                "success": True,
                "time_ms": elapsed_ms,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "tokens_per_sec": tokens_per_sec,
                "content": content
            }
        else:
            print(f"\n❌ ERROR: HTTP {response.status_code}")
            print(response.text)
            return {"success": False, "error": response.text}
            
    except requests.exceptions.ConnectionError:
        print(f"\n❌ ERROR: Could not connect to {url}")
        print("Make sure the VLM server is running:")
        print("  cd /compile/vlm && make serve")
        return {"success": False, "error": "Connection failed"}
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        return {"success": False, "error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Quick VLM benchmark")
    parser.add_argument("--url", default="http://localhost:8006", help="VLM server URL")
    parser.add_argument("--image", default="/compile/image.jpg", help="Path to test image")
    parser.add_argument("--prompt", default="Describe this image in detail.", help="Prompt to use")
    parser.add_argument("--repeat", type=int, default=1, help="Number of times to repeat test")
    
    args = parser.parse_args()
    
    # Check image exists
    if not Path(args.image).exists():
        print(f"❌ ERROR: Image not found: {args.image}")
        return 1
    
    print(f"\n{'='*80}")
    print(f"🎯 VLM Quick Benchmark")
    print(f"{'='*80}")
    print(f"Server: {args.url}")
    print(f"Image: {args.image}")
    print(f"Runs: {args.repeat}")
    print(f"{'='*80}\n")
    
    results = []
    for i in range(args.repeat):
        if args.repeat > 1:
            print(f"\n📌 Run {i+1}/{args.repeat}")
        
        result = test_vlm(args.url, args.image, args.prompt)
        if result["success"]:
            results.append(result)
        
        if i < args.repeat - 1:
            time.sleep(1)  # Brief pause between runs
    
    # Summary if multiple runs
    if len(results) > 1:
        avg_time = sum(r["time_ms"] for r in results) / len(results)
        avg_tps = sum(r["tokens_per_sec"] for r in results) / len(results)
        
        print(f"\n{'='*80}")
        print(f"📈 Summary ({len(results)} successful runs)")
        print(f"{'='*80}")
        print(f"Average time: {avg_time:.0f} ms")
        print(f"Average speed: {avg_tps:.1f} tokens/s")
        print(f"{'='*80}\n")
    
    return 0 if results else 1


if __name__ == "__main__":
    exit(main())

