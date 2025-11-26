#!/usr/bin/env python3
"""
Analyze vLLM logs to extract and visualize performance metrics.
Usage: python analyze_vllm_logs.py <logfile>
       or: tail -f vllm.log | python analyze_vllm_logs.py
"""

import sys
import re
import json
from datetime import datetime
from collections import defaultdict


def parse_log_line(line):
    """Extract relevant metrics from log lines."""
    data = {}
    
    # Extract timestamp
    ts_match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
    if ts_match:
        data['timestamp'] = ts_match.group(1)
    
    # Extract request ID if present
    req_id_match = re.search(r'request_id[=:]?\s*([a-f0-9-]+)', line, re.IGNORECASE)
    if req_id_match:
        data['request_id'] = req_id_match.group(1)
    
    # Extract TTFT (Time to First Token)
    ttft_match = re.search(r'time[_\s]to[_\s]first[_\s]token[:\s=]+([0-9.]+)\s*(ms|s)?', line, re.IGNORECASE)
    if ttft_match:
        ttft = float(ttft_match.group(1))
        if ttft_match.group(2) == 's':
            ttft *= 1000  # Convert to ms
        data['ttft_ms'] = ttft
    
    # Extract total time
    total_time_match = re.search(r'total[_\s]time[:\s=]+([0-9.]+)\s*(ms|s)?', line, re.IGNORECASE)
    if total_time_match:
        total_time = float(total_time_match.group(1))
        if total_time_match.group(2) == 's':
            total_time *= 1000
        data['total_time_ms'] = total_time
    
    # Extract prompt and generation tokens
    prompt_match = re.search(r'prompt[_\s]tokens?[:\s=]+(\d+)', line, re.IGNORECASE)
    if prompt_match:
        data['prompt_tokens'] = int(prompt_match.group(1))
    
    gen_match = re.search(r'(?:generation|generated|output)[_\s]tokens?[:\s=]+(\d+)', line, re.IGNORECASE)
    if gen_match:
        data['generation_tokens'] = int(gen_match.group(1))
    
    # Extract throughput metrics from aggregate stats
    prompt_throughput = re.search(r'Avg prompt throughput:\s*([0-9.]+)\s*tokens/s', line)
    if prompt_throughput:
        data['avg_prompt_throughput'] = float(prompt_throughput.group(1))
    
    gen_throughput = re.search(r'Avg generation throughput:\s*([0-9.]+)\s*tokens/s', line)
    if gen_throughput:
        data['avg_gen_throughput'] = float(gen_throughput.group(1))
    
    # Extract prefix cache hit rate
    cache_hit = re.search(r'Prefix cache hit rate:\s*([0-9.]+)%', line)
    if cache_hit:
        data['cache_hit_rate'] = float(cache_hit.group(1))
    
    # Extract running/waiting requests
    running = re.search(r'Running:\s*(\d+)\s*reqs', line)
    if running:
        data['running_reqs'] = int(running.group(1))
    
    waiting = re.search(r'Waiting:\s*(\d+)\s*reqs', line)
    if waiting:
        data['waiting_reqs'] = int(waiting.group(1))
    
    return data if data else None


def analyze_metrics(metrics):
    """Calculate statistics from collected metrics."""
    if not metrics:
        print("No metrics found in logs.")
        return
    
    # Group by metric type
    ttfts = [m['ttft_ms'] for m in metrics if 'ttft_ms' in m]
    total_times = [m['total_time_ms'] for m in metrics if 'total_time_ms' in m]
    prompt_tokens = [m['prompt_tokens'] for m in metrics if 'prompt_tokens' in m]
    gen_tokens = [m['generation_tokens'] for m in metrics if 'generation_tokens' in m]
    
    print("\n" + "="*80)
    print("PERFORMANCE ANALYSIS")
    print("="*80)
    
    if ttfts:
        print(f"\nTime to First Token (TTFT):")
        print(f"  Min:    {min(ttfts):.2f} ms")
        print(f"  Max:    {max(ttfts):.2f} ms")
        print(f"  Avg:    {sum(ttfts)/len(ttfts):.2f} ms")
        print(f"  Median: {sorted(ttfts)[len(ttfts)//2]:.2f} ms")
        print(f"  Count:  {len(ttfts)} requests")
    
    if total_times:
        print(f"\nTotal Request Time:")
        print(f"  Min:    {min(total_times):.2f} ms")
        print(f"  Max:    {max(total_times):.2f} ms")
        print(f"  Avg:    {sum(total_times)/len(total_times):.2f} ms")
        print(f"  Median: {sorted(total_times)[len(total_times)//2]:.2f} ms")
    
    if prompt_tokens:
        print(f"\nPrompt Tokens:")
        print(f"  Min:    {min(prompt_tokens)}")
        print(f"  Max:    {max(prompt_tokens)}")
        print(f"  Avg:    {sum(prompt_tokens)/len(prompt_tokens):.1f}")
        print(f"  Total:  {sum(prompt_tokens)}")
    
    if gen_tokens:
        print(f"\nGeneration Tokens:")
        print(f"  Min:    {min(gen_tokens)}")
        print(f"  Max:    {max(gen_tokens)}")
        print(f"  Avg:    {sum(gen_tokens)/len(gen_tokens):.1f}")
        print(f"  Total:  {sum(gen_tokens)}")
    
    # Throughput analysis
    throughput_data = [m for m in metrics if 'avg_prompt_throughput' in m]
    if throughput_data:
        prompt_tps = [m['avg_prompt_throughput'] for m in throughput_data]
        gen_tps = [m['avg_gen_throughput'] for m in throughput_data if 'avg_gen_throughput' in m]
        
        print(f"\nThroughput (from aggregate stats):")
        print(f"  Prompt throughput:")
        print(f"    Min: {min(prompt_tps):.1f} tokens/s")
        print(f"    Max: {max(prompt_tps):.1f} tokens/s")
        print(f"    Avg: {sum(prompt_tps)/len(prompt_tps):.1f} tokens/s")
        
        if gen_tps:
            print(f"  Generation throughput:")
            print(f"    Min: {min(gen_tps):.1f} tokens/s")
            print(f"    Max: {max(gen_tps):.1f} tokens/s")
            print(f"    Avg: {sum(gen_tps)/len(gen_tps):.1f} tokens/s")
    
    # Cache hit rate analysis
    cache_data = [m['cache_hit_rate'] for m in metrics if 'cache_hit_rate' in m]
    if cache_data:
        print(f"\nPrefix Cache Hit Rate:")
        print(f"  Min:    {min(cache_data):.1f}%")
        print(f"  Max:    {max(cache_data):.1f}%")
        print(f"  Avg:    {sum(cache_data)/len(cache_data):.1f}%")
        print(f"  Latest: {cache_data[-1]:.1f}%")
    
    print("\n" + "="*80)


def main():
    """Main function to process logs."""
    metrics = []
    
    try:
        if len(sys.argv) > 1:
            # Read from file
            with open(sys.argv[1], 'r') as f:
                for line in f:
                    data = parse_log_line(line.strip())
                    if data:
                        metrics.append(data)
                        if 'ttft_ms' in data or 'avg_prompt_throughput' in data:
                            print(f"[{data.get('timestamp', 'N/A')}] ", end='')
                            if 'ttft_ms' in data:
                                print(f"TTFT: {data['ttft_ms']:.1f}ms", end=' ')
                            if 'avg_prompt_throughput' in data:
                                print(f"Throughput: {data['avg_prompt_throughput']:.0f} t/s (prompt), "
                                      f"{data.get('avg_gen_throughput', 0):.1f} t/s (gen)", end=' ')
                            if 'cache_hit_rate' in data:
                                print(f"Cache: {data['cache_hit_rate']:.1f}%", end='')
                            print()
        else:
            # Read from stdin (streaming)
            print("Reading from stdin (streaming mode)... Press Ctrl+C to show summary")
            for line in sys.stdin:
                data = parse_log_line(line.strip())
                if data:
                    metrics.append(data)
                    if 'ttft_ms' in data or 'avg_prompt_throughput' in data:
                        print(f"\r[{data.get('timestamp', 'N/A')}] ", end='')
                        if 'ttft_ms' in data:
                            print(f"TTFT: {data['ttft_ms']:.1f}ms", end=' ')
                        if 'avg_prompt_throughput' in data:
                            print(f"Throughput: {data['avg_prompt_throughput']:.0f} t/s", end=' ')
                        if 'cache_hit_rate' in data:
                            print(f"Cache: {data['cache_hit_rate']:.1f}%", end='')
                        print()
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
    
    finally:
        if metrics:
            analyze_metrics(metrics)


if __name__ == '__main__':
    main()

