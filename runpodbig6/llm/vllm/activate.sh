#!/bin/bash
# Convenience script to activate vLLM venv
# Usage: source /wrk/llm/vllm/activate.sh

source /wrk/llm/vllm/venv/bin/activate
echo "✅ vLLM virtual environment activated!"
echo ""
echo "Available commands:"
echo "  ./run_vllm_fp4.sh          - Start vLLM with FP4 model (RECOMMENDED)"
echo "  python test_vllm_fp4.py    - Test FP4 model"
echo "  ./run_vllm_server.sh       - Start vLLM with GGUF model"
echo "  python test_vllm.py        - Test GGUF model"
echo "  vllm serve <model>         - Start vLLM server (CLI)"
echo ""
echo "📖 See README.md for more details"
echo ""
