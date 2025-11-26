#!/bin/bash
# Quick test script for Granite 4.0 H Micro API

set -euo pipefail

# Use localhost if testing locally, or exposed IP if testing externally
LOCAL_ENDPOINT="http://localhost:8080/v1"
EXTERNAL_ENDPOINT="http://38.80.152.249:30446/v1"

# Change this to test from outside
ENDPOINT="${1:-$LOCAL_ENDPOINT}"

echo "Testing Granite 4.0 H Micro API..."
echo "Endpoint: $ENDPOINT"
echo ""

# Test 1: Check if server is alive
echo "=== Test 1: Models List ==="
curl -s "$ENDPOINT/models" | python3 -m json.tool
echo ""
echo ""

# Test 2: Simple completion
echo "=== Test 2: Chat Completion ==="
time curl -s "$ENDPOINT/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "granite-micro",
    "messages": [
      {"role": "user", "content": "What is the capital of France? Answer in one sentence."}
    ],
    "max_tokens": 50,
    "temperature": 0.7
  }' | python3 -m json.tool

echo ""
echo ""
echo "✅ Test complete!"
echo ""
echo "To test from external network, run:"
echo "  ./test_granite.sh external"

