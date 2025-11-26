#!/bin/bash
# Test Higgs Audio vLLM server
# Run this after starting the server with run_vllm_higgs_audio.sh

set -euo pipefail

API_BASE="http://localhost:8000"
OUTPUT_FILE="higgs_test_speech.wav"

echo "========================================"
echo "Testing Higgs Audio v2 Server"
echo "========================================"
echo "API Base: $API_BASE"
echo "Output file: $OUTPUT_FILE"
echo ""

# Test if server is running
echo "1. Checking if server is responding..."
if curl -s -f "$API_BASE/v1/models" > /dev/null 2>&1; then
    echo "✓ Server is running!"
    echo ""
else
    echo "✗ Server is not responding at $API_BASE"
    echo "Please start the server first with: ./run_vllm_higgs_audio.sh"
    exit 1
fi

# Test audio generation
echo "2. Testing audio generation..."
echo "Input text: 'Today is a wonderful day to build something people love!'"
echo "Voice: en_woman"
echo ""

curl -X POST "$API_BASE/v1/audio/speech" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "higgs-audio-v2-generation-3B-base",
    "voice": "en_woman",
    "input": "Today is a wonderful day to build something people love!",
    "response_format": "pcm"
  }' \
  --output - 2>/dev/null | ffmpeg -f s16le -ar 24000 -ac 1 -i - "$OUTPUT_FILE" -y 2>&1 | grep -E "(Duration|size)"

if [ -f "$OUTPUT_FILE" ]; then
    echo ""
    echo "✓ Audio generation successful!"
    echo "Output saved to: $OUTPUT_FILE"
    echo ""
    echo "File info:"
    ls -lh "$OUTPUT_FILE"
    echo ""
    echo "You can play it with: ffplay $OUTPUT_FILE"
else
    echo "✗ Audio generation failed"
    exit 1
fi

echo ""
echo "========================================"
echo "Test completed successfully!"
echo "========================================"
echo ""
echo "Next steps:"
echo "  - Try different voices: en_man, belinda, broom_salesman"
echo "  - Use the Python client for more advanced features"
echo "  - See examples at: https://github.com/boson-ai/higgs-audio"












