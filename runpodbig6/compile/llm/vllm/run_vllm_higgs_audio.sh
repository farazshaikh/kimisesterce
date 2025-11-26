#!/bin/bash
# Run Higgs Audio v2 with vLLM (Docker-based)
# Text-audio foundation model from Boson AI
# Source: https://github.com/boson-ai/higgs-audio

set -euo pipefail

PORT=8007
HOST="0.0.0.0"
MODEL_NAME="higgs-audio-v2-generation-3B-base"
DOCKER_IMAGE="bosonai/higgs-audio-vllm:latest"

# Optional: Custom voice presets directory
# Uncomment and set this if you want to use custom voices
# VOICE_PRESETS_DIR="/path/to/your/voice_presets"

echo "========================================"
echo "Starting Higgs Audio v2 with vLLM"
echo "========================================"
echo "Model: $MODEL_NAME"
echo "Port: $PORT"
echo "Docker Image: $DOCKER_IMAGE"
echo ""
echo "API Endpoints:"
echo "  - Audio Speech API: http://$HOST:$PORT/v1/audio/speech"
echo "  - Chat Completion API: http://$HOST:$PORT/v1/chat/completions"
echo ""
echo "Performance (from benchmarks):"
echo "  - A100 40GB: ~1500 tokens/s (~60s audio/s)"
echo "  - RTX 4090 24GB: ~600 tokens/s (~24s audio/s)"
echo ""
echo "Available voices: en_woman, en_man, belinda, broom_salesman, etc."
echo "See voice_prompts folder for full list"
echo ""

# Build Docker command
DOCKER_CMD="docker run --gpus all --ipc=host --shm-size=20gb --network=host"

# Add custom voice presets if directory is set
if [ -n "${VOICE_PRESETS_DIR:-}" ]; then
    if [ -d "$VOICE_PRESETS_DIR" ]; then
        echo "Using custom voice presets from: $VOICE_PRESETS_DIR"
        DOCKER_CMD="$DOCKER_CMD -v $VOICE_PRESETS_DIR:/voice_presets"
        VOICE_FLAG="--voice-presets-dir /voice_presets"
    else
        echo "Warning: VOICE_PRESETS_DIR set but directory not found: $VOICE_PRESETS_DIR"
        echo "Continuing without custom voice presets..."
        VOICE_FLAG=""
    fi
else
    VOICE_FLAG=""
fi

echo "Starting Docker container..."
echo ""

# Run Higgs Audio vLLM server
$DOCKER_CMD \
    $DOCKER_IMAGE \
    --served-model-name "$MODEL_NAME" \
    --model "bosonai/$MODEL_NAME" \
    --audio-tokenizer-type "bosonai/higgs-audio-v2-tokenizer" \
    --limit-mm-per-prompt audio=50 \
    --max-model-len 8192 \
    --port $PORT \
    --gpu-memory-utilization 0.8 \
    --disable-mm-preprocessor-cache \
    $VOICE_FLAG

# Note: To stop the container, use:
# docker ps  # Find container ID
# docker stop <container_id>

