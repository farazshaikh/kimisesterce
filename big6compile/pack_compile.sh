#!/bin/bash

# Output file
OUTPUT_TAR="compile_bundle.tar.gz"

echo "Creating $OUTPUT_TAR..."

# Create tarball including:
# 1. /compile/vlm/sglang (The directory itself)
# 2. /compile/llm/vllm (The directory itself)
# We exclude venv directories as they are usually not portable.
# We use -P to preserve absolute paths so they extract to /compile/... on the remote.

tar -P -czf $OUTPUT_TAR \
    --exclude='*/venv' \
    --exclude='*/__pycache__' \
    --exclude='*.pyc' \
    /compile/vlm/sglang \
    /compile/llm/vllm

echo "Bundle created: $OUTPUT_TAR"
echo "Size: $(du -h $OUTPUT_TAR | cut -f1)"

