#!/bin/bash
# Activate SGLang virtual environment

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="$SCRIPT_DIR/venv"

if [ ! -d "$VENV_PATH" ]; then
    echo "Error: Virtual environment not found at $VENV_PATH"
    echo "Please run one of the SGLang scripts first to set up the environment."
    exit 1
fi

echo "Activating SGLang virtual environment..."
source "$VENV_PATH/bin/activate"

echo "SGLang environment activated!"
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo ""
echo "To verify SGLang installation, run:"
echo "  python -c 'import sglang; print(sglang.__version__)'"

