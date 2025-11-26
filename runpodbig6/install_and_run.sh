#!/bin/bash

# Define variables
BUNDLE_FILE="compile_bundle.tar.gz"

# 1. Install dependencies (tmux, make, tar)
echo "Installing dependencies..."
if [ -f /etc/debian_version ]; then
    sudo apt-get update
    sudo apt-get install -y tmux make tar
elif [ -f /etc/redhat-release ]; then
    sudo yum install -y tmux make tar
else
    echo "Warning: Unknown OS. Ensure tmux, make, and tar are installed."
fi

# 2. Extract Bundle
if [ -f "$BUNDLE_FILE" ]; then
    echo "Extracting $BUNDLE_FILE..."
    # We used -P (absolute paths) during creation, so we use -P here to restore to /compile/...
    sudo tar -P -xzf "$BUNDLE_FILE"
    echo "Extraction complete."
else
    echo "Warning: $BUNDLE_FILE not found in current directory. Assuming files are already in place."
fi

# Ensure log directory exists
sudo mkdir -p /compile/logs

# 3. Run commands in tmux
SESSION_NAME="compile_services"

# Check if session exists
tmux has-session -t $SESSION_NAME 2>/dev/null

if [ $? != 0 ]; then
    echo "Starting new tmux session: $SESSION_NAME"
    
    # Create the session and first window (sglang)
    tmux new-session -d -s $SESSION_NAME -n "sglang"
    # We need to ensure we are in the correct directory
    tmux send-keys -t $SESSION_NAME:sglang "cd /compile/vlm/sglang && make serve" C-m

    # Create second window (granite)
    tmux new-window -t $SESSION_NAME -n "granite"
    tmux send-keys -t $SESSION_NAME:granite "/compile/llm/vllm/run_granite_micro_dense.sh" C-m

    # Create third window (qwen)
    tmux new-window -t $SESSION_NAME -n "qwen"
    tmux send-keys -t $SESSION_NAME:qwen "/compile/llm/vllm/run_vllm_qwen3_embedding.sh" C-m

    echo "Services started in tmux session '$SESSION_NAME'."
    echo "Attach to it using: tmux attach -t $SESSION_NAME"
else
    echo "Session $SESSION_NAME already exists. Attaching..."
fi
