#!/bin/bash
set -e

echo "🔧 Setting up SYSTEMD-LLM-SWITCH..."

# 1. Enable linger (services will continue to run even after the user logs out)
sudo loginctl enable-linger $USER

# 2. Creating a directory for user-level systemd units
mkdir -p ~/.config/systemd/user/

# 3. Virtual environment and dependency installation
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi
source .venv/bin/activate
pip install --upgrade pip
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
fi

# 4. Connecting ALL services (so that they can be run via systemctl)
echo "🔗 Configuring and linking service files..."
PROJECT_ROOT=$(pwd)

# Default values
DEFAULT_LLAMA_SERVER="$HOME/Develop/llama.cpp/build/bin/llama-server"
DEFAULT_MODELS_DIR="$HOME/.cache/llama.cpp"

# Ask for llama-server path if not already set
if [ -z "$LLAMA_SERVER_PATH" ]; then
    read -p "📂 Path to your llama-server binary [default: $DEFAULT_LLAMA_SERVER]: " LLAMA_SERVER_PATH
    LLAMA_SERVER_PATH=${LLAMA_SERVER_PATH:-$DEFAULT_LLAMA_SERVER}
fi

# Ask for models directory if not already set
if [ -z "$MODELS_DIR" ]; then
    read -p "📂 Directory where your GGUF models are stored [default: $DEFAULT_MODELS_DIR]: " MODELS_DIR
    MODELS_DIR=${MODELS_DIR:-$DEFAULT_MODELS_DIR}
fi

# Expand ~ to full path if necessary
LLAMA_SERVER_PATH="${LLAMA_SERVER_PATH/#\~/$HOME}"
MODELS_DIR="${MODELS_DIR/#\~/$HOME}"
LLAMA_BIN_DIR=$(dirname "$LLAMA_SERVER_PATH")

# Patch all .service files with current paths
for service in deploy/systemd/*.service; do
    echo "  - Patching and linking $service"
    # Replace placeholders
    sed -i "s|{{PROJECT_ROOT}}|$PROJECT_ROOT|g" "$service"
    sed -i "s|{{LLAMA_SERVER_PATH}}|$LLAMA_SERVER_PATH|g" "$service"
    sed -i "s|{{MODELS_DIR}}|$MODELS_DIR|g" "$service"
    sed -i "s|{{LLAMA_BIN_DIR}}|$LLAMA_BIN_DIR|g" "$service"
    
    # Create symlink in systemd user directory
    ln -sf "$PROJECT_ROOT/$service" ~/.config/systemd/user/
done

systemctl --user daemon-reload

# 5. Activation of ONLY the main proxy service
if [ -f "~/.config/systemd/user/llm-switch.service" ] || [ -f "deploy/systemd/llm-switch.service" ]; then
    echo "🚀 Enabling core llm-switch.service..."
    systemctl --user enable llm-switch.service
    systemctl --user start llm-switch.service
fi

echo "---"
echo "✅ Done! Only llm-switch.service is enabled to start on boot."
echo "✅ Other models are linked and ready for main.py to manage them."
echo "💡 Check status: systemctl --user status llm-switch.service"