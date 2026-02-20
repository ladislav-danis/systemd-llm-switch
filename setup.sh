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
echo "🔗 Patching and linking service files..."
PROJECT_ROOT=$(pwd)

# Patch all .service files with current path
for service in deploy/systemd/*.service; do
    echo "  - Patching and linking $service"
    # Replace the placeholder {{PROJECT_ROOT}}
    # with the actual current directory
    sed -i "s|{{PROJECT_ROOT}}|$PROJECT_ROOT|g" "$service"
    
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