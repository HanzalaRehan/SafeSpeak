#!/usr/bin/env bash
# SafeSpeak — macOS M4 setup script
# Run once after cloning / replacing files:  bash setup.sh
set -e

echo "═══════════════════════════════════════"
echo "  SafeSpeak — macOS Setup"
echo "═══════════════════════════════════════"

# 1. Create directory layout
mkdir -p inputs outputs config

# 2. Install Python deps
echo ""
echo "→ Installing Python requirements..."
pip install -r requirements.txt

echo ""
echo "✓ Setup complete."
echo ""
echo "To start SafeSpeak:"
echo "  python main.py"
echo ""
echo "Then open index.html in your browser for the dashboard."
