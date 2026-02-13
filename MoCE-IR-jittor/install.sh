#!/bin/bash
# MoCE-IR Jittor Version Installation Script

set -e  # Exit on error

echo "========================================"
echo "Installing MoCE-IR (Jittor Version)"
echo "========================================"

# Install Jittor
echo "[1/3] Installing Jittor..."
pip install jittor>=1.3.0

# Install dependencies
echo "[2/3] Installing dependencies..."
pip install -r requirements.txt

# Verify installation
echo "[3/3] Verifying installation..."
python -c "import jittor as jt; print(f'Jittor version: {jt.__version__}')"

echo "========================================"
echo "Installation completed successfully!"
echo "========================================"

