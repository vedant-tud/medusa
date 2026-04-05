#!/bin/bash

# setup.sh — Automated environment setup for Medusa emotion recognition pipeline
# Usage: bash setup.sh

set -e  # Exit on any error

echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║         Medusa Pipeline — Environment Setup Script            ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""

# Check Python version
echo "[1/5] Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "      Found Python $python_version"
if ! python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)'; then
    echo "      ❌ ERROR: Python 3.10+ required. Current: $python_version"
    exit 1
fi
echo "      ✓ Python 3.10+ confirmed"
echo ""

# Create virtual environment
echo "[2/5] Setting up virtual environment..."
if [ -d "venv" ]; then
    echo "      Virtual environment already exists. Skipping..."
else
    python3 -m venv venv
    echo "      ✓ Virtual environment created"
fi
source venv/bin/activate
echo "      ✓ Activated: venv"
echo ""

# Upgrade pip
echo "[3/5] Upgrading pip..."
pip install --upgrade pip setuptools wheel --quiet
echo "      ✓ pip upgraded"
echo ""

# Install PyTorch with CUDA 11.8
echo "[4/5] Installing PyTorch (CUDA 11.8)..."
echo "      This may take 2-3 minutes..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --quiet
echo "      ✓ PyTorch installed"
echo ""

# Install dependencies
echo "[5/5] Installing dependencies from requirements.txt..."
pip install -r requirements.txt --quiet
echo "      ✓ All dependencies installed"
echo ""

echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║                    Setup Complete! ✓                          ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""
echo "Next steps:"
echo "  1. Verify installation: python backend/test_imports.py"
echo "  2. Test pipeline:       python backend/test_inference.py --video <path_to_video>"
echo ""
echo "For more details, see README.md"
