#!/bin/bash
# =============================================================================
# Apple Silicon Setup Script for Cricket Match Predictor
# =============================================================================
# This script sets up a Python 3.11 virtual environment with TensorFlow Metal
# GPU acceleration for Apple Silicon Macs (M1/M2/M3).
#
# Usage: ./scripts/setup_apple_silicon.sh
# =============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=============================================${NC}"
echo -e "${BLUE}  Cricket Match Predictor - Apple Silicon Setup${NC}"
echo -e "${BLUE}=============================================${NC}"
echo ""

# Check if we're on macOS
if [[ "$(uname)" != "Darwin" ]]; then
    echo -e "${RED}Error: This script is for macOS only.${NC}"
    echo "For other platforms, use: pip install -r requirements.txt"
    exit 1
fi

# Check if we're on Apple Silicon
ARCH=$(uname -m)
if [[ "$ARCH" != "arm64" ]]; then
    echo -e "${YELLOW}Warning: This script is optimized for Apple Silicon (arm64).${NC}"
    echo "Your architecture: $ARCH"
    echo "Metal GPU acceleration may not be available."
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check for Homebrew
if ! command -v brew &> /dev/null; then
    echo -e "${RED}Error: Homebrew is not installed.${NC}"
    echo "Install it from: https://brew.sh"
    exit 1
fi

echo -e "${GREEN}✓ Homebrew found${NC}"

# Check for Python 3.11
PYTHON311=""
if command -v python3.11 &> /dev/null; then
    PYTHON311=$(which python3.11)
    echo -e "${GREEN}✓ Python 3.11 found: $PYTHON311${NC}"
elif [ -f "/opt/homebrew/bin/python3.11" ]; then
    PYTHON311="/opt/homebrew/bin/python3.11"
    echo -e "${GREEN}✓ Python 3.11 found: $PYTHON311${NC}"
elif [ -f "/usr/local/bin/python3.11" ]; then
    PYTHON311="/usr/local/bin/python3.11"
    echo -e "${GREEN}✓ Python 3.11 found: $PYTHON311${NC}"
else
    echo -e "${YELLOW}Python 3.11 not found. Installing via Homebrew...${NC}"
    brew install python@3.11
    
    # Find the installed Python
    if [ -f "/opt/homebrew/bin/python3.11" ]; then
        PYTHON311="/opt/homebrew/bin/python3.11"
    elif [ -f "/usr/local/bin/python3.11" ]; then
        PYTHON311="/usr/local/bin/python3.11"
    else
        echo -e "${RED}Error: Could not find Python 3.11 after installation.${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓ Python 3.11 installed: $PYTHON311${NC}"
fi

# Verify Python version
PY_VERSION=$($PYTHON311 --version 2>&1)
echo -e "${BLUE}Python version: $PY_VERSION${NC}"

# Get project root (parent of scripts directory)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo -e "${BLUE}Project root: $PROJECT_ROOT${NC}"

# Create virtual environment
VENV_DIR="$PROJECT_ROOT/venv311"
if [ -d "$VENV_DIR" ]; then
    echo -e "${YELLOW}Virtual environment already exists at $VENV_DIR${NC}"
    read -p "Delete and recreate? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "$VENV_DIR"
    else
        echo "Using existing environment."
    fi
fi

if [ ! -d "$VENV_DIR" ]; then
    echo -e "${BLUE}Creating virtual environment with Python 3.11...${NC}"
    $PYTHON311 -m venv "$VENV_DIR"
    echo -e "${GREEN}✓ Virtual environment created at $VENV_DIR${NC}"
fi

# Activate virtual environment
echo -e "${BLUE}Activating virtual environment...${NC}"
source "$VENV_DIR/bin/activate"

# Upgrade pip
echo -e "${BLUE}Upgrading pip...${NC}"
pip install --upgrade pip

# Install Apple Silicon requirements
echo -e "${BLUE}Installing Apple Silicon optimized packages...${NC}"
echo "This may take a few minutes..."
pip install -r requirements-apple-silicon.txt

# Install Playwright browsers (for CREX scraping)
echo -e "${BLUE}Installing Playwright browsers...${NC}"
python -m playwright install chromium

# Verify TensorFlow Metal
echo ""
echo -e "${BLUE}=============================================${NC}"
echo -e "${BLUE}  Verifying TensorFlow Metal Installation${NC}"
echo -e "${BLUE}=============================================${NC}"
echo ""

python << 'EOF'
import tensorflow as tf
import sys

print(f"TensorFlow version: {tf.__version__}")
print(f"Python version: {sys.version}")

# Check for GPU devices
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"\n✅ Metal GPU detected: {gpus}")
    print("\nGPU acceleration is ENABLED!")
    print("Expected performance: ~400-600 simulations/second")
else:
    print("\n⚠️  No GPU detected. TensorFlow will run on CPU.")
    print("This is normal if tensorflow-metal failed to install.")
    print("\nTroubleshooting:")
    print("1. Ensure you're using Python 3.11 (not 3.12 or 3.13)")
    print("2. Try: pip install tensorflow-metal --upgrade")
    print("3. Restart your terminal and try again")

# Test basic TensorFlow operation
print("\nRunning quick GPU test...")
try:
    with tf.device('/GPU:0' if gpus else '/CPU:0'):
        a = tf.random.normal([1000, 1000])
        b = tf.random.normal([1000, 1000])
        c = tf.matmul(a, b)
    print("✅ TensorFlow computation test passed!")
except Exception as e:
    print(f"❌ TensorFlow test failed: {e}")
EOF

echo ""
echo -e "${GREEN}=============================================${NC}"
echo -e "${GREEN}  Setup Complete!${NC}"
echo -e "${GREEN}=============================================${NC}"
echo ""
echo "To activate this environment, run:"
echo -e "  ${BLUE}source venv311/bin/activate${NC}"
echo ""
echo "To start the web app:"
echo -e "  ${BLUE}python app/main.py${NC}"
echo ""
echo "To run simulations with GPU acceleration:"
echo -e "  ${BLUE}Visit http://localhost:5001/predict${NC}"
echo ""
