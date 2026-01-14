#!/bin/bash
set -e

echo "=== VAST.AI SETUP ==="

# 1. System Deps
apt-get update && apt-get install -y python3-venv git build-essential

# 2. Python Environment
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "Created venv"
fi

source venv/bin/activate

# 3. Pip Deps
pip install --upgrade pip
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cu121

echo "=== SETUP COMPLETE ==="
echo "To run experiments:"
echo "source venv/bin/activate"
echo "python -m src.run_eval --config configs/paper_hero.yaml"
