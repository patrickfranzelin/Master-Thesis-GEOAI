#!/bin/bash
set -euo pipefail

echo "=== Setting up Master-Thesis-GEOAI environment ==="

# install uv if missing
if ! command -v uv &>/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

# set up cache paths (works local + container)
source ./cache_env.sh

# create venv
uv venv --python 3.10

# activate
source .venv/bin/activate

# install requirements
uv pip install -r requirements.txt

echo "✅ Environment ready! Activate with: source .venv/bin/activate"
