#!/usr/bin/env bash
set -euo pipefail

# Persist heavy caches
export HF_HOME=/data/hf_cache
export HUGGINGFACE_HUB_CACHE=/data/hf_cache
export TRANSFORMERS_CACHE=/data/hf_cache
export TORCH_HOME=/data/torch_cache
export XDG_CACHE_HOME=/data/.cache
export TMPDIR=/data/tmp
export TOKENIZERS_PARALLELISM=false
export SAFETENSORS_FAST_GPU=1

mkdir -p "$HF_HOME" "$TORCH_HOME" "$XDG_CACHE_HOME" "$TMPDIR"

# venv via uv or python
if [ ! -d ".venv" ]; then
  if command -v uv >/dev/null 2>&1; then
    uv venv
  else
    python3 -m venv .venv
  fi
fi

# Activate & install
source .venv/bin/activate
pip install -U pip wheel
pip install -r requirements.txt

echo "✅ Env ready. To use later: source .venv/bin/activate"
