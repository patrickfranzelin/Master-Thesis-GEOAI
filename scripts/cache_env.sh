#!/bin/bash
set -euo pipefail

# -----------------------------------------
# Configure persistent caches for large models
# Works on local machines AND containers
# -----------------------------------------

# Default to /data if it exists (e.g. RunPod),
# otherwise fallback to the project directory.
if [ -d "/data" ]; then
  CACHE_ROOT="/data"
else
  CACHE_ROOT="$(pwd)/.cache"
fi

export HF_HOME="${CACHE_ROOT}/hf_cache"
export HUGGINGFACE_HUB_CACHE="${CACHE_ROOT}/hf_cache"
export TRANSFORMERS_CACHE="${CACHE_ROOT}/hf_cache"
export TORCH_HOME="${CACHE_ROOT}/torch_cache"
export XDG_CACHE_HOME="${CACHE_ROOT}/.cache"
export TMPDIR="${CACHE_ROOT}/tmp"
export TOKENIZERS_PARALLELISM=false
export SAFETENSORS_FAST_GPU=1

# Ensure directories exist
mkdir -p "$HF_HOME" "$TORCH_HOME" "$XDG_CACHE_HOME" "$TMPDIR"

echo "✅ Cache directories configured under: $CACHE_ROOT"
echo "HF cache: $HF_HOME"
echo "Torch cache: $TORCH_HOME"
echo "Temp dir: $TMPDIR"
