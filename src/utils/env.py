import os

def set_cache_env():
    os.environ.setdefault("HF_HOME", "/data/hf_cache")
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/data/hf_cache")
    os.environ.setdefault("TRANSFORMERS_CACHE", "/data/hf_cache")
    os.environ.setdefault("TORCH_HOME", "/data/torch_cache")
    os.environ.setdefault("XDG_CACHE_HOME", "/data/.cache")
    os.environ.setdefault("TMPDIR", "/data/tmp")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("SAFETENSORS_FAST_GPU", "1")
