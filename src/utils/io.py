from __future__ import annotations
import os, json, yaml

def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def ensure_dir(p: str) -> None:
    os.makedirs(os.path.dirname(p), exist_ok=True)

def save_json(obj, path: str) -> None:
    ensure_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
