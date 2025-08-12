from __future__ import annotations
import os, json, time

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def now_ts() -> str:
    return time.strftime("%Y%m%d-%H%M%S")
