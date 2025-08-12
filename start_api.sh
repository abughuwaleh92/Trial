#!/usr/bin/env bash
set -euo pipefail
PORT="${PORT:-8000}"
export ARTIFACT_DIR="${ARTIFACT_DIR:-artifacts}"
export USE_TORCH="${USE_TORCH:-1}"
uvicorn app.api.main:app --host 0.0.0.0 --port "$PORT"
