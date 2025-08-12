#!/usr/bin/env bash
set -euo pipefail
PORT="${PORT:-8501}"
streamlit run app/ui/streamlit_app.py --server.port "$PORT" --server.address 0.0.0.0
