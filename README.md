# Master Generators • ODE + ML + DL Lab

End‑to‑end system that:
- Generates infinite ODEs from *master generators* (linear, non‑linear, general).
- Batch synthesizes datasets; attempts symbolic solve (timeout) to label solvability.
- Trains **scikit‑learn** classifier to predict solvability.
- Trains a tiny **Transformer (PyTorch)** to propose novel *general* generators.
- Integrates an **LLM (OpenAI)** to critique novelty/difficulty and suggest variants.
- Ships **API (FastAPI)** and **UI (Streamlit)** services ready for Railway.

## Layout

```
app/
  api/                 # FastAPI app + endpoints
  ui/                  # Streamlit UI
  ode/                 # Symbolic generator builders and helpers
  ml/                  # Dataset synthesis + classic ML
  dl/                  # Transformer + LLM tools
  common/              # Shared pydantic schema and utilities
artifacts/             # Saved models, datasets, logs
```

## Quick start (local)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Terminal 1: API
export API_KEY=yoursecret
export OPENAI_API_KEY=sk-...           # optional
uvicorn app.api.main:app --host 0.0.0.0 --port 8000

# Terminal 2: UI
export API_URL=http://localhost:8000
streamlit run app/ui/streamlit_app.py --server.port 8501 --server.address 0.0.0.0
```

## Railway deployment

Create **two services** from the same repo:

### API service
- Start command:
  ```
  bash start_api.sh
  ```
- Vars: `API_KEY` (required), `OPENAI_API_KEY` (optional), `ARTIFACT_DIR=artifacts`

### UI service
- Start command:
  ```
  bash start_ui.sh
  ```
- Vars: `API_URL=<public URL of API service>`, `API_KEY` (same as API)

> Railway sets `$PORT`. Both start scripts respect that.

## Useful endpoints

- `GET /health`
- `POST /generate` — build one ODE (linear / nonlinear / general)
- `POST /batch/generate` — synthesize many ODEs (dataset)
- `POST /ml/train` — scikit‑learn classifier
- `POST /ml/predict` — predict p(solvable)
- `POST /dl/train` — train tiny Transformer on general terms
- `POST /dl/propose` — propose k new general generators via Transformer
- `POST /llm/critique` — ask LLM to judge novelty/difficulty and suggest edits

## Notes

- Heavy DL is **optional**. Set `USE_TORCH=1` to enable PyTorch features in API.
- LLM is **optional**. Provide `OPENAI_API_KEY` to unlock `/llm/*` features.
- Artifacts (models, datasets) are saved under `artifacts/` by default.
