from __future__ import annotations
import os, json
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException, Header, Depends
import sympy as sp

from app.common.schema import GenerateRequest, TrainRequest, PredictRequest, SuggestRequest, BatchRequest
from app.ode.generators import build_f_callable, linear_equation, nonlinear_equation, general_equation, ansatz
from app.ode.solver import try_solve_with_timeout, fingerprint_equation
from app.ml.feature_extractor import extract_features
from app.ml.dataset import batch_generate, build_equation
from app.ml.models import train_clf, save_clf, load_clf
from app.dl.transformer import load_vocab, encode_terms, decode_terms, train_tiny_transformer, sample_terms
from app.dl.llm import critique_and_suggest, have_llm

API_KEY = os.getenv("API_KEY")
USE_TORCH = os.getenv("USE_TORCH","1") != "0"

app = FastAPI(title="Master Generators API", version="2.0")

def require_key(x_api_key: Optional[str] = Header(default=None)):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")

@app.get("/health")
def health():
    return {"status":"ok","llm": have_llm(), "torch": USE_TORCH}

@app.post("/generate")
def generate(req: GenerateRequest, _=Depends(require_key)):
    F = build_f_callable(req.f_spec.name, custom_expr=req.f_spec.expr, nu=req.f_spec.nu)
    if req.template_type == "linear":
        if req.template_id is None:
            raise HTTPException(400, "template_id required for linear")
        eq = linear_equation(int(req.template_id), F, req.params.dict())
    elif req.template_type == "nonlinear":
        if req.template_id is None:
            raise HTTPException(400, "template_id required for nonlinear")
        eq = nonlinear_equation(int(req.template_id), F, req.params.dict())
    else:
        eq = general_equation([t.dict() for t in req.terms], F)
    solv = try_solve_with_timeout(eq, timeout_sec=2.0)
    return {
        "equation_latex": sp.latex(eq),
        "solution_latex": sp.latex(sp.Eq(sp.Function("y")(sp.Symbol("x", real=True)), ansatz(F))),
        "attempted_dsolve": {"success": solv.success, "elapsed": solv.elapsed, "solution_latex": solv.solution_latex, "error": solv.error},
        "fingerprint": fingerprint_equation(eq),
    }

@app.post("/batch/generate")
def batch(req: BatchRequest, _=Depends(require_key)):
    out = batch_generate(n_samples=req.n_samples, mode=req.template_mix if req.template_mix!="both" else "both",
                         f_pool=req.f_pool, timeout_sec=req.timeout_sec, out_dir=os.environ.get("ARTIFACT_DIR","artifacts"),
                         name=req.dataset_name)
    return out

@app.post("/ml/train")
def ml_train(req: TrainRequest, _=Depends(require_key)):
    import pandas as pd, numpy as np, glob, os
    # Train on the most recent dataset, or synthesize one on the fly
    art = os.environ.get("ARTIFACT_DIR","artifacts")
    files = sorted(glob.glob(os.path.join(art, "*.csv")))
    if not files:
        gen = batch_generate(n_samples=req.n_samples, out_dir=art, name="dataset", timeout_sec=req.timeout_sec)
        files = [gen["path"]]
    df = pd.read_csv(files[-1])
    Xcols = [c for c in df.columns if c.startswith("f_")]
    if not Xcols:
        raise HTTPException(500, "No features found in dataset")
    X = df[Xcols].values.astype("float32")
    y = df["label_solvable"].values.astype("int32")
    clf, metrics = train_clf(X, y)
    from app.ml.feature_extractor import extract_features
    # names are the same as columns order we produced
    save_clf(clf, Xcols)
    return {"acc": metrics.acc, "auc": metrics.auc, "n": int(len(df)), "dataset": files[-1]}

@app.post("/ml/predict")
def ml_predict(req: PredictRequest, _=Depends(require_key)):
    clf = load_clf()
    if clf is None:
        raise HTTPException(400, "No classifier trained. Call /ml/train first.")
    X, names = extract_features(req.payload.dict())
    import numpy as np
    p = float(clf.predict_proba(X.reshape(1,-1))[0,1])
    return {"p_solvable": p}

@app.post("/dl/train")
def dl_train(n_epochs: int = 6, _=Depends(require_key)):
    if not USE_TORCH:
        raise HTTPException(400, "Torch disabled (set USE_TORCH=1).")
    # Build corpus from latest dataset general rows (or synthesize general quickly)
    import pandas as pd, glob, os, json
    art = os.environ.get("ARTIFACT_DIR","artifacts")
    files = sorted(glob.glob(os.path.join(art, "*.csv")))
    if not files:
        from app.ml.dataset import batch_generate
        batch_generate(n_samples=200, mode="general", out_dir=art, name="dataset", timeout_sec=1.0)
        files = sorted(glob.glob(os.path.join(art, "*.csv")))
    df = pd.read_csv(files[-1])
    v = load_vocab()
    corpus = []
    for payload in df["payload"].values.tolist()[:500]:
        try:
            p = json.loads(payload)
        except Exception:
            continue
        if p.get("template_type") != "general": continue
        terms = p.get("terms", [])
        ids = encode_terms(terms, v)
        corpus.append(ids)
    if not corpus:
        raise HTTPException(400, "No general samples found to train on.")
    loss = train_tiny_transformer(corpus, epochs=n_epochs)
    return {"status":"ok","loss": loss}

@app.post("/dl/propose")
def dl_propose(k: int = 10, _=Depends(require_key)):
    if not USE_TORCH:
        raise HTTPException(400, "Torch disabled (set USE_TORCH=1).")
    v = load_vocab()
    seqs = sample_terms(k=k)
    out = []
    for s in seqs:
        terms = [{"deriv":int(t["deriv"]), "power":int(t["power"]), "coeff":float(t["coeff"])} for t in decode_terms(s, v)]
        out.append({
            "template_type":"general",
            "template_id": None,
            "f_spec":{"name":"sin","expr":None,"nu":None},
            "params":{"alpha":0.0,"beta":1.0,"M":0.0,"a":1.0,"q":2.0,"v":3.0},
            "terms": terms
        })
    return {"candidates": out}

@app.post("/llm/critique")
def llm_critique(payloads: List[Dict[str,Any]], _=Depends(require_key)):
    res = critique_and_suggest(payloads)
    return {"results": res}
