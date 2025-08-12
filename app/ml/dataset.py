from __future__ import annotations
from typing import Dict, Any, Tuple, List
import numpy as np, pandas as pd, os, json, random
from tqdm import tqdm
import sympy as sp
from app.ode.generators import build_f_callable, linear_equation, nonlinear_equation, general_equation, ansatz
from app.ode.solver import try_solve_with_timeout, fingerprint_equation
from app.ml.feature_extractor import extract_features
from app.common.utils import ensure_dir, now_ts

F_POOL_DEFAULT = ["sin","cos","tanh","sinh","cosh","exp","log","Ai","gamma","erf"]

def random_payload(mode: str = "both") -> Dict[str,Any]:
    t = mode
    if mode == "both":
        t = random.choice(["linear","nonlinear"])
    if t == "linear":
        tid = random.choice([1,2,3,4,5])
    elif t == "nonlinear":
        tid = random.choice(list(range(1,11)))
    else: # general
        tid = None
    f_name = random.choice(F_POOL_DEFAULT)
    params = {
        "alpha": random.uniform(-2,2),
        "beta": random.uniform(0.2, 2.0),
        "M": random.uniform(-1,1),
        "a": random.choice([0.5, 1.0, 2.0]),
        "q": random.choice([2,3,4]),
        "v": random.choice([2,3])
    }
    if t == "general":
        n_terms = random.choice([2,3,4])
        terms = []
        for _ in range(n_terms):
            terms.append({"deriv": random.choice([0,1,2,3,4]), "power": random.choice([1,1,2,3]), "coeff": random.choice([1,-1,2,0.5])})
    else:
        terms = []
    return {"template_type": t, "template_id": tid, "f_spec": {"name": f_name, "expr": None, "nu": None}, "params": params, "terms": terms}

def build_equation(payload: Dict[str,Any]) -> sp.Eq:
    fs = payload["f_spec"]; F = build_f_callable(fs["name"], custom_expr=fs.get("expr"), nu=fs.get("nu"))
    t = payload["template_type"]
    if t == "linear":
        return linear_equation(int(payload["template_id"]), F, payload["params"])
    elif t == "nonlinear":
        return nonlinear_equation(int(payload["template_id"]), F, payload["params"])
    else:
        return general_equation(payload.get("terms",[]), F)

def batch_generate(n_samples: int = 200, mode: str = "both", f_pool: list[str] | None = None, timeout_sec: float = 1.5, out_dir: str = "artifacts", name: str = "dataset") -> dict:
    f_pool = f_pool or F_POOL_DEFAULT
    ensure_dir(out_dir)
    rows = []
    for i in tqdm(range(n_samples), desc="Synthesizing ODEs"):
        pld = random_payload(mode)
        # enforce f_pool
        pld["f_spec"]["name"] = random.choice(f_pool)
        try:
            eq = build_equation(pld)
        except Exception as e:
            # skip malformed
            continue
        solv = try_solve_with_timeout(eq, timeout_sec=timeout_sec)
        y = 1 if solv.success else 0
        feats, names = extract_features(pld)
        rows.append({
            "payload": json.dumps(pld),
            "equation_latex": sp.latex(eq),
            "ansatz_latex": sp.latex(sp.Eq(sp.Function("y")(sp.Symbol("x", real=True)), ansatz(build_f_callable(pld["f_spec"]["name"])))),
            "label_solvable": y,
            "elapsed": solv.elapsed,
            "error": solv.error or "",
            "fingerprint": fingerprint_equation(eq),
            **{f"f_{k}": float(v) for k,v in zip(names, feats.tolist())}
        })
    df = pd.DataFrame(rows)
    ts = now_ts()
    csv_path = os.path.join(out_dir, f"{name}-{mode}-{ts}.csv")
    df.to_csv(csv_path, index=False)
    return {"path": csv_path, "n": int(len(df)), "preview": df.head(12).to_dict(orient="records")}
