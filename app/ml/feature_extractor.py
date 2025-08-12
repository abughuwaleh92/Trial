from __future__ import annotations
import numpy as np
from typing import Dict, Any, List

F_TYPES = ["sin","cos","tanh","sinh","cosh","exp","log","Ai","gamma","erf","besselj","custom"]

def one_hot(name: str, choices: List[str]) -> List[int]:
    name = (name or "").lower()
    return [1 if name == c.lower() else 0 for c in choices]

def extract_features(payload: Dict[str,Any]) -> tuple[np.ndarray, List[str]]:
    ttype = payload.get("template_type","linear")
    tid = int(payload.get("template_id") or 0)
    f_spec = payload.get("f_spec",{}) or {}
    params = payload.get("params",{}) or {}
    terms = payload.get("terms",[]) or []

    alpha = float(params.get("alpha", 0.0))
    beta  = float(params.get("beta", 0.0))
    M     = float(params.get("M", 0.0))
    a     = float(params.get("a", 1.0))
    q     = float(params.get("q", 2.0))
    v     = float(params.get("v", 3.0))

    if ttype == "general":
        derivs = [int(t.get("deriv",0)) for t in terms] or [0]
        powers = [int(t.get("power",1)) for t in terms] or [1]
        n_terms = len(terms) or 1
        max_deriv = max(derivs); sum_deriv = sum(derivs)
        max_power = max(powers); sum_power = sum(powers)
        nonlin = 1 if max_power > 1 else 0
        uses_delay = 1 if a not in (0.0,1.0) else 0
    else:
        uses_delay = 1 if (ttype=="linear" and tid in (4,5)) or (ttype=="nonlinear" and tid in (4,5,9,10)) else 0
        nonlin = 0 if ttype=="linear" else 1
        max_deriv = 2; sum_deriv = 2
        max_power = 1 if ttype=="linear" else (2 if tid in (1,2,4) else 3)
        sum_power = max_power
        n_terms = 2

    f_name = f_spec.get("name","custom")
    f_oh = one_hot(f_name, F_TYPES)

    feats = [
        alpha, abs(alpha), beta, abs(beta), M, abs(M), a, q, v,
        max_deriv, sum_deriv, max_power, sum_power, n_terms, uses_delay, nonlin
    ] + f_oh
    names = [
        "alpha","|alpha|","beta","|beta|","M","|M|","a","q","v",
        "max_deriv","sum_deriv","max_power","sum_power","n_terms","uses_delay","nonlin"
    ] + [f"f_is_{n}" for n in F_TYPES]
    return np.array(feats, dtype=np.float32), names
