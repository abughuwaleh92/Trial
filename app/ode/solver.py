from __future__ import annotations
import time, multiprocessing as mp
from dataclasses import dataclass
import sympy as sp

@dataclass
class SolveResult:
    success: bool
    elapsed: float
    solution_latex: str | None
    error: str | None

def _dsolve_worker(eq_expr: sp.Eq, q: mp.Queue):
    try:
        sol = sp.dsolve(eq_expr)
        q.put(("ok", sp.latex(sol)))
    except Exception as e:
        q.put(("err", str(e)))

def try_solve_with_timeout(eq_expr: sp.Eq, timeout_sec: float = 2.0) -> SolveResult:
    q: mp.Queue = mp.Queue()
    p = mp.Process(target=_dsolve_worker, args=(eq_expr, q))
    t0 = time.monotonic()
    p.start()
    p.join(timeout=timeout_sec)
    elapsed = time.monotonic() - t0
    if p.is_alive():
        p.terminate(); p.join()
        return SolveResult(False, elapsed, None, "timeout")
    try:
        status, payload = q.get_nowait()
    except Exception:
        return SolveResult(False, elapsed, None, "no-result")
    if status == "ok":
        return SolveResult(True, elapsed, payload, None)
    return SolveResult(False, elapsed, None, payload)

def fingerprint_equation(eq_expr: sp.Eq) -> str:
    y = sp.Function("y"); x = sp.Symbol("x")
    deriv_orders = []
    for node in sp.preorder_traversal(eq_expr.lhs):
        if isinstance(node, sp.Derivative):
            deriv_orders.append(node.derivative_count)
    for node in sp.preorder_traversal(eq_expr.rhs):
        if isinstance(node, sp.Derivative):
            deriv_orders.append(node.derivative_count)
    h = hash(tuple(sorted(deriv_orders))) ^ hash(eq_expr.lhs.func.__class__.__name__) ^ hash(eq_expr.rhs.func.__class__.__name__)
    return hex((h) & ((1<<64)-1))
