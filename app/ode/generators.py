from __future__ import annotations
import sympy as sp
from typing import Any, Dict

x = sp.Symbol("x", real=True)
alpha, beta, M, a, q_sym, v_sym = sp.symbols("alpha beta M a q v", real=True)
pi = sp.pi
y = sp.Function("y")

def build_f_callable(name: str, custom_expr: str | None = None, **kwargs):
    z = sp.Symbol("z", complex=True)
    name = (name or "").strip().lower()
    if name == "sin":  return lambda Z: sp.sin(Z)
    if name == "cos":  return lambda Z: sp.cos(Z)
    if name == "tanh": return lambda Z: sp.tanh(Z)
    if name == "sinh": return lambda Z: sp.sinh(Z)
    if name == "cosh": return lambda Z: sp.cosh(Z)
    if name in ("log","ln"): return lambda Z: sp.log(Z)
    if name == "exp":  return lambda Z: sp.exp(Z)
    if name in ("ai","airyai","airy_ai"): return lambda Z: sp.airyai(Z)
    if name == "gamma": return lambda Z: sp.gamma(Z)
    if name == "erf": return lambda Z: sp.erf(Z)
    if name == "besselj":
        nu = kwargs.get("nu", 0)
        return lambda Z: sp.besselj(sp.Integer(nu) if isinstance(nu,int) else sp.sympify(nu), Z)
    # custom
    local_ns = {
        "z": z, "sin": sp.sin, "cos": sp.cos, "tanh": sp.tanh, "sinh": sp.sinh, "cosh": sp.cosh,
        "log": sp.log, "ln": sp.log, "exp": sp.exp, "gamma": sp.gamma, "erf": sp.erf,
        "airyai": sp.airyai, "Ai": sp.airyai, "besselj": sp.besselj, "pi": sp.pi, "E": sp.E
    }
    expr = sp.sympify(custom_expr or "sin(z)", locals=local_ns)
    return lambda Z: expr.subs(z, Z)

def _pieces(F):
    z_beta = alpha + beta
    z_exp = alpha + beta*sp.exp(-x)
    f_ab = F(z_beta)
    f_a_be = F(z_exp)
    df_da = sp.diff(F(z_exp), alpha)
    d2f_da2 = sp.diff(F(z_exp), alpha, 2)
    return f_ab, f_a_be, df_da, d2f_da2

def ansatz(F):  # y(x) = π(f(α+β) − f(α+βe^{-x}) + M)
    f_ab, f_a_be, *_ = _pieces(F)
    return pi*(f_ab - f_a_be + M)

def linear_equation(template_id: int, F, params: Dict[str,Any]) -> sp.Eq:
    f_ab, f_a_be, df_da, d2f_da2 = _pieces(F)
    aval = sp.sympify(params.get("a", a))
    yfun = y(x); y1 = sp.diff(yfun, x); y2 = sp.diff(yfun, x, 2)
    if template_id == 1:
        rhs = pi*(f_ab - f_a_be + M) - pi*(beta*sp.exp(-x)*df_da + beta**2*sp.exp(-2*x)*d2f_da2)
        return sp.Eq(y2 + yfun, rhs)
    if template_id == 2:
        rhs = -pi*(beta**2)*sp.exp(-2*x)*d2f_da2
        return sp.Eq(y2 + y1, rhs)
    if template_id == 3:
        rhs = pi*(f_ab - f_a_be + M + beta*sp.exp(-x)*df_da)
        return sp.Eq(yfun + y1, rhs)
    if template_id == 4:
        z_exp_a = alpha + beta*sp.exp(-x/aval if aval != 0 else -x)
        rhs = pi*(F(alpha + beta*sp.exp(-x)) - F(z_exp_a) - beta*sp.exp(-x)*df_da - (beta**2)*sp.exp(-2*x)*d2f_da2)
        return sp.Eq(y2 + yfun/aval - yfun, rhs)
    if template_id == 5:
        z_exp_a = alpha + beta*sp.exp(-x/aval if aval != 0 else -x)
        rhs = pi*(f_ab - F(z_exp_a) + M + beta*sp.exp(-x)*df_da)
        return sp.Eq(y(x/aval if aval != 0 else x) + sp.diff(yfun, x), rhs)
    raise ValueError("Linear templates 1–5 only (this build).")

def nonlinear_equation(template_id: int, F, params: Dict[str,Any]) -> sp.Eq:
    f_ab, f_a_be, df_da, d2f_da2 = _pieces(F)
    aval = sp.sympify(params.get("a", a)); qv = sp.sympify(params.get("q", q_sym)); vv = sp.sympify(params.get("v", v_sym))
    yfun = y(x); y1 = sp.diff(yfun, x); y2 = sp.diff(yfun, x, 2)
    common = -beta*sp.exp(-x)*df_da - beta**2*sp.exp(-2*x)*d2f_da2
    rhs_pi = pi*(f_ab - f_a_be + M)
    if template_id == 1:
        return sp.Eq((y2)**qv + yfun, rhs_pi + (pi**qv)*(common**qv))
    if template_id == 2:
        return sp.Eq((y2)**qv + (y1)**vv, (pi**qv)*(common**qv) + (pi**vv)*(beta*sp.exp(-x)*df_da)**vv)
    if template_id == 3:
        return sp.Eq(yfun + (y1)**vv, rhs_pi + (pi**vv)*(beta*sp.exp(-x)*df_da)**vv)
    if template_id == 4:
        z_exp_a = alpha + beta*sp.exp(-x/aval if aval != 0 else -x)
        return sp.Eq((y2)**qv + y(x/aval if aval != 0 else x) - yfun, pi*(F(alpha + beta*sp.exp(-x)) - F(z_exp_a)) - (pi**qv)*(common**qv))
    if template_id == 5:
        z_exp_a = alpha + beta*sp.exp(-x/aval if aval != 0 else -x)
        return sp.Eq(y(x/aval if aval != 0 else x) + (y1)**vv, pi*(f_ab - F(z_exp_a)) + (pi**vv)*(beta*sp.exp(-x)*df_da)**vv)
    if template_id == 6:
        return sp.Eq(sp.sin(y2) + yfun, rhs_pi + sp.sin(pi*common))
    if template_id == 7:
        return sp.Eq(sp.exp(y2) + sp.exp(y1), sp.exp(pi*common) + sp.exp(pi*(beta*sp.exp(-x)*df_da)))
    if template_id == 8:
        return sp.Eq(yfun + sp.exp(y1), rhs_pi + sp.exp(pi*(beta*sp.exp(-x)*df_da)))
    if template_id == 9:
        z_exp_a = alpha + beta*sp.exp(-x/aval if aval != 0 else -x)
        return sp.Eq(sp.exp(y2) + y(x/aval if aval != 0 else x) - yfun, pi*(F(alpha + beta*sp.exp(-x)) - F(z_exp_a) + 2*M) + sp.exp(pi*common))
    if template_id == 10:
        z_exp_a = alpha + beta*sp.exp(-x/aval if aval != 0 else -x)
        return sp.Eq(y(x/aval if aval != 0 else x) + sp.log(sp.Abs(sp.diff(yfun, x))), pi*(f_ab - F(z_exp_a) + M) + sp.log(sp.Abs(beta*sp.exp(-x)*df_da)))
    raise ValueError("Unknown nonlinear template id.")

def general_equation(terms: list[dict], F) -> sp.Eq:
    yfun = y(x); lhs = 0
    for t in terms:
        deriv = int(t.get("deriv",0)); power = int(t.get("power",1)); coeff = sp.sympify(t.get("coeff",1))
        term = yfun if deriv == 0 else sp.diff(yfun, x, deriv)
        lhs += coeff * (term ** power)
    rhs = ansatz(F)
    return sp.Eq(lhs, rhs)
