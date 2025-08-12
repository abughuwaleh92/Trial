from __future__ import annotations
import os, json, requests, streamlit as st

st.set_page_config(page_title="Master Generators • Full Lab", page_icon="∫", layout="wide")

API_URL = os.environ.get("API_URL","").rstrip("/")
API_KEY = os.environ.get("API_KEY")
HEADERS = {"x-api-key": API_KEY} if API_KEY else {}

def call(path: str, payload=None, method="POST", timeout=300):
    if not API_URL:
        raise RuntimeError("API_URL not set")
    url = API_URL + path
    if method=="GET":
        r = requests.get(url, headers=HEADERS, timeout=timeout)
    else:
        r = requests.post(url, json=payload, headers=HEADERS, timeout=timeout)
    r.raise_for_status()
    return r.json()

def healthy():
    try:
        h = call("/health", method="GET")
        st.success(f"API: healthy • LLM={'on' if h.get('llm') else 'off'} / Torch={'on' if h.get('torch') else 'off'}")
    except Exception as e:
        st.error(f"API not reachable: {e}")

st.title("Master Generators • ODE + ML + DL + LLM")
if not API_URL:
    st.warning("Set `API_URL` env var in the UI service.")
else:
    healthy()

# Sidebar config
st.sidebar.header("f(z) and params")
func = st.sidebar.selectbox("f(z)", ["sin","cos","tanh","sinh","cosh","exp","log","Ai","gamma","erf","besselj","custom"], index=0)
nu = st.sidebar.number_input("ν (for BesselJ)", value=0, step=1) if func=="besselj" else None
expr = st.sidebar.text_input("Custom SymPy expression", value="sin(z)") if func=="custom" else None
alpha = st.sidebar.number_input("α", value=0.0)
beta = st.sidebar.number_input("β", value=1.0)
M = st.sidebar.number_input("M", value=0.0)
a = st.sidebar.number_input("a", value=1.0)
q = st.sidebar.number_input("q", value=2.0)
v = st.sidebar.number_input("v", value=3.0)

tabs = st.tabs(["🔧 Single Generate","📦 Batch & Datasets","🧮 ML (Classifier)","🧠 DL (Transformer)","🤖 LLM Studio"])

# -------- Single Generate --------
with tabs[0]:
    st.subheader("Generate one ODE")
    ttype = st.selectbox("Template Type", ["linear","nonlinear","general"], index=0)
    tid = st.selectbox("Template ID", list(range(1,6)) if ttype=="linear" else list(range(1,11)), index=0) if ttype!="general" else None
    terms_txt = None
    if ttype == "general":
        st.caption("Enter terms JSON list. Example shown below.")
        example = [{"deriv":2,"power":1,"coeff":1},{"deriv":0,"power":1,"coeff":1}]
        st.code(json.dumps(example, indent=2), language="json")
        terms_txt = st.text_area("Terms JSON", value=json.dumps(example, indent=2), height=170)

    if st.button("Generate", type="primary"):
        payload = {"template_type": ttype,"template_id": tid,
                   "f_spec":{"name": func, "expr": expr if func=='custom' else None, "nu": nu if func=='besselj' else None},
                   "params":{"alpha":alpha,"beta":beta,"M":M,"a":a,"q":q,"v":v},
                   "terms": []}
        if ttype == "general":
            try:
                payload["terms"] = json.loads(terms_txt or "[]")
            except Exception as e:
                st.error(f"Terms JSON invalid: {e}")
                st.stop()
        with st.spinner("Calling /generate ..."):
            res = call("/generate", payload)
        c1,c2 = st.columns(2)
        with c1:
            st.subheader("Equation")
            st.latex(res["equation_latex"])
            st.caption(f"fingerprint: `{res['fingerprint']}`")
        with c2:
            st.subheader("Ansatz solution")
            st.latex(res["solution_latex"])
        att = res.get("attempted_dsolve",{})
        if att.get("success"):
            st.success(f"Solved (in {att['elapsed']:.3f}s)")
            st.latex(att.get("solution_latex",""))
        else:
            st.warning(f"Not solved (reason: {att.get('error','unknown')} in {att.get('elapsed',0):.3f}s)")

# -------- Batch & Datasets --------
with tabs[1]:
    st.subheader("Synthesize a dataset")
    n = st.number_input("n samples", value=300, min_value=20, max_value=5000, step=20)
    mix = st.selectbox("template mix", ["both","linear","nonlinear","general"], index=0)
    fpool = st.multiselect("f(z) pool", ["sin","cos","tanh","sinh","cosh","exp","log","Ai","gamma","erf"], default=["sin","cos","exp"])
    tsec = st.number_input("dsolve timeout per sample (sec)", value=1.5, min_value=0.2, max_value=6.0, step=0.1)
    name = st.text_input("dataset name", value="dataset")
    if st.button("Start synthesis"):
        req = {"n_samples": int(n), "template_mix": mix, "f_pool": fpool, "timeout_sec": float(tsec), "dataset_name": name}
        with st.spinner("Synthesizing ... this may take a bit"):
            res = call("/batch/generate", req)
        st.success(f"Saved: {res['path']}  |  rows={res['n']}")
        st.dataframe(res["preview"])

# -------- ML (Classifier) --------
with tabs[2]:
    st.subheader("Train solvability classifier")
    ntr = st.number_input("If no dataset, synthesize this many:", value=400, min_value=50, max_value=5000, step=50)
    tsec = st.number_input("Timeout during on-the-fly synthesis (sec)", value=1.5, min_value=0.2, max_value=6.0, step=0.1, key="mlt")
    if st.button("Train /ml/train"):
        with st.spinner("Training ..."):
            res = call("/ml/train", {"n_samples": int(ntr), "timeout_sec": float(tsec)})
        st.success(f"acc={res['acc']:.3f}, auc={res['auc']:.3f}, n={res['n']}")
        st.caption(res['dataset'])

    st.markdown("---")
    st.subheader("Predict p(solvable) for current configuration")
    ttype = st.selectbox("Type", ["linear","nonlinear","general"], index=0, key="pred_ttype")
    tid = st.selectbox("ID", list(range(1,6)) if ttype=="linear" else list(range(1,11)), index=0, key="pred_tid") if ttype!="general" else None
    terms_txt = None
    if ttype == "general":
        terms_txt = st.text_area("Terms JSON", value='[{"deriv":2,"power":1,"coeff":1}]', height=120, key="pred_terms")
    if st.button("Predict /ml/predict"):
        payload = {"template_type": ttype,"template_id": tid,
                   "f_spec":{"name": func, "expr": expr if func=='custom' else None, "nu": nu if func=='besselj' else None},
                   "params":{"alpha":alpha,"beta":beta,"M":M,"a":a,"q":q,"v":v},
                   "terms": json.loads(terms_txt) if (ttype=="general" and terms_txt) else []}
        res = call("/ml/predict", {"payload": payload})
        st.info(f"p(solvable) = {res['p_solvable']:.3f}")

# -------- DL (Transformer) --------
with tabs[3]:
    st.subheader("Train Tiny Transformer on general generators")
    ne = st.number_input("epochs", value=8, min_value=1, max_value=100, step=1)
    if st.button("Train /dl/train"):
        with st.spinner("Training (tiny) ..."):
            res = call("/dl/train?n_epochs=%d" % int(ne))
        st.success(f"done • loss={res['loss']:.4f}")

    st.markdown("---")
    st.subheader("Propose new general generators")
    k = st.number_input("how many?", value=8, min_value=1, max_value=50, step=1)
    if st.button("Propose /dl/propose"):
        res = call("/dl/propose?k=%d" % int(k))
        cands = res.get("candidates", [])
        for i, p in enumerate(cands):
            with st.expander(f"Candidate #{i+1}"):
                st.json(p)
                try:
                    g = call("/generate", p)
                    st.latex(g["equation_latex"])
                    att = g.get("attempted_dsolve",{})
                    st.caption("solved" if att.get("success") else f"unsolved: {att.get('error','')}")
                except Exception as e:
                    st.error(e)

# -------- LLM Studio --------
with tabs[4]:
    st.subheader("Critique and nudge candidates with an LLM")
    st.caption("Paste JSON array of payloads (from Propose or manual). Requires OPENAI_API_KEY on API service.")
    sample = [{
        "template_type":"general",
        "template_id": None,
        "f_spec":{"name":"sin","expr":None,"nu":None},
        "params":{"alpha":0.0,"beta":1.0,"M":0.0,"a":1.0,"q":2.0,"v":3.0},
        "terms":[{"deriv":2,"power":1,"coeff":1},{"deriv":0,"power":1,"coeff":1}]
    }]
    txt = st.text_area("Payloads JSON", value=json.dumps(sample, indent=2), height=220)
    if st.button("Critique /llm/critique"):
        try:
            arr = json.loads(txt)
        except Exception as e:
            st.error(f"Invalid JSON: {e}")
            st.stop()
        with st.spinner("Asking LLM ..."):
            res = call("/llm/critique", arr)
        st.json(res)
