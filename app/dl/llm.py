from __future__ import annotations
import os, json
from typing import Dict, Any, List
from openai import OpenAI

def have_llm() -> bool:
    return bool(os.getenv("OPENAI_API_KEY"))

def critique_and_suggest(payloads: List[Dict[str,Any]]) -> List[Dict[str,Any]]:
    if not have_llm():
        return [{"error":"OPENAI_API_KEY not set"}]
    client = OpenAI()
    sys = "You are an expert in differential equations. Evaluate the novelty and difficulty of ODE generators. Score difficulty (0-1) and suggest a small edit to make it slightly harder but still analyzable."
    user = "Here are JSON payloads describing ODE generator specs (template_type/id, f_spec, params, terms). Return JSON list with fields: difficulty, novelty, suggestion."
    content = json.dumps(payloads, indent=2)
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role":"system","content":sys},{"role":"user","content":user + "\n" + content}],
        temperature=0.4,
    )
    try:
        txt = resp.choices[0].message.content
        data = json.loads(txt)
        return data if isinstance(data, list) else [{"error":"LLM did not return JSON"}]
    except Exception as e:
        return [{"error": str(e)}]
