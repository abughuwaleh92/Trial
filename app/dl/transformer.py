from __future__ import annotations
import os, json, math, random
from typing import List, Dict, Any, Tuple
import torch, torch.nn as nn
import numpy as np

ARTIFACT_DIR = os.environ.get("ARTIFACT_DIR","artifacts")
os.makedirs(ARTIFACT_DIR, exist_ok=True)
TOK_PATH = os.path.join(ARTIFACT_DIR, "tok.json")
CKPT_PATH = os.path.join(ARTIFACT_DIR, "tiny_transformer.pt")

# Simple tokenization over "general" term triples
# token space: d0..d4, p1..p3, c{-1,0.5,1,2}, plus special tokens
DERIVS = [0,1,2,3,4]
POWERS = [1,2,3]
COEFFS = [-1, 0.5, 1, 2]
SPECIAL = ["<bos>","<eos>","<sep>"]

def build_vocab():
    tokens = []
    for d in DERIVS: tokens.append(f"d{d}")
    for p in POWERS: tokens.append(f"p{p}")
    for c in COEFFS: tokens.append(f"c{str(c).replace('.','_')}")
    tokens += SPECIAL
    stoi = {t:i for i,t in enumerate(tokens)}
    itos = {i:t for t,i in stoi.items()}
    return {"tokens": tokens, "stoi": stoi, "itos": itos}

def save_vocab(v):
    with open(TOK_PATH,"w") as f: json.dump(v, f)

def load_vocab():
    try:
        with open(TOK_PATH) as f: return json.load(f)
    except Exception:
        v = build_vocab(); save_vocab(v); return v

def encode_terms(terms: List[Dict[str,Any]], v) -> List[int]:
    ids = [v["stoi"]["<bos>"]]
    for t in terms:
        ids.append(v["stoi"][f"d{int(t['deriv'])}"])
        ids.append(v["stoi"][f"p{int(t['power'])}"])
        c = t.get("coeff",1)
        c = -1 if c < -0.5 else (0.5 if 0 < c < 1 else (2 if c>1.5 else 1))
        ctag = f"c{str(c).replace('.','_')}"
        ids.append(v["stoi"][ctag])
        ids.append(v["stoi"]["<sep>"])
    ids.append(v["stoi"]["<eos>"])
    return ids

def decode_terms(ids: List[int], v) -> List[Dict[str,Any]]:
    itos = v["itos"]; out = []; cur = {"deriv":0,"power":1,"coeff":1.0}; state = 0
    for i in ids:
        tok = itos[str(i)] if isinstance(i,str) else itos.get(i, "")
        if tok.startswith("d"):
            cur["deriv"] = int(tok[1:]); state = 1
        elif tok.startswith("p"):
            cur["power"] = int(tok[1:]); state = 2
        elif tok.startswith("c"):
            val = tok[1:].replace("_",".")
            cur["coeff"] = float(val)
            state = 3
        elif tok == "<sep>":
            out.append(cur); cur = {"deriv":0,"power":1,"coeff":1.0}; state = 0
        elif tok == "<eos>":
            if state>0: out.append(cur)
            break
    return out

class TinyTransformer(nn.Module):
    def __init__(self, vocab_size: int, n_emb=64, n_heads=4, n_layers=2, block_size=64):
        super().__init__()
        self.block_size = block_size
        self.tok = nn.Embedding(vocab_size, n_emb)
        self.pos = nn.Embedding(block_size, n_emb)
        encoder_layer = nn.TransformerEncoderLayer(d_model=n_emb, nhead=n_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.lm_head = nn.Linear(n_emb, vocab_size)

    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device).unsqueeze(0).expand(B, T)
        h = self.tok(idx) + self.pos(pos)
        h = self.encoder(h)
        return self.lm_head(h)

def train_tiny_transformer(corpus: List[List[int]], epochs: int = 8, lr: float = 2e-3, bs: int = 64, block_size: int = 64, device: str | None = None) -> float:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    vocab_size = max(max(seq) for seq in corpus) + 1
    model = TinyTransformer(vocab_size=vocab_size, block_size=block_size).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    def batch():
        import random
        while True:
            batch_idx = random.sample(range(len(corpus)), min(bs, len(corpus)))
            X = []
            Y = []
            for i in batch_idx:
                seq = corpus[i][:block_size]
                x = seq[:-1]; y = seq[1:]
                X.append(x + [0]*(block_size - len(x)))
                Y.append(y + [0]*(block_size - len(y)))
            yield torch.tensor(X, dtype=torch.long, device=device), torch.tensor(Y, dtype=torch.long, device=device)
    gen = batch()
    model.train()
    last_loss = None
    for ep in range(epochs):
        Xb, Yb = next(gen)
        logits = model(Xb)
        loss = loss_fn(logits.view(-1, logits.size(-1)), Yb.view(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        last_loss = float(loss.item())
    torch.save(model.state_dict(), CKPT_PATH)
    return last_loss

def sample_terms(k: int = 10, max_tokens: int = 30, device: str | None = None) -> List[List[int]]:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    v = load_vocab()
    vocab_size = len(v["tokens"])
    model = TinyTransformer(vocab_size=vocab_size).to(device)
    try:
        sd = torch.load(CKPT_PATH, map_location=device)
        model.load_state_dict(sd)
    except Exception:
        # untrained; return trivial sequences
        bos = v["stoi"]["<bos>"]; eos = v["stoi"]["<eos>"]
        return [[bos, v["stoi"]["d2"], v["stoi"]["p1"], v["stoi"]["c1"], v["stoi"]["<sep>"], eos] for _ in range(k)]
    model.eval()
    out = []
    bos = v["stoi"]["<bos>"]; eos = v["stoi"]["<eos>"]
    for _ in range(k):
        idx = torch.tensor([[bos]], dtype=torch.long, device=device)
        seq = [bos]
        for _t in range(max_tokens):
            logits = model(idx)[:,-1,:]
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1).item()
            seq.append(next_id)
            if next_id == eos: break
            idx = torch.tensor([seq], dtype=torch.long, device=device)
        out.append(seq)
    return out
