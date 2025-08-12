from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
import numpy as np, os, json, joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

ARTIFACT_DIR = os.environ.get("ARTIFACT_DIR","artifacts")
os.makedirs(ARTIFACT_DIR, exist_ok=True)
CLF_PATH = os.path.join(ARTIFACT_DIR, "solvability_clf.joblib")
FEATS_PATH = os.path.join(ARTIFACT_DIR, "feature_names.json")

@dataclass
class ClfMetrics:
    acc: float
    auc: float

def train_clf(X: np.ndarray, y: np.ndarray) -> Tuple[HistGradientBoostingClassifier, ClfMetrics]:
    n = len(X); ntr = max(1, int(0.8*n))
    perm = np.random.RandomState(42).permutation(n)
    Xtr, Xte = X[perm[:ntr]], X[perm[ntr:]]
    ytr, yte = y[perm[:ntr]], y[perm[ntr:]]
    clf = HistGradientBoostingClassifier(max_depth=6, learning_rate=0.1, max_iter=400)
    clf.fit(Xtr, ytr)
    p = clf.predict_proba(Xte)[:,1] if len(Xte) else np.array([0.5])
    acc = accuracy_score(yte, (p>=0.5).astype(int)) if len(Xte) else float("nan")
    auc = roc_auc_score(yte, p) if len(Xte) and len(np.unique(yte))>1 else float("nan")
    return clf, ClfMetrics(acc=acc, auc=auc)

def save_clf(clf: HistGradientBoostingClassifier, feature_names: list[str]):
    joblib.dump(clf, CLF_PATH)
    with open(FEATS_PATH,"w") as f: json.dump(feature_names, f)

def load_clf():
    try: return joblib.load(CLF_PATH)
    except Exception: return None
