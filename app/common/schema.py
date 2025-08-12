from __future__ import annotations
from typing import Optional, List
from pydantic import BaseModel, Field

class FSpec(BaseModel):
    name: str = "sin"
    expr: Optional[str] = None
    nu: Optional[float] = None

class TermSpec(BaseModel):
    deriv: int
    power: int = 1
    coeff: float = 1.0

class Params(BaseModel):
    alpha: float = 0.0
    beta: float = 1.0
    M: float = 0.0
    a: float = 1.0
    q: float = 2.0
    v: float = 3.0

class GenerateRequest(BaseModel):
    template_type: str = Field(pattern="^(linear|nonlinear|general)$")
    template_id: Optional[int] = None
    f_spec: FSpec = FSpec()
    params: Params = Params()
    terms: List[TermSpec] = []

class TrainRequest(BaseModel):
    n_samples: int = 400
    timeout_sec: float = 1.5

class PredictRequest(BaseModel):
    payload: GenerateRequest

class SuggestRequest(BaseModel):
    n: int = 10

class BatchRequest(BaseModel):
    n_samples: int = 500
    template_mix: str = Field(default="both", pattern="^(linear|nonlinear|both|general)$")
    f_pool: Optional[list[str]] = None
    timeout_sec: float = 1.5
    save_csv: bool = True
    dataset_name: str = "dataset"
