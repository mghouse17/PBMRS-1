from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

from .models import SimConfig
from .simulation import run_sim

app = FastAPI(title="PBMRS API", version="0.2.2")


class RunRequest(BaseModel):
    seed: int = 42
    timesteps: int = 200
    n_agents: int = 200
    q0: float = 0.005


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/run")
def run(request: RunRequest) -> dict:
    cfg = SimConfig(**request.model_dump())
    result = run_sim(cfg)
    return {
        "price_path_length": len(result.prices),
        "return_length": len(result.r),
        "final_price": float(result.prices[-1]),
        "max_drawdown": float(result.prices.min() / max(result.prices.max(), 1e-12) - 1.0),
    }
