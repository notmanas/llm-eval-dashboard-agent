from typing import TypedDict, Optional, Dict, Any


class EvalState(TypedDict, total=False):
    prompt: str
    model: str
    response: str
    tokens_used: int
    latency_ms: float
    model_version: str
    metadata: Dict[str, Any]
    ground_truth: str
    hallucination_score: float
    hallucination_verdict: str
    hallucination_debug: Dict[str, Any]
