from pydantic import BaseModel, Field


class ModerationRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=128)
    threshold: float | None = Field(default=None, ge=0.0, le=1.0)


class LabelScore(BaseModel):
    label: str
    score: float
    flagged: bool


class ModerationResult(BaseModel):
    text: str
    flagged: bool
    max_score: float
    labels: list[LabelScore]


class ModerationResponse(BaseModel):
    backend: str
    threshold: float
    latency_ms: float
    results: list[ModerationResult]


class HealthResponse(BaseModel):
    status: str
    backend: str
    model_dir: str
