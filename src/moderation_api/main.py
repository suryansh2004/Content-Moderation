from contextlib import asynccontextmanager

from fastapi import FastAPI

from moderation_api.config import get_settings
from moderation_api.model import ModerationModel, load_model
from moderation_api.schemas import HealthResponse, ModerationRequest, ModerationResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.model = load_model(
        model_dir=settings.model_dir,
        backend=settings.backend,
        max_length=settings.max_length,
        onnx_provider=settings.onnx_provider,
    )
    yield


app = FastAPI(
    title="Real-Time Content Moderation",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    settings = get_settings()
    return HealthResponse(
        status="ok",
        backend=settings.backend,
        model_dir=settings.model_dir,
    )


@app.post("/moderate", response_model=ModerationResponse)
def moderate(request: ModerationRequest) -> ModerationResponse:
    settings = get_settings()
    model: ModerationModel = app.state.model
    threshold = request.threshold if request.threshold is not None else settings.threshold
    results, latency_ms = model.predict(request.texts, threshold=threshold)
    return ModerationResponse(
        backend=model.backend,
        threshold=threshold,
        latency_ms=latency_ms,
        results=results,
    )
