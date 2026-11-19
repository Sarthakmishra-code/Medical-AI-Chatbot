
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    message: str


class ReportRequest(BaseModel):
    predictions: dict[str, float]
    threshold: float = Field(default=0.5, ge=0, le=1)


class ReportResponse(BaseModel):
    title: str
    findings: str
    model_predictions: dict[str, float]
    disclaimer: str


class RetrievalRequest(BaseModel):
    image_path: str
    top_k: int = Field(default=3, ge=1, le=10)


class RetrievalResponse(BaseModel):
    results: list[dict]