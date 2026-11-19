import os

from fastapi import APIRouter, HTTPException

from retrieval.search import VisualRetriever
from api.schemas import (
    HealthResponse,
    ReportRequest,
    ReportResponse,
    RetrievalRequest, 
    RetrievalResponse
)

from reports.template_generator import generate_report

router = APIRouter()


@router.get("/", tags=["General"])
def home():
    return {
        "message": "Medical AI Chatbot API is running"
    }


@router.get(
    "/health",
    response_model=HealthResponse,
    tags=["General"]
)
def health_check():
    return HealthResponse(
        status="healthy",
        message="Backend is running successfully"
    )


@router.post(
    "/generate-report",
    response_model=ReportResponse,
    tags=["Reports"]
)
def create_report(request: ReportRequest):

    try:
        report = generate_report(
            predictions=request.predictions,
            threshold=request.threshold
        )

        return report

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error)
        )


retriever = None

@router.on_event("startup")
def load_retriever():
    global retriever
    try:
        retriever = VisualRetriever()
    except Exception as error:
        print(f"Retriever initialization failed: {error}")


@router.post(
    "/retrieve",
    response_model=RetrievalResponse,
    tags=["Visual Retrieval"]
)
def retrieve_images(request: RetrievalRequest):

    if retriever is None:
        raise HTTPException(
            status_code=503,
            detail="Visual retrieval system is not available."
        )

    if not os.path.isfile(request.image_path):
        raise HTTPException(
            status_code=404,
            detail="Query image not found."
        )

    try:
        results = retriever.search(
            request.image_path,
            top_k=request.top_k
        )
        return {"results": results}

    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))