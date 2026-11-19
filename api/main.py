
from fastapi import FastAPI

from api.routes import router

app = FastAPI(
    title="Medical AI Chatbot API",
    description=(
        "Backend API for chest X-ray classification, "
        "visual retrieval, and draft report generation."
    ),
    version="0.1.0"
)

app.include_router(router)
