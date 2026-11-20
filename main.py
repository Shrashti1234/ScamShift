"""
FastAPI Model Serving Application
=================================
Phase-1 Step-7: Implement FastAPI and create endpoints.
Serves the deployed Champion model for real-time scam classification.
"""

import pickle
import unicodedata
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent
CHAMPION_PATH = ROOT / "models" / "champion_model.pkl"
active_model = None
THRESHOLD = 0.5


def normalize_text(text: str) -> str:
    """Normalize text by folding unicode and whitespace."""
    if not isinstance(text, str):
        return ""
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def get_model():
    """Load model if not already in memory."""
    global active_model
    if active_model is None and CHAMPION_PATH.exists():
        with open(CHAMPION_PATH, "rb") as f:
            active_model = pickle.load(f)
    return active_model


class ScanRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Message text to analyze for scam intent")


class ScanResponse(BaseModel):
    text: str
    score: float
    threshold: float
    flagged: bool
    prediction: str
    model: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_model()
    yield


app = FastAPI(
    title="ScamShift API",
    description="Production Scam & Spam SMS Detection Service",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
def root():
    return {
        "service": "ScamShift",
        "status": "online",
        "active_model": "champion",
        "threshold": THRESHOLD,
    }


@app.get("/health")
def health():
    model = get_model()
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/predict", response_model=ScanResponse)
@app.post("/api/scan", response_model=ScanResponse)
def predict(request: ScanRequest):
    model = get_model()
    if model is None:
        raise HTTPException(status_code=503, detail="Active model is not loaded. Train champion first.")

    clean_text = normalize_text(request.text)
    if not clean_text:
        raise HTTPException(status_code=422, detail="Text cannot be empty.")

    score = float(model.predict_proba([clean_text])[0, 1])
    flagged = score >= THRESHOLD

    return ScanResponse(
        text=request.text,
        score=round(score, 4),
        threshold=THRESHOLD,
        flagged=flagged,
        prediction="scam" if flagged else "genuine",
        model="champion",
    )
