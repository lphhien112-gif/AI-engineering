"""
🚀 FastAPI ML Serving Demo
Chạy: pip install fastapi uvicorn pydantic scikit-learn numpy
       python -m uvicorn fastapi_demo:app --reload
       
Truy cập: http://localhost:8000/docs (Swagger UI)
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
import numpy as np
from datetime import datetime
import time


# ============================================================
# ML Model (Dummy — thay bằng model thật trong production)
# ============================================================

class DummyModel:
    """Simulate a trained ML model."""
    
    def __init__(self):
        self.version = "1.0.0"
        self.trained_at = datetime.now().isoformat()
        # Simulate trained weights
        np.random.seed(42)
        self.weights = np.random.randn(4)
        self.bias = 0.5
    
    def predict(self, features: list[float]) -> dict:
        """Make prediction."""
        x = np.array(features)
        score = float(np.dot(x, self.weights) + self.bias)
        probability = 1 / (1 + np.exp(-score))  # Sigmoid
        label = "positive" if probability > 0.5 else "negative"
        return {
            "label": label,
            "confidence": round(probability, 4),
            "raw_score": round(score, 4),
        }


# ============================================================
# Lifespan (load model once on startup)
# ============================================================

model: DummyModel = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load ML model when API starts, cleanup when stops."""
    global model
    print("📥 Loading ML model...")
    model = DummyModel()
    print(f"✅ Model v{model.version} loaded")
    yield
    print("🧹 Cleaning up...")


# ============================================================
# FastAPI App
# ============================================================

app = FastAPI(
    title="🤖 ML Prediction API",
    version="1.0.0",
    description="Production-ready ML model serving with FastAPI",
    lifespan=lifespan,
)

# Request/Response schemas
class PredictionRequest(BaseModel):
    features: list[float] = Field(
        ..., 
        min_length=4, max_length=4,
        description="4 numeric features",
        json_schema_extra={"examples": [[0.5, -0.3, 1.2, 0.8]]}
    )

class PredictionResponse(BaseModel):
    label: str
    confidence: float
    raw_score: float
    model_version: str
    inference_time_ms: float

class HealthResponse(BaseModel):
    status: str
    model_version: str
    uptime_seconds: float

class BatchRequest(BaseModel):
    inputs: list[list[float]] = Field(
        ..., max_length=100,
        description="Batch of feature vectors (max 100)"
    )


# Track uptime
start_time = time.time()


# ============================================================
# Endpoints
# ============================================================

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Liveness probe — check if API is running."""
    return HealthResponse(
        status="healthy",
        model_version=model.version if model else "not loaded",
        uptime_seconds=round(time.time() - start_time, 2),
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(request: PredictionRequest):
    """Single prediction endpoint."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    start = time.perf_counter()
    result = model.predict(request.features)
    inference_time = (time.perf_counter() - start) * 1000
    
    return PredictionResponse(
        label=result["label"],
        confidence=result["confidence"],
        raw_score=result["raw_score"],
        model_version=model.version,
        inference_time_ms=round(inference_time, 3),
    )


@app.post("/predict/batch", tags=["Prediction"])
async def batch_predict(request: BatchRequest):
    """Batch prediction — up to 100 inputs."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    start = time.perf_counter()
    results = []
    for features in request.inputs:
        if len(features) != 4:
            raise HTTPException(
                status_code=422, 
                detail=f"Each input must have exactly 4 features, got {len(features)}"
            )
        results.append(model.predict(features))
    
    inference_time = (time.perf_counter() - start) * 1000
    
    return {
        "predictions": results,
        "count": len(results),
        "total_inference_time_ms": round(inference_time, 3),
        "avg_inference_time_ms": round(inference_time / len(results), 3),
    }


@app.get("/model/info", tags=["Model"])
async def model_info():
    """Get model metadata."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "version": model.version,
        "trained_at": model.trained_at,
        "input_features": 4,
        "output_type": "binary_classification",
        "classes": ["negative", "positive"],
    }


# ============================================================
# Error handling
# ============================================================

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    from fastapi.responses import JSONResponse
    return JSONResponse(
        status_code=500,
        content={
            "error": "InternalServerError",
            "message": str(exc),
            "path": str(request.url),
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
