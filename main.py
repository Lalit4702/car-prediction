from pathlib import Path
from contextlib import asynccontextmanager
from typing import Dict, List, Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from schema import CarFeatures, PredictionResponse
from model import (
    predict_price,
    load_artifacts,
    get_known_cars,
    get_car_defaults,
    get_feature_importances,
    get_dataset
)

BASE_DIR = Path(__file__).resolve().parent
STATIC_INDEX = BASE_DIR / "static" / "index.html"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model and feature artifacts on startup
    load_artifacts()
    yield


app = FastAPI(
    title="AutoValuate AI - Car Price Prediction API",
    description="Machine Learning API for predicting used car selling prices with Random Forest Regression.",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", summary="Web Dashboard or API Root")
def root(request: Request):
    """
    Returns the rich Web UI for browser requests, or JSON overview for API clients.
    """
    accept = request.headers.get("accept", "")
    if "text/html" in accept and STATIC_INDEX.exists():
        return FileResponse(STATIC_INDEX)
    return JSONResponse(
        status_code=200,
        content={
            "app": "AutoValuate AI",
            "version": "2.0.0",
            "status": "online",
            "endpoints": {
                "web_ui": "/",
                "predict": "/predict",
                "health": "/health",
                "cars": "/cars",
                "feature_importance": "/feature-importance",
                "docs": "/docs"
            }
        }
    )


@app.get("/app", response_class=HTMLResponse, summary="Direct Web Application UI")
def serve_web_ui():
    """Serves the standalone modern HTML5 application."""
    if STATIC_INDEX.exists():
        return FileResponse(STATIC_INDEX)
    return HTMLResponse("<h3>Web UI template not found</h3>", status_code=404)


@app.get("/health", summary="Health Check")
def health_check():
    load_artifacts()
    return {
        "status": "healthy",
        "model_loaded": True,
        "service": "AutoValuate Car Price Prediction"
    }


@app.get("/cars", summary="Available Car Models")
def get_cars():
    return {
        "cars": get_known_cars()
    }


@app.get("/cars/{car_name}/defaults", summary="Get Default Specs for a Car")
def car_defaults(car_name: str):
    return get_car_defaults(car_name)


@app.get("/feature-importance", summary="Get Model Feature Importance")
def feature_importance():
    return {
        "features": get_feature_importances()
    }


@app.post("/predict", response_model=PredictionResponse, summary="Predict Car Price")
def predict(features: CarFeatures):
    try:
        price, meta = predict_price(features.model_dump())
        return PredictionResponse(
            prediction_price=price,
            inr_formatted=meta.get("inr_formatted"),
            fair_price_low=meta.get("fair_price_low"),
            fair_price_high=meta.get("fair_price_high"),
            depreciation_pct=meta.get("depreciation_pct"),
            currency="Lakhs (INR)",
            status="success"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
