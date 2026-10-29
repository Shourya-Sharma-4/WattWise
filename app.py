"""
WattWise - ML-Based Electricity Consumption Anomaly Detection System
FastAPI Backend: app.py

Provides production-ready REST API endpoints for energy anomaly detection,
readings data streaming, summary analytics, and dashboard static file serving.
"""

import os
import json
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from predict import predict_reading, get_model_bundle

# Path definitions relative to project directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "electricity_readings.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "wattwise_model.joblib")
METADATA_PATH = os.path.join(BASE_DIR, "models", "model_metadata.json")
STATIC_DIR = os.path.join(BASE_DIR, "static")

# Cached dataset for fast API response
_DATASET_CACHE: Optional[pd.DataFrame] = None
_METADATA_CACHE: Optional[dict] = None


def load_dataset() -> pd.DataFrame:
    """Loads and caches the electricity readings dataset."""
    global _DATASET_CACHE
    if _DATASET_CACHE is None:
        if not os.path.exists(DATA_PATH):
            import generate_data
            generate_data.main()
        df = pd.read_csv(DATA_PATH)
        # Ensure boolean/numeric types
        df["is_anomaly_simulated"] = df["is_anomaly_simulated"].astype(int)
        df["power_consumption"] = df["power_consumption"].astype(float)
        df["baseline_consumption"] = df["baseline_consumption"].astype(float)
        _DATASET_CACHE = df
    return _DATASET_CACHE


def load_metadata() -> dict:
    """Loads model metadata from JSON artifact."""
    global _METADATA_CACHE
    if _METADATA_CACHE is None:
        if os.path.exists(METADATA_PATH):
            with open(METADATA_PATH, "r") as f:
                _METADATA_CACHE = json.load(f)
        else:
            _METADATA_CACHE = {
                "model_name": "Isolation Forest Anomaly Detector",
                "status": "Awaiting initial training run",
            }
    return _METADATA_CACHE


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event to ensure model and data are initialized on startup."""
    # Ensure data and model exist
    if not os.path.exists(DATA_PATH):
        print("Dataset not found. Generating synthetic readings...", flush=True)
        import generate_data
        generate_data.main()
    
    if not os.path.exists(MODEL_PATH):
        print("Model not found. Executing training pipeline...", flush=True)
        import train
        train.train_and_evaluate()
        
    # Preload caches
    load_dataset()
    load_metadata()
    get_model_bundle()
    print("WattWise backend initialized successfully.", flush=True)
    yield


app = FastAPI(
    title="WattWise API",
    description="ML-Based Electricity Consumption Anomaly Detection System using Isolation Forest",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for external access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files directory
if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# Pydantic Schemas
class PredictionRequest(BaseModel):
    hour: int = Field(..., ge=0, le=23, description="Hour of day (0-23)", example=14)
    day_of_week: int = Field(..., ge=0, le=6, description="Day of week (0=Monday, 6=Sunday)", example=2)
    appliance: str = Field(..., description="Appliance category (e.g. HVAC, Refrigeration, EV Charger)", example="HVAC")
    power_consumption: float = Field(..., ge=0.0, description="Measured power consumption in Watts", example=1850.0)
    baseline_consumption: float = Field(..., gt=0.0, description="Expected baseline consumption in Watts", example=1800.0)


class PredictionResponse(BaseModel):
    status: str
    is_anomaly: bool
    anomaly_score: float
    anomaly_score_percentage: float
    risk_level: str
    features: Dict[str, Any]
    explanation: str
    disclaimer: str


# =====================================================================
# API Endpoints
# =====================================================================

@app.get("/", summary="Root endpoint - Dashboard or Project Information")
async def root(request: Request):
    """
    Returns the WattWise interactive dashboard for browser visitors,
    or JSON project information when requested via JSON client.
    """
    accept_header = request.headers.get("accept", "")
    index_file = os.path.join(STATIC_DIR, "index.html")

    # If client explicitly wants JSON or index.html is missing
    if "application/json" in accept_header and not ("text/html" in accept_header):
        return await project_info()

    if os.path.exists(index_file):
        return FileResponse(index_file)
    
    return await project_info()


@app.get("/api/info", summary="Project Information")
async def project_info():
    """Returns project details, algorithm details, and system architecture."""
    meta = load_metadata()
    return {
        "project": "WattWise — ML-Based Electricity Consumption Anomaly Detection System",
        "description": "Unsupervised machine learning system detecting anomalous electricity consumption using Isolation Forest.",
        "algorithm": "Isolation Forest (scikit-learn)",
        "features": [
            "Hour of day",
            "Day of week",
            "Appliance category",
            "Power consumption (W)",
            "Baseline consumption (W)",
            "Consumption ratio & difference"
        ],
        "model_metadata": meta,
        "endpoints": {
            "dashboard": "/",
            "health": "/health",
            "summary": "/api/summary",
            "readings": "/api/readings",
            "predict": "/api/predict",
            "docs": "/docs"
        },
        "disclaimer": "WattWise identifies statistical irregularities in power readings. It is not a hardware fault diagnosis tool."
    }


@app.get("/health", summary="Health check endpoint")
async def health_check():
    """Returns application health status, model readiness, and dataset count."""
    try:
        model_ready = get_model_bundle() is not None
        df = load_dataset()
        return {
            "status": "healthy",
            "model_loaded": model_ready,
            "readings_count": len(df),
            "version": "1.0.0"
        }
    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={"status": "unhealthy", "error": str(exc)}
        )


@app.get("/api/summary", summary="Readings summary & anomaly statistics")
async def get_summary():
    """
    Aggregates overall consumption metrics, anomaly count, anomaly percentage,
    and appliance-level distribution for dashboard KPI cards and charts.
    """
    df = load_dataset()
    total_readings = int(len(df))
    anomaly_mask = df["is_anomaly_simulated"] == 1
    anomaly_count = int(anomaly_mask.sum())
    normal_count = total_readings - anomaly_count
    anomaly_percentage = round((anomaly_count / total_readings) * 100.0, 2) if total_readings > 0 else 0.0

    avg_consumption = round(float(df["power_consumption"].mean()), 2)
    avg_baseline = round(float(df["baseline_consumption"].mean()), 2)
    max_consumption = round(float(df["power_consumption"].max()), 2)
    min_consumption = round(float(df["power_consumption"].min()), 2)

    # Anomaly counts grouped by appliance
    by_appliance = []
    for app_name, group in df.groupby("appliance"):
        app_total = len(group)
        app_anom = int((group["is_anomaly_simulated"] == 1).sum())
        by_appliance.append({
            "appliance": app_name,
            "total_readings": app_total,
            "anomalies": app_anom,
            "anomaly_rate": round((app_anom / app_total) * 100.0, 2) if app_total > 0 else 0.0,
            "avg_consumption": round(float(group["power_consumption"].mean()), 1),
            "avg_baseline": round(float(group["baseline_consumption"].mean()), 1),
        })

    # Hourly distribution of anomalies (0-23)
    hourly_anomalies = [0] * 24
    for hour, group in df.groupby("hour"):
        if 0 <= hour < 24:
            hourly_anomalies[int(hour)] = int((group["is_anomaly_simulated"] == 1).sum())

    meta = load_metadata()

    return {
        "total_readings": total_readings,
        "anomaly_count": anomaly_count,
        "normal_count": normal_count,
        "anomaly_percentage": anomaly_percentage,
        "average_consumption_watts": avg_consumption,
        "average_baseline_watts": avg_baseline,
        "max_consumption_watts": max_consumption,
        "min_consumption_watts": min_consumption,
        "by_appliance": by_appliance,
        "hourly_anomalies": hourly_anomalies,
        "model_metrics": meta.get("metrics", {}),
        "data_source": "Simulated Realistic Electricity Telemetry"
    }


@app.get("/api/readings", summary="Fetch electricity readings for charts & tables")
async def get_readings(
    limit: int = Query(default=100, ge=1, le=1000, description="Number of records to return"),
    offset: int = Query(default=0, ge=0, description="Offset for pagination"),
    anomaly_only: bool = Query(default=False, description="Filter only anomalous readings"),
    appliance: Optional[str] = Query(default=None, description="Filter by appliance name")
):
    """
    Returns electricity consumption readings with anomaly classifications
    for time-series charts and recent readings table.
    """
    df = load_dataset()
    filtered = df.copy()

    if anomaly_only:
        filtered = filtered[filtered["is_anomaly_simulated"] == 1]
    
    if appliance and appliance.strip():
        filtered = filtered[filtered["appliance"].str.lower() == appliance.strip().lower()]

    total_filtered = len(filtered)
    paged = filtered.iloc[offset : offset + limit]

    records = []
    for _, row in paged.iterrows():
        power = float(row["power_consumption"])
        baseline = float(row["baseline_consumption"])
        ratio = round(power / max(baseline, 1e-4), 2)
        records.append({
            "timestamp": str(row["timestamp"]),
            "hour": int(row["hour"]),
            "day_of_week": int(row["day_of_week"]),
            "appliance": str(row["appliance"]),
            "power_consumption": power,
            "baseline_consumption": baseline,
            "consumption_ratio": ratio,
            "is_anomaly": bool(row["is_anomaly_simulated"] == 1),
            "anomaly_reason": str(row.get("anomaly_reason", "Normal profile"))
        })

    return {
        "total": total_filtered,
        "limit": limit,
        "offset": offset,
        "readings": records
    }


@app.post("/api/predict", response_model=PredictionResponse, summary="Predict anomaly for a reading")
async def predict_endpoint(payload: PredictionRequest):
    """
    Performs unsupervised anomaly detection on an electricity consumption reading.
    Returns whether the reading is anomalous, the calibrated anomaly score, risk level,
    and a clear explanatory diagnosis.
    """
    try:
        result = predict_reading(
            hour=payload.hour,
            day_of_week=payload.day_of_week,
            appliance=payload.appliance,
            power_consumption=payload.power_consumption,
            baseline_consumption=payload.baseline_consumption
        )
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(exc)}")


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    print(f"Starting WattWise server on http://0.0.0.0:{port}...")
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
