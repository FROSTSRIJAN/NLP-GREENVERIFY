"""
GreenVerify Climate Risk Prediction API

FastAPI backend that serves a trained XGBoost model to predict
climate risk levels (Low / Medium / High) based on energy and
sustainability indicators.
"""

import os
from pathlib import Path
from contextlib import asynccontextmanager

import numpy as np
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Pydantic request model – validates incoming JSON payloads
# ---------------------------------------------------------------------------

class ClimateInput(BaseModel):
    gdp: float
    population: float
    coal_consumption: float
    gas_consumption: float
    oil_consumption: float
    renewables_consumption: float
    solar_consumption: float
    wind_consumption: float
    hydro_consumption: float

# ---------------------------------------------------------------------------
# Global references populated once at startup via the lifespan handler
# ---------------------------------------------------------------------------

model = None
scaler = None

# Mapping from numeric prediction to human-readable label
RISK_LABELS = {
    0: "Low Risk",
    1: "Medium Risk",
    2: "High Risk",
}

# Feature order must match the order used during training
FEATURE_ORDER = [
    "gdp",
    "population",
    "coal_consumption",
    "gas_consumption",
    "oil_consumption",
    "renewables_consumption",
    "solar_consumption",
    "wind_consumption",
    "hydro_consumption",
]

# ---------------------------------------------------------------------------
# Resolve model directory relative to this file so it works both locally
# and on Render regardless of the working directory.
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

# ---------------------------------------------------------------------------
# Lifespan: load model + scaler once when the application starts
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, scaler

    model_path = MODELS_DIR / "green_esg_model.pkl"
    scaler_path = MODELS_DIR / "scaler.pkl"

    if not model_path.exists():
        raise RuntimeError(f"Model file not found: {model_path}")
    if not scaler_path.exists():
        raise RuntimeError(f"Scaler file not found: {scaler_path}")

    # Load artifacts once – they stay in memory for the lifetime of the process
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    print(f"✔ Model loaded from {model_path}")
    print(f"✔ Scaler loaded from {scaler_path}")

    yield  # application is running

    # Cleanup (if needed) goes here
    print("Shutting down GreenVerify API")

# ---------------------------------------------------------------------------
# FastAPI application instance
# ---------------------------------------------------------------------------

app = FastAPI(
    title="GreenVerify Climate Risk API",
    description="Predicts climate risk levels using a trained XGBoost model.",
    version="1.0.0",
    lifespan=lifespan,
)

# Allow cross-origin requests so frontend dashboards can call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],     # restrict in production as needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
def health_check():
    """Health check – confirms the API is running."""
    return {"message": "GreenVerify Climate Risk API running"}


@app.post("/predict")
def predict(data: ClimateInput):
    """
    Accept energy/sustainability indicators and return the predicted
    climate risk level.
    """
    # 1. Extract features in the exact order used during training
    features = [getattr(data, f) for f in FEATURE_ORDER]

    # 2. Convert to NumPy array and reshape to a single-sample row
    input_array = np.array(features).reshape(1, -1)

    # 3. Apply the same scaling that was used during training
    scaled_input = scaler.transform(input_array)

    # 4. Run the model prediction
    prediction = model.predict(scaled_input)

    # 5. Map the numeric label to a readable string
    risk_code = int(prediction[0])
    risk_level = RISK_LABELS.get(risk_code, "Unknown Risk")

    return {"risk_level": risk_level}
