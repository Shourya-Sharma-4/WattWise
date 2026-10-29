"""
WattWise - ML-Based Electricity Consumption Anomaly Detection System
Prediction Engine: predict.py

Loads the serialized Isolation Forest model and ColumnTransformer preprocessor.
Provides high-performance single-reading and batch anomaly inference with calibrated
anomaly scoring and contextual diagnostic explanations.
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Union, List

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "wattwise_model.joblib")

_MODEL_BUNDLE = None

def get_model_bundle() -> dict:
    """
    Loads and caches the model bundle in memory.
    """
    global _MODEL_BUNDLE
    if _MODEL_BUNDLE is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Trained model not found at {MODEL_PATH}. Run 'python train.py' to generate artifacts."
            )
        _MODEL_BUNDLE = joblib.load(MODEL_PATH)
    return _MODEL_BUNDLE

def generate_explanation(
    appliance: str,
    hour: int,
    power: float,
    baseline: float,
    is_anomaly: bool,
    anomaly_score: float
) -> str:
    """
    Produces a transparent, human-interpretable rationale for why a reading
    was classified as normal or anomalous.
    """
    diff = power - baseline
    ratio = power / max(baseline, 1e-3)
    pct_change = ((power - baseline) / max(baseline, 1e-3)) * 100.0

    days_part = "early morning" if 0 <= hour <= 5 else (
        "morning" if 6 <= hour <= 11 else (
            "afternoon" if 12 <= hour <= 17 else "evening"
        )
    )

    if not is_anomaly:
        if abs(pct_change) < 15.0:
            return (
                f"Nominal operating profile. Power draw ({power:,.1f}W) matches expected baseline "
                f"({baseline:,.1f}W ±{abs(pct_change):.1f}%) for {appliance} during {days_part} (hour {hour:02d}:00)."
            )
        else:
            return (
                f"Within expected operating variance. At {power:,.1f}W vs {baseline:,.1f}W baseline, "
                f"{appliance} exhibits routine non-critical load fluctuation during {days_part}."
            )

    # Anomalous scenarios
    if ratio >= 2.0:
        pct_excess = int(ratio * 100)
        return (
            f"Excessive consumption spike detected! Current load of {power:,.1f}W is {pct_excess}% of "
            f"the standard baseline ({baseline:,.1f}W) for {appliance} at hour {hour:02d}:00 ({days_part}). "
            f"Potential cause: continuous heating element stuck on, unsealed thermal envelope, or compressor fault."
        )
    elif ratio <= 0.2 and baseline > 200:
        return (
            f"Unusual under-consumption detected! Current load of {power:,.1f}W is significantly below "
            f"the expected active baseline ({baseline:,.1f}W) for {appliance} during peak scheduled hours ({hour:02d}:00). "
            f"Potential cause: tripped breaker, component shutdown, or control circuit failure."
        )
    else:
        return (
            f"Multivariate irregularity detected. While consumption is {power:,.1f}W (baseline: {baseline:,.1f}W), "
            f"the combination of hour {hour:02d}:00, day cycle, and appliance profile deviates significantly from the "
            f"learned normative cluster in Isolation Forest feature space."
        )

def predict_reading(
    hour: int,
    day_of_week: int,
    appliance: str,
    power_consumption: float,
    baseline_consumption: float
) -> Dict[str, Any]:
    """
    Performs anomaly inference on a single electricity reading.
    
    Returns:
        dict: Classification status, anomaly score (0-1), confidence, and diagnostic explanation.
    """
    bundle = get_model_bundle()
    preprocessor = bundle["preprocessor"]
    model = bundle["model"]
    calib = bundle["score_calibration"]

    # Validate inputs
    hour = int(hour) % 24
    day_of_week = int(day_of_week) % 7
    power_consumption = float(max(0.0, power_consumption))
    baseline_consumption = float(max(1.0, baseline_consumption))
    appliance = str(appliance).strip()

    # Derived features
    eps = 1e-4
    consumption_ratio = power_consumption / (baseline_consumption + eps)
    consumption_diff = power_consumption - baseline_consumption

    input_df = pd.DataFrame([{
        "hour": hour,
        "day_of_week": day_of_week,
        "power_consumption": power_consumption,
        "baseline_consumption": baseline_consumption,
        "consumption_ratio": consumption_ratio,
        "consumption_diff": consumption_diff,
        "appliance": appliance
    }])

    # Preprocess
    X_processed = preprocessor.transform(input_df)

    # Predict: 1 = normal, -1 = anomaly
    raw_pred = model.predict(X_processed)[0]
    is_anomaly = bool(raw_pred == -1)

    # Raw decision function: higher is normal, lower/negative is anomalous
    raw_score = model.decision_function(X_processed)[0]
    inverted_score = -raw_score

    # Normalize score to [0.0, 1.0] using calibration from training set
    s_min = calib["score_min"]
    s_max = calib["score_max"]
    normalized_score = float(np.clip((inverted_score - s_min) / (s_max - s_min + 1e-6), 0.0, 1.0))
    normalized_score = round(normalized_score, 4)

    # Determine risk category
    if is_anomaly:
        risk_level = "High" if normalized_score > 0.70 else "Elevated"
        status = "Anomalous"
    else:
        risk_level = "Low" if normalized_score < 0.40 else "Moderate"
        status = "Normal"

    explanation = generate_explanation(
        appliance=appliance,
        hour=hour,
        power=power_consumption,
        baseline=baseline_consumption,
        is_anomaly=is_anomaly,
        anomaly_score=normalized_score
    )

    return {
        "status": status,
        "is_anomaly": is_anomaly,
        "anomaly_score": normalized_score,
        "anomaly_score_percentage": round(normalized_score * 100.0, 1),
        "risk_level": risk_level,
        "features": {
            "hour": hour,
            "day_of_week": day_of_week,
            "appliance": appliance,
            "power_consumption_watts": power_consumption,
            "baseline_consumption_watts": baseline_consumption,
            "consumption_ratio": round(consumption_ratio, 2),
            "consumption_diff_watts": round(consumption_diff, 2),
        },
        "explanation": explanation,
        "disclaimer": (
            "WattWise detects statistical irregularities via unsupervised Isolation Forest. "
            "Flagged readings indicate mathematical deviation from baseline profiles, not confirmed physical defects."
        )
    }

if __name__ == "__main__":
    print("Testing WattWise Prediction Engine...\n")
    
    test_cases = [
        {
            "name": "Normal Daytime HVAC",
            "hour": 14,
            "day_of_week": 2,
            "appliance": "HVAC",
            "power_consumption": 1820.0,
            "baseline_consumption": 1800.0
        },
        {
            "name": "Off-Peak Massive HVAC Spike (Anomaly)",
            "hour": 3,
            "day_of_week": 1,
            "appliance": "HVAC",
            "power_consumption": 3850.0,
            "baseline_consumption": 450.0
        },
        {
            "name": "Normal Refrigeration Cycle",
            "hour": 9,
            "day_of_week": 4,
            "appliance": "Refrigeration",
            "power_consumption": 165.0,
            "baseline_consumption": 160.0
        },
        {
            "name": "Refrigeration Seal Leak Anomaly",
            "hour": 2,
            "day_of_week": 0,
            "appliance": "Refrigeration",
            "power_consumption": 680.0,
            "baseline_consumption": 110.0
        }
    ]

    for tc in test_cases:
        res = predict_reading(
            hour=tc["hour"],
            day_of_week=tc["day_of_week"],
            appliance=tc["appliance"],
            power_consumption=tc["power_consumption"],
            baseline_consumption=tc["baseline_consumption"]
        )
        print(f"Scenario: {tc['name']}")
        print(f" -> Status: {res['status']} | Score: {res['anomaly_score']} ({res['anomaly_score_percentage']}%) | Risk: {res['risk_level']}")
        print(f" -> Rationale: {res['explanation']}")
        print("-" * 65)
