"""
WattWise - ML-Based Electricity Consumption Anomaly Detection System
Model Training: train.py

Loads electricity consumption readings, engineers temporal and consumption features,
trains an unsupervised Isolation Forest model, evaluates performance against
simulated ground truth labels, and exports the serialized model and preprocessor artifacts.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes derived statistical features for anomaly detection.
    """
    df = df.copy()
    # Avoid division by zero
    eps = 1e-4
    df["consumption_ratio"] = df["power_consumption"] / (df["baseline_consumption"] + eps)
    df["consumption_diff"] = df["power_consumption"] - df["baseline_consumption"]
    return df

def train_and_evaluate(
    data_path: str = "data/electricity_readings.csv",
    model_dir: str = "models",
    contamination: float = 0.08
):
    """
    Trains the Isolation Forest model and evaluates against synthetic labels.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at {data_path}. Run generate_data.py first.")

    os.makedirs(model_dir, exist_ok=True)

    print("=" * 70, flush=True)
    print("  WATTWISE - ISOLATION FOREST ANOMALY DETECTION TRAINING", flush=True)
    print("=" * 70, flush=True)

    # 1. Load Dataset
    print(f"\n[1/5] Loading dataset from: {data_path}...", flush=True)
    df = pd.read_csv(data_path)
    print(f"Total readings loaded: {len(df):,}", flush=True)

    # 2. Feature Engineering
    print("\n[2/5] Engineering domain-specific features...", flush=True)
    df = engineer_features(df)
    
    numeric_features = [
        "hour",
        "day_of_week",
        "power_consumption",
        "baseline_consumption",
        "consumption_ratio",
        "consumption_diff"
    ]
    categorical_features = ["appliance"]

    X = df[numeric_features + categorical_features]
    y_ground_truth = df["is_anomaly_simulated"].values

    actual_contamination = float(np.mean(y_ground_truth))
    print(f"Features: {numeric_features + categorical_features}", flush=True)
    print(f"Ground-truth simulated anomaly rate: {actual_contamination * 100:.2f}% ({np.sum(y_ground_truth)} anomalies)", flush=True)

    # 3. Build Preprocessing & Isolation Forest Pipeline
    print("\n[3/5] Constructing ColumnTransformer and Isolation Forest Pipeline...", flush=True)
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features)
        ]
    )

    # Fit preprocessor
    X_processed = preprocessor.fit_transform(X)

    # Train Isolation Forest
    # n_estimators=150, contamination tuned to the expected anomaly rate
    iso_forest = IsolationForest(
        n_estimators=150,
        contamination=contamination,
        max_samples="auto",
        random_state=42,
        n_jobs=-1
    )
    iso_forest.fit(X_processed)
    print("Isolation Forest training complete.", flush=True)

    # 4. Evaluation against synthetic ground truth
    print("\n[4/5] Evaluating unsupervised detection against synthetic labels...", flush=True)
    
    # IsolationForest: 1 = normal, -1 = anomaly
    raw_preds = iso_forest.predict(X_processed)
    y_pred = (raw_preds == -1).astype(int)

    # Decision function: lower/negative = more anomalous
    decision_scores = iso_forest.decision_function(X_processed)
    # Convert to anomaly score where higher = more anomalous
    # decision_function typically ranges from -0.5 to 0.5; offset so higher is anomalous
    anomaly_scores = -decision_scores

    # Calibrate min/max for normalization to [0, 1]
    score_min = float(np.min(anomaly_scores))
    score_max = float(np.max(anomaly_scores))
    norm_scores = (anomaly_scores - score_min) / (score_max - score_min + 1e-6)

    # Classification Metrics
    prec = precision_score(y_ground_truth, y_pred, zero_division=0)
    rec = recall_score(y_ground_truth, y_pred, zero_division=0)
    f1 = f1_score(y_ground_truth, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_ground_truth, norm_scores)
    cm = confusion_matrix(y_ground_truth, y_pred)
    tn, fp, fn, tp = cm.ravel()

    print("\n" + "-" * 50, flush=True)
    print("             MODEL EVALUATION METRICS", flush=True)
    print("-" * 50, flush=True)
    print(f"  Precision:   {prec:.4f}  (True Positive / Total Flagged)", flush=True)
    print(f"  Recall:      {rec:.4f}  (True Positive / Total Simulated)", flush=True)
    print(f"  F1-Score:    {f1:.4f}  (Harmonic mean of precision & recall)", flush=True)
    print(f"  ROC-AUC:     {roc_auc:.4f}  (Separation capability)", flush=True)
    print("-" * 50, flush=True)
    print(f"  Confusion Matrix:", flush=True)
    print(f"    True Negatives  (Normal detected as Normal):   {tn:5d}", flush=True)
    print(f"    False Positives (Normal flagged as Anomaly):   {fp:5d}", flush=True)
    print(f"    False Negatives (Anomaly missed as Normal):    {fn:5d}", flush=True)
    print(f"    True Positives  (Anomaly correctly detected):  {tp:5d}", flush=True)
    print("-" * 50, flush=True)

    print("\nDetailed Classification Report:")
    print(classification_report(y_ground_truth, y_pred, target_names=["Normal", "Anomaly"]), flush=True)

    # 5. Save Artifacts
    print("\n[5/5] Exporting model artifacts and metadata...", flush=True)
    
    # Save composite bundle for fast loading
    model_bundle = {
        "preprocessor": preprocessor,
        "model": iso_forest,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "score_calibration": {
            "score_min": score_min,
            "score_max": score_max,
            "decision_threshold": float(iso_forest.offset_),
        },
        "appliances": sorted(list(df["appliance"].unique())),
    }

    bundle_path = os.path.join(model_dir, "wattwise_model.joblib")
    joblib.dump(model_bundle, bundle_path)
    print(f"Model bundle saved to: {bundle_path}", flush=True)

    # Save summary metadata JSON for transparency and backend consumption
    metadata = {
        "model_name": "Isolation Forest Anomaly Detector",
        "algorithm": "IsolationForest(n_estimators=150, contamination=0.08)",
        "training_samples": len(df),
        "features": numeric_features + categorical_features,
        "appliances": sorted(list(df["appliance"].unique())),
        "contamination": contamination,
        "actual_anomaly_rate": round(actual_contamination, 4),
        "metrics": {
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1_score": round(float(f1), 4),
            "roc_auc": round(float(roc_auc), 4),
            "true_positives": int(tp),
            "false_positives": int(fp),
            "true_negatives": int(tn),
            "false_negatives": int(fn),
        },
        "score_calibration": {
            "score_min": round(score_min, 4),
            "score_max": round(score_max, 4),
            "offset": round(float(iso_forest.offset_), 4),
        },
        "trained_timestamp": pd.Timestamp.now().isoformat(),
        "disclaimer": "Unsupervised model detects statistical irregularities in power consumption. Not an engineering fault certificate."
    }

    metadata_path = os.path.join(model_dir, "model_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Model metadata saved to: {metadata_path}", flush=True)

    print("\n" + "=" * 70, flush=True)
    print("  TRAINING PIPELINE FINISHED SUCCESSFULLY!", flush=True)
    print("=" * 70, flush=True)

    return metadata

if __name__ == "__main__":
    train_and_evaluate()
