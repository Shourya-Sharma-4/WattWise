"""
WattWise - ML-Based Electricity Consumption Anomaly Detection System
Dataset Generator: generate_data.py

Generates a realistic, reproducible synthetic dataset of electricity consumption readings
with realistic household appliance profiles, seasonal/hourly baselines, and deliberately
injected anomaly scenarios for training and evaluating unsupervised anomaly detection.
"""

import os
import random
import numpy as np
import pandas as pd

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

APPLIANCES = {
    "HVAC": {
        "base_watt_day": 1800.0,
        "base_watt_night": 450.0,
        "noise_std": 120.0,
        "peak_hours": [12, 13, 14, 15, 16, 17, 18, 19],
    },
    "Refrigeration": {
        "base_watt_day": 160.0,
        "base_watt_night": 110.0,
        "noise_std": 15.0,
        "peak_hours": [8, 12, 19, 20],  # meal prep opening doors
    },
    "Water Heater": {
        "base_watt_day": 2800.0,
        "base_watt_night": 80.0,
        "noise_std": 140.0,
        "peak_hours": [6, 7, 8, 19, 20, 21],  # showers and evening dishes
    },
    "Lighting": {
        "base_watt_day": 60.0,
        "base_watt_night": 20.0,
        "noise_std": 10.0,
        "peak_hours": [18, 19, 20, 21, 22, 23],  # evening lighting
    },
    "EV Charger": {
        "base_watt_day": 40.0,
        "base_watt_night": 3600.0,  # overnight charging
        "noise_std": 90.0,
        "peak_hours": [0, 1, 2, 3, 4, 5],
    },
    "Laundry / Kitchen": {
        "base_watt_day": 1200.0,
        "base_watt_night": 15.0,  # idle standby
        "noise_std": 80.0,
        "peak_hours": [10, 11, 17, 18, 19],
    },
}

ANOMALY_SCENARIOS = [
    {
        "type": "HVAC Overload Off-Peak",
        "appliance": "HVAC",
        "description": "HVAC running continuously at full load during early morning hours (possible thermostat stuck or window left open).",
        "hour_range": [1, 2, 3, 4],
        "fixed_power": 3800.0,
    },
    {
        "type": "Refrigeration Seal Leak",
        "appliance": "Refrigeration",
        "description": "Refrigerator compressor running without cycling off (door gasket leak or door left ajar).",
        "hour_range": [0, 1, 2, 3, 4, 5, 22, 23],
        "fixed_power": 650.0,
    },
    {
        "type": "Water Heater Stuck Element",
        "appliance": "Water Heater",
        "description": "Continuous high-draw water heater heating element stuck active during middle of the night.",
        "hour_range": [1, 2, 3, 4],
        "fixed_power": 4200.0,
    },
    {
        "type": "Daytime Lighting Waste",
        "appliance": "Lighting",
        "description": "High-intensity outdoor/commercial floodlights left fully on under bright midday sunlight.",
        "hour_range": [11, 12, 13, 14, 15],
        "fixed_power": 1450.0,
    },
    {
        "type": "EV Charging Midday Spike",
        "appliance": "EV Charger",
        "description": "High-draw fast charging during peak electricity tariff hours with erratic power fluctuation.",
        "hour_range": [13, 14, 15, 16],
        "fixed_power": 7200.0,
    },
    {
        "type": "Kitchen Stove Left On",
        "appliance": "Laundry / Kitchen",
        "description": "Electric range / oven element left active overnight with no kitchen activity.",
        "hour_range": [2, 3, 4, 5],
        "fixed_power": 2600.0,
    },
    {
        "type": "Appliance Degradation Spike",
        "appliance": "HVAC",
        "description": "Compressor motor starting failure causing massive electrical surges.",
        "hour_range": [8, 9, 10, 11, 12],
        "fixed_power": 5200.0,
    },
]


def generate_readings(num_samples: int = 2500, anomaly_rate: float = 0.08) -> pd.DataFrame:
    """
    Generates a DataFrame of electricity consumption readings.
    Includes simulated baseline, actual power consumption, and realistic anomalies.
    """
    appliance_names = list(APPLIANCES.keys())
    
    # Generate sequential hourly timestamps starting 60 days ago
    start_time = pd.Timestamp("2026-08-01 00:00:00")
    timestamps = [start_time + pd.Timedelta(hours=i) for i in range(num_samples)]
    
    rows = []
    num_anomalies_target = int(num_samples * anomaly_rate)
    anomaly_indices = set(random.sample(range(num_samples), num_anomalies_target))
    
    for i in range(num_samples):
        ts = timestamps[i]
        hour = ts.hour
        day_of_week = ts.day_of_week  # 0=Monday, 6=Sunday
        is_weekend = 1 if day_of_week >= 5 else 0
        
        # Select appliance for this timestamp
        appliance = appliance_names[i % len(appliance_names)]
        cfg = APPLIANCES[appliance]
        
        # Determine baseline for this appliance and hour
        is_peak = hour in cfg["peak_hours"]
        if is_peak:
            baseline = cfg["base_watt_day"] * (1.15 if is_weekend else 1.0)
        else:
            baseline = cfg["base_watt_night"] * (1.10 if is_weekend else 1.0)
            
        # Add normal variation (Gaussian noise)
        normal_power = baseline + np.random.normal(0, cfg["noise_std"])
        normal_power = max(5.0, round(float(normal_power), 2))
        
        # Check if this index was chosen as an anomaly
        if i in anomaly_indices:
            # Pick a suitable anomaly scenario
            scenario_candidates = [s for s in ANOMALY_SCENARIOS if s["appliance"] == appliance]
            if scenario_candidates:
                scenario = random.choice(scenario_candidates)
            else:
                scenario = random.choice(ANOMALY_SCENARIOS)
                appliance = scenario["appliance"]
            
            # Anomaly power
            power = scenario["fixed_power"] + np.random.normal(0, 60.0)
            power = max(baseline * 2.2, round(float(power), 2))
            is_anomaly = 1
            reason = scenario["description"]
        else:
            power = normal_power
            is_anomaly = 0
            reason = "Normal operating profile"
            
        baseline_clean = round(float(baseline), 2)
        
        rows.append({
            "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
            "hour": int(hour),
            "day_of_week": int(day_of_week),
            "is_weekend": int(is_weekend),
            "appliance": appliance,
            "power_consumption": power,
            "baseline_consumption": baseline_clean,
            "is_anomaly_simulated": is_anomaly,
            "anomaly_reason": reason,
        })
        
    df = pd.DataFrame(rows)
    return df


def main():
    os.makedirs("data", exist_ok=True)
    output_path = os.path.join("data", "electricity_readings.csv")
    
    print("Generating simulated electricity consumption dataset...", flush=True)
    df = generate_readings(num_samples=2500, anomaly_rate=0.08)
    
    df.to_csv(output_path, index=False)
    print(f"Dataset successfully saved to: {output_path}", flush=True)
    print(f"Total readings: {len(df)}", flush=True)
    print(f"Simulated anomalies: {df['is_anomaly_simulated'].sum()} ({df['is_anomaly_simulated'].mean() * 100:.2f}%)", flush=True)
    print(f"Appliances included: {list(df['appliance'].unique())}", flush=True)
    print("\nSample records:", flush=True)
    print(df.head(), flush=True)


if __name__ == "__main__":
    main()
