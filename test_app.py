"""
WattWise - Test Suite for FastAPI Endpoints
test_app.py
"""

import asyncio
import httpx
from app import app

async def run_tests():
    print("Testing WattWise API Endpoints...\n")

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        # 1. Health check
        res = await client.get("/health")
        assert res.status_code == 200, f"Health check failed: {res.text}"
        health_data = res.json()
        print(f"[PASS] /health -> status: {health_data['status']}, model_loaded: {health_data['model_loaded']}")

        # 2. Project Info
        res = await client.get("/api/info")
        assert res.status_code == 200, f"Info check failed: {res.text}"
        info = res.json()
        print(f"[PASS] /api/info -> project: {info['project']}")

        # 3. Summary
        res = await client.get("/api/summary")
        assert res.status_code == 200, f"Summary failed: {res.text}"
        summary = res.json()
        print(f"[PASS] /api/summary -> total: {summary['total_readings']}, anomalies: {summary['anomaly_count']} ({summary['anomaly_percentage']}%)")
        assert summary["total_readings"] > 0
        assert summary["anomaly_count"] > 0

        # 4. Readings
        res = await client.get("/api/readings?limit=5")
        assert res.status_code == 200, f"Readings failed: {res.text}"
        readings = res.json()
        print(f"[PASS] /api/readings?limit=5 -> returned {len(readings['readings'])} records (total: {readings['total']})")
        assert len(readings["readings"]) == 5

        # 5. Predict Normal
        normal_payload = {
            "hour": 14,
            "day_of_week": 2,
            "appliance": "HVAC",
            "power_consumption": 1820.0,
            "baseline_consumption": 1800.0
        }
        res = await client.post("/api/predict", json=normal_payload)
        assert res.status_code == 200, f"Predict normal failed: {res.text}"
        pred_normal = res.json()
        print(f"[PASS] /api/predict (Normal) -> status: {pred_normal['status']}, score: {pred_normal['anomaly_score']}")
        assert pred_normal["status"] == "Normal"
        assert pred_normal["is_anomaly"] is False

        # 6. Predict Anomaly
        anomaly_payload = {
            "hour": 3,
            "day_of_week": 1,
            "appliance": "HVAC",
            "power_consumption": 3850.0,
            "baseline_consumption": 450.0
        }
        res = await client.post("/api/predict", json=anomaly_payload)
        assert res.status_code == 200, f"Predict anomaly failed: {res.text}"
        pred_anomaly = res.json()
        print(f"[PASS] /api/predict (Anomaly) -> status: {pred_anomaly['status']}, score: {pred_anomaly['anomaly_score']}")
        assert pred_anomaly["status"] == "Anomalous"
        assert pred_anomaly["is_anomaly"] is True

        print("\nALL BACKEND API TESTS PASSED SUCCESSFULLY! :)")

if __name__ == "__main__":
    asyncio.run(run_tests())
