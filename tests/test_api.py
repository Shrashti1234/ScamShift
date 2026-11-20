"""
Automated Tests for FastAPI Endpoints
====================================
Phase-3: Automated Testing with pytest and FastAPI TestClient.
"""

from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "ScamShift"
    assert data["status"] == "online"


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_predict_scam_message():
    payload = {"text": "Aapka SBI account freeze ho gaya hai. KYC update karne ke liye OTP abhi share karein"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "score" in data
    assert "flagged" in data
    assert "prediction" in data


def test_predict_genuine_message():
    payload = {"text": "Hi Mom, I reached the hostel safely. Will call you after dinner."}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "score" in data
    assert data["score"] < 0.5
    assert data["prediction"] == "genuine"
    assert data["flagged"] is False


def test_empty_text_validation():
    response = client.post("/predict", json={"text": "   "})
    assert response.status_code == 422
