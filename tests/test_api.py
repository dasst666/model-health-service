from fastapi.testclient import TestClient
from src.app import app

client = TestClient(app)

class FakeModel:
    def predict(self, X):
        return [0]

def test_health():
    app.state.model = FakeModel()
    app.state.metadata = {"model_version": "1.0.0"}

    response = client.get("/health")
    data = response.json()

    assert response.status_code == 200
    assert data["status"] == "healthy"
    assert "model_version" in data

def test_predict():
    app.state.model = FakeModel()
    app.state.metadata = {
        "target_classes": ["setosa", "versicolor", "virginica"],
        "model_version": "1.0.0"
    }
    
    response = client.post("/predict", json={"features": [5.1, 3.5, 1.4, 0.2]})
    data = response.json()

    assert response.status_code == 200
    assert "prediction_id" in data