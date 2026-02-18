from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
import joblib
import json
from pydantic import BaseModel

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = joblib.load("models/iris_model_v1.pkl")

    with open("models/metadata.json") as f:
        app.state.metadata = json.load(f)
    yield

app = FastAPI(lifespan=lifespan)

class PredictRequest(BaseModel):
    features: list[float]

@app.get("/health")
def healthcheck():
    try:
        test_features = [5.1, 3.5, 1.4, 0.2]
        X = [test_features]
        model = app.state.model
        _ = model.predict(X)
        return {
                "status": "healthy",
                "model_version": app.state.metadata["model_version"]
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }



@app.post("/predict")
def predict(request: PredictRequest):
    model = app.state.model
    meta = app.state.metadata

    pred_id = int(model.predict([request.features])[0])
    pred_label = meta["target_classes"][pred_id]

    return {
        "prediction_id": pred_id,
        "prediction_label": pred_label,
        "model_version": meta["model_version"]
    }
    