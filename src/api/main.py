from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# Load your saved best model and preprocessing objects
model = joblib.load("best_model.pkl")

class PredictRequest(BaseModel):
    features: list[float]  # assuming input is a list of features

@app.post("/predict")
def predict(data: PredictRequest):
    X = np.array(data.features).reshape(1, -1)
    # Make prediction (adjust preprocessing here if needed)
    pred = model.predict(X)
    proba = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else None

    return {
        "prediction": int(pred[0]),
        "probability": float(proba[0]) if proba is not None else None
    }
