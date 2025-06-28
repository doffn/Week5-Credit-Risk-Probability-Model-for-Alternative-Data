from fastapi import FastAPI
from .pydantic_models import CustomerData

app = FastAPI()

@app.get("/")
def home():
    return {"status": "Credit Risk API running"}

@app.post("/predict")
def predict_risk(data: CustomerData):
    # Dummy return for now
    return {"risk_score": 0.5}
