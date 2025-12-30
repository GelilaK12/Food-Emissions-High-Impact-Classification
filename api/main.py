import joblib
from pathlib import Path
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np

# -------------------------
# Load numeric arrays (no class needed)
# -------------------------
posterior_path = Path(__file__).resolve().parent.parent / "scripts/artifacts/posterior_means.pkl"
coef_means, bias_mean = joblib.load(posterior_path)

# -------------------------
# Define wrapper class locally
# -------------------------
class BayesianLogisticWrapper:
    def __init__(self, coef, bias):
        self.coef_ = coef.reshape(1, -1)
        self.intercept_ = np.array([bias])

    def predict(self, X):
        # Convert DataFrame to NumPy array to avoid attribute errors
        X_np = X.to_numpy()
        z = X_np @ self.coef_.T + self.intercept_
        return (1 / (1 + np.exp(-z)) > 0.5).astype(int).ravel()



model = BayesianLogisticWrapper(coef_means, bias_mean)

# -------------------------
# FastAPI app
# -------------------------
app = FastAPI(title="Food Emissions Predictor API")

class FoodFeatures(BaseModel):
    Land_use_change: float
    Animal_Feed: float
    Farm: float
    Processing: float
    Transport: float
    Packaging: float
    Retail: float

@app.get("/")
def root():
    return {"message": "API is running!"}

@app.post("/predict")
def predict(data: FoodFeatures):
    df = pd.DataFrame([data.dict()])
    prediction = model.predict(df)[0]
    return {"high_impact": int(prediction)}
