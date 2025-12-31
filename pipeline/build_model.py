from pathlib import Path
import joblib
import numpy as np
from pipeline.validation import validate
import pandas as pd

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "scripts/artifacts"
POSTERIOR_PATH = ARTIFACTS_DIR / "posterior_means.pkl"
DEPLOYABLE_MODEL_PATH = ARTIFACTS_DIR / "deployable_model.pkl"
RAW_DATA_PATH = Path(__file__).resolve().parent.parent / "data/Food_Production.csv"

# -------------------------------
# Wrapper class
# -------------------------------
class BayesianLogisticWrapper:
    def __init__(self, coef, bias):
        self.coef_ = coef.reshape(1, -1)
        self.intercept_ = np.array([bias])

    def predict(self, X):
        z = X @ self.coef_.T + self.intercept_
        return (1 / (1 + np.exp(-z)) > 0.5).astype(int).ravel()

# -------------------------------
# Pipeline functions
# -------------------------------
def load_and_validate_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = validate(df)  # run your Pandera or Great Expectations checks
    print(f"Data loaded and validated: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

def load_posterior_and_wrap_model(posterior_path: Path) -> BayesianLogisticWrapper:
    coef_means, bias_mean = joblib.load(posterior_path)
    model = BayesianLogisticWrapper(coef_means, bias_mean)
    print("Deployable Bayesian wrapper created")
    return model

def save_model(model: BayesianLogisticWrapper, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    print(f"Deployable model saved to {path}")

# -------------------------------
# Main pipeline flow
# -------------------------------
if __name__ == "__main__":
    df = load_and_validate_data(RAW_DATA_PATH)

    model = load_posterior_and_wrap_model(POSTERIOR_PATH)

    save_model(model, DEPLOYABLE_MODEL_PATH)
