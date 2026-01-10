from pathlib import Path
import numpy as np
import joblib

# -----------------------
# Paths
# -----------------------
ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"
POSTERIOR_PATH = ARTIFACTS_DIR / "posterior_means.pkl"
DEPLOYABLE_MODEL_PATH = ARTIFACTS_DIR / "deployable_model.pkl"

# -----------------------
# Feature columns used in training
# -----------------------
def clean_columns(columns):
    cleaned = []
    for col in columns:
        col = col.strip().lower()                        
        col = col.replace(" ", "_")                      
        cleaned.append(col)
    return cleaned

FEATURE_COLS = [
    "land_use_change",
    "animal_feed",
    "farm",
    "processing",
    "transport",
    "packaging",
    "retail"
]
# -----------------------
# Model wrapper
# -----------------------
class BayesianLogisticWrapper:
    def __init__(self, coef_, intercept_, feature_names):
        self.coef_ = coef_.reshape(1, -1)
        self.intercept_ = np.array([intercept_])
        self.feature_names = feature_names

    def predict(self, X):
        X_subset = X[self.feature_names]
        z = X_subset.values @ self.coef_.T + self.intercept_
        return (1 / (1 + np.exp(-z)) > 0.5).astype(int).ravel()

    def predict_proba(self, X):
        X_subset = X[self.feature_names]
        z = X_subset.values @ self.coef_.T + self.intercept_
        p = 1 / (1 + np.exp(-z))
        return np.column_stack([1 - p, p])

# -----------------------
# Build and save deployable model
# -----------------------
def train_and_save_model(df=None):
    coef_means, bias_mean = joblib.load(POSTERIOR_PATH)
    model = BayesianLogisticWrapper(coef_means, bias_mean, FEATURE_COLS)
    joblib.dump(model, DEPLOYABLE_MODEL_PATH)
    print(f"Deployable model saved to {DEPLOYABLE_MODEL_PATH}")
    return model

# -----------------------
# CLI entry
# -----------------------
if __name__ == "__main__":
    train_and_save_model()

from pathlib import Path
import numpy as np
import joblib

# -----------------------
# Paths
# -----------------------
ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"
POSTERIOR_PATH = ARTIFACTS_DIR / "posterior_means.pkl"
DEPLOYABLE_MODEL_PATH = ARTIFACTS_DIR / "deployable_model.pkl"

# -----------------------
# Feature columns used in training
# -----------------------
def clean_columns(columns):
    cleaned = []
    for col in columns:
        col = col.strip().lower()                        
        col = col.replace(" ", "_")                      
        cleaned.append(col)
    return cleaned

FEATURE_COLS = [
    "land_use_change",
    "animal_feed",
    "farm",
    "processing",
    "transport",
    "packaging",
    "retail"
]

# -----------------------
# Model wrapper
# -----------------------
class BayesianLogisticWrapper:
    def __init__(self, coef_, intercept_, feature_names):
        self.coef_ = coef_.reshape(1, -1)
        self.intercept_ = np.array([intercept_])
        self.feature_names = feature_names

    def predict(self, X):
        X_subset = X[self.feature_names]
        z = X_subset.values @ self.coef_.T + self.intercept_
        return (1 / (1 + np.exp(-z)) > 0.5).astype(int).ravel()

    def predict_proba(self, X):
        X_subset = X[self.feature_names]
        z = X_subset.values @ self.coef_.T + self.intercept_
        p = 1 / (1 + np.exp(-z))
        return np.column_stack([1 - p, p])

# -----------------------
# Build and save deployable model
# -----------------------
def train_and_save_model(df=None):
    coef_means, bias_mean = joblib.load(POSTERIOR_PATH)
    model = BayesianLogisticWrapper(coef_means, bias_mean, FEATURE_COLS)
    joblib.dump(model, DEPLOYABLE_MODEL_PATH)
    print(f"Deployable model saved to {DEPLOYABLE_MODEL_PATH}")
    return model

# -----------------------
# CLI entry
# -----------------------
if __name__ == "__main__":
    train_and_save_model()
