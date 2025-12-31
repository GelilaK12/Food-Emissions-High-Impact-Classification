import pandas as pd
import sys
from pathlib import Path
from prefect import flow, task
from prefect import task
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
from pathlib import Path
from scripts.regular.export_deployable_model import train_and_save_model
from pipeline.validation import validate

# -----------------------
# Paths
# -----------------------

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
DATA_PATH = ROOT / "data" / "Food_Production.csv"
OUTPUT_DIR = ROOT / "outputs" / "bayesian_logistic"

# -----------------------
# Prefect Tasks
# -----------------------
@task
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    print(f"Loaded {len(df)} rows from {DATA_PATH}")
    print(f"Columns: {df.columns.tolist()}")

   
    threshold = df['total_emissions'].quantile(0.75)  
    df['high_impact'] = (df['total_emissions'] > threshold).astype(int)

    return df

@task
def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    validated_df = validate(df)
    print("Data validation passed")
    return validated_df

@task
def build_and_save_model(df: pd.DataFrame):
    model = train_and_save_model(df)
    print("Deployable Bayesian model built and saved")
    return model

@task
def save_predictions(df: pd.DataFrame, model):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if hasattr(model, "predict_proba"):
        df["predicted_prob"] = model.predict_proba(df)[:, 1]
    else:
        df["prediction"] = model.predict(df)

    out_path = OUTPUT_DIR / "data_sample_with_predictions.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved predictions to {out_path}")

@task
def evaluate_metrics(df: pd.DataFrame, model):
    y_true = df["high_impact"]
    y_pred_prob = model.predict_proba(df)[:, 1]
    y_pred = model.predict(df)

    metrics = {
        "ROC-AUC": roc_auc_score(y_true, y_pred_prob),
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
    }

    print("Model evaluation metrics:", metrics)

    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv("outputs/bayesian_logistic/metrics.csv", index=False)

    return metrics


# -----------------------
# Prefect Flow
# -----------------------
@flow(name="Food Emissions Bayesian ETL Pipeline")
def etl_pipeline():
    df = load_data()
    df_valid = validate_data(df)
    model = build_and_save_model(df_valid)
    save_predictions(df_valid, model)
    metrics = evaluate_metrics(df_valid, model)
    print("ETL pipeline completed successfully!")

# -----------------------
# Entry Point
# -----------------------
if __name__ == "__main__":
    etl_pipeline()
    