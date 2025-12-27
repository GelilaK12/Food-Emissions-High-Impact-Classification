import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pymc as pm
import arviz as az

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
data_path = os.path.join(BASE_DIR, "data", "Food_Production.csv")

# ================= Folder Setup =================
OUTPUT_FOLDER = os.path.join(BASE_DIR, "outputs", "bayesian_logistic")
IMAGE_FOLDER = os.path.join(BASE_DIR, "images", "bayesian_logistic")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(IMAGE_FOLDER, exist_ok=True)

# ================= Data Loading =================
data = pd.read_csv(data_path)

print("\n=== Dataset shape ===")
print(data.shape)
data.head().to_csv(f"{OUTPUT_FOLDER}/data_sample.csv", index=False)
data.isnull().sum().to_csv(f"{OUTPUT_FOLDER}/missing_values_per_column.csv")

# ============================================
# Data Cleaning
# ============================================
data["Food product"] = data["Food product"].str.strip().str.title()
object_cols = data.select_dtypes(include="object").columns
for col in object_cols:
    try:
        data[col] = pd.to_numeric(data[col].str.replace(",", ""))
    except Exception:
        pass
data.describe().to_csv(f"{OUTPUT_FOLDER}/data_describe_summary.csv")
data.to_csv(f"{OUTPUT_FOLDER}/food_production_cleaned.csv", index=False)

# ============================================
# Target Definition
# ============================================
threshold = data["Total_emissions"].quantile(0.75)
data["High_Impact"] = (data["Total_emissions"] >= threshold).astype(int)
features = ["Land use change", "Animal Feed", "Farm", "Processing", "Transport", "Packaging", "Retail"]
X = data[features]
y = data["High_Impact"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# ============================================
# Feature Scaling
# ============================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================
# Windows-safe Bayesian Logistic Regression
# ============================================
if __name__ == "__main__":
    with pm.Model() as bayes_logistic:
        w = pm.Normal("w", mu=0, sigma=1, shape=X_train_scaled.shape[1])
        b = pm.Normal("b", mu=0, sigma=1)
        logits = pm.math.dot(X_train_scaled, w) + b
        p = pm.Deterministic("p", pm.math.sigmoid(logits))
        y_obs = pm.Bernoulli("y_obs", p=p, observed=y_train)
        trace = pm.sample(2000, tune=1000, cores=1, chains=1, random_seed=42)

    # ============================================
    # Posterior Summaries
    # ============================================
    summary_df = az.summary(trace, var_names=["w"])
    feature_names = ["Land use change", "Animal Feed", "Farm", "Processing", "Transport", "Packaging", "Retail"]
    summary_df.index = feature_names
    summary_df.to_csv(f"{OUTPUT_FOLDER}/bayesian_logistic_summary.csv")
    
    print(summary_df)

    fig = az.plot_forest(trace, var_names=["w"], combined=True)
    plt.title("Bayesian Logistic Regression Weights with Uncertainty")
    plt.tight_layout()
    plt.savefig(f"{IMAGE_FOLDER}/bayesian_logistic_forest_plot.png")
    plt.close()

    # ============================================
    # Posterior Prediction on Test Set
    # ============================================
    w_samples = trace.posterior["w"].stack(sample=("chain", "draw")).values  
    b_samples = trace.posterior["b"].stack(sample=("chain", "draw")).values  

    logits_test = np.dot(X_test_scaled, w_samples) + b_samples  
    probs_test = 1 / (1 + np.exp(-logits_test))  
    pred_means = probs_test.mean(axis=1) 
    y_pred = (pred_means >= 0.5).astype(int)  

    pred_df = pd.DataFrame({
    "Food Product": X_test.index,  
    "Predicted_Probability": pred_means,
    "Predicted_Class": y_pred})
    pred_df.to_csv(f"{OUTPUT_FOLDER}/bayesian_logistic_predictions.csv", index=False)

    # ============================================
    # Evaluation
    # ============================================
    report = classification_report(y_test, y_pred, output_dict=True)
    pd.DataFrame(report).transpose().to_csv(f"{OUTPUT_FOLDER}/bayesian_logistic_classification_report.csv")
    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Bayesian Logistic Regression Confusion Matrix")
    print(cm)
    plt.savefig(f"{IMAGE_FOLDER}/bayesian_logistic_confusion_matrix.png")
    plt.close()

# ============================================
# Insight
# ============================================

# - Land use change is the most important feature, consistent with classical logistic regression.  
# - Overall classification is similar to logistic regression, with comparable accuracy and F1 score.  
# - Bayesian approach provides uncertainty estimates for each weight, highlighting which features are more or less reliable.