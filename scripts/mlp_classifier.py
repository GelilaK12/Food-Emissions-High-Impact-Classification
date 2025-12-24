import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.inspection import permutation_importance
from sklearn.calibration import calibration_curve

# ================= Folder Setup =================
OUTPUT_FOLDER = "outputs/mlp_classifier"
IMAGE_FOLDER = "images/mlp_classifier"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(IMAGE_FOLDER, exist_ok=True)

# ============================================
# Data Loading
# ============================================
data_path = os.path.join("data", "Food_Production.csv")
data = pd.read_csv(data_path)

print("\n=== Dataset shape (rows, columns) ===")
print(data.shape)

data.head().to_csv(f"{OUTPUT_FOLDER}/data_sample.csv", index=False)
print("\n=== Dataset info ===")
print(data.info())

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
# Feature Scaling (Required for MLP)
# ============================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================
# Baseline MLP Model
# ============================================
mlp = MLPClassifier(hidden_layer_sizes=(32, 16), activation="relu", solver="adam", max_iter=1000, random_state=42)
mlp.fit(X_train_scaled, y_train)

# ============================================
# Evaluation
# ============================================
y_pred = mlp.predict(X_test_scaled)

report = classification_report(y_test, y_pred, output_dict=True)
pd.DataFrame(report).transpose().to_csv(f"{OUTPUT_FOLDER}/mlp_classification_report.csv")

print("\n=== MLP Classification Report ===")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
plt.figure()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("MLP Confusion Matrix")
plt.savefig(f"{IMAGE_FOLDER}/mlp_confusion_matrix.png")
plt.close()

# ============================================
# Training Loss Curve
# ============================================
plt.figure()
plt.plot(mlp.loss_curve_)
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.title("MLP Training Loss Curve")
plt.savefig(f"{IMAGE_FOLDER}/mlp_loss_curve.png")
plt.close()

# ============================================
# Regularized MLP
# ============================================
mlp_reg = MLPClassifier(hidden_layer_sizes=(32, 16), activation="relu", solver="adam", alpha=0.01, max_iter=1000, random_state=42)
mlp_reg.fit(X_train_scaled, y_train)

# ============================================
# Hyperparameter Tuning
# ============================================
param_grid = {
    "hidden_layer_sizes": [(16, 16), (32, 16), (32, 32)],
    "activation": ["relu", "tanh"],
    "alpha": [0.0001, 0.001, 0.01],
}

grid = GridSearchCV(MLPClassifier(max_iter=1000, random_state=42), param_grid, cv=5, scoring="f1", n_jobs=-1)
grid.fit(X_train_scaled, y_train)

best_mlp = grid.best_estimator_
print("Best hyperparameters:", grid.best_params_)

y_pred_best = best_mlp.predict(X_test_scaled)
print(classification_report(y_test, y_pred_best))

# ============================================
# Calibration Curve
# ============================================
y_prob = best_mlp.predict_proba(X_test_scaled)[:, 1]
prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10)

plt.figure(figsize=(6, 6))
plt.plot(prob_pred, prob_true, marker="o", label="MLP")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect Calibration")
plt.xlabel("Predicted Probability")
plt.ylabel("True Probability")
plt.title("MLP Calibration Curve")
plt.legend()
plt.savefig(f"{IMAGE_FOLDER}/mlp_calibration_curve.png")
plt.close()

# ============================================
# Permutation Importance (Interpretability)
# ============================================
perm = permutation_importance(best_mlp, X_test_scaled, y_test, n_repeats=10, random_state=42)

importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": perm.importances_mean,
    "Std": perm.importances_std,
}).sort_values(by="Importance", ascending=True)

importance_df.to_csv(f"{OUTPUT_FOLDER}/mlp_permutation_importance.csv", index=False)
print(importance_df)

plt.figure(figsize=(8, 6))
plt.barh(importance_df["Feature"], importance_df["Importance"], xerr=importance_df["Std"])
plt.xlabel("Permutation Importance")
plt.ylabel("Feature")
plt.title("Permutation Feature Importance - MLP")
plt.tight_layout()
plt.savefig(f"{IMAGE_FOLDER}/mlp_permutation_importance.png")
plt.close()

# ============================================
# Insight
# ============================================
print("\n===Insight ===")
print("1. Logistic Regression and Random Forest performed similarly to MLP on this small dataset.")
print("2. MLP shows potential nonlinear interactions (packaging and transport importance), but metrics did not improve.")
