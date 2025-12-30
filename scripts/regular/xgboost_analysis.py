import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.inspection import permutation_importance
import shap
import matplotlib.pyplot as plt
import os

# ================= BASE DIR =================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
data_path = os.path.join(BASE_DIR, "data", "Food_Production.csv")

# ================= Folder Setup =================
OUTPUT_FOLDER = os.path.join(BASE_DIR, "outputs", "xgboost")
IMAGE_FOLDER = os.path.join(BASE_DIR, "images", "xgboost")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(IMAGE_FOLDER, exist_ok=True)

shap_folder = os.path.join(IMAGE_FOLDER, "shap")
fi_folder = os.path.join(IMAGE_FOLDER, "feature_importances")
os.makedirs(shap_folder, exist_ok=True)
os.makedirs(fi_folder, exist_ok=True)

# ================= Load & Clean Data =================
data = pd.read_csv(data_path)
data["Food product"] = data["Food product"].str.strip().str.title()
for col in data.select_dtypes(include="object").columns:
    data[col] = pd.to_numeric(data[col].str.replace(",", ""), errors="ignore")

data.describe().to_csv(os.path.join(OUTPUT_FOLDER, "data_describe_summary.csv"))
data.to_csv(os.path.join(OUTPUT_FOLDER, "food_production_cleaned.csv"), index=False)

# ================= Target & Features =================
threshold = data["Total_emissions"].quantile(0.75)
data["High_Impact"] = (data["Total_emissions"] >= threshold).astype(int)
features = ["Land use change", "Animal Feed", "Farm", "Processing", "Transport", "Packaging", "Retail"]
X = data[features]
y = data["High_Impact"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# ================= Reusable Model Function =================
def train_evaluate_model(model, X_train, y_train, X_test, y_test, features, prefix):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    report_path = os.path.join(OUTPUT_FOLDER, f"{prefix}_classification_report.csv")
    pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose().to_csv(report_path)
    cm_path = os.path.join(OUTPUT_FOLDER, f"{prefix}_confusion_matrix.csv")
    pd.DataFrame(confusion_matrix(y_test, y_pred), index=["Actual 0", "Actual 1"], columns=["Predicted 0", "Predicted 1"]).to_csv(cm_path)
    if hasattr(model, "feature_importances_"):
        importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
        fi_path = os.path.join(OUTPUT_FOLDER, f"{prefix}_feature_importances.csv")
        importances.to_csv(fi_path)
        plt.figure(figsize=(8,6))
        importances.plot(kind="bar", title=f"{prefix} Feature Importances")
        plt.tight_layout()
        plt.savefig(os.path.join(fi_folder, f"{prefix}_feature_importances.png"))
        plt.close()
    return y_pred, model

# ================= Train XGBoost =================
xgb_model = xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, objective="binary:logistic", eval_metric="logloss", use_label_encoder=False, random_state=42)
y_pred_xgb, xgb_model = train_evaluate_model(xgb_model, X_train, y_train, X_test, y_test, features, "xgboost")

# ================= ROC-AUC =================
y_true = y_test
y_scores = xgb_model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)
print(f"XGBoost ROC-AUC: {roc_auc:.3f}")

plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, color="blue", lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
plt.plot([0, 1], [0, 1], color="red", lw=1, linestyle="--", label="Random guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("XGBoost ROC Curve - High Impact Food Classification")
plt.legend(loc="lower right")
plt.grid(True)
plt.savefig(f"{IMAGE_FOLDER}/roc_curve_xgboost.png")
plt.close()

# ================= SHAP Explanations =================
explainer = shap.Explainer(xgb_model, X_train)
shap_values = explainer(X_test)

plt.figure(figsize=(8,6))
shap.summary_plot(shap_values, X_test, max_display=5, show=False)
plt.tight_layout()
plt.savefig(os.path.join(shap_folder, "shap_summary_top5.png"))
plt.close()

plt.figure(figsize=(8,6))
shap.summary_plot(shap_values, X_test, show=False)
plt.tight_layout()
plt.savefig(os.path.join(shap_folder, "shap_summary_all.png"))
plt.close()

# ================= Permutation Importance =================
perm = permutation_importance(xgb_model, X_test, y_test, n_repeats=10, random_state=42)
perm_result = pd.Series(perm.importances_mean, index=features).sort_values(ascending=False)
perm_result.to_csv(os.path.join(OUTPUT_FOLDER, "xgboost_perm_result.csv"))

plt.figure(figsize=(8,6))
perm_result.plot(kind="bar", title="Permutation Importance")
plt.tight_layout()
plt.savefig(os.path.join(fi_folder, "xgboost_perm_importance.png"))
plt.close()

# ================= GridSearchCV Hyperparameter Tuning =================
xgb_model_grid = xgb.XGBClassifier(objective="binary:logistic", eval_metric="logloss", use_label_encoder=False, random_state=42)
param_grid = {"n_estimators":[50,100,200], "max_depth":[3,5,7], "learning_rate":[0.01,0.1,0.2]}
grid_search = GridSearchCV(estimator=xgb_model_grid, param_grid=param_grid, scoring="f1", cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)

pd.DataFrame(grid_search.cv_results_).to_csv(os.path.join(OUTPUT_FOLDER, "xgboost_gridsearch_cv_results.csv"), index=False)
best_model = grid_search.best_estimator_
print("Best parameters found:", grid_search.best_params_)

y_pred_grid, _ = train_evaluate_model(best_model, X_train, y_train, X_test, y_test, features, "xgboost_gridsearch")
