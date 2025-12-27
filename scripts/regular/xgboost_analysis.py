import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.inspection import permutation_importance
import shap
import matplotlib.pyplot as plt
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
data_path = os.path.join(BASE_DIR, "data", "Food_Production.csv")

# ================= Folder Setup =================
OUTPUT_FOLDER = os.path.join(BASE_DIR, "outputs", "xgboost_analysis")
IMAGE_FOLDER = os.path.join(BASE_DIR, "images", "xgboost_analysis")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(IMAGE_FOLDER, exist_ok=True)

data_path = os.path.join("..", "..", "data", "Food_Production.csv")
data = pd.read_csv(data_path)
data["Food product"] = data["Food product"].str.strip().str.title()
for col in data.select_dtypes(include="object").columns:
    data[col] = pd.to_numeric(data[col].str.replace(",", ""), errors="ignore")
data.describe().to_csv("outputs/data_describe_summary.csv")
data.to_csv("outputs/food_production_cleaned.csv", index=False)

# ============================================
# Target & Features
# ============================================
threshold = data["Total_emissions"].quantile(0.75)
data["High_Impact"] = (data["Total_emissions"] >= threshold).astype(int)
features = ["Land use change","Animal Feed","Farm","Processing","Transport","Packaging","Retail"]
X, y = data[features], data["High_Impact"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# ============================================
# Reusable Function
# ============================================
def train_evaluate_model(model, X_train, y_train, X_test, y_test, features, prefix, output_folder="baseline"):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose().to_csv(f"outputs/xgboost/{output_folder}/{prefix}_classification_report.csv")
    pd.DataFrame(confusion_matrix(y_test, y_pred), index=["Actual 0","Actual 1"], columns=["Predicted 0","Predicted 1"]).to_csv(f"outputs/xgboost/{output_folder}/{prefix}_confusion_matrix.csv")
    
    if hasattr(model, "feature_importances_"):
        importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
        importances.to_csv(f"outputs/xgboost/{output_folder}/{prefix}_feature_importances.csv")
        importances.plot(kind="bar", title=f"{prefix} Feature Importances")
        plt.tight_layout()
        plt.savefig(f"images/xgboost/feature_importances/{prefix}_feature_importances.png")
        plt.close()
    
    return y_pred, model

# ============================================
# Baseline XGBoost
# ============================================
xgb_model = xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, objective="binary:logistic", eval_metric="logloss", use_label_encoder=False, random_state=42)
y_pred_xgb, xgb_model = train_evaluate_model(xgb_model, X_train, y_train, X_test, y_test, features, "xgboost_baseline", "baseline")

# ============================================
# SHAP Explanations
# ============================================
explainer = shap.Explainer(xgb_model, X_train)
shap_values = explainer(X_test)
plt.figure(figsize=(8,6))
shap.summary_plot(shap_values, X_test, max_display=5, show=False)
plt.tight_layout()
plt.savefig("images/xgboost/shap/shap_summary_top5.png")
shap.summary_plot(shap_values, X_test, show=False)
plt.tight_layout()
plt.savefig("images/xgboost/shap/shap_summary_all.png")

# ============================================
# Permutation Importance
# ============================================
perm = permutation_importance(xgb_model, X_test, y_test, n_repeats=10, random_state=42)
perm_result = pd.Series(perm.importances_mean, index=features).sort_values(ascending=False)
perm_result.to_csv("outputs/xgboost/baseline/xgboost_perm_result.csv")
perm_result.plot(kind="bar", title="Permutation Importance")
plt.tight_layout()
plt.savefig("images/xgboost/feature_importances/xgboost_perm_importance.png")
plt.close()

# ============================================
# GridSearchCV Hyperparameter Tuning
# ============================================
xgb_model_grid = xgb.XGBClassifier(objective="binary:logistic", eval_metric="logloss", use_label_encoder=False, random_state=42)
param_grid = {"n_estimators":[50,100,200], "max_depth":[3,5,7], "learning_rate":[0.01,0.1,0.2]}
grid_search = GridSearchCV(estimator=xgb_model_grid, param_grid=param_grid, scoring="f1", cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)
pd.DataFrame(grid_search.cv_results_).to_csv("outputs/xgboost/gridsearch/xgboost_gridsearch_cv_results.csv", index=False)

best_model = grid_search.best_estimator_
print("Best parameters found:", grid_search.best_params_)
y_pred_grid, _ = train_evaluate_model(best_model, X_train, y_train, X_test, y_test, features, "xgboost_gridsearch", "gridsearch")
