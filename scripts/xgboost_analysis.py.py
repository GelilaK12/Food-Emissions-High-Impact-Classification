import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.inspection import permutation_importance
import shap
import matplotlib.pyplot as plt
import os

# ================= Folder Setup =================
OUTPUT_FOLDER = "outputs/xgboost"
IMAGE_FOLDER = "images/xgboost"
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
    data[col] = pd.to_numeric(data[col].str.replace(",", ""), errors="ignore")

data.describe().to_csv(f"{OUTPUT_FOLDER}/data_describe_summary.csv")
data.to_csv(f"{OUTPUT_FOLDER}/food_production_cleaned.csv", index=False)

# ============================================
# Target Definition
# ============================================

threshold = data["Total_emissions"].quantile(0.75)
data["High_Impact"] = (data["Total_emissions"] >= threshold).astype(int)

features = ["Land use change","Animal Feed","Farm","Processing","Transport","Packaging","Retail"]
X = data[features]
y = data["High_Impact"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# ============================================
# XGBoost Baseline
# ============================================

xgb_model = xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, objective="binary:logistic", eval_metric="logloss", use_label_encoder=False, random_state=42)
xgb_model.fit(X_train, y_train)

y_pred_xgb = xgb_model.predict(X_test)

report = classification_report(y_test, y_pred_xgb, output_dict=True)
pd.DataFrame(report).transpose().to_csv(f"{OUTPUT_FOLDER}/xgboost_classification_report.csv")

cm = confusion_matrix(y_test, y_pred_xgb)
pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Predicted 0", "Predicted 1"]).to_csv(f"{OUTPUT_FOLDER}/xgboost_confusion_matrix.csv")

# ============================================
# Feature Importance
# ============================================

xgb_importances = pd.Series(xgb_model.feature_importances_, index=features).sort_values(ascending=False)
xgb_importances.to_csv(f"{OUTPUT_FOLDER}/xgboost_feature_importances.csv")

xgb_importances.plot(kind="bar", title="XGBoost Feature Importances")
plt.tight_layout()
plt.savefig(f"{IMAGE_FOLDER}/xgboost_feature_importances.png")
plt.close()

# ============================================
# SHAP Explanations
# ============================================

explainer = shap.Explainer(xgb_model, X_train)
shap_values = explainer(X_test)

shap.summary_plot(shap_values, X_test, max_display=5, show=False)
plt.tight_layout()
plt.savefig(f"{IMAGE_FOLDER}/shap_summary_top5.png")
plt.close()

shap.summary_plot(shap_values, X_test, show=False)
plt.savefig(f"{IMAGE_FOLDER}/shap_summary_all.png")
plt.close()

# ============================================
# Permutation Importance
# ============================================

perm = permutation_importance(xgb_model, X_test, y_test, n_repeats=10, random_state=42)
perm_result = pd.Series(perm.importances_mean, index=features).sort_values(ascending=False)
perm_result.to_csv(f"{OUTPUT_FOLDER}/xgboost_permutation_importance.csv")

# ============================================
# GridSearchCV
# ============================================

xgb_model = xgb.XGBClassifier(objective="binary:logistic", eval_metric="logloss", use_label_encoder=False, random_state=42)
param_grid = {"n_estimators":[50,100,200], "max_depth":[3,5,7], "learning_rate":[0.01,0.1,0.2]}

grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, scoring="f1", cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
print("Best parameters found:", grid_search.best_params_)

y_pred = best_model.predict(X_test)

report = classification_report(y_test, y_pred, output_dict=True)
pd.DataFrame(report).transpose().to_csv(f"{OUTPUT_FOLDER}/xgboost_gridsearch_classification_report.csv")

cm = confusion_matrix(y_test, y_pred)
pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Predicted 0", "Predicted 1"]).to_csv(f"{OUTPUT_FOLDER}/xgboost_gridsearch_confusion_matrix.csv")

pd.DataFrame(grid_search.cv_results_).to_csv(f"{OUTPUT_FOLDER}/xgboost_gridsearch_cv_results.csv", index=False)
