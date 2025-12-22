import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.inspection import permutation_importance

# ================= Folder Setup =================
OUTPUT_FOLDER = "outputs/logistic_rf"
IMAGE_FOLDER = "images/logistic_rf"
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
# Standardize food names
data["Food product"] = data["Food product"].str.strip().str.title()

# Convert numeric columns accidentally read as objects
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

features = ["Land use change", "Animal Feed", "Farm", "Processing", "Transport", "Packaging", "Retail"]
X = data[features]
y = data["High_Impact"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# ============================================
# Logistic Regression (Baseline Model)
# ============================================
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)
y_pred_log = log_model.predict(X_test)

# Classification report
log_report = classification_report(y_test, y_pred_log, output_dict=True)
pd.DataFrame(log_report).transpose().to_csv(f"{OUTPUT_FOLDER}/logistic_regression_classification_report.csv")

# Confusion matrix
log_cm = confusion_matrix(y_test, y_pred_log)
pd.DataFrame(log_cm, index=["Actual 0", "Actual 1"], columns=["Predicted 0", "Predicted 1"]).to_csv(f"{OUTPUT_FOLDER}/logistic_regression_confusion_matrix.csv")

# Coefficient interpretation
coef_df = pd.DataFrame({"Stage": features, "Coefficient": log_model.coef_[0]}).sort_values(by="Coefficient", ascending=False)
coef_df.to_csv(f"{OUTPUT_FOLDER}/logistic_regression_coefficients.csv")

plt.figure(figsize=(8, 5))
plt.barh(coef_df["Stage"], coef_df["Coefficient"])
plt.xlabel("Logistic Regression Coefficient")
plt.title("Logistic Regression Coefficients (High-Impact Prediction)")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(f"{IMAGE_FOLDER}/logistic_regression_coefficients.png")
plt.close()

# ============================================
# Random Forest Classifier (Nonlinear Benchmark)
# ============================================
rf_model = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42, class_weight="balanced")
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# Classification report
rf_report = classification_report(y_test, y_pred_rf, output_dict=True)
pd.DataFrame(rf_report).transpose().to_csv(f"{OUTPUT_FOLDER}/random_forest_classification_report.csv")

# Confusion matrix
rf_cm = confusion_matrix(y_test, y_pred_rf)
pd.DataFrame(rf_cm, index=["Actual 0", "Actual 1"], columns=["Predicted 0", "Predicted 1"]).to_csv(f"{OUTPUT_FOLDER}/random_forest_confusion_matrix.csv")

# Feature importance (impurity-based)
rf_importances = pd.Series(rf_model.feature_importances_, index=features).sort_values(ascending=False)
rf_importances.to_csv(f"{OUTPUT_FOLDER}/random_forest_feature_importances.csv")

plt.figure(figsize=(8, 5))
rf_importances.plot(kind="bar")
plt.ylabel("Importance")
plt.title("Random Forest Feature Importances")
plt.tight_layout()
plt.savefig(f"{IMAGE_FOLDER}/random_forest_feature_importances.png")
plt.close()

# ============================================
# Diagnostics & Interpretation
# ============================================
# Correlation with target
corr_df = X_train.copy()
corr_df["High_Impact"] = y_train
corr_df.corr()["High_Impact"].sort_values(ascending=False).to_csv(f"{OUTPUT_FOLDER}/feature_correlations.csv")

# Distribution comparison for key features
plt.figure(figsize=(8, 5))
sns.boxplot(x=y_train, y=X_train["Farm"])
plt.title("Farm Emissions by High-Impact Class")
plt.tight_layout()
plt.savefig(f"{IMAGE_FOLDER}/farm_by_class.png")
plt.close()

plt.figure(figsize=(8, 5))
sns.boxplot(x=y_train, y=X_train["Land use change"])
plt.title("Land Use Change by High-Impact Class")
plt.tight_layout()
plt.savefig(f"{IMAGE_FOLDER}/land_use_change_by_class.png")
plt.close()

# Permutation importance (unique contribution)
perm_result = permutation_importance(rf_model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=1)
perm_importances = pd.Series(perm_result.importances_mean, index=features).sort_values(ascending=False)
perm_importances.to_csv(f"{OUTPUT_FOLDER}/random_forest_permutation_importance.csv")
print("\nPermutation Importances:")
print(perm_importances)
