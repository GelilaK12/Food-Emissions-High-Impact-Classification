import pandas as pd
import os

# Purpose: Linear regression used as a linear aggregation check to validate data integrity

# ================= Folder Setup =================
OUTPUT_FOLDER = "outputs/linear_agg"
IMAGE_FOLDER = "images/linear_agg"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(IMAGE_FOLDER, exist_ok=True)

# ================= Data Loading =================
data_path = os.path.join("data", "Food_Production.csv")
data = pd.read_csv(data_path)

print("\n=== Dataset shape (rows, columns) ===\n")
print(data.shape)
print("\n")

data.head().to_csv(f"{OUTPUT_FOLDER}/data_sample.csv", index=False)

print("\n=== Dataset info ===\n")
print(data.info())
print("\n")

data.isnull().sum().to_csv(f"{OUTPUT_FOLDER}/missing_values_per_column.csv")

# ================= Data Cleaning =================
# Standardize food names
data["Food product"] = data["Food product"].str.strip().str.title()

# Convert numeric columns accidentally read as objects
object_cols = data.select_dtypes(include="object").columns
for col in object_cols:
    data[col] = pd.to_numeric(data[col].str.replace(",", ""), errors="ignore")

summary_stats = data.describe()
summary_stats.to_csv(f"{OUTPUT_FOLDER}/data_describe_summary.csv")

# ================= Feature Definition =================
features = ["Land use change", "Animal Feed", "Farm", "Processing", "Transport", "Packaging", "Retail"]

X = data[features]
y = data["Total_emissions"]

animal_foods = ["Beef (beef herd)", "Beef (dairy herd)", "Lamb & Mutton", "Pig Meat",
                "Poultry Meat", "Milk", "Cheese", "Eggs", "Fish (farmed)", "Shrimps (farmed)"]

data["animal_based"] = data["Food product"].isin(animal_foods).astype(int)

# Save cleaned data
data.to_csv(f"{OUTPUT_FOLDER}/food_production_cleaned.csv", index=False)

# ================= Train/Test Split =================
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ================= Linear Regression Model =================
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)

# ================= Model Evaluation =================
from sklearn.metrics import mean_squared_error, r2_score

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R² Score:", r2)

# ================= Coefficient Analysis =================
coef_df = pd.DataFrame({"Stage": X.columns, "Coefficient": model.coef_}).sort_values(by="Coefficient", ascending=False)
coef_df.to_csv(f"{OUTPUT_FOLDER}/stage_coefficient.csv", index=False)
