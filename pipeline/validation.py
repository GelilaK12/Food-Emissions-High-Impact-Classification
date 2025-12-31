import pandas as pd
import pandera as pa
from pandera import Column, Check, DataFrameSchema
from pathlib import Path
import re
# -----------------------
# Paths
# -----------------------
ROOT = Path(__file__).parent.parent.resolve()
DATA_PATH = ROOT / "data" / "Food_Production.csv"

# -----------------------
# Function to clean column names
# -----------------------
def clean_columns(columns):
    cleaned = []
    for col in columns:
        col = col.strip().lower()                        
        col = col.replace(" ", "_")                      
        col = re.sub(r"[^0-9a-zA-Z_]", "", col)         
        cleaned.append(col)
    return cleaned

# -----------------------
# Pandera Schema
# -----------------------
schema = DataFrameSchema({
    "land_use_change": Column(float, Check.ge(-5)),
    "animal_feed": Column(float, Check.ge(0)),
    "farm": Column(float, Check.ge(0)),
    "processing": Column(float, Check.ge(0)),
    "transport": Column(float, Check.ge(0)),
    "packaging": Column(float, Check.ge(0)),
    "retail": Column(float, Check.ge(0)),
})

# -----------------------
# Validation Function
# -----------------------
def validate(df: pd.DataFrame) -> pd.DataFrame:
    # Standardize column names
    df.columns = clean_columns(df.columns)
    validated_df = schema.validate(df)
    return validated_df

# -----------------------
# CLI / Test Run
# -----------------------
if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} rows from {DATA_PATH}")
    validated_df = validate(df)
    print("Data validated successfully!")
    print(f"Columns after cleaning: {validated_df.columns.tolist()}")
