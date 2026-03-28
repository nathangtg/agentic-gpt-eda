import pandas as pd
from langchain.tools import tool
import json

df_store = {}

@tool
def load_dataset(name: str, path: str) -> str:
    # Load dataset from a file and store it in the df_store with the given name
    if ".csv" in path:
        df = pd.read_csv(path)
    elif ".json" in path:
        df = pd.read_json(path)
    elif ".xlsx" in path: 
        df = pd.read_excel(path)

    df_store[name] = df
    return f"Dataset '{name}' loaded successfully with {len(df)} rows and {len(df.columns)} columns."

@tool
def get_dataset_info(name: str) -> str:
    # Get basic information about the dataset
    if name not in df_store:
        return f"Dataset '{name}' not found."

    df = df_store[name]
    info = {
        "num_rows": len(df),
        "num_columns": len(df.columns),
        "columns": list(df.columns),
        "data_types": df.dtypes.apply(lambda x: str(x)).to_dict(),
        "missing_values": df.isnull().sum().to_dict()
    }
    return json.dumps(info, indent=2)

@tool
def get_column_stats(name: str, column: str) -> str:
    # Get statistics for a specific column in the dataset
    if name not in df_store:
        return f"Dataset '{name}' not found."
    
    df = df_store[name]
    if column not in df.columns:
        return f"Column '{column}' not found in dataset '{name}'."

    col_data = df[column]
    stats = {
        "data_type": str(col_data.dtype),
        "num_missing": col_data.isnull().sum(),
        "num_unique": col_data.nunique(),
        "mean": col_data.mean() if pd.api.types.is_numeric_dtype(col_data) else None,
        "std": col_data.std() if pd.api.types.is_numeric_dtype(col_data) else None,
        "min": col_data.min() if pd.api.types.is_numeric_dtype(col_data) else None,
        "max": col_data.max() if pd.api.types.is_numeric_dtype(col_data) else None,
        "top_values": col_data.value_counts().head(5).to_dict()
    }
    return json.dumps(stats, indent=2)