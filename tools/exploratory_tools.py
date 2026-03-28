from langchain.tools import tool
import pandas as pd
import json

@tool
def summarize_dataset(dataset: pd.DataFrame) -> str:
    summary = {
        "num_rows": len(dataset),
        "num_columns": len(dataset.columns),
        "columns": list(dataset.columns),
        "data_types": dataset.dtypes.apply(lambda x: str(x)).to_dict(),
        "missing_values": dataset.isnull().sum().to_dict()
    }
    return json.dumps(summary, indent=2)

@tool
def generate_correlation_matrix(dataset: pd.DataFrame) -> str:
    numeric_cols = dataset.select_dtypes(include='number').columns
    if len(numeric_cols) < 2:
        return "Not enough numeric columns to generate a correlation matrix."
    
    corr_matrix = dataset[numeric_cols].corr()
    return corr_matrix.to_json(orient='split')

@tool
def get_top_n_unique_values(dataset: pd.DataFrame, column: str, n: int = 5) -> str:
    if column not in dataset.columns:
        return f"Column '{column}' not found in dataset."
    
    top_values = dataset[column].value_counts().head(n).to_dict()
    return json.dumps(top_values, indent=2)

@tool
def generate_histogram(dataset: pd.DataFrame, column: str) -> str:
    if column not in dataset.columns:
        return f"Column '{column}' not found in dataset."
    
    if not pd.api.types.is_numeric_dtype(dataset[column]):
        return f"Column '{column}' is not numeric and cannot be used to generate a histogram."
    
    histogram_data = dataset[column].value_counts().to_dict()
    return json.dumps(histogram_data, indent=2)

@tool
def generate_bar_chart(dataset: pd.DataFrame, column: str) -> str:
    if column not in dataset.columns:
        return f"Column '{column}' not found in dataset."
    
    if pd.api.types.is_numeric_dtype(dataset[column]):
        return f"Column '{column}' is numeric and may not be suitable for a bar chart."
    
    bar_chart_data = dataset[column].value_counts().to_dict()
    return json.dumps(bar_chart_data, indent=2)