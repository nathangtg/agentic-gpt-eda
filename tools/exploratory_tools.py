import json

import numpy as np
import pandas as pd
from langchain.tools import tool

from tools.common_tools import df_store

@tool
def summarize_dataset(name: str) -> str:
    """Return a compact summary of rows, columns, dtypes, and missing values."""
    if name not in df_store:
        return f"Dataset '{name}' not found."

    dataset = df_store[name]
    summary = {
        "num_rows": len(dataset),
        "num_columns": len(dataset.columns),
        "columns": list(dataset.columns),
        "data_types": dataset.dtypes.apply(lambda x: str(x)).to_dict(),
        "missing_values": dataset.isnull().sum().to_dict()
    }
    return json.dumps(summary, indent=2)

@tool
def generate_correlation_matrix(name: str, method: str = "pearson") -> str:
    """Compute a numeric correlation matrix for a loaded dataset."""
    if name not in df_store:
        return f"Dataset '{name}' not found."

    dataset = df_store[name]
    numeric_cols = dataset.select_dtypes(include='number').columns
    if len(numeric_cols) < 2:
        return "Not enough numeric columns to generate a correlation matrix."

    if method not in {"pearson", "spearman", "kendall"}:
        return "Invalid method. Choose one of: pearson, spearman, kendall."
    
    corr_matrix = dataset[numeric_cols].corr(method=method)
    return corr_matrix.to_json(orient='split')

@tool
def get_top_n_unique_values(name: str, column: str, n: int = 5) -> str:
    """Return top-N most frequent values for a dataset column."""
    if name not in df_store:
        return f"Dataset '{name}' not found."

    dataset = df_store[name]
    if column not in dataset.columns:
        return f"Column '{column}' not found in dataset '{name}'."
    
    top_values = dataset[column].value_counts().head(n).to_dict()
    return json.dumps(top_values, indent=2)

@tool
def generate_histogram(name: str, column: str, bins: int = 10) -> str:
    """Compute histogram bin edges and counts for a numeric column."""
    if name not in df_store:
        return f"Dataset '{name}' not found."

    dataset = df_store[name]
    if column not in dataset.columns:
        return f"Column '{column}' not found in dataset '{name}'."
    
    if not pd.api.types.is_numeric_dtype(dataset[column]):
        return f"Column '{column}' is not numeric and cannot be used to generate a histogram."
    
    values = dataset[column].dropna().to_numpy()
    counts, edges = np.histogram(values, bins=bins)
    histogram_data = {
        "bin_edges": [float(v) for v in edges.tolist()],
        "counts": [int(v) for v in counts.tolist()],
    }
    return json.dumps(histogram_data, indent=2)

@tool
def generate_bar_chart(name: str, column: str, n: int = 20) -> str:
    """Return top category frequencies for a categorical column."""
    if name not in df_store:
        return f"Dataset '{name}' not found."

    dataset = df_store[name]
    if column not in dataset.columns:
        return f"Column '{column}' not found in dataset '{name}'."
    
    if pd.api.types.is_numeric_dtype(dataset[column]):
        return f"Column '{column}' is numeric and may not be suitable for a bar chart."
    
    bar_chart_data = dataset[column].value_counts().head(n).to_dict()
    return json.dumps(bar_chart_data, indent=2)