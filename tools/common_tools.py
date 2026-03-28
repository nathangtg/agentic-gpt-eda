import json
from typing import Any

import numpy as np
import pandas as pd
from langchain.tools import tool
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

df_store = {}


def _to_python(value: Any) -> Any:
    if isinstance(value, (np.integer, np.int64, np.int32)):
        return int(value)
    if isinstance(value, (np.floating, np.float64, np.float32)):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _parse_column_list(columns: str) -> list[str]:
    return [col.strip() for col in columns.split(",") if col.strip()]


def _resolve_model_type(target: pd.Series, model_type: str) -> str:
    if model_type in {"regression", "classification"}:
        return model_type

    # Auto mode: numeric target with enough unique values => regression; otherwise classification.
    if pd.api.types.is_numeric_dtype(target) and target.nunique(dropna=True) > 10:
        return "regression"
    return "classification"


def _build_preprocessor(X: pd.DataFrame) -> tuple[ColumnTransformer, list[str], list[str]]:
    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = [col for col in X.columns if col not in numeric_features]

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return preprocessor, numeric_features, categorical_features

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
def list_loaded_datasets() -> str:
    """List datasets that are currently loaded in memory."""
    datasets = {
        name: {
            "rows": int(len(df)),
            "columns": int(len(df.columns)),
        }
        for name, df in df_store.items()
    }
    return json.dumps(datasets, indent=2)

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
def get_data_quality_report(name: str) -> str:
    """Return deterministic data quality metrics to ground model reasoning."""
    if name not in df_store:
        return f"Dataset '{name}' not found."

    df = df_store[name]
    duplicate_count = int(df.duplicated().sum())
    null_counts = df.isnull().sum()
    row_count = len(df)

    report = {
        "num_rows": int(row_count),
        "num_columns": int(len(df.columns)),
        "duplicate_rows": duplicate_count,
        "duplicate_row_rate": float(duplicate_count / row_count) if row_count else 0.0,
        "missing_values": {
            col: {
                "count": int(null_counts[col]),
                "rate": float(null_counts[col] / row_count) if row_count else 0.0,
            }
            for col in df.columns
        },
    }
    return json.dumps(report, indent=2)

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


@tool
def get_numeric_distribution_report(name: str, columns: str = "", bins: int = 10) -> str:
    """Profile numeric columns with robust distribution statistics."""
    if name not in df_store:
        return f"Dataset '{name}' not found."

    df = df_store[name]
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    requested = _parse_column_list(columns)
    if requested:
        numeric_cols = [col for col in numeric_cols if col in requested]

    if not numeric_cols:
        return "No numeric columns available for distribution report."

    report = {}
    for col in numeric_cols:
        series = df[col].dropna()
        if series.empty:
            report[col] = {"error": "Column has only missing values."}
            continue

        q1 = float(series.quantile(0.25))
        q3 = float(series.quantile(0.75))
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outlier_mask = (series < lower_bound) | (series > upper_bound)

        hist_counts, hist_edges = np.histogram(series.to_numpy(), bins=bins)

        report[col] = {
            "count": int(series.shape[0]),
            "mean": float(series.mean()),
            "std": float(series.std(ddof=1)) if series.shape[0] > 1 else 0.0,
            "min": float(series.min()),
            "q1": q1,
            "median": float(series.median()),
            "q3": q3,
            "max": float(series.max()),
            "skew": float(series.skew()),
            "kurtosis": float(series.kurtosis()),
            "iqr": float(iqr),
            "outlier_count_iqr": int(outlier_mask.sum()),
            "outlier_rate_iqr": float(outlier_mask.mean()) if series.shape[0] else 0.0,
            "histogram": {
                "bin_edges": [float(v) for v in hist_edges.tolist()],
                "counts": [int(v) for v in hist_counts.tolist()],
            },
        }

    return json.dumps(report, indent=2)


@tool
def get_correlation_report(name: str, target: str = "", method: str = "pearson", top_k: int = 10) -> str:
    """Return pairwise numeric correlations or top correlations to a target column."""
    if name not in df_store:
        return f"Dataset '{name}' not found."

    df = df_store[name]
    numeric_df = df.select_dtypes(include=["number"])
    if numeric_df.shape[1] < 2:
        return "Not enough numeric columns to compute correlations."

    if method not in {"pearson", "spearman", "kendall"}:
        return "Invalid method. Choose one of: pearson, spearman, kendall."

    corr = numeric_df.corr(method=method)

    if target:
        if target not in corr.columns:
            return f"Target column '{target}' is not numeric or not found."

        target_corr = corr[target].drop(labels=[target], errors="ignore").dropna()
        ranked = target_corr.reindex(target_corr.abs().sort_values(ascending=False).index)
        result = {
            "target": target,
            "method": method,
            "top_correlations": {
                k: float(v)
                for k, v in ranked.head(max(1, top_k)).to_dict().items()
            },
        }
        return json.dumps(result, indent=2)

    return corr.to_json(orient="split")


@tool
def fit_regularized_feature_importance(
    name: str,
    target_column: str,
    regularization: str = "ridge",
    model_type: str = "auto",
    alpha: float = 1.0,
    test_size: float = 0.2,
    random_state: int = 42,
) -> str:
    """Fit Ridge/Lasso/Logistic regularized model and return ranked feature impacts."""
    if name not in df_store:
        return f"Dataset '{name}' not found."

    if regularization not in {"ridge", "lasso"}:
        return "regularization must be one of: ridge, lasso"

    if not 0 < test_size < 0.9:
        return "test_size must be between 0 and 0.9"

    df = df_store[name].copy()
    if target_column not in df.columns:
        return f"Target column '{target_column}' not found in dataset '{name}'."

    y = df[target_column]
    X = df.drop(columns=[target_column])
    if X.shape[1] == 0:
        return "No feature columns available after removing target column."

    resolved_model_type = _resolve_model_type(y, model_type)

    valid_mask = y.notna()
    X = X.loc[valid_mask]
    y = y.loc[valid_mask]

    if resolved_model_type == "classification":
        y = y.astype("category")
        if y.nunique() < 2:
            return "Classification target must have at least 2 classes."

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if resolved_model_type == "classification" and y.nunique() <= 20 else None,
    )

    preprocessor, _, _ = _build_preprocessor(X)

    if resolved_model_type == "regression":
        model = Ridge(alpha=alpha) if regularization == "ridge" else Lasso(alpha=alpha)
    else:
        if regularization == "ridge":
            model = LogisticRegression(
                penalty="l2",
                C=1.0 / max(alpha, 1e-8),
                solver="lbfgs",
                max_iter=5000,
            )
        else:
            model = LogisticRegression(
                penalty="l1",
                C=1.0 / max(alpha, 1e-8),
                solver="saga",
                max_iter=5000,
            )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])
    pipeline.fit(X_train, y_train)

    model_obj = pipeline.named_steps["model"]
    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out().tolist()

    if resolved_model_type == "classification" and getattr(model_obj, "coef_", None) is not None:
        coef_matrix = np.abs(model_obj.coef_)
        if coef_matrix.ndim == 2 and coef_matrix.shape[0] > 1:
            coefficients = coef_matrix.mean(axis=0)
        else:
            coefficients = coef_matrix.ravel()
    else:
        coefficients = np.abs(np.asarray(model_obj.coef_).ravel())

    feature_impacts = sorted(
        [
            {
                "feature": feature,
                "absolute_coefficient": float(abs_coef),
            }
            for feature, abs_coef in zip(feature_names, coefficients)
        ],
        key=lambda row: row["absolute_coefficient"],
        reverse=True,
    )

    predictions = pipeline.predict(X_test)
    metrics: dict[str, Any]
    if resolved_model_type == "regression":
        mse = mean_squared_error(y_test, predictions)
        metrics = {
            "r2": float(r2_score(y_test, predictions)),
            "mae": float(mean_absolute_error(y_test, predictions)),
            "rmse": float(np.sqrt(mse)),
        }
    else:
        metrics = {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "f1_weighted": float(f1_score(y_test, predictions, average="weighted")),
        }

        # ROC AUC is only straightforward for binary classification with probabilities.
        if y.nunique() == 2 and hasattr(pipeline, "predict_proba"):
            positive_scores = pipeline.predict_proba(X_test)[:, 1]
            metrics["roc_auc"] = float(roc_auc_score(y_test, positive_scores))

    result = {
        "dataset": name,
        "target_column": target_column,
        "model_type": resolved_model_type,
        "regularization": regularization,
        "alpha": float(alpha),
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "metrics": {k: _to_python(v) for k, v in metrics.items()},
        "top_feature_impacts": feature_impacts[:25],
    }
    return json.dumps(result, indent=2)