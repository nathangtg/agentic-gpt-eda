import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from main import run_eda_pipeline
from tools.common_tools import (
    df_store,
    fit_regularized_feature_importance,
    get_data_quality_report,
    get_dataset_info,
)
from tools.exploratory_tools import (
    generate_bar_chart,
    generate_correlation_matrix,
    generate_histogram,
)


st.set_page_config(page_title="Agentic EDA", layout="wide")
st.title("Agentic EDA Dashboard")
st.caption("Plan, execute, reason, and validate insights with grounded dataset tools.")


if "active_dataset" not in st.session_state:
    st.session_state.active_dataset = "main"

if "pipeline_result" not in st.session_state:
    st.session_state.pipeline_result = None


def _read_uploaded_dataset(uploaded_file) -> pd.DataFrame:
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(uploaded_file)
    if suffix == ".json":
        return pd.read_json(uploaded_file)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(uploaded_file)

    raise ValueError("Unsupported file type. Please upload CSV, JSON, or XLSX.")


def _to_str_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if value is None:
        return []
    return [str(value)]


def _render_plan(plan: list[dict[str, Any]]) -> None:
    st.markdown("### Execution Plan")
    if not plan:
        st.warning("No plan steps were returned.")
        return

    for step in plan:
        step_num = step.get("step", "?")
        title = step.get("title", "Untitled step")
        objective = step.get("objective", "")
        methods = _to_str_list(step.get("methods", []))
        expected_output = step.get("expected_output", "")

        with st.expander(f"Step {step_num}: {title}", expanded=False):
            if objective:
                st.markdown(f"**Objective:** {objective}")
            if methods:
                st.markdown("**Methods:**")
                for method in methods:
                    st.markdown(f"- {method}")
            if expected_output:
                st.markdown(f"**Expected output:** {expected_output}")


def _render_execution_results(execution_results: list[dict[str, Any]]) -> None:
    st.markdown("### Step Execution Results")
    if not execution_results:
        st.warning("No execution results were returned.")
        return

    status_counts = {}
    for result in execution_results:
        status = str(result.get("status", "unknown"))
        status_counts[status] = status_counts.get(status, 0) + 1

    metric_cols = st.columns(max(1, len(status_counts)))
    for idx, (status, count) in enumerate(status_counts.items()):
        metric_cols[idx].metric(label=f"{status.title()} steps", value=count)

    for result in execution_results:
        step_num = result.get("step", "?")
        status = str(result.get("status", "unknown")).lower()
        badge = "OK" if status == "completed" else "WARN" if status == "needs_input" else "ERR"

        with st.expander(f"[{badge}] Step {step_num} ({status})", expanded=False):
            actions = _to_str_list(result.get("actions_taken", []))
            observations = _to_str_list(result.get("observations", []))
            artifacts = _to_str_list(result.get("artifacts", []))
            next_action = result.get("next_recommended_action", "")

            if actions:
                st.markdown("**Actions taken:**")
                for action in actions:
                    st.markdown(f"- {action}")

            if observations:
                st.markdown("**Observations:**")
                for observation in observations:
                    st.markdown(f"- {observation}")

            if artifacts:
                st.markdown("**Artifacts:**")
                for artifact in artifacts:
                    st.markdown(f"- {artifact}")

            if next_action:
                st.markdown(f"**Next recommended action:** {next_action}")


def _render_insights(insights: list[dict[str, Any]]) -> None:
    st.markdown("### Final Insights and Recommendations")
    if not insights:
        st.warning("No insights were returned.")
        return

    insights_rows: list[dict[str, Any]] = []
    for insight in insights:
        insights_rows.append(
            {
                "insight": insight.get("insight", ""),
                "confidence": insight.get("confidence", None),
                "recommended_action": insight.get("recommended_action", ""),
            }
        )

    insight_df = pd.DataFrame(insights_rows)
    if "confidence" in insight_df.columns:
        insight_df = insight_df.sort_values(by="confidence", ascending=False, na_position="last")

    st.dataframe(insight_df, use_container_width=True)

    top_insights = insight_df.head(3).to_dict(orient="records")
    st.markdown("**Executive summary:**")
    for row in top_insights:
        insight_text = row.get("insight") or "No insight text"
        action_text = row.get("recommended_action") or "No recommended action"
        confidence = row.get("confidence")
        conf_text = f" (confidence: {confidence:.2f})" if isinstance(confidence, (int, float)) else ""
        st.markdown(f"- {insight_text}{conf_text}. Action: {action_text}")


with st.sidebar:
    st.header("Dataset")
    uploaded_file = st.file_uploader(
        "Upload dataset",
        type=["csv", "json", "xlsx", "xls"],
        help="Supported formats: CSV, JSON, XLSX",
    )
    default_name = (
        Path(uploaded_file.name).stem if uploaded_file is not None else st.session_state.active_dataset
    )
    dataset_name = st.text_input("Dataset name", value=default_name)

    if st.button("Load uploaded dataset", type="primary"):
        if uploaded_file is None:
            st.error("Please upload a dataset file first.")
        elif not dataset_name.strip():
            st.error("Please provide a dataset name.")
        else:
            try:
                df_store[dataset_name] = _read_uploaded_dataset(uploaded_file)
                st.session_state.active_dataset = dataset_name
                rows = len(df_store[dataset_name])
                cols = len(df_store[dataset_name].columns)
                st.success(
                    f"Dataset '{dataset_name}' uploaded successfully with {rows} rows and {cols} columns."
                )
            except Exception as exc:  # noqa: BLE001
                st.error(f"Failed to load uploaded dataset: {exc}")

active_dataset = st.session_state.active_dataset

if active_dataset not in df_store:
    st.info("Load a dataset from the sidebar to begin analysis.")
    st.stop()


df = df_store[active_dataset]
left_col, right_col = st.columns([1, 1])

with left_col:
    st.subheader("Dataset preview")
    st.dataframe(df.head(100), use_container_width=True)

with right_col:
    st.subheader("Dataset profile")
    try:
        info = json.loads(get_dataset_info.invoke({"name": active_dataset}))
        quality = json.loads(get_data_quality_report.invoke({"name": active_dataset}))
        st.json(info)
        st.json(quality)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Unable to build profile: {exc}")

st.divider()
st.subheader("Charts")
chart_left, chart_right, chart_bottom = st.columns([1, 1, 1])

numeric_cols = df.select_dtypes(include="number").columns.tolist()
category_cols = [col for col in df.columns if col not in numeric_cols]

with chart_left:
    st.markdown("Histogram")
    if numeric_cols:
        hist_col = st.selectbox("Numeric column", options=numeric_cols, key="hist_col")
        hist_bins = st.slider("Bins", min_value=5, max_value=50, value=15, key="hist_bins")
        hist_raw = generate_histogram.invoke(
            {"name": active_dataset, "column": hist_col, "bins": hist_bins}
        )
        hist_json = json.loads(hist_raw)
        hist_df = pd.DataFrame(
            {
                "bin": [
                    f"{hist_json['bin_edges'][i]:.2f}-{hist_json['bin_edges'][i + 1]:.2f}"
                    for i in range(len(hist_json["counts"]))
                ],
                "count": hist_json["counts"],
            }
        )
        st.bar_chart(hist_df.set_index("bin"), use_container_width=True)
    else:
        st.warning("No numeric columns available.")

with chart_right:
    st.markdown("Category frequency")
    if category_cols:
        bar_col = st.selectbox("Categorical column", options=category_cols, key="bar_col")
        top_n = st.slider("Top N", min_value=5, max_value=50, value=20, key="bar_top_n")
        bar_raw = generate_bar_chart.invoke({"name": active_dataset, "column": bar_col, "n": top_n})
        bar_json = json.loads(bar_raw)
        bar_df = pd.DataFrame(
            {
                "category": list(bar_json.keys()),
                "count": list(bar_json.values()),
            }
        )
        st.bar_chart(bar_df.set_index("category"), use_container_width=True)
    else:
        st.warning("No categorical columns available.")

with chart_bottom:
    st.markdown("Correlation matrix")
    if len(numeric_cols) >= 2:
        corr_method = st.selectbox("Method", options=["pearson", "spearman", "kendall"])
        corr_raw = generate_correlation_matrix.invoke({"name": active_dataset, "method": corr_method})
        corr_json = json.loads(corr_raw)
        corr_df = pd.DataFrame(
            data=corr_json["data"],
            columns=corr_json["columns"],
            index=corr_json["index"],
        )
        st.dataframe(corr_df, use_container_width=True)
    else:
        st.warning("Need at least 2 numeric columns.")

st.divider()
st.subheader("Regularized feature impact")

if len(df.columns) >= 2:
    target_col = st.selectbox("Target column", options=df.columns.tolist())
    reg_col1, reg_col2, reg_col3, reg_col4 = st.columns(4)

    with reg_col1:
        reg_type = st.selectbox("Regularization", options=["ridge", "lasso"])
    with reg_col2:
        model_type = st.selectbox("Model type", options=["auto", "regression", "classification"])
    with reg_col3:
        alpha = st.number_input("Alpha", min_value=0.0001, max_value=100.0, value=1.0, step=0.1)
    with reg_col4:
        test_size = st.slider("Test split", min_value=0.1, max_value=0.4, value=0.2, step=0.05)

    if st.button("Run regularized model"):
        try:
            reg_raw = fit_regularized_feature_importance.invoke(
                {
                    "name": active_dataset,
                    "target_column": target_col,
                    "regularization": reg_type,
                    "model_type": model_type,
                    "alpha": float(alpha),
                    "test_size": float(test_size),
                }
            )
            reg_json = json.loads(reg_raw)

            st.markdown("Metrics")
            st.json(reg_json.get("metrics", {}))

            impact_df = pd.DataFrame(reg_json.get("top_feature_impacts", []))
            if not impact_df.empty:
                impact_df = impact_df.head(20)
                st.markdown("Top feature impacts")
                st.bar_chart(impact_df.set_index("feature")["absolute_coefficient"], use_container_width=True)
            else:
                st.warning("No feature impacts returned.")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Regularized analysis failed: {exc}")
else:
    st.warning("Dataset needs at least one feature and one target column for regularized modeling.")

st.divider()
st.subheader("Orchestrated LLM analysis")

query = st.text_area(
    "Analysis question",
    value="Analyze this dataset and identify key risk drivers and data quality issues.",
)
max_steps = st.slider("Max plan steps", min_value=1, max_value=10, value=3)

if st.button("Run planner -> executor -> reasoner"):
    with st.spinner("Running orchestrated analysis..."):
        try:
            st.session_state.pipeline_result = run_eda_pipeline(query=query, max_steps=max_steps)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Pipeline failed: {exc}")

result = st.session_state.pipeline_result
if result:
    st.success("Analysis complete. Review the readable report below.")

    plan_result = result.get("plan", [])
    execution_result = result.get("execution_results", [])
    insights_result = result.get("insights", [])

    _render_plan(plan_result if isinstance(plan_result, list) else [])
    _render_execution_results(execution_result if isinstance(execution_result, list) else [])
    _render_insights(insights_result if isinstance(insights_result, list) else [])

    with st.expander("Raw JSON output (debug)", expanded=False):
        st.json(result)
