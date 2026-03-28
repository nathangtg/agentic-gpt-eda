import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from agents.base import BaseAgent
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

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []


@st.cache_resource
def _get_chat_llm():
    return BaseAgent.create_default_llm()


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


def _build_chat_context(
    dataset_name: str,
    df: pd.DataFrame,
    quality: dict[str, Any],
    pipeline_result: dict[str, Any] | None,
) -> str:
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = [col for col in df.columns if col not in numeric_cols]

    missing_items = quality.get("missing_values", {}) if isinstance(quality, dict) else {}
    top_missing = sorted(
        [
            (col, vals.get("count", 0), vals.get("rate", 0.0))
            for col, vals in missing_items.items()
            if isinstance(vals, dict)
        ],
        key=lambda row: row[1],
        reverse=True,
    )[:5]

    top_missing_text = "\n".join(
        [f"- {col}: {count} ({rate:.2%})" for col, count, rate in top_missing]
    ) or "- No missing values summary available"

    pipeline_summary = "No orchestrated analysis has been run yet."
    if isinstance(pipeline_result, dict):
        insights = pipeline_result.get("insights", [])
        if isinstance(insights, list) and insights:
            insight_lines = []
            for item in insights[:5]:
                if isinstance(item, dict):
                    insight_lines.append(
                        f"- {item.get('insight', '')} | action: {item.get('recommended_action', '')}"
                    )
            if insight_lines:
                pipeline_summary = "\n".join(insight_lines)

    return (
        f"Dataset name: {dataset_name}\n"
        f"Rows: {len(df)}\n"
        f"Columns: {len(df.columns)}\n"
        f"Numeric columns ({len(numeric_cols)}): {', '.join(numeric_cols[:20])}\n"
        f"Categorical columns ({len(categorical_cols)}): {', '.join(categorical_cols[:20])}\n"
        f"Top missing-value columns:\n{top_missing_text}\n\n"
        f"Latest analysis insights:\n{pipeline_summary}"
    )


def _chat_answer(question: str, context: str) -> str:
    llm = _get_chat_llm()
    prompt = (
        "You are a BI data assistant inside an EDA dashboard. "
        "Answer clearly for non-technical and technical users. "
        "Ground your answer in the provided dataset context and latest insights. "
        "If data is insufficient, say what additional data is needed.\n\n"
        f"Context:\n{context}\n\n"
        f"User question:\n{question}\n\n"
        "Return concise markdown with:\n"
        "1) Direct answer\n"
        "2) Why this matters\n"
        "3) Suggested next chart or analysis step"
    )
    response = llm.invoke(prompt)
    return response.content if hasattr(response, "content") else str(response)


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


def _render_plan_trace(plan: list[dict[str, Any]], execution_results: list[dict[str, Any]]) -> None:
    st.markdown("### AI Plan Trace")
    if not plan:
        st.warning("No plan available to trace.")
        return

    status_by_step: dict[Any, str] = {
        item.get("step"): str(item.get("status", "unknown")) for item in execution_results
    }

    rows = []
    for step in plan:
        step_num = step.get("step")
        methods = _to_str_list(step.get("methods", []))
        rows.append(
            {
                "step": step_num,
                "title": step.get("title", "Untitled"),
                "methods": ", ".join(methods[:2]) + (" ..." if len(methods) > 2 else ""),
                "status": status_by_step.get(step_num, "not_executed"),
            }
        )

    trace_df = pd.DataFrame(rows)
    completed_steps = int((trace_df["status"] == "completed").sum()) if not trace_df.empty else 0
    st.metric("Execution coverage", f"{completed_steps}/{len(trace_df)} steps completed")
    st.dataframe(trace_df, use_container_width=True, hide_index=True)


def _render_bi_dashboard(df: pd.DataFrame, quality: dict[str, Any]) -> None:
    st.subheader("BI Overview")
    num_rows = len(df)
    num_cols = len(df.columns)
    total_cells = max(1, num_rows * num_cols)
    missing_cells = int(df.isna().sum().sum())
    missing_rate = 100.0 * missing_cells / total_cells
    duplicate_rows = int(df.duplicated().sum())

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Rows", num_rows)
    kpi2.metric("Columns", num_cols)
    kpi3.metric("Missing cells", f"{missing_cells} ({missing_rate:.2f}%)")
    kpi4.metric("Duplicate rows", duplicate_rows)

    visual_left, visual_right = st.columns(2)
    with visual_left:
        st.markdown("**Missing Values by Column**")
        missing_items = quality.get("missing_values", {}) if isinstance(quality, dict) else {}
        if missing_items:
            missing_df = pd.DataFrame(
                [
                    {"column": col, "missing_count": vals.get("count", 0)}
                    for col, vals in missing_items.items()
                ]
            ).sort_values("missing_count", ascending=False)
            st.bar_chart(missing_df.set_index("column")["missing_count"], use_container_width=True)
        else:
            st.info("No missing-value details available.")

    with visual_right:
        st.markdown("**Data Types Mix**")
        dtype_counts = df.dtypes.astype(str).value_counts()
        dtype_df = pd.DataFrame({"dtype": dtype_counts.index, "count": dtype_counts.values})
        st.bar_chart(dtype_df.set_index("dtype")["count"], use_container_width=True)

    corr_cols = df.select_dtypes(include="number").columns.tolist()
    if len(corr_cols) >= 2:
        st.markdown("**Top Absolute Correlations**")
        corr = df[corr_cols].corr().abs()
        corr_pairs = []
        for i, left in enumerate(corr.columns):
            for right in corr.columns[i + 1 :]:
                corr_pairs.append({"pair": f"{left} vs {right}", "abs_corr": float(corr.loc[left, right])})
        if corr_pairs:
            pair_df = pd.DataFrame(corr_pairs).sort_values("abs_corr", ascending=False).head(15)
            st.bar_chart(pair_df.set_index("pair")["abs_corr"], use_container_width=True)


def _parse_artifact_item(artifact: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(artifact)
        if isinstance(parsed, dict):
            return parsed
    except Exception:  # noqa: BLE001
        return None
    return None


def _render_generated_visuals(execution_results: list[dict[str, Any]]) -> None:
    st.markdown("### Auto-Generated Visual Outputs")

    artifact_items: list[dict[str, Any]] = []
    for result in execution_results:
        for artifact in _to_str_list(result.get("artifacts", [])):
            item = _parse_artifact_item(artifact)
            if item:
                artifact_items.append(item)

    if not artifact_items:
        st.info("No structured execution artifacts were found to render visuals.")
        return

    rendered_any = False
    for idx, item in enumerate(artifact_items):
        tool_name = str(item.get("tool", ""))
        output = item.get("output")
        if not isinstance(output, str):
            continue

        try:
            payload = json.loads(output)
        except Exception:  # noqa: BLE001
            continue

        if tool_name == "generate_histogram" and isinstance(payload, dict):
            if "bin_edges" in payload and "counts" in payload:
                hist_df = pd.DataFrame(
                    {
                        "bin": [
                            f"{payload['bin_edges'][i]:.2f}-{payload['bin_edges'][i + 1]:.2f}"
                            for i in range(len(payload["counts"]))
                        ],
                        "count": payload["counts"],
                    }
                )
                st.markdown(f"**Histogram (artifact {idx + 1})**")
                st.bar_chart(hist_df.set_index("bin"), use_container_width=True)
                rendered_any = True

        elif tool_name == "generate_bar_chart" and isinstance(payload, dict):
            bar_df = pd.DataFrame({"category": list(payload.keys()), "count": list(payload.values())})
            st.markdown(f"**Category Frequency (artifact {idx + 1})**")
            st.bar_chart(bar_df.set_index("category"), use_container_width=True)
            rendered_any = True

        elif tool_name in {"generate_correlation_matrix", "get_correlation_report"} and isinstance(payload, dict):
            if {"data", "columns", "index"}.issubset(payload.keys()):
                corr_df = pd.DataFrame(
                    data=payload["data"],
                    columns=payload["columns"],
                    index=payload["index"],
                )
                st.markdown(f"**Correlation Matrix (artifact {idx + 1})**")
                st.dataframe(corr_df, use_container_width=True)
                rendered_any = True

        elif tool_name == "fit_regularized_feature_importance" and isinstance(payload, dict):
            impacts = payload.get("top_feature_impacts", [])
            if isinstance(impacts, list) and impacts:
                impact_df = pd.DataFrame(impacts).head(20)
                if {"feature", "absolute_coefficient"}.issubset(impact_df.columns):
                    st.markdown(f"**Regularized Feature Impact (artifact {idx + 1})**")
                    st.bar_chart(
                        impact_df.set_index("feature")["absolute_coefficient"],
                        use_container_width=True,
                    )
                    rendered_any = True

    if not rendered_any:
        st.info("Execution finished, but no chart-compatible artifacts were produced yet.")


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
quality = {}
try:
    info = json.loads(get_dataset_info.invoke({"name": active_dataset}))
    quality = json.loads(get_data_quality_report.invoke({"name": active_dataset}))
except Exception as exc:  # noqa: BLE001
    st.error(f"Unable to build profile: {exc}")

numeric_cols = df.select_dtypes(include="number").columns.tolist()
category_cols = [col for col in df.columns if col not in numeric_cols]

tab_overview, tab_analytics, tab_orchestrated, tab_chat = st.tabs(
    ["Overview", "Analytics", "Orchestrated AI", "AI Chat"]
)

with tab_overview:
    left_col, right_col = st.columns([1, 1])

    with left_col:
        st.subheader("Dataset Preview")
        st.dataframe(df.head(100), use_container_width=True)

    with right_col:
        st.subheader("Dataset Profile")
        if info if "info" in locals() else None:
            st.json(info)
        if quality:
            st.json(quality)

    st.divider()
    _render_bi_dashboard(df, quality if isinstance(quality, dict) else {})

with tab_analytics:
    st.subheader("Interactive Charts")
    chart_left, chart_right, chart_bottom = st.columns([1, 1, 1])

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
        st.markdown("Category Frequency")
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
        st.markdown("Correlation Matrix")
        if len(numeric_cols) >= 2:
            corr_method = st.selectbox("Method", options=["pearson", "spearman", "kendall"], key="corr_method")
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
    st.subheader("Regularized Feature Impact")

    if len(df.columns) >= 2:
        target_col = st.selectbox("Target column", options=df.columns.tolist(), key="target_col")
        reg_col1, reg_col2, reg_col3, reg_col4 = st.columns(4)

        with reg_col1:
            reg_type = st.selectbox("Regularization", options=["ridge", "lasso"], key="reg_type")
        with reg_col2:
            model_type = st.selectbox(
                "Model type", options=["auto", "regression", "classification"], key="model_type"
            )
        with reg_col3:
            alpha = st.number_input("Alpha", min_value=0.0001, max_value=100.0, value=1.0, step=0.1, key="alpha")
        with reg_col4:
            test_size = st.slider("Test split", min_value=0.1, max_value=0.4, value=0.2, step=0.05, key="test_size")

        if st.button("Run regularized model", key="run_regularized"):
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
                    st.bar_chart(
                        impact_df.set_index("feature")["absolute_coefficient"],
                        use_container_width=True,
                    )
                else:
                    st.warning("No feature impacts returned.")
            except Exception as exc:  # noqa: BLE001
                st.error(f"Regularized analysis failed: {exc}")
    else:
        st.warning("Dataset needs at least one feature and one target column for regularized modeling.")

with tab_orchestrated:
    st.subheader("Orchestrated LLM Analysis")

    query = st.text_area(
        "Analysis question",
        value="Analyze this dataset and identify key risk drivers and data quality issues.",
        key="orchestrated_query",
    )
    run_full_plan = st.checkbox("Execute full plan", value=True, key="run_full_plan")
    max_steps = None
    if not run_full_plan:
        max_steps = st.slider("Max plan steps", min_value=1, max_value=10, value=5, key="max_steps")

    if st.button("Run planner -> executor -> reasoner", key="run_orchestrated"):
        with st.spinner("Running orchestrated analysis..."):
            try:
                pipeline_query = (
                    f"Dataset already loaded in memory as '{active_dataset}'. "
                    "Do not call load_dataset unless a valid file path is explicitly provided. "
                    f"Use dataset name '{active_dataset}' for all tool calls. "
                    f"User objective: {query}"
                )
                st.session_state.pipeline_result = run_eda_pipeline(query=pipeline_query, max_steps=max_steps)
            except Exception as exc:  # noqa: BLE001
                st.error(f"Pipeline failed: {exc}")

    result = st.session_state.pipeline_result
    if result:
        st.success("Analysis complete. Review the readable report below.")

        plan_result = result.get("plan", [])
        execution_result = result.get("execution_results", [])
        insights_result = result.get("insights", [])

        _render_plan_trace(
            plan_result if isinstance(plan_result, list) else [],
            execution_result if isinstance(execution_result, list) else [],
        )
        _render_plan(plan_result if isinstance(plan_result, list) else [])
        _render_execution_results(execution_result if isinstance(execution_result, list) else [])
        _render_generated_visuals(execution_result if isinstance(execution_result, list) else [])
        _render_insights(insights_result if isinstance(insights_result, list) else [])

        with st.expander("Raw JSON output (debug)", expanded=False):
            st.json(result)

with tab_chat:
    st.subheader("AI Data Assistant")
    st.caption("Ask questions about the loaded dataset, charts, and orchestrated analysis results.")

    actions_col1, actions_col2 = st.columns([1, 6])
    if actions_col1.button("Clear Chat", key="clear_chat_main"):
        st.session_state.chat_messages = []
        st.rerun()

    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_prompt = st.chat_input("Ask about patterns, anomalies, feature impact, or next best analysis step...")
    if user_prompt and user_prompt.strip():
        st.session_state.chat_messages.append({"role": "user", "content": user_prompt.strip()})
        with st.chat_message("user"):
            st.markdown(user_prompt.strip())

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                context_text = _build_chat_context(
                    dataset_name=active_dataset,
                    df=df,
                    quality=quality if isinstance(quality, dict) else {},
                    pipeline_result=st.session_state.pipeline_result,
                )
                answer = _chat_answer(user_prompt.strip(), context=context_text)
            st.markdown(answer)

        st.session_state.chat_messages.append({"role": "assistant", "content": answer})
