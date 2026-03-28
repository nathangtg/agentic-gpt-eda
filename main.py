import json

from agents.base import BaseAgent
from agents.executor import ExecutorAgent
from agents.orchestrator import EDAOrchestrator
from agents.planner import PlannerAgent
from agents.reason import ReasonAgent
from tools.common_tools import (
    fit_regularized_feature_importance,
    get_column_stats,
    get_correlation_report,
    get_data_quality_report,
    get_dataset_info,
    get_numeric_distribution_report,
    list_loaded_datasets,
    load_dataset,
)
from tools.exploratory_tools import (
    generate_bar_chart,
    generate_correlation_matrix,
    generate_histogram,
    get_top_n_unique_values,
    summarize_dataset,
)

llm = BaseAgent.create_default_llm()

# Tools loading 
def load_tools():
    return [
        load_dataset,
        list_loaded_datasets,
        get_dataset_info,
        get_data_quality_report,
        get_column_stats,
        get_numeric_distribution_report,
        get_correlation_report,
        fit_regularized_feature_importance,
        summarize_dataset,
        generate_correlation_matrix,
        get_top_n_unique_values,
        generate_histogram,
        generate_bar_chart,
    ]


tools = load_tools()
planner = PlannerAgent(llm=llm, tools=tools)
reasoner = ReasonAgent(llm=llm, tools=tools)
executor = ExecutorAgent(llm=llm, tools=tools)
orchestrator = EDAOrchestrator(planner=planner, executor=executor, reasoner=reasoner)


def run_eda_pipeline(query: str, max_steps: int | None = None) -> dict:
    return orchestrator.run(query=query, max_steps=max_steps)


if __name__ == "__main__":
    demo_query = "Analyze customer churn dataset and identify key risk drivers."
    result = run_eda_pipeline(query=demo_query, max_steps=3)
    print(json.dumps(result, indent=2))