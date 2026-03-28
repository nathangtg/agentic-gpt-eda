from agents.base import BaseAgent
from agents.planner import PlannerAgent
from agents.reason import ReasonAgent
from tools.common_tools import get_column_stats, get_dataset_info, load_dataset
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
        get_dataset_info,
        get_column_stats,
        summarize_dataset,
        generate_correlation_matrix,
        get_top_n_unique_values,
        generate_histogram,
        generate_bar_chart,
    ]


tools = load_tools()
planner = PlannerAgent(llm=llm, tools=tools)
reasoner = ReasonAgent(llm=llm, tools=tools)