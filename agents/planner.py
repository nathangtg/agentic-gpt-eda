from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from collections.abc import Sequence
from typing import Any

from agents.base import BaseAgent

class PlannerAgent(BaseAgent):
    def __init__(self, llm: ChatOpenAI | None = None, tools: Sequence[Any] | None = None):
        super().__init__(llm=llm, tools=tools)

        self.prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are a senior data scientist specializing in Exploratory Data Analysis (EDA). "
                "Your task is to generate a structured, step-by-step EDA plan that maximizes actionable insights.\n\n"

                "Requirements:\n"
                "- Output MUST be a JSON array of steps\n"
                "- Each step must contain: step (int), title (string), objective (string), methods (list of strings), expected_output (string)\n"
                "- Steps must be logically ordered and non-overlapping\n"
                "- Be specific and technical (avoid vague steps like 'analyze data')\n\n"

                "EDA Coverage Requirements:\n"
                "1. Data Understanding (schema, data types, missingness)\n"
                "2. Data Cleaning (null handling, outliers, duplicates)\n"
                "3. Univariate Analysis (distributions, skewness)\n"
                "4. Bivariate/Multivariate Analysis (correlation, relationships)\n"
                "5. Feature Engineering (transformations, encoding, scaling)\n"
                "6. Statistical Analysis (hypothesis testing, significance)\n"
                "7. Visualization (Matplotlib, Seaborn, Plotly where appropriate)\n"
                "8. Insight Extraction (patterns, anomalies, business implications)\n\n"

                "Constraints:\n"
                "- Prefer concrete techniques (e.g., 'Pearson correlation', 'IQR outlier detection')\n"
                "- Mention specific plots when relevant (e.g., histogram, boxplot, heatmap)\n"
                "- Ensure steps are actionable and executable by an engineer\n"
                "- Do NOT include explanations outside JSON\n\n"

                "Goal: Produce a high-quality EDA plan that a data scientist can directly execute."
            ),
            ("user", "{query}")
        ])

    def generate_plan(self, query: str):
        formatted_prompt = self.prompt.format_messages(query=query)
        response = self.llm.invoke(formatted_prompt)
        plan = self.parse_json_response(response.content, expected_type=list)
        return plan
    
    