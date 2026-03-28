from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from collections.abc import Sequence
from typing import Any

from agents.base import BaseAgent

class ReasonAgent(BaseAgent):
    def __init__(self, llm: ChatOpenAI | None = None, tools: Sequence[Any] | None = None):
        super().__init__(llm=llm, tools=tools)

        self.prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are a senior data scientist specializing in Exploratory Data Analysis (EDA). "
                "Your task is to reason through the results of EDA steps and generate actionable insights.\n\n"

                "Requirements:\n"
                "- Output MUST be a JSON array of insights\n"
                "- Each insight must contain: insight (string), confidence (float 0-1), recommended_action (string)\n"
                "- Insights must be specific, actionable, and directly tied to EDA results\n"
                "- Avoid vague insights like 'the data has issues' - be specific about what the issues are and how to address them\n\n"

                "Reasoning Coverage Requirements:\n"
                "1. Data Quality Issues (missingness patterns, outliers, duplicates)\n"
                "2. Distributional Insights (skewness, modality, variance)\n"
                "3. Relationships (correlations, interactions)\n"
                "4. Feature Importance (key drivers of target variable)\n"
                "5. Business Implications (what do the findings mean for the business?)\n\n"

                "Constraints:\n"
                "- Provide confidence scores for each insight to indicate how strongly the data supports it\n"
                "- Recommend specific actions for each insight (e.g., 'impute missing values with median', 'consider log transformation')\n"
                "- Ensure insights are directly actionable by an engineer or data scientist\n"
                "- Do NOT include explanations outside JSON\n\n"

                "Goal: Produce high-quality insights that can guide the next steps in data analysis or modeling."
            ),
            ("user", "{query}")
        ])

    def generate_insights(self, query: str):
        formatted_prompt = self.prompt.format_messages(query=query)
        response = self.llm.invoke(formatted_prompt)
        insights = self.parse_json_response(response.content, expected_type=list)
        return insights