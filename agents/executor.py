import json
from collections.abc import Sequence
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from agents.base import BaseAgent


class ExecutorAgent(BaseAgent):
	def __init__(self, llm: ChatOpenAI | None = None, tools: Sequence[Any] | None = None):
		super().__init__(llm=llm, tools=tools)

		self.prompt = ChatPromptTemplate.from_messages([
			(
				"system",
				"You are an EDA execution agent. Execute or operationalize one EDA step at a time. "
				"Use available tools when relevant and return a strict JSON object only.\n\n"
				"Required JSON schema:\n"
				"{{\n"
				"  \"step\": int,\n"
				"  \"status\": \"completed\" | \"needs_input\" | \"failed\",\n"
				"  \"actions_taken\": [string],\n"
				"  \"observations\": [string],\n"
				"  \"artifacts\": [string],\n"
				"  \"next_recommended_action\": string\n"
				"}}\n\n"
				"Constraints:\n"
				"- Tie outputs to the requested step objective and methods\n"
				"- Keep actions concrete and reproducible\n"
				"- If information is missing, set status='needs_input' and explain what is needed\n"
				"- Do not include any text outside JSON"
			),
			(
				"user",
				"Execution context:\n{context}\n\n"
				"Step to execute (JSON):\n{step_json}"
			),
		])

	def execute_step(self, step: dict[str, Any], context: str = "") -> dict[str, Any]:
		formatted_prompt = self.prompt.format_messages(
			context=context or "No prior execution context provided.",
			step_json=json.dumps(step)
		)
		response = self.llm.invoke(formatted_prompt)

		if response.content:
			return self.parse_json_response(response.content, expected_type=dict)

		# Some tool-enabled calls may return tool calls with no content.
		return {
			"step": step.get("step"),
			"status": "needs_input",
			"actions_taken": [],
			"observations": ["Model returned tool calls without final JSON output."],
			"artifacts": [json.dumps(getattr(response, "tool_calls", []))],
			"next_recommended_action": "Execute the tool calls and rerun execute_step with updated context.",
		}

	def run(self, input: str) -> str:
		# Compatibility wrapper for BaseAgent interface.
		return json.dumps(
			self.execute_step(step={"step": 0, "title": "ad-hoc", "objective": input, "methods": []}),
			indent=2,
		)
