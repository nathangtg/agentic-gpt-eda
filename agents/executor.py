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

		tool_calls = getattr(response, "tool_calls", []) or []
		if tool_calls:
			tool_results = [self.execute_tool_call(tool_call) for tool_call in tool_calls]

			success_count = sum(1 for res in tool_results if res.get("ok"))
			failure_count = len(tool_results) - success_count

			# Try to let the model summarize executed tool outputs into the required schema.
			summary_prompt = (
				"You are an EDA execution summarizer. Using executed tool results, return only a valid JSON object.\n"
				"Required schema keys: step, status, actions_taken, observations, artifacts, next_recommended_action.\n"
				"Status must be one of: completed, needs_input, failed.\n"
				f"Step JSON:\n{json.dumps(step)}\n\n"
				f"Tool results:\n{json.dumps(tool_results)}"
			)

			try:
				summary_response = self.llm.invoke(summary_prompt)
				if getattr(summary_response, "content", ""):
					return self.parse_json_response(summary_response.content, expected_type=dict)
			except Exception:  # noqa: BLE001
				pass

			status = "completed" if success_count > 0 else "failed"
			actions_taken = [f"Executed tool '{res.get('name', 'unknown')}'" for res in tool_results]
			observations = []
			artifacts = []

			for res in tool_results:
				if res.get("ok"):
					observations.append(f"Tool '{res.get('name')}' completed successfully.")
					artifacts.append(
						json.dumps(
							{
								"tool": res.get("name"),
								"args": res.get("args", {}),
								"output": str(res.get("output", ""))[:5000],
							}
						)
					)
				else:
					observations.append(
						f"Tool '{res.get('name')}' failed: {res.get('error', 'Unknown error')}"
					)
					artifacts.append(
						json.dumps(
							{
								"tool": res.get("name"),
								"args": res.get("args", {}),
								"error": res.get("error", "Unknown error"),
							}
						)
					)

			next_action = (
				"Proceed to the next step using generated artifacts."
				if failure_count == 0
				else "Review failed tool calls and update step methods or tool parameters."
			)

			return {
				"step": step.get("step"),
				"status": status,
				"actions_taken": actions_taken,
				"observations": observations,
				"artifacts": artifacts,
				"next_recommended_action": next_action,
			}

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
