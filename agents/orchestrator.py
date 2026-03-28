import json
from typing import Any

from agents.executor import ExecutorAgent
from agents.planner import PlannerAgent
from agents.reason import ReasonAgent


class EDAOrchestrator:
    def __init__(
        self,
        planner: PlannerAgent,
        executor: ExecutorAgent,
        reasoner: ReasonAgent,
    ):
        self.planner = planner
        self.executor = executor
        self.reasoner = reasoner

    def run(self, query: str, max_steps: int | None = None) -> dict[str, Any]:
        plan = self.planner.generate_plan(query)

        if not isinstance(plan, list):
            raise ValueError("Planner output must be a JSON array of steps.")

        steps = plan[:max_steps] if max_steps is not None else plan
        execution_results: list[dict[str, Any]] = []

        context_parts = [f"Original analysis goal: {query}"]
        for step in steps:
            execution_context = "\n".join(context_parts)
            step_result = self.executor.execute_step(step=step, context=execution_context)
            execution_results.append(step_result)
            context_parts.append(
                f"Step {step_result.get('step', 'unknown')} result: {json.dumps(step_result)}"
            )

        reasoning_input = (
            "Use the following EDA artifacts to generate insights.\n\n"
            f"Goal:\n{query}\n\n"
            f"Plan:\n{json.dumps(plan)}\n\n"
            f"Execution Results:\n{json.dumps(execution_results)}"
        )
        insights = self.reasoner.generate_insights(reasoning_input)

        return {
            "query": query,
            "plan": plan,
            "execution_results": execution_results,
            "insights": insights,
        }
