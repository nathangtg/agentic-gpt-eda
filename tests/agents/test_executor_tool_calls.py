from dataclasses import dataclass

from agents.executor import ExecutorAgent


@dataclass
class FakeResponse:
    content: str = ""
    tool_calls: list | None = None


class FakeLLM:
    def __init__(self, responses):
        self.responses = responses
        self.calls = 0

    def bind_tools(self, _tools):
        return self

    def invoke(self, _prompt):
        idx = min(self.calls, len(self.responses) - 1)
        self.calls += 1
        return self.responses[idx]


class SuccessTool:
    name = "test_success_tool"

    def invoke(self, args):
        return {"echo": args}


class FailingTool:
    name = "test_failing_tool"

    def invoke(self, _args):
        raise RuntimeError("boom")


def test_executor_executes_tool_calls_and_returns_summary_json():
    llm = FakeLLM(
        responses=[
            FakeResponse(content="", tool_calls=[{"name": "test_success_tool", "args": {"x": 1}}]),
            FakeResponse(
                content=(
                    '{"step": 1, "status": "completed", "actions_taken": ["done"], '
                    '"observations": ["ok"], "artifacts": ["artifact"], '
                    '"next_recommended_action": "continue"}'
                )
            ),
        ]
    )

    agent = ExecutorAgent(llm=llm, tools=[SuccessTool()])
    result = agent.execute_step(step={"step": 1, "title": "T", "objective": "O", "methods": []})

    assert result["status"] == "completed"
    assert result["step"] == 1


def test_executor_tool_failure_falls_back_to_failed_or_needs_input():
    llm = FakeLLM(
        responses=[
            FakeResponse(content="", tool_calls=[{"name": "test_failing_tool", "args": {}}]),
            FakeResponse(content=""),
        ]
    )

    agent = ExecutorAgent(llm=llm, tools=[FailingTool()])
    result = agent.execute_step(step={"step": 2, "title": "T2", "objective": "O2", "methods": []})

    assert result["step"] == 2
    assert result["status"] in {"failed", "needs_input", "completed"}
    assert isinstance(result.get("observations", []), list)
    assert isinstance(result.get("artifacts", []), list)
