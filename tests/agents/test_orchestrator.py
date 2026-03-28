from agents.orchestrator import EDAOrchestrator


class FakePlanner:
    def generate_plan(self, _query):
        return [
            {"step": 1, "title": "A", "objective": "oa", "methods": ["m1"]},
            {"step": 2, "title": "B", "objective": "ob", "methods": ["m2"]},
        ]


class FakeExecutor:
    def __init__(self):
        self.calls = 0

    def execute_step(self, step, context=""):
        self.calls += 1
        return {
            "step": step.get("step"),
            "status": "completed",
            "actions_taken": [f"ran {step.get('title')}"] ,
            "observations": [context[:50]],
            "artifacts": [],
            "next_recommended_action": "next",
        }


class FakeReasoner:
    def generate_insights(self, _query):
        return [
            {
                "insight": "top driver found",
                "confidence": 0.9,
                "recommended_action": "prioritize feature",
            }
        ]


def test_orchestrator_runs_end_to_end_with_mocks():
    planner = FakePlanner()
    executor = FakeExecutor()
    reasoner = FakeReasoner()

    orchestrator = EDAOrchestrator(planner=planner, executor=executor, reasoner=reasoner)
    result = orchestrator.run(query="test query", max_steps=1)

    assert "plan" in result
    assert "execution_results" in result
    assert "insights" in result
    assert len(result["execution_results"]) == 1
    assert result["insights"][0]["confidence"] == 0.9
