from dataclasses import dataclass

from agents.base import BaseAgent


@dataclass
class FakeResponse:
    content: str


class FakeLLM:
    def __init__(self, outputs: list[str]):
        self.outputs = outputs
        self.calls = 0

    def invoke(self, _prompt):
        idx = min(self.calls, len(self.outputs) - 1)
        self.calls += 1
        return FakeResponse(content=self.outputs[idx])


def test_parse_json_response_valid_array():
    llm = FakeLLM(outputs=["[]"])
    agent = BaseAgent(llm=llm)

    parsed = agent.parse_json_response('[{"step": 1}]', expected_type=list)

    assert isinstance(parsed, list)
    assert parsed[0]["step"] == 1


def test_parse_json_response_extracts_fenced_json():
    llm = FakeLLM(outputs=["[]"])
    agent = BaseAgent(llm=llm)

    raw = "Some text before\n```json\n[{\"insight\": \"x\"}]\n```\nSome text after"
    parsed = agent.parse_json_response(raw, expected_type=list)

    assert isinstance(parsed, list)
    assert parsed[0]["insight"] == "x"


def test_parse_json_response_repairs_invalid_json_with_llm():
    llm = FakeLLM(outputs=['[{"insight": "fixed"}]'])
    agent = BaseAgent(llm=llm)

    parsed = agent.parse_json_response('not valid json', expected_type=list)

    assert isinstance(parsed, list)
    assert parsed[0]["insight"] == "fixed"
    assert llm.calls == 1
