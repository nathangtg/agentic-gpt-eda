import os
import json
import re
from collections.abc import Sequence
from typing import Any

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

class BaseAgent:
    def __init__(self, llm: ChatOpenAI | None = None, tools: Sequence[Any] | None = None):
        self.tools = list(tools) if tools else []
        self.tool_map = {
            getattr(tool, "name", ""): tool
            for tool in self.tools
            if getattr(tool, "name", "")
        }
        base_llm = llm or self.create_default_llm()
        self.llm = base_llm.bind_tools(self.tools) if self.tools else base_llm

    @classmethod
    def create_default_llm(cls) -> ChatOpenAI:
        load_dotenv()
        return ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL")
        )

    def _extract_json_candidate(self, text: str) -> str:
        cleaned = text.strip()
        if not cleaned:
            return cleaned

        # Prefer fenced JSON content first when present.
        fenced = re.findall(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned, flags=re.IGNORECASE)
        if fenced:
            return fenced[0].strip()

        # Fallback: pull from first opening bracket/brace to last closing counterpart.
        start_positions = [idx for idx in (cleaned.find("{"), cleaned.find("[")) if idx != -1]
        if not start_positions:
            return cleaned

        start = min(start_positions)
        end_obj = cleaned.rfind("}")
        end_arr = cleaned.rfind("]")
        end = max(end_obj, end_arr)
        if end == -1 or end < start:
            return cleaned

        return cleaned[start:end + 1].strip()

    def _repair_json_with_llm(self, raw_text: str, expected_kind: str) -> str:
        repair_prompt = (
            "Convert the content below into valid JSON only.\n"
            f"Expected top-level type: {expected_kind}.\n"
            "Do not include markdown, comments, or explanations.\n\n"
            f"Content:\n{raw_text}"
        )
        repaired = self.llm.invoke(repair_prompt)
        return repaired.content if hasattr(repaired, "content") else str(repaired)

    def parse_json_response(self, raw_text: str, expected_type: type[list] | type[dict]):
        candidate = self._extract_json_candidate(raw_text)
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            expected_kind = "array" if expected_type is list else "object"
            repaired_text = self._repair_json_with_llm(raw_text, expected_kind=expected_kind)
            parsed = json.loads(self._extract_json_candidate(repaired_text))

        if not isinstance(parsed, expected_type):
            expected_name = "array" if expected_type is list else "object"
            raise ValueError(f"Parsed JSON must be a top-level {expected_name}.")

        return parsed

    def execute_tool_call(self, tool_call: dict[str, Any]) -> dict[str, Any]:
        name = str(tool_call.get("name", ""))
        args = tool_call.get("args", {})

        if name not in self.tool_map:
            return {
                "name": name,
                "args": args,
                "ok": False,
                "error": f"Tool '{name}' is not registered.",
            }

        tool = self.tool_map[name]
        try:
            output = tool.invoke(args)
            if not isinstance(output, str):
                output = json.dumps(output, ensure_ascii=False)
            return {
                "name": name,
                "args": args,
                "ok": True,
                "output": output,
            }
        except Exception as exc:  # noqa: BLE001
            return {
                "name": name,
                "args": args,
                "ok": False,
                "error": str(exc),
            }

    def run(self, input: str) -> str:
        raise NotImplementedError("Subclasses must implement the run method.")