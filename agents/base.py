import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

class BaseAgent:
    def __init__(self, llm: ChatOpenAI | None = None):
        self.llm = llm or self.create_default_llm()

    @classmethod
    def create_default_llm(cls) -> ChatOpenAI:
        load_dotenv()
        return ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL")
        )

    def run(self, input: str) -> str:
        raise NotImplementedError("Subclasses must implement the run method.")