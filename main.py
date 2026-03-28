from agents.base import BaseAgent
from agents.planner import PlannerAgent
from agents.reason import ReasonAgent

llm = BaseAgent.create_default_llm()

planner = PlannerAgent(llm=llm)
reasoner = ReasonAgent(llm=llm)