"""HR specialist: answers HR, onboarding and payroll questions from the HR part of the knowledge base."""

from __future__ import annotations

from typing import Literal

from langgraph.types import Command

from .common import AgentDeps, answer_from_kb, graceful
from .state import SmartDeskState

GROUP = "HR"
PERSONA = "You are the HR & People specialist of SmartDesk. You help with leave, policies, onboarding, benefits, payroll and compensation."


def make_hr_agent(deps: AgentDeps):
    @graceful
    def hr_agent(state: SmartDeskState) -> Command[Literal["escalate", "__end__"]]:
        return answer_from_kb(state, deps, GROUP, PERSONA)

    return hr_agent
