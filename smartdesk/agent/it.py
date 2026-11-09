"""IT specialist: answers IT questions from the IT part of the knowledge base."""

from __future__ import annotations

from typing import Literal

from langgraph.types import Command

from .common import AgentDeps, answer_from_kb, graceful
from .state import SmartDeskState

GROUP = "IT"
PERSONA = "You are the IT Support specialist of SmartDesk. You help with accounts, VPN, Wi-Fi, devices, software and email."


def make_it_agent(deps: AgentDeps):
    @graceful
    def it_agent(state: SmartDeskState) -> Command[Literal["escalate", "__end__"]]:
        return answer_from_kb(state, deps, GROUP, PERSONA)

    return it_agent
