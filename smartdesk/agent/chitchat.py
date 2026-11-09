"""Chitchat agent: greetings, thanks and polite redirection of out-of-scope requests. States no policy facts."""

from __future__ import annotations

from typing import Literal

from langchain_core.messages import AIMessage, SystemMessage
from langgraph.graph import END
from langgraph.types import Command

from .. import llm
from ..llm import LLMUnavailable
from . import prompts
from .state import SmartDeskState


def chitchat(state: SmartDeskState) -> Command[Literal["__end__"]]:
    try:
        reply = llm.invoke_safely(
            llm.chat(temperature=0.3),
            [SystemMessage(content=prompts.CHITCHAT_SYSTEM)] + state["messages"][-6:],
        ).content
    except LLMUnavailable:
        reply = ("Hi! I'm SmartDesk, NovaTech's IT & HR assistant. I can answer policy and IT questions, "
                 "raise support tickets and check ticket status.")
    return Command(goto=END, update={"messages": [AIMessage(content=reply)]})
