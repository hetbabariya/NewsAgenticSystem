from __future__ import annotations

from typing import Annotated, Any, Dict, List, Literal, Optional, Sequence, TypedDict, Union

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """
    Production-grade state for the News Agentic System.
    Tracks messages, context, and routing information.
    """
    # Standard LangGraph messages list with additive logic
    messages: Annotated[Sequence[BaseMessage], add_messages]
    
    # Context dictionary for metadata, trace IDs, and trigger information
    context: Dict[str, Any]
    
    # Next node to execute (used for conditional routing)
    next: Optional[str]
    
    # Error tracking for resilience
    errors: List[str]
    
    # Shared data for the current run (e.g., article IDs, summary counts)
    shared_data: Dict[str, Any]
