from __future__ import annotations

import logging
from typing import Any, Dict, List, Literal, Optional, Sequence, Union

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from app.core.settings import settings
from app.orchestrator.state import AgentState
# LOCAL_TOOLS will be imported later or we can define the graph logic here

log = logging.getLogger("orchestrator.orchestrator")

# --- Coordinator Agent (The "Router") ---

def get_coordinator_model():
    """Returns the primary model for coordination with key rotation support."""
    # For simplicity in this layer, we use the settings-defined model.
    # Key rotation logic is typically handled at the graph.py level or via a wrapper.
    return ChatGroq(
        api_key=settings.groq_keys[0],
        model_name=settings.model_coordinator,
        temperature=0
    ).bind_tools(LOCAL_TOOLS)

async def coordinator_agent(state: AgentState) -> Dict[str, Any]:
    """
    The main decision-maker. Analyzes intent and routes to specialized agents or tools.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are the Lead Coordinator for the News Agentic System.
        Your goal is to fulfill user requests by orchestrating specialized sub-agents and tools.

        CURRENT CONTEXT:
        - Trigger: {trigger}
        - History: {history}

        AVAILABLE AGENTS:
        1. collector: Fetches raw news from Reddit, Tavily, Twitter, GitHub.
        2. filter: Scores and filters articles based on user preferences.
        3. summarizer: Generates concise summaries of relevant news.
        4. publisher: Handles newspaper generation and delivery.
        5. memory: Manages user preferences and long-term facts.

        GUIDELINES:
        - For news ingest (cron), follow the sequence: collector -> filter -> summarizer.
        - For conversational queries, use the most relevant tool or agent.
        - If the user provides a fact about themselves, route to 'memory'.
        - Always return a clear 'next' step.
        """),
        MessagesPlaceholder(variable_name="messages"),
    ])

    chain = prompt | get_coordinator_model()

    # Prepare context for the prompt
    trigger = state["context"].get("trigger", "manual")
    history = state["context"].get("history", "")

    response = await chain.ainvoke({
        "messages": state["messages"],
        "trigger": trigger,
        "history": history
    })

    # Determine the next node based on tool calls or content
    next_node = END
    if response.tool_calls:
        next_node = "tools"

    return {
        "messages": [response],
        "next": next_node
    }

# --- Specialized Agent Logic (Production Grade) ---

async def subagent_node(state: AgentState, name: str, system_prompt: str, model_name: str, provider: str = "groq") -> Dict[str, Any]:
    """Generic node for specialized agents with focused prompts and models."""
    log.info(f"Executing sub-agent: {name}")

    if provider == "groq":
        llm = ChatGroq(api_key=settings.groq_keys[0], model_name=model_name)
    else:
        llm = ChatOpenAI(
            api_key=settings.openrouter_keys[0],
            base_url="https://openrouter.ai/api/v1",
            model_name=model_name
        )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
    ])

    chain = prompt | llm
    response = await chain.ainvoke({"messages": state["messages"]})

    return {
        "messages": [response],
        "next": "coordinator" # Always return to coordinator for next steps
    }

# Specific Sub-Agent Definitions (Wrappers for the generic node)

async def collector_agent(state: AgentState) -> Dict[str, Any]:
    system = "You are the News Collector. Fetch news from specified sources based on preferences."
    return await subagent_node(state, "collector", system, settings.model_collector)

async def filter_agent(state: AgentState) -> Dict[str, Any]:
    system = "You are the News Filter. Score articles for relevance and urgency."
    return await subagent_node(state, "filter", system, settings.model_filter)

async def summarizer_agent(state: AgentState) -> Dict[str, Any]:
    system = "You are the News Summarizer. Create factual, high-quality summaries."
    return await subagent_node(state, "summarizer", system, settings.model_summarizer, provider="openrouter")

async def publisher_agent(state: AgentState) -> Dict[str, Any]:
    system = "You are the News Publisher. Generate PDFs and deliver newspapers."
    return await subagent_node(state, "publisher", system, settings.model_publisher, provider="openrouter")

async def memory_agent(state: AgentState) -> Dict[str, Any]:
    system = "You are the Memory Agent. Update user preferences and store long-term facts."
    return await subagent_node(state, "memory", system, settings.model_memory)


# --- LangGraph Definition (Production Grade) ---

def build_production_graph(tools: list):
    """Constructs the production-grade LangGraph for news orchestration."""
    workflow = StateGraph(AgentState)

    # Add Nodes
    workflow.add_node("coordinator", coordinator_agent)
    workflow.add_node("collector", collector_agent)
    workflow.add_node("filter", filter_agent)
    workflow.add_node("summarizer", summarizer_agent)
    workflow.add_node("publisher", publisher_agent)
    workflow.add_node("memory", memory_agent)
    workflow.add_node("tools", ToolNode(tools))

    # Add Edges
    workflow.set_entry_point("coordinator")

    # Routing logic from coordinator
    def route_coordinator(state: AgentState) -> Literal["tools", "collector", "filter", "summarizer", "publisher", "memory", END]:
        next_node = state.get("next")
        if next_node == "tools":
            return "tools"

        # Check messages for specific intent if 'next' isn't set by tool calls
        last_message = state["messages"][-1]
        if isinstance(last_message, AIMessage) and not last_message.tool_calls:
            content = last_message.content.lower()
            if "route to collector" in content: return "collector"
            if "route to filter" in content: return "filter"
            if "route to summarizer" in content: return "summarizer"
            if "route to publisher" in content: return "publisher"
            if "route to memory" in content: return "memory"

        return END

    workflow.add_conditional_edges("coordinator", route_coordinator)

    # Tools always return to coordinator
    workflow.add_edge("tools", "coordinator")

    # Sub-agents always return to coordinator
    workflow.add_edge("collector", "coordinator")
    workflow.add_edge("filter", "coordinator")
    workflow.add_edge("summarizer", "coordinator")
    workflow.add_edge("publisher", "coordinator")
    workflow.add_edge("memory", "coordinator")

    return workflow.compile()
