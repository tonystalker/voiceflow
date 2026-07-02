"""
app/agent/graph.py

LangGraph state machine for VoiceFlow.

Flow:
  START
    │
    ▼
  intent_classification_node
    │
    ├─(faq)──────────────────► rag_retrieval_node
    │                                │
    │                  ┌────────────►│ (low_confidence)──► fallback_node
    │                  │             │
    │                  │             ▼
    │                  │       response_generation_node
    │                  │             │
    ├─(account_query)──┤             ▼
    ├─(dispute_query)──┤           END
    │                  │
    │          tool_calling_node
    │                  │
    │                  ▼
    │          response_generation_node ──► END
    │
    ├─(escalate)────► fallback_node ──► END
    └─(out_of_scope)─► fallback_node ──► END
"""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from app.agent.state import ConversationState
from app.agent.nodes import (
    fallback_node,
    intent_classification_node,
    rag_retrieval_node,
    response_generation_node,
    tool_calling_node,
)


def _route_intent(state: ConversationState) -> str:
    intent = state.get("intent", "out_of_scope")
    if intent in ("account_query", "dispute_query"):
        return "tool"
    if intent in ("escalate", "out_of_scope"):
        return "fallback"
    return "rag"  # faq or anything else


def _route_rag(state: ConversationState) -> str:
    if state.get("escalate_flag"):
        return "fallback"
    return "generate"


def build_graph() -> StateGraph:
    builder = StateGraph(ConversationState)

    # ── Add nodes ─────────────────────────────────────────────────────────
    builder.add_node("classify", intent_classification_node)  # renamed: 'intent' conflicts with state key
    builder.add_node("rag", rag_retrieval_node)
    builder.add_node("tool", tool_calling_node)
    builder.add_node("generate", response_generation_node)
    builder.add_node("fallback", fallback_node)

    # ── Entry point ───────────────────────────────────────────────────────
    builder.set_entry_point("classify")

    # ── Edges ─────────────────────────────────────────────────────────────
    builder.add_conditional_edges(
        "classify",
        _route_intent,
        {"rag": "rag", "tool": "tool", "fallback": "fallback"},
    )
    builder.add_conditional_edges(
        "rag",
        _route_rag,
        {"fallback": "fallback", "generate": "generate"},
    )
    builder.add_edge("tool", "generate")
    builder.add_edge("generate", END)
    builder.add_edge("fallback", END)

    return builder.compile()


# Module-level compiled graph (singleton)
agent_graph = build_graph()
