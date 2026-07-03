"""
app/agent/nodes.py

LangGraph node functions for the VoiceFlow voice agent.
Each function receives ConversationState and returns a partial update dict.

Nodes:
  turn_detection_node         — (handled in media_stream.py; stub here for graph clarity)
  intent_classification_node  — classify caller utterance
  rag_retrieval_node          — fetch relevant FAQ chunks
  tool_calling_node           — account/dispute lookup
  response_generation_node    — stream LLM answer
  fallback_node               — graceful out-of-scope response
"""

from __future__ import annotations

import re
import time
from typing import Any, Dict

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger

from app.agent.state import ConversationState
from app.config import settings
from app.rag.retriever import RAGRetriever
from app.tools.account_lookup import get_account_status, get_dispute_status

# ── Shared singletons (initialised once at import) ─────────────────────────
_retriever = RAGRetriever()

_llm = ChatGroq(
    model=settings.groq_model,
    api_key=settings.groq_api_key,
    temperature=0.3,
    max_tokens=256,          # Keep responses short — it's voice
    streaming=True,
)

# ── System prompt ──────────────────────────────────────────────────────────
_SYSTEM_PROMPT = """You are Aria, a friendly AI banking assistant for VoiceBank speaking over the phone.

STRICT RULES:
1. Keep every response to 2-3 SHORT sentences maximum. You are on a phone call.
2. Use natural spoken language. No bullet points, no markdown, no lists.
3. If "Relevant information" is provided below, use it DIRECTLY to answer — do not make up alternative steps.
4. If no relevant information is provided or you are unsure, say: "Let me connect you with a specialist who can help with that."
5. Never ask for account IDs unless the customer's question is specifically about their account balance or transactions.
6. Never invent policy details, fees, or procedures not given to you.
"""


# ── Node 1: Intent Classification ─────────────────────────────────────────
def intent_classification_node(state: ConversationState) -> Dict[str, Any]:
    """Classify the caller's utterance into one of three intents."""
    transcript = state["transcript_final"]

    # Fast rule-based pre-check to save LLM call
    lower = transcript.lower()
    
    # Check for "account" followed closely by 4 digits (e.g. "account is 1001", "account one zero zero one" converted to digits)
    has_acc = re.search(r"(?:acc|account).*?(\d{4})", lower)
    has_dis = re.search(r"(?:dis|dispute).*?(\d{4})", lower)
    
    if has_acc:
        intent = "account_query"
    elif has_dis:
        intent = "dispute_query"
    elif any(w in lower for w in ["human", "agent", "representative", "person", "speak to someone"]):
        intent = "escalate"
    else:
        # LLM classification for ambiguous cases
        clf_llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            api_key=settings.groq_api_key,
            temperature=0,
            max_tokens=10,
        )
        prompt = (
            f"Classify this banking support query into exactly one label.\n"
            f"Labels: faq, account_query, dispute_query, escalate, out_of_scope\n"
            f"Query: {transcript}\n"
            f"Label:"
        )
        result = clf_llm.invoke([HumanMessage(content=prompt)])
        intent = result.content.strip().lower().split()[0]
        if intent not in {"faq", "account_query", "dispute_query", "escalate", "out_of_scope"}:
            intent = "faq"  # default

    logger.info(f"[INTENT] '{transcript}' → {intent}")
    return {"intent": intent}


# ── Node 2: RAG Retrieval ─────────────────────────────────────────────────
def rag_retrieval_node(state: ConversationState) -> Dict[str, Any]:
    """Fetch relevant FAQ chunks from Qdrant."""
    chunks = _retriever.search(state["transcript_final"], top_k=3)
    escalate = bool(chunks and chunks[0].get("low_confidence"))
    return {"retrieved_context": chunks, "escalate_flag": escalate}


# ── Node 3: Tool Calling ──────────────────────────────────────────────────
def tool_calling_node(state: ConversationState) -> Dict[str, Any]:
    """Call the appropriate mock tool based on intent and extracted IDs."""
    transcript = state["transcript_final"]

    # Extract account ID
    acc_match = re.search(r"(?:acc|account).*?(\d{4})", transcript, re.IGNORECASE)
    dis_match = re.search(r"(?:dis|dispute).*?(\d{4})", transcript, re.IGNORECASE)

    if acc_match:
        account_id = f"ACC-{acc_match.group(1)}"
        result = get_account_status(account_id)
    elif dis_match:
        dispute_id = f"DIS-{dis_match.group(1)}"
        result = get_dispute_status(dispute_id)
    else:
        result = (
            "I couldn't identify an account or dispute number in your request. "
            "Could you please say your account ID, for example: A C C dash 1001?"
        )

    logger.info(f"[TOOL] result: {result[:80]}…")
    return {"tool_result": result}


# ── Node 4: Response Generation ───────────────────────────────────────────
def response_generation_node(state: ConversationState) -> Dict[str, Any]:
    """Generate the LLM response, streaming tokens, grounded in retrieved context."""
    t0 = time.monotonic()

    # Build context block
    context_block = ""
    if state.get("retrieved_context"):
        snippets = [c["text"] for c in state["retrieved_context"] if not c.get("low_confidence")]
        if snippets:
            context_block = "\n\nRelevant information:\n" + "\n".join(f"- {s}" for s in snippets)

    tool_result = state.get("tool_result")
    if tool_result:
        context_block += f"\n\nAccount/Dispute lookup result:\n{tool_result}"

    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=f"Customer said: {state['transcript_final']}{context_block}"),
    ]

    # Stream and collect full response
    response_text = ""
    for chunk in _llm.stream(messages):
        if chunk.content:
            response_text += chunk.content

    llm_ms = (time.monotonic() - t0) * 1000
    logger.info(f"[LLM] response in {llm_ms:.0f} ms: '{response_text[:80]}…'")

    latency_log = dict(state.get("latency_log", {}))
    latency_log["llm_ms"] = llm_ms

    return {
        "llm_response": response_text,
        "latency_log": latency_log,
    }


# ── Node 5: Fallback ──────────────────────────────────────────────────────
def fallback_node(state: ConversationState) -> Dict[str, Any]:
    """Return a graceful handoff line for out-of-scope or low-confidence queries."""
    intent = state.get("intent", "out_of_scope")

    if intent == "escalate":
        msg = (
            "Of course — I'll connect you to one of our human agents right away. "
            "Please hold for just a moment."
        )
    else:
        msg = (
            "I'm sorry, I don't have enough information to answer that confidently. "
            "Let me connect you with a specialist who can help. Please hold."
        )

    logger.info(f"[FALLBACK] escalating — intent={intent}")
    return {"llm_response": msg, "escalate_flag": True}
