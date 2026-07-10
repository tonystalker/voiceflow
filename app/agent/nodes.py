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
_SYSTEM_PROMPT = """You are Tony, a personal voice assistant. You help with questions,
account lookups, and can order food/groceries or book a table via Swiggy when asked.
Introduce yourself as Tony if asked who you are. Keep responses concise and
conversational — you're being read aloud by a TTS engine, not displayed as text.

STRICT RULES:
1. Keep every response to 2-3 SHORT sentences maximum. You are speaking aloud.
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
    
    if state.get("pending_action"):
        if any(w in lower for w in ["yes", "yeah", "yep", "sure", "go ahead", "confirm", "place it", "do it"]):
            return {"intent": "confirm_yes", "confirmed": True}
        else:
            return {"intent": "confirm_no", "confirmed": False}
    
    # Check for "account" followed closely by 4 digits (e.g. "account is 1001", "account one zero zero one" converted to digits)
    has_acc = re.search(r"(?:acc|account).*?(\d{4})", lower)
    has_dis = re.search(r"(?:dis|dispute).*?(\d{4})", lower)
    
    food_keywords = ["order", "food", "biryani", "swiggy", "book", "table", "groceries", "chicken", "roll", "pizza", "burger", "buy"]
    has_swiggy = any(w in lower for w in food_keywords) and not has_acc and not has_dis
    
    if has_swiggy:
        intent = "action_intent"
    elif has_acc:
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
            f"Labels: action_intent, faq, account_query, dispute_query, escalate, out_of_scope\n"
            f"Note: action_intent includes ordering food, groceries, booking tables, or Swiggy.\n"
            f"Query: {transcript}\n"
            f"Label:"
        )
        result = clf_llm.invoke([HumanMessage(content=prompt)])
        intent = result.content.strip().lower().split()[0]
        if intent not in {"action_intent", "faq", "account_query", "dispute_query", "escalate", "out_of_scope"}:
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


# ── Node 3.2: Confirmation Node ───────────────────────────────────────────
def confirmation_node(state: ConversationState) -> Dict[str, Any]:
    """Request confirmation from the user before taking a high-stakes action."""
    transcript = state["transcript_final"]
    logger.info(f"[CONFIRM] Requesting confirmation for action")
    
    # We ask for confirmation and set pending_action so the next turn expects a yes/no
    msg = f"You want to use Swiggy for this request. Shall I go ahead and proceed?"
    
    return {
        "llm_response": msg,
        "pending_action": {"transcript": transcript},
    }


# ── Node 3.5: Swiggy Action Node ──────────────────────────────────────────
async def swiggy_tool_node(state: ConversationState) -> Dict[str, Any]:
    intent = state.get("intent")
    transcript = state["transcript_final"]
    
    if intent == "confirm_yes":
        instruction = "The user has confirmed the order. Check the active cart and place the order now."
        next_pending = None
    else:
        instruction = f"User Request: {transcript}\n\nIMPORTANT: Do NOT place the order yet. Search for the items, add them to the cart, and then read back the items and the total price. Ask the user if they want to place the order.\nCRITICAL: When calling tools (e.g. search_restaurants), ensure numeric parameters like 'offset' are passed as raw JSON integers (e.g. 0), NOT strings (e.g. \"0\")."
        next_pending = {"transcript": transcript}
        
    logger.info(f"[SWIGGY] Handling confirmed action: {intent}")
    
    from app.mcp.swiggy_client import get_swiggy_tools
    from langgraph.prebuilt import create_react_agent
    
    # In a full system, you would cache this agent/tools or instantiate once.
    # For now, we open the connection per action.
    try:
        async with get_swiggy_tools("food") as tools:
            # Use a lighter model for tools to avoid Groq's 12k TPM limit on 70B models
            from langchain_groq import ChatGroq
            from app.config import settings
            tool_llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=settings.groq_api_key, temperature=0.0)
            
            essential_tools = {"search_restaurants", "search_menu", "get_food_cart", "update_food_cart", "place_food_order", "confirm_order"}
            filtered_tools = [t for t in tools if t.name in essential_tools]
            
            agent = create_react_agent(tool_llm, tools=filtered_tools)
            
            # execute
            res = await agent.ainvoke({"messages": [("user", instruction)]})
            result_text = res["messages"][-1].content
            
            return {"tool_result": result_text, "pending_action": next_pending}
    except Exception as e:
        import traceback
        logger.error(f"[SWIGGY] Traceback:\n{traceback.format_exc()}")
            
        logger.error(f"[SWIGGY] Error: {e}")
        return {"tool_result": f"Sorry, there was an error connecting to Swiggy: {str(e)}", "pending_action": None}



# ── Node 4: Response Generation ───────────────────────────────────────────
def response_generation_node(state: ConversationState) -> Dict[str, Any]:
    """Generate the LLM response, streaming tokens, grounded in retrieved context."""
    t0 = time.monotonic()

    # Build context block
    context_block = ""
    if state.get("retrieved_context"):
        # Include all chunks — let the LLM decide if context is relevant.
        # Filtering by low_confidence here causes unnecessary fallbacks on
        # minor transcription errors (e.g. 'backing' vs 'banking').
        snippets = [c["text"] for c in state["retrieved_context"] if c.get("text")]
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
