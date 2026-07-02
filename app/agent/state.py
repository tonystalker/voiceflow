"""
app/agent/state.py
ConversationState — the single shared TypedDict that flows through every
LangGraph node for the duration of one caller turn.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict


class LatencyLog(TypedDict):
    stt_ms: Optional[float]
    llm_ms: Optional[float]
    tts_first_chunk_ms: Optional[float]
    total_ms: Optional[float]


class ConversationState(TypedDict):
    # ── Call identity ──────────────────────────────────────────────────────
    call_sid: str                        # Twilio CallSid
    turn_count: int                      # How many full turns completed

    # ── Transcript ────────────────────────────────────────────────────────
    transcript_partial: str             # Latest Deepgram interim result
    transcript_final: str               # Committed final transcript for this turn

    # ── Agent reasoning ───────────────────────────────────────────────────
    intent: Optional[str]               # "faq" | "account_query" | "out_of_scope"
    retrieved_context: List[Dict[str, Any]]   # RAG chunks [{text, score}]
    llm_response: Optional[str]         # Final text sent to TTS

    # ── Tool outputs ──────────────────────────────────────────────────────
    tool_result: Optional[str]          # Stringified result from a tool call

    # ── Control flags ─────────────────────────────────────────────────────
    escalate_flag: bool                 # True → trigger fallback/human handoff
    barge_in_detected: bool             # True → caller interrupted TTS

    # ── Observability ─────────────────────────────────────────────────────
    latency_log: LatencyLog
