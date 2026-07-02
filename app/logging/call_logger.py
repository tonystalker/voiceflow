"""
app/logging/call_logger.py

Structured per-turn call logger.
Appends JSON lines to logs/calls.jsonl for the Streamlit dashboard.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from loguru import logger as _loguru

from app.config import settings

_LOG_PATH = Path(settings.call_log_path)


class CallLogger:
    def __init__(self) -> None:
        _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    def log_turn(
        self,
        *,
        call_sid: str,
        turn_count: int,
        transcript: str,
        intent: Optional[str],
        llm_response: Optional[str],
        latency_log: Dict[str, Any],
        escalated: bool,
    ) -> None:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "call_sid": call_sid,
            "turn_count": turn_count,
            "transcript": transcript,
            "intent": intent,
            "response": llm_response,
            "latency": latency_log,
            "escalated": escalated,
        }
        with open(_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        _loguru.debug(f"[LOG] call={call_sid} turn={turn_count} intent={intent}")
