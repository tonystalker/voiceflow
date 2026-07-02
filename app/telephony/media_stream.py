"""
app/telephony/media_stream.py

FastAPI WebSocket endpoint that handles Twilio Media Streams.

Protocol:
  • Twilio → us: JSON frames with event types: connected, start, media, stop
  • us → Twilio: JSON frames with event type: media  (base64 μ-law audio)

Pipeline per call:
  audio bytes → DeepgramSTTClient → [on_partial] barge-in check
                                  → [on_final]   LangGraph agent
                                                        ↓
                                               ElevenLabsTTSClient
                                                        ↓
                                              Twilio WebSocket (out)
"""

from __future__ import annotations

import asyncio
import base64
import json
import time
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from loguru import logger

from app.agent.graph import agent_graph
from app.agent.state import ConversationState
from app.logging.call_logger import CallLogger
from app.stt.deepgram_client import DeepgramSTTClient
from app.tts.elevenlabs_client import ElevenLabsTTSClient

router = APIRouter()
_call_logger = CallLogger()


@router.websocket("/media-stream")
async def media_stream(ws: WebSocket) -> None:
    await ws.accept()
    logger.info("Twilio Media Stream WebSocket connected")

    # Per-call state
    call_sid: str = ""
    stream_sid: str = ""
    turn_count: int = 0
    tts_playing: bool = False
    greeting_sent: bool = False
    greeting_text: str = ""

    tts_client = ElevenLabsTTSClient()

    # ── Callbacks for STT ──────────────────────────────────────────────────
    def on_partial(text: str) -> None:
        nonlocal tts_playing
        if tts_playing:
            logger.info("[BARGE-IN] Caller interrupted — stopping TTS")
            tts_client.stop()
            tts_playing = False

    def on_final(text: str, stt_latency_ms: float) -> None:
        nonlocal turn_count
        turn_count += 1
        asyncio.create_task(
            _run_agent_turn(
                ws=ws,
                stream_sid=stream_sid,
                call_sid=call_sid,
                transcript=text,
                stt_latency_ms=stt_latency_ms,
                turn_count=turn_count,
                tts_client=tts_client,
            )
        )

    stt_client = DeepgramSTTClient(on_partial=on_partial, on_final=on_final)

    try:
        # ── Send greeting ─────────────────────────────────────────────────
        async for raw in ws.iter_text():
            data = json.loads(raw)
            event = data.get("event")

            if event == "connected":
                logger.info("Media Stream: connected event")

            elif event == "start":
                call_sid = data["start"].get("callSid", "")
                stream_sid = data["start"].get("streamSid", "")
                greeting_text = (
                    data["start"]
                    .get("customParameters", {})
                    .get("greeting", "Hello, how can I help you?")
                )
                logger.info(f"Media Stream: start  call_sid={call_sid}")
                await stt_client.connect()

                # Send greeting TTS immediately
                asyncio.create_task(
                    _send_tts(ws, stream_sid, greeting_text, tts_client)
                )

            elif event == "media":
                payload_b64 = data["media"]["payload"]
                audio_bytes = base64.b64decode(payload_b64)
                await stt_client.send_audio(audio_bytes)

            elif event == "stop":
                logger.info("Media Stream: stop event")
                break

    except WebSocketDisconnect:
        logger.info("Media Stream: WebSocket disconnected")
    finally:
        await stt_client.finish()
        logger.info(f"Call {call_sid} ended after {turn_count} turns")


# ── Agent turn (runs as asyncio Task) ────────────────────────────────────

async def _run_agent_turn(
    *,
    ws: WebSocket,
    stream_sid: str,
    call_sid: str,
    transcript: str,
    stt_latency_ms: float,
    turn_count: int,
    tts_client: ElevenLabsTTSClient,
) -> None:
    """Invoke LangGraph, get response text, stream to TTS → Twilio."""
    logger.info(f"[TURN {turn_count}] transcript='{transcript}'")

    initial_state: ConversationState = {
        "call_sid": call_sid,
        "turn_count": turn_count,
        "transcript_partial": "",
        "transcript_final": transcript,
        "intent": None,
        "retrieved_context": [],
        "llm_response": None,
        "tool_result": None,
        "escalate_flag": False,
        "barge_in_detected": False,
        "latency_log": {
            "stt_ms": stt_latency_ms,
            "llm_ms": None,
            "tts_first_chunk_ms": None,
            "total_ms": None,
        },
    }

    t0 = time.monotonic()
    final_state = await asyncio.get_event_loop().run_in_executor(
        None, lambda: agent_graph.invoke(initial_state)
    )
    response_text = final_state.get("llm_response", "I'm sorry, I didn't catch that. Could you repeat?")

    # Stream TTS back to Twilio
    await _send_tts(ws, stream_sid, response_text, tts_client)

    total_ms = (time.monotonic() - t0) * 1000
    latency_log = dict(final_state.get("latency_log", {}))
    latency_log["total_ms"] = total_ms

    # Log the turn
    _call_logger.log_turn(
        call_sid=call_sid,
        turn_count=turn_count,
        transcript=transcript,
        intent=final_state.get("intent"),
        llm_response=response_text,
        latency_log=latency_log,
        escalated=final_state.get("escalate_flag", False),
    )
    logger.info(f"[TURN {turn_count}] total latency={total_ms:.0f} ms")


async def _send_tts(
    ws: WebSocket,
    stream_sid: str,
    text: str,
    tts_client: ElevenLabsTTSClient,
) -> None:
    """Stream ElevenLabs TTS audio chunks back through Twilio WebSocket."""
    t0 = time.monotonic()
    first = True
    async for chunk in tts_client.synthesize(text):
        if first:
            logger.info(f"[TTS→TWILIO] first chunk in {(time.monotonic()-t0)*1000:.0f} ms")
            first = False
        payload = base64.b64encode(chunk).decode("utf-8")
        await ws.send_text(
            json.dumps({
                "event": "media",
                "streamSid": stream_sid,
                "media": {"payload": payload},
            })
        )
    # Mark end of playback
    await ws.send_text(json.dumps({"event": "mark", "streamSid": stream_sid, "mark": {"name": "tts_end"}}))
