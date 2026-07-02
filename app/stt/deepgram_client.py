"""
app/stt/deepgram_client.py

Wraps Deepgram's LiveTranscription WebSocket.
Accepts raw μ-law (8 kHz, mono) bytes from Twilio and fires callbacks:
  • on_partial(text)  — interim results   → feed to barge-in handler
  • on_final(text)    — committed transcript → trigger LangGraph turn
"""

from __future__ import annotations

import asyncio
import time
from typing import Callable, Optional

from deepgram import (
    DeepgramClient,
    DeepgramClientOptions,
    LiveOptions,
    LiveTranscriptionEvents,
)
from loguru import logger

from app.config import settings


class DeepgramSTTClient:
    def __init__(
        self,
        on_partial: Callable[[str], None],
        on_final: Callable[[str, float], None],  # text, stt_latency_ms
    ):
        self._on_partial = on_partial
        self._on_final = on_final
        self._connection = None
        self._start_time: Optional[float] = None

        cfg = DeepgramClientOptions(options={"keepalive": "true"})
        self._client = DeepgramClient(settings.deepgram_api_key, cfg)

    async def connect(self) -> None:
        """Open a live transcription session."""
        options = LiveOptions(
            model="nova-2",
            language="en-US",
            encoding="mulaw",
            sample_rate=8000,
            channels=1,
            interim_results=True,
            utterance_end_ms="1000",
            vad_events=True,
            smart_format=True,
        )

        self._connection = self._client.listen.asynclive.v("1")

        self._connection.on(LiveTranscriptionEvents.Transcript, self._on_transcript)
        self._connection.on(LiveTranscriptionEvents.Error, self._on_error)
        self._connection.on(LiveTranscriptionEvents.Close, self._on_close)

        await self._connection.start(options)
        logger.info("Deepgram STT connection opened")

    async def send_audio(self, chunk: bytes) -> None:
        """Forward raw audio bytes (μ-law) to Deepgram."""
        if self._connection:
            if self._start_time is None:
                self._start_time = time.monotonic()
            await self._connection.send(chunk)

    async def finish(self) -> None:
        """Signal end of stream and close connection."""
        if self._connection:
            await self._connection.finish()
            self._connection = None
            self._start_time = None
            logger.info("Deepgram STT connection closed")

    # ── Private callbacks ─────────────────────────────────────────────────

    def _on_transcript(self, _self, result, **kwargs) -> None:
        try:
            alt = result.channel.alternatives[0]
            text = alt.transcript.strip()
            if not text:
                return

            if result.is_final:
                latency_ms = (
                    (time.monotonic() - self._start_time) * 1000
                    if self._start_time
                    else 0.0
                )
                self._start_time = None  # reset for next utterance
                logger.info(f"[STT FINAL] '{text}'  ({latency_ms:.0f} ms)")
                self._on_final(text, latency_ms)
            else:
                logger.debug(f"[STT PARTIAL] '{text}'")
                self._on_partial(text)
        except Exception as exc:
            logger.error(f"STT transcript callback error: {exc}")

    def _on_error(self, _self, error, **kwargs) -> None:
        logger.error(f"Deepgram error: {error}")

    def _on_close(self, _self, close, **kwargs) -> None:
        logger.info("Deepgram connection closed by server")
