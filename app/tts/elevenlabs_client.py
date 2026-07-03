"""
app/tts/elevenlabs_client.py

Wraps ElevenLabs streaming TTS.
Yields μ-law 8 kHz audio chunks suitable for Twilio Media Streams.
Exposes stop() for barge-in interruption mid-stream.
"""

from __future__ import annotations

import asyncio
import audioop
import time
from typing import AsyncIterator

import httpx
from loguru import logger

from app.config import settings

# output_format must be a QUERY PARAM, not a body field
_STREAM_URL = (
    "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream"
    "?output_format=pcm_8000&optimize_streaming_latency=3"
)


class ElevenLabsTTSClient:
    def __init__(self) -> None:
        self._stop_event: asyncio.Event = asyncio.Event()

    def stop(self) -> None:
        """Signal an in-progress stream to abort (barge-in)."""
        self._stop_event.set()
        logger.info("[TTS] stop() called — barge-in interrupt")

    def reset(self) -> None:
        """Clear stop flag before starting a new utterance."""
        self._stop_event.clear()

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        """
        Stream TTS for *text*.
        Yields μ-law 8 kHz mono byte chunks.
        Stops early if stop() is called.
        """
        self.reset()
        url = _STREAM_URL.format(voice_id=settings.elevenlabs_voice_id)

        headers = {
            "xi-api-key": settings.elevenlabs_api_key,
            "Content-Type": "application/json",
            # No Accept header — output_format in URL determines content type
        }
        payload = {
            "text": text,
            "model_id": "eleven_turbo_v2",
            "voice_settings": {
                "stability": 0.4,
                "similarity_boost": 0.75,
                "style": 0.0,
                "use_speaker_boost": True,
            },
        }

        first_chunk = True
        t0 = time.monotonic()

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                async with client.stream("POST", url, headers=headers, json=payload) as resp:
                    if resp.status_code != 200:
                        body = await resp.aread()
                        print(f"\n[TTS ERROR] ElevenLabs HTTP {resp.status_code}: {body.decode()[:200]}\n")
                        return

                    async for chunk in resp.aiter_bytes(chunk_size=4096):
                        if self._stop_event.is_set():
                            print("[TTS] stream aborted by barge-in")
                            return
                        if not chunk:
                            continue
                        if first_chunk:
                            elapsed = (time.monotonic() - t0) * 1000
                            print(f"[TTS] first chunk in {elapsed:.0f} ms")
                            first_chunk = False
                        # PCM 16-bit signed (pcm_8000) → μ-law 8-bit for Twilio
                        mulaw_chunk = audioop.lin2ulaw(chunk, 2)
                        yield mulaw_chunk

        except httpx.HTTPStatusError as exc:
            print(f"\n[TTS ERROR] ElevenLabs HTTP exception {exc.response.status_code}\n")
        except Exception as exc:
            print(f"\n[TTS ERROR] TTS synthesis exception: {exc}\n")
