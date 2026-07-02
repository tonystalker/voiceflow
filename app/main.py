"""
app/main.py

FastAPI application entrypoint.
Mounts telephony routes and exposes a health check.

Run with:
    uvicorn app.main:app --reload --port 8000
"""

from __future__ import annotations

from fastapi import FastAPI
from loguru import logger

from app.telephony.twilio_webhook import router as twilio_router
from app.telephony.media_stream import router as media_router

app = FastAPI(
    title="VoiceFlow AI",
    description="Real-time voice AI agent — Twilio + Deepgram + ElevenLabs + LangGraph",
    version="1.0.0",
)

# ── Routes ────────────────────────────────────────────────────────────────
app.include_router(twilio_router)
app.include_router(media_router)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "service": "VoiceFlow AI"}


@app.on_event("startup")
async def startup() -> None:
    logger.info("VoiceFlow AI server started ✓")
