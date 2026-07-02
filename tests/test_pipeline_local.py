"""
tests/test_pipeline_local.py

Day 1–2 local pipeline test — NO TWILIO REQUIRED.
Tests the full loop: mic → Deepgram → LangGraph → ElevenLabs → speaker.
Measures per-stage latency and prints a summary.

Usage:
    python tests/test_pipeline_local.py

Press Ctrl+C to stop.
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parents[1]))

import numpy as np
import sounddevice as sd
import soundfile as sf

from app.stt.deepgram_client import DeepgramSTTClient
from app.tts.elevenlabs_client import ElevenLabsTTSClient
from app.agent.graph import agent_graph
from app.agent.state import ConversationState

SAMPLE_RATE = 8000
CHANNELS = 1
BLOCK_SIZE = 320   # 40ms blocks at 8kHz (Twilio-equivalent chunk size)

turn_count = 0
tts_client = ElevenLabsTTSClient()


def on_partial(text: str) -> None:
    print(f"\r[PARTIAL] {text}   ", end="", flush=True)


def on_final(text: str, stt_ms: float) -> None:
    global turn_count
    turn_count += 1
    print(f"\n[FINAL] '{text}'  (STT: {stt_ms:.0f} ms)")
    asyncio.create_task(_run_turn(text, stt_ms))


async def _run_turn(transcript: str, stt_ms: float) -> None:
    t0 = time.monotonic()

    state: ConversationState = {
        "call_sid": "LOCAL_TEST",
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
            "stt_ms": stt_ms,
            "llm_ms": None,
            "tts_first_chunk_ms": None,
            "total_ms": None,
        },
    }

    # Run agent in thread executor (LangGraph is sync)
    loop = asyncio.get_event_loop()
    final_state = await loop.run_in_executor(None, lambda: agent_graph.invoke(state))
    response_text = final_state.get("llm_response", "Sorry, I couldn't process that.")

    llm_ms = final_state.get("latency_log", {}).get("llm_ms", 0)
    print(f"[AGENT] '{response_text}'  (LLM: {llm_ms:.0f} ms)")

    # Play TTS through speaker
    audio_pcm_chunks = []
    tts_start = time.monotonic()
    first_chunk_logged = False
    async for mulaw_chunk in tts_client.synthesize(response_text):
        # Convert μ-law back to PCM 16-bit for local playback
        import audioop
        pcm_chunk = audioop.ulaw2lin(mulaw_chunk, 2)
        audio_array = np.frombuffer(pcm_chunk, dtype=np.int16).astype(np.float32) / 32768.0
        audio_pcm_chunks.append(audio_array)
        if not first_chunk_logged:
            tts_first_ms = (time.monotonic() - tts_start) * 1000
            print(f"[TTS] first chunk in {tts_first_ms:.0f} ms")
            first_chunk_logged = True

    if audio_pcm_chunks:
        full_audio = np.concatenate(audio_pcm_chunks)
        sd.play(full_audio, samplerate=SAMPLE_RATE)
        sd.wait()

    total_ms = (time.monotonic() - t0) * 1000
    print(f"[LATENCY] STT={stt_ms:.0f}ms  LLM={llm_ms:.0f}ms  Total={total_ms:.0f}ms")
    print("-" * 60)
    print("Listening... (speak now)")


async def main() -> None:
    print("=" * 60)
    print("VoiceFlow AI — Local Pipeline Test")
    print("Speak into your mic. Press Ctrl+C to stop.")
    print("=" * 60)

    stt = DeepgramSTTClient(on_partial=on_partial, on_final=on_final)
    await stt.connect()

    # μ-law encoding of mic input
    import audioop

    def audio_callback(indata: np.ndarray, frames: int, time_info, status) -> None:
        pcm_bytes = (indata[:, 0] * 32767).astype(np.int16).tobytes()
        mulaw_bytes = audioop.lin2ulaw(pcm_bytes, 2)
        asyncio.get_event_loop().call_soon_threadsafe(
            asyncio.ensure_future, stt.send_audio(mulaw_bytes)
        )

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="float32",
        blocksize=BLOCK_SIZE,
        callback=audio_callback,
    ):
        print("\nListening... (speak now)")
        try:
            await asyncio.sleep(3600)  # Run for 1 hour max
        except asyncio.CancelledError:
            pass

    await stt.finish()
    print("\nTest ended.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nStopped by user.")
