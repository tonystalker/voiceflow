"""
tests/test_pipeline_local.py

Local pipeline test -- NO TWILIO REQUIRED.
mic -> Deepgram -> LangGraph -> ElevenLabs -> speaker

Fixes applied:
  Bug 1: Mic gated off during TTS playback (no more acoustic echo)
  Bug 2: _busy only silences the mic; final transcripts are queued, not dropped
  Bug 3: STT timer resets cleanly at listen-start, not mid-TTS
  Bug 4: speech_final (not is_final) drives turn boundaries to prevent fragmentation

Run:  python tests/test_pipeline_local.py
Stop: Ctrl+C
"""
from __future__ import annotations

import asyncio
import audioop
import io
import queue
import sys
import time
import traceback
from pathlib import Path

# ── Force UTF-8 output (Windows cp1252 terminal can't display some chars) ─
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
if hasattr(sys.stderr, "buffer"):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace", line_buffering=True)

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parents[1]))

import numpy as np
import sounddevice as sd
from deepgram import Deepgram

from app.config import settings
from app.agent.graph import agent_graph
from app.agent.state import ConversationState
from app.tts.elevenlabs_client import ElevenLabsTTSClient

SAMPLE_RATE = 8000
BLOCK_SIZE  = 320      # 40 ms at 8 kHz

_audio_q: queue.SimpleQueue = queue.SimpleQueue()
_tts       = ElevenLabsTTSClient()
_loop: asyncio.AbstractEventLoop | None = None
_turn       = 0

# _speaking: True while TONY is playing audio.
# The mic callback checks this flag — if True it discards the frame,
# implementing half-duplex (Bug 1 fix).  This also indirectly prevents
# Deepgram from receiving echo and avoids silent drops (Bug 2).
_speaking   = False


# ── sounddevice callback (runs on a C audio thread) ──────────────────────

def _mic_callback(indata: np.ndarray, frames: int, time_info, status) -> None:
    if _speaking:
        return           # gate mic off during TTS playback (Bug 1)
    pcm   = (indata[:, 0] * 32767).astype(np.int16).tobytes()
    mulaw = audioop.lin2ulaw(pcm, 2)
    _audio_q.put_nowait(mulaw)


# ── Agent + TTS (runs as an asyncio Task) ────────────────────────────────

async def _run_turn(text: str, stt_ms: float) -> None:
    """Run one full agent turn: LLM → TTS → playback."""
    global _turn, _speaking

    _turn += 1
    t0 = time.monotonic()

    state: ConversationState = {
        "call_sid":            "LOCAL_TEST",
        "turn_count":          _turn,
        "transcript_partial":  "",
        "transcript_final":    text,
        "intent":              None,
        "retrieved_context":   [],
        "llm_response":        None,
        "tool_result":         None,
        "escalate_flag":       False,
        "barge_in_detected":   False,
        "latency_log":         {
            "stt_ms":             stt_ms,
            "llm_ms":             None,
            "tts_first_chunk_ms": None,
            "total_ms":           None,
        },
    }

    # LLM inference — run in thread pool to avoid blocking the event loop
    loop  = asyncio.get_event_loop()
    final = await loop.run_in_executor(None, lambda: agent_graph.invoke(state))
    response = final.get("llm_response", "Sorry, I could not process that.")
    llm_ms   = final.get("latency_log", {}).get("llm_ms", 0) or 0

    print(f"\n[TONY]  {response}")
    print(f"        (LLM: {llm_ms:.0f} ms)")

    # Gate mic OFF before we start playing (Bug 1)
    _speaking = True
    try:
        # Stream TTS → collect PCM frames
        chunks  = []
        tts_t0  = time.monotonic()
        first   = True
        async for mulaw_chunk in _tts.synthesize(response):
            pcm = audioop.ulaw2lin(mulaw_chunk, 2)
            arr = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
            chunks.append(arr)
            if first:
                print(f"        (TTS first chunk: {(time.monotonic()-tts_t0)*1000:.0f} ms)")
                first = False

        if chunks:
            sd.play(np.concatenate(chunks), samplerate=SAMPLE_RATE)
            await asyncio.to_thread(sd.wait)   # non-blocking wait
    finally:
        # Always un-gate the mic, even if TTS errors (Bug 1)
        _speaking = False

    total_ms = (time.monotonic() - t0) * 1000
    print(f"        [total: {total_ms:.0f} ms]")
    print("-" * 50)
    print("Listening... (speak now)\n")


# ── Main ──────────────────────────────────────────────────────────────────

async def main() -> None:
    global _loop
    _loop = asyncio.get_event_loop()

    print("=" * 50)
    print("VoiceFlow AI  --  Local Pipeline Test")
    print("Speak into your mic.  Ctrl+C to stop.")
    print("=" * 50)

    dg   = Deepgram(settings.deepgram_api_key)
    conn = await dg.transcription.live({
        "encoding":        "mulaw",
        "sample_rate":     SAMPLE_RATE,
        "channels":        1,
        "interim_results": True,
        "smart_format":    True,
    })
    print("[OK] Deepgram connected\n")

    # STT timer: measures from first audio byte → final transcript (Bug 3 fix)
    # Only set when listening (_speaking is False), so TTS playback time
    # is never included in the measurement.
    start_time: list[float] = [0.0]

    # Pending turn text (Bug 2 fix): accumulate speech_final fragments that
    # arrive while the agent is busy, then fire once mic re-opens.
    _pending: list[str] = []

    def on_transcript(msg: dict) -> None:
        alt   = msg.get("channel", {}).get("alternatives", [{}])[0]
        text  = alt.get("transcript", "").strip()
        if not text:
            return

        # Use speech_final (utterance boundary) rather than is_final
        # (packet boundary) so we don't split mid-sentence (Bug 4 fix).
        # speech_final is True when Deepgram detects end of an utterance.
        speech_final = msg.get("speech_final", False)
        is_final     = msg.get("is_final",     False)

        if not is_final:
            # Interim/partial — show live caption only when listening
            if not _speaking:
                print(f"\r[...] {text}   ", end="", flush=True)
            return

        # is_final packet — text is committed
        if speech_final:
            # Utterance complete: fire a turn
            full_text = text
            if _pending:
                # If we were mid-accumulation flush them together
                _pending.append(text)
                full_text = " ".join(_pending)
                _pending.clear()

            stt_ms = (time.monotonic() - start_time[0]) * 1000 if start_time[0] else 0.0
            start_time[0] = 0.0

            print(f"\n[YOU]   {full_text}  (STT: {stt_ms:.0f} ms)")

            if _speaking:
                # Agent is still talking — queue the text, print a note
                print(f"        [queued — agent still speaking]")
                _pending.append(full_text)
            else:
                asyncio.ensure_future(_run_turn(full_text, stt_ms), loop=_loop)
        else:
            # is_final but not speech_final → partial utterance chunk
            # Accumulate until we get speech_final
            _pending.append(text)

    conn.registerHandler(conn.event.TRANSCRIPT_RECEIVED, on_transcript)
    conn.registerHandler(conn.event.CLOSE, lambda c: print(f"\n[Deepgram closed code={c}]"))

    # Audio pump: drains _audio_q and forwards to Deepgram.
    # Also sets start_time on first chunk after mic re-opens (Bug 3 fix).
    async def pump() -> None:
        while True:
            try:
                chunk = _audio_q.get_nowait()
                # Only start the STT timer when we're actually listening
                if not _speaking and not start_time[0]:
                    start_time[0] = time.monotonic()
                conn.send(chunk)
            except queue.Empty:
                await asyncio.sleep(0.005)

    with sd.InputStream(
        samplerate=SAMPLE_RATE, channels=1,
        dtype="float32", blocksize=BLOCK_SIZE,
        callback=_mic_callback,
    ):
        print("Listening... (speak now)\n")
        pump_task = asyncio.ensure_future(pump(), loop=_loop)
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            pass
        pump_task.cancel()

    await conn.finish()
    print("\nSession ended.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nStopped by user.")
    except Exception:
        crash_path = Path(__file__).parent / "crash.log"
        with open(crash_path, "w", encoding="utf-8") as f:
            traceback.print_exc(file=f)
        print(f"\n[CRASH] Full traceback written to: {crash_path}")
        traceback.print_exc()
