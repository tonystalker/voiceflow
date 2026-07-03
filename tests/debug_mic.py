"""
tests/debug_mic.py — minimal diagnostic script

Checks:
  1. Is the microphone capturing audio?
  2. Are chunks reaching Deepgram?
  3. Is Deepgram sending anything back?

Run: python tests/debug_mic.py
"""
import asyncio
import audioop
import queue
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

import numpy as np
import sounddevice as sd
from deepgram import Deepgram
from app.config import settings

SAMPLE_RATE = 8000
BLOCK_SIZE = 320
_q: queue.SimpleQueue = queue.SimpleQueue()
chunk_count = 0


def _callback(indata: np.ndarray, frames: int, time_info, status) -> None:
    global chunk_count
    pcm = (indata[:, 0] * 32767).astype(np.int16).tobytes()
    rms = np.sqrt(np.mean(indata ** 2))
    _q.put_nowait((audioop.lin2ulaw(pcm, 2), rms))


async def main() -> None:
    print("=== MIC DEBUG ===")
    print(f"Deepgram API key: {settings.deepgram_api_key[:8]}...")
    print(f"Mic sample rate : {SAMPLE_RATE} Hz")

    dg = Deepgram(settings.deepgram_api_key)
    conn = await dg.transcription.live({
        "encoding": "mulaw",
        "sample_rate": SAMPLE_RATE,
        "channels": 1,
        "interim_results": True,
        "smart_format": True,
    })
    print("[OK] Deepgram connected\n")

    # Print EVERY message from Deepgram (raw)
    def on_message(msg: dict) -> None:
        alt = msg.get("channel", {}).get("alternatives", [{}])[0]
        text = alt.get("transcript", "").strip()
        is_final = msg.get("is_final", False)
        if text:
            tag = "[FINAL]" if is_final else "[partial]"
            print(f"\n{tag} {text}")

    def on_any(msg) -> None:
        # catches anything not caught by TRANSCRIPT_RECEIVED
        print(f"[RAW] {msg}")

    conn.registerHandler(conn.event.TRANSCRIPT_RECEIVED, on_message)
    conn.registerHandler(conn.event.CLOSE, lambda c: print(f"[CLOSE] code={c}"))

    sent = 0

    async def pump() -> None:
        nonlocal sent
        while True:
            try:
                chunk, rms = _q.get_nowait()
                conn.send(chunk)
                sent += 1
                if sent % 50 == 0:   # every ~2 seconds
                    print(f"  [mic] chunks sent={sent}  RMS={rms:.4f}  "
                          f"{'(silence)' if rms < 0.002 else '(AUDIO DETECTED)'}")
            except queue.Empty:
                await asyncio.sleep(0.005)

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=BLOCK_SIZE,
        callback=_callback,
    ):
        print("Speak now — you have 30 seconds.")
        print("You should see RMS values and Deepgram transcripts appear.\n")
        pump_task = asyncio.create_task(pump())
        await asyncio.sleep(30)
        pump_task.cancel()

    await conn.finish()
    print(f"\nDone. Total chunks sent to Deepgram: {sent}")


if __name__ == "__main__":
    asyncio.run(main())
