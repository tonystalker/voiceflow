"""
app/wakeword/openwakeword_listener.py

Standalone wake-word listener. Trigger phrase: "Hey Jarvis" (built-in openWakeWord model,
no custom training required). Assistant persona is "Tony" — the wake phrase and the spoken
name are intentionally decoupled; see Phase A update notes.

Run standalone to verify in isolation before wiring into the state machine:
    python app/wakeword/openwakeword_listener.py
"""

import numpy as np
import sounddevice as sd
from openwakeword.model import Model

SAMPLE_RATE = 16000
FRAME_SIZE  = 1280   # 80ms at 16kHz — openWakeWord's expected chunk size
THRESHOLD   = 0.5

oww = Model(wakeword_models=["hey_jarvis"], inference_framework="onnx")

print("=" * 50)
print("Tony — Wake Word Listener (openWakeWord)")
print('Say "Hey Jarvis" to wake.  Ctrl+C to stop.')
print("=" * 50)


def _callback(indata: np.ndarray, frames: int, time_info, status) -> None:
    audio = (indata[:, 0] * 32767).astype(np.int16)
    
    # Debug: print volume bar and score occasionally
    vol = np.max(np.abs(audio))
    prediction = oww.predict(audio)
    score = prediction.get("hey_jarvis", 0.0)
    
    if vol > 500:  # Only print if there's actual sound
        # Create a simple volume bar
        bars = int((vol / 32767) * 40)
        bar_str = "#" * bars + "-" * (40 - bars)
        print(f"[DEBUG] Vol: [{bar_str}] {vol:5d} | Score: {score:.4f}      ", end="\r")

    if score > THRESHOLD:
        print("\n[WAKE] Tony is listening...")


if __name__ == "__main__":
    import time
    with sd.InputStream(
        samplerate=SAMPLE_RATE, channels=1,
        dtype="float32", blocksize=FRAME_SIZE,
        callback=_callback,
    ):
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nStopped by user.")
