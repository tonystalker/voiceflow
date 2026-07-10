"""
main.py

Entrypoint for VoiceFlow v2 — Personal Assistant Mode.
Runs the state machine (IDLE -> LISTENING -> THINKING -> SPEAKING)
which uses openWakeWord for wake detection and Deepgram for STT.
"""

import asyncio
import sys
import io
import traceback
from pathlib import Path

# Force UTF-8 and line-buffering so logs appear immediately
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
if hasattr(sys.stderr, "buffer"):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace", line_buffering=True)

from app.assistant.state_machine import StateMachine

async def async_main():
    machine = StateMachine()
    await machine.run()

if __name__ == "__main__":
    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        print("\nStopped by user.")
    except Exception:
        crash_path = Path(__file__).parent / "crash.log"
        with open(crash_path, "w", encoding="utf-8") as f:
            traceback.print_exc(file=f)
        print(f"\n[CRASH] Full traceback written to: {crash_path}")
        traceback.print_exc()
