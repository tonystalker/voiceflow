import asyncio
import audioop
import queue
import time
import traceback
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np
import sounddevice as sd
from deepgram import (
    DeepgramClient,
    LiveTranscriptionEvents,
    LiveOptions,
)
import webrtcvad
import openwakeword
from openwakeword.model import Model

from app.config import settings
from app.agent.graph import agent_graph
from app.agent.state import ConversationState
from app.tts.elevenlabs_client import ElevenLabsTTSClient


class State(Enum):
    IDLE      = "IDLE"      # Waiting for wake word (Hey Jarvis)
    LISTENING = "LISTENING" # Capturing user speech for Deepgram
    THINKING  = "THINKING"  # LangGraph LLM inference
    SPEAKING  = "SPEAKING"  # ElevenLabs TTS playback


class StateMachine:
    def __init__(self):
        # Configuration
        self.sample_rate = 16000
        self.block_size  = 1280  # 80ms at 16kHz for openWakeWord
        
        # Audio components
        openwakeword.utils.download_models()
        self.oww = Model(wakeword_models=["hey_jarvis"], inference_framework="onnx")
        self.tts = ElevenLabsTTSClient()
        self.dg  = DeepgramClient(settings.deepgram_api_key)
        
        # State
        self.current_state = State.IDLE
        self.turn = 0
        self.dg_conn = None
        self.audio_q = queue.SimpleQueue()
        self.loop = None
        
        # VAD for Barge-in
        self.vad = webrtcvad.Vad(3)
        self.barge_in_counter = 0
        self.barge_in_triggered = False
        
        # Tracking
        self.start_time = 0.0
        self.pending_text = []
        self.session_state = {}  # Persisted across turns

    def audio_callback(self, indata: np.ndarray, frames: int, time_info, status):
        """Single mic callback handling both wake-word (IDLE) and STT (LISTENING)."""
        if status:
            print(status)
            
        if self.current_state not in (State.IDLE, State.LISTENING, State.SPEAKING):
            return

        # PCM 16-bit 16kHz (both openWakeWord and Deepgram accept this)
        # Apply a software gain multiplier (e.g., 10.0) since the mic input is very quiet
        MIC_GAIN = 10.0
        audio_float = indata[:, 0] * 32767 * MIC_GAIN
        audio_int16 = np.clip(audio_float, -32768, 32767).astype(np.int16)

        if self.current_state == State.IDLE:
            # Feed openWakeWord
            prediction = self.oww.predict(audio_int16)
            score = prediction.get("hey_jarvis", 0.0)
            
            # Print a subtle volume bar for debug (optional, can be disabled)
            vol = np.max(np.abs(audio_int16))
            if vol > 500:
                bars = int((vol / 32767) * 40)
                print(f"[IDLE] [{('#'*bars).ljust(40, '-')}] Vol: {vol:5d} | Score: {score:.4f}      ", end="\r")

            # Lowered threshold so you don't have to shout
            if score > 0.03:
                print("\n[WAKE] Tony is listening...                        ")
                self.transition_to(State.LISTENING)

        elif self.current_state == State.LISTENING:
            # Feed Deepgram (needs mulaw at 8000Hz or 16000Hz PCM)
            # Since Deepgram is configured for 16kHz mulaw below, we convert it here:
            pcm_bytes = audio_int16.tobytes()
            mulaw = audioop.lin2ulaw(pcm_bytes, 2)
            self.audio_q.put_nowait(mulaw)
            
        elif self.current_state == State.SPEAKING:
            # Barge-in VAD (requires headphones to prevent echo)
            pcm_bytes = audio_int16.tobytes()
            # webrtcvad needs 10/20/30ms frames. 30ms @ 16kHz = 480 samples = 960 bytes
            is_speech = False
            for i in range(0, len(pcm_bytes) - 960, 960):
                if self.vad.is_speech(pcm_bytes[i:i+960], self.sample_rate):
                    is_speech = True
                    break
            
            if is_speech:
                self.barge_in_counter += 1
                if self.barge_in_counter >= 3 and not self.barge_in_triggered:  # ~240ms of sustained speech
                    print("\n[BARGE-IN] Detected! Interrupting TTS...")
                    self.barge_in_triggered = True
                    sd.stop()
                    self.barge_in_counter = 0
            else:
                self.barge_in_counter = 0

    def transition_to(self, new_state: State):
        """Handle state transitions and necessary side-effects."""
        old_state = self.current_state
        self.current_state = new_state
        
        if new_state == State.LISTENING and old_state == State.IDLE:
            # Wake word triggered -> connect to deepgram
            asyncio.run_coroutine_threadsafe(self.start_deepgram(), self.loop)
            
        elif new_state == State.IDLE:
            # Going back to idle -> clean up any deepgram connection
            if self.dg_conn:
                asyncio.run_coroutine_threadsafe(self.dg_conn.finish(), self.loop)
                self.dg_conn = None
            print("\n[IDLE] Waiting for wake word (Hey Jarvis)...")

    async def start_deepgram(self):
        """Open Deepgram connection for a new conversation turn."""
        self.start_time = 0.0
        self.pending_text.clear()
        
        # Sleep briefly to let the trailing audio of the wake word pass
        await asyncio.sleep(0.5)
        
        # Clear out any old audio frames (including the wake word tail)
        while not self.audio_q.empty():
            self.audio_q.get_nowait()
            
        self.dg_conn = self.dg.listen.asyncwebsocket.v("1")
        
        async def on_close(c, **kwargs):
            print(f"[Deepgram closed]")
            
        async def on_error(c, error, **kwargs):
            print(f"[Deepgram error] {error}")
            
        self.dg_conn.on(LiveTranscriptionEvents.Transcript, self.on_transcript)
        self.dg_conn.on(LiveTranscriptionEvents.Close, on_close)
        self.dg_conn.on(LiveTranscriptionEvents.Error, on_error)
        
        options = LiveOptions(
            encoding="mulaw",
            sample_rate=self.sample_rate,
            channels=1,
            interim_results=True,
            smart_format=True,
            endpointing=150,  # 150ms endpointing for quick responses
            utterance_end_ms=1000 # 1000ms max silence
        )
        
        await self.dg_conn.start(options)
        
        # Start pump
        asyncio.create_task(self.audio_pump())

    async def audio_pump(self):
        """Pumps audio from the mic queue to Deepgram while in LISTENING state."""
        while self.current_state == State.LISTENING and self.dg_conn:
            try:
                chunk = self.audio_q.get_nowait()
                if not self.start_time:
                    self.start_time = time.monotonic()
                await self.dg_conn.send(chunk)
            except queue.Empty:
                await asyncio.sleep(0.005)

    async def on_transcript(self, client, result, **kwargs):
        if self.current_state != State.LISTENING:
            return
            
        try:
            alt = result.channel.alternatives[0]
            text = alt.transcript.strip()
            if not text:
                return

            speech_final = result.speech_final
            is_final = result.is_final
        except (AttributeError, IndexError):
            return

        if not is_final:
            print(f"\r[...] {text}   ", end="", flush=True)
            return

        if speech_final:
            # End of utterance
            full_text = text
            if self.pending_text:
                self.pending_text.append(text)
                full_text = " ".join(self.pending_text)
                self.pending_text.clear()

            stt_ms = (time.monotonic() - self.start_time) * 1000 if self.start_time else 0.0
            print(f"\n[YOU]   {full_text}  (STT: {stt_ms:.0f} ms)")
            
            # Transition to thinking and run LLM
            self.transition_to(State.THINKING)
            asyncio.create_task(self.run_turn(full_text, stt_ms))
            
        else:
            self.pending_text.append(text)

    async def run_turn(self, text: str, stt_ms: float):
        """Execute the LangGraph agent and ElevenLabs TTS."""
        self.turn += 1
        t0 = time.monotonic()

        state: ConversationState = {
            "call_sid":            "LOCAL_WAKEWORD",
            "turn_count":          self.turn,
            "transcript_partial":  "",
            "transcript_final":    text,
            "intent":              None,
            "retrieved_context":   [],
            "llm_response":        None,
            "tool_result":         None,
            "pending_action":      self.session_state.get("pending_action"),
            "confirmed":           None,
            "escalate_flag":       False,
            "barge_in_detected":   False,
            "latency_log":         {
                "stt_ms":             stt_ms,
                "llm_ms":             None,
                "tts_first_chunk_ms": None,
                "total_ms":           None,
            },
        }

        # Run LLM (THINKING state)
        final = await agent_graph.ainvoke(state)
        
        # Persist fields for next turn
        self.session_state["pending_action"] = final.get("pending_action")
        
        response = final.get("llm_response", "Sorry, I could not process that.")
        llm_ms = final.get("latency_log", {}).get("llm_ms", 0) or 0
        
        print(f"\n[TONY]  {response}")
        print(f"        (LLM: {llm_ms:.0f} ms)")

        # Play TTS (SPEAKING state)
        self.transition_to(State.SPEAKING)
        
        try:
            chunks = []
            tts_t0 = time.monotonic()
            first = True
            
            async for mulaw_chunk in self.tts.synthesize(response):
                pcm = audioop.ulaw2lin(mulaw_chunk, 2)
                arr = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
                chunks.append(arr)
                if first:
                    print(f"        (TTS first chunk: {(time.monotonic()-tts_t0)*1000:.0f} ms)")
                    first = False

            if chunks:
                audio_data = np.concatenate(chunks)
                sd.play(audio_data, samplerate=8000)
                await asyncio.to_thread(sd.wait)
                
        finally:
            total_ms = (time.monotonic() - t0) * 1000
            print(f"        [total: {total_ms:.0f} ms]")
            print("-" * 50)
            
            if self.barge_in_triggered:
                # Go directly to LISTENING to capture the user's interruption
                self.barge_in_triggered = False
                self.transition_to(State.LISTENING)
                import logging
                task = asyncio.create_task(self.start_deepgram())
                task.add_done_callback(lambda t: t.exception() and print(f"Task crashed: {t.exception()!r}"))
            else:
                # Back to IDLE state to listen for wake word again
                self.transition_to(State.IDLE)

    async def run(self):
        """Main loop that keeps the program alive and opens the audio stream."""
        self.loop = asyncio.get_event_loop()
        
        print("=" * 50)
        print("VoiceFlow AI — v2 Assistant Mode")
        print("Say 'Hey Jarvis' to wake the assistant.")
        print("Ctrl+C to stop.")
        print("=" * 50)
        
        with sd.InputStream(
            samplerate=self.sample_rate, channels=1,
            dtype="float32", blocksize=self.block_size,
            callback=self.audio_callback,
        ):
            print("\n[IDLE] Waiting for wake word (Hey Jarvis)...")
            try:
                while True:
                    await asyncio.sleep(0.1)
            except asyncio.CancelledError:
                pass
