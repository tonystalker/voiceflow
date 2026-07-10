# VoiceFlow AI — v1 Development Log

> Complete record of bottlenecks encountered, how we fixed them, our design architecture,
> and why we chose each technology (and rejected the alternatives).
>
> **Created:** July 10, 2026 — before starting v2 development.

---

## Table of Contents

1. [Bottlenecks & Fixes](#1-bottlenecks--fixes)
2. [Design Architecture](#2-design-architecture)
3. [Technology Choices & Rationale](#3-technology-choices--rationale)
4. [Latency Observations](#4-latency-observations)
5. [Lessons Learned for v2](#5-lessons-learned-for-v2)

---

## 1. Bottlenecks & Fixes

### Bug 1 — Acoustic Echo / Self-Transcription

**Problem:**
The microphone was capturing Aria's own TTS audio output, and Deepgram was transcribing it as
if the user had spoken. This caused a feedback loop where the agent would respond to its own
speech, creating an infinite conversation with itself.

**Root Cause:**
No mechanism existed to mute the mic during TTS playback. The system was full-duplex by default —
mic was always hot, even while the speaker was playing Aria's response.

**Fix (commit `ace54b3`):**
Added a `_speaking` / `_busy` boolean flag. The `_mic_callback` function in `test_pipeline_local.py`
checks this flag on every audio frame — if `_speaking` is `True`, the frame is silently discarded
instead of being enqueued to Deepgram.

```python
# test_pipeline_local.py — mic callback
def _mic_callback(indata, frames, time_info, status):
    if _speaking:
        return  # gate mic off during TTS playback (Bug 1 fix)
    pcm = (indata[:, 0] * 32767).astype(np.int16).tobytes()
    mulaw = audioop.lin2ulaw(pcm, 2)
    _audio_q.put_nowait(mulaw)
```

The flag is set to `True` before `sd.play()` starts and reset to `False` in a `finally` block to
guarantee it's always cleared, even if TTS errors out.

**Trade-off:** This is half-duplex — the user cannot interrupt Aria mid-sentence in this
implementation. Acceptable for v1; true barge-in with VAD is planned for v2.

---

### Bug 2 — Dropped Transcripts During Playback (Silent Drops)

**Problem:**
If the user spoke while Aria was still talking (and the mic was gated off via Bug 1 fix),
their speech was completely lost. After Aria finished, the system went silent with no prompt
to the user — it just sat there waiting for speech that had already been discarded.

**Root Cause:**
The mic gating fix (Bug 1) discarded audio at the callback level, so Deepgram never received
the speech. Any `is_final` transcript that happened to arrive during playback was also not
being queued — it was simply ignored.

**Fix (commit `ace54b3`):**
Instead of dropping final transcripts that arrive during `_speaking = True`, we queue them into
a `_pending` list. When `_speaking` flips back to `False`, the queued transcript is flushed and
a new agent turn is triggered.

```python
if _speaking:
    # Agent is still talking — queue the text, print a note
    _pending.append(full_text)
else:
    asyncio.ensure_future(_run_turn(full_text, stt_ms))
```

**Result:** No more silent dead-ends. Speech that arrives during playback is deferred, not lost.

---

### Bug 3 — Inflated STT Latency Measurements

**Problem:**
STT latency was showing absurd numbers (10+ seconds) even though Deepgram was clearly
responding fast. The latency metric was useless.

**Root Cause:**
The STT timer (`start_time`) was being set when the mic re-opened after TTS playback, but the
`time.monotonic()` measurement included the entire TTS playback duration. The timer was
started at "mic open" but the first audio byte didn't reach Deepgram until the user actually
started speaking — seconds later.

**Fix (commit `ace54b3`):**
Moved the timer start into the audio pump loop. `start_time` is now only set when:
1. `_speaking` is `False` (we're actually listening), AND
2. `start_time` is currently zero (not already timing)

```python
# Audio pump — set timer only on first real audio chunk
if not _speaking and not start_time[0]:
    start_time[0] = time.monotonic()
```

**Result:** STT latency now correctly measures "time from first audio byte sent → final
transcript received" — typically 200–500ms for Deepgram Nova 2.

---

### Bug 4 — Turn Fragmentation (Mid-Sentence Cutoffs)

**Problem:**
Aria was responding to half-sentences. The user would say "I want to check my account
balance for" and the agent would fire a turn on "I want to check my" — producing a confused
response before the user finished speaking.

**Root Cause:**
We were using `is_final` from Deepgram to trigger agent turns. But Deepgram sets `is_final`
on every committed packet boundary — which happens multiple times per sentence as the streaming
decoder commits partial results. `is_final` does NOT mean "the user stopped talking."

**Fix (commit `ace54b3`):**
Switched to `speech_final` as the turn boundary signal. `speech_final` is only `True` when
Deepgram detects an actual utterance end (via its internal voice activity detection + endpointing).
Intermediate `is_final` packets are accumulated into `_pending` and only flushed when
`speech_final` arrives.

```python
speech_final = msg.get("speech_final", False)
is_final     = msg.get("is_final",     False)

if is_final and speech_final:
    # Utterance complete — fire a turn with all accumulated text
    ...
elif is_final:
    # Committed packet but utterance continues — accumulate
    _pending.append(text)
```

**Result:** Agent only responds after complete utterances. No more premature firings.

---

### Bug 5 — LangGraph Node Name Collision

**Problem:**
LangGraph crashed at graph compilation with an error about reserved names.

**Root Cause:**
We named a graph node `"intent"`, which collided with the `intent` key in our
`ConversationState` TypedDict. LangGraph forbids node names that match state keys because
it uses them in the same namespace for routing.

**Fix (commit `0985e66`):**
Renamed the node from `"intent"` to `"classify"`:

```python
builder.add_node("classify", intent_classification_node)  # was "intent"
```

**Lesson:** Always choose node names that don't overlap with state field names in LangGraph.

---

### Bug 6 — `audioop-lts` Import Failure on Python 3.9

**Problem:**
`pip install` pulled in `audioop-lts`, which then failed to import because it's designed for
Python 3.13+ (where `audioop` was removed from stdlib). On Python 3.9–3.12, `audioop` is a
built-in C extension — installing the PyPI shim breaks it.

**Root Cause:**
A transitive dependency or careless `pip install` brought in `audioop-lts`. Since we're on
Python 3.9, `import audioop` should resolve to the stdlib module, but the installed package
shadowed it.

**Fix (commit `a7495c5`):**
Removed `audioop-lts` from requirements.txt and added a comment:

```
# audioop is built into Python stdlib on 3.9–3.12; audioop-lts only needed on 3.13+
```

**Lesson:** Be explicit about stdlib-vs-PyPI boundaries in requirements.

---

### Bug 7 — Excessive RAG Fallback on Transcription Errors

**Problem:**
Aria was routing to the fallback node ("I don't have enough information") for questions she
should have been able to answer. For example, asking about "banking fees" would get a confident
FAQ match, but slight transcription errors like "backing fees" would drop the cosine score
below the threshold and trigger fallback.

**Root Cause:**
Two compounding issues:
1. The RAG fallback threshold was set too high (0.50). STT transcription errors (which are
   inevitable in a voice pipeline) would degrade the embedding similarity just enough to
   cross the threshold.
2. In `response_generation_node`, we were filtering out chunks with `low_confidence` before
   passing them to the LLM. So even when chunks were retrieved, the LLM never saw them if
   the score was below threshold.

**Fix (commit `256b862`):**
1. Lowered `_FALLBACK_THRESHOLD` from `0.50` to `0.30` in `retriever.py`
2. Removed the score filtering in `response_generation_node` — now all retrieved chunks are
   passed to the LLM regardless of confidence score. The LLM is better than a hard threshold
   at judging whether a chunk is relevant to the query.

```python
# Before (broken):
snippets = [c["text"] for c in state["retrieved_context"]
            if c.get("text") and not c.get("low_confidence")]

# After (fixed):
snippets = [c["text"] for c in state["retrieved_context"]
            if c.get("text")]
```

**Result:** Agent now answers FAQ questions correctly even with minor transcription noise.

---

### Bug 8 — TTS Buffering Hiding Real Latency

**Problem (architectural, not a crash):**
An early version of the TTS playback code would collect ALL audio chunks from ElevenLabs into
a buffer, then play the entire audio at once via `sd.play(np.concatenate(chunks))`. This
worked functionally, but it hid the true latency from the user — Aria would go silent for
2–3 seconds while buffering, then speak the full response instantly.

**Root Cause:**
The natural instinct was to buffer-then-play for simplicity. But in a voice pipeline, time-to-
first-byte matters more than clean playback — the user perceives responsiveness by how quickly
Aria starts speaking, not how smoothly she finishes.

**Fix:**
Although the current implementation in `test_pipeline_local.py` still does buffer-then-play
(due to `sounddevice` limitations with streaming small chunks), the TTS client itself
(`elevenlabs_client.py`) yields chunks as they arrive from the streaming API. The Twilio path
(`media_stream.py`) sends each chunk to the WebSocket immediately — no buffering.

The local test file notes this as a known compromise:

```python
# Stream TTS → collect PCM frames (local playback limitation)
chunks = []
async for mulaw_chunk in _tts.synthesize(response):
    pcm = audioop.ulaw2lin(mulaw_chunk, 2)
    chunks.append(arr)
# Play all at once (buffered — not ideal, but sd.play needs full array)
sd.play(np.concatenate(chunks), samplerate=SAMPLE_RATE)
```

**Status:** Fixed for Twilio path (true streaming). Local test still buffers. v2 will use a
streaming audio queue (e.g., `pyaudio` with a ring buffer) for true local streaming.

---

### Bug 9 — Windows Terminal UTF-8 Encoding Crash

**Problem:**
The local test script crashed on Windows when Aria's response contained non-ASCII characters
(e.g., currency symbols, accented names). The terminal's default `cp1252` encoding couldn't
render them.

**Root Cause:**
Windows PowerShell and CMD use `cp1252` by default. Python's `sys.stdout` inherits this
encoding, so any `print()` with characters outside that codepage throws `UnicodeEncodeError`.

**Fix:**
Added explicit UTF-8 wrapping at the top of `test_pipeline_local.py`:

```python
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8",
        errors="replace", line_buffering=True
    )
```

**Lesson:** Always force UTF-8 on Windows for any script that prints API responses.

---

## 2. Design Architecture

### High-Level Pipeline

```
Caller (phone)
    │  PSTN/SIP
    ▼
Twilio ── Media Stream (WebSocket, μ-law 8kHz audio)
    │
    ▼
FastAPI WebSocket Handler  (Real-Time Media Server)
    │
    ├──► Deepgram (streaming STT) ──► partial + final transcripts
    │
    ▼
LangGraph Agent Orchestrator
    ├─ Intent Classification Node   (Groq, fast classify → faq/account/dispute/escalate)
    ├─ RAG Retrieval Node           (Qdrant — 20-item fintech FAQ knowledge base)
    ├─ Tool Calling Node            (mock account status + dispute lookup)
    ├─ Response Generation Node     (Groq Llama 3.3 70B, streaming)
    └─ Fallback/Escalation Node     (low confidence → graceful handoff)
    │
    ▼
ElevenLabs (streaming TTS) ──► μ-law audio chunks
    │
    ▼
Twilio Media Stream ──► Caller hears response

Cross-cutting: Barge-in (interrupt TTS if caller speaks mid-response)
Cross-cutting: Structured logging → Streamlit dashboard
```

### LangGraph State Machine

```
START → classify
         │
         ├── (faq) ──────────► rag ──┬── (low_confidence) → fallback → END
         │                           └── (ok) → generate → END
         ├── (account_query) ──► tool → generate → END
         ├── (dispute_query) ──► tool → generate → END
         ├── (escalate) ────────► fallback → END
         └── (out_of_scope) ────► fallback → END
```

### Key Design Principles

1. **Stream everything.** Every stage (STT, LLM, TTS) uses streaming APIs. No stage
   waits for a previous stage to fully complete before starting. This is how you hit a
   <1.5s round-trip target.

2. **Confidence-gated fallback.** If the RAG retrieval score is below 0.30, the agent
   doesn't hallucinate — it routes to a fallback node that says "let me connect you with
   a specialist." This prevents the LLM from inventing bank policies.

3. **Per-stage latency logging.** Every turn records STT, LLM, TTS, and total latency
   in a structured JSON log. This makes it trivially easy to identify which stage is the
   bottleneck on any given turn.

4. **Graceful degradation.** If the user asks something completely unrelated to banking,
   the intent classifier routes to `out_of_scope` → `fallback` instead of attempting an
   answer.

### Data Flow Per Turn

1. **Audio in:** Twilio sends μ-law 8kHz audio frames via WebSocket → decoded and forwarded
   to Deepgram's live transcription API.
2. **Partial transcripts:** Deepgram sends interim results → used for barge-in detection
   (if caller speaks while TTS is playing, stop playback).
3. **Final transcript:** Deepgram sends `speech_final=True` → triggers a LangGraph turn.
4. **Intent classification:** Rule-based pre-check first (regex for account/dispute IDs,
   keyword match for escalation). Falls back to Groq LLM classification for ambiguous
   queries. This saves an LLM call for 60%+ of queries.
5. **RAG or Tool:** Based on intent, either query Qdrant for FAQ chunks or call the mock
   account/dispute lookup tool.
6. **Response generation:** Groq Llama 3.3 70B streams tokens. System prompt enforces
   short (2–3 sentence), spoken-language responses with no markdown.
7. **TTS synthesis:** ElevenLabs streaming API converts text → PCM 8kHz → μ-law for Twilio.
8. **Audio out:** μ-law chunks are base64-encoded and sent back through the Twilio WebSocket
   in real time.
9. **Logging:** The entire turn is logged as a JSON-lines record with all latency metrics.

---

## 3. Technology Choices & Rationale

### Telephony: Twilio

**Why Twilio:**
- Industry standard for PSTN access from code. Buy a phone number for $1/month, configure
  a webhook, and you're receiving calls in 10 minutes.
- Media Streams API provides raw audio over WebSocket — gives us full control over the
  audio pipeline instead of being locked into Twilio's TwiML `<Gather>` / `<Say>` which
  doesn't support real-time streaming.
- Excellent documentation and Python SDK.

**Why NOT alternatives:**
| Alternative | Why Rejected |
|---|---|
| **Vonage / Nexmo** | Smaller ecosystem, fewer examples for raw audio streaming |
| **Plivo** | Less mature WebSocket media streaming support |
| **Telnyx** | Good alternative, but Twilio has more community resources for debugging |
| **Direct SIP** | Too much infra overhead for a portfolio project |

---

### STT: Deepgram (Nova 2)

**Why Deepgram:**
- Streaming API with sub-300ms partial transcripts — critical for real-time voice.
- `speech_final` / `UtteranceEnd` events give us proper turn boundary detection (not just
  raw word chunks).
- Free tier: 200 hours/month — more than enough for development and demos.
- Nova 2 is best-in-class for accuracy at this latency tier.

**Why NOT alternatives:**
| Alternative | Why Rejected |
|---|---|
| **Google Cloud STT** | Higher latency for streaming, more complex auth setup, overshoot for a portfolio project |
| **AWS Transcribe** | Minimum 1s latency for streaming, no free tier without AWS account overhead |
| **Azure Speech** | Good streaming, but SDK is heavier and pricing less transparent for dev usage |
| **OpenAI Whisper** | No streaming API — batch only. Whisper requires full audio → full transcript. Unusable for real-time. |
| **AssemblyAI** | Good alternative, but Deepgram was faster for streaming and had a more generous free tier at the time |

---

### TTS: ElevenLabs (Turbo v2)

**Why ElevenLabs:**
- Streaming HTTP endpoint — start receiving audio within 200-400ms of the API call.
- Natural-sounding voices that don't sound robotic on a phone call.
- `eleven_turbo_v2` model optimized for low latency, not just quality.
- 10,000 characters/month free — sufficient for development.

**Why NOT alternatives:**
| Alternative | Why Rejected |
|---|---|
| **Google Cloud TTS** | Batch-only for high-quality voices (WaveNet/Journey). Streaming exists but quality is significantly lower. |
| **Amazon Polly** | Decent streaming, but voice quality feels synthetic for a conversational agent |
| **Azure Neural TTS** | Good quality, but streaming API is more complex to integrate |
| **OpenAI TTS** | No streaming endpoint at the time of building. Good quality but batch-only. |
| **Cartesia (Sonic)** | Emerging competitor with ultra-low latency. Wasn't mature enough when we started. Worth evaluating for v2. |
| **XTTS (Coqui)** | Open-source / self-hosted, but inference latency on CPU is too high for real-time. Needs GPU. |

---

### LLM: Groq (Llama 3.3 70B)

**Why Groq:**
- **Speed.** Groq's custom LPU hardware delivers 500–800 tokens/sec for Llama 3.3 70B.
  Typical first-token latency is 100–200ms. This is 3–5x faster than running the same
  model on GPU cloud.
- **Free tier.** No cost during development — critical for a portfolio project where you're
  iterating hundreds of times.
- **Model quality.** Llama 3.3 70B is competitive with GPT-4-turbo for classification and
  short-form generation. For 2-3 sentence voice responses, it's more than sufficient.

**Why NOT alternatives:**
| Alternative | Why Rejected |
|---|---|
| **OpenAI GPT-4 / GPT-4o** | Higher quality, but 2–3x slower for first token. Costs money per call. For a voice agent, latency matters more than marginal quality. |
| **Anthropic Claude** | Same latency/cost concern as OpenAI. Overkill for 2-sentence banking responses. |
| **OpenAI GPT-3.5-turbo** | Faster and cheaper than GPT-4, but Groq + Llama 3.3 70B is both faster AND free. No reason to choose GPT-3.5. |
| **Local Ollama** | Inference on a local machine (without dedicated GPU) is 5–20x slower. Real-time voice requires sub-600ms LLM response. |
| **Together AI** | Good Llama hosting, but Groq is faster at inference. |
| **Fireworks AI** | Comparable speed to Groq, but Groq's free tier was more generous. |

---

### Orchestration: LangGraph

**Why LangGraph:**
- The call flow is inherently a state machine: classify → retrieve/tool → generate → respond.
  LangGraph models this directly as a graph with conditional edges.
- Built on LangChain — reuse `ChatGroq`, `HumanMessage`, etc. without adapter code.
- Supports `stream()` natively for token-by-token LLM streaming.
- TypedDict-based state (`ConversationState`) gives us type safety and IDE autocomplete
  across all nodes.

**Why NOT alternatives:**
| Alternative | Why Rejected |
|---|---|
| **Raw LangChain (Chains)** | Chains are linear — our flow has branching (intent → 3 paths). Would require awkward `if/else` inside a single chain. |
| **Vanilla Python (no framework)** | Works for v1 complexity, but adding confirmation nodes, multi-step tools, and conversation memory in v2 would require reinventing state management. LangGraph scales better. |
| **AutoGen / CrewAI** | Multi-agent frameworks. Overkill — we have one agent, not a team. Adds latency from agent coordination loops. |
| **Haystack** | Good for RAG pipelines, but less natural for branching agent flows. |

---

### Vector DB: Qdrant (Local Docker)

**Why Qdrant:**
- Simple to run locally: `docker compose up -d` and it's ready.
- Python client with straightforward `search()` API.
- Cosine similarity search with score thresholds for confidence gating.
- Persistent storage (survives container restarts).
- Zero cost — runs locally.

**Why NOT alternatives:**
| Alternative | Why Rejected |
|---|---|
| **Pinecone** | Cloud-hosted — adds network latency to RAG retrieval. Free tier has limits. Overkill for 20 FAQ items. |
| **Weaviate** | Good, but heavier Docker footprint and more complex schema setup. |
| **ChromaDB** | In-process (no Docker needed), but less production-like. We wanted to practice with a proper vector DB. |
| **FAISS** | In-memory only, no persistence by default. Good for notebooks, not for a server that restarts. |
| **pgvector** | Requires PostgreSQL. We don't need a relational DB — adding one just for vectors is overhead. |

---

### Embeddings: `all-MiniLM-L6-v2`

**Why this model:**
- 384-dimensional embeddings — small, fast to encode, low memory.
- Runs locally on CPU in <50ms per query — no API call needed for embedding.
- Well-suited for short text (FAQ questions and answers are 1–2 sentences).
- Free and open-source.

**Why NOT alternatives:**
| Alternative | Why Rejected |
|---|---|
| **OpenAI `text-embedding-3-small`** | Costs money per embedding. Adds network latency. For 20 FAQs, local encoding is faster. |
| **Cohere Embed** | Same cost/latency argument. |
| **`all-mpnet-base-v2`** | Better quality (768d), but 2x slower to encode and 2x more memory. Marginal gain for short FAQ text. |

---

### Backend: FastAPI

**Why FastAPI:**
- Native WebSocket support — critical for Twilio Media Streams and Deepgram.
- Async-first (`async def` handlers) — the entire voice pipeline is I/O-bound (waiting for
  Deepgram, Groq, ElevenLabs). Async is the natural fit.
- Auto-generated OpenAPI docs for webhook endpoints.
- The standard for modern Python APIs.

**Why NOT alternatives:**
| Alternative | Why Rejected |
|---|---|
| **Flask** | No native async. No native WebSocket. Would need Flask-SocketIO + eventlet — fragile combo. |
| **Django** | Too heavyweight. Channels for WebSocket adds complexity. ORM/admin is wasted here. |
| **Node.js / Express** | Language mismatch — LangGraph, sentence-transformers, and the ML ecosystem are Python-native. |

---

### Dashboard: Streamlit

**Why Streamlit:**
- Build a functional analytics dashboard in <100 lines of Python.
- Reads directly from `logs/calls.jsonl` — no database needed.
- Good enough for a demo video and portfolio screenshots.
- Free.

**Why NOT alternatives:**
| Alternative | Why Rejected |
|---|---|
| **React** | 10x more code for the same dashboard. Not worth the time for a demo. |
| **Grafana** | Good for production monitoring, but requires InfluxDB/Prometheus setup. Overkill. |
| **Gradio** | More suited for ML model demos than analytics dashboards. |

---

### SDK Versions: Deepgram v2 (Not v3)

**Why v2:**
- Python 3.9 compatible. Deepgram's v3 SDK (`deepgram-sdk >= 3.0`) requires Python 3.10+.
- v2 uses `Deepgram()` client with `registerHandler` — callback-based, synchronous-feeling
  API that maps well to our audio pump architecture.

**Note for v2 (of VoiceFlow):** If we upgrade to Python 3.10+, we should migrate to Deepgram
SDK v3 which has a cleaner async API with `UtteranceEnd` events as first-class citizens.

---

## 4. Latency Observations

### Target vs Actual (local test, approximate)

| Stage | Target | Observed (typical) | Notes |
|---|---|---|---|
| STT (first audio → final transcript) | <300ms | 200–500ms | After Bug 3 fix. Depends on utterance length. |
| LLM response (Groq) | <600ms | 300–600ms | Groq is consistently fast. Occasional 800ms spikes. |
| TTS first chunk (ElevenLabs) | <400ms | 250–500ms | `optimize_streaming_latency=3` helps. |
| **Total round-trip** | **<1.5s** | **1.0–2.0s** | Meets target on most turns. Spikes on first turn (cold start). |

### Cold Start Penalty

First turn after boot has additional overhead:
- SentenceTransformer model load: ~2–3s
- Deepgram WebSocket handshake: ~200ms
- Qdrant first query: ~100ms

Subsequent turns don't pay this cost (singletons are warm).

---

## 5. Lessons Learned for v2

### Carry Forward (Don't Regress)

1. **Mic gating during playback** — `_mic_callback` must not enqueue audio while SPEAKING.
2. **`speech_final` turn detection** — never use raw `is_final` for turn boundaries.
3. **Streaming TTS** — play chunks as they arrive, don't buffer full response.
4. **STT timer accuracy** — measure from last-audio-sent to final-transcript-received.
5. **Generous RAG threshold** — 0.30, not 0.50. Let the LLM judge relevance, not a hard cutoff.
6. **UTF-8 stdout on Windows** — always force it.

### Things to Improve in v2

1. **True streaming playback locally** — replace `sd.play(concatenated)` with a ring-buffer
   approach (e.g., `pyaudio` callback) so the user hears audio as chunks arrive.
2. **Deepgram SDK v3** — if we move to Python 3.10+, use the modern async client with
   native `UtteranceEnd` events.
3. **Real barge-in with VAD** — instead of half-duplex mic gating, run a lightweight VAD
   in parallel during SPEAKING to detect user interruption and stop TTS within ~300ms.
4. **Wake word replaces Twilio** — v2 is a local assistant, not a phone agent. Porcupine
   replaces Twilio as the session trigger.
5. **Confirmation node** — before any real-world action (Swiggy order, etc.), require
   explicit verbal confirmation. Never spend money without a "yes."

---

*End of v1 development log. Ready for v2.*

---

## 6. V2 Bottlenecks & Fixes (Assistant Mode & Swiggy MCP)

### Bug 1 — Async Callback Crash in Deepgram SDK v3

**Problem:**
Deepgram silently closed its connection, waiting for audio, because the background task to listen for the query crashed with `TypeError: a coroutine was expected, got None`. The mic never started listening.

**Root Cause:**
Deepgram v3 SDK uses `asyncio.create_task` inside its `_emit` handler for callbacks. We mistakenly ran `asyncio.create_task(self.listen_for_query())` from a synchronous loop context and didn't wait or retrieve exceptions properly, so the task crashed silently and the error was swallowed.

**Fix:**
Renamed the method to `start_deepgram` and attached a done-callback: `task.add_done_callback(lambda t: t.exception() and logger.error(...))`. This exposed the underlying error.

### Bug 2 — Wake-Word Audio Bleeding into STT

**Problem:**
STT latencies appeared as 34s long. The wake word "Hey Jarvis" audio tail was fed into Deepgram, which then waited for more speech that didn't arrive, delaying endpointing.

**Root Cause:**
Transition from `IDLE` to `LISTENING` was so fast that physical trailing audio in the PyAudio buffer was pumped to Deepgram.

**Fix:**
Added `await asyncio.sleep(0.5)` and a secondary queue flush loop before feeding Deepgram in `LISTENING` state.

### Bug 3 — Groq Token Rate Limits (HTTP 413) with Swiggy MCP

**Problem:**
The Swiggy node crashed with `unhandled errors in a TaskGroup (1 sub-exception)` which was a `groq.APIStatusError: Error code 413`.

**Root Cause:**
Swiggy MCP exposes 18 complex tools. Injecting all 18 tool schemas into LangChain consumed 12,349 tokens, completely exceeding `llama-3.1-8b-instant`'s 6,000 TPM limit.

**Fix:**
Filtered the toolset before passing to the ReAct agent, leaving only 6 essential tools (Cart, Search, Order). Upgraded to `llama-3.3-70b-versatile` which has a 12,000 TPM limit.

### Bug 4 — Missing State Context for Confirmation (Hallucination Loop)

**Problem:**
Saying "I would like to buy a chicken" routed to the `faq` node instead of Swiggy. The FAQ node asked "Would you like to order via Swiggy?", but saying "Yes" routed to `out_of_scope`.

**Root Cause:**
`intent_classification_node` relied on strict keywords. Because "buy" and "chicken" weren't in the list, it routed to `faq`. Since `faq` is handled by `response_generation_node`, it didn't set `pending_action`, leaving the graph with no context for "Yes".

**Fix:**
Expanded the heuristic keyword list in `intent_classification_node` to include "buy", "chicken", "roll", etc., routing direct food actions to the `action_intent` node.

### Bug 5 — Groq Schema Type Validation (String vs Int)

**Problem:**
Groq threw a 400 error on the Swiggy `search_restaurants` tool call because it generated `"0"` (string) for the `offset` parameter, which the MCP schema expects as an integer.

**Fix:**
Appended a strict prompt instruction directly into the Swiggy ReAct agent: `CRITICAL: When calling tools ... ensure numeric parameters like 'offset' are passed as raw JSON integers (e.g. 0), NOT strings`.
