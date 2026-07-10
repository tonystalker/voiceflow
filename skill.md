# VoiceFlow AI — Implementation Plan

Voice agent for automated customer service calls: STT → LLM reasoning + RAG → TTS, over a real phone line.

---

## 1. Architecture

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
    ├─ Turn Detection Node        (silence threshold → "user finished speaking")
    ├─ Intent Classification Node (cheap/fast model)
    ├─ RAG Retrieval Node         (Qdrant knowledge base)
    ├─ Response Generation Node   (Groq-hosted LLM)
    ├─ Tool Calling Node          (mock order-status lookup)
    └─ Fallback/Escalation Node   (low confidence → human handoff message)
    │
    ▼
ElevenLabs (streaming TTS) ──► audio chunks
    │
    ▼
Twilio Media Stream ──► Caller hears response

Cross-cutting: Barge-in handler (interrupt TTS playback if caller starts speaking)
Cross-cutting: Structured logging → Streamlit dashboard (call logs, transcripts, intents)
```

**Design principle for v1:** build the pipeline yourself (Twilio + Deepgram + ElevenLabs) rather than a platform like Vapi/LiveKit. Slower to start, but it's what actually demonstrates systems understanding to a founder reading your repo.

---

## 2. Tech Stack

| Layer | Choice | Notes |
|---|---|---|
| Orchestration | LangGraph | State machine maps naturally to call flow |
| Telephony | Twilio | Buy 1 number (~$1), use Media Streams |
| STT | Deepgram | Streaming, low latency, free tier |
| TTS | ElevenLabs | 10K free credits/mo, streaming API |
| LLM | Groq (Llama 3.3 70B or GPT-OSS) | Free, fast — latency matters more than raw quality here |
| Vector DB | Qdrant (local Docker) | Reuse pattern from Flowdesk |
| Backend | FastAPI | Webhooks + WebSocket audio |
| Dashboard | Streamlit | Call logs, transcripts, intent breakdown |
| Infra | Docker Compose (local) | No k8s/autoscaling until a real deployment need exists |

---

## 3. Prerequisites / Accounts to Set Up (Day 0)

- [ ] Twilio account + phone number purchased
- [ ] Deepgram API key (free tier)
- [ ] ElevenLabs API key (free tier, 10K credits)
- [ ] Groq API key (free)
- [ ] Docker installed (for Qdrant)
- [ ] `ngrok` or similar (to expose local FastAPI server to Twilio during dev)

---

## 4. Repo Structure

```
voiceflow-ai/
├── app/
│   ├── main.py                  # FastAPI entrypoint
│   ├── telephony/
│   │   ├── twilio_webhook.py    # Incoming call webhook, TwiML response
│   │   └── media_stream.py      # WebSocket handler for audio in/out
│   ├── stt/
│   │   └── deepgram_client.py   # Streaming STT wrapper
│   ├── tts/
│   │   └── elevenlabs_client.py # Streaming TTS wrapper
│   ├── agent/
│   │   ├── graph.py             # LangGraph definition
│   │   ├── nodes.py             # Individual node functions
│   │   └── state.py             # Conversation state schema
│   ├── rag/
│   │   ├── ingest.py            # Load knowledge base into Qdrant
│   │   └── retriever.py         # Query wrapper
│   ├── tools/
│   │   └── order_status.py      # Mock business tool
│   └── logging/
│       └── call_logger.py       # Structured logs for dashboard
├── dashboard/
│   └── streamlit_app.py
├── data/
│   └── faq_knowledge_base.json  # Sample company FAQ content
├── tests/
│   └── test_pipeline_local.py   # Day 1-2 local loop test (no phone)
├── docker-compose.yml           # Qdrant service
├── requirements.txt
├── .env.example
└── README.md
```

---

## 5. Phased Build Plan

### Day 1–2: Local pipeline, no phone
**Goal:** prove the loop works before adding Twilio complexity.

- Script: mic input → Deepgram → Groq (LangChain call) → ElevenLabs → speaker output
- Measure round-trip latency at each stage, log it
- Exit criteria: you can have a spoken back-and-forth locally with <2s response lag

### Day 3–4: Twilio integration
**Goal:** same pipeline, now over a real phone call.

- Buy Twilio number, configure webhook URL (via ngrok in dev)
- Implement `twilio_webhook.py` → returns TwiML that opens a Media Stream
- Implement `media_stream.py` → FastAPI WebSocket receiving μ-law audio chunks from Twilio, decoding, forwarding to Deepgram
- Send ElevenLabs audio back through the same WebSocket to Twilio
- Exit criteria: call the Twilio number, hear a canned response back

### Day 5: RAG layer
**Goal:** agent answers from real knowledge, not just LLM improvisation.

- Write `data/faq_knowledge_base.json` (10–20 sample Q&A pairs for a fictional or real company)
- `ingest.py`: chunk + embed + load into Qdrant
- `retriever.py`: similarity search wrapper
- Wire retrieval into the LangGraph flow before the response generation node
- Exit criteria: agent correctly answers an FAQ that's only in the knowledge base, not general knowledge

### Day 6: Intent routing, tools, turn detection, fallback
**Goal:** the parts that separate a toy from something demo-worthy.

- Turn detection: silence-threshold logic to detect "caller finished speaking" (avoid cutting them off)
- Barge-in handling: if caller speaks while TTS is playing, stop playback and listen
- Intent classification node: route between "FAQ answer," "order status check," "escalate to human"
- Mock tool: `order_status.py` — fake DB lookup by order ID
- Fallback node: if RAG retrieval score is low or LLM is unsure, respond with a graceful handoff line instead of hallucinating
- Exit criteria: you can interrupt the agent mid-sentence and it responds correctly; an out-of-scope question triggers fallback instead of a made-up answer

### Day 7: Dashboard + demo polish
**Goal:** what you screen-record and put in the cold email.

- Streamlit dashboard: call list, transcript viewer, intent breakdown chart
- Record a 60–90s demo video: call the number, ask an FAQ, ask an order status, ask something out of scope (show the fallback), interrupt it once (show barge-in)
- Write README with the architecture diagram (based on the one you already have) and a "what makes this production-aware" section covering: streaming at every stage, barge-in, fallback design, latency logging

---

## 6. Latency Budget (keep it in view while building)

| Stage | Target |
|---|---|
| STT (partial transcript) | <300ms |
| LLM response generation | <600ms |
| TTS first audio chunk | <400ms |
| **Total round trip** | **<1.5s** |

If you're not streaming (i.e. waiting for full STT transcript, full LLM response, full TTS audio before moving to the next stage), you will blow this budget. Stream everything.

---

## 7. What to Cut for v1 (don't overbuild)

- No Kubernetes / autoscaling — Docker Compose locally is enough
- No real CRM/payment integration — one mock tool is enough to prove the pattern
- No React frontend — Streamlit is enough for the demo
- No LiveKit/Vapi — build the pipeline yourself for the portfolio signal

---

## 8. Post-v1 (only if a founder wants to pilot it)

- Swap mock tool for a real API integration (their CRM/booking system)
- Add proper autoscaling/observability if call volume requires it
- Multi-language support if relevant to their customer base

v2 plan 

# VoiceFlow AI → Personal Assistant — Implementation Plan v2

Pivot from phone-based customer service agent to a local, always-on, wake-word-triggered
personal assistant with real-world action capability (Swiggy: food, groceries, dining).

This plan assumes the v1 pipeline already works: mic → Deepgram STT → LangGraph → RAG →
LLM → ElevenLabs TTS → speaker, with dashboard, call logger, intent routing, fallback
node, and mock tool calling all built and running locally.

---

## 1. What's Changing

| | v1 (customer service) | v2 (personal assistant) |
|---|---|---|
| Trigger | Twilio inbound call | Local wake word ("Hey Aria") |
| Channel | Phone (PSTN) | Mic/speaker on your machine |
| Mode | Half-duplex, one call = one session | Always-on idle loop, repeated wake/sleep cycles |
| Tools | Mock order-status lookup | Real-world actions via Swiggy MCP (order food, groceries, book table) |
| Cost model | Twilio number + per-minute | $0 — wake word runs 100% local, cloud APIs only fire after wake |
| Twilio/ngrok | Required | Parked — optional future feature ("call home") |

---

## 2. Updated Architecture

```
                    ┌────────────────────────────┐
                    │   IDLE                      │
                    │   Porcupine wake-word engine │  ← runs locally, no network calls,
                    │   listening for "Hey Aria"   │     ~0 ongoing cost
                    └──────────────┬──────────────┘
                                   │ wake word detected
                                   ▼
                    ┌────────────────────────────┐
                    │   LISTENING                 │
                    │   Deepgram streaming STT     │  ← mic gated OFF during playback
                    │   UtteranceEnd → turn end     │     (echo fix from v1 debugging)
                    └──────────────┬──────────────┘
                                   │ final transcript
                                   ▼
                    ┌────────────────────────────┐
                    │   THINKING (LangGraph)       │
                    │   Intent Classification       │
                    │     ├─ faq        → RAG (Qdrant)
                    │     ├─ order_status → mock tool
                    │     └─ action_intent → Swiggy MCP tool node
                    │            └─ CONFIRMATION NODE (required before
                    │               any checkout/booking tool call fires)
                    └──────────────┬──────────────┘
                                   │ response text
                                   ▼
                    ┌────────────────────────────┐
                    │   SPEAKING                   │
                    │   ElevenLabs streaming TTS     │  ← play-as-chunks-arrive
                    │   (barge-in listener active)    │     (v1 fix: don't buffer
                    └──────────────┬──────────────┘        full response first)
                                   │ done / interrupted
                                   ▼
                              back to IDLE
```

---

## 3. Tech Stack Additions

| Layer | Choice | Notes |
|---|---|---|
| Wake word | **Porcupine (Picovoice)** | Free personal-use tier, custom "Hey Aria" trainable in console, low CPU |
| MCP client | **langchain-mcp-adapters** | Officially supported LangGraph integration path for MCP servers |
| Action provider | **Swiggy MCP** (`food`, `instamart`, `dineout` servers) | OAuth 2.1 + PKCE, phone/OTP auth, free on localhost during dev |
| Auth flow | One-time browser OAuth via setup script | Not per-call — this is personal use, token persists |

Everything else (Deepgram, ElevenLabs, Groq, Qdrant, FastAPI, Streamlit) is unchanged from v1.

---

## 4. Repo Structure Additions

```
voiceflow-ai/
├── app/
│   ├── wakeword/
│   │   └── porcupine_listener.py   # NEW — idle-loop wake detection
│   ├── assistant/
│   │   └── state_machine.py        # NEW — IDLE/LISTENING/THINKING/SPEAKING loop
│   ├── mcp/
│   │   ├── swiggy_client.py        # NEW — langchain-mcp-adapters wrapper
│   │   └── setup_auth.py           # NEW — one-time OAuth browser flow, saves token
│   ├── agent/
│   │   ├── graph.py                # UPDATED — add action_intent branch
│   │   ├── nodes.py                # UPDATED — add confirmation_node
│   │   └── state.py                # UPDATED — add pending_action / confirmed fields
├── main.py                         # NEW — replaces test_pipeline_local.py as entrypoint
```

---

## 5. Phased Build Plan

### Phase A: Wake-word idle loop
- `pip install pvporcupine pvrecorder`
- Get free Picovoice access key, train/select "Hey Aria" wake word in console
- `porcupine_listener.py`: standalone script, prints "wake word detected!" — verify in isolation before touching the pipeline
- Exit criteria: reliably detects wake word across a room, near-zero false positives during normal speech

### Phase B: State machine refactor
- Wrap your existing mic→STT→LangGraph→TTS→speaker loop (from `test_pipeline_local.py`) inside the IDLE/LISTENING/THINKING/SPEAKING state machine
- IDLE: only Porcupine running, everything else dormant (no Deepgram connection open, no API cost)
- On wake: open Deepgram connection, transition to LISTENING
- Reuse the mic-gating fix (don't capture mic during SPEAKING) and `UtteranceEnd`-based turn detection from v1 debugging — carry those fixes forward, don't regress
- Exit criteria: say "Hey Aria," ask a question, get a spoken answer, system returns to idle listening for the wake word again

### Phase C: Swiggy MCP integration (build/test as text first, before wiring into voice)
- Read Swiggy MCP developer quickstart, run steps 1–5 locally (free, no production access needed yet)
- `setup_auth.py`: one-time script, opens browser OAuth+PKCE flow, phone/OTP, saves session token locally
- `swiggy_client.py`: load Swiggy's MCP tools via `langchain-mcp-adapters`, bind to your existing LLM node the same way as any other tool
- Add `action_intent` branch to intent classification node
- Test with a text-only harness first (skip mic/TTS) — validate `search_restaurants` → `get_addresses` → order flow works end-to-end via the LLM before adding voice on top
- Exit criteria: via text input, agent can search a restaurant, pick an item, and reach the checkout step (not yet placing real orders)

### Phase D: Confirmation node (safety-critical, do not skip)
- New LangGraph node: sits between "user requested an action" and "tool actually executes"
- Reads back the concrete action in plain language ("2 butter naan and dal makhani from X, ₹340, COD — should I place it?")
- Requires an unambiguous affirmative before the checkout/booking tool call fires
- Any ambiguity ("uh maybe," silence, unclear response) → do NOT execute, ask again or abandon
- Exit criteria: agent never places a real order without an explicit yes; test with an intentionally ambiguous response to confirm it does NOT proceed

### Phase E: Real barge-in
- Currently half-duplex (mic gated off during SPEAKING) — upgrade to true interrupt
- While SPEAKING: keep a lightweight VAD (voice activity detector) listening in parallel; if user speech is detected, immediately stop TTS playback and transition to LISTENING
- This matters more here than it did for the phone-call version — a wake-word assistant that can't be interrupted mid-sentence feels broken fast
- Exit criteria: interrupt the agent mid-response, it stops talking within ~300ms and starts listening

### Phase F: Deepgram endpointing tuning
- Set `endpointing: 300`, `utterance_end_ms: "1000"`, `vad_events: True` (carried over from v1 fix)
- Confirms STT wait drops from the earlier 10s+ readings to a true 1–2s
- Exit criteria: "STT latency" measured correctly (from last audio chunk sent → final transcript), not from mic-reopen to final

### Phase G: Polish + demo
- Streamlit dashboard: extend to show assistant sessions (not "calls") — wake events, intents handled, actions taken/confirmed
- Record a 60–90s demo: wake word → ask an FAQ → order food end-to-end (with confirmation) → interrupt it mid-sentence to show barge-in
- README: reframe from "answers FAQs, routes to human" → "personal assistant that takes real-world actions, with explicit confirmation before spending money" — this is a stronger signal for founder outreach than a pure FAQ bot

---

## 6. Fixes Carried Forward From v1 Debugging (don't regress these)

1. **Mic gating during playback** — `_mic_callback` must not enqueue audio while `_busy`/SPEAKING is true, or Deepgram transcribes your own TTS output as user speech
2. **`UtteranceEnd`-based turn detection**, not raw `is_final` — Deepgram finalizes fragments mid-sentence; only `UtteranceEnd` (or `speech_final`) means the user actually stopped talking
3. **Streaming TTS playback** — play audio chunks as they arrive, don't buffer the full response before calling `sd.play()`; the old pattern hides real latency behind a misleading "first chunk" timestamp
4. **Latency measurement** — measure STT latency from last-audio-sent to final-transcript-received, not from mic-reopen (which includes think-time)

---

## 7. References

**Wake word**
- Porcupine (Picovoice): https://picovoice.ai/platform/porcupine/
- openWakeWord (fully open-source alternative): https://github.com/dscripka/openWakeWord

**STT**
- Deepgram streaming/live docs: https://developers.deepgram.com/docs/live-streaming-audio
- Deepgram endpointing & UtteranceEnd: https://developers.deepgram.com/docs/endpointing
- Deepgram Python SDK (v3+, recommended over the deprecated v2 client): https://github.com/deepgram/deepgram-python-sdk

**TTS**
- ElevenLabs streaming API: https://elevenlabs.io/docs/api-reference/streaming

**LLM**
- Groq API docs: https://console.groq.com/docs

**Orchestration / MCP**
- LangGraph docs: https://langchain-ai.github.io/langgraph/
- langchain-mcp-adapters: https://github.com/langchain-ai/langchain-mcp-adapters
- Model Context Protocol spec: https://modelcontextprotocol.io/

**Swiggy MCP**
- Developer quickstart: https://mcp.swiggy.com/builders/docs/start/developer/

**Vector DB**
- Qdrant docs: https://qdrant.tech/documentation/

---

## 8. What to Still Skip for v2 (don't overbuild)

- No Kubernetes/autoscaling — this runs on your machine
- No multi-user auth — single personal token via one-time OAuth setup
- No React frontend — Streamlit remains sufficient
- Twilio/phone access stays parked — revisit only if "call home" becomes a demo priority