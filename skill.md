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