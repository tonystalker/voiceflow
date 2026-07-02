# 🎙️ VoiceFlow AI

> Real-time AI voice agent for automated banking support calls — built from first principles without Vapi or LiveKit.

**Stack:** Twilio · Deepgram · ElevenLabs · LangGraph · Groq · Qdrant · FastAPI · Streamlit

---

## Architecture

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

---

## What Makes This Production-Aware

| Feature | Implementation |
|---|---|
| **Streaming at every stage** | Deepgram `interim_results`, Groq `stream=True`, ElevenLabs `/stream` endpoint — no stage buffers the full response |
| **Barge-in interruption** | Deepgram partial transcript fires `tts.stop()` mid-stream; caller can interrupt Aria at any time |
| **Confidence-gated fallback** | RAG cosine score < 0.50 → routes to fallback node instead of hallucinating |
| **Per-stage latency logging** | STT / LLM / TTS / total all logged per turn; viewable in dashboard |
| **Structured turn logs** | JSON-lines log per turn: call SID, transcript, intent, response, latency, escalation flag |
| **Graceful escalation** | Low-confidence or explicit "speak to agent" → clean handoff line, not a confused response |

**Latency budget (target):**

| Stage | Target |
|---|---|
| STT (partial) | < 300 ms |
| LLM response | < 600 ms |
| TTS first chunk | < 400 ms |
| **Total round-trip** | **< 1.5 s** |

---

## Setup

### 1. Accounts & API Keys

| Service | Free Tier | Link |
|---|---|---|
| Twilio | $15 trial credit | [twilio.com](https://twilio.com) |
| Deepgram | 200 hrs/mo | [deepgram.com](https://deepgram.com) |
| ElevenLabs | 10K chars/mo | [elevenlabs.io](https://elevenlabs.io) |
| Groq | Free | [console.groq.com](https://console.groq.com) |

### 2. Clone & configure

```bash
git clone https://github.com/yourname/voiceflow-ai
cd voiceflow-ai
cp .env.example .env
# Fill in your API keys in .env
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Start Qdrant

```bash
docker compose up -d
```

### 5. Ingest the knowledge base

```bash
python -m app.rag.ingest
# Output: Ingested 20 FAQ items into Qdrant ✓
```

### 6. Day 1–2: Local pipeline test (no phone needed)

```bash
python tests/test_pipeline_local.py
# Speak into your mic → hear Aria respond → latency printed per stage
```

### 7. Day 3+: Full Twilio integration

```bash
# Terminal 1 — Start the server
uvicorn app.main:app --reload --port 8000

# Terminal 2 — Expose to Twilio via ngrok
ngrok http 8000

# Copy the ngrok HTTPS URL → update PUBLIC_BASE_URL in .env
# In Twilio console:
#   Phone Numbers → your number → Voice → Webhook URL:
#   https://<ngrok-url>/incoming-call  [HTTP POST]
```

Call your Twilio number — you should hear Aria greet you.

### 8. Dashboard

```bash
streamlit run dashboard/streamlit_app.py
# Opens at http://localhost:8501
```

---

## Project Structure

```
voiceflow-ai/
├── app/
│   ├── main.py                  # FastAPI entrypoint
│   ├── config.py                # Centralised settings (pydantic-settings)
│   ├── telephony/
│   │   ├── twilio_webhook.py    # POST /incoming-call → TwiML
│   │   └── media_stream.py      # WS /media-stream — audio I/O + barge-in
│   ├── stt/
│   │   └── deepgram_client.py   # Streaming STT (μ-law → transcripts)
│   ├── tts/
│   │   └── elevenlabs_client.py # Streaming TTS → μ-law chunks
│   ├── agent/
│   │   ├── graph.py             # LangGraph state machine
│   │   ├── nodes.py             # Intent / RAG / Tool / Generate / Fallback
│   │   └── state.py             # ConversationState TypedDict
│   ├── rag/
│   │   ├── ingest.py            # Embed + load FAQ into Qdrant
│   │   └── retriever.py         # Cosine search + confidence threshold
│   ├── tools/
│   │   └── account_lookup.py    # Mock account + dispute lookup tool
│   └── logging/
│       └── call_logger.py       # JSON-lines turn logger
├── dashboard/
│   └── streamlit_app.py         # Analytics dashboard
├── data/
│   └── faq_knowledge_base.json  # 20-item fintech FAQ (Qdrant source)
├── tests/
│   └── test_pipeline_local.py   # Mic → agent → speaker (no phone)
├── docker-compose.yml           # Qdrant
├── requirements.txt
├── .env.example
└── README.md
```

---

## Knowledge Base (Fintech / Banking)

20 Q&A pairs covering:
- Password reset & PIN management
- Lost/stolen card blocking
- International transfers & SWIFT codes
- KYC verification documents
- Fraud reporting & dispute filing
- Account limits & overdraft
- Savings interest rates
- Joint accounts
- Standing orders
- Deposit insurance

---

## Post-v1 Roadmap

- [ ] Swap mock tool for real CRM/core banking API integration
- [ ] Add proper autoscaling (Kubernetes + horizontal pod autoscaler)
- [ ] Multi-language support (Deepgram + ElevenLabs both support 30+ languages)
- [ ] Call recording & PII redaction pipeline
- [ ] Sentiment analysis node (flag distressed callers for priority escalation)

---

## License

MIT
