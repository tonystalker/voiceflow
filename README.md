# 🎙️ VoiceFlow AI (Tony)

> Real-time AI voice assistant for automated food ordering — built 100% locally with wake-word detection and MCP tool integration.

**Stack:** openWakeWord · Deepgram · ElevenLabs · LangGraph · Groq · MCP (Swiggy) · FastAPI · Streamlit

---

## Architecture

```
Caller (Microphone)
    │
    ▼
openWakeWord ("Hey Jarvis") ──► Triggers LISTENING state
    │
    ▼
Deepgram (streaming STT v3) ──► partial + final transcripts
    │
    ▼
LangGraph Agent Orchestrator (Tony)
    ├─ Intent Classification Node   (Groq, fast classify → action_intent/faq/etc)
    ├─ Confirmation Node            (Requires explicit confirmation for actions)
    ├─ Swiggy MCP Tool Node         (Model Context Protocol integration for food ordering)
    ├─ Response Generation Node     (Groq Llama 3.3 70B, streaming)
    └─ Fallback/Escalation Node     (graceful handoff)
    │
    ▼
ElevenLabs (streaming TTS) ──► μ-law audio chunks
    │
    ▼
Speaker ──► Caller hears response

Cross-cutting: Barge-in (webrtcvad interrupts TTS if caller speaks mid-response)
Cross-cutting: Structured logging → Streamlit dashboard
```

---

## What Makes This Production-Aware

| Feature | Implementation |
|---|---|
| **Streaming at every stage** | Deepgram `interim_results`, Groq `stream=True`, ElevenLabs `/stream` endpoint — no stage buffers the full response |
| **Real Barge-in** | webrtcvad actively monitors the microphone during TTS playback and stops the stream immediately when speech is detected. |
| **Model Context Protocol (MCP)** | Fully integrated with LangChain MCP Adapters to interact with the Swiggy API, utilizing OAuth PKCE for secure authentication. |
| **Action Confirmation** | Graph state machine ensures the agent pauses and asks for explicit confirmation ("Shall I go ahead?") before executing real-world tool actions. |
| **Structured turn logs** | JSON-lines log per turn: transcript, intent, response, latency |

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

### 4. Swiggy Auth

```bash
uv run -m app.mcp.setup_auth
# Output: Will open a browser window to authenticate with Swiggy
```

### 5. Run the Local Assistant

```bash
uv run main.py
# Speak "Hey Jarvis" into your mic to wake Tony up.
# Try saying: "Can you order some biryani from a nearby restaurant?"
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
