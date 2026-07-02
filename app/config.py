"""
app/config.py

Centralised settings loaded from .env via pydantic-settings.
Import `settings` everywhere — never import os.environ directly.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Twilio
    twilio_account_sid: str = ""
    twilio_auth_token: str = ""
    twilio_phone_number: str = ""

    # Deepgram
    deepgram_api_key: str = ""

    # ElevenLabs
    elevenlabs_api_key: str = ""
    elevenlabs_voice_id: str = "21m00Tcm4TlvDq8ikWAM"

    # Groq
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"

    # Qdrant
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "voiceflow_faq"

    # App
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    public_base_url: str = "https://your-ngrok-url.ngrok-free.app"

    # Logging
    log_level: str = "INFO"
    call_log_path: str = "logs/calls.jsonl"


settings = Settings()
