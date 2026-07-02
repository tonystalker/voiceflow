"""
app/telephony/twilio_webhook.py

Handles the initial POST webhook from Twilio when a call arrives.
Returns TwiML that opens a Media Stream WebSocket back to our server.
"""

from __future__ import annotations

from fastapi import APIRouter, Form, Request
from fastapi.responses import Response
from loguru import logger

from app.config import settings

router = APIRouter()

_TWIML_TEMPLATE = """\
<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Connect>
    <Stream url="wss://{host}/media-stream">
      <Parameter name="greeting" value="Hello, thank you for calling VoiceBank. I am Aria, your AI banking assistant. How can I help you today?"/>
    </Stream>
  </Connect>
</Response>
"""


@router.post("/incoming-call")
async def incoming_call(
    request: Request,
    CallSid: str = Form(default=""),
    From: str = Form(default=""),
    To: str = Form(default=""),
) -> Response:
    """Twilio sends a POST here when the call connects."""
    logger.info(f"Incoming call: SID={CallSid} from={From} to={To}")

    # Strip scheme from public_base_url to get the hostname for WSS
    host = settings.public_base_url.replace("https://", "").replace("http://", "")
    twiml = _TWIML_TEMPLATE.format(host=host)

    return Response(content=twiml, media_type="application/xml")
