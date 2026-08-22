"""The language model client.

The only module that knows which LLM provider is in use. Providers are reached
through their OpenAI-compatible endpoints, so switching from Gemini to Groq,
Qwen or Mistral means changing .env - not this file, and not any other.
"""

import logging
import os

from dotenv import load_dotenv
from openai import OpenAI, RateLimitError, APIError

log = logging.getLogger(__name__)

load_dotenv(override=True)

API_KEY = os.getenv("LLM_API_KEY")
BASE_URL = os.getenv("LLM_BASE_URL")
MODEL = os.getenv("LLM_MODEL")

# low temperature: this bot quotes prices and USSD codes, so consistency
# matters far more than variety
TEMPERATURE = 0.2
MAX_TOKENS = 800

_client = None


def get_client():
    """Create the client once and reuse it."""
    global _client

    if _client is None:
        if not API_KEY:
            raise SystemExit(
                "LLM_API_KEY is missing. Add it to the .env file in the "
                "project root."
            )
        _client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
        log.info("LLM client ready: %s", MODEL)

    return _client


def generate(system_prompt, user_prompt):
    """Send one request and return the model's reply as text."""
    client = get_client()

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
    except RateLimitError:
        log.warning("rate limit hit on %s", MODEL)
        raise RuntimeError(
            "The free tier rate limit was reached. Wait a moment and try again."
        )
    except APIError as error:
        log.error("LLM request failed: %s", error)
        raise RuntimeError(f"The language model could not be reached: {error}")

    reply = response.choices[0].message.content
    log.info("generated %d characters", len(reply or ""))

    return reply or ""