"""The language model client.

The only module that knows which LLM provider is in use. Providers are reached
through their OpenAI-compatible endpoints, so switching from Gemini to Groq,
Qwen or Mistral means changing .env - not this file, and not any other.

Two ways to call the model. generate() waits for the whole reply and returns a
string. generate_stream() yields it in pieces as the provider produces them,
so an interface can show the answer being written. Streaming does not make the
reply arrive sooner - it makes it start arriving sooner, which is most of what
waiting feels like.
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

# This budget covers the model's internal reasoning as well as the words the
# customer sees, and for gpt-oss-120b the reasoning is most of it. Measured:
# an answer of 483 visible characters - roughly 120 tokens of text - spent the
# whole of a 600 token budget and was cut off mid-sentence, because ~480
# tokens went on reasoning first.
#
# The logs also settle what this costs. Usage is billed on tokens actually
# produced, not on the ceiling: ordinary answers come back at 100-300
# completion tokens whatever this is set to. So a high ceiling is close to
# free and a low one truncates. It was briefly 600, which bought nothing and
# clipped answers.
MAX_TOKENS = 1200

# Ask the provider to report token usage on streamed responses too. Not every
# OpenAI-compatible endpoint accepts it, so the first refusal turns it off for
# the rest of the run rather than failing the request.
_stream_usage = True

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


def _request(system_prompt, user_prompt, stream):
    """One chat completion request. Shared so the streaming and blocking calls
    cannot drift apart in temperature, token limit or message shape."""
    global _stream_usage

    client = get_client()

    options = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "stream": stream,
    }

    if stream and _stream_usage:
        options["stream_options"] = {"include_usage": True}

        try:
            return client.chat.completions.create(**options)
        except TypeError:
            # an SDK too old to know the argument
            _stream_usage = False
        except APIError as error:
            if "stream_options" not in str(error):
                raise
            log.info("provider rejected stream_options; usage will not be "
                     "reported for streamed replies")
            _stream_usage = False

        options.pop("stream_options", None)

    return client.chat.completions.create(**options)


def _log_usage(label, usage):
    """Record what a call actually cost.

    Guessing at token counts is how a tokens-per-minute limit sneaks up on
    you. The provider reports the real numbers on every response, so the log
    can answer 'which call is expensive' instead of leaving it to arithmetic.
    """
    if usage is None:
        return

    log.info("%s cost %s prompt + %s completion = %s tokens",
             label,
             getattr(usage, "prompt_tokens", "?"),
             getattr(usage, "completion_tokens", "?"),
             getattr(usage, "total_tokens", "?"))


def _as_runtime_error(error):
    """Turn a provider error into the one exception type the interfaces
    already know how to display."""
    if isinstance(error, RateLimitError):
        log.warning("rate limit hit on %s", MODEL)
        return RuntimeError(
            "The free tier rate limit was reached. Wait a moment and try again."
        )

    log.error("LLM request failed: %s", error)
    return RuntimeError(f"The language model could not be reached: {error}")


def generate(system_prompt, user_prompt):
    """Send one request and return the model's reply as text."""
    try:
        response = _request(system_prompt, user_prompt, stream=False)
    except (RateLimitError, APIError) as error:
        raise _as_runtime_error(error)

    reply = response.choices[0].message.content

    _log_usage("blocking call", getattr(response, "usage", None))
    log.info("generated %d characters", len(reply or ""))

    return reply or ""


def generate_stream(system_prompt, user_prompt):
    """Yield the model's reply in pieces as they arrive.

    The caller is responsible for joining the pieces if it also wants the whole
    answer. Errors surface while iterating rather than at the call, because
    nothing is sent until the first piece is asked for.
    """
    try:
        response = _request(system_prompt, user_prompt, stream=True)

        length = 0
        usage = None

        for chunk in response:
            # the usage report arrives as a final chunk of its own, after the
            # text has finished and with no choices in it
            if getattr(chunk, "usage", None) is not None:
                usage = chunk.usage

            if not chunk.choices:
                continue

            piece = chunk.choices[0].delta.content

            if piece:
                length += len(piece)
                yield piece

        _log_usage("streamed call", usage)
        log.info("streamed %d characters", length)

    except (RateLimitError, APIError) as error:
        raise _as_runtime_error(error)
