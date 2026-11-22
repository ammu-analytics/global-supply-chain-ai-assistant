"""Optional LLM layer (OpenAI).

The LLM never calculates anything. The numbers come from pandas / the ML
models, and the LLM is only asked to rewrite that answer in friendlier
words. If there's no API key or the call fails, the app just shows the
original answer, so everything works offline too.
"""

from __future__ import annotations

import os

DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-mini")

SYSTEM_PROMPT = (
    "You rewrite answers for a supply chain analytics app. "
    "Use ONLY the facts and numbers in the draft answer and table. "
    "Do not add new numbers, countries, causes or predictions. "
    "Keep it under 90 words, plain English, no headings."
)


def available() -> bool:
    if not os.getenv("OPENAI_API_KEY"):
        return False
    try:
        import openai  # noqa: F401
    except ImportError:
        return False
    return True


def rephrase(question: str, draft: str, table_text: str = "") -> str | None:
    if not available():
        return None
    try:
        from openai import OpenAI

        client = OpenAI()
        prompt = f"User question: {question}\n\nDraft answer:\n{draft}\n\nData table:\n{table_text or '(none)'}"
        response = client.responses.create(
            model=DEFAULT_MODEL,
            instructions=SYSTEM_PROMPT,
            input=prompt,
        )
        text = (response.output_text or "").strip()
        return text or None
    except Exception:
        return None
