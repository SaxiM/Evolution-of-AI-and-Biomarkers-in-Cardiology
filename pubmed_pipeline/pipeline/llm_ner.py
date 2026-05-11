"""NER przez lokalny Ollama (JSON mode) — zgodnie z metodologią LLM."""

from __future__ import annotations

import json
import logging
import re
import urllib.error
import urllib.request
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_OLLAMA_MODEL = "llama3.2"
MAX_ABSTRACT_CHARS = 14_000


def ollama_server_reachable(base_url: str, timeout: float = 5.0) -> bool:
    url = base_url.rstrip("/") + "/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.status == 200
    except (urllib.error.URLError, OSError, TimeoutError):
        return False


def _strip_code_fence(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.IGNORECASE)
        t = re.sub(r"\s*```\s*$", "", t)
    return t.strip()


def _build_prompt(abstract: str, biomarker_keys: list[str], ml_keys: list[str]) -> str:
    bio = ", ".join(biomarker_keys)
    ml = ", ".join(ml_keys)
    body = abstract.strip()
    if len(body) > MAX_ABSTRACT_CHARS:
        body = body[:MAX_ABSTRACT_CHARS] + "\n[... truncated ...]"

    return f"""You are a biomedical Named Entity Recognition system for PubMed abstracts.

Task: From the abstract, identify biomarkers and statistical / machine-learning methods that are used for prediction, prognosis, risk stratification, diagnosis, or outcome modeling.

ALLOWED_BIOMARKER_IDS (output these strings exactly if they apply):
{bio}

ALLOWED_METHOD_IDS (output these strings exactly if they apply):
{ml}

Abstract:
{body}

Return ONLY valid JSON with this exact shape:
{{"biomarkers": ["id1", ...], "ml_methods": ["id1", ...]}}
Use empty arrays if nothing applies. Do not invent IDs outside the allowlists."""


def call_ollama_ner(
    abstract: str,
    biomarker_keys: list[str],
    ml_keys: list[str],
    model: str,
    base_url: str,
    timeout: int = 180,
) -> tuple[dict[str, int], dict[str, int]]:
    """Wywołuje Ollama /api/chat z format=json; zwraca słowniki flag 0/1 (tylko obecne → 1)."""
    base_url = base_url.rstrip("/")
    bio_set = set(biomarker_keys)
    ml_set = set(ml_keys)

    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": _build_prompt(abstract, biomarker_keys, ml_keys),
            }
        ],
        "stream": False,
        "format": "json",
    }

    req = urllib.request.Request(
        f"{base_url}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw_resp = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        logger.warning("Ollama HTTP error: %s", e)
        return {}, {}
    except (urllib.error.URLError, OSError, TimeoutError, json.JSONDecodeError) as e:
        logger.warning("Ollama request failed: %s", e)
        return {}, {}
    except Exception as e:
        logger.warning("Unexpected Ollama error: %s", e)
        return {}, {}

    content = raw_resp.get("message", {}).get("content", "")
    if not content:
        return {}, {}

    try:
        content = _strip_code_fence(content)
        data = json.loads(content)
    except json.JSONDecodeError:
        logger.warning("Ollama returned non-JSON content (first 200 chars): %r", content[:200])
        return {}, {}

    bio_out: dict[str, int] = {}
    for x in data.get("biomarkers", []):
        if isinstance(x, str) and x in bio_set:
            bio_out[x] = 1

    ml_out: dict[str, int] = {}
    for x in data.get("ml_methods", []):
        if isinstance(x, str) and x in ml_set:
            ml_out[x] = 1

    return bio_out, ml_out
