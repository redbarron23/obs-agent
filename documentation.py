"""Read-only client for rag-observability's retrieval API."""

import json
import os
from http.client import HTTPException
from typing import Literal
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from pydantic import BaseModel, ConfigDict, Field


class _Chunk(BaseModel):
    model_config = ConfigDict(strict=True)

    id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    source: str = Field(min_length=1)
    section: str
    distance: float | None = Field(allow_inf_nan=False)


class _IndexedCorpus(BaseModel):
    status: Literal["available"]
    corpus_sha256: str = Field(min_length=1, strict=True)
    ingested_at: str = Field(min_length=1, strict=True)
    embedding_model: str = Field(min_length=1, strict=True)


class _UnknownCorpus(BaseModel):
    status: Literal["unavailable"]


class _Evidence(BaseModel):
    chunks: list[_Chunk] = Field(max_length=20)
    corpus: _IndexedCorpus | _UnknownCorpus = Field(discriminator="status")


def search_observability_docs(question: str, n_results: int = 6) -> str:
    """Send only the explicit question/result count; never read review workspaces."""
    if not isinstance(question, str) or not 1 <= len(question.strip()) <= 4000:
        return "Error: documentation question must contain 1–4000 characters."
    if type(n_results) is not int or not 1 <= n_results <= 20:
        return "Error: documentation result count must be an integer from 1 to 20."
    base_url = os.environ.get("OBS_AGENT_RAG_URL", "").strip().rstrip("/")
    if not base_url:
        return "Documentation search is disabled; configure OBS_AGENT_RAG_URL to enable it."
    try:
        url = urlsplit(base_url)
        _ = url.port  # Validate malformed ports before making a request.
        if (
            url.scheme not in {"http", "https"}
            or not url.hostname
            or url.username
            or url.password
            or url.query
            or url.fragment
        ):
            return (
                "Error: OBS_AGENT_RAG_URL must be an HTTP(S) base URL without credentials or query."
            )
        request = Request(
            base_url + "/retrieve",
            data=json.dumps({"question": question.strip(), "n_results": n_results}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urlopen(request, timeout=30) as response:
            content = response.read(1_048_577)
        if len(content) > 1_048_576:
            return "Error: documentation response exceeds 1 MiB."
        evidence = _Evidence.model_validate(json.loads(content))
    except HTTPError as exc:
        return (
            f"Error: documentation service returned HTTP {exc.code}; no policy evidence available."
        )
    except (URLError, OSError, HTTPException):
        return (
            "Error: documentation retrieval unavailable or timed out; no policy evidence available."
        )
    except ValueError:
        return "Error: invalid documentation response; no policy evidence available."
    return evidence.model_dump_json()
