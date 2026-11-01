"""Read-only client for rag-observability's retrieval API."""

import json
import os
from http.client import HTTPException
from typing import Literal
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from pydantic import BaseModel, ConfigDict, Field


class _Chunk(BaseModel):
    model_config = ConfigDict(strict=True)

    rank: int = Field(ge=1)
    chunk_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    source: str = Field(min_length=1)
    file: str
    section: str
    distance: float | None = Field(allow_inf_nan=False)
    exact_term_match: bool


class _IndexedCorpus(BaseModel):
    provenance: Literal["available"]
    corpus_sha256: str = Field(min_length=1, strict=True)
    ingested_at: str = Field(min_length=1, strict=True)
    embedding_model: str = Field(min_length=1, strict=True)
    embedding_revision: str | None
    chunk_size: int | None
    document_count: int = Field(ge=0)
    collection: str = Field(min_length=1)
    chunk_count: int = Field(ge=0)


class _UnknownCorpus(BaseModel):
    provenance: Literal["unavailable"]
    embedding_model: str = Field(min_length=1)
    collection: str = Field(min_length=1)
    chunk_count: int = Field(ge=0)


class _Freshness(BaseModel):
    status: Literal["current", "stale", "unavailable"]
    checked_at: str = Field(min_length=1)
    added: list[str] | None = None
    modified: list[str] | None = None
    deleted: list[str] | None = None
    reason: str | None = None


class _Evidence(BaseModel):
    hits: list[_Chunk] = Field(max_length=20)
    corpus: _IndexedCorpus | _UnknownCorpus = Field(discriminator="provenance")
    freshness: _Freshness
    content_trust: Literal["untrusted"]
    generated: Literal[False]


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None  # Do not forward questions or bearer credentials to another URL.


_opener = build_opener(_NoRedirect())


def search_observability_docs(question: str, k: int = 6) -> str:
    """Send only the explicit question/result count; never read review workspaces."""
    if (
        not isinstance(question, str)
        or not 1 <= len(question.strip()) <= 2000
        or "\x00" in question
    ):
        return "Error: documentation question must contain 1–2000 characters and no NUL."
    if type(k) is not int or not 1 <= k <= 20:
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
        headers = {"Content-Type": "application/json"}
        if key := os.environ.get("OBS_AGENT_RAG_API_KEY"):
            headers["Authorization"] = f"Bearer {key}"
        request = Request(
            base_url + "/retrieve",
            data=json.dumps({"question": question.strip(), "k": k}, ensure_ascii=False).encode(),
            headers=headers,
        )
        with _opener.open(request, timeout=30) as response:
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
    return evidence.model_dump_json(exclude_unset=True)
