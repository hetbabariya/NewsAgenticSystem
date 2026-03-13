from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx

from app.core.settings import settings

log = logging.getLogger("semantic_memory")

_pinecone_index = None
_semantic_initialized = False


def _guess_embedding_dimension(model_name: str | None) -> int:
    m = (model_name or "").lower()
    if "bge-small" in m:
        return 384
    if "minilm" in m:
        return 384
    return 768


def init_semantic_memory() -> tuple[Any | None, bool | None]:
    global _pinecone_index, _semantic_initialized
    if _semantic_initialized:
        return _pinecone_index, True

    _semantic_initialized = True

    if not settings.pinecone_api_key:
        log.info("Pinecone not configured (PINECONE_API_KEY missing) — semantic memory disabled.")
        return None, None

    try:
        from pinecone import Pinecone, ServerlessSpec
    except Exception as exc:
        log.warning("pinecone failed to import: %s", exc)
        return None, None

    try:
        pc = Pinecone(api_key=settings.pinecone_api_key)
        index_name = (settings.pinecone_index_name or "newsagent").strip() or "newsagent"

        existing: list[str] = []
        try:
            li = pc.list_indexes()
            if hasattr(li, "names"):
                existing = list(li.names())
            elif isinstance(li, list):
                existing = [str(idx.get("name")) for idx in li if isinstance(idx, dict) and idx.get("name")]
        except Exception:
            existing = []

        if index_name not in existing:
            pc.create_index(
                name=index_name,
                dimension=_guess_embedding_dimension(getattr(settings, "hf_embedding_model", None)),
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )

        _pinecone_index = pc.Index(index_name)
    except Exception as exc:
        log.warning("Failed to initialize Pinecone index: %s", exc)
        _pinecone_index = None

    if _pinecone_index is None:
        return None, None

    return _pinecone_index, True


async def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []

    if not settings.hf_token:
        log.warning("HF_TOKEN missing, skipping embeddings")
        return []

    model = settings.hf_embedding_model or "intfloat/multilingual-e5-large"
    url = f"https://router.huggingface.co/hf-inference/models/{model}/pipeline/feature-extraction"

    headers = {
        "Authorization": f"Bearer {settings.hf_token}",
        "Content-Type": "application/json",
    }

    async def _fetch_one(text: str) -> list[float] | None:
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(url, json={"inputs": text}, headers=headers)
                if resp.status_code != 200:
                    log.warning("HF API error %d: %s", resp.status_code, resp.text)
                    return None

                data = resp.json()
                if isinstance(data, list) and data:
                    vec = data[0] if isinstance(data[0], list) else data
                    return [float(x) for x in vec]
                return None
        except Exception as exc:
            log.warning("HF embedding request failed: %s", exc)
            return None

    results: list[list[float]] = []
    for text in texts:
        vec = await _fetch_one(text)
        if vec:
            results.append(vec)

    return results


async def semantic_upsert(*, item_id: str, text: str, metadata: dict) -> None:
    if not text or not item_id:
        return

    index, _ = init_semantic_memory()
    if index is None:
        return

    vectors = await embed_texts([text])
    if not vectors:
        return

    vec = vectors[0]

    meta = dict(metadata or {})
    if not meta.get("text"):
        meta["text"] = (text or "")[:2000]

    def _do_upsert() -> None:
        try:
            index.upsert(vectors=[{"id": item_id, "values": vec, "metadata": meta}])
        except Exception as exc:
            log.warning("Pinecone upsert failed for %s: %s", item_id, exc)

    await asyncio.to_thread(_do_upsert)
