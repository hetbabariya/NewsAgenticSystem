import asyncio
import os
import time
import uuid
from dotenv import load_dotenv

load_dotenv()

def _require_env(name: str) -> str:
    val = (os.getenv(name) or "").strip()
    if not val:
        raise RuntimeError(f"Missing required env var: {name}")
    return val


async def _embed_text(text: str) -> list[float]:
    """Get embeddings via direct HF Inference API call using aiohttp."""
    import aiohttp
    import json

    token = _require_env("HF_TOKEN")
    model = (os.getenv("HF_EMBEDDING_MODEL") or "intfloat/multilingual-e5-large").strip()

    # Using the exact URL pattern from user request
    url = f"https://router.huggingface.co/hf-inference/models/{model}/pipeline/feature-extraction"

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    payload = {
        "inputs": text,
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, headers=headers, timeout=30) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    try:
                        err_json = json.loads(error_text)
                        msg = err_json.get("error", error_text)
                    except:
                        msg = error_text
                    raise RuntimeError(f"HF API Error {resp.status}: {msg}")

                data = await resp.json()
    except Exception as exc:
        if isinstance(exc, RuntimeError): raise
        raise RuntimeError(f"HuggingFace embedding request failed: {exc}") from exc

    # Response for feature-extraction is usually list[float] or list[list[float]]
    if isinstance(data, list) and data:
        vec = data[0] if isinstance(data[0], list) else data
    else:
        raise RuntimeError(f"Unexpected HF response format: {type(data)}")

    return [float(x) for x in vec]

def _ensure_index(*, dimension: int) -> "object":
    from pinecone import Pinecone, ServerlessSpec

    api_key = _require_env("PINECONE_API_KEY")
    index_name = (os.getenv("PINECONE_INDEX_NAME") or "newsagent").strip() or "newsagent"

    pc = Pinecone(api_key=api_key)

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
            dimension=int(dimension),
            metric="cosine",
            spec=ServerlessSpec(cloud=os.getenv("PINECONE_CLOUD") or "aws", region=os.getenv("PINECONE_REGION") or "us-east-1"),
        )

        # Give Pinecone a moment to provision.
        time.sleep(2)

    return pc.Index(index_name)


def _cosine_query(index: "object", vector: list[float], top_k: int = 5) -> list[dict]:
    res = index.query(vector=vector, top_k=max(1, int(top_k)), include_metadata=True)
    matches = []
    for m in getattr(res, "matches", []) or []:
        matches.append(
            {
                "id": getattr(m, "id", None),
                "score": float(getattr(m, "score", 0.0)),
                "metadata": dict(getattr(m, "metadata", {}) or {}),
            }
        )
    return matches


async def main() -> None:
    # Inputs
    upsert_text = (os.getenv("SEMANTIC_TEST_UPSERT_TEXT") or "User likes AI agents and Pinecone semantic search.").strip()
    query_text = (os.getenv("SEMANTIC_TEST_QUERY_TEXT") or "semantic search for AI agents").strip()

    print("[semantic_memory_test] embedding upsert text...")
    upsert_vec = await _embed_text(upsert_text)

    print("[semantic_memory_test] ensuring pinecone index...")
    index = _ensure_index(dimension=len(upsert_vec))

    test_id = os.getenv("SEMANTIC_TEST_ID") or f"semantic-test-{uuid.uuid4()}"
    metadata = {
        "type": "semantic_test",
        "text": upsert_text[:2000],
        "created_at": int(time.time()),
    }

    print(f"[semantic_memory_test] upserting id={test_id} dim={len(upsert_vec)}")
    index.upsert(vectors=[{"id": test_id, "values": upsert_vec, "metadata": metadata}])

    print("[semantic_memory_test] embedding query text...")
    query_vec = await _embed_text(query_text)

    print("[semantic_memory_test] querying...")
    matches = _cosine_query(index, query_vec, top_k=int(os.getenv("SEMANTIC_TEST_TOP_K") or "5"))

    print("\n[semantic_memory_test] top matches:")
    for i, m in enumerate(matches, start=1):
        meta = m.get("metadata") or {}
        preview = (meta.get("text") or "")[:120].replace("\n", " ")
        print(f"  {i}. id={m.get('id')} score={m.get('score'):.4f} preview={preview!r}")

    ok = any(str(m.get("id")) == str(test_id) for m in matches)
    print("\n[semantic_memory_test] result:")
    print("  ok=" + str(ok))
    if not ok:
        raise SystemExit(2)


if __name__ == "__main__":
    asyncio.run(main())
