import asyncio
import hashlib
import json
import logging
import os
import urllib.parse
from datetime import datetime
from typing import Any, List, Optional

import httpx
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from weasyprint import HTML

from app.core.settings import settings
from app.core.logger import agent_logger
from app.orchestrator.models import (
    CollectorResponse,
    FilterResponse,
    SummarizerResponse,
    PublisherResponse,
    MemoryResponse,
    SupportResponse,
)
from app.db.neon import execute, fetch_all, fetch_val, get_preferences, write_episodic
from app.db.neon import bump_key_usage
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

log = logging.getLogger("orchestrator.tools")


async def _upload_file_to_supabase_storage(*, file_path: str, content_type: str) -> dict | None:
    if not settings.supabase_url or not settings.supabase_service_role_key:
        return None

    if not (settings.supabase_url.startswith("https://") or settings.supabase_url.startswith("http://")):
        log.warning("SUPABASE_URL must start with http(s):// (got %s...) — skipping upload", settings.supabase_url[:20])
        return None

    bucket = (settings.supabase_storage_bucket or "newspapers").strip()
    if not bucket:
        return None

    filename = os.path.basename(file_path)
    object_path = f"{datetime.now().strftime('%Y/%m/%d')}/{filename}"
    encoded_path = urllib.parse.quote(object_path)

    base = settings.supabase_url.rstrip("/")
    put_url = f"{base}/storage/v1/object/{bucket}/{encoded_path}"

    headers = {
        "authorization": f"Bearer {settings.supabase_service_role_key}",
        "apikey": settings.supabase_service_role_key,
        "content-type": content_type,
        "x-upsert": "true",
    }

    try:
        with open(file_path, "rb") as f:
            data = f.read()
    except Exception:
        return None

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(put_url, headers=headers, content=data)
            if resp.status_code >= 300:
                log.warning("Supabase upload failed status=%s body=%s", resp.status_code, resp.text[:300])
                return None

            public_url = None
            signed_url = None
            if settings.supabase_storage_public:
                public_url = f"{base}/storage/v1/object/public/{bucket}/{encoded_path}"
            else:
                sign_url = f"{base}/storage/v1/object/sign/{bucket}/{encoded_path}"
                sign_resp = await client.post(
                    sign_url,
                    headers={
                        "authorization": f"Bearer {settings.supabase_service_role_key}",
                        "apikey": settings.supabase_service_role_key,
                        "content-type": "application/json",
                    },
                    json={"expiresIn": 60 * 60 * 24},
                )
                if sign_resp.status_code < 300:
                    signed_path = (sign_resp.json() or {}).get("signedURL")
                    if signed_path:
                        signed_url = f"{base}{signed_path}"
                else:
                    log.warning(
                        "Supabase signed URL failed status=%s body=%s",
                        sign_resp.status_code,
                        sign_resp.text[:300],
                    )
    except (httpx.InvalidURL, httpx.RequestError) as exc:
        log.warning("Supabase upload skipped due to request error: %s", exc)
        return None
    except Exception as exc:
        log.warning("Supabase upload skipped due to unexpected error: %s", exc)
        return None

    return {
        "bucket": bucket,
        "object_path": object_path,
        "public_url": public_url,
        "signed_url": signed_url,
    }


def _clamp_int(value: int, min_v: int, max_v: int) -> int:
    try:
        v = int(value)
    except Exception:
        v = min_v
    return max(min_v, min(max_v, v))


def _truncate(value: str | None, max_len: int) -> str:
    if not value:
        return ""
    s = str(value)
    return s if len(s) <= max_len else s[:max_len]


def _sanitize_article(a: dict) -> dict:
    url = _truncate((a.get("url") or "").strip(), settings.max_article_url_chars)
    title = _truncate(a.get("title"), settings.max_article_title_chars)
    content = _truncate(a.get("content"), settings.max_article_content_chars)
    source = _truncate(a.get("source") or "mcp", settings.max_article_source_chars)
    return {"url": url, "title": title, "content": content, "source": source}

_groq_key_idx = 0
_openrouter_key_idx = 0

# --- Semantic memory / Pinecone (lazy-init) ---

_pinecone_index = None
_emb_model = None
_semantic_initialized = False


def _guess_embedding_dimension(model_name: str | None) -> int:
    m = (model_name or "").lower()
    # Common lightweight sentence embedding models.
    if "bge-small" in m:
        return 384
    if "minilm" in m:
        return 384
    # Conservative fallback.
    return 768


def _init_semantic_memory():
    """
    Lazy initializer for Pinecone index.
    Uses Hugging Face API for embeddings, so local model is not needed.

    Fails gracefully (returns (None, None)) if Pinecone is not configured.
    """
    global _pinecone_index, _semantic_initialized
    if _semantic_initialized:
        return _pinecone_index, True

    _semantic_initialized = True

    if not settings.pinecone_api_key:
        log.info("Pinecone not configured (PINECONE_API_KEY missing) — semantic memory disabled.")
        return None, None

    try:
        from pinecone import Pinecone, ServerlessSpec
    except Exception as exc:  # ImportError or other
        log.warning("pinecone-client not installed or failed to import: %s", exc)
        return None, None

    try:
        pc = Pinecone(api_key=settings.pinecone_api_key)
        index_name = (settings.pinecone_index_name or "newsagent").strip()
        if not index_name:
            index_name = "newsagent"

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
            # Default small serverless index suitable for semantic memory.
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
        log.info("Semantic memory partially initialized (index failed)")
        return None, None

    return _pinecone_index, True


async def _embed_texts(texts: list[str]) -> list[list[float]]:
    """Compute embeddings for a list of texts using Langchain HuggingFaceEndpointEmbeddings."""
    if not texts:
        return []

    index, is_ready = _init_semantic_memory()
    if not is_ready:
        return []

    try:
        embeddings_model = HuggingFaceEndpointEmbeddings(
            model=settings.hf_embedding_model,
            huggingfacehub_api_token=settings.hf_token,
        )
        # aembed_documents uses asyncio under the hood
        result = await embeddings_model.aembed_documents(texts)
        return result
    except Exception as exc:
        log.warning("Embedding encode failed via LangChain HF endpoint: %s", exc)
        return []


async def _semantic_upsert(
    *,
    item_id: str,
    text: str,
    metadata: dict,
) -> None:
    """Upsert a single semantic memory item into Pinecone."""
    if not text or not item_id:
        return

    index, _ = _init_semantic_memory()
    if index is None:
        return

    vectors = await _embed_texts([text])
    if not vectors:
        return

    vec = vectors[0]

    # Ensure there is always some searchable text in metadata.
    meta = dict(metadata or {})
    if not meta.get("text"):
        meta["text"] = (text or "")[:2000]

    def _do_upsert() -> None:
        try:
            index.upsert(vectors=[{"id": item_id, "values": vec, "metadata": meta}])
        except Exception as exc:
            log.warning("Pinecone upsert failed for %s: %s", item_id, exc)

    await asyncio.to_thread(_do_upsert)

# --- Pydantic Models for Structured Output ---

class ScoredArticle(BaseModel):
    article_id: str = Field(description="The unique ID of the article")
    relevance_score: float = Field(description="Relevance score from 0.0-1.0 (0.0 if not relevant)", ge=0.0, le=1.0)
    is_urgent: bool = Field(description="Whether the article is urgent (relevance_score > SCORE_URGENT)")
    reasoning: str = Field(description="Brief explanation for the score")

class ScoringResults(BaseModel):
    scores: List[ScoredArticle] = Field(description="List of scored articles")

class ArticleSummary(BaseModel):
    source_id: str = Field(description="ID of the source article")
    summary_text: str = Field(description="Concise, factual summary")
    key_points: List[str] = Field(description="3-5 key takeaways")
    tags: List[str] = Field(description="Relevant topic tags")

class SummarizationResults(BaseModel):
    summaries: List[ArticleSummary] = Field(description="List of generated summaries")

class PreferenceUpdate(BaseModel):
    new_topics: List[str] = Field(description="Updated list of interest topics")
    new_keywords: List[str] = Field(description="Updated list of keywords")
    excluded_topics: List[str] = Field(description="Topics to explicitly ignore")
    summary: str = Field(description="Summary of changes made")

class UserFact(BaseModel):
    fact: str = Field(description="The normalized fact about the user")
    category: str = Field(description="Category of the fact (e.g., job, preference, schedule)")
    importance: int = Field(description="How much this should influence behavior (1-5)", ge=1, le=5)


class SemanticMatch(BaseModel):
    id: str = Field(description="The internal id of the memory item")
    score: float = Field(description="Similarity score (higher is more similar)")
    type: str = Field(description="Type of memory item (summary, fact, preference)")
    text: str = Field(description="Primary text for this memory (summary text or fact)")
    metadata: dict = Field(description="Additional metadata (tags, importance, etc.)")


class InterestPlan(BaseModel):
    """Plan of what to search given everything the system knows about the user."""

    topics: List[str] = Field(description="High-level topics representing the user's current interests")
    reddit_subreddits: List[str] = Field(description="Concrete subreddit names to use for Reddit fetches")
    web_queries: List[str] = Field(description="Short news-style web search queries (NOT definitions)")
    github_queries: List[str] = Field(description="Short queries to discover repos/issues relevant to the user")
    twitter_queries: List[str] = Field(description="Short queries to discover tweets/threads relevant to the user")
    summary: str = Field(description="Natural language explanation of why these queries match the user's interests")


class NewspaperArticleEdited(BaseModel):
    """LLM-generated editorial fields for one newspaper article."""
    summary_id: str = Field(description="The id of the summary this refers to (pass through unchanged)")
    headline: str = Field(description="A crisp, punchy newspaper headline (max 12 words, title case, NO full-stop)")
    deck: str = Field(description="One-sentence subheading elaborating the headline (max 25 words)")
    source_label: str = Field(description="Reader-friendly attribution, e.g. 'Reddit · r/MachineLearning' or 'Web' or 'Twitter'")


class NewspaperEdition(BaseModel):
    """Full set of editorial overrides for the day's newspaper."""
    articles: List[NewspaperArticleEdited] = Field(description="One entry per article, in the same order as input")

# --- LLM Helper ---

async def _get_model(provider: str = "groq"):
    """Helper to get the appropriate model based on agent routing."""
    if provider == "groq":
        global _groq_key_idx
        if not settings.groq_keys:
            raise RuntimeError("No Groq API keys configured.")
        idx = _groq_key_idx % len(settings.groq_keys)
        _groq_key_idx += 1
        await bump_key_usage("groq", idx)
        return ChatGroq(
            api_key=settings.groq_keys[idx],
            model_name=settings.model_filter
        )
    else:
        global _openrouter_key_idx
        if not settings.openrouter_keys:
            raise RuntimeError("No OpenRouter API keys configured.")
        idx = _openrouter_key_idx % len(settings.openrouter_keys)
        _openrouter_key_idx += 1
        await bump_key_usage("openrouter", idx)
        return ChatOpenAI(
            api_key=settings.openrouter_keys[idx],
            base_url="https://openrouter.ai/api/v1",
            model_name=settings.model_summarizer
        )

# --- Tools ---

@tool
async def get_user_preferences() -> dict:
    """Fetch the user's current preference profile."""
    prefs = await get_preferences()
    # Ensure we return a structure the agents expect
    if not prefs:
        return {"topics": ["AI", "Machine Learning", "Space"], "keywords": ["innovation"], "excluded_topics": []}
    return prefs

@tool
async def query_recent_summaries(hours: int = 6, keyword: str = "") -> dict:
    """Query recently generated summaries from storage."""
    base = (
        "SELECT id, summary_text, relevance_score, created_at "
        "FROM summaries "
        "WHERE created_at >= NOW() - ($1 || ' hours')::interval"
    )
    params: list[Any] = [str(hours)]
    if keyword:
        base += " AND summary_text ILIKE $2"
        params.append(f"%{keyword}%")
    base += " ORDER BY relevance_score DESC LIMIT 10"

    rows = await fetch_all(base, *params)
    summaries = [dict(r) for r in rows]
    return {"summaries": summaries, "count": len(summaries), "fresh": len(summaries) > 0}

@tool
async def get_system_stats() -> dict:
    """Return basic health/status counters for today's activity."""
    articles = int(await fetch_val("SELECT COUNT(*) FROM raw_articles WHERE fetched_at >= CURRENT_DATE") or 0)
    summaries = int(await fetch_val("SELECT COUNT(*) FROM summaries WHERE created_at >= CURRENT_DATE") or 0)
    prefs = await get_preferences()
    return {
        "articles_ingested_today": articles,
        "summaries_created_today": summaries,
        "topics": prefs.get("topics", []),
    }

@tool
async def fetch_summaries_for_newspaper() -> dict:
    """Fetch candidate summaries for the daily newspaper."""
    rows = await fetch_all(
        """
        SELECT id, summary_text, relevance_score
        FROM summaries
        WHERE created_at >= NOW() - INTERVAL '24 hours'
          AND sent_newspaper = FALSE
        ORDER BY relevance_score DESC
        """
    )
    return {"summaries": [dict(r) for r in rows], "count": len(rows)}

@tool
async def mark_newspaper_sent() -> dict:
    """Mark the last 24 hours of summaries as delivered via newspaper."""
    await execute(
        """
        UPDATE summaries SET sent_newspaper = TRUE
        WHERE created_at >= NOW() - INTERVAL '24 hours'
        """
    )
    return {"ok": True}

# Anti-blocking headers for Reddit
REDDIT_HEADERS = {
    "User-Agent": "NewsAgenticSystem/1.0.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
}

def _extract_reddit_post(post_data: dict) -> dict:
    """Extract only necessary fields from Reddit post data."""
    data = post_data.get("data", {})
    return {
        "id": data.get("id"),
        "title": data.get("title", "")[:500],
        "url": data.get("url", ""),
        "score": data.get("score", 0),
        "created": data.get("created_utc", 0),
        "selftext": data.get("selftext", "")[:1000],
        "subreddit": data.get("subreddit", ""),
        "author": data.get("author", "[unknown]"),
        "num_comments": data.get("num_comments", 0),
    }

@tool
async def fetch_reddit_posts(subreddit: str = "MachineLearning", limit: int = 10) -> dict:
    """Fetch top posts from a specific subreddit using Reddit's JSON API with retry logic."""
    # Clean subreddit name: remove /r/ prefix if present, strip whitespace
    sub = subreddit.strip()
    if sub.lower().startswith("/r/"):
        sub = sub[3:]
    elif sub.lower().startswith("r/"):
        sub = sub[2:]

    sub = _truncate(sub, 80) or "MachineLearning"

    # Special case for 'AI' which sometimes needs to be 'ai' or 'ArtificialIntelligence'
    if sub.upper() == "AI":
        sub = "ArtificialIntelligence" # /r/AI often redirects or 404s depending on user-agent

    safe_limit = _clamp_int(limit, 1, max(1, settings.reddit_max_posts))
    url = f"https://www.reddit.com/r/{sub}/top.json?limit={safe_limit}&t=day"
    max_retries = _clamp_int(settings.reddit_max_retries, 1, 10)
    base_delay = float(settings.reddit_retry_base_delay_seconds)

    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient(timeout=float(settings.reddit_timeout_seconds)) as client:
                # Add random delay to avoid rate limiting
                if attempt > 0:
                    await asyncio.sleep(base_delay * attempt)

                resp = await client.get(url, headers=REDDIT_HEADERS)

                # Handle rate limiting
                if resp.status_code == 429:
                    retry_after = int(resp.headers.get("retry-after", base_delay * (attempt + 1)))
                    log.warning("Reddit rate limited, waiting %d seconds (attempt %d/%d)", retry_after, attempt + 1, max_retries)
                    if attempt < max_retries - 1:
                        await asyncio.sleep(retry_after)
                        continue
                    return {"error": "Rate limited after retries", "posts": []}

                resp.raise_for_status()
                data = resp.json()

                posts = []
                for post in data.get("data", {}).get("children", []):
                    posts.append(_extract_reddit_post(post))

                log.info("Fetched %d posts from r/%s", len(posts), subreddit)
                return {"posts": posts, "count": len(posts), "source": "reddit"}

        except httpx.HTTPStatusError as e:
            log.warning("Reddit HTTP error (attempt %d/%d): %s", attempt + 1, max_retries, e)
            if attempt == max_retries - 1:
                return {"error": f"HTTP error after {max_retries} retries: {e}", "posts": []}
        except httpx.RequestError as e:
            log.warning("Reddit request error (attempt %d/%d): %s", attempt + 1, max_retries, e)
            if attempt == max_retries - 1:
                return {"error": f"Request error after {max_retries} retries: {e}", "posts": []}
        except Exception as e:
            log.error("Reddit fetch failed unexpectedly: %s", e)
            return {"error": str(e), "posts": []}

    return {"error": "Max retries exceeded", "posts": []}

def _generate_url_hash(url: str) -> str:
    """Generate a consistent SHA-256 hash for a URL."""
    return hashlib.sha256(url.encode('utf-8')).hexdigest()

@tool
async def run_news_collector(
    sources: list[str],
    topics: list[str] | None = None,
    articles: list[dict] | None = None,
) -> dict:
    """Collect fresh news articles from configured sources.

    If `articles` is provided, they will be inserted directly into `raw_articles` (this is the
    intended path for MCP tool results).
    Expected keys per item: url (required), title, content, source.
    """
    agent_logger.log_tool_call("run_news_collector", {"sources": sources, "topics": topics})
    sources = [str(s).strip().lower() for s in (sources or []) if str(s).strip()]
    sources = sources[: max(1, settings.collector_max_sources)]

    if topics is None:
        try:
            prefs = await get_preferences()
            pref_topics = prefs.get("topics") if isinstance(prefs, dict) else None
            if isinstance(pref_topics, list) and pref_topics:
                search_topics = pref_topics
            else:
                search_topics = ["MachineLearning", "Programming"]
        except Exception:
            search_topics = ["MachineLearning", "Programming"]
    else:
        search_topics = topics
    search_topics = [str(t).strip() for t in (search_topics or []) if str(t).strip()]
    search_topics = search_topics[: max(1, settings.collector_max_topics)]

    max_total = max(1, int(settings.collector_max_articles_total))
    total_collected = 0
    details = []

    if articles:
        inserted = 0
        for a in list(articles)[: max(0, int(settings.collector_max_mcp_articles))]:
            try:
                sa = _sanitize_article(a if isinstance(a, dict) else {})
                if not sa["url"]:
                    continue
                url_hash = _generate_url_hash(sa["url"])
                await execute(
                    """
                    INSERT INTO raw_articles (title, content, url, url_hash, source, fetched_at)
                    VALUES ($1, $2, $3, $4, $5, NOW())
                    ON CONFLICT (url_hash) DO NOTHING
                    """,
                    sa["title"], sa["content"], sa["url"], url_hash, sa["source"],
                )
                inserted += 1
                if total_collected + inserted >= max_total:
                    break
            except Exception as e:
                log.error("Collector failed to insert MCP article: %s", e)

        total_collected += inserted
        details.append(f"mcp: {inserted}")

    for source in [s.lower() for s in sources]:
        if total_collected >= max_total:
            break
        if source == "reddit":
            for topic in search_topics:
                if total_collected >= max_total:
                    break
                try:
                    log.info("Fetching reddit posts for topic: %s", topic)
                    reddit_data = await fetch_reddit_posts.ainvoke({"subreddit": topic, "limit": settings.reddit_max_posts})
                    posts = reddit_data.get("posts", [])

                    for p in posts:
                        if total_collected >= max_total:
                            break
                        url = p.get("url")
                        if not url:
                            continue
                        url_hash = _generate_url_hash(url)
                        title = _truncate(p.get("title"), settings.max_article_title_chars)
                        content = _truncate(p.get("selftext", "") or p.get("title", ""), settings.max_article_content_chars)
                        src = _truncate(f"reddit/{topic}", settings.max_article_source_chars)
                        await execute(
                            """
                            INSERT INTO raw_articles (title, content, url, url_hash, source, fetched_at)
                            VALUES ($1, $2, $3, $4, $5, NOW())
                            ON CONFLICT (url_hash) DO NOTHING
                            """,
                            title, content, _truncate(url, settings.max_article_url_chars), url_hash, src
                        )
                    total_collected += min(len(posts), max_total - total_collected)
                    details.append(f"reddit/{topic}: {len(posts)}")
                except Exception as e:
                    log.error("Collector failed to fetch reddit for %s: %s", topic, e)

        elif source == "tavily":
            # This is a placeholder for when the agent uses MCP.
            # The collector agent should ideally call the MCP tool directly,
            # but we can provide a fallback or helper logic here if needed.
            details.append("tavily: delegated to MCP")

        elif source == "twitter":
            details.append("twitter: delegated to MCP")

    res = {
        "status": "success",
        "total_articles": total_collected,
        "details": details,
        "message": f"Collected {total_collected} articles. Details: {', '.join(details)}"
    }
    agent_logger.log_tool_result("run_news_collector", str(res))
    return res

@tool
async def run_preference_scoring() -> dict:
    """Score/filter collected articles against the user's preferences.

    Expects the model to return a 0.0–1.0 `relevance_score`. An article is marked urgent when
    `relevance_score > settings.score_urgent`.
    """
    agent_logger.log_tool_call("run_preference_scoring", {})
    prefs = await get_preferences()
    articles = await fetch_all("SELECT id, title, content FROM raw_articles WHERE relevance_score IS NULL LIMIT 10")

    if not articles:
        return {"scored_count": 0, "message": "No new articles to score."}

    model = await _get_model("groq")
    parser = PydanticOutputParser(pydantic_object=ScoringResults)

    template = PromptTemplate(
        template="""You are a news filter. Score these articles based on the user's preferences.

        Scoring rubric:
        - Return `relevance_score` as a FLOAT between 0.0 and 1.0 inclusive.
        - 0.0 means not relevant at all.
        - 1.0 means extremely relevant.
        - Use a calibrated scale (do NOT make everything 0.0):
          - 0.90-1.00: strongly matches multiple preference topics/keywords.
          - 0.60-0.89: clearly relevant to at least one preference topic.
          - 0.30-0.59: partially relevant / adjacent topic.
          - 0.10-0.29: weak connection.
          - 0.00-0.09: no meaningful connection.
        - Set `is_urgent` to true ONLY when relevance_score > {score_urgent}.

        Guidance:
        - Prefer giving a small non-zero score (e.g., 0.10-0.30) if there is ANY plausible connection.
        - Use the title AND content excerpt.
        - Keep reasoning short.

        User Preferences: {prefs}

        Articles:
        {articles_text}

        {format_instructions}

        CRITICAL: Your response must be a valid JSON object matching the format instructions exactly.
        Do not include any introductory text, markdown blocks, or commentary.
        """,
        input_variables=["prefs", "articles_text"],
        partial_variables={
            "format_instructions": parser.get_format_instructions(),
            "score_urgent": str(settings.score_urgent),
        }
    )

    articles_text = "\n\n".join(
        [
            (
                f"ID: {a['id']}\n"
                f"Title: {a.get('title')}\n"
                f"Content: {(a.get('content') or '')[:1200]}"
            )
            for a in articles
        ]
    )
    chain = template | model | parser
    try:
        log.info("Invoking LLM for scoring with %d articles", len(articles))
        # Use ainvoke for consistency and logging
        results: ScoringResults = await chain.ainvoke({"prefs": str(prefs), "articles_text": articles_text})

        for score in results.scores:
            try:
                is_urgent = bool(score.relevance_score > float(settings.score_urgent))
                # Ensure we use the UUID id if it's already a UUID in the DB
                await execute(
                    "UPDATE raw_articles SET relevance_score = $1, is_urgent = $2 WHERE id = $3",
                    float(score.relevance_score), is_urgent, score.article_id
                )
            except Exception as update_err:
                log.error("Failed to update score for article %s: %s", score.article_id, update_err)

        res = {
            "scored_count": len(results.scores),
            "urgent_ids": [s.article_id for s in results.scores if float(s.relevance_score) > float(settings.score_urgent)],
            "message": f"Successfully scored {len(results.scores)} articles."
        }
        agent_logger.log_tool_result("run_preference_scoring", str(res))
        return res
    except Exception as e:
        log.error("Scoring failed: %s", e)
        return {"error": str(e), "scored_count": 0}

@tool
async def run_summarizer(
    min_score: float | None = None,
    max_score: float | None = None,
    limit: int = 5,
    article_ids: list[str] | None = None,
) -> dict:
    """Summarize scored articles and store summaries.

    Use `min_score`/`max_score` to control the score range to summarize. Use `article_ids`
    to force summarization of specific articles.
    """
    agent_logger.log_tool_call("run_summarizer", {})

    resolved_min = float(settings.score_minimum) if min_score is None else float(min_score)
    resolved_max = None if max_score is None else float(max_score)
    resolved_limit = int(limit) if limit and int(limit) > 0 else 5

    if article_ids:
        articles = await fetch_all(
            """
            SELECT a.id, a.title, a.content, a.relevance_score
            FROM raw_articles a
            WHERE a.id = ANY($1::uuid[])
              AND a.relevance_score IS NOT NULL
              AND (a.status = 'raw' OR a.status IS NULL)
              AND NOT EXISTS (
                SELECT 1
                FROM summaries s
                WHERE s.source_id = a.id
              )
            LIMIT $2
            """,
            article_ids,
            resolved_limit,
        )
    elif resolved_max is None:
        articles = await fetch_all(
            """
            SELECT a.id, a.title, a.content, a.relevance_score
            FROM raw_articles a
            WHERE a.relevance_score >= $1
              AND (a.status = 'raw' OR a.status IS NULL)
              AND NOT EXISTS (
                SELECT 1
                FROM summaries s
                WHERE s.source_id = a.id
              )
            ORDER BY a.relevance_score DESC
            LIMIT $2
            """,
            resolved_min,
            resolved_limit,
        )
    else:
        articles = await fetch_all(
            """
            SELECT a.id, a.title, a.content, a.relevance_score
            FROM raw_articles a
            WHERE a.relevance_score >= $1
              AND a.relevance_score <= $2
              AND (a.status = 'raw' OR a.status IS NULL)
              AND NOT EXISTS (
                SELECT 1
                FROM summaries s
                WHERE s.source_id = a.id
              )
            ORDER BY a.relevance_score DESC
            LIMIT $3
            """,
            resolved_min,
            resolved_max,
            resolved_limit,
        )

    if not articles:
        return {"summarized_count": 0, "message": "No new articles to summarize."}

    model = await _get_model("openrouter")
    parser = PydanticOutputParser(pydantic_object=SummarizationResults)

    template = PromptTemplate(
        template="""You are a news summarizer. Create concise, factual summaries for these articles.

        Articles:
        {articles_text}

        {format_instructions}

        CRITICAL: Your response must be a valid JSON object matching the format instructions exactly.
        Do not include any introductory text, markdown blocks, or commentary.
        """,
        input_variables=["articles_text"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )

    articles_text = "\n\n".join([f"ID: {a['id']}\nTitle: {a['title']}\nContent: {a['content'][:2000]}" for a in articles])
    chain = template | model | parser

    try:
        log.info("Invoking LLM for summarization of %d articles", len(articles))
        results: SummarizationResults = await chain.ainvoke({"articles_text": articles_text})

        inserted_ids: list[str] = []

        for summary in results.summaries:
            article = next((a for a in articles if str(a['id']) == str(summary.source_id)), None)
            score = article['relevance_score'] if article else 0

            inserted = False

            try:
                meta = {"key_points": summary.key_points, "tags": summary.tags}
                await execute(
                    """
                    INSERT INTO summaries (source_id, summary_text, relevance_score, metadata, pinecone_id)
                    VALUES ($1, $2, $3, $4, $5)
                    ON CONFLICT (source_id) DO UPDATE SET
                        summary_text = EXCLUDED.summary_text,
                        relevance_score = EXCLUDED.relevance_score,
                        metadata = EXCLUDED.metadata,
                        pinecone_id = EXCLUDED.pinecone_id
                    """,
                    summary.source_id,
                    summary.summary_text,
                    score,
                    json.dumps(meta),
                    str(summary.source_id),
                )

                inserted = True

                await execute(
                    "UPDATE raw_articles SET status = 'summarized' WHERE id = $1",
                    summary.source_id,
                )

                # Upsert semantic representation into Pinecone (best-effort).
                try:
                    full_text = f"{summary.summary_text}\n\nTags: {', '.join(summary.tags)}"
                    await _semantic_upsert(
                        item_id=str(summary.source_id),
                        text=full_text,
                        metadata={
                            "type": "summary",
                            "score": float(score or 0.0),
                            "tags": summary.tags,
                        },
                    )
                except Exception as sem_exc:
                    log.warning("Failed to upsert summary %s into semantic memory: %s", summary.source_id, sem_exc)
            except Exception as ins_err:
                log.error("Failed to insert summary for source %s: %s", summary.source_id, ins_err)

            if inserted:
                inserted_ids.append(summary.source_id)

        res = {
            "summarized_ids": inserted_ids,
            "summarized_count": len(inserted_ids),
            "message": f"Successfully summarized {len(inserted_ids)} articles."
        }
        agent_logger.log_tool_result("run_summarizer", str(res))
        return res
    except Exception as e:
        log.error("Summarization failed: %s", e)
        return {"error": str(e), "summarized_count": 0}

@tool
async def update_user_preferences(change_description: str) -> dict:
    """Update the stored user preference profile from a natural-language request."""
    log.info("update_user_preferences: %s", change_description)
    current_prefs = await get_preferences()

    model = await _get_model("groq")
    parser = PydanticOutputParser(pydantic_object=PreferenceUpdate)

    template = PromptTemplate(
        template="""Update the user's news preferences based on their request.
        Current Preferences: {current_prefs}
        User Request: {change_description}

        {format_instructions}

        CRITICAL: Your response must be a valid JSON object matching the format instructions exactly.
        Ensure all fields (new_topics, new_keywords, excluded_topics, summary) are present.
        If no change is needed for a field, return an empty list.
        Do not include any introductory text, markdown blocks, or commentary.
        Do not include "properties", "title", or "type" keys in your final object.
        """,
        input_variables=["current_prefs", "change_description"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )

    chain = template | model | parser
    try:
        update: PreferenceUpdate = await chain.ainvoke({
            "current_prefs": str(current_prefs),
            "change_description": change_description
        })

        new_prefs = {
            "topics": list(set(update.new_topics)),
            "keywords": list(set(update.new_keywords)),
            "excluded_topics": list(set(update.excluded_topics)),
        }

        await execute("INSERT INTO preferences (prefs_json) VALUES ($1)", json.dumps(new_prefs))

        # Also store a semantic preference snapshot so future runs can reason over it.
        try:
            prefs_text = (
                "User preference profile. "
                f"Topics: {', '.join(new_prefs['topics'])}. "
                f"Keywords: {', '.join(new_prefs['keywords'])}. "
                f"Excluded: {', '.join(new_prefs['excluded_topics'])}."
            )
            # Use a deterministic id so future updates overwrite rather than explode memory.
            pref_id = "user-preferences"
            await _semantic_upsert(
                item_id=pref_id,
                text=prefs_text,
                metadata={
                    "type": "preference",
                    "importance": 5,
                },
            )
        except Exception as sem_exc:
            log.warning("Failed to upsert preferences into semantic memory: %s", sem_exc)

        return {"updated": True, "message": update.summary}
    except Exception as e:
        log.error("Preference update failed: %s", e)
        return {"error": str(e), "updated": False}

@tool
async def store_user_fact(fact: str) -> dict:
    """Store a durable user fact for personalization."""
    log.info("store_user_fact: %s", fact)
    model = await _get_model("groq")
    parser = PydanticOutputParser(pydantic_object=UserFact)

    template = PromptTemplate(
        template="""Extract a durable fact from the user's statement.
        User Statement: {fact}

        {format_instructions}

        CRITICAL: Your response must be a valid JSON object matching the format instructions exactly.
        Do not include "properties", "title", or "type" keys in your final object - only the fields "fact", "category", and "importance".
        """,
        input_variables=["fact"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )

    chain = template | model | parser
    try:
        extracted: UserFact = await chain.ainvoke({"fact": fact})

        # Persist as episodic memory (for audit/history)...
        from app.db.neon import write_episodic, fetch_val

        await write_episodic(
            event_type="user_fact",
            description=extracted.fact,
            metadata={"category": extracted.category, "importance": extracted.importance},
        )

        # ...and as long-term semantic memory in semantic_facts + Pinecone.
        try:
            fact_id = await fetch_val(
                """
                INSERT INTO semantic_facts (fact_text, pinecone_id)
                VALUES ($1, NULL)
                RETURNING id
                """,
                extracted.fact,
            )
        except Exception as db_exc:
            log.warning("Failed to insert semantic_fact row: %s", db_exc)
            fact_id = None

        if fact_id is not None:
            try:
                fact_id_str = str(fact_id)
                await _semantic_upsert(
                    item_id=fact_id_str,
                    text=extracted.fact,
                    metadata={
                        "type": "fact",
                        "category": extracted.category,
                        "importance": int(extracted.importance),
                    },
                )
                try:
                    await execute(
                        "UPDATE semantic_facts SET pinecone_id = $1 WHERE id = $2",
                        fact_id_str,
                        fact_id,
                    )
                except Exception as upd_exc:
                    log.warning("Failed to update semantic_facts.pinecone_id: %s", upd_exc)
            except Exception as sem_exc:
                log.warning("Failed to upsert user fact into semantic memory: %s", sem_exc)

        return {"stored": True, "message": f"Stored fact: {extracted.fact}"}
    except Exception as e:
        log.error("Fact storage failed: %s", e)
        return {"error": str(e), "stored": False}

@tool
async def generate_newspaper_pdf(summaries: list[dict] | None = None) -> dict:
    """Render a newspaper-style PDF from summaries.

    If `summaries` is not provided, this tool will use `fetch_summaries_for_newspaper()`
    (unsent summaries from the last 24 hours).
    """
    agent_logger.log_tool_call("generate_newspaper_pdf", {"summaries_provided": bool(summaries)})

    if summaries is None:
        rows = await fetch_all(
            """
            SELECT
                s.id,
                s.summary_text,
                s.relevance_score,
                s.created_at,
                s.metadata,
                a.title   AS article_title,
                a.url     AS article_url,
                a.source  AS article_source
            FROM summaries s
            LEFT JOIN raw_articles a ON a.id = s.source_id
            WHERE s.created_at >= NOW() - INTERVAL '24 hours'
              AND s.sent_newspaper = FALSE
            ORDER BY s.relevance_score DESC
            """
        )
        summaries = [dict(r) for r in rows]

    if not summaries:
        return {"pdf_path": None, "articles_count": 0, "message": "No summaries found."}

    date_str = datetime.now().strftime("%B %d, %Y")
    day_str  = datetime.now().strftime("%A").upper()

    # --- Separate company/finance articles from general news ---
    COMPANY_TAGS = {"company","stock","earnings","market","finance","startup",
                    "ipo","acquisition","merger","investment","funding"}

    regular_articles: list[dict] = []
    company_insights: list[dict] = []

    for s in summaries:
        raw_meta = s.get("metadata")
        meta: dict = {}
        if isinstance(raw_meta, dict):
            meta = raw_meta
        elif isinstance(raw_meta, str) and raw_meta.strip():
            try:
                parsed = json.loads(raw_meta)
                if isinstance(parsed, dict):
                    meta = parsed
            except Exception:
                meta = {}
        s["_meta"] = meta
        tags_lower = {t.lower() for t in (meta.get("tags") or [])}
        if tags_lower & COMPANY_TAGS:
            company_insights.append(s)
        else:
            regular_articles.append(s)

    # -----------------------------------------------------------------------
    # LLM pass: generate newspaper-quality headlines, decks, and source labels
    # for every article in one batch call.
    # -----------------------------------------------------------------------
    async def _generate_editorial(articles: list[dict]) -> dict[str, NewspaperArticleEdited]:
        """Return a mapping of summary id -> editorial fields. Fails gracefully."""
        if not articles:
            return {}
        try:
            model = await _get_model("groq")
            parser = PydanticOutputParser(pydantic_object=NewspaperEdition)
            articles_text = "\n\n".join(
                (
                    f"ID: {s['id']}\n"
                    f"Summary: {(s.get('summary_text') or '')[:600]}\n"
                    f"Key points: {'; '.join((s.get('_meta') or {}).get('key_points') or [])}\n"
                    f"Tags: {', '.join((s.get('_meta') or {}).get('tags') or [])}\n"
                    f"Raw source label: {s.get('article_source') or 'unknown'}"
                )
                for s in articles
            )
            template = PromptTemplate(
                template=(
                    "You are the chief editor of a newspaper called 'The Agentic Daily'.\n"
                    "For each article below, produce:\n"
                    "  1. `headline` — punchy, title-case, max 12 words, no trailing full-stop.\n"
                    "  2. `deck` — one-sentence subheading, max 25 words.\n"
                    "  3. `source_label` — friendly attribution such as:\n"
                    "       'Reddit · r/MachineLearning', 'Web', 'Twitter', 'GitHub', etc.\n"
                    "     Derive it from the raw_source field; keep it very short.\n"
                    "ARTICLES:\n{articles_text}\n\n{format_instructions}\n"
                    "CRITICAL: Return ONLY the JSON object — no markdown, no commentary."
                ),
                input_variables=["articles_text"],
                partial_variables={"format_instructions": parser.get_format_instructions()},
            )
            chain = template | model | parser
            edition: NewspaperEdition = await chain.ainvoke({"articles_text": articles_text})
            return {art.summary_id: art for art in edition.articles}
        except Exception as exc:
            log.warning("Newspaper editorial LLM call failed: %s", exc)
            return {}

    # Run editorial generation for all articles together
    all_summaries_for_editorial = regular_articles + company_insights
    editorial_map = await _generate_editorial(all_summaries_for_editorial)

    def _apply_editorial(s: dict) -> dict:
        """Merge LLM-generated editorial fields onto a summary dict."""
        art = editorial_map.get(str(s.get("id")))
        if art:
            s["_headline"]     = art.headline
            s["_deck"]         = art.deck
            s["_source_label"] = art.source_label
        else:
            # Graceful fallback: first sentence of summary_text as headline
            raw = (s.get("summary_text") or "").strip()
            s["_headline"] = raw[:80] + ("…" if len(raw) > 80 else "")
            s["_deck"]     = ""
            s["_source_label"] = (s.get("article_source") or "").replace("/", " · ")
        return s

    regular_articles  = [_apply_editorial(s) for s in regular_articles]
    company_insights  = [_apply_editorial(s) for s in company_insights]

    # -----------------------------------------------------------------------
    def _make_source_line(art_url: str, source_label: str) -> str:
        if art_url and source_label:
            return (f'<p class="source-line">Source: <strong>{source_label}</strong>'
                    f' &mdash; <a href="{art_url}">{art_url}</a></p>')
        elif art_url:
            return f'<p class="source-line"><a href="{art_url}">{art_url}</a></p>'
        elif source_label:
            return f'<p class="source-line">Source: <strong>{source_label}</strong></p>'
        return ""

    def _article_html(s: dict, lead: bool = False) -> str:
        meta         = s.get("_meta", {})
        key_points   = meta.get("key_points") or []
        tags         = meta.get("tags") or []
        summary      = (s.get("summary_text") or "").strip()
        headline     = s.get("_headline") or summary[:80]
        deck         = s.get("_deck") or ""
        source_label = s.get("_source_label") or ""
        art_url      = (s.get("article_url") or "").strip()
        bullets_html = "".join(f"<li>{p}</li>" for p in key_points)
        tag_pills    = "".join(f'<span class="tag">{t}</span>' for t in tags[:6])
        source_html  = _make_source_line(art_url, source_label)
        deck_html    = f'<p class="deck">{deck}</p>' if deck else ""
        if lead:
            return (
                '<article class="article lead">'
                f'<h2 class="headline">{headline}</h2>'
                f'{deck_html}'
                f'<ul class="bullets">{bullets_html}</ul>'
                f'<p class="summary-body">{summary}</p>'
                f'<div class="tag-row">{tag_pills}</div>'
                f'{source_html}</article><hr class="divider"/>'
            )
        return (
            '<article class="article">'
            f'<h3 class="headline-small">{headline}</h3>'
            f'{deck_html}'
            f'<ul class="bullets">{bullets_html}</ul>'
            f'<p class="summary-body">{summary}</p>'
            f'<div class="tag-row">{tag_pills}</div>'
            f'{source_html}</article>'
        )

    articles_html = "".join(
        _article_html(s, lead=(idx == 0)) for idx, s in enumerate(regular_articles)
    )

    # Company insights block
    insights_html = ""
    if company_insights:
        cards = ""
        for s in company_insights:
            meta         = s.get("_meta", {})
            key_points   = meta.get("key_points") or []
            tags         = meta.get("tags") or []
            summary      = (s.get("summary_text") or "").strip()
            headline     = s.get("_headline") or summary[:80]
            deck         = s.get("_deck") or ""
            source_label = s.get("_source_label") or ""
            art_url      = (s.get("article_url") or "").strip()
            bullets_html = "".join(f"<li>{p}</li>" for p in key_points)
            tag_pills    = "".join(f'<span class="tag">{t}</span>' for t in tags[:6])
            source_html  = _make_source_line(art_url, source_label)
            deck_html    = f'<p class="deck">{deck}</p>' if deck else ""
            cards += (
                '<div class="insight-card">'
                f'<h4 class="insight-title">{headline}</h4>'
                f'{deck_html}'
                f'<ul class="bullets">{bullets_html}</ul>'
                f'<p class="summary-body">{summary}</p>'
                f'<div class="tag-row">{tag_pills}</div>'
                f'{source_html}</div>'
            )
        insights_html = (
            '<section class="insights-section">'
            '<h2 class="insights-heading">&#9642; Company &amp; Market Insights</h2>'
            f'<div class="insights-grid">{cards}</div>'
            '</section>'
        )

    try:
        os.makedirs("temp", exist_ok=True)
        pdf_path = os.path.join("temp", f"newspaper_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf")

        css = """
        @page { size: A4; margin: 1.8cm 1.5cm; }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: "Georgia", "Times New Roman", serif;
            line-height: 1.55;
            color: #1a1a1a;
            background: #fdfaf4;
            font-size: 10pt;
        }
        .masthead {
            text-align: center;
            border-top: 4px double #1a1a1a;
            border-bottom: 4px double #1a1a1a;
            padding: 10px 0 8px 0;
            margin-bottom: 6px;
        }
        .masthead-eyebrow {
            font-size: 7.5pt;
            letter-spacing: 0.22em;
            text-transform: uppercase;
            color: #555;
            margin-bottom: 2px;
        }
        .masthead-title {
            font-size: 36pt;
            font-weight: bold;
            line-height: 1;
            color: #111;
            letter-spacing: 0.04em;
        }
        .masthead-tagline {
            font-size: 8pt;
            letter-spacing: 0.16em;
            text-transform: uppercase;
            color: #666;
            margin-top: 4px;
            font-style: italic;
        }
        .masthead-meta {
            display: flex;
            justify-content: space-between;
            font-size: 7.5pt;
            color: #555;
            border-top: 1px solid #aaa;
            margin-top: 6px;
            padding-top: 4px;
        }
        .section-banner {
            background: #1a1a1a;
            color: #fdfaf4;
            text-align: center;
            padding: 3px 0;
            font-size: 8pt;
            letter-spacing: 0.2em;
            text-transform: uppercase;
            margin-bottom: 10px;
        }
        .front-page {
            column-count: 2;
            column-gap: 20px;
            column-rule: 1px solid #bbb;
        }
        .article {
            break-inside: avoid;
            margin-bottom: 16px;
            padding-bottom: 12px;
            border-bottom: 1px solid #ccc;
        }
        .article:last-child { border-bottom: none; }
        .article.lead {
            column-span: all;
            border-bottom: 2px solid #1a1a1a;
            margin-bottom: 14px;
            padding-bottom: 12px;
        }
        .headline {
            font-size: 22pt;
            font-weight: bold;
            line-height: 1.15;
            margin-bottom: 6px;
            color: #111;
        }
        .headline-small {
            font-size: 13pt;
            font-weight: bold;
            line-height: 1.2;
            margin-bottom: 5px;
            color: #111;
            border-bottom: 1px solid #ddd;
            padding-bottom: 3px;
        }
        .bullets {
            margin: 5px 0 6px 16px;
            font-size: 9.5pt;
        }
        .bullets li { margin-bottom: 3px; }
        .deck {
            font-size: 10pt;
            font-style: italic;
            color: #444;
            margin: 2px 0 6px 0;
            line-height: 1.4;
        }
        .summary-body {
            font-size: 9.5pt;
            line-height: 1.6;
            color: #333;
            margin-bottom: 5px;
        }
        .tag-row { margin: 4px 0; }
        .tag {
            display: inline-block;
            font-size: 7pt;
            font-family: Arial, sans-serif;
            background: #e8e4da;
            border: 1px solid #c8c0ad;
            color: #444;
            border-radius: 2px;
            padding: 1px 5px;
            margin: 1px 2px 1px 0;
            letter-spacing: 0.05em;
            text-transform: uppercase;
        }
        .source-line {
            font-size: 7.5pt;
            color: #777;
            font-family: Arial, sans-serif;
            margin-top: 4px;
            border-top: 1px dotted #ccc;
            padding-top: 3px;
            word-break: break-all;
        }
        .source-line a { color: #3a5fa0; text-decoration: none; }
        .divider { display: none; }
        .insights-section {
            margin-top: 18px;
            border-top: 3px double #1a1a1a;
            padding-top: 10px;
        }
        .insights-heading {
            font-size: 11pt;
            font-weight: bold;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 10px;
            color: #111;
        }
        .insights-grid {
            column-count: 2;
            column-gap: 20px;
            column-rule: 1px solid #bbb;
        }
        .insight-card {
            break-inside: avoid;
            background: #f4f0e6;
            border: 1px solid #d4cbb5;
            border-radius: 2px;
            padding: 8px 10px;
            margin-bottom: 10px;
        }
        .insight-title {
            font-size: 10pt;
            font-weight: bold;
            margin-bottom: 4px;
            color: #111;
        }
        """

        edition_num = datetime.now().strftime("Vol. %Y, No. %j")
        total_count = len(summaries)
        full_html = f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"/><style>{css}</style></head>
<body>
  <header class="masthead">
    <p class="masthead-eyebrow">Your Personalised Intelligence Digest</p>
    <div class="masthead-title">The Agentic Daily</div>
    <p class="masthead-tagline">Intelligent News &mdash; Curated Just For You</p>
    <div class="masthead-meta">
      <span>{edition_num}</span>
      <span>{day_str}, {date_str.upper()}</span>
      <span>{total_count} Articles</span>
    </div>
  </header>
  <div class="section-banner">Today&rsquo;s Top Stories</div>
  <section class="front-page">{articles_html}</section>
  {insights_html}
</body>
</html>"""

        HTML(string=full_html).write_pdf(pdf_path)
        agent_logger.log_tool_result("generate_newspaper_pdf", f"PDF path: {pdf_path}")
        upload = await _upload_file_to_supabase_storage(file_path=pdf_path, content_type="application/pdf")
        return {
            "pdf_path": pdf_path,
            "articles_count": len(summaries),
            "message": "PDF generated successfully.",
            "storage": upload,
        }
    except Exception as e:
        log.error("PDF generation failed: %s", e)
        # Fallback to a simple text file if PDF fails
        try:
            txt_path = pdf_path.replace(".pdf", ".txt")
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(f"The Agentic Daily \u2014 {date_str}\n")
                f.write("=" * 60 + "\n\n")
                for s in summaries:
                    meta       = s.get("_meta") or {}
                    key_points = meta.get("key_points") or []
                    art_title  = (s.get("article_title") or "").strip()
                    art_url    = (s.get("article_url") or "").strip()
                    art_source = (s.get("article_source") or "").strip()
                    summary    = (s.get("summary_text") or "").strip()
                    f.write(f"## {art_title or 'News Briefing'}\n")
                    for pt in key_points:
                        f.write(f"  \u2022 {pt}\n")
                    if summary:
                        f.write(f"\n{summary}\n")
                    if art_source or art_url:
                        f.write(f"\nSource: {art_source}  {art_url}\n")
                    f.write("\n" + "-" * 60 + "\n\n")
            upload = await _upload_file_to_supabase_storage(file_path=txt_path, content_type="text/plain")
            return {
                "pdf_path": txt_path,
                "articles_count": len(summaries),
                "message": f"PDF failed ({str(e)}), generated text fallback.",
                "storage": upload,
            }
        except Exception as txt_e:
            log.error("Text fallback also failed: %s", txt_e)
            return {"error": str(e), "pdf_path": None, "articles_count": len(summaries)}


@tool
async def semantic_search_memory(query: str, top_k: int = 5) -> dict:
    """
    Search long-term semantic memory (summaries, user facts, preferences) using Pinecone.

    Returns the most similar items across:
      - Summaries stored by the summarizer (type='summary')
      - User facts captured via store_user_fact (type='fact')
      - Preference snapshots (type='preference')
    """
    q = (query or "").strip()
    if not q:
        return {"matches": [], "count": 0, "message": "Empty query."}

    index, _ = _init_semantic_memory()
    if index is None:
        return {"matches": [], "count": 0, "message": "Semantic memory not configured."}

    vectors = await _embed_texts([q])
    if not vectors:
        return {"matches": [], "count": 0, "message": "Embedding model not available."}

    vec = vectors[0]

    def _do_query() -> Any:
        try:
            return index.query(
                vector=vec,
                top_k=max(1, min(int(top_k or 5), 20)),
                include_metadata=True,
            )
        except Exception as exc:
            log.warning("Pinecone query failed: %s", exc)
            return None

    res = await asyncio.to_thread(_do_query)
    if not res or not getattr(res, "matches", None):
        return {"matches": [], "count": 0, "message": "No semantic matches found."}

    matches: list[dict] = []
    for m in res.matches:
        meta = dict(getattr(m, "metadata", {}) or {})
        text = meta.get("text") or meta.get("summary_text") or meta.get("fact_text") or ""
        matches.append(
            {
                "id": getattr(m, "id", None),
                "score": float(getattr(m, "score", 0.0)),
                "type": meta.get("type") or "unknown",
                "text": text,
                "metadata": meta,
            }
        )

    return {"matches": matches, "count": len(matches)}


@tool
async def plan_interest_queries(hint: str | None = None) -> dict:
    """
    Plan concrete search queries based on the user's interests and long-term memory.

    This tool makes the system intelligent about WHAT to search:
      - It looks at stored preferences.
      - It can leverage semantic memory indirectly via those preferences and facts.
      - It returns source-specific queries that should yield news, not generic explanations.
    """
    current_prefs = await get_preferences()

    # Compact representation for the LLM; we keep table as a backing store but the plan
    # itself is recomputed dynamically each time this tool is called.
    model = await _get_model("groq")
    parser = PydanticOutputParser(pydantic_object=InterestPlan)

    template = PromptTemplate(
        template="""You are an interest planner for a news agent.

User preference snapshot (may be partial or noisy):
{current_prefs}

Optional hint or latest user statement:
{hint}

Your job is to decide WHAT to search to get fresh, real news that matches the user's interests.

Guidelines:
- Never propose definition-style queries like "what is AI" or "what is machine learning".
- Always propose news-style, event- or trend-focused queries such as
  "latest AI research breakthroughs", "LLM safety incidents", "startup funding for AI infra".
- Subreddits should be concrete and aligned with interests (e.g., 'MachineLearning', 'LocalLLaMA', 'DataScience', 'programming').
- Queries must be SHORT (ideal length: 3–10 words) and suitable for the target source.
- Prefer a small, focused set of topics and queries over many generic ones.
- Use the excluded topics (if any) to AVOID proposing queries in those areas.

{format_instructions}

CRITICAL:
- Return a single JSON object that matches the format instructions exactly.
- Do NOT include definitions or tutorials as queries; everything should be news-oriented.
""",
        input_variables=["current_prefs", "hint"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    chain = template | model | parser
    try:
        plan: InterestPlan = await chain.ainvoke(
            {
                "current_prefs": json.dumps(current_prefs),
                "hint": (hint or "").strip(),
            }
        )
        return plan.model_dump()
    except Exception as e:
        log.error("plan_interest_queries failed: %s", e)
        # Fallback: derive a trivial plan that is at least consumable.
        topics = current_prefs.get("topics") or ["AI", "Machine Learning"]
        safe_topics = [str(t) for t in topics][:5]
        return {
            "topics": safe_topics,
            "reddit_subreddits": ["MachineLearning", "LocalLLaMA", "programming"],
            "web_queries": [f"latest {t} news" for t in safe_topics],
            "github_queries": [f"{t} tools" for t in safe_topics],
            "twitter_queries": [f"{t} trend" for t in safe_topics],
            "summary": "Fallback plan derived directly from preferences due to planner error.",
        }

LOCAL_TOOLS = [
    get_user_preferences,
    query_recent_summaries,
    get_system_stats,
    fetch_summaries_for_newspaper,
    mark_newspaper_sent,
    run_news_collector,
    run_preference_scoring,
    run_summarizer,
    update_user_preferences,
    store_user_fact,
    generate_newspaper_pdf,
    fetch_reddit_posts,
    semantic_search_memory,
    plan_interest_queries,
]
