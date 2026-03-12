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
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Referer": "https://www.google.com/",
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
    subreddit = _truncate(subreddit.strip(), 80) or "MachineLearning"
    safe_limit = _clamp_int(limit, 1, max(1, settings.reddit_max_posts))
    url = f"https://www.reddit.com/r/{subreddit}/top.json?limit={safe_limit}&t=day"
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

    search_topics = topics or ["MachineLearning", "Programming"]
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
                await execute(
                    """
                    INSERT INTO summaries (source_id, summary_text, relevance_score, metadata)
                    VALUES ($1, $2, $3, $4)
                    ON CONFLICT (source_id) DO UPDATE SET
                        summary_text = EXCLUDED.summary_text,
                        relevance_score = EXCLUDED.relevance_score,
                        metadata = EXCLUDED.metadata
                    """,
                    summary.source_id, summary.summary_text, score,
                    json.dumps({"key_points": summary.key_points, "tags": summary.tags})
                )

                inserted = True

                await execute(
                    "UPDATE raw_articles SET status = 'summarized' WHERE id = $1",
                    summary.source_id,
                )
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
            "excluded_topics": list(set(update.excluded_topics))
        }

        await execute("INSERT INTO preferences (prefs_json) VALUES ($1)", json.dumps(new_prefs))
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
        from app.db.neon import write_episodic
        await write_episodic(
            event_type="user_fact",
            description=extracted.fact,
            metadata={"category": extracted.category, "importance": extracted.importance}
        )
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
            SELECT id, summary_text, relevance_score, created_at, metadata
            FROM summaries
            WHERE created_at >= NOW() - INTERVAL '24 hours'
              AND sent_newspaper = FALSE
            ORDER BY relevance_score DESC
            """
        )
        summaries = [dict(r) for r in rows]

    if not summaries:
        return {"pdf_path": None, "articles_count": 0, "message": "No summaries found."}

    date_str = datetime.now().strftime("%B %d, %Y")
    articles_html = ""
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

        key_points = "".join([f"<li>{p}</li>" for p in (meta.get("key_points") or [])])
        tags = ", ".join(meta.get("tags") or [])
        articles_html += f"""
        <div class="article">
            <p class="summary">{s.get('summary_text')}</p>
            <ul>{key_points}</ul>
            <p class="meta">Tags: {tags} | Score: {s.get('relevance_score')}</p>
        </div>
        <hr/>"""

    html_template = f"<html><body><h1>The Agentic Daily</h1><p>{date_str}</p>{articles_html}</body></html>"

    try:
        os.makedirs("temp", exist_ok=True)
        pdf_path = os.path.join("temp", f"newspaper_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf")

        # Injected CSS to handle potential Fontconfig issues by using generic fonts
        css = """
        @page { size: A4; margin: 2cm; }
        body { font-family: serif; line-height: 1.6; color: #333; }
        h1 { color: #2c3e50; text-align: center; border-bottom: 2px solid #2c3e50; padding-bottom: 10px; }
        .date { text-align: center; font-style: italic; color: #7f8c8d; margin-bottom: 30px; }
        .article { margin-bottom: 25px; page-break-inside: avoid; }
        .summary { font-weight: bold; font-size: 1.1em; margin-bottom: 10px; }
        ul { margin-top: 5px; margin-bottom: 10px; }
        .meta { font-size: 0.9em; color: #95a5a6; border-top: 1px solid #eee; padding-top: 5px; }
        """

        # Use a simpler HTML structure for better reliability
        full_html = f"""
        <!DOCTYPE html>
        <html>
        <head><style>{css}</style></head>
        <body>
            <h1>The Agentic Daily</h1>
            <p class="date">{date_str}</p>
            {articles_html}
        </body>
        </html>
        """

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
                f.write(f"The Agentic Daily - {date_str}\n\n")
                # Strip HTML for text fallback
                import re
                clean_html = re.sub('<[^<]+?>', '', articles_html)
                f.write(clean_html)
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
]
