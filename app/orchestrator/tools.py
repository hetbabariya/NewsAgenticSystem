import json
import logging
import os
from datetime import datetime
from typing import Any, List, Optional

import httpx
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from weasyprint import HTML

from app.core.logger import agent_logger
from app.db.neon import execute, fetch_all, fetch_val, get_preferences
from app.core.settings import settings
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

log = logging.getLogger("orchestrator.tools")

# --- Pydantic Models for Structured Output ---

class ScoredArticle(BaseModel):
    article_id: str = Field(description="The unique ID of the article")
    relevance_score: int = Field(description="Relevance score from 1-10", ge=1, le=10)
    is_urgent: bool = Field(description="Whether the article is urgent based on user thresholds")
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
        return ChatGroq(
            api_key=settings.groq_keys[0],
            model_name=settings.model_filter
        )
    else:
        return ChatOpenAI(
            api_key=settings.openrouter_keys[0],
            base_url="https://openrouter.ai/api/v1",
            model_name=settings.model_summarizer
        )

# --- Tools ---

@tool
async def get_user_preferences() -> dict:
    """Fetch the user's current preference profile."""
    return await get_preferences()

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

@tool
async def fetch_reddit_posts(subreddit: str = "MachineLearning", limit: int = 10) -> dict:
    """Fetch top posts from a specific subreddit using Reddit's JSON API."""
    url = f"https://www.reddit.com/r/{subreddit}/top.json?limit={limit}"
    headers = {"User-Agent": "news-agent/0.1"}

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(url, headers=headers)
            resp.raise_for_status()
            data = resp.json()

            posts = []
            for post in data.get("data", {}).get("children", []):
                p = post.get("data", {})
                posts.append({
                    "id": p.get("id"),
                    "title": p.get("title"),
                    "url": p.get("url"),
                    "score": p.get("score"),
                    "created": p.get("created_utc"),
                    "selftext": p.get("selftext", "")[:500]
                })
            return {"posts": posts, "count": len(posts)}
    except Exception as e:
        log.error("Reddit fetch failed: %s", e)
        return {"error": str(e), "posts": []}

@tool
async def run_news_collector(sources: list[str], topics: list[str] | None = None) -> dict:
    """Collect fresh news articles from configured sources."""
    agent_logger.log_tool_call("run_news_collector", {"sources": sources, "topics": topics})
    search_topics = topics or ["MachineLearning", "Programming"]
    total_collected = 0
    details = []

    for source in [s.lower() for s in sources]:
        if source == "reddit":
            for topic in search_topics:
                try:
                    log.info("Fetching reddit posts for topic: %s", topic)
                    reddit_data = await fetch_reddit_posts.ainvoke({"subreddit": topic, "limit": 5})
                    posts = reddit_data.get("posts", [])

                    for p in posts:
                        content = p.get("selftext", "") or p.get("title", "")
                        await execute(
                            """
                            INSERT INTO raw_articles (title, content, url, source, fetched_at)
                            VALUES ($1, $2, $3, $4, NOW())
                            ON CONFLICT (url) DO NOTHING
                            """,
                            p.get("title"), content, p.get("url"), f"reddit/{topic}"
                        )
                    total_collected += len(posts)
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
    """Score/filter collected articles against the user's preferences."""
    agent_logger.log_tool_call("run_preference_scoring", {})
    prefs = await get_preferences()
    articles = await fetch_all("SELECT id, title, content FROM raw_articles WHERE relevance_score IS NULL LIMIT 10")

    if not articles:
        return {"scored_count": 0, "message": "No new articles to score."}

    model = await _get_model("groq")
    parser = PydanticOutputParser(pydantic_object=ScoringResults)

    template = PromptTemplate(
        template="""You are a news filter. Score these articles based on the user's preferences.
        User Preferences: {prefs}

        Articles:
        {articles_text}

        {format_instructions}

        CRITICAL: Your response must be a valid JSON object matching the format instructions exactly.
        Do not include any introductory text, markdown blocks, or commentary.
        """,
        input_variables=["prefs", "articles_text"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )

    articles_text = "\n".join([f"ID: {a['id']} - Title: {a['title']}" for a in articles])
    chain = template | model | parser

    try:
        log.info("Invoking LLM for scoring with %d articles", len(articles))
        results: ScoringResults = await chain.ainvoke({"prefs": str(prefs), "articles_text": articles_text})

        for score in results.scores:
            await execute(
                "UPDATE raw_articles SET relevance_score = $1, is_urgent = $2 WHERE id = $3",
                score.relevance_score, score.is_urgent, int(score.article_id) if score.article_id.isdigit() else score.article_id
            )

        res = {
            "scored_count": len(results.scores),
            "urgent_ids": [s.article_id for s in results.scores if s.is_urgent],
            "message": f"Successfully scored {len(results.scores)} articles."
        }
        agent_logger.log_tool_result("run_preference_scoring", str(res))
        return res
    except Exception as e:
        log.error("Scoring failed: %s", e)
        return {"error": str(e), "scored_count": 0}

@tool
async def run_summarizer() -> dict:
    """Summarize the relevant scored articles and store summaries."""
    agent_logger.log_tool_call("run_summarizer", {})
    articles = await fetch_all(
        "SELECT id, title, content, relevance_score FROM raw_articles WHERE relevance_score >= $1 AND id NOT IN (SELECT source_id FROM summaries) LIMIT 5",
        settings.score_minimum
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

        for summary in results.summaries:
            article = next((a for a in articles if str(a['id']) == str(summary.source_id)), None)
            score = article['relevance_score'] if article else 0

            await execute(
                """
                INSERT INTO summaries (source_id, summary_text, relevance_score, metadata)
                VALUES ($1, $2, $3, $4)
                """,
                summary.source_id, summary.summary_text, score,
                json.dumps({"key_points": summary.key_points, "tags": summary.tags})
            )

        res = {
            "summarized_ids": [s.source_id for s in results.summaries],
            "message": f"Successfully summarized {len(results.summaries)} articles."
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
async def generate_newspaper_pdf() -> dict:
    """Render a newspaper-style PDF from existing summaries."""
    agent_logger.log_tool_call("generate_newspaper_pdf", {})
    summaries = await fetch_all(
        """
        SELECT summary_text, relevance_score, created_at, metadata
        FROM summaries
        WHERE created_at >= NOW() - INTERVAL '24 hours'
        ORDER BY relevance_score DESC
        """
    )

    if not summaries:
        return {"pdf_path": None, "message": "No summaries found."}

    date_str = datetime.now().strftime("%B %d, %Y")
    articles_html = ""
    for s in summaries:
        meta = json.loads(s['metadata']) if isinstance(s['metadata'], str) else s['metadata']
        key_points = "".join([f"<li>{p}</li>" for p in meta.get("key_points", [])])
        tags = ", ".join(meta.get("tags", []))
        articles_html += f"""
        <div class="article">
            <p class="summary">{s['summary_text']}</p>
            <ul>{key_points}</ul>
            <p class="meta">Tags: {tags} | Score: {s['relevance_score']}</p>
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
        return {"pdf_path": pdf_path, "message": "PDF generated successfully."}
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
            return {"pdf_path": txt_path, "message": f"PDF failed ({str(e)}), generated text fallback."}
        except Exception as txt_e:
            log.error("Text fallback also failed: %s", txt_e)
            return {"error": str(e), "pdf_path": None}

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
