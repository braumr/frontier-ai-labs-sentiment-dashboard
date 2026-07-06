"""Refresh Reddit and NewsAPI records for the Streamlit dashboard."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import re
import time
import tomllib
from typing import Any

import pandas as pd

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
except ImportError:  # pragma: no cover - dashboard still works without VADER installed.
    SentimentIntensityAnalyzer = None


DATA_DIR = Path(__file__).resolve().parent / "data"
RECORDS_FILE = DATA_DIR / "powerbi_record_level_export.csv"
WEEKLY_FILE = DATA_DIR / "powerbi_weekly_trend_export.csv"
COMPANY_FILE = DATA_DIR / "powerbi_company_topic_summary.csv"
REFRESH_STATE_FILE = DATA_DIR / "refresh_state.json"
REFRESH_LOCK_FILE = DATA_DIR / ".refresh.lock"
START_DATE = "2023-01-01"
START_TIMESTAMP = pd.Timestamp(START_DATE, tz="UTC")

DEFAULT_SUBREDDITS = [
    "artificial",
    "ArtificialInteligence",
    "technology",
    "MachineLearning",
    "singularity",
    "OpenAI",
    "ChatGPT",
]

DEFAULT_SEARCH_TERMS = [
    "AI risk",
    "AI safety",
    "AI regulation",
    "AI privacy",
    "AI security",
    "AI misinformation",
    "AI hallucination",
    "AI job loss",
    "AI copyright",
    "AI adoption risk",
]

DEFAULT_NEWS_QUERIES = [
    "OpenAI OR ChatGPT risk OR safety OR privacy",
    "Anthropic OR Claude risk OR safety OR security",
    "Google Gemini OR DeepMind risk OR safety",
    "Microsoft Copilot risk OR privacy OR security",
    "Meta Llama OR Meta AI risk OR privacy",
    "xAI OR Grok risk OR misinformation",
]

COMPANY_TERMS = {
    "OpenAI": ["openai", "chatgpt", "gpt-4", "gpt-5", "gpt"],
    "Anthropic": ["anthropic", "claude"],
    "Google": ["google", "gemini", "deepmind", "vertex ai"],
    "Microsoft": ["microsoft", "copilot", "azure ai"],
    "Meta": ["meta", "llama"],
    "xAI": ["xai", "grok"],
}

AI_CONTEXT_TERMS = [
    "ai",
    "artificial intelligence",
    "generative ai",
    "machine learning",
    "deep learning",
    "large language model",
    "large language models",
    "llm",
    "llms",
    "chatbot",
    "neural network",
    "foundation model",
    "agentic ai",
]

RISK_TERMS = [
    "bias",
    "breach",
    "compliance",
    "copyright",
    "deepfake",
    "disinformation",
    "ethics",
    "hallucination",
    "job loss",
    "lawsuit",
    "misinformation",
    "privacy",
    "regulation",
    "risk",
    "safety",
    "security",
]


@dataclass
class RefreshConfig:
    """Runtime settings for a refresh job."""

    newsapi_key: str | None = None
    reddit_client_id: str | None = None
    reddit_client_secret: str | None = None
    reddit_user_agent: str = "enterprise-ai-risk-monitor/1.0"
    subreddits: list[str] = field(default_factory=lambda: DEFAULT_SUBREDDITS.copy())
    search_terms: list[str] = field(default_factory=lambda: DEFAULT_SEARCH_TERMS.copy())
    news_queries: list[str] = field(default_factory=lambda: DEFAULT_NEWS_QUERIES.copy())
    total_reddit_comments: int = 1100
    posts_per_search: int = 40
    max_comments_per_post: int = 6
    max_comments_per_week: int = 30
    sleep_seconds: float = 0
    newsapi_page_size: int = 20
    append_existing: bool = True

    @classmethod
    def from_env(cls) -> "RefreshConfig":
        """Build config from environment variables."""
        secrets = load_streamlit_secrets()
        subreddits = os.getenv("REFRESH_SUBREDDITS")
        search_terms = os.getenv("REFRESH_SEARCH_TERMS")
        news_queries = os.getenv("REFRESH_NEWS_QUERIES")
        return cls(
            newsapi_key=get_config_value(secrets, "NEWSAPI_KEY", "NEWS_API_KEY"),
            reddit_client_id=get_config_value(secrets, "REDDIT_CLIENT_ID"),
            reddit_client_secret=get_config_value(secrets, "REDDIT_CLIENT_SECRET"),
            reddit_user_agent=os.getenv(
                "REDDIT_USER_AGENT",
                secrets.get("REDDIT_USER_AGENT", "enterprise-ai-risk-monitor/1.0"),
            ),
            subreddits=[
                value.strip()
                for value in subreddits.split(",")
                if value.strip()
            ]
            if subreddits
            else DEFAULT_SUBREDDITS.copy(),
            search_terms=[
                value.strip()
                for value in search_terms.split("|")
                if value.strip()
            ]
            if search_terms
            else DEFAULT_SEARCH_TERMS.copy(),
            news_queries=[
                value.strip()
                for value in news_queries.split("|")
                if value.strip()
            ]
            if news_queries
            else DEFAULT_NEWS_QUERIES.copy(),
            total_reddit_comments=int(os.getenv("TOTAL_REDDIT_COMMENTS", "1100")),
            posts_per_search=int(os.getenv("POSTS_PER_SEARCH", "40")),
            max_comments_per_post=int(os.getenv("MAX_COMMENTS_PER_POST", "6")),
            max_comments_per_week=int(os.getenv("MAX_COMMENTS_PER_WEEK", "30")),
            sleep_seconds=float(os.getenv("REDDIT_SLEEP_SECONDS", "0")),
            newsapi_page_size=int(os.getenv("MAX_NEWS_ARTICLES_PER_QUERY", "20")),
            append_existing=os.getenv("REFRESH_REPLACE_EXISTING", "").lower()
            not in {"1", "true", "yes"},
        )


def load_streamlit_secrets() -> dict[str, Any]:
    """Read local Streamlit secrets for CLI refreshes without exposing them in app code."""
    secrets_path = Path(__file__).resolve().parent / ".streamlit" / "secrets.toml"
    try:
        with secrets_path.open("rb") as file:
            return tomllib.load(file)
    except (FileNotFoundError, tomllib.TOMLDecodeError, OSError):
        return {}


def get_config_value(secrets: dict[str, Any], *names: str) -> str | None:
    """Return the first configured secret or environment variable for a key."""
    for name in names:
        value = os.getenv(name) or secrets.get(name)
        if value:
            return str(value)
    return None


def _clean_text(value: object) -> str:
    text = "" if value is None else str(value)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _contains_term(text: str, term: str) -> bool:
    """Match complete terms so company names do not fire inside unrelated words."""
    normalized_text = text.lower()
    normalized_term = term.lower()
    pattern = rf"(?<![a-z0-9]){re.escape(normalized_term)}(?![a-z0-9])"
    return re.search(pattern, normalized_text) is not None


def _contains_any_term(text: str, terms: list[str]) -> bool:
    return any(_contains_term(text, term) for term in terms)


def _company_for_text(text: str) -> str:
    for company, terms in COMPANY_TERMS.items():
        if _contains_any_term(text, terms):
            return company
    return "General AI"


def _is_ai_related(text: str) -> bool:
    company_terms = [
        term
        for terms in COMPANY_TERMS.values()
        for term in terms
    ]
    return _contains_any_term(text, AI_CONTEXT_TERMS + company_terms)


def _risk_count(text: str) -> int:
    return sum(1 for term in RISK_TERMS if _contains_term(text, term))


def _sentiment_score(text: str) -> float:
    if SentimentIntensityAnalyzer is None:
        return 0.0
    analyzer = SentimentIntensityAnalyzer()
    return float(analyzer.polarity_scores(text)["compound"])


def _requests():
    """Import requests only when an API refresh is actually requested."""
    try:
        import requests
    except ImportError as exc:
        raise RuntimeError(
            "The refresh workflow requires the `requests` package. "
            "Install dependencies with `pip install -r requirements.txt`."
        ) from exc
    return requests


def _praw():
    """Import PRAW only when Reddit refreshes are configured."""
    try:
        import praw
    except ImportError as exc:
        raise RuntimeError(
            "Reddit refreshes require the `praw` package. "
            "Install dependencies with `pip install -r requirements.txt`."
        ) from exc
    return praw


def _record_from_item(
    *,
    created_at: datetime,
    source: str,
    title: str,
    body: str,
    url: str,
) -> dict[str, Any]:
    text = _clean_text(f"{title} {body}")
    risk_keyword_count = _risk_count(text)
    vader_compound = _sentiment_score(text)
    negative_sentiment_score = (1 - vader_compound) / 2
    risk_label = 1 if risk_keyword_count > 0 else 0
    risk_weight = min(1.0, risk_keyword_count / 4)
    record_risk_sentiment_score = (negative_sentiment_score * 0.75) + (risk_weight * 0.25)
    created_at = created_at.astimezone(timezone.utc)

    return {
        "created_at": created_at.isoformat(),
        "week": created_at.date().isoformat(),
        "month": created_at.strftime("%Y-%m"),
        "source": source,
        "company_or_topic": _company_for_text(text),
        "risk_label": risk_label,
        "risk_keyword_count": risk_keyword_count,
        "vader_compound": vader_compound,
        "negative_sentiment_score": negative_sentiment_score,
        "record_risk_sentiment_score": record_risk_sentiment_score,
        "title": _clean_text(title),
        "clean_text": text,
        "url": url,
    }


def fetch_reddit_records(config: RefreshConfig) -> list[dict[str, Any]]:
    """Fetch Reddit comments using the original notebook subreddits and searches."""
    if not config.reddit_client_id or not config.reddit_client_secret:
        return []

    praw = _praw()
    reddit = praw.Reddit(
        client_id=config.reddit_client_id,
        client_secret=config.reddit_client_secret,
        user_agent=config.reddit_user_agent,
    )
    records: list[dict[str, Any]] = []
    seen_comment_ids: set[str] = set()
    weekly_counts: dict[pd.Timestamp, int] = {}
    sort_options = ["relevance", "top", "comments", "new"]
    time_filters = ["year", "all"]

    for subreddit_name in config.subreddits:
        subreddit = reddit.subreddit(subreddit_name)
        for search_term in config.search_terms:
            for sort_option in sort_options:
                for time_filter in time_filters:
                    submissions = subreddit.search(
                        search_term,
                        sort=sort_option,
                        time_filter=time_filter,
                        limit=config.posts_per_search,
                    )
                    for submission in submissions:
                        submission.comments.replace_more(limit=0)
                        comments = submission.comments.list()[: config.max_comments_per_post]
                        for comment in comments:
                            if len(records) >= config.total_reddit_comments:
                                return records
                            if comment.id in seen_comment_ids:
                                continue

                            created_at = datetime.fromtimestamp(
                                float(comment.created_utc), tz=timezone.utc
                            )
                            if pd.Timestamp(created_at) < START_TIMESTAMP:
                                continue
                            week = (
                                pd.Timestamp(created_at)
                                .tz_localize(None)
                                .to_period("W")
                                .start_time
                            )
                            if weekly_counts.get(week, 0) >= config.max_comments_per_week:
                                continue

                            seen_comment_ids.add(comment.id)
                            weekly_counts[week] = weekly_counts.get(week, 0) + 1
                            record = _record_from_item(
                                created_at=created_at,
                                source="reddit_comment",
                                title=submission.title,
                                body=comment.body,
                                url=f"https://www.reddit.com{comment.permalink}",
                            )
                            if not _is_ai_related(record["clean_text"]):
                                continue
                            record["subreddit"] = subreddit_name
                            record["search_term"] = search_term
                            record["post_url"] = f"https://www.reddit.com{submission.permalink}"
                            record["comment_id"] = comment.id
                            record["post_id"] = submission.id
                            records.append(record)

                    if config.sleep_seconds:
                        time.sleep(config.sleep_seconds)
    return records


def fetch_newsapi_records(config: RefreshConfig) -> list[dict[str, Any]]:
    """Fetch recent NewsAPI articles."""
    if not config.newsapi_key:
        return []

    requests = _requests()
    records = []
    seen_urls: set[str] = set()
    for query in config.news_queries:
        response = requests.get(
            "https://newsapi.org/v2/everything",
            params={
                "q": query,
                "language": "en",
                "sortBy": "publishedAt",
                "pageSize": config.newsapi_page_size,
                "apiKey": config.newsapi_key,
            },
            timeout=30,
        )
        response.raise_for_status()

        for article in response.json().get("articles", []):
            url = article.get("url") or ""
            if url in seen_urls:
                continue
            published_at = article.get("publishedAt")
            if published_at:
                created_at = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
            else:
                created_at = datetime.now(timezone.utc)
            if pd.Timestamp(created_at) < START_TIMESTAMP:
                continue
            record = _record_from_item(
                created_at=created_at,
                source="newsapi_article",
                title=article.get("title") or "",
                body=article.get("description") or article.get("content") or "",
                url=url,
            )
            if not _is_ai_related(record["clean_text"]):
                continue
            if record["company_or_topic"] == "General AI":
                continue
            record["search_term"] = query
            record["post_url"] = url
            seen_urls.add(url)
            records.append(record)
    return records


def aggregate_weekly(records_df: pd.DataFrame) -> pd.DataFrame:
    """Create the weekly trend export used by the dashboard."""
    if records_df.empty:
        return pd.DataFrame()

    df = records_df.copy()
    created_dates = pd.to_datetime(df["created_at"], errors="coerce", utc=True)
    df["week"] = created_dates.dt.tz_localize(None).dt.to_period("W").dt.start_time
    numeric_columns = [
        "vader_compound",
        "negative_sentiment_score",
        "record_risk_sentiment_score",
        "risk_label",
        "risk_keyword_count",
    ]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0)

    weekly = (
        df.groupby("week", dropna=True)
        .agg(
            total_records=("title", "size"),
            risk_records=("risk_label", "sum"),
            avg_sentiment=("vader_compound", "mean"),
            risk_sentiment_score=("record_risk_sentiment_score", "mean"),
            avg_risk_keyword_count=("risk_keyword_count", "mean"),
        )
        .reset_index()
        .sort_values("week")
    )
    weekly["week"] = weekly["week"].dt.date.astype(str)
    weekly["risk_sentiment_score_4wk_avg"] = weekly["risk_sentiment_score"].rolling(
        window=4, min_periods=1
    ).mean()
    weekly["avg_sentiment_4wk_avg"] = weekly["avg_sentiment"].rolling(
        window=4, min_periods=1
    ).mean()
    return weekly


def aggregate_company(records_df: pd.DataFrame) -> pd.DataFrame:
    """Create the company/topic summary export used by the dashboard."""
    if records_df.empty or "company_or_topic" not in records_df.columns:
        return pd.DataFrame()

    df = records_df.copy()
    for column in ["record_risk_sentiment_score", "vader_compound", "risk_label"]:
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0)

    return (
        df.groupby("company_or_topic", dropna=False)
        .agg(
            total_records=("title", "size"),
            risk_records=("risk_label", "sum"),
            avg_sentiment=("vader_compound", "mean"),
            avg_risk_sentiment_score=("record_risk_sentiment_score", "mean"),
        )
        .reset_index()
        .sort_values("avg_risk_sentiment_score", ascending=False)
    )


def apply_record_filters(records_df: pd.DataFrame) -> pd.DataFrame:
    """Keep only records that satisfy the project date and AI-relevance rules."""
    if records_df.empty:
        return records_df.copy()
    filtered_df = records_df.copy()
    record_dates = pd.to_datetime(filtered_df["created_at"], errors="coerce", utc=True)
    filtered_df = filtered_df[record_dates >= START_TIMESTAMP].copy()
    ai_related = filtered_df["clean_text"].fillna("").map(_is_ai_related)
    return filtered_df[ai_related].copy()


def clean_existing_exports() -> dict[str, int]:
    """Prune invalid records already on disk and rebuild dependent exports."""
    if not RECORDS_FILE.exists():
        return {"changed": 0, "removed": 0, "remaining": 0}

    records_df = pd.read_csv(RECORDS_FILE)
    before = len(records_df)
    records_df = apply_record_filters(records_df)
    records_df = records_df.drop_duplicates(subset=["url", "title"], keep="last")
    records_df = records_df.sort_values("created_at", ascending=False)
    removed = before - len(records_df)

    if removed:
        records_df.to_csv(RECORDS_FILE, index=False)
        aggregate_weekly(records_df).to_csv(WEEKLY_FILE, index=False)
        aggregate_company(records_df).to_csv(COMPANY_FILE, index=False)

    return {"changed": int(removed > 0), "removed": removed, "remaining": len(records_df)}


def refresh_public_data(config: RefreshConfig | None = None) -> dict[str, Any]:
    """Fetch public records and update dashboard CSV exports."""
    config = config or RefreshConfig.from_env()
    reddit_enabled = bool(config.reddit_client_id and config.reddit_client_secret)
    newsapi_enabled = bool(config.newsapi_key)
    if not reddit_enabled and not newsapi_enabled:
        raise RuntimeError(
            "No refresh credentials are configured. Set Reddit credentials, "
            "NEWSAPI_KEY, or both before running a data refresh."
        )
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    new_records = fetch_reddit_records(config) + fetch_newsapi_records(config)
    new_df = pd.DataFrame(new_records)

    if config.append_existing and RECORDS_FILE.exists():
        existing_df = pd.read_csv(RECORDS_FILE)
        records_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        records_df = new_df

    if records_df.empty:
        raise RuntimeError("No records were fetched and no existing records were available.")

    records_df = apply_record_filters(records_df)
    if records_df.empty:
        raise RuntimeError(
            f"No records remained after applying {START_DATE} and AI relevance filters."
        )

    records_df = records_df.drop_duplicates(subset=["url", "title"], keep="last")
    records_df = records_df.sort_values("created_at", ascending=False)
    records_df.to_csv(RECORDS_FILE, index=False)
    aggregate_weekly(records_df).to_csv(WEEKLY_FILE, index=False)
    aggregate_company(records_df).to_csv(COMPANY_FILE, index=False)

    return {
        "new_records": len(new_df),
        "total_records": len(records_df),
        "reddit_enabled": reddit_enabled,
        "newsapi_enabled": newsapi_enabled,
        "records_file": str(RECORDS_FILE),
        "weekly_file": str(WEEKLY_FILE),
        "company_file": str(COMPANY_FILE),
    }


def _read_refresh_state() -> dict[str, Any]:
    try:
        return json.loads(REFRESH_STATE_FILE.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def _write_refresh_state(state: dict[str, Any]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    REFRESH_STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")


def refresh_due(now: datetime | None = None, hours: int = 24) -> bool:
    """Return True when the last successful refresh is older than the schedule."""
    now = now or datetime.now(timezone.utc)
    state = _read_refresh_state()
    last_success = state.get("last_success_at")
    if not last_success:
        return True
    try:
        last_success_at = datetime.fromisoformat(last_success)
    except ValueError:
        return True
    return now - last_success_at >= timedelta(hours=hours)


def refresh_lock_is_stale(now: datetime | None = None, hours: int = 2) -> bool:
    """Return True when a lock was left behind by an interrupted refresh."""
    now = now or datetime.now(timezone.utc)
    try:
        lock_created_at = datetime.fromisoformat(
            REFRESH_LOCK_FILE.read_text(encoding="utf-8")
        )
    except (FileNotFoundError, OSError, ValueError):
        return True
    return now - lock_created_at >= timedelta(hours=hours)


def auto_refresh_daily(config: RefreshConfig | None = None) -> dict[str, Any]:
    """Refresh once per day when credentials exist and a refresh is due."""
    config = config or RefreshConfig.from_env()
    reddit_enabled = bool(config.reddit_client_id and config.reddit_client_secret)
    newsapi_enabled = bool(config.newsapi_key)
    if not reddit_enabled and not newsapi_enabled:
        return {"status": "skipped", "reason": "missing_credentials"}
    if not refresh_due():
        return {"status": "skipped", "reason": "not_due"}
    if REFRESH_LOCK_FILE.exists():
        if refresh_lock_is_stale():
            REFRESH_LOCK_FILE.unlink()
        else:
            return {"status": "skipped", "reason": "refresh_in_progress"}

    try:
        REFRESH_LOCK_FILE.write_text(
            datetime.now(timezone.utc).isoformat(), encoding="utf-8"
        )
        result = refresh_public_data(config)
        _write_refresh_state(
            {
                "last_success_at": datetime.now(timezone.utc).isoformat(),
                "last_result": result,
            }
        )
        return {"status": "refreshed", **result}
    except Exception as exc:  # noqa: BLE001 - persisted so the app can keep rendering.
        _write_refresh_state(
            {
                **_read_refresh_state(),
                "last_failure_at": datetime.now(timezone.utc).isoformat(),
                "last_failure": str(exc),
            }
        )
        return {"status": "failed", "reason": str(exc)}
    finally:
        try:
            REFRESH_LOCK_FILE.unlink()
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    result = refresh_public_data()
    print(result)
