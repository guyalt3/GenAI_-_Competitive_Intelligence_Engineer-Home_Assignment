"""Autonomous background worker for proactive competitor intelligence scanning."""

from __future__ import annotations

import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tavily import TavilyClient

import config
import prompts

from agent import (
    create_embeddings,
    create_llm,
    create_tavily_client,
    initialize_environment,
)

LOG_PREFIX = "[Cron Worker]"


def log(message: str) -> None:
    """Print a standardized worker log line."""
    print(f"{LOG_PREFIX} {message}")


def connect_daily_news_store(embeddings: OpenAIEmbeddings) -> Chroma:
    """Connect to the existing daily_news Chroma collection."""
    db_storage_dir = Path(config.DB_STORAGE_DIR)

    if not db_storage_dir.exists():
        raise FileNotFoundError(
            f"Vector DB directory not found at '{config.DB_STORAGE_DIR}/'. "
            "Run ingestion before scheduling this worker."
        )

    daily_news_store = Chroma(
        collection_name=config.DAILY_NEWS_COLLECTION,
        embedding_function=embeddings,
        persist_directory=str(config.DB_STORAGE_DIR),
    )

    return daily_news_store


def build_search_query(competitor_name: str) -> str:
    """Build a Tavily query focused on recent competitive intelligence."""
    search_query = (
        f"{competitor_name} latest product announcements, "
        "security updates, and market news"
    )

    return search_query


def get_cached_source_urls(
    daily_news_store: Chroma,
    competitor_name: str,
) -> set[str]:
    """Return URLs already stored for a competitor to avoid duplicates."""
    results = daily_news_store._collection.get(
        where={
            "$and": [
                {"competitor": competitor_name},
                {"type": "dynamic"},
            ]
        },
    )

    cached_urls: set[str] = set()
    for metadata in results.get("metadatas") or []:
        if not metadata:
            continue
        source = str(metadata.get("source", "")).strip()
        if source:
            cached_urls.add(source)

    return cached_urls


def clean_text(text: str) -> str:
    """Normalize whitespace in short snippet text."""
    return " ".join(text.split()).strip()


def parse_and_filter_results(
    raw_results: list[dict[str, Any]],
    existing_urls: set[str],
    limit: int = config.MAX_ARTICLES_PER_COMPETITOR,
) -> list[dict[str, Any]]:
    """Clean Tavily results, deduplicate by URL, and keep top-scoring items."""
    ranked_results = sorted(
        raw_results,
        key=lambda result: result.get("score", 0),
        reverse=True,
    )

    filtered_results: list[dict[str, Any]] = []
    for result in ranked_results:
        url = str(result.get("url", "")).strip()
        content = clean_text(str(result.get("content", "")))
        title = clean_text(str(result.get("title", "")))
        raw_content = str(result.get("raw_content") or result.get("content", "")).strip()

        if not url or not content:
            continue
        if url in existing_urls:
            continue

        filtered_results.append(
            {
                "url": url,
                "content": content,
                "raw_content": raw_content,
                "title": title or "Untitled Article",
                "score": result.get("score", 0),
            }
        )
        existing_urls.add(url)

        if len(filtered_results) >= limit:
            break

    return filtered_results


def fetch_competitor_news(
    tavily_client: TavilyClient,
    competitor_name: str,
) -> list[dict[str, Any]]:
    """Fetch recent competitor news from Tavily."""
    search_query = build_search_query(competitor_name)
    log(f"Querying Tavily for '{competitor_name}' (last {config.RAG_RECENCY_DAYS} days)...")

    response = tavily_client.search(
        query=search_query,
        search_depth="advanced",
        max_results=config.MAX_ARTICLES_PER_COMPETITOR + config.TAVILY_FETCH_BUFFER,
        days=config.RAG_RECENCY_DAYS,
        include_raw_content="markdown",
    )

    return response.get("results", [])


def evaluate_chunk_relevance(
    llm: ChatOpenAI,
    chunk_text: str,
    competitor_name: str,
) -> bool:
    """Use the LLM to decide whether a chunk is high-value competitive intelligence."""
    relevance_prompt = ChatPromptTemplate.from_messages(prompts.RELEVANCE_PROMPT)
    chain = relevance_prompt | llm | StrOutputParser()
    response = chain.invoke(
        {
            "competitor": competitor_name,
            "chunk": chunk_text,
        }
    )

    return response.strip().upper().startswith("YES")


def build_documents(
    llm: ChatOpenAI,
    articles: list[dict[str, Any]],
    competitor_name: str,
) -> tuple[list[Document], int, int]:
    """
    Split raw page content into chunks and keep only LLM-approved intelligence.

    Returns:
        A tuple of (documents, parsed_chunk_count, preserved_chunk_count).
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CRON_SCANNER_CHUNK_SIZE,
        chunk_overlap=config.CRON_SCANNER_CHUNK_OVERLAP,
    )

    documents: list[Document] = []
    parsed_chunks = 0
    preserved_chunks = 0

    current_date = datetime.now().strftime("%Y-%m-%d")

    for article in articles:
        page_text = str(article.get("raw_content") or article.get("content", "")).strip()
        if not page_text:
            continue

        raw_chunks = splitter.split_text(page_text)

        for chunk in raw_chunks:
            chunk = chunk.strip()
            if len(chunk) <= config.CRON_SCANNER_MIN_CHUNK_LENGTH:
                continue

            parsed_chunks += 1
            if not evaluate_chunk_relevance(llm, chunk, competitor_name):
                continue

            preserved_chunks += 1
            documents.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "competitor": competitor_name,
                        "type": "dynamic",
                        "date": current_date,
                        "source": article["url"],
                        "title": article["title"],
                        "summary": article["content"],
                    },
                )
            )

    return documents, parsed_chunks, preserved_chunks


def cache_articles(
    daily_news_store: Chroma,
    documents: list[Document],
) -> int:
    """Persist new documents into the daily_news collection."""
    if not documents:
        return 0

    daily_news_store.add_documents(documents)

    return len(documents)


def scan_competitor(
    competitor_name: str,
    tavily_client: TavilyClient,
    daily_news_store: Chroma,
    llm: ChatOpenAI,
) -> int:
    """Run the full scan pipeline for a single competitor."""
    log(f"Starting autonomous scan for {competitor_name}...")

    existing_urls = get_cached_source_urls(daily_news_store, competitor_name)
    log(f"Found {len(existing_urls)} existing cached URL(s) for {competitor_name}.")

    raw_results = fetch_competitor_news(tavily_client, competitor_name)
    log(f"Received {len(raw_results)} raw result(s) from Tavily for {competitor_name}.")

    articles = parse_and_filter_results(raw_results, existing_urls)
    if not articles:
        log(f"No new articles to cache for {competitor_name}.")
        return 0

    log(
        f"Processing {len(articles)} article(s) through offline chunking "
        f"and relevance filtering for {competitor_name}..."
    )
    documents, parsed_chunks, preserved_chunks = build_documents(
        llm,
        articles,
        competitor_name,
    )
    filtered_out = parsed_chunks - preserved_chunks
    log(
        f"Filtered out {filtered_out} noisy chunk(s), preserved "
        f"{preserved_chunks} high-value chunk(s) for {competitor_name}."
    )

    cached_count = cache_articles(daily_news_store, documents)

    log(f"Successfully cached {cached_count} high-value chunk(s) for {competitor_name}.")
    for index, article in enumerate(articles, start=1):
        log(
            f"  [{index}] {article['title']} "
            f"(score={article.get('score', 'n/a')}) -> {article['url']}"
        )

    return cached_count


def run_scan() -> tuple[int, int]:
    """
    Execute scans for all core competitors.

    Returns:
        A tuple of (total_cached_chunks, failed_competitor_count).
    """
    initialize_environment()

    embeddings = create_embeddings()
    daily_news_store = connect_daily_news_store(embeddings)
    tavily_client = create_tavily_client()
    llm = create_llm()

    total_cached = 0
    failed_competitors = 0

    core_competitors = config.CORE_COMPETITORS

    log(
        f"Connected to '{config.DAILY_NEWS_COLLECTION}' at '{config.DB_STORAGE_DIR}/'. "
        f"Scanning {len(core_competitors)} competitor(s)."
    )

    for competitor_name in core_competitors:
        try:
            cached_count = scan_competitor(
                competitor_name,
                tavily_client,
                daily_news_store,
                llm,
            )
            total_cached += cached_count
        except Exception as exc:
            failed_competitors += 1
            log(f"ERROR: Scan failed for {competitor_name}: {exc}")
            log(traceback.format_exc())

    return total_cached, failed_competitors


def main() -> int:
    """Entry point for cron, Airflow, or manual execution."""
    log("Worker started.")

    try:
        total_cached, failed_competitors = run_scan()
    except Exception as exc:
        log(f"FATAL: Worker initialization failed: {exc}")
        log(traceback.format_exc())
        return 1

    log(
        f"Worker finished. Cached {total_cached} high-value chunk(s) across "
        f"{len(config.CORE_COMPETITORS)} competitor(s). Failures: {failed_competitors}."
    )

    if failed_competitors == len(config.CORE_COMPETITORS):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
