"""Streamlit frontend for the JFrog Competitive Intelligence Agent."""

from __future__ import annotations

import io
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

import streamlit as st

import config
from agent import (
    create_embeddings,
    create_llm,
    create_tavily_client,
    connect_vector_stores,
    get_competitor_insights,
    initialize_environment,
)

ALL_COMPETITORS = "All Competitors"


def apply_custom_styles() -> None:
    """Apply lightweight styling for a cleaner dashboard layout."""
    st.markdown(
        """
        <style>
            .block-container {
                padding-top: 2rem;
                padding-bottom: 2rem;
            }
            .radar-card-title {
                font-size: 1.05rem;
                font-weight: 600;
                margin-bottom: 0.35rem;
            }
            .radar-card-meta {
                color: #64748b;
                font-size: 0.85rem;
                margin-bottom: 0.75rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def load_agent_resources() -> dict[str, Any]:
    """Initialize and cache LLM, Tavily, and vector store connections."""
    initialize_environment()
    embeddings = create_embeddings()
    static_store, daily_news_store = connect_vector_stores(embeddings)

    resources = {
        "llm": create_llm(),
        "tavily_client": create_tavily_client(),
        "static_store": static_store,
        "daily_news_store": daily_news_store,
    }

    return resources


def extract_competitors_from_collection(collection: Any) -> set[str]:
    """Extract unique competitor names from a Chroma collection's metadata."""
    results = collection.get()
    competitors: set[str] = set()

    for metadata in results.get("metadatas") or []:
        if not metadata:
            continue

        competitor = str(metadata.get("competitor", "")).strip()
        if competitor and competitor != ALL_COMPETITORS:
            competitors.add(competitor)

    return competitors


def fetch_competitors_from_db(resources: dict[str, Any]) -> list[str]:
    """Build a sorted list of competitors found across Chroma collections."""
    competitors: set[str] = set()
    fallback_competitors = config.FALLBACK_COMPETITORS

    try:
        competitors.update(
            extract_competitors_from_collection(resources["static_store"]._collection)
        )
        competitors.update(
            extract_competitors_from_collection(
                resources["daily_news_store"]._collection
            )
        )
    except Exception:
        return fallback_competitors.copy()

    if not competitors:
        return fallback_competitors.copy()

    return sorted(competitors)


def build_competitor_options(resources: dict[str, Any]) -> list[str]:
    """Return selectbox options with 'All Competitors' first."""
    competitor_options = [ALL_COMPETITORS] + fetch_competitors_from_db(resources)

    return competitor_options


def render_sidebar(competitor_options: list[str]) -> str:
    """Render competitor selection and system status."""
    st.sidebar.title("Competitive Intelligence")
    st.sidebar.caption("JFrog Agentic RAG Console")

    selected_competitor = st.sidebar.selectbox(
        "Target Competitor",
        competitor_options,
        help="Choose which competitor the agent should analyze.",
    )

    st.sidebar.divider()
    st.sidebar.subheader("System Status")

    db_storage_dir = Path(config.DB_STORAGE_DIR)

    if db_storage_dir.exists():
        st.sidebar.success("Local Vector DB connected")
        st.sidebar.caption(f"Storage: `{db_storage_dir}/`")
        st.sidebar.caption(f"Collections: static + daily news")
    else:
        st.sidebar.error("Vector DB not found")
        st.sidebar.caption("Run `python ingestion.py` to initialize local storage.")

    return selected_competitor


def run_agent_with_logs(
    query_text: str,
    competitor_name: str,
    resources: dict[str, Any],
) -> tuple[str, list[str]]:
    """Run the agent and capture stdout log lines for the UI status block."""
    log_buffer = io.StringIO()

    with redirect_stdout(log_buffer):
        response, _ = get_competitor_insights(
            query_text,
            competitor_name,
            llm=resources["llm"],
            tavily_client=resources["tavily_client"],
            static_store=resources["static_store"],
            daily_news_store=resources["daily_news_store"],
        )

    logs = [
        line.strip()
        for line in log_buffer.getvalue().splitlines()
        if line.strip()
    ]

    return response, logs


def render_chat_tab(selected_competitor: str, resources: dict[str, Any]) -> None:
    """Render the deep-dive analyst chat interface."""
    st.subheader("Deep-Dive Analyst Chat")
    st.caption(
        f"Ask strategic questions about **{selected_competitor}**. "
        "The agent checks local intelligence first, then searches the web when needed."
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("logs"):
                with st.expander("Agent trace", expanded=False):
                    for log_line in message["logs"]:
                        st.write(log_line)

    if prompt := st.chat_input(f"Ask about {selected_competitor}..."):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            try:
                with st.status("Agent is analyzing...", expanded=True) as status:
                    response, logs = run_agent_with_logs(
                        prompt,
                        selected_competitor,
                        resources,
                    )
                    for log_line in logs:
                        st.write(log_line)
                    status.update(
                        label="Analysis complete",
                        state="complete",
                        expanded=False,
                    )

                st.markdown(response)
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": response,
                        "logs": logs,
                    }
                )
            except Exception as exc:
                error_message = f"Unable to complete analysis: {exc}"
                st.error(error_message)
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_message}
                )


def fetch_radar_articles(
    daily_news_store: Any,
    competitor_name: str,
) -> list[dict[str, str]]:
    """Load cached web articles for a competitor from the daily_news collection."""
    if competitor_name == ALL_COMPETITORS:
        where_filter: dict[str, Any] = {"type": "dynamic"}
    else:
        where_filter = {
            "$and": [
                {"competitor": competitor_name},
                {"type": "dynamic"},
            ]
        }

    results = daily_news_store._collection.get(where=where_filter)

    documents = results.get("documents") or []
    metadatas = results.get("metadatas") or []

    articles: list[dict[str, str]] = []
    seen_urls: set[str] = set()

    for content, metadata in zip(documents, metadatas):
        metadata = metadata or {}
        url = str(metadata.get("source", "")).strip()
        if url and url in seen_urls:
            continue
        if url:
            seen_urls.add(url)

        snippet = metadata.get("summary") or content.strip().replace("\n", " ")
        snippet_length = config.APP_SNIPPET_LENGTH

        if len(snippet) > snippet_length:
            snippet = f"{snippet[:snippet_length].rstrip()}..."

        articles.append(
            {
                "title": metadata.get("title") or "Untitled Article",
                "date": metadata.get("date") or "Unknown date",
                "snippet": snippet or "No preview available.",
                "url": url,
            }
        )

    articles.sort(key=lambda article: article["date"], reverse=True)

    return articles


def render_radar_tab(selected_competitor: str, resources: dict[str, Any]) -> None:
    """Render cached competitive intelligence news cards."""
    st.subheader("Competitive Intelligence Radar")
    st.caption(
        f"Recently cached web intelligence for **{selected_competitor}** "
        f"from `{config.DAILY_NEWS_COLLECTION}`."
    )

    try:
        articles = fetch_radar_articles(
            resources["daily_news_store"],
            selected_competitor,
        )
    except Exception as exc:
        st.error(f"Unable to load radar data: {exc}")
        return

    if not articles:
        st.info(
            "No cached articles yet for this competitor. "
            "Ask a question in the chat tab to trigger a web search and populate the radar."
        )
        return

    st.metric("Cached Articles", len(articles))

    for article in articles:
        with st.container(border=True):
            st.markdown(
                f'<p class="radar-card-title">{article["title"]}</p>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<p class="radar-card-meta">{article["date"]}</p>',
                unsafe_allow_html=True,
            )
            st.write(article["snippet"])

            if article["url"]:
                st.link_button("Read full article", article["url"])
            else:
                st.caption("Source URL unavailable")


def main() -> None:
    st.set_page_config(
        page_title="JFrog Competitive Intelligence",
        page_icon="🎯",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    apply_custom_styles()

    try:
        resources = load_agent_resources()
    except Exception as exc:
        st.error(f"Failed to initialize agent resources: {exc}")
        st.stop()

    competitor_options = build_competitor_options(resources)
    selected_competitor = render_sidebar(competitor_options)

    st.title("Competitive Intelligence Agent")
    st.markdown(
        "Monitor competitor moves, run deep-dive analysis, and review cached web intelligence."
    )

    chat_tab, radar_tab = st.tabs(
        ["💬 Deep-Dive Analyst Chat", "📺 Competitive Intelligence Radar"]
    )

    with chat_tab:
        render_chat_tab(selected_competitor, resources)

    with radar_tab:
        render_radar_tab(selected_competitor, resources)


if __name__ == "__main__":
    main()
