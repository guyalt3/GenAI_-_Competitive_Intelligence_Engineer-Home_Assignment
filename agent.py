"""Agentic RAG pipeline for competitor intelligence using local cache and web search."""

from __future__ import annotations

import os
from pathlib import Path

from datetime import datetime, timedelta
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from sentence_transformers import CrossEncoder
from tavily import TavilyClient

import config
import prompts

RERANKER_MODEL = CrossEncoder(config.CROSS_ENCODER_MODEL)

CACHE_DATE = datetime.now().strftime("%Y-%m-%d")

load_dotenv()


def initialize_environment() -> None:
    """Load environment variables from a .env file."""
    load_dotenv()


def create_llm() -> ChatOpenAI:
    """Return a fast, cost-effective OpenRouter chat model."""
    llm = ChatOpenAI(
        openai_api_base=config.OPENROUTER_API_BASE,
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        model=config.LLM_MODEL,
        temperature=0,
    )

    return llm


def create_tavily_client() -> TavilyClient:
    """Return an authenticated Tavily search client."""
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        raise ValueError("TAVILY_API_KEY is not set in the environment.")

    return TavilyClient(api_key=api_key)


def create_embeddings() -> OpenAIEmbeddings:
    """Return an OpenRouter embeddings model."""
    openai_embeddings = OpenAIEmbeddings(
        openai_api_base=config.OPENROUTER_API_BASE,
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        model=config.EMBEDDING_MODEL,
    )

    return openai_embeddings


def connect_vector_stores(
    embeddings: OpenAIEmbeddings,
) -> tuple[Chroma, Chroma]:
    """Connect to existing ChromaDB collections in db_storage/."""
    db_storage_dir = Path(config.DB_STORAGE_DIR)

    static_store = Chroma(
        collection_name=config.STATIC_COLLECTION,
        embedding_function=embeddings,
        persist_directory=str(db_storage_dir),
    )
    daily_news_store = Chroma(
        collection_name=config.DAILY_NEWS_COLLECTION,
        embedding_function=embeddings,
        persist_directory=str(db_storage_dir),
    )

    return static_store, daily_news_store


def _format_documents(documents: list[Document], label: str) -> str:
    """Render retrieved documents into a readable context block."""
    if not documents:
        return f"[{label}] No relevant documents found."

    sections: list[str] = []
    for index, document in enumerate(documents, start=1):
        metadata = document.metadata or {}
        source = metadata.get("source") or metadata.get("title") or label
        sections.append(
            f"Document {index} ({source}):\n{document.page_content.strip()}"
        )

    return "\n\n".join(sections)


def _extract_sources(documents: list[Document], label: str) -> list[str]:
    """Collect human-readable source labels from document metadata."""
    sources: list[str] = []
    for document in documents:
        metadata = document.metadata or {}
        source = metadata.get("source") or metadata.get("title")
        if source:
            sources.append(str(source))
        else:
            sources.append(f"Local cache ({label})")

    return sources


def search_local_collections(
    query_text: str,
    competitor_name: str,
    static_store: Chroma,
    daily_news_store: Chroma,
) -> tuple[list[Document], list[Document]]:
    """Perform similarity search across static and daily news collections."""
    competitor_filter = (
        None if competitor_name == "All Competitors" 
        else {"competitor": competitor_name}
    )
    static_results = static_store.similarity_search(
        query_text,
        k=config.VECTOR_SEARCH_TOP_K,
        filter=competitor_filter,
    )
    daily_results = daily_news_store.similarity_search(
        query_text,
        k=config.VECTOR_SEARCH_TOP_K,
        filter=competitor_filter,
    )

    return static_results, daily_results


def evaluate_context_sufficiency(
    llm: ChatOpenAI,
    query_text: str,
    competitor_name: str,
    local_context: str,
) -> bool:
    """Ask the LLM whether local context alone is sufficient (YES/NO)."""
    evaluation_prompt = ChatPromptTemplate.from_messages(prompts.EVALUATION_PROMPT)
    chain = evaluation_prompt | llm | StrOutputParser()
    evaluation = chain.invoke(
        {
            "query": query_text,
            "competitor": competitor_name,
            "context": local_context,
        }
    )
    print(f"Router decision: {evaluation.strip()}")

    return evaluation.strip().upper().startswith("YES")


def fetch_web_results(
    tavily_client: TavilyClient,
    query_text: str,
    competitor_name: str,
) -> list[dict]:
    """Fetch latest web results from Tavily for a cache miss."""
    search_query = f"{competitor_name} {query_text}"
    response = tavily_client.search(
        query=search_query,
        search_depth=config.TAVILY_SEARCH_DEPTH,
        max_results=config.TAVILY_MAX_RESULTS,
    )

    return response.get("results", [])


def cache_web_results(
    daily_news_store: Chroma,
    web_results: list[dict],
    competitor_name: str,
) -> list[Document]:
    """Persist fresh web results into the daily_news collection."""
    documents: list[Document] = []
    for result in web_results:
        content = result.get("content", "").strip()
        if not content:
            continue

        documents.append(
            Document(
                page_content=content,
                metadata={
                    "competitor": competitor_name,
                    "type": "dynamic",
                    "date": CACHE_DATE,
                    "source": result.get("url", ""),
                    "title": result.get("title", ""),
                },
            )
        )

    if documents:
        daily_news_store.add_documents(documents)
        print(f"Cached {len(documents)} new web result(s) into '{config.DAILY_NEWS_COLLECTION}'.")

    return documents


def synthesize_answer(
    llm: ChatOpenAI,
    query_text: str,
    competitor_name: str,
    context: str,
    sources: list[str],
    include_jfrog_impact: bool,
) -> str:
    """Generate the final structured intelligence brief."""
    unique_sources = list(dict.fromkeys(sources))
    sources_text = "\n".join(f"- {source}" for source in unique_sources)

    if include_jfrog_impact:
        raw_synthesis_prompt = prompts.SYNTHESIS_PROMPT_WITH_JFROG_IMPACT
    else:
        raw_synthesis_prompt = prompts.SYNTHESIS_PROMPT_WITHOUT_JFROG_IMPACT

    synthesis_prompt = ChatPromptTemplate.from_messages(raw_synthesis_prompt)
    chain = synthesis_prompt | llm | StrOutputParser()

    answer = chain.invoke(
        {
            "query": query_text,
            "competitor": competitor_name,
            "context": context,
            "sources": sources_text,
        }
    )

    return answer


def rerank_chunks(query: str, chunks: list[Document]) -> list[Document]:
    if not chunks:
        return []

    pairs = [[query, chunk.page_content] for chunk in chunks]
    scores = RERANKER_MODEL.predict(pairs)

    for chunk, score in zip(chunks, scores):
        if chunk.metadata is None:
            chunk.metadata = {}
        chunk.metadata["rerank_score"] = float(score)

    sorted_chunks = sorted(
        chunks,
        key=lambda document: document.metadata["rerank_score"],
        reverse=True,
    )

    filtered_chunks = sorted_chunks[:config.RERANK_TOP_K]

    return filtered_chunks


def retrieve_and_filter_context(
    query_text: str,
    competitor_name: str,
    static_store: Chroma,
    daily_news_store: Chroma,
) -> list[Document]:
    print("Checking local cache...")
    static_results, daily_results = search_local_collections(
        query_text,
        competitor_name,
        static_store,
        daily_news_store,
    )
    print(
        f"Retrieved {len(static_results)} static chunk(s) and "
        f"{len(daily_results)} daily news chunk(s)."
    )

    combined_chunks = static_results + daily_results

    current_dt = datetime.strptime(CACHE_DATE, "%Y-%m-%d")
    cutoff_dt = current_dt - timedelta(days=config.RAG_RECENCY_DAYS)
    cutoff_str = cutoff_dt.strftime("%Y-%m-%d")

    fresh_chunks: list[Document] = []
    for chunk in combined_chunks:
        doc_type = chunk.metadata.get("type", "static")
        doc_date = chunk.metadata.get("date", "")

        if doc_type == "static":
            fresh_chunks.append(chunk)
        elif doc_type == "dynamic" and doc_date >= cutoff_str:
            fresh_chunks.append(chunk)
        else:
            print(f"[Recency Filter] Discarding stale chunk from {doc_date} (Cutoff: {cutoff_str})")

    return fresh_chunks


def process_and_format_cache(
    query_text: str,
    fresh_chunks: list[Document]
) -> tuple[str, list[str]]:
    """Rerank fresh chunks and extract formatted context and source labels."""
    print(
        f"[Two-Stage RAG] Re-ranking {len(fresh_chunks)} raw chunks "
        "using local Cross-Encoder..."
    )
    filtered_chunks = rerank_chunks(query_text, fresh_chunks)

    local_context = _format_documents(filtered_chunks, "Re-ranked Cache")
    sources = _extract_sources(filtered_chunks, "Re-ranked Cache")

    return local_context, sources


def resolve_final_context(
    llm: ChatOpenAI,
    tavily_client: TavilyClient,
    daily_news_store: Chroma,
    query_text: str,
    competitor_name: str,
    local_context: str,
    sources: list[str],
) -> tuple[str, list[str]]:
    """Evaluate local context sufficiency and complement with web search on a cache miss."""
    print("Evaluating whether local context is sufficient...")
    cache_hit = evaluate_context_sufficiency(
        llm,
        query_text,
        competitor_name,
        local_context,
    )

    web_documents: list[Document] = []

    if cache_hit:
        print("Cache Hit! Using local context to generate response.")
        compiled_context = local_context
    else:
        print("Cache Miss! Triggering Web Search...")
        web_results = fetch_web_results(tavily_client, query_text, competitor_name)
        print(f"Fetched {len(web_results)} web result(s) from Tavily.")

        web_documents = cache_web_results(
            daily_news_store,
            web_results,
            competitor_name,
        )

        if web_documents:
            web_context = _format_documents(web_documents, "Tavily Web Search")
            web_sources = _extract_sources(web_documents, "Tavily Web Search")
            compiled_context = f"{local_context}\n\n{web_context}".strip()
            sources.extend(web_sources)
        else:
            print("Web search returned no actionable results. Relying on local cache fallback.")
            compiled_context = local_context

    return compiled_context, sources


def get_competitor_insights(
    query_text: str,
    competitor_name: str,
    include_jfrog_impact: bool = True,
    llm: ChatOpenAI | None = None,
    tavily_client: TavilyClient | None = None,
    static_store: Chroma | None = None,
    daily_news_store: Chroma | None = None,
) -> tuple[str, str]:
    """
    Route a competitor query through local vector search, optional web search,
    and final synthesis.
    """
    print(f"\nAnalyzing query for competitor '{competitor_name}'...")
    print(f"Query: {query_text}")

    llm = llm or create_llm()
    tavily_client = tavily_client or create_tavily_client()

    if static_store is None or daily_news_store is None:
        embeddings = create_embeddings()
        static_store, daily_news_store = connect_vector_stores(embeddings)

    fresh_chunks = retrieve_and_filter_context(
        query_text, competitor_name, static_store, daily_news_store
    )

    local_context, sources = process_and_format_cache(query_text, fresh_chunks)

    compiled_context, sources = resolve_final_context(llm, tavily_client, daily_news_store, query_text, competitor_name,
                                                      local_context, sources)

    print("Synthesizing final intelligence brief...")
    answer = synthesize_answer(
        llm,
        query_text,
        competitor_name,
        compiled_context,
        sources,
        include_jfrog_impact=include_jfrog_impact
    )
    print("Done.\n")
    return answer, compiled_context


def main() -> None:
    initialize_environment()

    sample_query = "What are Snyk's main advantages over JFrog in SCA?"
    sample_competitor = "Snyk"

    response, _ = get_competitor_insights(sample_query, sample_competitor)
    print(response)


if __name__ == "__main__":
    main()
