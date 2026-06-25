"""Ingest competitor intelligence documents into ChromaDB vector stores."""
import os

import config

from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


load_dotenv()


def create_embeddings() -> OpenAIEmbeddings:
    """Return an OpenAI embeddings model via OpenRouter."""
    openai_embeddings = OpenAIEmbeddings(
        openai_api_base=config.OPENROUTER_API_BASE,
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        model=config.EMBEDDING_MODEL,
    )

    return openai_embeddings


def create_vector_stores(
    embeddings: OpenAIEmbeddings,
) -> tuple[Chroma, Chroma]:
    """Initialize persistent ChromaDB collections for static and daily news data."""
    db_storage_dir = Path(config.DB_STORAGE_DIR)
    db_storage_dir.mkdir(parents=True, exist_ok=True)

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


def load_source_text(path: Path) -> str:
    """Read and return the contents of a source text file."""
    source_file_context = path.read_text(encoding="utf-8")

    return source_file_context


def split_into_chunks(text: str) -> list[str]:
    """Split text into overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.INGESTION_CHUNK_SIZE,
        chunk_overlap=config.INGESTION_CHUNK_OVERLAP,
    )

    return splitter.split_text(text)


def build_documents(chunks: list[str]) -> list[Document]:
    """Attach static intelligence metadata to each chunk."""
    metadata = {"competitor": "Snyk", "type": "static"}

    return [Document(page_content=chunk, metadata=metadata) for chunk in chunks]


def ingest_static_intelligence(
    static_store: Chroma,
    documents: list[Document],
) -> int:
    """Add documents to the static intelligence collection."""
    static_store.add_documents(documents)

    return len(documents)


def clear_static_collection(static_store: Chroma) -> None:
    """Safely clear the static collection to ensure ingestion is idempotent."""
    try:
        existing_docs = static_store.get()
        if existing_docs and "ids" in existing_docs and existing_docs["ids"]:
            static_store.delete(ids=existing_docs["ids"])
            print(f"Cleared {len(existing_docs['ids'])} legacy chunk(s) from '{config.STATIC_COLLECTION}' to avoid duplication.")
    except Exception as e:
        print(f"Note on collection cleanup: {e}")


def main() -> None:
    source_file = Path(config.SOURCE_FILE)

    if not source_file.exists():
        raise FileNotFoundError(
            f"Critical Error: Source file not found at '{source_file}'. "
            f"Please ensure the data_input directory and file exist."
        )

    embeddings = create_embeddings()
    static_store, _daily_news_store = create_vector_stores(embeddings)

    clear_static_collection(static_store)

    source_text = load_source_text(source_file)
    chunks = split_into_chunks(source_text)
    documents = build_documents(chunks)

    chunk_count = ingest_static_intelligence(static_store, documents)
    db_storage_dir = Path(config.DB_STORAGE_DIR)

    print(
        f"Successfully added {chunk_count} chunk(s) to "
        f"'{config.STATIC_COLLECTION}' collection in '{db_storage_dir}/'."
    )


if __name__ == "__main__":
    main()
