"""
Document Chunker — Splits parsed documents into chunks with full metadata.
Uses RecursiveCharacterTextSplitter (proven best in 2026 benchmarks).
Each chunk carries: doc_name, page_number, section_title, chunk_index.
"""

from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger


def chunk_document(
    parsed_doc: dict[str, Any],
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    separators: list[str] = None,
    is_distractor: bool = False,
    agenda_relevance: list[str] = None,
) -> list[dict[str, Any]]:
    """
    Chunk a parsed document into smaller pieces with metadata.

    Args:
        parsed_doc: Output from parsers.parse_document()
        chunk_size: Target chunk size in characters (~tokens * 4)
        chunk_overlap: Overlap between chunks
        separators: Text split boundaries
        is_distractor: Mark as test distractor document
        agenda_relevance: Which agenda points this doc is relevant for (test only)

    Returns:
        List of chunk dicts with 'text' and 'metadata' keys.
    """
    if separators is None:
        separators = ["\n\n", "\n", ". ", " "]

    # Character-based sizes (rough: 1 token ≈ 4 chars for English/German)
    char_chunk_size = chunk_size * 4
    char_overlap = chunk_overlap * 4

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=char_chunk_size,
        chunk_overlap=char_overlap,
        separators=separators,
        add_start_index=True,
        strip_whitespace=True,
    )

    chunks = []
    chunk_index = 0

    for page in parsed_doc["pages"]:
        text = page["text"]
        if not text or len(text.strip()) < 20:
            continue

        # Split page text into chunks
        splits = splitter.create_documents(
            texts=[text],
            metadatas=[{
                "doc_id": parsed_doc["doc_id"],
                "doc_name": parsed_doc["doc_name"],
                "doc_type": parsed_doc["doc_type"],
                "page_number": page["page_number"],
                "section_title": page.get("section_title", ""),
                "file_path": parsed_doc["metadata"]["file_path"],
                "ingested_at": parsed_doc["metadata"]["ingested_at"],
                "is_distractor": is_distractor,
                "agenda_relevance": agenda_relevance or [],
            }],
        )

        for split in splits:
            chunk = {
                "text": split.page_content,
                "metadata": {
                    **split.metadata,
                    "chunk_index": chunk_index,
                    "start_index": split.metadata.get("start_index", 0),
                    "chunk_char_length": len(split.page_content),
                },
            }
            chunks.append(chunk)
            chunk_index += 1

    logger.info(
        f"Chunked '{parsed_doc['doc_name']}': "
        f"{len(parsed_doc['pages'])} pages → {len(chunks)} chunks "
        f"(size={chunk_size}t, overlap={chunk_overlap}t)"
    )
    return chunks


def chunk_documents(
    parsed_docs: list[dict[str, Any]],
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    separators: list[str] = None,
) -> list[dict[str, Any]]:
    """Chunk multiple parsed documents."""
    all_chunks = []
    for doc in parsed_docs:
        is_distractor = "distractor" in doc["metadata"].get("file_path", "").lower()
        doc_chunks = chunk_document(
            doc,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
            is_distractor=is_distractor,
        )
        all_chunks.extend(doc_chunks)

    logger.info(f"Total: {len(parsed_docs)} documents → {len(all_chunks)} chunks")
    return all_chunks
