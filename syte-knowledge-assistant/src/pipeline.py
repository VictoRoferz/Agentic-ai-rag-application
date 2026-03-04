"""
RAG Pipeline — Orchestrates the full Retrieval-Augmented Generation flow.
This is the central entry point: query → retrieve → rerank → generate.
"""

import sys
from pathlib import Path
from typing import Any, Generator, Optional

from loguru import logger

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import Settings, get_settings
from src.ingestion.parsers import parse_document, parse_directory
from src.ingestion.chunker import chunk_document, chunk_documents
from src.ingestion.embedder import EmbeddingService
from src.retrieval.search import SearchService
from src.retrieval.reranker import RerankerService
from src.generation.llm_client import LLMClient
from src.generation.prompts import build_prompt, format_sources_summary


class RAGPipeline:
    """
    Full RAG pipeline: Ingest → Search → Rerank → Generate.
    Initialize once, then call query() for each user request.
    """

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self._initialized = False

        # Components (lazy-loaded)
        self._embedding_service = None
        self._search_service = None
        self._reranker = None
        self._llm_client = None

    def initialize(self):
        """Initialize all pipeline components."""
        if self._initialized:
            return

        s = self.settings
        logger.info("Initializing RAG Pipeline...")

        # 1. Embedding Service (also manages Qdrant collection)
        self._embedding_service = EmbeddingService(
            model_name=s.embedding_model,
            qdrant_host=s.qdrant_host,
            qdrant_port=s.qdrant_port,
            collection_name=s.qdrant_collection,
            vector_size=s.embedding_dimension,
            device=s.embedding_device,
            batch_size=s.embedding_batch_size,
            query_instruction=s.query_instruction,
            passage_instruction=s.passage_instruction,
        )

        # 2. Search Service
        self._search_service = SearchService(
            embedding_service=self._embedding_service,
            qdrant_host=s.qdrant_host,
            qdrant_port=s.qdrant_port,
            collection_name=s.qdrant_collection,
            top_k=s.retrieval_top_k,
            score_threshold=s.score_threshold,
        )

        # 3. Reranker
        if s.reranker_enabled:
            self._reranker = RerankerService(
                model_name=s.reranker_model,
                device=s.embedding_device,
                top_n=s.reranker_top_n,
            )
        else:
            self._reranker = None

        # 4. LLM Client
        self._llm_client = LLMClient(
            provider=s.llm_provider,
            config=s.llm_config,
        )

        self._initialized = True
        logger.info("RAG Pipeline ready!")

    # ── Ingestion ─────────────────────────────────────────────

    def ingest_file(self, file_path: str | Path, is_distractor: bool = False) -> int:
        """Parse, chunk, embed and upload a single document."""
        self.initialize()
        parsed = parse_document(file_path)
        chunks = chunk_document(
            parsed,
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
            separators=self.settings.chunking_separators,
            is_distractor=is_distractor,
        )
        count = self._embedding_service.upload_chunks(chunks)
        return count

    def ingest_directory(self, dir_path: str | Path, is_distractor: bool = False) -> int:
        """Parse and ingest all documents in a directory."""
        self.initialize()
        parsed_docs = parse_directory(dir_path)
        all_chunks = []
        for doc in parsed_docs:
            is_dist = is_distractor or "distractor" in doc["metadata"]["file_path"].lower()
            chunks = chunk_document(
                doc,
                chunk_size=self.settings.chunk_size,
                chunk_overlap=self.settings.chunk_overlap,
                separators=self.settings.chunking_separators,
                is_distractor=is_dist,
            )
            all_chunks.extend(chunks)

        count = self._embedding_service.upload_chunks(all_chunks)
        return count

    # ── Query ─────────────────────────────────────────────────

    def query(
        self,
        question: str,
        mode: str = "output_a",
        top_k: Optional[int] = None,
    ) -> dict[str, Any]:
        """
        Full RAG query: search → rerank → generate.

        Args:
            question: User question in natural language
            mode: "output_a" (narrative 1-pager) or "output_b" (research list)
            top_k: Override retrieval count

        Returns:
            {
                "answer": str,           # Generated text
                "sources": list[dict],   # Used source chunks
                "sources_summary": str,  # Formatted source list
                "mode": str,
                "query": str,
            }
        """
        self.initialize()

        # Step 1: Retrieve
        logger.info(f"Query: '{question[:80]}...' (mode={mode})")
        results = self._search_service.search(question, top_k=top_k)

        if not results:
            return {
                "answer": "Keine relevanten Dokumente gefunden. Bitte prüfe, ob Dokumente in der Wissensbasis indexiert sind.",
                "sources": [],
                "sources_summary": "Keine Quellen gefunden.",
                "mode": mode,
                "query": question,
            }

        # Step 2: Rerank
        if self._reranker:
            results = self._reranker.rerank(question, results)
        else:
            results = results[: self.settings.reranker_top_n]

        # Step 3: Build prompt
        system_prompt, user_message = build_prompt(question, results, mode=mode)

        # Step 4: Generate
        answer = self._llm_client.generate(system_prompt, user_message)

        # Step 5: Format sources
        sources_summary = format_sources_summary(results)

        return {
            "answer": answer,
            "sources": results,
            "sources_summary": sources_summary,
            "mode": mode,
            "query": question,
        }

    def query_stream(
        self,
        question: str,
        mode: str = "output_a",
        top_k: Optional[int] = None,
    ) -> Generator[dict[str, Any], None, None]:
        """
        Streaming RAG query — yields partial results as they arrive.
        First yields sources, then streams the answer token by token.
        """
        self.initialize()

        # Retrieve + Rerank (not streamed)
        results = self._search_service.search(question, top_k=top_k)
        if not results:
            yield {"type": "answer", "content": "Keine relevanten Dokumente gefunden."}
            return

        if self._reranker:
            results = self._reranker.rerank(question, results)
        else:
            results = results[: self.settings.reranker_top_n]

        # Yield sources first
        yield {
            "type": "sources",
            "sources": results,
            "sources_summary": format_sources_summary(results),
        }

        # Build prompt and stream answer
        system_prompt, user_message = build_prompt(question, results, mode=mode)

        for token in self._llm_client.generate_stream(system_prompt, user_message):
            yield {"type": "token", "content": token}

    # ── Info ──────────────────────────────────────────────────

    def get_status(self) -> dict:
        """Get pipeline status."""
        self.initialize()
        info = self._embedding_service.get_collection_info()
        return {
            "collection": info,
            "embedding_model": self.settings.embedding_model,
            "llm_provider": self.settings.llm_provider,
            "reranker_enabled": self.settings.reranker_enabled,
        }

    def reset(self):
        """Reset the knowledge base (delete all vectors)."""
        self.initialize()
        self._embedding_service.delete_collection()
        logger.warning("Knowledge base reset!")
