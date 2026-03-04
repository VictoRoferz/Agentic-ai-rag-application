"""
Search — Hybrid vector search in Qdrant with metadata filtering.
Retrieves top-K chunks with scores for downstream reranking.
"""

from typing import Any, Optional

from loguru import logger
from qdrant_client import QdrantClient, models


class SearchService:
    """Semantic search over the Qdrant knowledge base."""

    def __init__(
        self,
        embedding_service,  # EmbeddingService instance (for query embedding)
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        collection_name: str = "syte_knowledge_base",
        top_k: int = 25,
        score_threshold: float = 0.30,
    ):
        self.embedding_service = embedding_service
        self.collection_name = collection_name
        self.top_k = top_k
        self.score_threshold = score_threshold

        self.client = QdrantClient(host=qdrant_host, port=qdrant_port)
        logger.info(
            f"SearchService ready: collection='{collection_name}', "
            f"top_k={top_k}, threshold={score_threshold}"
        )

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        doc_type_filter: Optional[str] = None,
        exclude_distractors: bool = False,
        score_threshold: Optional[float] = None,
    ) -> list[dict[str, Any]]:
        """
        Search the knowledge base.

        Args:
            query: Natural language query
            top_k: Override default top_k
            doc_type_filter: Filter by doc type ("pdf", "docx", "pptx")
            exclude_distractors: Filter out distractor documents
            score_threshold: Override default threshold

        Returns:
            List of result dicts sorted by score (highest first):
            [{"text", "metadata", "score", "id"}, ...]
        """
        k = top_k or self.top_k
        threshold = score_threshold or self.score_threshold

        # Embed the query
        query_vector = self.embedding_service.embed_query(query)

        # Build filters
        conditions = []
        if doc_type_filter:
            conditions.append(
                models.FieldCondition(
                    key="doc_type",
                    match=models.MatchValue(value=doc_type_filter),
                )
            )
        if exclude_distractors:
            conditions.append(
                models.FieldCondition(
                    key="is_distractor",
                    match=models.MatchValue(value=False),
                )
            )

        query_filter = None
        if conditions:
            query_filter = models.Filter(must=conditions)

        # Execute search
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=query_filter,
            limit=k,
            score_threshold=threshold,
            with_payload=True,
        )

        # Format results
        formatted = []
        for point in results.points:
            payload = point.payload or {}
            formatted.append({
                "id": point.id,
                "score": point.score,
                "text": payload.get("text", ""),
                "metadata": {
                    "doc_id": payload.get("doc_id", ""),
                    "doc_name": payload.get("doc_name", ""),
                    "doc_type": payload.get("doc_type", ""),
                    "page_number": payload.get("page_number", 0),
                    "section_title": payload.get("section_title", ""),
                    "chunk_index": payload.get("chunk_index", 0),
                    "is_distractor": payload.get("is_distractor", False),
                    "file_path": payload.get("file_path", ""),
                },
            })

        logger.info(
            f"Search '{query[:60]}...' → {len(formatted)} results "
            f"(top score: {formatted[0]['score']:.3f})" if formatted else
            f"Search '{query[:60]}...' → 0 results"
        )
        return formatted

    def search_with_diversity(
        self,
        query: str,
        top_k: Optional[int] = None,
        mmr_lambda: float = 0.7,
    ) -> list[dict[str, Any]]:
        """
        Search with MMR (Maximal Marginal Relevance) for diverse results.
        Avoids returning multiple chunks from the same section.
        """
        # Get more results than needed, then diversify
        k = top_k or self.top_k
        raw_results = self.search(query, top_k=k * 2)

        if not raw_results:
            return []

        # Simple MMR: penalize results from same document + page
        selected = [raw_results[0]]
        seen_pages = {(raw_results[0]["metadata"]["doc_name"], raw_results[0]["metadata"]["page_number"])}

        for result in raw_results[1:]:
            if len(selected) >= k:
                break

            page_key = (result["metadata"]["doc_name"], result["metadata"]["page_number"])

            if page_key in seen_pages:
                # Apply diversity penalty
                result["score"] *= (1 - mmr_lambda) + mmr_lambda * 0.5
            else:
                seen_pages.add(page_key)

            selected.append(result)

        # Re-sort by adjusted scores
        selected.sort(key=lambda x: x["score"], reverse=True)
        return selected[:k]
