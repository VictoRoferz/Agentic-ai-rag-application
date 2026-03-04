"""
Embedder — Generates vector embeddings and uploads chunks to Qdrant.
Handles collection creation, embedding generation, and batch upsert.
"""

import uuid
from typing import Any

from loguru import logger
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer


class EmbeddingService:
    """Manages the embedding model and Qdrant collection."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Embedding-0.6B",
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        collection_name: str = "syte_knowledge_base",
        vector_size: int = 1024,
        device: str = "cpu",
        batch_size: int = 32,
        query_instruction: str = "",
        passage_instruction: str = "",
    ):
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.batch_size = batch_size
        self.query_instruction = query_instruction
        self.passage_instruction = passage_instruction

        # Load embedding model
        logger.info(f"Loading embedding model: {model_name} (device={device})")
        self.model = SentenceTransformer(
            model_name,
            trust_remote_code=True,
            device=device,
        )

        actual_dim = self.model.get_sentence_embedding_dimension()
        if actual_dim != vector_size:
            logger.warning(
                f"Model dimension ({actual_dim}) != config ({vector_size}). "
                f"Using model dimension: {actual_dim}"
            )
            self.vector_size = actual_dim

        # Connect to Qdrant
        logger.info(f"Connecting to Qdrant at {qdrant_host}:{qdrant_port}")
        self.client = QdrantClient(host=qdrant_host, port=qdrant_port)
        self._ensure_collection()

    def _ensure_collection(self):
        """Create the Qdrant collection if it doesn't exist."""
        collections = [c.name for c in self.client.get_collections().collections]

        if self.collection_name in collections:
            info = self.client.get_collection(self.collection_name)
            logger.info(
                f"Collection '{self.collection_name}' exists: "
                f"{info.points_count} points"
            )
            return

        logger.info(f"Creating collection '{self.collection_name}' (dim={self.vector_size})")
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=self.vector_size,
                distance=models.Distance.COSINE,
            ),
        )

        # Create payload indexes for filtering
        for field, schema in [
            ("doc_name", models.PayloadSchemaType.KEYWORD),
            ("doc_type", models.PayloadSchemaType.KEYWORD),
            ("page_number", models.PayloadSchemaType.INTEGER),
            ("is_distractor", models.PayloadSchemaType.BOOL),
        ]:
            try:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field,
                    field_schema=schema,
                )
            except Exception:
                pass  # Index may already exist

        logger.info(f"Collection '{self.collection_name}' created with indexes")

    def embed_texts(self, texts: list[str], is_query: bool = False) -> list[list[float]]:
        """Generate embeddings for a list of texts."""
        if is_query and self.query_instruction:
            texts = [self.query_instruction + t for t in texts]
        elif not is_query and self.passage_instruction:
            texts = [self.passage_instruction + t for t in texts]

        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=len(texts) > 10,
            normalize_embeddings=True,
        )
        return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        """Generate embedding for a single query."""
        return self.embed_texts([query], is_query=True)[0]

    def upload_chunks(self, chunks: list[dict[str, Any]]) -> int:
        """
        Embed and upload chunks to Qdrant.

        Args:
            chunks: List of chunk dicts with 'text' and 'metadata' keys.

        Returns:
            Number of uploaded points.
        """
        if not chunks:
            logger.warning("No chunks to upload")
            return 0

        texts = [c["text"] for c in chunks]
        logger.info(f"Embedding {len(texts)} chunks...")
        embeddings = self.embed_texts(texts, is_query=False)

        # Build Qdrant points
        points = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            point_id = str(uuid.uuid4())
            payload = {
                **chunk["metadata"],
                "text": chunk["text"],  # Store text in payload for retrieval
            }
            # Clean payload: convert non-serializable types
            for k, v in payload.items():
                if isinstance(v, set):
                    payload[k] = list(v)

            points.append(models.PointStruct(
                id=point_id,
                vector=embedding,
                payload=payload,
            ))

        # Batch upsert
        batch_size = 100
        total_uploaded = 0
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            self.client.upsert(
                collection_name=self.collection_name,
                points=batch,
            )
            total_uploaded += len(batch)
            if len(points) > batch_size:
                logger.info(f"Uploaded {total_uploaded}/{len(points)} points...")

        logger.info(f"Upload complete: {total_uploaded} points → '{self.collection_name}'")
        return total_uploaded

    def get_collection_info(self) -> dict:
        """Get collection stats."""
        info = self.client.get_collection(self.collection_name)
        return {
            "name": self.collection_name,
            "points_count": info.points_count,
            "vectors_count": getattr(info, "vectors_count", info.points_count),
            "status": info.status.value if hasattr(info.status, "value") else str(info.status),
        }

    def delete_collection(self):
        """Delete the collection (for reset/testing)."""
        self.client.delete_collection(self.collection_name)
        logger.warning(f"Deleted collection '{self.collection_name}'")
        self._ensure_collection()
