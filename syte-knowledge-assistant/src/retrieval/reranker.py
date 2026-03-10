"""
Reranker — Cross-encoder reranking for precision.
Takes initial retrieval results and re-scores them using a cross-encoder
that sees query + document together (much more accurate than bi-encoder).
"""

from typing import Any, Optional

from loguru import logger


class RerankerService:
    """Rerank search results using a cross-encoder model."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Reranker-0.6B",
        device: str = "cpu",
        top_n: int = 7,
    ):
        self.top_n = top_n
        self.model_name = model_name
        self._model = None
        self._device = device

        logger.info(f"RerankerService configured: model={model_name}, top_n={top_n}")

    def _load_model(self):
        """Lazy-load the reranker model."""
        if self._model is not None:
            return

        logger.info(f"Loading reranker model: {self.model_name}")

        try:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(
                self.model_name,
                trust_remote_code=True,
                device=self._device,
            )
            # Fix: Qwen3 tokenizer has no padding token
            if self._model.tokenizer.pad_token is None:
                self._model.tokenizer.pad_token = self._model.tokenizer.eos_token
                self._model.tokenizer.pad_token_id = self._model.tokenizer.eos_token_id
                self._model.tokenizer.padding_side = "right"
            self._model_type = "cross_encoder"
            logger.info("Reranker loaded (CrossEncoder)")
        except Exception as e:
            logger.warning(
                f"CrossEncoder load failed ({e}). "
                f"Falling back to transformers AutoModelForSequenceClassification."
            )
            try:
                from transformers import AutoModelForSequenceClassification, AutoTokenizer
                import torch

                self._tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name, trust_remote_code=True
                )
                self._hf_model = AutoModelForSequenceClassification.from_pretrained(
                    self.model_name, trust_remote_code=True
                ).to(self._device)
                self._hf_model.eval()
                self._model_type = "hf_classifier"
                self._model = True  # Mark as loaded
                logger.info("Reranker loaded (HF classifier)")
            except Exception as e2:
                logger.error(f"Reranker load failed completely: {e2}")
                self._model = "disabled"
                self._model_type = "disabled"

    def rerank(
        self,
        query: str,
        results: list[dict[str, Any]],
        top_n: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        """
        Rerank search results by cross-encoder score.

        Args:
            query: The user query
            results: List of search results (each with 'text' and 'metadata')
            top_n: Number of top results to return

        Returns:
            Reranked results with added 'rerank_score' field.
        """
        self._load_model()
        n = top_n or self.top_n

        if not results:
            return []

        if self._model_type == "disabled":
            logger.warning("Reranker disabled, returning original order")
            return results[:n]

        texts = [r["text"] for r in results]
        pairs = [[query, text] for text in texts]

        try:
            if self._model_type == "cross_encoder":
                scores = self._model.predict(pairs, show_progress_bar=False)
                scores = scores.tolist() if hasattr(scores, "tolist") else list(scores)

            elif self._model_type == "hf_classifier":
                import torch
                scores = []
                batch_size = 16
                for i in range(0, len(pairs), batch_size):
                    batch = pairs[i:i + batch_size]
                    inputs = self._tokenizer(
                        batch,
                        padding=True,
                        truncation=True,
                        max_length=512,
                        return_tensors="pt",
                    ).to(self._device)
                    with torch.no_grad():
                        outputs = self._hf_model(**inputs)
                    logits = outputs.logits.squeeze(-1)
                    scores.extend(logits.cpu().tolist())

        except Exception as e:
            logger.error(f"Reranking failed: {e}. Returning original order.")
            return results[:n]

        # Attach scores and sort
        for result, score in zip(results, scores):
            result["rerank_score"] = float(score)

        reranked = sorted(results, key=lambda x: x["rerank_score"], reverse=True)

        logger.info(
            f"Reranked {len(results)} results → top {n} "
            f"(best: {reranked[0]['rerank_score']:.3f}, "
            f"doc: {reranked[0]['metadata']['doc_name']})"
        )
        return reranked[:n]
