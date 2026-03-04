"""
Configuration loader for Syte Knowledge Assistant.
Loads settings.yaml and .env, provides typed access to all config values.
"""

import os
from pathlib import Path
from typing import Any, Optional

import yaml
from dotenv import load_dotenv
from loguru import logger


# Project root = parent of config/
PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"


def load_config(config_path: Optional[str] = None) -> dict:
    """Load settings.yaml and merge with environment variables."""

    # Load .env file
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        logger.info(f"Loaded .env from {env_path}")
    else:
        logger.warning(f"No .env file found at {env_path}. Copy .env.template to .env and add your API keys.")

    # Load settings.yaml
    if config_path is None:
        config_path = CONFIG_DIR / "settings.yaml"

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    logger.info(f"Loaded config from {config_path}")
    return config


class Settings:
    """Typed access to configuration values."""

    def __init__(self, config: Optional[dict] = None):
        if config is None:
            config = load_config()
        self._config = config

    # --- Qdrant ---
    @property
    def qdrant_host(self) -> str:
        return os.getenv("QDRANT_URL", self._config["qdrant"]["host"])

    @property
    def qdrant_port(self) -> int:
        return self._config["qdrant"]["port"]

    @property
    def qdrant_collection(self) -> str:
        return self._config["qdrant"]["collection_name"]

    @property
    def vector_size(self) -> int:
        return self._config["qdrant"]["vector_size"]

    @property
    def distance_metric(self) -> str:
        return self._config["qdrant"]["distance"]

    # --- Embedding ---
    @property
    def embedding_model(self) -> str:
        return self._config["embedding"]["model_name"]

    @property
    def embedding_dimension(self) -> int:
        return self._config["embedding"]["dimension"]

    @property
    def embedding_device(self) -> str:
        return self._config["embedding"]["device"]

    @property
    def embedding_batch_size(self) -> int:
        return self._config["embedding"]["batch_size"]

    @property
    def query_instruction(self) -> str:
        return self._config["embedding"].get("query_instruction", "")

    @property
    def passage_instruction(self) -> str:
        return self._config["embedding"].get("passage_instruction", "")

    # --- Reranker ---
    @property
    def reranker_enabled(self) -> bool:
        return self._config["reranker"]["enabled"]

    @property
    def reranker_model(self) -> str:
        return self._config["reranker"]["model_name"]

    @property
    def reranker_top_n(self) -> int:
        return self._config["reranker"]["top_n"]

    # --- LLM ---
    @property
    def llm_provider(self) -> str:
        return self._config["llm"]["provider"]

    @property
    def llm_config(self) -> dict:
        provider = self.llm_provider
        return self._config["llm"].get(provider, {})

    @property
    def anthropic_api_key(self) -> str:
        return os.getenv("ANTHROPIC_API_KEY", "")

    # --- Chunking ---
    @property
    def chunk_size(self) -> int:
        return self._config["chunking"]["chunk_size"]

    @property
    def chunk_overlap(self) -> int:
        return self._config["chunking"]["chunk_overlap"]

    @property
    def chunking_separators(self) -> list:
        return self._config["chunking"]["separators"]

    # --- Retrieval ---
    @property
    def retrieval_top_k(self) -> int:
        return self._config["retrieval"]["top_k"]

    @property
    def dense_weight(self) -> float:
        return self._config["retrieval"]["dense_weight"]

    @property
    def sparse_weight(self) -> float:
        return self._config["retrieval"]["sparse_weight"]

    @property
    def score_threshold(self) -> float:
        return self._config["retrieval"]["score_threshold"]

    # --- Ingestion ---
    @property
    def supported_formats(self) -> list:
        return self._config["ingestion"]["supported_formats"]

    @property
    def watch_folder(self) -> Path:
        return PROJECT_ROOT / self._config["ingestion"]["watch_folder"]

    @property
    def indexed_folder(self) -> Path:
        return PROJECT_ROOT / self._config["ingestion"]["indexed_folder"]

    # --- Paths ---
    @property
    def data_dir(self) -> Path:
        return DATA_DIR

    @property
    def project_root(self) -> Path:
        return PROJECT_ROOT


# Global settings instance (lazy-loaded)
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get or create global settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
