"""Ingestion module — Document parsing, chunking, and embedding."""
from src.ingestion.parsers import parse_document, parse_directory
from src.ingestion.chunker import chunk_document, chunk_documents
from src.ingestion.embedder import EmbeddingService
