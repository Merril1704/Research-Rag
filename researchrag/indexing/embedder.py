"""
Embedder — Wraps SentenceTransformers for chunk and query embedding.

Provides a swappable interface for generating embeddings.
Model name is pulled from config and can be changed without modifying this code.

Usage:
    from researchrag.indexing.embedder import Embedder

    embedder = Embedder()
    vectors = embedder.embed_texts(["text1", "text2"])
    query_vec = embedder.embed_query("my search query")
"""

import numpy as np
from sentence_transformers import SentenceTransformer

from researchrag.config import EMBEDDING_MODEL, EMBEDDING_BATCH_SIZE
from researchrag.logger import get_logger

logger = get_logger(__name__)


class Embedder:
    """
    Embedding wrapper around SentenceTransformers.

    Attributes:
        model_name: Name/path of the SentenceTransformer model.
        model: Loaded SentenceTransformer instance.
        dimension: Embedding vector dimension.
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        """
        Initialize the embedder.

        Args:
            model_name: HuggingFace model name or local path.
        """
        self.model_name = model_name
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        logger.info(
            f"Embedding model loaded: dim={self.dimension}, "
            f"max_seq_length={self.model.max_seq_length}"
        )

    def embed_texts(
        self, texts: list[str], batch_size: int = EMBEDDING_BATCH_SIZE
    ) -> np.ndarray:
        """
        Batch-embed a list of text strings.

        Args:
            texts: List of text strings to embed.
            batch_size: Number of texts per batch.

        Returns:
            numpy array of shape (len(texts), dimension), L2-normalized.
        """
        if not texts:
            logger.warning("embed_texts called with empty list")
            return np.array([]).reshape(0, self.dimension)

        logger.info(f"Embedding {len(texts)} texts (batch_size={batch_size})")
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,  # L2 normalize for cosine similarity via dot product
        )
        logger.debug(f"Embeddings shape: {embeddings.shape}")
        return np.array(embeddings, dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a single query string.

        Args:
            query: Query text.

        Returns:
            numpy array of shape (dimension,), L2-normalized.
        """
        logger.debug(f"Embedding query: '{query[:80]}...'")
        embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
        )
        return np.array(embedding[0], dtype=np.float32)
