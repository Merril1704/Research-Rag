"""
Vector Store — FAISS-backed local vector index with metadata persistence.

Builds, saves, loads, and searches a FAISS index.
Chunk metadata is stored separately as JSON for traceability.

Usage:
    from researchrag.indexing.vector_store import VectorStore

    store = VectorStore()
    store.build(embeddings, metadata_list)
    store.save("data/index/faiss_index.bin", "data/index/chunk_metadata.json")
    store.load("data/index/faiss_index.bin", "data/index/chunk_metadata.json")
    results = store.search(query_embedding, top_k=10)
"""

import json
from pathlib import Path

import faiss
import numpy as np

from researchrag.config import FAISS_INDEX_FILE, METADATA_FILE, INDEX_DIR
from researchrag.logger import get_logger

logger = get_logger(__name__)


class VectorStore:
    """
    Local FAISS vector index with JSON metadata persistence.

    Attributes:
        index: FAISS index instance.
        metadata: List of chunk metadata dicts (aligned by row index).
        dimension: Embedding dimension.
    """

    def __init__(self):
        self.index: faiss.Index | None = None
        self.metadata: list[dict] = []
        self.dimension: int | None = None

    def build(self, embeddings: np.ndarray, metadata: list[dict]) -> None:
        """
        Build a FAISS index from embeddings and aligned metadata.

        Args:
            embeddings: numpy array of shape (n, dimension), L2-normalized.
            metadata: List of chunk metadata dicts, same length as embeddings.

        Raises:
            ValueError: If embeddings and metadata lengths don't match.
        """
        if len(embeddings) != len(metadata):
            raise ValueError(
                f"Embeddings ({len(embeddings)}) and metadata ({len(metadata)}) "
                f"length mismatch"
            )

        self.dimension = embeddings.shape[1]
        # Inner product on L2-normalized vectors = cosine similarity
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings)
        self.metadata = metadata

        logger.info(
            f"FAISS index built: {self.index.ntotal} vectors, "
            f"dim={self.dimension}"
        )

    def save(
        self,
        index_path: str | None = None,
        metadata_path: str | None = None,
    ) -> None:
        """
        Save FAISS index and metadata to disk.

        Args:
            index_path: Path for the FAISS index file.
            metadata_path: Path for the JSON metadata file.
        """
        if self.index is None:
            raise RuntimeError("No index to save. Call build() first.")

        index_path = Path(index_path or (INDEX_DIR / FAISS_INDEX_FILE))
        metadata_path = Path(metadata_path or (INDEX_DIR / METADATA_FILE))

        # Ensure directories exist
        index_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(index_path))
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)

        logger.info(
            f"Index saved: {index_path} ({self.index.ntotal} vectors), "
            f"metadata: {metadata_path}"
        )

    def load(
        self,
        index_path: str | None = None,
        metadata_path: str | None = None,
    ) -> None:
        """
        Load FAISS index and metadata from disk.

        Args:
            index_path: Path to the FAISS index file.
            metadata_path: Path to the JSON metadata file.

        Raises:
            FileNotFoundError: If index or metadata file doesn't exist.
            ValueError: If index and metadata sizes don't match.
        """
        index_path = Path(index_path or (INDEX_DIR / FAISS_INDEX_FILE))
        metadata_path = Path(metadata_path or (INDEX_DIR / METADATA_FILE))

        if not index_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {index_path}")
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        self.index = faiss.read_index(str(index_path))
        with open(metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        self.dimension = self.index.d

        if self.index.ntotal != len(self.metadata):
            logger.error(
                f"Index/metadata mismatch: {self.index.ntotal} vectors vs "
                f"{len(self.metadata)} metadata entries"
            )
            raise ValueError("FAISS index and metadata size mismatch")

        logger.info(
            f"Index loaded: {self.index.ntotal} vectors, dim={self.dimension}"
        )

    def search(self, query_embedding: np.ndarray, top_k: int = 10) -> list[dict]:
        """
        Search the index for the top-k most similar chunks.

        Args:
            query_embedding: Query vector of shape (dimension,).
            top_k: Number of results to return.

        Returns:
            List of result dicts with score and full chunk metadata.

        Raises:
            RuntimeError: If no index is loaded.
        """
        if self.index is None:
            raise RuntimeError("No index loaded. Call build() or load() first.")

        # Reshape for FAISS (expects 2D array)
        query = query_embedding.reshape(1, -1).astype(np.float32)
        actual_k = min(top_k, self.index.ntotal)

        scores, indices = self.index.search(query, actual_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:  # FAISS returns -1 for empty slots
                continue
            result = {
                "score": float(score),
                **self.metadata[idx],
            }
            results.append(result)

        logger.debug(
            f"Search returned {len(results)} results "
            f"(top score: {results[0]['score']:.4f})" if results else
            f"Search returned 0 results"
        )

        return results

    @property
    def is_loaded(self) -> bool:
        """Check if an index is currently loaded."""
        return self.index is not None and self.index.ntotal > 0
