"""
Retriever — Dual-mode retrieval over the FAISS index.

Supports both Related Work (standard dense) and Gap Analysis
(section-aware filtering/boosting) modes through a single interface.

Usage:
    from researchrag.retrieval.retriever import Retriever

    retriever = Retriever(embedder, vector_store)
    results = retriever.retrieve("my query", mode="related_work", top_k=10)
    results = retriever.retrieve("my query", mode="gap_analysis", top_k=10)
"""

from researchrag.config import (
    TOP_K_RELATED_WORK,
    TOP_K_GAP_ANALYSIS,
    GAP_SECTION_TYPES,
    MIN_GAP_RESULTS,
)
from researchrag.indexing.embedder import Embedder
from researchrag.indexing.vector_store import VectorStore
from researchrag.logger import get_logger

logger = get_logger(__name__)

VALID_MODES = {"related_work", "gap_analysis"}


class Retriever:
    """
    Unified retriever for both app modes.

    Attributes:
        embedder: Embedder instance for query embedding.
        store: VectorStore instance for similarity search.
    """

    def __init__(self, embedder: Embedder, store: VectorStore):
        self.embedder = embedder
        self.store = store

    def retrieve(
        self,
        query: str,
        mode: str = "related_work",
        top_k: int | None = None,
    ) -> list[dict]:
        """
        Retrieve top-k relevant chunks for a query.

        Args:
            query: User query string.
            mode: "related_work" or "gap_analysis".
            top_k: Number of results (uses config defaults if None).

        Returns:
            Ranked list of chunk dicts with scores and metadata.

        Raises:
            ValueError: If mode is not recognized.
        """
        if mode not in VALID_MODES:
            raise ValueError(f"Invalid mode '{mode}'. Must be one of {VALID_MODES}")

        if mode == "related_work":
            return self._retrieve_related_work(query, top_k)
        else:
            return self._retrieve_gap_analysis(query, top_k)

    def _retrieve_related_work(
        self, query: str, top_k: int | None = None
    ) -> list[dict]:
        """Standard dense top-k retrieval for Related Work mode."""
        k = top_k or TOP_K_RELATED_WORK

        logger.info(f"[Related Work] Retrieving top-{k} for: '{query[:80]}...'")

        query_embedding = self.embedder.embed_query(query)
        results = self.store.search(query_embedding, top_k=k)

        self._log_results("related_work", query, results, filters_applied=None)
        return results

    def _retrieve_gap_analysis(
        self, query: str, top_k: int | None = None
    ) -> list[dict]:
        """
        Section-aware retrieval for Gap Analysis mode.

        Strategy:
        1. Retrieve a larger pool (top_k * 3).
        2. Filter/boost results from limitation-oriented sections.
        3. If too few filtered results, fall back to broader retrieval.
        """
        k = top_k or TOP_K_GAP_ANALYSIS
        pool_size = k * 3  # over-retrieve for filtering

        logger.info(
            f"[Gap Analysis] Retrieving pool of {pool_size} for: '{query[:80]}...'"
        )

        query_embedding = self.embedder.embed_query(query)
        pool = self.store.search(query_embedding, top_k=pool_size)

        # Filter to gap-relevant sections
        filtered = [
            r for r in pool
            if r.get("section_type", "other") in GAP_SECTION_TYPES
        ]

        logger.debug(
            f"Gap filtering: {len(pool)} pool -> {len(filtered)} from "
            f"sections {GAP_SECTION_TYPES}"
        )

        # Fallback: if too few filtered results, use the full pool
        if len(filtered) < MIN_GAP_RESULTS:
            logger.info(
                f"Only {len(filtered)} gap-section results (min={MIN_GAP_RESULTS}). "
                f"Falling back to broader retrieval."
            )
            # Boost gap-section results by putting them first, then others
            boosted = filtered + [
                r for r in pool
                if r.get("section_type", "other") not in GAP_SECTION_TYPES
            ]
            results = boosted[:k]
            self._log_results("gap_analysis", query, results,
                              filters_applied="fallback_boosted")
        else:
            results = filtered[:k]
            self._log_results("gap_analysis", query, results,
                              filters_applied=str(GAP_SECTION_TYPES))

        return results

    def _log_results(
        self,
        mode: str,
        query: str,
        results: list[dict],
        filters_applied: str | None,
    ) -> None:
        """Log retrieval details for debugging and evaluation."""
        logger.info(
            f"Retrieval complete | mode={mode} | "
            f"query='{query[:60]}' | "
            f"results={len(results)} | "
            f"filters={filters_applied or 'none'}"
        )
        for i, r in enumerate(results[:5]):  # log top-5 details
            logger.debug(
                f"  [{i+1}] score={r['score']:.4f} | "
                f"paper='{r.get('paper_title', 'N/A')[:40]}' | "
                f"section={r.get('section_type', '?')} | "
                f"chunk_id={r.get('chunk_id', '?')}"
            )
