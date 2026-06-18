"""
Gap Finder — Generates a research gap analysis report from retrieved evidence.

Usage:
    from researchrag.generation.gap_finder import find_gaps

    result = find_gaps(query, retrieved_chunks, llm_client)
"""

from researchrag.generation.llm_client import LLMClient
from researchrag.generation.prompts import build_gap_analysis_prompt
from researchrag.logger import get_logger

logger = get_logger(__name__)


def find_gaps(
    query: str,
    retrieved_chunks: list[dict],
    llm_client: LLMClient,
    custom_instructions: str | None = None,
) -> dict:
    """
    Generate a research gap analysis report from retrieved evidence.

    Args:
        query: User's research query.
        retrieved_chunks: List of chunk dicts from retriever.
        llm_client: LLMClient instance for generation.
        custom_instructions: Optional user instructions for tone/focus.

    Returns:
        Dict with: mode, answer_text, citations, used_chunks.
    """
    logger.info(f"Finding gaps for query: '{query[:80]}...'")

    if not retrieved_chunks:
        logger.warning("No retrieved chunks provided for gap analysis")
        return {
            "mode": "gap_analysis",
            "answer_text": "No relevant evidence was retrieved for this query.",
            "citations": [],
            "used_chunks": [],
        }

    # Build prompt and generate
    prompt = build_gap_analysis_prompt(query, retrieved_chunks, custom_instructions)
    answer_text = llm_client.generate(prompt)

    # Extract citation metadata from used chunks
    citations = []
    used_chunk_ids = []
    seen_papers = set()

    for chunk in retrieved_chunks:
        chunk_id = chunk.get("chunk_id", "")
        paper_title = chunk.get("paper_title", "Unknown")
        used_chunk_ids.append(chunk_id)

        if paper_title not in seen_papers:
            seen_papers.add(paper_title)
            citations.append({
                "paper_title": paper_title,
                "paper_id": chunk.get("paper_id", ""),
                "authors": chunk.get("authors"),
                "year": chunk.get("year"),
                "section_type": chunk.get("section_type", ""),
                "page_number": chunk.get("page_number"),
            })

    result = {
        "mode": "gap_analysis",
        "answer_text": answer_text,
        "citations": citations,
        "used_chunks": used_chunk_ids,
    }

    logger.info(
        f"Gap analysis complete: {len(answer_text)} chars, "
        f"{len(citations)} unique papers cited"
    )
    return result
