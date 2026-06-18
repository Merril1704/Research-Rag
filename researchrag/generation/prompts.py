"""
Prompt Builder — Centralized prompt construction for both generation modes.

Templates are derived from the project's Docs/gap_analysis.md and
Docs/related_work.md specifications. All prompt logic lives here;
no other module should contain hard-coded prompts.

Usage:
    from researchrag.generation.prompts import (
        build_related_work_prompt,
        build_gap_analysis_prompt,
    )

    prompt = build_related_work_prompt("my query", retrieved_chunks)
"""

from researchrag.logger import get_logger

logger = get_logger(__name__)


def _format_evidence(chunks: list[dict]) -> str:
    """Format retrieved chunks into a numbered evidence block for the LLM."""
    evidence_parts = []
    for i, chunk in enumerate(chunks, 1):
        title = chunk.get("paper_title", "Unknown Paper")
        section = chunk.get("section_heading") or chunk.get("section_type", "N/A")
        page = chunk.get("page_number", "N/A")
        text = chunk.get("chunk_text", "")

        evidence_parts.append(
            f"[Evidence {i}]\n"
            f"Paper: {title}\n"
            f"Section: {section}\n"
            f"Page: {page}\n"
            f"Text: {text}\n"
        )
    return "\n---\n".join(evidence_parts)


def build_related_work_prompt(
    query: str, chunks: list[dict], custom_instructions: str | None = None
) -> str:
    """
    Build the Related Work Drafter prompt.

    Args:
        query: User's research query.
        chunks: Retrieved evidence chunks with metadata.
        custom_instructions: Optional user instructions for tone/focus.

    Returns:
        Complete prompt string for the LLM.
    """
    evidence = _format_evidence(chunks)
    num_papers = len(set(c.get("paper_title", "") for c in chunks))

    custom_block = ""
    if custom_instructions and custom_instructions.strip():
        custom_block = f"\n## Custom User Instructions\n{custom_instructions.strip()}\n"

    prompt = f"""You are an academic writing assistant drafting a related work section.

## Task
Given a user query and a set of retrieved passages from research papers, write a concise, well-structured related work discussion grounded only in the provided evidence.

## Instructions
- Use only the retrieved evidence.
- Do not add unsupported background knowledge.
- Organize the literature thematically rather than paper-by-paper when possible.
- Compare methods, assumptions, findings, or limitations only when supported by the retrieved text.
- Maintain an academic tone suitable for a project report or literature review.
{custom_block}
## Output requirements
Produce:
1. an opening framing sentence for the literature area,
2. one or more thematic paragraphs,
3. clear attribution for synthesized claims,
4. smooth transitions across themes,
5. no unsupported claims.

## Citation rule
Every synthesized point should be attributable to one or more retrieved chunks or source papers. Use inline citations like (Author, Year) or [Paper Title] based on the evidence metadata.

## Style rule
Write prose that reads like a related work section, not like bullet-point notes.

## User Query
{query}

## Retrieved Evidence ({len(chunks)} chunks from {num_papers} papers)
{evidence}

## Your Response
Write the related work section below:"""

    logger.debug(f"Related Work prompt built: {len(prompt)} chars, {len(chunks)} chunks")
    return prompt


def build_gap_analysis_prompt(
    query: str, chunks: list[dict], custom_instructions: str | None = None
) -> str:
    """
    Build the Gap Analysis / Gap Finder prompt.

    Args:
        query: User's research query.
        chunks: Retrieved evidence chunks with metadata.
        custom_instructions: Optional user instructions for tone/focus.

    Returns:
        Complete prompt string for the LLM.
    """
    evidence = _format_evidence(chunks)
    num_papers = len(set(c.get("paper_title", "") for c in chunks))

    custom_block = ""
    if custom_instructions and custom_instructions.strip():
        custom_block = f"\n## Custom User Instructions\n{custom_instructions.strip()}\n"

    prompt = f"""You are an academic literature analysis assistant.

## Task
Given a user query and a set of retrieved passages from research papers, identify research gaps, recurring limitations, and underexplored directions that are supported by the evidence.

## Instructions
- Use only the retrieved evidence.
- Do not invent claims that are not supported by the passages.
- Prefer explicit limitations and future-work statements when present.
- Group similar limitations across papers into themes.
- Distinguish between:
  - directly stated limitations,
  - inferred cross-paper opportunities.
- If evidence is weak or incomplete, say so clearly.
- Keep the tone academic and analytical.
{custom_block}
## Output requirements
Produce:
1. a short overview,
2. a set of thematic gap areas,
3. evidence-backed explanation for each area,
4. source attribution for each claim,
5. a final section listing promising future directions.

## Citation rule
Every meaningful claim should include source attribution based on the retrieved chunks. Use inline citations like (Author, Year) or [Paper Title] based on the evidence metadata.

## Safety rule
If the retrieved context does not support a strong conclusion, avoid overclaiming novelty or certainty.

## User Query
{query}

## Retrieved Evidence ({len(chunks)} chunks from {num_papers} papers)
{evidence}

## Your Response
Write the gap analysis report below:"""

    logger.debug(f"Gap Analysis prompt built: {len(prompt)} chars, {len(chunks)} chunks")
    return prompt
