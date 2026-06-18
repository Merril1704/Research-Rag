"""
Chunker — Section detection, normalization, and overlapping text chunking.

Splits parsed paper text into overlapping chunks with section metadata.
Uses LangChain RecursiveCharacterTextSplitter for the splitting logic.

Usage:
    from researchrag.indexing.chunker import chunk_paper

    chunks = chunk_paper(paper_doc, chunk_size=1500, chunk_overlap=200)
"""

import re
from langchain_text_splitters import RecursiveCharacterTextSplitter

from researchrag.config import CHUNK_SIZE, CHUNK_OVERLAP, SECTION_TYPES
from researchrag.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Section heading patterns (common academic paper formats)
# ---------------------------------------------------------------------------
# Matches numbered headings like "1. Introduction", "3.2 Methods"
# or standalone ALL-CAPS headings like "ABSTRACT", "METHODOLOGY"
HEADING_PATTERNS = [
    # Numbered: "1. Introduction", "2.1 Background", "III. Methods"
    re.compile(
        r"^(?:[IVXivx]+\.?\s+|[0-9]+\.(?:[0-9]+\.?)*\s+)"
        r"([A-Z][A-Za-z\s&:,\-]+)$",
        re.MULTILINE,
    ),
    # All-caps standalone: "ABSTRACT", "INTRODUCTION"
    re.compile(r"^([A-Z][A-Z\s&:,\-]{3,})$", re.MULTILINE),
    # Title-case standalone (at least 2 words): "Related Work", "Future Work"
    re.compile(r"^([A-Z][a-z]+(?:\s+[A-Za-z]+){0,5})$", re.MULTILINE),
]

# ---------------------------------------------------------------------------
# Section type normalization mapping
# ---------------------------------------------------------------------------
SECTION_KEYWORD_MAP = {
    "abstract": ["abstract"],
    "introduction": ["introduction"],
    "background": ["background", "preliminaries", "preliminary"],
    "related_work": ["related work", "related literature", "literature review",
                     "prior work", "previous work"],
    "methodology": ["method", "methodology", "approach", "framework",
                     "proposed method", "model", "system design",
                     "experimental setup", "materials and methods"],
    "results": ["result", "results", "findings", "experimental results",
                "experiments"],
    "discussion": ["discussion"],
    "limitations": ["limitation", "limitations"],
    "future_work": ["future work", "future direction", "future directions",
                    "future research"],
    "conclusion": ["conclusion", "conclusions", "concluding remarks",
                   "summary", "summary and conclusion"],
}


def normalize_section_type(heading: str) -> str:
    """
    Map a raw section heading to one of the canonical section types.

    Args:
        heading: Raw heading text from the paper.

    Returns:
        One of the canonical section type strings from SECTION_TYPES.
    """
    if not heading:
        return "other"

    heading_lower = heading.lower().strip()
    # Remove numbering prefix for matching
    heading_lower = re.sub(r"^[0-9ivx]+\.?\s*", "", heading_lower)
    heading_lower = heading_lower.strip()

    for section_type, keywords in SECTION_KEYWORD_MAP.items():
        for keyword in keywords:
            if keyword in heading_lower:
                return section_type

    return "other"


def detect_sections(paper_doc: dict) -> list[dict]:
    """
    Detect section boundaries in a paper's full text using heading heuristics.

    Args:
        paper_doc: Parsed paper document dict with 'full_text' field.

    Returns:
        List of section dicts, each with:
        - heading: raw heading text
        - section_type: normalized type
        - start_idx: character start in full_text
        - end_idx: character end in full_text
        - text: section body text
    """
    full_text = paper_doc.get("full_text", "")
    if not full_text:
        logger.warning(f"Paper {paper_doc.get('paper_id', '?')} has no text")
        return [{"heading": None, "section_type": "other",
                 "start_idx": 0, "end_idx": 0, "text": ""}]

    # Find all heading matches with positions
    headings = []
    for pattern in HEADING_PATTERNS:
        for match in pattern.finditer(full_text):
            heading_text = match.group(1).strip() if match.lastindex else match.group(0).strip()
            # Skip very short or very long "headings" (likely false positives)
            if 3 <= len(heading_text) <= 80:
                headings.append({
                    "heading": heading_text,
                    "position": match.start(),
                    "section_type": normalize_section_type(heading_text),
                })

    # Deduplicate headings that overlap in position (keep first)
    headings.sort(key=lambda h: h["position"])
    deduped = []
    last_pos = -10
    for h in headings:
        if h["position"] - last_pos > 5:  # at least 5 chars apart
            deduped.append(h)
            last_pos = h["position"]
    headings = deduped

    logger.debug(
        f"Detected {len(headings)} section headings in "
        f"paper {paper_doc.get('paper_id', '?')}"
    )

    # Build sections from heading positions
    sections = []
    for i, h in enumerate(headings):
        start_idx = h["position"]
        end_idx = headings[i + 1]["position"] if i + 1 < len(headings) else len(full_text)
        section_text = full_text[start_idx:end_idx].strip()

        sections.append({
            "heading": h["heading"],
            "section_type": h["section_type"],
            "start_idx": start_idx,
            "end_idx": end_idx,
            "text": section_text,
        })

    # If no headings detected, treat entire text as one section
    if not sections:
        logger.info(
            f"No section headings found in paper "
            f"{paper_doc.get('paper_id', '?')}, treating as single section"
        )
        sections.append({
            "heading": None,
            "section_type": "other",
            "start_idx": 0,
            "end_idx": len(full_text),
            "text": full_text,
        })

    # Capture any text before the first detected heading
    if sections and sections[0]["start_idx"] > 0:
        preamble_text = full_text[: sections[0]["start_idx"]].strip()
        if preamble_text:
            sections.insert(0, {
                "heading": None,
                "section_type": "other",
                "start_idx": 0,
                "end_idx": sections[0]["start_idx"],
                "text": preamble_text,
            })

    return sections


def _estimate_page_number(char_offset: int, pages: list[dict]) -> int | None:
    """Estimate which page a character offset falls on."""
    running_offset = 0
    for page in pages:
        page_len = len(page["text"]) + 2  # +2 for the \n\n join separator
        if running_offset + page_len > char_offset:
            return page["page_number"]
        running_offset += page_len
    return pages[-1]["page_number"] if pages else None


def chunk_paper(
    paper_doc: dict,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[dict]:
    """
    Split a parsed paper into overlapping chunks with full metadata.

    Args:
        paper_doc: Parsed paper document from pdf_parser.
        chunk_size: Target chunk size in characters.
        chunk_overlap: Overlap between consecutive chunks in characters.

    Returns:
        List of chunk dicts matching the canonical chunk schema.
    """
    paper_id = paper_doc["paper_id"]
    logger.info(f"Chunking paper: {paper_id}")

    sections = detect_sections(paper_doc)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []
    chunk_counter = 0

    for section in sections:
        section_text = section["text"]
        if not section_text.strip():
            continue

        # If section is short enough, keep it as one chunk
        if len(section_text) <= chunk_size:
            text_splits = [section_text]
        else:
            text_splits = splitter.split_text(section_text)

        for split_text in text_splits:
            # Find approximate position in full text for page mapping
            char_start = paper_doc["full_text"].find(split_text[:50])
            char_end = char_start + len(split_text) if char_start >= 0 else None
            page_num = _estimate_page_number(
                section["start_idx"], paper_doc.get("pages", [])
            )

            chunk = {
                "chunk_id": f"{paper_id}_chunk_{chunk_counter:03d}",
                "paper_id": paper_id,
                "paper_title": paper_doc.get("paper_title"),
                "source_file": paper_doc.get("source_file"),
                "section_heading": section["heading"],
                "section_type": section["section_type"],
                "page_number": page_num,
                "chunk_text": split_text.strip(),
                "char_start": char_start if char_start >= 0 else None,
                "char_end": char_end,
                "token_estimate": len(split_text) // 4,
                "authors": paper_doc.get("authors"),
                "year": paper_doc.get("year"),
                "doi": None,
            }
            chunks.append(chunk)
            chunk_counter += 1

    logger.info(
        f"Paper {paper_id}: {len(sections)} sections -> {len(chunks)} chunks "
        f"(avg {sum(c['token_estimate'] for c in chunks) // max(len(chunks), 1)} tokens/chunk)"
    )

    return chunks


def chunk_corpus(papers: list[dict], **kwargs) -> list[dict]:
    """
    Chunk all papers in a corpus.

    Args:
        papers: List of parsed paper dicts from parse_folder().
        **kwargs: Passed to chunk_paper (chunk_size, chunk_overlap).

    Returns:
        Flat list of all chunk dicts across all papers.
    """
    all_chunks = []
    for paper in papers:
        try:
            paper_chunks = chunk_paper(paper, **kwargs)
            all_chunks.extend(paper_chunks)
        except Exception as e:
            logger.error(
                f"Failed to chunk paper {paper.get('paper_id', '?')}: {e}",
                exc_info=True,
            )

    logger.info(f"Total corpus: {len(all_chunks)} chunks from {len(papers)} papers")
    return all_chunks
