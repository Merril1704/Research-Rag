"""
PDF Parser — Extracts text and metadata from research paper PDFs.

Uses PyMuPDF (fitz) for fast, lightweight PDF text extraction.
Produces one structured paper document dict per PDF file.

Usage:
    from researchrag.ingestion.pdf_parser import parse_pdf, parse_folder

    paper = parse_pdf("path/to/paper.pdf")
    papers = parse_folder("path/to/pdf_folder/")
"""

import hashlib
import re
from pathlib import Path

import fitz  # PyMuPDF

from researchrag.logger import get_logger

logger = get_logger(__name__)


def _generate_paper_id(filename: str) -> str:
    """Generate a deterministic paper ID from the filename."""
    name_hash = hashlib.md5(filename.encode()).hexdigest()[:8]
    clean_name = re.sub(r"[^a-zA-Z0-9]", "_", Path(filename).stem)[:30]
    return f"{clean_name}_{name_hash}"


def _clean_text(text: str) -> str:
    """Basic text cleanup: collapse whitespace, remove page-break artifacts."""
    # Remove form-feed characters (page breaks)
    text = text.replace("\f", " ")
    # Collapse multiple spaces/tabs into single space
    text = re.sub(r"[ \t]+", " ", text)
    # Collapse 3+ newlines into 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _extract_title(first_page_text: str) -> str | None:
    """
    Heuristic title extraction from the first page.
    Takes the first non-empty line that looks like a title.
    """
    lines = first_page_text.strip().split("\n")
    for line in lines:
        line = line.strip()
        # Skip very short lines, page numbers, dates
        if len(line) < 5:
            continue
        if re.match(r"^\d+$", line):
            continue
        if re.match(r"^(page|vol|issue|doi|http|www)", line, re.IGNORECASE):
            continue
        # First substantial line is likely the title
        return line
    return None


def _extract_authors(first_page_text: str) -> list[str] | None:
    """
    Heuristic author extraction. Looks for lines after the title
    that contain comma-separated names or 'and' patterns.
    """
    lines = first_page_text.strip().split("\n")
    title_found = False
    for line in lines:
        line = line.strip()
        if not title_found:
            if len(line) >= 5:
                title_found = True
            continue
        # Look for author-like patterns after the title
        if re.search(r"[A-Z][a-z]+\s+[A-Z][a-z]+", line) and len(line) < 300:
            # Split on commas and 'and'
            parts = re.split(r",\s*|\s+and\s+", line)
            authors = [p.strip() for p in parts if len(p.strip()) > 2]
            if 1 <= len(authors) <= 15:
                return authors
    return None


def parse_pdf(pdf_path: str) -> dict:
    """
    Parse a single PDF file into a structured paper document.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        Dict with keys: paper_id, source_file, paper_title, authors,
        year, pages, full_text.

    Raises:
        FileNotFoundError: If the PDF file does not exist.
        RuntimeError: If PyMuPDF fails to open or parse the file.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    logger.info(f"Parsing PDF: {pdf_path.name}")

    try:
        doc = fitz.open(str(pdf_path))
    except Exception as e:
        logger.error(f"Failed to open PDF {pdf_path.name}: {e}", exc_info=True)
        raise RuntimeError(f"PyMuPDF failed to open {pdf_path.name}") from e

    pages = []
    full_text_parts = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        raw_text = page.get_text("text")
        cleaned = _clean_text(raw_text)
        pages.append({
            "page_number": page_num + 1,
            "text": cleaned,
        })
        full_text_parts.append(cleaned)

    doc.close()
    full_text = "\n\n".join(full_text_parts)

    # Extract metadata from first page
    first_page_text = pages[0]["text"] if pages else ""
    title = _extract_title(first_page_text)
    authors = _extract_authors(first_page_text)

    # Try to extract year from the text (look for 4-digit year)
    year = None
    year_match = re.search(r"\b(19|20)\d{2}\b", first_page_text)
    if year_match:
        year = year_match.group(0)

    paper_id = _generate_paper_id(pdf_path.name)

    paper_doc = {
        "paper_id": paper_id,
        "source_file": pdf_path.name,
        "paper_title": title,
        "authors": authors,
        "year": year,
        "pages": pages,
        "full_text": full_text,
    }

    logger.info(
        f"Parsed '{pdf_path.name}': {len(pages)} pages, "
        f"title='{title or 'N/A'}', year={year or 'N/A'}"
    )
    logger.debug(f"Paper ID: {paper_id}, text length: {len(full_text)} chars")

    return paper_doc


def parse_folder(folder_path: str) -> list[dict]:
    """
    Parse all PDF files in a folder.

    Args:
        folder_path: Path to directory containing PDF files.

    Returns:
        List of paper document dicts.

    Raises:
        FileNotFoundError: If the folder does not exist.
    """
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    pdf_files = sorted(folder.glob("*.pdf"))
    if not pdf_files:
        logger.warning(f"No PDF files found in {folder}")
        return []

    logger.info(f"Found {len(pdf_files)} PDF(s) in {folder}")

    papers = []
    for pdf_file in pdf_files:
        try:
            paper = parse_pdf(str(pdf_file))
            papers.append(paper)
        except Exception as e:
            logger.error(f"Skipping {pdf_file.name}: {e}", exc_info=True)

    logger.info(f"Successfully parsed {len(papers)}/{len(pdf_files)} PDF(s)")
    return papers
