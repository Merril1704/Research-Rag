"""
Centralized configuration for ResearchRAG.

All tunable constants, paths, and defaults live here.
Modules should import from this file rather than hard-coding values.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Load environment variables from .env (if present)
# ---------------------------------------------------------------------------
load_dotenv()

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PDF_DIR = DATA_DIR / "pdfs"
INDEX_DIR = DATA_DIR / "index"
EVAL_DIR = DATA_DIR / "eval_results"
LOG_DIR = PROJECT_ROOT / "logs"

# Ensure directories exist
for _dir in [PDF_DIR, INDEX_DIR, EVAL_DIR, LOG_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Chunking defaults
# ---------------------------------------------------------------------------
CHUNK_SIZE = 1500          # characters (~375-500 tokens)
CHUNK_OVERLAP = 200        # characters of overlap between chunks

# ---------------------------------------------------------------------------
# Embedding defaults
# ---------------------------------------------------------------------------
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_BATCH_SIZE = 64

# ---------------------------------------------------------------------------
# Retrieval defaults
# ---------------------------------------------------------------------------
TOP_K_RELATED_WORK = 10
TOP_K_GAP_ANALYSIS = 15    # pre-filter pool size (filtered down after)
GAP_SECTION_TYPES = {"limitations", "future_work", "discussion", "conclusion"}
MIN_GAP_RESULTS = 3        # minimum filtered results before fallback

# ---------------------------------------------------------------------------
# Generation / LLM defaults
# ---------------------------------------------------------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_FALLBACK_MODEL = "llama3-8b-8192"

# Ollama local fallback
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")

# Generation parameters
LLM_TEMPERATURE = 0.3
LLM_MAX_TOKENS = 4096

# ---------------------------------------------------------------------------
# Section type normalization
# ---------------------------------------------------------------------------
SECTION_TYPES = [
    "abstract",
    "introduction",
    "background",
    "related_work",
    "methodology",
    "results",
    "discussion",
    "limitations",
    "future_work",
    "conclusion",
    "other",
]

# ---------------------------------------------------------------------------
# Index file names
# ---------------------------------------------------------------------------
FAISS_INDEX_FILE = "faiss_index.bin"
METADATA_FILE = "chunk_metadata.json"
