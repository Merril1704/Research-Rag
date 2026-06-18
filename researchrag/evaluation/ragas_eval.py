"""
RAGAS Evaluation — Offline evaluation harness for ResearchRAG.

Evaluates retrieval and generation quality on sampled queries
using RAGAS metrics: Faithfulness, Answer Relevance,
Context Precision, and Context Recall.

Can be run standalone:
    python -m researchrag.evaluation.ragas_eval

Usage:
    from researchrag.evaluation.ragas_eval import run_evaluation

    results = run_evaluation(eval_rows)
"""

import json
from datetime import datetime
from pathlib import Path

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from langchain_groq import ChatGroq
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_community.embeddings import HuggingFaceEmbeddings

from researchrag.config import EVAL_DIR, GROQ_API_KEY, GROQ_MODEL, EMBEDDING_MODEL
from researchrag.logger import get_logger

logger = get_logger(__name__)

# Default metrics to evaluate
DEFAULT_METRICS = [
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
]


def _get_ragas_llm():
    """Get a Groq-backed LLM wrapper for RAGAS evaluation."""
    chat_model = ChatGroq(
        api_key=GROQ_API_KEY,
        model=GROQ_MODEL,
        temperature=0,
    )
    logger.info(f"RAGAS LLM configured: Groq ({GROQ_MODEL})")
    return LangchainLLMWrapper(chat_model)


def _get_ragas_embeddings():
    """Get local SentenceTransformer embeddings for RAGAS evaluation."""
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    logger.info(f"RAGAS embeddings configured: {EMBEDDING_MODEL}")
    return LangchainEmbeddingsWrapper(embeddings)


def build_eval_dataset(eval_rows: list[dict]) -> Dataset:
    """
    Build a RAGAS-compatible HuggingFace Dataset from evaluation rows.

    Args:
        eval_rows: List of dicts, each containing:
            - user_input (str): The query
            - retrieved_contexts (list[str]): Retrieved chunk texts
            - response (str): Generated answer
            - reference (str, optional): Gold reference answer

    Returns:
        HuggingFace Dataset for RAGAS evaluation.
    """
    logger.info(f"Building evaluation dataset from {len(eval_rows)} rows")

    # Validate and normalize rows
    dataset_dict = {
        "user_input": [],
        "retrieved_contexts": [],
        "response": [],
        "reference": [],
    }

    for i, row in enumerate(eval_rows):
        if "user_input" not in row or "response" not in row:
            logger.warning(f"Skipping eval row {i}: missing required fields")
            continue

        dataset_dict["user_input"].append(row["user_input"])
        dataset_dict["retrieved_contexts"].append(
            row.get("retrieved_contexts", [])
        )
        dataset_dict["response"].append(row["response"])
        dataset_dict["reference"].append(row.get("reference", ""))

    dataset = Dataset.from_dict(dataset_dict)
    logger.info(f"Evaluation dataset built: {len(dataset)} rows")
    return dataset


def run_evaluation(
    eval_rows: list[dict],
    metrics: list | None = None,
) -> dict:
    """
    Run RAGAS evaluation on a set of evaluation rows.

    Uses Groq as the judge LLM and local SentenceTransformer embeddings,
    so no OpenAI API key is needed.

    Args:
        eval_rows: List of evaluation row dicts.
        metrics: RAGAS metrics to compute (defaults to all four).

    Returns:
        Dict with metric scores and per-row details.
    """
    metrics = metrics or DEFAULT_METRICS

    logger.info(
        f"Running RAGAS evaluation: {len(eval_rows)} rows, "
        f"{len(metrics)} metrics"
    )

    dataset = build_eval_dataset(eval_rows)

    if len(dataset) == 0:
        logger.error("No valid evaluation rows. Aborting.")
        return {"error": "No valid evaluation rows"}

    # Use Groq LLM + local embeddings instead of OpenAI defaults
    ragas_llm = _get_ragas_llm()
    ragas_embeddings = _get_ragas_embeddings()

    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=ragas_llm,
        embeddings=ragas_embeddings,
    )

    logger.info(f"Evaluation complete. Scores: {result}")
    return dict(result)


def save_results(
    results: dict,
    eval_rows: list[dict] | None = None,
    output_dir: str | None = None,
) -> Path:
    """
    Save evaluation results and metadata to disk.

    Args:
        results: Metric results dict from run_evaluation.
        eval_rows: Original evaluation rows (saved for inspection).
        output_dir: Output directory (defaults to data/eval_results/).

    Returns:
        Path to the saved results file.
    """
    out_dir = Path(output_dir or EVAL_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = out_dir / f"eval_{timestamp}.json"

    output = {
        "timestamp": timestamp,
        "metrics": results,
        "num_rows": len(eval_rows) if eval_rows else 0,
        "rows": eval_rows,
    }

    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)

    logger.info(f"Evaluation results saved to {results_file}")
    return results_file


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("RAGAS Evaluation Harness")
    print("=" * 50)
    print(
        "To run evaluation, prepare eval rows and call:\n"
        "  from researchrag.evaluation.ragas_eval import run_evaluation\n"
        "  results = run_evaluation(eval_rows)\n"
        "  save_results(results, eval_rows)\n"
    )
    print(
        "Each eval row should have: user_input, retrieved_contexts, "
        "response, and optionally reference."
    )
