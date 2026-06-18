"""
ResearchRAG — Streamlit Application

Full UI for PDF upload, corpus building, querying, and result display.
This is the only module that imports Streamlit.

Usage:
    streamlit run researchrag/app.py
"""

import streamlit as st
from pathlib import Path

from researchrag.config import PDF_DIR, INDEX_DIR
from researchrag.ingestion.pdf_parser import parse_folder
from researchrag.indexing.chunker import chunk_corpus
from researchrag.indexing.embedder import Embedder
from researchrag.indexing.vector_store import VectorStore
from researchrag.retrieval.retriever import Retriever
from researchrag.generation.llm_client import LLMClient
from researchrag.generation.related_work_drafter import draft_related_work
from researchrag.generation.gap_finder import find_gaps
from researchrag.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="ResearchRAG",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------
if "embedder" not in st.session_state:
    st.session_state.embedder = None
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "llm_client" not in st.session_state:
    st.session_state.llm_client = None
if "corpus_built" not in st.session_state:
    st.session_state.corpus_built = False

# ---------------------------------------------------------------------------
# Sidebar — PDF Upload & Corpus Management
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("📚 ResearchRAG")
    st.markdown("---")

    # --- PDF Upload ---
    st.header("📄 Upload Papers")
    uploaded_files = st.file_uploader(
        "Upload research paper PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload 10-20 research papers in PDF format.",
    )

    if uploaded_files:
        # Save uploaded files to data/pdfs/
        PDF_DIR.mkdir(parents=True, exist_ok=True)
        saved_count = 0
        for uploaded_file in uploaded_files:
            save_path = PDF_DIR / uploaded_file.name
            if not save_path.exists():
                with open(save_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                saved_count += 1
        if saved_count > 0:
            st.success(f"Saved {saved_count} new PDF(s) to corpus.")

    # Show current PDFs
    existing_pdfs = list(PDF_DIR.glob("*.pdf")) if PDF_DIR.exists() else []
    if existing_pdfs:
        st.caption(f"📁 {len(existing_pdfs)} PDF(s) in corpus folder")
        with st.expander("View files"):
            for pdf in existing_pdfs:
                st.text(f"  • {pdf.name}")
    else:
        st.info("No PDFs uploaded yet.")

    st.markdown("---")

    # --- Build Corpus ---
    st.header("🔨 Build Corpus")
    col1, col2 = st.columns(2)

    with col1:
        build_btn = st.button(
            "Build Index",
            disabled=len(existing_pdfs) == 0,
            use_container_width=True,
        )

    with col2:
        load_btn = st.button(
            "Load Index",
            disabled=not (INDEX_DIR / "faiss_index.bin").exists(),
            use_container_width=True,
        )

    # --- Build flow ---
    if build_btn and existing_pdfs:
        with st.status("Building corpus...", expanded=True) as status:
            # Step 1: Parse PDFs
            st.write("📄 Parsing PDFs...")
            logger.info(f"Building corpus from {len(existing_pdfs)} PDFs")
            papers = parse_folder(str(PDF_DIR))
            st.write(f"  ✓ Parsed {len(papers)} papers")

            # Step 2: Chunk
            st.write("✂️ Chunking papers...")
            all_chunks = chunk_corpus(papers)
            st.write(f"  ✓ Created {len(all_chunks)} chunks")

            # Step 3: Embed
            st.write("🧠 Generating embeddings...")
            if st.session_state.embedder is None:
                st.session_state.embedder = Embedder()
            embedder = st.session_state.embedder
            chunk_texts = [c["chunk_text"] for c in all_chunks]
            embeddings = embedder.embed_texts(chunk_texts)
            st.write(f"  ✓ Embedded {len(chunk_texts)} chunks")

            # Step 4: Build index
            st.write("📊 Building FAISS index...")
            # Store metadata without embeddings (chunk_text is in metadata)
            metadata = [
                {k: v for k, v in c.items() if k != "embedding"}
                for c in all_chunks
            ]
            store = VectorStore()
            store.build(embeddings, metadata)
            store.save()
            st.session_state.vector_store = store
            st.write(f"  ✓ Index built with {store.index.ntotal} vectors")

            # Step 5: Initialize retriever
            st.session_state.retriever = Retriever(embedder, store)
            st.session_state.corpus_built = True

            # Step 6: Initialize LLM client
            if st.session_state.llm_client is None:
                st.session_state.llm_client = LLMClient()

            status.update(label="Corpus built successfully! ✅", state="complete")
            logger.info("Corpus build complete")

    # --- Load flow ---
    if load_btn:
        with st.status("Loading index...", expanded=True) as status:
            st.write("🧠 Loading embedding model...")
            if st.session_state.embedder is None:
                st.session_state.embedder = Embedder()

            st.write("📊 Loading FAISS index...")
            store = VectorStore()
            store.load()
            st.session_state.vector_store = store

            st.session_state.retriever = Retriever(
                st.session_state.embedder, store
            )
            st.session_state.corpus_built = True

            if st.session_state.llm_client is None:
                st.session_state.llm_client = LLMClient()

            status.update(
                label=f"Index loaded ({store.index.ntotal} vectors) ✅",
                state="complete",
            )
            logger.info(f"Index loaded: {store.index.ntotal} vectors")

    # --- Status ---
    st.markdown("---")
    if st.session_state.corpus_built:
        st.success("✅ Corpus ready")
    else:
        st.warning("⚠️ Build or load a corpus to start querying")

    # --- Generation Settings ---
    st.markdown("---")
    st.header("⚙️ Generation Settings")
    
    selected_model = st.selectbox(
        "LLM Model (via Groq)",
        options=[
            "llama-3.3-70b-versatile",
            "llama3-70b-8192",
            "gemma2-9b-it"
        ],
        index=0,
        help="Select the generation model. Llama 3.3 70B is currently the best available on Groq."
    )
    
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.2,
        step=0.1,
        help="Lower values are more factual, higher values are more creative."
    )
    
    max_tokens = st.slider(
        "Max Tokens",
        min_value=500,
        max_value=4000,
        value=2000,
        step=100,
        help="Maximum length of the generated response."
    )

# ---------------------------------------------------------------------------
# Main area — Query & Results
# ---------------------------------------------------------------------------
st.title("📚 ResearchRAG")
st.markdown(
    "A modular RAG system for **Research Gap Finding** and "
    "**Related Work Drafting** over your paper corpus."
)

# --- Query input ---
query = st.text_area(
    "Enter your research query",
    placeholder="e.g., What are the main limitations of transformer-based models for low-resource NLP?",
    height=100,
)

with st.expander("📝 Custom Instructions (Optional)"):
    custom_instructions = st.text_area(
        "Add specific instructions for the LLM (tone, structure, focus areas):",
        placeholder="e.g., Keep the response concise. Focus primarily on data privacy limitations. Use bullet points for the main gaps.",
        height=80,
    )

col_mode, col_topk, col_run = st.columns([2, 1, 1])

with col_mode:
    mode = st.selectbox(
        "Mode",
        options=["Related Work", "Gap Analysis"],
        help="Related Work: thematic literature synthesis. Gap Analysis: identify limitations and open problems.",
    )

with col_topk:
    top_k = st.number_input("Top-K", min_value=3, max_value=30, value=10)

with col_run:
    st.write("")  # spacing
    run_btn = st.button(
        "🚀 Generate",
        disabled=not st.session_state.corpus_built or not query,
        use_container_width=True,
    )

# --- Run pipeline ---
if run_btn and query and st.session_state.corpus_built:
    mode_key = "related_work" if mode == "Related Work" else "gap_analysis"

    with st.spinner("Retrieving evidence and generating..."):
        logger.info(f"Query: '{query}', mode={mode_key}, top_k={top_k}")

        # Retrieve
        retriever = st.session_state.retriever
        retrieved = retriever.retrieve(query, mode=mode_key, top_k=top_k)

        # Apply UI hyperparameters to the LLM client
        llm = st.session_state.llm_client
        llm.model = selected_model
        llm.temperature = temperature
        llm.max_tokens = max_tokens

        # Generate
        if mode_key == "related_work":
            result = draft_related_work(query, retrieved, llm, custom_instructions)
        else:
            result = find_gaps(query, retrieved, llm, custom_instructions)

    # --- Display results ---
    st.markdown("---")
    st.subheader(f"{'📝' if mode_key == 'related_work' else '🔍'} {mode} Output")
    st.markdown(result["answer_text"])

    # --- Source evidence ---
    st.markdown("---")
    st.subheader("📎 Source Evidence")
    st.caption(f"{len(retrieved)} chunks retrieved from {len(result['citations'])} papers")

    for i, chunk in enumerate(retrieved):
        with st.expander(
            f"[{i+1}] {chunk.get('paper_title', 'Unknown')} — "
            f"{chunk.get('section_type', 'N/A')} "
            f"(score: {chunk.get('score', 0):.4f})"
        ):
            st.markdown(f"**Section:** {chunk.get('section_heading', 'N/A')}")
            st.markdown(f"**Page:** {chunk.get('page_number', 'N/A')}")
            st.markdown(f"**Chunk ID:** `{chunk.get('chunk_id', 'N/A')}`")
            st.text(chunk.get("chunk_text", ""))

    # --- Citations ---
    if result["citations"]:
        st.markdown("---")
        st.subheader("📖 Papers Referenced")
        for cit in result["citations"]:
            authors = ", ".join(cit["authors"]) if cit.get("authors") else "Unknown"
            year = cit.get("year", "N/A")
            st.markdown(f"- **{cit['paper_title']}** — {authors} ({year})")

elif not st.session_state.corpus_built:
    st.info("👈 Upload PDFs and build your corpus using the sidebar to get started.")
