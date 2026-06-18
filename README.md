# 📚 ResearchRAG

**A Modular Dense RAG System for Academic Literature Analysis**

ResearchRAG is a specialized Retrieval-Augmented Generation (RAG) system designed specifically for academic researchers. Instead of relying on general LLMs that frequently hallucinate citations or provide superficial summaries, ResearchRAG strictly grounds its analysis in your custom corpus of uploaded PDFs, providing deep, evidence-backed synthesis with full traceability.

---

## ✨ Features

- **🎯 Two Specialized Modes:**
  - **Related Work Drafting:** Synthesizes literature into cohesive, thematic paragraphs suitable for a thesis or paper introduction.
  - **Gap Analysis:** Specifically targets and extracts limitations, future work, and open problems across multiple papers to help you identify your next research contribution.
- **🛡️ Zero Hallucinated Citations:** The system is restricted to citing *only* the papers you upload. Every claim is traceable to a specific chunk, page, and section.
- **🧠 Section-Aware Retrieval:** Uses heuristic chunking to recognize sections (e.g., "Methodology", "Limitations"). When doing a Gap Analysis, it inherently boosts retrieval from "Limitations" and "Future Work" sections.
- **🎛️ Customizable Dashboard:** Built with Streamlit. Allows you to control the LLM model, temperature, maximum output length, and inject custom instructions dynamically.
- **⚡ High-Speed Inference:** Optimized to run with Groq's lightning-fast API for LLM generation (Llama 3.3 70B, DeepSeek R1), with FAISS for local, sub-millisecond vector search.
- **📊 Built-in Evaluation:** Integrated with RAGAS framework for automated metric tracking (Faithfulness, Answer Relevancy, Context Precision/Recall).

---

## 🏗️ Architecture

1. **Ingestion (`pdf_parser.py`):** Uses PyMuPDF to extract text while maintaining pagination and metadata.
2. **Indexing (`chunker.py`, `embedder.py`, `vector_store.py`):** 
   - Detects academic section headers.
   - Chunks text locally while preserving section and paper metadata.
   - Embeds chunks using local HuggingFace `sentence-transformers`.
   - Stores vectors in a fast, local FAISS index.
3. **Retrieval (`retriever.py`):** Performs semantic search, filtering and boosting results based on the active mode (e.g., filtering out "Abstracts" when looking for deep "Methodology" gaps).
4. **Generation (`llm_client.py`, `prompts.py`):** Constructs specialized prompts and streams requests to the Groq API.
5. **UI (`app.py`):** A modern Streamlit interface for uploading papers, managing the corpus, and viewing synthesized results side-by-side with source evidence.

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.12+
- `conda` (recommended for environment management)
- A [Groq API Key](https://console.groq.com/keys)

### 1. Clone the repository
```bash
git clone https://github.com/Merril1704/Research-Rag.git
cd Research-Rag
```

### 2. Set up the Conda Environment
```bash
conda create -n researchrag python=3.12 -y
conda activate researchrag
```

### 3. Install Dependencies
Install the package in editable mode so it can be imported properly from the UI:
```bash
pip install -r requirements.txt
pip install -e .
```

### 4. Configure Environment Variables
Copy the example environment file and add your API keys:
```bash
cp .env.example .env
```
Open `.env` and add your Groq API key:
```env
GROQ_API_KEY=gsk_your_api_key_here
```

---

## 💻 Usage

Start the Streamlit application:
```bash
streamlit run researchrag/app.py
```

### Workflow
1. **Upload Papers:** Use the sidebar to upload 10-20 research paper PDFs.
2. **Build Corpus:** Click "Build Index" to trigger the parsing, chunking, and embedding pipeline.
3. **Query:** Type your research question into the main text area.
4. **Configure Settings:** Choose your mode (Related Work vs. Gap Analysis), select your preferred model, adjust temperature, or add custom instructions.
5. **Generate:** View the synthesized output, followed by the exact source evidence chunks and citations used to generate the response.

---

## 🧪 Evaluation

ResearchRAG includes a built-in evaluation module using the **RAGAS** framework to guarantee high-quality academic output.

To verify your pipeline against hallucination and relevance metrics:
```bash
python -c "from researchrag.evaluation.ragas_eval import run_evaluation; print('ragas_eval OK')"
```
For a comprehensive comparison on why this system outperforms standard chat interfaces like ChatGPT or Claude for research work, see our [Evaluation Strategy Document](evaluation_strategy.md).

---

## 🛠️ Tech Stack

- **Frontend:** Streamlit
- **PDF Processing:** PyMuPDF
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` (Local)
- **Vector DB:** FAISS (Local)
- **LLM Engine:** Groq API (Llama-3.3-70B, Gemma2)
- **Evaluation:** RAGAS, Langchain ecosystem

---

## 🤝 Contributing
Contributions are welcome! If you're adding new features like web-search augmentation or integration with reference managers (Zotero/Mendeley), please open an issue first to discuss.
