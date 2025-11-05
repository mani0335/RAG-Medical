# Medical-Chatbot — Project README

Last updated: 2025-11-04

## Project origin — where we started
This project started as a small Retrieval-Augmented Generation (RAG) demo to make a Streamlit-based chatbot that answers medical questions from a set of PDF documents. Goals:
- Ingest PDF reference material (data/).
- Build a FAISS vector store of chunk embeddings.
- Use an LLM to synthesize concise, sourced answers from retrieved passages.
- Provide a simple Streamlit UI that returns single, factual answers and citations.

## What we did (progress, edits & engineering decisions)
- Built an indexing pipeline (`create_memory_for_llm.py`) that:
  - Loads PDFs from `data/`
  - Splits into chunks
  - Creates embeddings (HuggingFace sentence-transformers)
  - Saves a FAISS DB at `vectorstore/db_faiss`
- Created the Streamlit app (`medibot.py`) that:
  - Loads the FAISS vectorstore (cached)
  - Uses RetrievalQA (LangChain) to retrieve and synthesize answers (RAG)
  - Formats and displays readable source citations instead of raw Document objects
  - Loads `.env` (dotenv) so environment keys are read consistently
  - Adds exponential-backoff retries for Groq API calls and a fallback to a Hugging Face endpoint LLM
  - Implements robust prompt and chain-type wiring (supports `stuff`, `map_reduce`, `refine` in a safe way)
  - Provides a concise answer prompt that asks for a single factual answer and one best source
- Fixed multiple runtime issues encountered:
  - Keras 3 / Transformers compatibility — recommended installing `tf-keras`.
  - LangChain chain prompt validation errors — implemented `build_chain_type_kwargs` and aligned `context_str` for refine chains.
  - Source citation formatting — robust routine to extract filename/page/snippet from Document metadata.
  - UI debug prints removed or moved to a sidebar toggle to avoid leaking raw metadata.

## What the application uses (tech stack) and why
- Python >= 3.9 — runtime.
- Streamlit — web UI for chat interface.
- LangChain / langchain_core / langchain_community — orchestration, RetrievalQA, document chains.
- FAISS (faiss-cpu) — fast similarity search vector store for embeddings.
- sentence-transformers (via HuggingFaceEmbeddings) — lightweight embeddings model (all-MiniLM-L6-v2) for index.
- Groq (langchain_groq.ChatGroq) — primary LLM backend (requires GROQ_API_KEY). Fast hosted LLM option.
- Hugging Face Endpoint (`langchain_huggingface.HuggingFaceEndpoint`) — fallback LLM backend (requires HF_TOKEN / HF_REPO_ID).
- python-dotenv — load `.env` automatically.
- Other utilities: PyPDF loaders, text splitters, etc.

Purpose summary:
- FAISS + embeddings = retrieval (find relevant chunks).
- RetrievalQA + LLM = synthesis (use retrieved context to answer).
- Streamlit = interactive interface and UX.

## Requirements
- See `requirements.txt` in repo root. Notable items:
  - faiss-cpu, sentence-transformers, streamlit, langchain (and related connectors)
  - python-dotenv
- Windows specific notes:
  - Some binary wheels (faiss, torch) may be easier to install via conda/mamba.
  - If you hit Keras/Transformers issues install `tf-keras` (or pin transformers).

## Quick start (run locally)
1. Open PowerShell and change to project folder:
   Set-Location 'C:\Users\manish\Desktop\medical-chatbot-main'

2. Create & activate virtual environment (PowerShell):
   . .\.venv\Scripts\Activate.ps1

3. Upgrade pip and install requirements:
   python -m pip install --upgrade pip
   pip install -r requirements.txt

4. Add secrets to `.env` (in project root):
   - GROQ_API_KEY=...
   - HF_TOKEN=... (optional)
   - HF_REPO_ID=... (optional)
   (A `.env` example file may be present; do NOT commit secrets.)

5. Index PDFs (reads all PDFs in `data/` and builds `vectorstore/db_faiss`):
   python create_memory_for_llm.py

6. Start the app:
   streamlit run medibot.py
   - Open http://localhost:8501

## How to add PDFs
- Copy your PDF(s) into the `data/` directory.
- Re-run the indexer:
  python create_memory_for_llm.py
- Restart Streamlit and ask a question; the new PDFs will be included.

Optional: use an incremental add script to append single PDFs to the existing FAISS DB (can be added to the repo if desired).

## Features
- RAG pipeline: retrieval from FAISS + LLM synthesis.
- Single concise answer preference (prompt instructs single-paragraph answer + 1 source).
- Source citation formatting with filename + page + snippet.
- Retry + fallback for Groq LLM (automatic exponential backoff).
- `.env` support and safe .gitignore entries (.env, .venv, vectorstore).
- Debug toggle (sidebar) to view raw document metadata when needed.

## Troubleshooting & common errors
- Keras / Transformers:
  - Error: "Keras 3 not yet supported" — install `tf-keras` or pin `transformers` to a compatible version.
- Groq over-capacity (503):
  - The app retries automatically; check https://groqstatus.com. Ensure fallback HF_TOKEN is set if you want fallback to work.
- LangChain chain validation errors (extra_forbidden / missing input variables):
  - We added `build_chain_type_kwargs` and use `context_str` for refine prompts. If you edit prompt templates, ensure input_variables names match the chain's expected names (map_prompt uses `text`, combine uses `summaries`, refine uses `context_str`, `existing_answer`).
- FAISS issues on Windows:
  - Use conda to install faiss-cpu if pip wheels fail.

## File map (important files)
- medibot.py — Streamlit app, LLM/retrieval orchestration, UI.
- create_memory_for_llm.py — PDF loading, chunking, embedding, FAISS saving.
- connect_memory_with_llm.py — console QA / helper scripts.
- requirements.txt — Python dependencies.
- data/ — place PDFs here.
- vectorstore/ — generated FAISS DB (ignore from git).
- .env — local secrets (ignored by git).

## Workflow diagram
(mermaid diagram; many viewers render Mermaid automatically)

```mermaid
flowchart TD
  A[Place PDFs in data/] --> B[Index PDFs]
  B --> C[Chunking & Embeddings]
  C --> D[FAISS Vectorstore saved (vectorstore/db_faiss)]
  E[User asks question via Streamlit UI] --> F[Retrieve top-k chunks from FAISS]
  F --> G[RetrievalQA chain (LangChain)]
  G --> H[LLM: Primary (Groq) — retry/backoff]
  H -- if failed --> I[Fallback: HuggingFace Endpoint]
  G --> J[Synthesized concise answer + source]
  J --> K[Streamlit displays answer + formatted source(s)]
```

## What we deliver / output
- A running Streamlit RAG chatbot that:
  - Answers medical questions using your PDF corpus.
  - Shows the single best source (filename + page + snippet).
  - Uses retries and fallback for robustness.

## Next recommended improvements
- Add a Streamlit PDF uploader that triggers incremental indexing.
- Add UI controls to configure `k`, chain type (`stuff` / `refine` / `map_rerank`) and model choice.
- Add CI checks for dependency compatibility and a requirements lock file.
- Improve duplicate-detection on ingestion and store stronger metadata (doc_id).

---

If you want, I can:
- Render the Mermaid diagram to an image and add it to the repo.
- Add a sample `.env.example` (without secrets).
- Implement PDF upload + incremental indexing in the Streamlit app.

Open the file in your editor or refresh the repo to preview this README.



