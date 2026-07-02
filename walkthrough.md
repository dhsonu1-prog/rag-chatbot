# Local Offline RAG Chatbot Walkthrough

This document describes the structure of the RAG pipeline built for your 115 civil service PDF documents, how it operates completely offline, and how you can run and query it on your local system.

---

## What We Built

We implemented a modular Python RAG pipeline in the [rag-chatbot](file:///home/administrator/Downloads/rag-chatbot) directory:
1.  **[requirements.txt](file:///home/administrator/Downloads/rag-chatbot/requirements.txt)**: Defines the offline package dependencies.
2.  **[ingest.py](file:///home/administrator/Downloads/rag-chatbot/ingest.py)**: Processes all PDFs, extracts text pages, attaches metadata (parent folders, file names), and saves embeddings using a local Hugging Face model (`all-MiniLM-L6-v2`) to a local vector store. It also automatically invalidates the stale BM25 cache file when re-running ingestion.
3.  **[query_engine.py](file:///home/administrator/Downloads/rag-chatbot/query_engine.py)**: Connects the local Chroma DB and the local `BM25Retriever` into a unified `EnsembleRetriever` (hybrid search). Utilizes the modern `ConversationalRetrievalChain` from `langchain_classic` to enable conversational context tracking.
4.  **[app.py](file:///home/administrator/Downloads/rag-chatbot/app.py)**: A beautiful Streamlit-based web interface built with a glassmorphic design and multi-turn chat history, including collapsible citation check lists.

---

## Pipeline Execution Details & Statistics

1.  **Ingestion**:
    - **Files Processed**: 113 circulars and notifications (plus 2 root files).
    - **Total Local Chunks Indexed**: **1,600** text blocks.
    - **Chroma Database Location**: `[workspace]/chroma_db/`
    - **Local Embeddings Engine**: `sentence-transformers/all-MiniLM-L6-v2` (run completely locally on CPU).

2.  **Database & Search Verification**:
    - **Hybrid Search**: Integrates semantic vector similarity search with BM25 keyword keyword search. This solves search queries containing specific acronyms and administrative terms (e.g. `MACP`, `FR 26`, `DPC`).
    - **Startup Performance Cache**: Serializes/caches the BM25 index on disk (`bm25_index.pkl`) on first cold run. Subsequent queries load the index instantly (less than 0.05 seconds) instead of rebuilding it from the 1,600 chunks, resulting in lightning-fast response times.
    - **Conversational Memory**: The chatbot tracks conversational history (`chat_history` list of tuples) and dynamically translates follow-up queries (like "Does it apply to English or Hindi?") into standalone search queries based on prior turns.

---

## How to Launch and Use the Chatbot

Follow these simple steps in your terminal to start using the system:

### 1. Run the local LLM using Ollama
Make sure Ollama is installed and running:
```bash
# Start the Ollama local model service
ollama run gemma2:9b
```
*(You can also use other local models such as `gemma2:2b`, `qwen2:7b`, or `llama3.1:8b`. Just make sure to download them first using `ollama pull <model_name>` and select them in the Streamlit app sidebar).*

### 2. Launch the Streamlit Web Application
From the workspace folder `/home/administrator/Downloads/rag-chatbot`, execute:
```bash
./venv/bin/streamlit run app.py
```

Streamlit will boot up and print the local URL:
```text
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.1.XX:8501
```

Open `http://localhost:8501` in your browser. You will see the local RAG chatbot interface where you can submit questions, view citations, and modify settings.

---

## Recommended Example Queries to Try:
*   "What is the typing speed required for LDC in English?" (Keyword: LDC, typing speed)
*   "What is the MACP scheme and when is it applicable?" (Follow up: "Does this apply to Staff Car Drivers?")
*   "What is the process for recovery of wrongful excess payments to government servants?" (Keyword: recovery, excess payments)
*   "What is FR 26 regarding increments?" (Acronym keyword search verification)
