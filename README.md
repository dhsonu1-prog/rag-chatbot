# 📚 Conversational RAG Chatbot: Localized Civil Services Rules Assistant

An automated, incremental Retrieval-Augmented Generation (RAG) system with a localized hybrid search pipeline, dynamic LLM-based categorization, real-time directory automation, and conversational reasoning for Government Civil Services guidelines.

---

## 🛠️ Technology Stack & Specifications

The project leverages a robust, offline-first production stack:

### 1. Vector Search & Database Layer
*   **Vector Database**: [Qdrant](https://qdrant.tech/) running in a standalone Docker container (`http://localhost:6333`). Includes an automatic local SQLite fallback directory (`path=db_dir`) if the standalone server is offline.
*   **Dense Embeddings**: `Qwen/Qwen3-VL-Embedding-2B` (2048-dimensional dense vectors) loaded via HuggingFace Embeddings.
*   **Sparse Embeddings**: Custom token-hashing sparse vectorizer using the **Adler32 Checksum Hash** technique to generate deterministic indices on-the-fly without maintaining a static vocabulary dictionary.

### 2. Context Reranking Layer
*   **Cross-Encoder**: `Qwen/Qwen3-VL-Reranker-2B` (~4GB VRAM) used to compute relevance scores for the top 20 candidate documents and select the best 5.
*   **VRAM Protection**: Self-healing try-except fallback that automatically shifts Cross-Encoder operations to the **workstation CPU** if the GPU is out of memory.

### 3. Parsing & OCR Layer
*   **PDF Extraction**: [PyMuPDF](https://pymupdf.readthedocs.io/) for fast digital text extraction.
*   **Visual OCR Fallback**: If a page has < 50 characters, it is treated as a scanned image and sent to the remote **Gemma-4-Vision** model on Port 3001 (`http://172.16.172.4:3001/v1/`) to transcribe using visual reasoning.

### 4. Language Models (LLM) & Serving
*   **Remote vLLM Options**:
    *   **Sarvam-105B (Port 3002)**: Running on SGLang. Highly optimized for fast classification and generation.
    *   **Gemma-4-Vision (Port 3001)**: Vision-and-text generation.
    *   **Qwen3.5-397B (Port 3003)**: Running on vLLM. Massive-parameter reasoning for complex rule queries.
*   **Local Workstation Options**:
    *   **Ollama (`gemma2:9b`, `gemma2:2b`, `llama3.1:8b`)**: Evaluated at startup via the local tags API (`http://localhost:11434/api/tags`) for 100% offline generation.

### 5. Frontend UI
*   **Streamlit**: Host app running on Port 8501 (`http://localhost:8501`). Features dynamic Qdrant status banners, dynamic file counters, dynamic Ollama model detection, and remote model select options.

---

## 🏗️ Core Algorithms & Ingestion Flow

1.  **Stage 0: LLM Segregation**: The first page of each PDF is read and sent to **Sarvam-105B** to decide whether it belongs to *Central Procurement*, *Finance*, or *Personnel*. Files are automatically moved to their corresponding folders.
2.  **Stage 2: Chunking & Text Extraction**: PyMuPDF extracts text from pages. If low-text page detected, Gemma-Vision OCR is triggered.
3.  **Stage 3: Hybrid Writing (Self-Healing)**:
    *   *Payload Limit & OOM Protection*: If the batch upsert fails (for example, exceeding Qdrant's 32 MB HTTP payload upload limit or CUDA VRAM OOM), the handler flushes PyTorch memory and recursively subdivisions the batch into **sub-batches of 50** chunks. This ensures zero data loss.
4.  **Retrieval RRF (Reciprocal Rank Fusion)**: Combines dense similarity search and sparse token keyword results using the mathematical formula:
    \[RRF\_Score(d) = \sum_{m \in M} \frac{1}{k + r_m(d)}\]
    where \(k=60\) and \(r_m(d)\) is the rank of document \(d\) in retriever \(m\).

---

## 🚀 Running the Project

### Setup Docker Qdrant DB
```bash
docker run -d --name qdrant-server -p 6333:6333 -p 6334:6334 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant
```

### Ingest Documents (Single Command)
```bash
python run_pipeline.py --skip-scrape --skip-clean
```
*Note*: Processed documents are saved in `qdrant_db/file_registry.json`. If ingestion is stopped, running the command again resumes from where it left off. Delete this file to force a complete rebuild from scratch.

### Start the Chat UI
```bash
streamlit run app.py --server.address=0.0.0.0 --server.port=8501
```

---

## 📋 Developer Customization

For setup detailed configurations, model endpoints modifications, and system customization guides, please check out the **[Developer Guide](file:///home/administrator/Downloads/rag-chatbot/developer_guide.md)** located in the root directory.
