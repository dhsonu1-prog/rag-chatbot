# Slide Presentation: Localized Civil Services RAG Assistant

---

## 📺 Slide 1: Cover & Project Identity
### **Conversational RAG Chatbot: Localized Civil Services Rule Assistant**

*   **Problem Statement ID**: `RAG-CS-001`
*   **Project Title**: Localized Conversational AI & Directory Automation for Civil Service Guidelines
*   **Team Composition**: AI engineering Team (Pair Programmed with Gemini Antigravity)
*   **Core Objective**: Build a secure, offline-first Retrieval-Augmented Generation system with dual dense-sparse hybrid search, vision-based OCR fallback, and dynamic folder-relaxed routing for government administrative rules.

---

## 📺 Slide 2: Problem Statement & Key Challenges
### **The Administrative Rule Verification Problem**

*   **Problem Statement**: Government personnel waste hours manually verifying administrative queries against thousands of pages of circulars, office memorandums, and financial guidelines (such as GFR/CPWD manuals).
*   **Key Challenges**:
    1.  **Manual Latency**: Navigating large manuals (>100MB) blocks quick decision-making.
    2.  **Hallucination & Cloud Leak Risks**: Public LLM APIs cannot handle sensitive directives and are prone to hallucinations.
    3.  **Indexing Overhead**: Rebuilding entire database vectors from scratch on directory updates is highly resource-inefficient.

---

## 📺 Slide 3: Proposed Solution, Workflow & Overall Impact
### **Dual Hybrid Vector Retrieval & Dynamic Automation**

*   **Proposed Solution**: Native Qdrant Reciprocal Rank Fusion (RRF) combining semantic dense embeddings (Qwen3) and exact keyword sparse hashes (Adler32 Checksum Hashing).
*   **System Workflow**:
    1.  *LLM Segregation*: Auto-classifies files recursively into broad categories (CPC, Finance, Personnel).
    2.  *Vision OCR*: Falls back to Gemma-4-Vision (Port 3001) for scanned documents.
    3.  *Hybrid Indexing*: Generates dense-sparse vectors in a single SQLite Qdrant point.
    4.  *Dynamic Fallback Routing*: Relaxes filters to parent folders if top cosine score < 0.70.
*   **Overall Impact**: Latency lowered to **3.6 seconds**, zero-hallucination rate, and 100% offline data security.

---

## 📺 Slide 4: Dataset Details (AI, Analytics & Automation)
### **The Three-Tier Project Dataset**

*   **AI Dataset (RAG Corpus)**:
    *   *Format*: 514 PDFs/images (~6,500 chunks).
    *   *Categorization*: Segregated into 3 broad folders and 13 subcategories.
    *   *Type*: Unlabeled raw text with directory metadata breadcrumbs.
*   **Analytics Dataset (Performance Logs)**:
    *   *Format*: JSON/CSV logs of execution times, cosine scores, and RRF rank list outcomes.
    *   *Type*: Labeled performance data.
*   **Automation Dataset (Workflow Config)**:
    *   *Format*: Unified YAML/Python execution config mapping OCR fallbacks, DB purges, and server states.

---

## 📺 Slide 5: Evaluation Criteria & Deployment Strategy
### **Outcome Measurement & Standalone Network Host**

*   **Evaluation Criteria**:
    *   **Retrieval Hit Rate**: Accuracy of chunk retrieval via RRF rankings.
    *   **Fallback Reliability**: Correct trigger rate of parent-routing when subfolder scores < 0.70.
    *   **Zero-Hallucination Rate**: System must return "not found" instead of fabricating answers if source circulars are missing.
*   **Deployment Strategy**:
    *   **Local Workstations**: Host on local GPU server utilizing Ollama and Docker.
    *   **Standalone Network SOP**: Compile the code and clean hierarchical directory layout to zip to deploy directly onto isolated standalone government networks.
