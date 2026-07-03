import os
import pickle
from langchain_qdrant import Qdrant
from qdrant_client import QdrantClient
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever

workspace_dir = os.path.dirname(os.path.abspath(__file__))
db_dir = os.path.join(workspace_dir, "qdrant_db")
bm25_cache_path = os.path.join(workspace_dir, "bm25_index.pkl")

# Strict, robust prompt template for local RAG
RAG_PROMPT_TEMPLATE = """You are an expert administrative assistant for Government Civil Services rules and guidelines.
Answer the user's question as accurately and concisely as possible using ONLY the context provided below.

CRITICAL INSTRUCTIONS:
1. Rely ONLY on the facts directly mentioned in the Context. Do NOT use outside knowledge or make assumptions.
2. If the Context contains partial information, answer with those facts, but clearly state what is missing.
3. If the Context does not contain the answer at all, say: "I cannot find the answer in the downloaded documents." Do not say anything else.

Context:
{context}

Question: {question}

Helpful Answer:"""

# Cache to avoid rebuilding the chain on every query
_chain_cache = {}
_reranker_cache = {}

from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from typing import List

def get_reranker(model_name="Qwen/Qwen3-VL-Reranker-2B"):
    global _reranker_cache
    if model_name not in _reranker_cache:
        print(f"Loading reranker model '{model_name}' on GPU...")
        from sentence_transformers import CrossEncoder
        import torch
        model = CrossEncoder(
            model_name,
            automodel_args={"torch_dtype": torch.float16},
            device="cuda"
        )
        _reranker_cache[model_name] = model
    return _reranker_cache[model_name]

class RerankEnsembleRetriever(BaseRetriever):
    base_retriever: BaseRetriever
    use_reranker: bool = True
    top_n: int = 5
    
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None
    ) -> List[Document]:
        # 1. Retrieve candidates (e.g. get top 20 docs)
        docs = self.base_retriever.invoke(query)
        if not self.use_reranker or not docs:
            return docs[:self.top_n]
            
        try:
            model = get_reranker("Qwen/Qwen3-VL-Reranker-2B")
            
            # 2. Score documents
            scored_docs = []
            for doc in docs:
                score = model.predict([query, doc.page_content])
                scored_docs.append((score, doc))
                
            # 3. Sort by score descending and take top_n
            scored_docs.sort(key=lambda x: x[0], reverse=True)
            reranked_docs = [doc for score, doc in scored_docs[:self.top_n]]
            print(f"  [Reranker] Successfully reranked {len(docs)} documents to top {len(reranked_docs)}.")
            return reranked_docs
        except Exception as e:
            print(f"  [Reranker Warning] Reranking failed: {e}. Falling back to default retrieval.")
            return docs[:self.top_n]

def get_rag_chain(model_name="gemma2:9b", api_base=None, api_model=None, vector_weight=0.5, bm25_weight=0.5, use_reranker=True, top_n_rerank=5):
    """Initialize and return the hybrid conversational RAG chain."""
    global _chain_cache
    import time
    
    cache_key = (model_name, api_base, api_model, vector_weight, bm25_weight, use_reranker, top_n_rerank)
    
    # Check if last_ingest timestamp exists to handle hot-reloading
    last_ingest_time = 0.0
    last_ingest_path = os.path.join(db_dir, "last_ingest.txt")
    if os.path.exists(last_ingest_path):
        try:
            with open(last_ingest_path, "r") as f:
                last_ingest_time = float(f.read().strip())
        except Exception:
            pass
            
    if cache_key in _chain_cache:
        cached_entry = _chain_cache[cache_key]
        if cached_entry["timestamp"] >= last_ingest_time:
            return cached_entry["chain"]
        else:
            # Stale cache detected, delete it
            del _chain_cache[cache_key]
        
    if not os.path.exists(db_dir):
        raise FileNotFoundError(f"Qdrant DB directory not found at {db_dir}. Please run 'python ingest.py' first.")
        
    # 1. Load local embedding model
    embeddings = HuggingFaceEmbeddings(
        model_name="Qwen/Qwen3-VL-Embedding-2B",
        model_kwargs={'device': 'cuda'}
    )
    
    # 2. Load Qdrant DB
    client = QdrantClient(path=db_dir)
    if not client.collection_exists("government_rules"):
        raise ValueError(f"The Qdrant collection 'government_rules' does not exist. Please run 'python ingest.py' first.")
        
    db = Qdrant(
        client=client,
        collection_name="government_rules",
        embeddings=embeddings
    )
    
    # 3. Create Vector retriever
    candidate_k = 20 if use_reranker else 8
    vector_retriever = db.as_retriever(search_kwargs={"k": candidate_k})
    
    # 4. Load or build BM25 retriever
    bm25_retriever = None
    if os.path.exists(bm25_cache_path):
        try:
            with open(bm25_cache_path, "rb") as f:
                bm25_retriever = pickle.load(f)
            bm25_retriever.k = candidate_k
        except Exception as e:
            print(f"Warning: Failed to load BM25 cache: {e}. Rebuilding...")
            bm25_retriever = None
            
    if bm25_retriever is None:
        records = []
        offset = None
        while True:
            res_scroll, next_page = client.scroll(
                collection_name="government_rules",
                limit=1000,
                with_payload=True,
                with_vectors=False,
                offset=offset
            )
            records.extend(res_scroll)
            if next_page is None:
                break
            offset = next_page
            
        docs = []
        for rec in records:
            payload = rec.payload or {}
            if "page_content" in payload:
                docs.append(Document(
                    page_content=payload["page_content"],
                    metadata=payload.get("metadata", {})
                ))
                
        if not docs:
            raise ValueError("The Qdrant database is currently empty. Please run ingestion first using 'python ingest.py'.")
        bm25_retriever = BM25Retriever.from_documents(docs)
        bm25_retriever.k = candidate_k
        try:
            with open(bm25_cache_path, "wb") as f:
                pickle.dump(bm25_retriever, f)
        except Exception as e:
            print(f"Warning: Failed to save BM25 cache: {e}")
            
    # 5. Combine into Ensemble (Hybrid) retriever
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[bm25_weight, vector_weight]
    )
    
    retriever = RerankEnsembleRetriever(
        base_retriever=ensemble_retriever,
        use_reranker=use_reranker,
        top_n=top_n_rerank
    )
    
    # 6. Load local LLM (Ollama) or custom OpenAI-compatible endpoint (vLLM)
    if api_base:
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            openai_api_key="none",
            openai_api_base=api_base,
            model_name=api_model or "/mnt/ai_storage/models/Qwen3.5-397B-A17B-FP8-dynamic",
            temperature=0.0
        )
    else:
        llm = OllamaLLM(model=model_name)
    
    # 7. Define prompt
    prompt = PromptTemplate(
        template=RAG_PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )
    
    # 8. Build the ConversationalRetrievalChain
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": prompt}
    )
    
    _chain_cache[cache_key] = {
        "chain": qa_chain,
        "timestamp": time.time()
    }
    return qa_chain

def query_rag(query_text, model_name="gemma2:9b", chat_history=None, api_base=None, api_model=None, vector_weight=0.5, bm25_weight=0.5, use_reranker=True, top_n_rerank=5):
    """Query the local conversational RAG pipeline and return the answer and source documents."""
    if chat_history is None:
        chat_history = []
        
    try:
        chain = get_rag_chain(
            model_name=model_name, 
            api_base=api_base, 
            api_model=api_model, 
            vector_weight=vector_weight, 
            bm25_weight=bm25_weight,
            use_reranker=use_reranker,
            top_n_rerank=top_n_rerank
        )
        response = chain.invoke({"question": query_text, "chat_history": chat_history})
        
        answer = response["answer"]
        source_docs = response["source_documents"]
        
        # Deduplicate sources
        unique_sources = {}
        for doc in source_docs:
            src = doc.metadata.get('source', 'Unknown')
            page = doc.metadata.get('page', 0) + 1 # 0-indexed to 1-indexed
            if src not in unique_sources:
                unique_sources[src] = []
            unique_sources[src].append(page)
            
        return answer, unique_sources
    except Exception as e:
        return f"Error: {e}", {}

if __name__ == "__main__":
    import sys
    query = "What is the age limit for deputation?" if len(sys.argv) < 2 else sys.argv[1]
    model = "gemma2:9b" if len(sys.argv) < 3 else sys.argv[2]
    
    print(f"Querying local hybrid RAG using model '{model}' for: '{query}'...\n")
    ans, sources = query_rag(query, model_name=model)
    
    print("--- ANSWER ---")
    print(ans)
    print("\n--- SOURCES ---")
    for src, pages in sources.items():
        pages_str = ", ".join(map(str, sorted(set(pages))))
        print(f"- {src} (Page {pages_str})")
