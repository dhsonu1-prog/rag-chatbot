import os
import pickle
from langchain_qdrant import Qdrant
from qdrant_client import QdrantClient
from qdrant_client import models as qdrant_models
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from sparse_encoder import HashingSparseEncoder
import re

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
        from sentence_transformers import CrossEncoder
        import torch
        try:
            print(f"Loading reranker model '{model_name}' on GPU...")
            model = CrossEncoder(
                model_name,
                automodel_args={"torch_dtype": torch.float16},
                device="cuda"
            )
        except Exception as e:
            print(f"  [Reranker Warning] Failed to load reranker on GPU ({e}). Falling back to CPU...")
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            model = CrossEncoder(
                model_name,
                device="cpu"
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

class QdrantHybridRetriever(BaseRetriever):
    client: QdrantClient
    embeddings: HuggingFaceEmbeddings
    sparse_encoder: HashingSparseEncoder
    collection_name: str = "government_rules"
    top_k: int = 20
    
    # Active routing properties (can be updated dynamically)
    broad_category: str = None
    subcategory: str = None
    similarity_threshold: float = 0.70  # Cosine threshold for fallback routing

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None
    ) -> List[Document]:
        # 1. Generate query vectors
        query_dense = self.embeddings.embed_query(query)
        query_sparse = self.sparse_encoder.encode(query)
        
        # 2. Build filters dynamically
        sub_filter = None
        broad_filter = None
        
        if self.subcategory and self.subcategory != "All":
            sub_filter = qdrant_models.Filter(
                must=[
                    qdrant_models.FieldCondition(
                        key="metadata.subcategory",
                        match=qdrant_models.MatchValue(value=self.subcategory)
                    )
                ]
            )
            
        if self.broad_category and self.broad_category != "All":
            broad_filter = qdrant_models.Filter(
                must=[
                    qdrant_models.FieldCondition(
                        key="metadata.broad_category",
                        match=qdrant_models.MatchValue(value=self.broad_category)
                    )
                ]
            )
            
        # 3. Dynamic Fallback Routing Evaluation
        active_filter = None
        routing_level = "Global"
        
        if sub_filter:
            try:
                # Gauge quality of best semantic match within this specific subcategory
                test_res = self.client.query_points(
                    collection_name=self.collection_name,
                    query=query_dense,
                    filter=sub_filter,
                    limit=1
                ).points
                
                best_cosine = test_res[0].score if test_res else 0.0
                print(f"  [Router] Subcategory '{self.subcategory}' match quality (Cosine): {best_cosine:.4f}")
                
                if best_cosine >= self.similarity_threshold:
                    active_filter = sub_filter
                    routing_level = f"Subcategory ({self.subcategory})"
                elif broad_filter:
                    print(f"  [Router Warning] Match quality below {self.similarity_threshold}. Falling back to Broad Category...")
                    active_filter = broad_filter
                    routing_level = f"Broad Category ({self.broad_category})"
                else:
                    print(f"  [Router Warning] Match quality below {self.similarity_threshold}. Falling back to Global...")
                    active_filter = None
                    routing_level = "Global"
            except Exception as e:
                print(f"  [Router Warning] Fallback quality check failed: {e}. Defaulting to subcategory filter.")
                active_filter = sub_filter
                routing_level = f"Subcategory ({self.subcategory})"
        elif broad_filter:
            active_filter = broad_filter
            routing_level = f"Broad Category ({self.broad_category})"
            
        print(f"  [Router] Executing hybrid query at level: {routing_level}")
        
        # 4. Execute the actual pre-filtered hybrid search
        if not query_sparse["indices"]:
            # Fallback to dense only if no sparse terms are present
            results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_dense,
                filter=active_filter,
                limit=self.top_k
            ).points
        else:
            try:
                results = self.client.query_points(
                    collection_name=self.collection_name,
                    prefetch=[
                        qdrant_models.Prefetch(
                            query=query_dense,
                            using="",  # default unnamed dense vector
                            filter=active_filter,
                            limit=self.top_k * 2
                        ),
                        qdrant_models.Prefetch(
                            query=qdrant_models.SparseVector(
                                indices=query_sparse["indices"],
                                values=query_sparse["values"]
                            ),
                            using="sparse-text",
                            filter=active_filter,
                            limit=self.top_k * 2
                        )
                    ],
                    query=qdrant_models.FusionQuery(
                        fusion=qdrant_models.Fusion.RRF
                    ),
                    limit=self.top_k
                ).points
            except Exception as query_err:
                print(f"  [Qdrant Hybrid Warning] Hybrid query failed: {query_err}. Falling back to dense vector search.")
                results = self.client.query_points(
                    collection_name=self.collection_name,
                    query=query_dense,
                    filter=active_filter,
                    limit=self.top_k
                ).points
            
        # 5. Map results to LangChain Documents
        docs = []
        for point in results:
            payload = point.payload or {}
            metadata = payload.get("metadata", {})
            docs.append(Document(
                page_content=payload.get("page_content", ""),
                metadata=metadata
            ))
        return docs

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
        
    # 1. Load local embedding model on CPU to save 5GB+ GPU VRAM (query is fast on CPU)
    embeddings = HuggingFaceEmbeddings(
        model_name="Qwen/Qwen3-VL-Embedding-2B",
        model_kwargs={'device': 'cpu'}
    )
    
    # 2. Load Qdrant DB
    try:
        client = QdrantClient(url="http://localhost:6333", timeout=3.0)
        client.get_collections()
        print("Connected to standalone Qdrant server at http://localhost:6333")
    except Exception:
        if not os.path.exists(db_dir):
            raise FileNotFoundError(f"Qdrant DB directory not found at {db_dir}. Please run 'python ingest.py' first.")
        print(f"Standalone Qdrant server offline. Falling back to local db path: {db_dir}")
        client = QdrantClient(path=db_dir)
    if not client.collection_exists("government_rules"):
        raise ValueError(f"The Qdrant collection 'government_rules' does not exist. Please run 'python ingest.py' first.")
        
    db = Qdrant(
        client=client,
        collection_name="government_rules",
        embeddings=embeddings
    )
    
    # 3. Create Native Qdrant Hybrid retriever (replacing Python BM25 / pkl)
    candidate_k = 20 if use_reranker else 8
    sparse_encoder = HashingSparseEncoder()
    hybrid_retriever = QdrantHybridRetriever(
        client=client,
        embeddings=embeddings,
        sparse_encoder=sparse_encoder,
        collection_name="government_rules",
        top_k=candidate_k
    )
    
    retriever = RerankEnsembleRetriever(
        base_retriever=hybrid_retriever,
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
        # Strip Qwen/DeepSeek thinking blocks if present (e.g. <think>...</think>)
        answer = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL).strip()
        
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
