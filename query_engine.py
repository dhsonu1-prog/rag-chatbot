import os
import pickle
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever

workspace_dir = os.path.dirname(os.path.abspath(__file__))
db_dir = os.path.join(workspace_dir, "chroma_db")
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

def get_rag_chain(model_name="gemma2:9b", api_base=None, api_model=None):
    """Initialize and return the hybrid conversational RAG chain."""
    global _chain_cache
    cache_key = (model_name, api_base, api_model)
    if cache_key in _chain_cache:
        return _chain_cache[cache_key]
        
    if not os.path.exists(db_dir):
        raise FileNotFoundError(f"Chroma DB directory not found at {db_dir}. Please run 'python ingest.py' first.")
        
    # 1. Load local embedding model
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    
    # 2. Load Chroma DB
    db = Chroma(
        persist_directory=db_dir,
        embedding_function=embeddings
    )
    
    # 3. Create Vector retriever (k=8)
    vector_retriever = db.as_retriever(search_kwargs={"k": 8})
    
    # 4. Load or build BM25 retriever (k=8)
    bm25_retriever = None
    if os.path.exists(bm25_cache_path):
        try:
            with open(bm25_cache_path, "rb") as f:
                bm25_retriever = pickle.load(f)
            # Ensure k is set to 8
            bm25_retriever.k = 8
        except Exception as e:
            print(f"Warning: Failed to load BM25 cache: {e}. Rebuilding...")
            bm25_retriever = None
            
    if bm25_retriever is None:
        res = db.get(include=["documents", "metadatas"])
        docs = [Document(page_content=doc, metadata=meta) for doc, meta in zip(res["documents"], res["metadatas"])]
        bm25_retriever = BM25Retriever.from_documents(docs)
        bm25_retriever.k = 8
        try:
            with open(bm25_cache_path, "wb") as f:
                pickle.dump(bm25_retriever, f)
        except Exception as e:
            print(f"Warning: Failed to save BM25 cache: {e}")
            
    # 5. Combine into Ensemble (Hybrid) retriever
    retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.5, 0.5]
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
    
    _chain_cache[cache_key] = qa_chain
    return qa_chain

def query_rag(query_text, model_name="gemma2:9b", chat_history=None, api_base=None, api_model=None):
    """Query the local conversational RAG pipeline and return the answer and source documents."""
    if chat_history is None:
        chat_history = []
        
    try:
        chain = get_rag_chain(model_name=model_name, api_base=api_base, api_model=api_model)
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
