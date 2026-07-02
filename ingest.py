import os
import glob
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

workspace_dir = os.path.dirname(os.path.abspath(__file__))
db_dir = os.path.join(workspace_dir, "chroma_db")

def find_pdfs(root_dir):
    """Find all PDF files recursively, ignoring virtual environments or db folders."""
    exclude_dirs = {'.git', 'venv', '.venv', 'chroma_db', 'node_modules', '__pycache__'}
    pdf_files = []
    
    for root, dirs, files in os.walk(root_dir):
        # Prune excluded directories in-place
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        for file in files:
            if file.lower().endswith('.pdf'):
                pdf_files.append(os.path.join(root, file))
                
    return pdf_files

def main():
    print(f"Scanning workspace for PDF documents in: {workspace_dir}...")
    pdf_paths = find_pdfs(workspace_dir)
    print(f"Found {len(pdf_paths)} PDF files to index.")
    
    if not pdf_paths:
        print("No PDFs found. Exiting.")
        return
        
    all_chunks = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    
    for idx, pdf_path in enumerate(pdf_paths):
        rel_path = os.path.relpath(pdf_path, workspace_dir)
        parent_dir = os.path.basename(os.path.dirname(pdf_path))
        filename = os.path.basename(pdf_path)
        
        # Category is either the direct folder name or "Root" if it's in the workspace root
        category = parent_dir if os.path.dirname(pdf_path) != workspace_dir else "Root"
        
        print(f"[{idx+1}/{len(pdf_paths)}] Processing {rel_path}...")
        
        try:
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            
            # Enrich metadata
            for doc in docs:
                doc.metadata['source'] = rel_path
                doc.metadata['filename'] = filename
                doc.metadata['category'] = category
                
            chunks = text_splitter.split_documents(docs)
            all_chunks.extend(chunks)
        except Exception as e:
            print(f"  Error reading {rel_path}: {e}")
            
    print(f"\nTotal text chunks generated: {len(all_chunks)}")
    
    if not all_chunks:
        print("No text chunks generated. Ingestion aborted.")
        return
        
    print("\nLoading local embedding model ('all-MiniLM-L6-v2')...")
    # This downloads and runs the embedding model locally inside the python process
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'} # Use 'cuda' if GPU is available and configured
    )
    
    print(f"Saving embeddings and chunks to local Chroma DB at: {db_dir}...")
    # Initialize and persist Chroma DB
    db = Chroma.from_documents(
        documents=all_chunks,
        embedding=embeddings,
        persist_directory=db_dir
    )
    
    print("Ingestion and indexing completed successfully!")
    print(f"Chroma DB contains {len(all_chunks)} chunks.")
    
    # Delete stale BM25 cache
    bm25_cache_path = os.path.join(workspace_dir, "bm25_index.pkl")
    if os.path.exists(bm25_cache_path):
        try:
            os.remove(bm25_cache_path)
            print("Stale BM25 cache cleared.")
        except Exception as e:
            print(f"Warning: Could not clear BM25 cache: {e}")

if __name__ == "__main__":
    main()
