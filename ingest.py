import os
import glob
import json
import time
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
        
    registry_path = os.path.join(db_dir, "file_registry.json")
    registry = {}
    
    if os.path.exists(registry_path):
        try:
            with open(registry_path, "r") as f:
                registry = json.load(f)
        except Exception as e:
            print(f"Warning: Failed to load file registry: {e}. Rebuilding registry.")
            registry = {}
            
    # Calculate current state of files on disk
    current_state = {}
    for path in pdf_paths:
        try:
            stat = os.stat(path)
            rel_path = os.path.relpath(path, workspace_dir)
            current_state[rel_path] = {"mtime": stat.st_mtime, "size": stat.st_size, "abs_path": path}
        except Exception:
            pass
            
    # Identify changes
    added_or_modified = []
    deleted = []
    
    for rel_path, info in current_state.items():
        if rel_path not in registry:
            added_or_modified.append(rel_path)
        elif registry[rel_path]["mtime"] != info["mtime"] or registry[rel_path]["size"] != info["size"]:
            added_or_modified.append(rel_path)
            
    for rel_path in registry:
        if rel_path not in current_state:
            deleted.append(rel_path)
            
    if not added_or_modified and not deleted:
        print("No additions, deletions or modifications detected. database is up-to-date.")
        return
        
    print(f"\nIncremental changes detected:")
    print(f"  - New or Modified: {len(added_or_modified)}")
    print(f"  - Removed: {len(deleted)}")
    
    print("\nLoading local embedding model ('all-MiniLM-L6-v2')...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    
    print(f"Opening Chroma DB at: {db_dir}...")
    db = Chroma(
        persist_directory=db_dir,
        embedding_function=embeddings
    )
    
    # Process deletions & modifications (delete old chunks)
    files_to_remove = deleted + added_or_modified
    if files_to_remove:
        print(f"\nRemoving stale chunks from database for {len(files_to_remove)} files...")
        for rel_path in files_to_remove:
            try:
                db.delete(where={"source": rel_path})
                print(f"  Removed chunks for: {rel_path}")
            except Exception as e:
                # delete might fail if collection is empty/new, which is fine
                pass
                
    # Process additions & modifications (extract new chunks)
    all_chunks = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    
    if added_or_modified:
        print(f"\nProcessing {len(added_or_modified)} new/modified files...")
        for idx, rel_path in enumerate(added_or_modified):
            pdf_path = current_state[rel_path]["abs_path"]
            parent_dir = os.path.basename(os.path.dirname(pdf_path))
            filename = os.path.basename(pdf_path)
            category = parent_dir if os.path.dirname(pdf_path) != workspace_dir else "Root"
            
            print(f"[{idx+1}/{len(added_or_modified)}] Processing {rel_path}...")
            try:
                loader = PyPDFLoader(pdf_path)
                docs = loader.load()
                for doc in docs:
                    doc.metadata['source'] = rel_path
                    doc.metadata['filename'] = filename
                    doc.metadata['category'] = category
                    
                chunks = text_splitter.split_documents(docs)
                all_chunks.extend(chunks)
            except Exception as e:
                print(f"  Error reading {rel_path}: {e}")
                
    # Add new chunks to DB in batches
    if all_chunks:
        batch_size = 4000
        print(f"\nAdding {len(all_chunks)} chunks to Chroma DB (batch size: {batch_size})...")
        for i in range(0, len(all_chunks), batch_size):
            batch = all_chunks[i:i + batch_size]
            print(f"  Inserting batch {i // batch_size + 1}/{-(-len(all_chunks) // batch_size)} ({len(batch)} chunks)...")
            try:
                db.add_documents(batch)
            except Exception as e:
                print(f"Error writing batch starting at index {i}: {e}")
                return
            
    # Update file registry
    for rel_path in deleted:
        if rel_path in registry:
            del registry[rel_path]
            
    for rel_path in added_or_modified:
        registry[rel_path] = {
            "mtime": current_state[rel_path]["mtime"],
            "size": current_state[rel_path]["size"]
        }
        
    os.makedirs(db_dir, exist_ok=True)
    with open(registry_path, "w") as f:
        json.dump(registry, f, indent=2)
    print("File registry updated on disk.")
    
    # Update last ingest timestamp
    last_ingest_path = os.path.join(db_dir, "last_ingest.txt")
    try:
        with open(last_ingest_path, "w") as f:
            f.write(str(time.time()))
        print("Last ingest timestamp updated.")
    except Exception as e:
        print(f"Warning: Could not update last_ingest timestamp: {e}")
        
    # Delete stale BM25 cache
    bm25_cache_path = os.path.join(workspace_dir, "bm25_index.pkl")
    if os.path.exists(bm25_cache_path):
        try:
            os.remove(bm25_cache_path)
            print("Stale BM25 cache cleared.")
        except Exception as e:
            print(f"Warning: Could not clear BM25 cache: {e}")
            
    print("Ingestion completed successfully!")

if __name__ == "__main__":
    main()
