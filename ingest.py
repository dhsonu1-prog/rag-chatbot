import os
import glob
import json
import time
import base64
import requests
import uuid
import fitz  # PyMuPDF for converting PDF pages to images
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_qdrant import Qdrant
from qdrant_client import QdrantClient
from qdrant_client import models as qdrant_models
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from sparse_encoder import HashingSparseEncoder

sparse_encoder = HashingSparseEncoder()

def add_documents_hybrid(client, embeddings, sparse_encoder, collection_name, chunks):
    if not chunks:
        return
    # 1. Generate dense vectors
    texts = [chunk.page_content for chunk in chunks]
    dense_vectors = embeddings.embed_documents(texts)
    
    # 2. Map chunks to PointStructs with dual vectors (dense & sparse-text)
    points = []
    for doc, dense_vector in zip(chunks, dense_vectors):
        doc_id = str(uuid.uuid4())
        sparse_vec = sparse_encoder.encode(doc.page_content)
        
        points.append(
            qdrant_models.PointStruct(
                id=doc_id,
                vector={
                    "": dense_vector,  # default unnamed vector for LangChain compatibility
                    "sparse-text": qdrant_models.SparseVector(
                        indices=sparse_vec["indices"],
                        values=sparse_vec["values"]
                    )
                },
                payload={
                    "page_content": doc.page_content,
                    "metadata": doc.metadata
                }
            )
        )
    # 3. Upsert into Qdrant
    client.upsert(
        collection_name=collection_name,
        points=points
    )

workspace_dir = os.path.dirname(os.path.abspath(__file__))
db_dir = os.path.join(workspace_dir, "qdrant_db")

# Supported file formats
SUPPORTED_EXTENSIONS = ('.pdf', '.png', '.jpg', '.jpeg', '.tiff')

def find_documents(root_dir):
    """Find all supported files recursively, ignoring virtual environments or db folders."""
    exclude_dirs = {'.git', 'venv', '.venv', 'qdrant_db', 'node_modules', '__pycache__'}
    doc_files = []
    
    for root, dirs, files in os.walk(root_dir):
        # Prune excluded directories in-place
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        for file in files:
            if file.lower().endswith(SUPPORTED_EXTENSIONS):
                doc_files.append(os.path.join(root, file))
                
    return doc_files

# Keep find_pdfs function for backward compatibility
def find_pdfs(root_dir):
    return find_documents(root_dir)

def ocr_page_pymupdf(pdf_path, page_num):
    """Render a single PDF page to an image and run OCR on it using gemma-4-vision."""
    try:
        doc = fitz.open(pdf_path)
        page = doc.load_page(page_num)
        pix = page.get_pixmap(dpi=150)
        img_bytes = pix.tobytes("jpeg")
        img_b64 = base64.b64encode(img_bytes).decode("utf-8")
    except Exception as e:
        print(f"  [OCR Error] Failed to render PDF page {page_num} of {pdf_path}: {e}")
        return ""

    url = "http://172.16.172.4:3001/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    data = {
        "model": "gemma-4-vision",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Perform OCR on this page. Extract all readable text verbatim. Maintain layout and structure as much as possible. Respond ONLY with the extracted text, do not explain or add commentary."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img_b64}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 1500,
        "temperature": 0.0
    }
    
    try:
        response = requests.post(url, headers=headers, json=data, timeout=30)
        if response.status_code == 200:
            res_json = response.json()
            return res_json["choices"][0]["message"]["content"]
        else:
            print(f"  [OCR Error] API returned status {response.status_code}: {response.text}")
    except Exception as e:
        print(f"  [OCR Error] Connection failed during page OCR: {e}")
    return ""

def ocr_image(image_path):
    """Run OCR on a standalone image file using gemma-4-vision."""
    try:
        with open(image_path, "rb") as f:
            img_bytes = f.read()
        img_b64 = base64.b64encode(img_bytes).decode("utf-8")
    except Exception as e:
        print(f"  [OCR Error] Failed to read image {image_path}: {e}")
        return ""

    mime_type = "image/jpeg"
    ext = image_path.lower()
    if ext.endswith(".png"):
        mime_type = "image/png"
    elif ext.endswith(".tiff") or ext.endswith(".tif"):
        mime_type = "image/tiff"

    url = "http://172.16.172.4:3001/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    data = {
        "model": "gemma-4-vision",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Perform OCR on this image. Extract all readable text verbatim. Respond ONLY with the extracted text, do not explain or add commentary."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{img_b64}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 1500,
        "temperature": 0.0
    }
    
    try:
        response = requests.post(url, headers=headers, json=data, timeout=30)
        if response.status_code == 200:
            res_json = response.json()
            return res_json["choices"][0]["message"]["content"]
        else:
            print(f"  [OCR Error] API returned status {response.status_code}: {response.text}")
    except Exception as e:
        print(f"  [OCR Error] Connection failed during image OCR: {e}")
    return ""

def main():
    print(f"Scanning workspace for documents (PDFs and images) in: {workspace_dir}...")
    doc_paths = find_documents(workspace_dir)
    print(f"Found {len(doc_paths)} documents to index.")
    
    if not doc_paths:
        print("No documents found. Exiting.")
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
    for path in doc_paths:
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
    
    print("\nLoading local embedding model ('Qwen/Qwen3-VL-Embedding-2B') on GPU...")
    embeddings = HuggingFaceEmbeddings(
        model_name="Qwen/Qwen3-VL-Embedding-2B",
        model_kwargs={'device': 'cuda'}
    )
    
    print(f"Opening Qdrant DB at: {db_dir}...")
    client = QdrantClient(path=db_dir)
    try:
        if not client.collection_exists("government_rules"):
            client.create_collection(
                collection_name="government_rules",
                vectors_config=qdrant_models.VectorParams(
                    size=2048,
                    distance=qdrant_models.Distance.COSINE
                ),
                sparse_vectors_config={
                    "sparse-text": qdrant_models.SparseVectorParams(
                        index=qdrant_models.SparseIndexParams(
                            on_disk=True
                        )
                    )
                }
            )
            print("  Created new Qdrant collection: 'government_rules'")
            
            # Create payload indexes for fast O(1) pre-filtering
            try:
                client.create_payload_index(
                    collection_name="government_rules",
                    field_name="metadata.broad_category",
                    field_schema=qdrant_models.PayloadSchemaType.KEYWORD
                )
                client.create_payload_index(
                    collection_name="government_rules",
                    field_name="metadata.subcategory",
                    field_schema=qdrant_models.PayloadSchemaType.KEYWORD
                )
                client.create_payload_index(
                    collection_name="government_rules",
                    field_name="metadata.source",
                    field_schema=qdrant_models.PayloadSchemaType.KEYWORD
                )
                print("  Payload indexes created successfully for 'broad_category', 'subcategory', and 'source'.")
            except Exception as index_err:
                print(f"  Warning: Could not create payload indexes: {index_err}")
    except Exception as e:
        print(f"Error initializing Qdrant database collection: {e}")

    db = Qdrant(
        client=client,
        collection_name="government_rules",
        embeddings=embeddings
    )
    
    # Process deletions & modifications (delete old chunks)
    files_to_remove = deleted + added_or_modified
    if files_to_remove:
        print(f"\nRemoving stale chunks from database for {len(files_to_remove)} files...")
        try:
            if client.collection_exists("government_rules"):
                for rel_path in files_to_remove:
                    try:
                        client.delete(
                            collection_name="government_rules",
                            points_selector=qdrant_models.FilterSelector(
                                filter=qdrant_models.Filter(
                                    must=[
                                        qdrant_models.FieldCondition(
                                            key="metadata.source",
                                            match=qdrant_models.MatchValue(value=rel_path)
                                        )
                                    ]
                                )
                            )
                        )
                        print(f"  Removed chunks for: {rel_path}")
                    except Exception as e:
                        pass
        except Exception:
            pass
                
    # Process additions & modifications (extract new chunks)
    all_chunks = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    
    if added_or_modified:
        print(f"\nProcessing {len(added_or_modified)} new/modified files...")
        for idx, rel_path in enumerate(added_or_modified):
            file_path = current_state[rel_path]["abs_path"]
            filename = os.path.basename(file_path)
            
            # Parse broad_category and subcategory from relative path
            parts = rel_path.split(os.sep)
            broad_category = "General"
            subcategory = "General"
            
            if len(parts) >= 3:
                broad_category = parts[1]
                subcategory = parts[2]
            elif len(parts) == 2:
                broad_category = parts[1]
            
            print(f"[{idx+1}/{len(added_or_modified)}] Processing {rel_path}...")
            try:
                docs = []
                if file_path.lower().endswith('.pdf'):
                    loader = PyPDFLoader(file_path)
                    docs = loader.load()
                    for doc in docs:
                        page_num = doc.metadata.get('page', 0)
                        text = doc.page_content.strip()
                        # If page text is very short/empty, run OCR fallback
                        if len(text) < 50:
                            print(f"  [OCR] Low text content detected on page {page_num+1} ({len(text)} chars). Running Vision OCR fallback...")
                            ocr_text = ocr_page_pymupdf(file_path, page_num)
                            if ocr_text:
                                doc.page_content = ocr_text
                                print(f"  [OCR] Successfully extracted {len(ocr_text)} chars from page {page_num+1}.")
                            else:
                                print(f"  [OCR] Failed or empty result for page {page_num+1}.")
                else:
                    # Image file
                    print(f"  [OCR] Image file detected. Running Vision OCR...")
                    ocr_text = ocr_image(file_path)
                    if ocr_text:
                        print(f"  [OCR] Successfully extracted {len(ocr_text)} chars from image.")
                        doc = Document(page_content=ocr_text, metadata={"page": 0})
                        docs = [doc]
                    else:
                        print(f"  [OCR] Failed or empty result for image.")
                
                for doc in docs:
                    doc.metadata['source'] = rel_path
                    doc.metadata['filename'] = filename
                    doc.metadata['broad_category'] = broad_category
                    doc.metadata['subcategory'] = subcategory
                    
                    # Prepend category and document context to enrich vectors
                    broad_name = broad_category.split('_', 1)[-1]
                    sub_name = subcategory.replace('_', ' ')
                    context_prefix = f"[Category: {broad_name} > {sub_name}] [Document: {filename}]\n\n"
                    doc.page_content = context_prefix + doc.page_content
                    
                chunks = text_splitter.split_documents(docs)
                all_chunks.extend(chunks)
                
                # Write to DB incrementally in batches of 400 chunks to prevent memory bloat/OOM
                if len(all_chunks) >= 400:
                    print(f"  Writing batch of {len(all_chunks)} chunks to Qdrant DB...")
                    try:
                        add_documents_hybrid(client, embeddings, sparse_encoder, "government_rules", all_chunks)
                    except Exception as e:
                        print(f"  Error writing batch to Qdrant DB: {e}")
                    finally:
                        all_chunks = []
            except Exception as e:
                print(f"  Error reading {rel_path}: {e}")
                
        # Write any remaining chunks to Qdrant DB
        if all_chunks:
            print(f"  Writing final batch of {len(all_chunks)} chunks to Qdrant DB...")
            try:
                add_documents_hybrid(client, embeddings, sparse_encoder, "government_rules", all_chunks)
            except Exception as e:
                print(f"Error writing final batch to Qdrant DB: {e}")
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
