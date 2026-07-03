import os
import re
import hashlib
import shutil
import fitz  # Requires PyMuPDF (pip install pymupdf)

# Keywords indicating a garbage or error page
ERROR_PHRASES = [
    "404 not found", "page not found", "server error", "forbidden", "unauthorized",
    "access denied", "site maintenance", "temporarily unavailable", "under construction",
    "apache tomcat", "iis windows server", "internal server error", "sql exception",
    "database error", "connection failed", "error 404", "requested url was not found",
    "resource you are looking for"
]

# Keywords for garbage filenames
GARBAGE_KEYWORDS = [
    'accessibility', 'certificate', 'stqc', 'sitemap', 'feedback', 
    'contact', 'about-us', 'menu', 'logo', 'banner', 'holiday', 
    'website-policy', 'screen-reader', 'help', 'twitter', 'facebook'
]

def is_valid_pdf_file(filepath):
    """Checks if the file meets the minimum size and starts with the %PDF magic header."""
    # 1. Size check: minimum 5 KB
    if os.path.getsize(filepath) < 5120:
        return False
    # 2. PDF Header check
    try:
        with open(filepath, 'rb') as f:
            header = f.read(4)
            return header == b'%PDF'
    except Exception:
        return False

def get_file_hash(filepath):
    """Computes MD5 hash of the file content."""
    hasher = hashlib.md5()
    try:
        with open(filepath, 'rb') as f:
            buf = f.read(65536)
            while len(buf) > 0:
                hasher.update(buf)
                buf = f.read(65536)
        return hasher.hexdigest()
    except Exception:
        return None

def clean_documents_pipeline(directory):
    print("=" * 60)
    print("STARTING RULE DOCUMENT CLEANUP PIPELINE")
    print("=" * 60)
    
    deleted_garbage_files = 0
    deleted_corrupt_files = 0
    deleted_binary_duplicates = 0
    deleted_content_duplicates = 0
    cleaned_folders_count = 0
    empty_folders_count = 0
    scanned_image_pdfs = 0
    total_scanned = 0
    
    # ----------------------------------------------------
    # Step 1: Clean folder names and merge if necessary
    # ----------------------------------------------------
    print("\n[Step 1] Sanitizing category folder names...")
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isdir(item_path):
            cleaned_name = item.strip()
            cleaned_name = " ".join(cleaned_name.split())
            if cleaned_name != item:
                new_path = os.path.join(directory, cleaned_name)
                if os.path.exists(new_path):
                    print(f"  Merging folder '{item}' into '{cleaned_name}'...")
                    for file in os.listdir(item_path):
                        src_file = os.path.join(item_path, file)
                        dst_file = os.path.join(new_path, file)
                        if not os.path.exists(dst_file):
                            try:
                                shutil.move(src_file, dst_file)
                            except Exception as e:
                                print(f"    Error moving {src_file}: {e}")
                        else:
                            try:
                                os.remove(src_file)
                            except Exception:
                                pass
                    try:
                        os.rmdir(item_path)
                    except Exception:
                        pass
                else:
                    print(f"  Renaming folder '{item}' to '{cleaned_name}'...")
                    try:
                        shutil.move(item_path, new_path)
                    except Exception as e:
                        print(f"    Error renaming {item_path}: {e}")
                cleaned_folders_count += 1

    # ----------------------------------------------------
    # Step 2: Validate signatures, sizes, and clean garbage keywords
    # ----------------------------------------------------
    print("\n[Step 2] Validating file sizes, headers, and keyword exclusions...")
    for root, dirs, files in os.walk(directory):
        for file in files:
            if not file.lower().endswith('.pdf'):
                continue
                
            total_scanned += 1
            filepath = os.path.join(root, file)
            file_lower = file.lower()
            
            # Check filename for garbage keywords
            if any(kw in file_lower for kw in GARBAGE_KEYWORDS):
                print(f"  Garbage keyword matched: {file} -> Deleting.")
                try:
                    os.remove(filepath)
                    deleted_garbage_files += 1
                except Exception as e:
                    print(f"    Error deleting {file}: {e}")
                continue
                
            # Check valid PDF signature and size
            if not is_valid_pdf_file(filepath):
                print(f"  Invalid PDF signature/size check failed: {file} -> Deleting.")
                try:
                    os.remove(filepath)
                    deleted_corrupt_files += 1
                except Exception as e:
                    print(f"    Error deleting {file}: {e}")
                continue

    # ----------------------------------------------------
    # Step 3: Binary Hash Deduplication & Inner Text Content Verification
    # ----------------------------------------------------
    print("\n[Step 3] Performing binary hashing and inner text error analysis...")
    seen_hashes = {}
    seen_content_texts = {}
    
    # We walk again because some files were deleted in Step 2
    for root, dirs, files in os.walk(directory):
        for file in files:
            if not file.lower().endswith('.pdf'):
                continue
                
            filepath = os.path.join(root, file)
            
            # A. Binary hash deduplication
            file_hash = get_file_hash(filepath)
            if file_hash:
                if file_hash in seen_hashes:
                    print(f"  Binary duplicate found: {file} (Original: {os.path.basename(seen_hashes[file_hash])}) -> Deleting.")
                    try:
                        os.remove(filepath)
                        deleted_binary_duplicates += 1
                    except Exception as e:
                        print(f"    Error: {e}")
                    continue
                else:
                    seen_hashes[file_hash] = filepath
            
            # B. Read page content using PyMuPDF (fitz)
            try:
                doc = fitz.open(filepath)
                page_count = doc.page_count
                if page_count == 0:
                    print(f"  Read failed (0 pages): {file} -> Deleting.")
                    doc.close()
                    os.remove(filepath)
                    deleted_corrupt_files += 1
                    continue
                    
                full_text = ""
                first_page_text = ""
                for idx, page in enumerate(doc):
                    text = page.get_text()
                    full_text += text
                    if idx == 0:
                        first_page_text = text.strip()
                        
                doc.close()
                
                # Check for empty text (scanned page)
                full_text_strip = full_text.strip()
                if not full_text_strip:
                    scanned_image_pdfs += 1
                    continue
                    
                # Inspect text for embedded server-side error phrases
                full_text_lower = full_text_strip.lower()
                is_garbage_content = False
                matched_phrase = ""
                for phrase in ERROR_PHRASES:
                    if phrase in full_text_lower:
                        is_garbage_content = True
                        matched_phrase = phrase
                        break
                        
                if is_garbage_content:
                    print(f"  Embedded error text matched ('{matched_phrase}'): {file} -> Deleting.")
                    os.remove(filepath)
                    deleted_garbage_files += 1
                    continue
                    
                # C. Deep Content Deduplication (near-duplicates)
                normalized_first_page = "".join(first_page_text.split())
                if len(normalized_first_page) > 100:
                    content_key = normalized_first_page[:1000]
                    if content_key in seen_content_texts:
                        print(f"  Content-near duplicate found: {file} (Original: {os.path.basename(seen_content_texts[content_key])}) -> Deleting.")
                        os.remove(filepath)
                        deleted_content_duplicates += 1
                    else:
                        seen_content_texts[content_key] = filepath
                        
            except Exception as e:
                print(f"  Read error: {file} -> Deleting. Error: {e}")
                try:
                    os.remove(filepath)
                except Exception:
                    pass
                deleted_corrupt_files += 1

    # ----------------------------------------------------
    # Step 4: Clean up empty folders
    # ----------------------------------------------------
    print("\n[Step 4] Cleaning up empty directories...")
    for root, dirs, files in os.walk(directory, topdown=False):
        for d in dirs:
            dirpath = os.path.join(root, d)
            if not os.listdir(dirpath):
                print(f"  Removing empty directory: {dirpath}")
                try:
                    os.rmdir(dirpath)
                    empty_folders_count += 1
                except Exception:
                    pass

    # ----------------------------------------------------
    # Step 5: Report Cleanup Statistics
    # ----------------------------------------------------
    print("\n" + "=" * 60)
    print("CLEANUP PIPELINE REPORT")
    print("=" * 60)
    print(f"Total PDFs Scanned              : {total_scanned}")
    print(f"Garbage Excluded Files Deleted  : {deleted_garbage_files}")
    print(f"Corrupt / Signatures Deleted    : {deleted_corrupt_files}")
    print(f"Identical Binary Duplicates      : {deleted_binary_duplicates}")
    print(f"Content Near-Duplicates Deleted : {deleted_content_duplicates}")
    print(f"Scanned-Only Image PDFs Kept    : {scanned_image_pdfs}")
    print(f"Folder Names Merged/Cleaned     : {cleaned_folders_count}")
    print(f"Empty Folders Removed           : {empty_folders_count}")
    print(f"Pristine PDFs Remaining         : {total_scanned - deleted_garbage_files - deleted_corrupt_files - deleted_binary_duplicates - deleted_content_duplicates}")
    print("=" * 60)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Rule Document Cleanup and Verification Utility")
    parser.add_argument("--dir", type=str, default="documents", help="Target folder containing subdirectories of PDFs.")
    args = parser.parse_args()
    
    target_dir = os.path.abspath(args.dir)
    if not os.path.exists(target_dir):
        print(f"Error: Directory '{target_dir}' does not exist.")
    else:
        clean_documents_pipeline(target_dir)
