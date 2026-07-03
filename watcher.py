import os
import time
import subprocess
import sys

# Get workspace directory
workspace_dir = os.path.dirname(os.path.abspath(__file__))
monitor_dir = os.path.join(workspace_dir, "documents")

def find_pdfs(root_dir):
    """Find all PDF files recursively, ignoring virtual environments, git, or db folders."""
    exclude_dirs = {'.git', 'venv', '.venv', 'chroma_db', 'node_modules', '__pycache__'}
    pdf_files = []
    
    for root, dirs, files in os.walk(root_dir):
        # Prune excluded directories in-place
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        for file in files:
            if file.lower().endswith('.pdf'):
                pdf_files.append(os.path.join(root, file))
                
    return pdf_files

def get_pdf_state(pdf_paths):
    """Retrieve the current mtime and size state of all PDF files."""
    state = {}
    for path in pdf_paths:
        try:
            stat = os.stat(path)
            state[path] = (stat.st_mtime, stat.st_size)
        except Exception:
            # File might be temporarily locked or deleted
            pass
    return state

def run_ingestion():
    """Trigger the ingestion script as a subprocess."""
    print("\n[Watcher] 🔄 Re-indexing database due to directory changes...")
    try:
        # Run using virtual environment python executable
        python_exe = os.path.join(workspace_dir, "venv", "bin", "python")
        if not os.path.exists(python_exe):
            python_exe = sys.executable  # fallback to active interpreter
            
        ingest_script = os.path.join(workspace_dir, "ingest.py")
        
        result = subprocess.run([python_exe, ingest_script], check=True)
        print("[Watcher] ✅ Re-indexing completed successfully.\n")
    except subprocess.CalledProcessError as e:
        print(f"[Watcher] ❌ Re-indexing failed with exit code: {e.returncode}\n")
    except Exception as e:
        print(f"[Watcher] ❌ Error triggering ingestion: {e}\n")

def main():
    print("=" * 60)
    print("        📚 PDF Document Watcher Daemon initialized")
    print(f"        Monitoring: {monitor_dir}")
    print("        Sleep interval: 5 seconds")
    print("=" * 60)
    
    # Initialize initial state
    pdf_paths = find_pdfs(monitor_dir)
    last_state = get_pdf_state(pdf_paths)
    print(f"[Watcher] Monitoring {len(pdf_paths)} initial PDF documents...")
    
    try:
        while True:
            time.sleep(5)
            
            # Scan current state
            current_paths = find_pdfs(monitor_dir)
            current_state = get_pdf_state(current_paths)
            
            # Detect changes
            changed = False
            
            # 1. Check for additions or modifications
            for path, (mtime, size) in current_state.items():
                if path not in last_state:
                    print(f"[Watcher] ➕ Added: {os.path.relpath(path, monitor_dir)}")
                    changed = True
                elif last_state[path] != (mtime, size):
                    print(f"[Watcher] 📝 Modified: {os.path.relpath(path, monitor_dir)}")
                    changed = True
                    
            # 2. Check for deletions
            for path in last_state:
                if path not in current_state:
                    print(f"[Watcher] ❌ Removed: {os.path.relpath(path, monitor_dir)}")
                    changed = True
                    
            if changed:
                run_ingestion()
                # Update base state to current state
                last_state = current_state
            elif set(current_paths) != set(last_state.keys()):
                # Catch case where file list changed but no metadata modified
                last_state = current_state
                
    except KeyboardInterrupt:
        print("\n[Watcher] Monitoring stopped by user. Exiting.")

if __name__ == "__main__":
    main()
