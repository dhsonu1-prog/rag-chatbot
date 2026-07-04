import os
import argparse
import time
import subprocess

try:
    from scraper import run_scraper
except ImportError:
    run_scraper = None

try:
    from clean_documents import clean_documents_pipeline
except ImportError:
    clean_documents_pipeline = None

try:
    from ingest import main as run_ingestion
except ImportError:
    run_ingestion = None

def main():
    parser = argparse.ArgumentParser(description="Unified RAG pipeline: Scrape, Clean, Ingest, and Start Server.")
    parser.add_argument("--depth", type=int, default=2, help="Scraper crawling depth limit. Default is 2.")
    parser.add_argument("--skip-scrape", action="store_true", help="Skip crawling and downloading.")
    parser.add_argument("--skip-clean", action="store_true", help="Skip garbage cleanup and deduplication.")
    parser.add_argument("--skip-ingest", action="store_true", help="Skip database ingestion.")
    parser.add_argument("--start-server", action="store_true", help="Automatically start the Streamlit chat server at the end.")
    args = parser.parse_args()

    workspace_dir = os.path.dirname(os.path.abspath(__file__))
    doc_dir = os.path.join(workspace_dir, "documents")

    print("=" * 60)
    print("UNIFIED RAG CHATBOT PIPELINE AUTOMATION")
    print("=" * 60)
    start_time = time.time()

    # ----------------------------------------------------
    # Stage 1: Scrape
    # ----------------------------------------------------
    if args.skip_scrape:
        print("\n[Stage 1] Scraper disabled. Skipping.")
    elif run_scraper is None:
        print("\n[Stage 1] Warning: scraper.py not found. Skipping.")
    else:
        print("\n[Stage 1] Launching DoPT Scraper...")
        try:
            run_scraper(max_depth=args.depth)
        except Exception as e:
            print(f"Scraper error: {e}")

    # ----------------------------------------------------
    # Stage 2: Clean
    # ----------------------------------------------------
    if args.skip_clean:
        print("\n[Stage 2] Document cleanup disabled. Skipping.")
    elif clean_documents_pipeline is None:
        print("\n[Stage 2] Warning: clean_documents.py not found. Skipping.")
    else:
        print("\n[Stage 2] Launching Document Cleanup Pipeline...")
        try:
            clean_documents_pipeline(doc_dir)
        except Exception as e:
            print(f"Cleanup error: {e}")

    # ----------------------------------------------------
    # Stage 3: Ingest
    # ----------------------------------------------------
    if args.skip_ingest:
        print("\n[Stage 3] Database ingestion disabled. Skipping.")
    elif run_ingestion is None:
        print("\n[Stage 3] Warning: ingest.py not found. Skipping.")
    else:
        print("\n[Stage 3] Launching Database Ingestion...")
        try:
            run_ingestion()
        except Exception as e:
            print(f"Ingestion error: {e}")

    duration = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"PIPELINE EXECUTION FINISHED in {duration:.2f} seconds")
    print("=" * 60)

    # ----------------------------------------------------
    # Stage 4: Start Server
    # ----------------------------------------------------
    if args.start_server:
        print("\n[Stage 4] Starting Streamlit Chat Server...")
        try:
            venv_bin = os.path.join(workspace_dir, "venv", "bin")
            if not os.path.exists(venv_bin):
                venv_bin = os.path.join(workspace_dir, ".venv", "bin")
                
            if os.path.exists(venv_bin):
                streamlit_cmd = os.path.join(venv_bin, "streamlit")
            else:
                streamlit_cmd = "streamlit"
                
            app_path = os.path.join(workspace_dir, "app.py")
            print(f"Executing: {streamlit_cmd} run {app_path} --server.address=0.0.0.0")
            
            # Launch streamlit in the background and let it persist
            process = subprocess.Popen([streamlit_cmd, "run", app_path, "--server.address=0.0.0.0"])
            print(f"Streamlit server started successfully in the background (PID: {process.pid})!")
        except Exception as e:
            print(f"Error starting Streamlit server: {e}")

if __name__ == "__main__":
    main()
