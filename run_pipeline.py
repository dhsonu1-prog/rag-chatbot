import os
import argparse
import time

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
    parser = argparse.ArgumentParser(description="Unified RAG pipeline: Scrape, Clean, and Ingest.")
    parser.add_argument("--depth", type=int, default=2, help="Scraper crawling depth limit. Default is 2.")
    parser.add_argument("--skip-scrape", action="store_true", help="Skip crawling and downloading.")
    parser.add_argument("--skip-clean", action="store_true", help="Skip garbage cleanup and deduplication.")
    parser.add_argument("--skip-ingest", action="store_true", help="Skip database ingestion.")
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

if __name__ == "__main__":
    main()
