import os
import re
import time
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, unquote

def sanitize_filename(name):
    # Keep only alphanumeric, spaces, dashes, and underscores
    name = re.sub(r'[^\w\s\-\.]', '', name)
    # Replace multiple spaces with a single space
    name = re.sub(r'\s+', ' ', name).strip()
    return name

visited_urls = set()

def extract_pdf_links(pdf_path):
    links = []
    try:
        import fitz
        doc = fitz.open(pdf_path)
        for page in doc:
            for link in page.get_links():
                if "uri" in link:
                    uri = link["uri"]
                    # Filter for relevant government domains and PDF links
                    is_gov = any(domain in uri.lower() for domain in ['dopt.gov.in', 'doptcirculars.nic.in', 'legislative.gov.in', 'cvc.gov.in'])
                    is_valid = uri.lower().endswith('.pdf') or is_gov
                    if is_valid:
                        links.append(uri)
    except Exception:
        pass
    return list(set(links))

def is_valid_pdf_file(filepath):
    # 1. Size check: minimum 5 KB for a valid PDF
    if os.path.getsize(filepath) < 5120:
        return False
    # 2. PDF Header check: first 4 bytes must be %PDF
    try:
        with open(filepath, 'rb') as f:
            header = f.read(4)
            return header == b'%PDF'
    except Exception:
        return False

def download_file(url, dest_folder, custom_name=None):
    global visited_urls
    
    # Keyword exclusion list to filter out garbage links
    GARBAGE_KEYWORDS = [
        'accessibility', 'certificate', 'stqc', 'sitemap', 'feedback', 
        'contact', 'about-us', 'menu', 'logo', 'banner', 'holiday', 
        'website-policy', 'screen-reader', 'help', 'twitter', 'facebook'
    ]
    
    url_lower = url.lower()
    custom_name_lower = (custom_name or "").lower()
    
    if any(kw in url_lower or kw in custom_name_lower for kw in GARBAGE_KEYWORDS):
        print(f"  [Garbage Shield] Skipping excluded garbage link: {custom_name or url}")
        return False

    os.makedirs(dest_folder, exist_ok=True)
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, Guide/Admin) Chrome/91.0.4472.124'
    }
    try:
        response = requests.get(url, headers=headers, stream=True, timeout=30)
        if response.status_code == 200:
            filename = ""
            if "Content-Disposition" in response.headers:
                cd = response.headers["Content-Disposition"]
                filename_match = re.findall(r'filename="?([^";]+)"?', cd)
                if filename_match:
                    filename = filename_match[0]
            if not filename:
                filename = url.split('/')[-1]
                
            filename = unquote(filename)
            if not filename.lower().endswith('.pdf'):
                filename += '.pdf'
                
            if custom_name:
                sanitized = sanitize_filename(custom_name)
                if sanitized and len(sanitized) > 5:
                    filename = f"{sanitized}.pdf"
                    
            dest_path = os.path.join(dest_folder, filename)
            
            # Skip if already exists
            if os.path.exists(dest_path):
                print(f"  File already exists: {filename}")
                return False
                
            with open(dest_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    
            # Post-download validations (Header and size)
            if not is_valid_pdf_file(dest_path):
                print(f"  [Garbage Shield] Deleting invalid/garbage file: {filename}")
                try:
                    os.remove(dest_path)
                except Exception:
                    pass
                return False
                
            print(f"  Downloaded: {filename}")
            
            # Recursively download links found inside the PDF document
            if filename.lower().endswith('.pdf'):
                inner_links = extract_pdf_links(dest_path)
                if inner_links:
                    print(f"    [PDF Link Harvester] Found {len(inner_links)} sublinks inside PDF: {filename}")
                    for inner_url in inner_links:
                        absolute_inner_url = urljoin(url, inner_url)
                        if absolute_inner_url not in visited_urls:
                            visited_urls.add(absolute_inner_url)
                            download_file(absolute_inner_url, dest_folder)
            return True
        else:
            print(f"  Failed to download {url}: HTTP {response.status_code}")
    except Exception as e:
        print(f"  Error downloading {url}: {e}")
    return False

def get_category_folder(seed_url, dest_root):
    if "gazette-notifications" in seed_url:
        return os.path.join(dest_root, "Gazette Notifications")
    elif "recruitment-rules" in seed_url:
        return os.path.join(dest_root, "Recruitment Rules")
    elif "disciplinary-authorities" in seed_url:
        return os.path.join(dest_root, "Disciplinary Authorities")
    elif "download/acts" in seed_url:
        return os.path.join(dest_root, "Acts")
    return os.path.join(dest_root, "General")

def scrape_url_recursive(url, dest_folder, current_depth=1, max_depth=2):
    global visited_urls
    if url in visited_urls:
        return
    visited_urls.add(url)
    
    print(f"\n[Depth {current_depth}] Scraping page: {url}")
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, Guide/Admin) Chrome/91.0.4472.124'
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=20)
        if response.status_code != 200:
            print(f"  Failed to fetch page (HTTP {response.status_code})")
            return
            
        soup = BeautifulSoup(response.content, 'html.parser')
        sublinks_to_visit = []
        
        for link in soup.find_all('a', href=True):
            href = link['href']
            is_pdf = href.lower().endswith('.pdf') or 'doptcirculars.nic.in' in href or 'documents.doptcirculars.nic.in' in href
            absolute_url = urljoin(url, href)
            
            if is_pdf:
                link_text = link.get_text().strip()
                description = link_text
                if len(description) < 5 and link.get('title'):
                    description = link['title'].strip()
                if len(description) > 120:
                    description = description[:117] + "..."
                    
                print(f"  Found PDF link: {description} ({absolute_url})")
                download_file(absolute_url, dest_folder, custom_name=description)
            else:
                if current_depth < max_depth:
                    is_internal = 'dopt.gov.in' in absolute_url or absolute_url.startswith('/')
                    is_valid_page = not (href.startswith('#') or href.startswith('javascript:') or href.startswith('mailto:'))
                    if is_internal and is_valid_page and absolute_url not in visited_urls:
                        sublinks_to_visit.append(absolute_url)
                        
        for sublink in sublinks_to_visit:
            scrape_url_recursive(sublink, dest_folder, current_depth + 1, max_depth)
            
    except Exception as e:
        print(f"  Error scraping {url}: {e}")

def run_scraper(max_depth=2):
    global visited_urls
    seed_urls = [
        "https://dopt.gov.in/notifications/gazette-notifications",
        "https://dopt.gov.in/reports/hand-book/hand-book-recruitment-rules",
        "https://dopt.gov.in/reports/hand-book/hand-book-inquiry-officers-and-disciplinary-authorities-2013",
        "https://dopt.gov.in/download/acts"
    ]
    
    dest_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "documents")
    visited_urls.clear()
    
    print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting recursive DoPT PDF scraper (Max Depth: {max_depth})...")
    
    for seed_url in seed_urls:
        category_folder = get_category_folder(seed_url, dest_root)
        print(f"\n=== Scraping Category: {os.path.basename(category_folder)} ===")
        scrape_url_recursive(seed_url, category_folder, current_depth=1, max_depth=max_depth)
        
    print("\nScraper run finished.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Recursive DoPT PDF Scraper")
    parser.add_argument("--depth", type=int, default=2, help="Crawling depth limit. Default is 2 (crawls starting URLs and their direct sublinks).")
    parser.add_argument("--interval", type=int, default=0, help="Interval in seconds to run periodically. Default is 0 (run once).")
    args = parser.parse_args()
    
    if args.interval > 0:
        print(f"Running scraper in periodic mode. Interval: {args.interval} seconds.")
        while True:
            run_scraper(max_depth=args.depth)
            print(f"Sleeping for {args.interval} seconds...")
            time.sleep(args.interval)
    else:
        run_scraper(max_depth=args.depth)
