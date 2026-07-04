import os
import shutil
import base64
import requests
import fitz  # PyMuPDF
from concurrent.futures import ThreadPoolExecutor, as_completed

DOC_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "documents")
LLM_API_URL = "http://172.16.172.4:3002/v1/chat/completions"
OCR_API_URL = "http://172.16.172.4:3001/v1/chat/completions"

VALID_CATEGORIES = [
    "1_Central_Procurement_Commission/Procurement_Guidelines_&_GFR",
    "1_Central_Procurement_Commission/Tenders_&_Bidding",
    "1_Central_Procurement_Commission/GeM_&_Contracts",
    "2_Finance/Demands_for_Grants",
    "2_Finance/Accounts_&_Audits",
    "2_Finance/General_Expenditure",
    "2_Finance/Pay_&_Increments",
    "3_Personnel/Recruitment_&_Selection",
    "3_Personnel/Vigilance_Conduct_&_Discipline",
    "3_Personnel/Cadre_&_Promotion",
    "3_Personnel/Leave_LTC_&_Allowances",
    "3_Personnel/Retirement_&_Pension",
    "3_Personnel/Deputation_&_Transfer",
    "3_Personnel/Acts_&_Central_Rules",
    "3_Personnel/Training_&_Development",
    "3_Personnel/Forms_&_Annexures",
    "3_Personnel/General_Policies_&_Circulars"
]

SYSTEM_PROMPT = """You are an expert government document classifier. You will be given the first page of a government circular, office memorandum (OM), or handbook.
Your job is to read the text and classify the document into exactly ONE of the following valid folder paths.

Valid Folder Paths:
- 1_Central_Procurement_Commission/Procurement_Guidelines_&_GFR (For GFR rules, procurement policies, general buying directives)
- 1_Central_Procurement_Commission/Tenders_&_Bidding (For tender notices, bidding instructions, bidder qualifications, pre-bid queries)
- 1_Central_Procurement_Commission/GeM_&_Contracts (For GeM portal circulars, contract agreements, performance security, EMD)
- 2_Finance/Demands_for_Grants (For Detailed Demands for Grants documents, parliamentary financial committees)
- 2_Finance/Accounts_&_Audits (For Accounts at a Glance, controller of accounts structure, audit reconciliation, ledger balances)
- 2_Finance/General_Expenditure (For budget allocation sheets, financial sanctions, non-plan/plan expenditures)
- 2_Finance/Pay_&_Increments (For pay fixation, stepping up of pay, grade pay, pay matrix levels, dearness allowance, increments)
- 3_Personnel/Recruitment_&_Selection (For direct recruitment rules, vacancies, UPSC circulars, appointments, probation)
- 3_Personnel/Vigilance_Conduct_&_Discipline (For conduct rules, suspension, disciplinary proceedings, CVC inquiries, POSH/harassment)
- 3_Personnel/Cadre_&_Promotion (For APAR/ACR sparrow guidelines, seniority rosters, promotions, cadre reviews, MACP)
- 3_Personnel/Leave_LTC_&_Allowances (For LTC travel concession, maternity/special maternity leaves, casual/earned leave guidelines)
- 3_Personnel/Retirement_&_Pension (For superannuation, retirement gratuity, NPS national pension system rules, VRS schemes)
- 3_Personnel/Deputation_&_Transfer (For rotational transfer policy, inter-cadre deputations, transfer rules, lien retention)
- 3_Personnel/Acts_&_Central_Rules (For parliamentary acts like RTI Act, legislative rules, general constitution guidelines)
- 3_Personnel/Training_&_Development (For LBSNAA training schedules, mid-career courses, induction programs, fellowships)
- 3_Personnel/Forms_&_Annexures (For blank formats, application proformas, sports calendars, standard annexure templates)
- 3_Personnel/General_Policies_&_Circulars (If the document is a general circular or policy that doesn't fit any of the above)

CRITICAL INSTRUCTION:
Response format must be exactly one of the valid folder path strings listed above, with no extra spaces, quotes, punctuation, prefixes, or explanation.
Example Output:
3_Personnel/Recruitment_&_Selection"""

def ocr_page_one(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        if len(doc) == 0:
            return ""
        page = doc[0]
        pix = page.get_pixmap(dpi=150)
        img_data = pix.tobytes("jpg")
        base64_image = base64.b64encode(img_data).decode('utf-8')
        doc.close()
        
        payload = {
            "model": "gemma-4-vision",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Transcribe the printed text from this image accurately. Do not add comments."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }
            ],
            "temperature": 0.0
        }
        res = requests.post(OCR_API_URL, json=payload, timeout=60)
        if res.status_code == 200:
            return res.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"  [OCR Error] for {os.path.basename(pdf_path)}: {e}")
    return ""

def get_page_one_text(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        if len(doc) == 0:
            return ""
        text = doc[0].get_text().strip()
        doc.close()
        
        if len(text) < 50:
            text = ocr_page_one(pdf_path)
            
        return text
    except Exception as e:
        print(f"  Error reading text for {pdf_path}: {e}")
        return ""

def classify_document(filename, text_content):
    if not text_content:
        return "3_Personnel/General_Policies_&_Circulars"
        
    truncated_text = text_content[:1500]
    prompt = f"Filename: {filename}\n\nDocument Text (First Page):\n{truncated_text}"
    
    try:
        payload = {
            "model": "/mnt/ai_storage/models/sarvam-105b",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.0,
            "max_tokens": 1500
        }
        res = requests.post(LLM_API_URL, json=payload, timeout=60)
        if res.status_code == 200:
            classification = res.json()["choices"][0]["message"]["content"].strip()
            classification = classification.replace("`", "").replace("'", "").replace('"', "").strip()
            lines = [l.strip() for l in classification.split('\n') if l.strip()]
            classification_line = lines[0] if lines else ""
            
            if classification_line in VALID_CATEGORIES:
                return classification_line
                
            for cat in VALID_CATEGORIES:
                if cat.lower() == classification_line.lower():
                    return cat
                if classification_line.lower() in cat.lower() and len(classification_line) > 10:
                    return cat
            
            # Map common category confusion
            if "general_expenditure" in classification_line.lower():
                return "2_Finance/General_Expenditure"
            if "general_policies" in classification_line.lower():
                return "3_Personnel/General_Policies_&_Circulars"
            if "pay" in classification_line.lower() and "increment" in classification_line.lower():
                return "2_Finance/Pay_&_Increments"
    except Exception as e:
        print(f"  [LLM Error] for {filename}: {e}")
        
    return "3_Personnel/General_Policies_&_Circulars"

def process_file(filepath):
    filename = os.path.basename(filepath)
    text = get_page_one_text(filepath)
    category = classify_document(filename, text)
    return filepath, category

def run_llm_segregation():
    print("=" * 60)
    print("STARTING LLM-BASED SEGREGATION")
    print("=" * 60)
    
    pdf_files = []
    for root, dirs, files in os.walk(DOC_ROOT):
        for file in files:
            if file.lower().endswith('.pdf'):
                pdf_files.append(os.path.join(root, file))
                
    total_files = len(pdf_files)
    print(f"Found {total_files} files to classify.")
    
    results = []
    with ThreadPoolExecutor(max_workers=24) as executor:
        futures = {executor.submit(process_file, fp): fp for fp in pdf_files}
        for idx, future in enumerate(as_completed(futures)):
            filepath, category = future.result()
            results.append((filepath, category))
            if (idx + 1) % 10 == 0 or (idx + 1) == total_files:
                print(f"Progress: [{idx+1}/{total_files}] classified...")
                
    moved_count = 0
    for src, cat in results:
        dst_dir = os.path.join(DOC_ROOT, cat)
        dst_file = os.path.join(dst_dir, os.path.basename(src))
        
        if os.path.dirname(src) == dst_dir:
            continue
            
        os.makedirs(dst_dir, exist_ok=True)
        if os.path.exists(dst_file):
            base, ext = os.path.splitext(os.path.basename(src))
            dst_file = os.path.join(dst_dir, f"{base}_duplicate{ext}")
            
        print(f"Moving: {os.path.basename(src)} -> {cat}/")
        try:
            shutil.move(src, dst_file)
            moved_count += 1
        except Exception as e:
            print(f"  Error moving file: {e}")
            
    # Clean up empty subdirectories
    preserved = [
        "1_Central_Procurement_Commission",
        "1_Central_Procurement_Commission/Procurement_Guidelines_&_GFR",
        "1_Central_Procurement_Commission/Tenders_&_Bidding",
        "1_Central_Procurement_Commission/GeM_&_Contracts",
        "2_Finance",
        "2_Finance/Demands_for_Grants",
        "2_Finance/Accounts_&_Audits",
        "2_Finance/General_Expenditure",
        "2_Finance/Pay_&_Increments",
        "3_Personnel"
    ]
    for root, dirs, files in os.walk(DOC_ROOT, topdown=False):
        for d in dirs:
            dirpath = os.path.join(root, d)
            rel = os.path.relpath(dirpath, DOC_ROOT)
            if rel in preserved:
                continue
            if not os.listdir(dirpath):
                try:
                    os.rmdir(dirpath)
                except Exception:
                    pass
                    
    print(f"\nLLM SEGREGATION FINISHED. Re-categorized: {moved_count} files.")
    print("=" * 60)

if __name__ == "__main__":
    run_llm_segregation()
