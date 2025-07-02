import json
import time
import fitz
import concurrent.futures
from urllib.parse import urlparse
import requests
from typhoon_ocr import ocr_document
from datetime import datetime, timezone
from dotenv import load_dotenv
import os
from pathlib import Path

load_dotenv(dotenv_path=Path(__file__).parent / ".env")

def extract_text_from_pdf_ocr_only(pdf_url: str) -> str:
    try:
        response = requests.get(pdf_url)
        response.raise_for_status()

        save_path = Path("downloads")
        save_path.mkdir(exist_ok=True)
        filename = save_path / Path(urlparse(pdf_url).path).name

        with open(filename, "wb") as f:
            f.write(response.content)

        with fitz.open(str(filename)) as doc:
            total_pages = doc.page_count

        print(f"📄 Processing {filename.name} ({total_pages} pages)")

        all_text = []
        for page_index in range(total_pages):
            try:
                print(f"  🔍 OCR page {page_index + 1} of {total_pages}")
                page_text = ocr_document(
                    pdf_or_image_path=str(filename),
                    task_type="default",
                    page_num=page_index
                )
                all_text.append(page_text.strip())
            except Exception as e:
                print(f"❌ OCR error on page {page_index + 1}: {e}")
            time.sleep(1)

        return "\n\n".join(all_text)

    except Exception as e:
        print(f"Error extracting OCR text: {e}")
        return "OCR failed or unavailable"

def fetch_funds(page=1, per_page=10):
    """
    Fetch fund data from Finnomena public API.

    Args:
        page (int): Page number to fetch.
        per_page (int): Number of items per page (if supported).

    Returns:
        dict: Parsed JSON response.
    """
    url = "https://www.finnomena.com/fn3/api/fund/v2/public/filter"
    params = {
        "page": page,
        "per_page": per_page,
    }
    resp = requests.get(url, params=params)
    resp.raise_for_status()
    return resp.json()

# --------- RAG formatting ---------
def format_fund_for_rag(fund: dict) -> dict:
    fund_id = fund["fund_id"]
    text = extract_text_from_pdf_ocr_only(fund.get("fund_fact_sheet", "")) 
    return {
        "id": fund_id,
        "text": text.strip() or "OCR failed or unavailable",
        "metadata": {
   "last_updated": datetime.now(timezone.utc).isoformat(),
    "fund_id": fund.get("fund_id"),
    "short_code": fund.get("short_code"),
    "amc_name": fund.get("amc_name"),
    "nav": fund.get("nav"),
    "nav_date": fund.get("nav_date"),
    "return_1y": fund.get("return_1y"),
    "sharpe_ratio_1y": fund.get("sharpe_ratio_1y"),
    "max_drawdown_1y": fund.get("max_drawdown_1y"),
    "fund_fact_sheet": fund.get("fund_fact_sheet")
}
    }

if __name__ == "__main__":
    # Fetch pages 1 to 10 and aggregate all funds
    all_pages = []
    for page_num in range(2, 20):  # Fetch pages 1 to 10
        try:
            result = fetch_funds(page=page_num)
            all_pages.extend(result["data"]["funds"])
        except Exception as e:
            print(f"Failed to fetch page {page_num}: {e}")
        all_data = all_pages
        
        for fund in all_data:
            output_dir = Path("rag_outputs/ocr_only")
            output_dir.mkdir(exist_ok=True)
            fund_fact_sheet = fund.get("fund_fact_sheet", "")
            pdf_name = Path(urlparse(fund_fact_sheet).path).name if fund_fact_sheet else fund["fund_id"]
            fund_folder = output_dir / pdf_name.replace(".pdf", "")
            if fund_folder.exists():
                print(f"⚠️ Skipping {fund['fund_id']} (already exists)")
                continue

            doc = format_fund_for_rag(fund)
            print(doc["text"])
            fund_folder.mkdir(parents=True, exist_ok=True)
            filename = fund_folder / "content.txt"
            with open(filename, "w", encoding="utf-8") as f:
                f.write(doc["text"])
            meta_filename = fund_folder / "meta.json"
            with open(meta_filename, "w", encoding="utf-8") as mf:
                json.dump(doc["metadata"], mf, ensure_ascii=False, indent=2)