import csv
import requests
from typhoon_ocr import ocr_document
from pathlib import Path
from datetime import datetime, timezone
import json
from urllib.parse import urlparse
import fitz
import time
import concurrent.futures  

def extract_text_from_pdf_ocr_only(file_type: str, source: str) -> str:
    try:
        # Load PDF
        if file_type == "url":
            response = requests.get(source)
            response.raise_for_status()
            save_path = Path("downloads")
            save_path.mkdir(exist_ok=True)
            filename = save_path / Path(urlparse(source).path).name
            with open(filename, "wb") as f:
                f.write(response.content)
        elif file_type == "file":
            filename = Path(source)
            if not filename.exists():
                raise FileNotFoundError(f"Local file not found: {source}")
        else:
            raise ValueError(f"Unknown file type: {file_type}")

        # Count total pages
        with fitz.open(str(filename)) as doc:
            total_pages = doc.page_count

        print(f"📄 Processing {filename.name} ({total_pages} pages)")

        # Run OCR page-by-page
        all_text = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for page_index in range(total_pages):
                try:
                    print(f"  🔍 OCR page {page_index + 1} of {total_pages}")
                    future = executor.submit(
                        ocr_document,
                        pdf_or_image_path=str(filename),
                        task_type="default",
                        page_num=page_index
                    )
                    page_text = future.result(timeout=180)  # ⏱️ Timeout 60s
                    all_text.append(page_text.strip())
                except concurrent.futures.TimeoutError:
                    print(f"⏱️ OCR timeout on page {page_index + 1}, skipping.")
                    continue
                except Exception as e:
                    print(f"❌ OCR error on page {page_index + 1}: {e}")
                    continue
                time.sleep(0.5)

        return "\n\n".join(all_text)

    except Exception as e:
        print(f"❌ OCR error for {source}: {e}")
        return "OCR failed or unavailable"


def format_econ_for_rag(row: dict):
    file_type = row.get("type", "").strip()
    source = row.get("source_file", "").strip() if file_type == "file" else row.get("source_url", "").strip()

    # Run OCR
    text = extract_text_from_pdf_ocr_only(file_type, source)
    doc_id = Path(source).stem if file_type == "file" else Path(urlparse(source).path).stem

    # Create output folder
    output_dir = Path("rag_outputs/econ_ocr_only") / doc_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save content
    with open(output_dir / "content.txt", "w", encoding="utf-8") as f:
        f.write(text.strip())

    # Save metadata
    metadata = {
        "last_updated": datetime.now(timezone.utc).isoformat(),
        "source_name": row.get("source_name", "").strip(),
        "article": row.get("article", "").strip(),
        "source_url": row.get("source_url", "").strip(),
        "source_file": row.get("source_file", "").strip(),
        "source_type": file_type,
    }

    with open(output_dir / "meta.json", "w", encoding="utf-8") as mf:
        json.dump(metadata, mf, ensure_ascii=False, indent=2)

    print(f"✅ Saved: {doc_id}")


def process_economic_csv(csv_path: Path):
    with open(csv_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            format_econ_for_rag(row)


if __name__ == "__main__":
    process_economic_csv(Path("resources/economic.csv"))