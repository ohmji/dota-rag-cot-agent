import requests
from typhoon_ocr import ocr_document
from langchain.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import Optional
from dotenv import load_dotenv
import os
from pathlib import Path

load_dotenv(dotenv_path=Path(__file__).parent / ".env")

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
        "sort": "SR_10Y,DESC"
    }
    resp = requests.get(url, params=params)
    resp.raise_for_status()
    return resp.json()

def extract_policy_from_pdf(pdf_url: str) -> dict:
    from pathlib import Path

    class FundPolicy(BaseModel):
        policy_summary: str = Field(..., description="Summary of the fund's investment policy")
        risk_level: Optional[int] = Field(None, description="Numeric risk level, e.g. 6")
        risk_description: Optional[str] = Field(None, description="Textual risk label, e.g. 'เสี่ยงสูง'")
        underlying_fund: Optional[str] = Field(None, description="Name of the underlying fund if any")
        investment_theme: Optional[str] = Field(None, description="Investment theme such as AI, ESG, etc.")
        focus_country: Optional[str] = Field(None, description="Country or region the fund invests in")
        fund_type: Optional[str] = Field(None, description="Fund type such as RMF, SSF, LTF, or General")
        asset_allocation: Optional[str] = Field(None, description="Breakdown of asset allocation, e.g. 'Equity 60%, Bonds 40%'")

    try:
        response = requests.get(pdf_url)
        response.raise_for_status()

        save_path = Path("downloads")
        save_path.mkdir(exist_ok=True)
        filename = save_path / Path(pdf_url).name

        with open(filename, "wb") as f:
            f.write(response.content)


        print (f" typhoon_ocr: Processing {filename} for policy extraction...")
        markdown = ocr_document(
            pdf_or_image_path=str(filename),
            task_type="default",
            page_num=2
        )

        if not markdown or len(markdown.strip()) < 100:
            print(f"⚠️ Skipping {filename} due to empty or invalid OCR output.")
            return FundPolicy(policy_summary="OCR failed or empty content", risk_level=None, risk_description=None)

        print("llm: Initializing LLM for policy extraction...")
        parser = PydanticOutputParser(pydantic_object=FundPolicy)
        llm = ChatOpenAI(temperature=0, model="gpt-4o", openai_api_key=os.getenv("OPENAI_API_KEY"))

        try:
            prompt = (
                "You are a financial document parser that returns only valid JSON.\n"
                "From the input text, extract the following fields:\n\n"
                "1. policy_summary: A concise summary (in English) of the fund's investment policy, including objectives, asset classes, regions, and any restrictions.\n"
                "2. risk_description: Short English phrase describing the risk level (e.g. 'High risk').\n"
                "3. risk_level: Numeric risk level (0 to 8). If unreadable, infer from description or leave null.\n"
                "4. underlying_fund: Name of the underlying fund, if any.\n"
                "5. investment_theme: Investment theme such as AI, robotics, ESG, Next Gen, etc.\n"
                "6. focus_country: Primary country or region the fund invests in.\n"
                "7. fund_type: RMF, SSF, LTF, or General.\n"
                "8. asset_allocation: Breakdown of asset allocation, e.g. 'Equity 60%, Bonds 40%'.\n"
                "\n"
                "❗Exclude disclaimers, marketing, or unrelated text.\n"
                "Respond with raw JSON in this format:\n"
                '{\n'
                '  "policy_summary": "...",\n'
                '  "risk_description": "...",\n'
                '  "risk_level": 4,\n'
                '  "underlying_fund": "...",\n'
                '  "investment_theme": "...",\n'
                '  "focus_country": "...",\n'
                '  "fund_type": "...",\n'
                '  "asset_allocation": "..." \n'
                '}\n\n'
                f"--- DOCUMENT TEXT ---\n{markdown}\n"
            )
            raw = llm.invoke(prompt).content.strip()
            result = parser.invoke(raw)
            if result and result.policy_summary:
                return result
        except Exception as e:
            print(f"Error parsing document: {e}")

        return FundPolicy(policy_summary="No clear investment policy found", risk_level=None, risk_description=None)
    except Exception as e:
        return FundPolicy(policy_summary=f"PDF parsing error: {str(e)}", risk_level=None, risk_description=None)

def translate_fee_description(description: str) -> str:
    fee_translation = {
        "ค่าใช้จ่ายอื่นๆ": "Other expenses",
        "ค่าธรรมเนียมการขายหน่วยลงทุน (Front-end Fee)": "Front-end fee (unit purchase fee)",
        "ค่าธรรมเนียมการจัดการ": "Management fee",
        "ค่าธรรมเนียมการรับซื้อคืนหน่วยลงทุน (Back-end Fee)": "Back-end fee (unit redemption fee)",
        "ค่าธรรมเนียมการสับเปลี่ยนหน่วยลงทุนเข้า (SWITCHING IN)": "Switching-in fee",
        "ค่าธรรมเนียมการโอนหน่วยลงทุน": "Unit transfer fee",
        "ค่าธรรมเนียมนายทะเบียนหน่วย": "Registrar fee",
        "ค่าธรรมเนียมผู้ดูแลผลประโยชน์": "Trustee fee",
        "ค่าธรรมเนียมและค่าใช้จ่ายรวมทั้งหมด": "Total fees and expenses",
    }
    return fee_translation.get(description, description)

# --------- RAG formatting ---------
def format_fund_for_rag(fund: dict) -> dict:
    fund_id = fund["fund_id"]
    investment_data = extract_policy_from_pdf(fund.get('fund_fact_sheet', '')) if fund.get('fund_fact_sheet') else None
    if investment_data:
        investment_policy = investment_data.policy_summary
        risk_level = investment_data.risk_level
        risk_description = investment_data.risk_description
        asset_allocation = investment_data.asset_allocation
    else:
        investment_policy = 'N/A'
        risk_level = None
        risk_description = None
        asset_allocation = None

    text = f"""
Fund: {fund['short_code']} ({fund['amc_name']})
Latest NAV: {fund['nav']} THB as of {fund['nav_date']}

Returns:
- 1 month: {fund.get('return_1m', 'N/A')}%
- 3 months: {fund.get('return_3m', 'N/A')}%
- 6 months: {fund.get('return_6m', 'N/A')}%
- YTD: {fund.get('return_ytd', 'N/A')}%
- 1 year: {fund.get('return_1y', 'N/A')}%
- Sharpe Ratio 1Y: {fund.get('sharpe_ratio_1y', 'N/A')}
- Max Drawdown 1Y: {fund.get('max_drawdown_1y', 'N/A')}

Minimum Investment:
- Initial: {fund.get('minimum_initial', 'N/A')} THB
- Subsequent: {fund.get('minimum_subsequent', 'N/A')} THB

Dividends:
- Total amount: {fund.get('dividend_sum', 'N/A')} THB
- Number of times: {fund.get('dividend_count', 'N/A')} times
- Last XD date: {fund.get('last_xd_date', 'N/A')}
- Last payment date: {fund.get('last_pay_date', 'N/A')}

Investment Policy (from PDF):
{investment_policy}

Risk Level: {risk_level} - {risk_description}
Asset Allocation: {asset_allocation}

Fact Sheet: {fund.get('fund_fact_sheet', 'N/A')}

Key Fees:
""" + "\n".join([
        f"- {translate_fee_description(fee.get('description'))}: {fee.get('rate')} {fee.get('unit', '')}" for fee in fund.get("fees", [])
    ])

    return {
        "id": fund_id,
        "text": text.strip(),
        "metadata": {
            "short_code": fund['short_code'],
            "amc_name": fund['amc_name'],
            "nav_date": fund['nav_date'],
            "fund_fact_sheet": fund.get("fund_fact_sheet", None),
            "investment_policy": investment_policy,
            "risk_level": risk_level,
            "risk_description": risk_description,
            "underlying_fund": investment_data.underlying_fund if investment_data else None,
            "investment_theme": investment_data.investment_theme if investment_data else None,
            "focus_country": investment_data.focus_country if investment_data else None,
            "fund_type": investment_data.fund_type if investment_data else None,
            "asset_allocation": asset_allocation,
        }
    }

if __name__ == "__main__":
    # Fetch pages 1 to 10 and aggregate all funds
    all_pages = []
    for page_num in range(1, 11):  # Fetch pages 1 to 10
        try:
            result = fetch_funds(page=page_num)
            all_pages.extend(result["data"]["funds"])
        except Exception as e:
            print(f"Failed to fetch page {page_num}: {e}")
        all_data = all_pages
        
        rag_docs = [format_fund_for_rag(f) for f in all_data]
        for doc in rag_docs:  # print only first 2 for preview
            print(doc["text"])
            output_dir = Path("rag_outputs")
            output_dir.mkdir(exist_ok=True)
            from urllib.parse import urlparse
            fund_fact_sheet = doc["metadata"].get("fund_fact_sheet", "")
            pdf_name = Path(urlparse(fund_fact_sheet).path).name if fund_fact_sheet else doc['id']
            fund_folder = output_dir / pdf_name.replace(".pdf", "")
            fund_folder.mkdir(parents=True, exist_ok=True)
            filename = fund_folder / "content.txt"
            with open(filename, "w", encoding="utf-8") as f:
                f.write(doc["text"])
            import json
            meta_filename = fund_folder / "meta.json"
            with open(meta_filename, "w", encoding="utf-8") as mf:
                json.dump(doc["metadata"], mf, ensure_ascii=False, indent=2)