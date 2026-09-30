"""P0 Probe Script for Terminal V2 Phase 4 Non-Crypto Data Providers.

Tests live connectivity, HTTP response headers, schemas, units, and capturing
real payload fixtures for:
1. Cboe Commodity Volatility Indices (GVZ, VXSLV, OVX)
2. US Treasury Completed Auctions History (by type and term)
3. SEC EDGAR Company Facts (XBRL) & Submissions (Form 4 XML)
4. Google News RSS Search (per-ticker discovery)
"""
import json
import os
from pathlib import Path
import re
import sys
import time
import xml.etree.ElementTree as ET
import requests

FIXTURES_DIR = Path("tests/fixtures/terminal_v2")
FIXTURES_DIR.mkdir(parents=True, exist_ok=True)

SEC_USER_AGENT = "TrinityWealthEngine/2.0 (research-terminal@trinity-wealth.internal)"
BROWSER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

results = {}

# ---------------------------------------------------------
# 1. Cboe Commodity Volatility Indices
# ---------------------------------------------------------
print("--> Probing Cboe Volatility...")
cboe_indices = ["GVZ", "VXSLV", "OVX"]
for idx in cboe_indices:
    url = f"https://cdn.cboe.com/api/global/us_indices/daily_prices/{idx}_History.csv"
    try:
        r = requests.get(url, headers=BROWSER_HEADERS, timeout=12)
        print(f"Cboe {idx}: {r.status_code}, len={len(r.text)}")
        if r.status_code == 200:
            lines = r.text.strip().split("\n")
            header = lines[0].strip()
            sample_tail = lines[-5:]
            print(f"  Header: {header}")
            print(f"  Total rows: {len(lines) - 1}")
            print(f"  Last line: {lines[-1].strip()}")
            results[f"cboe_{idx}"] = {"status": r.status_code, "rows": len(lines) - 1, "header": header}
            # Save trimmed fixture (header + first 5 + last 260 rows for 52W percentile testing)
            trimmed_lines = [header] + lines[1:6] + lines[-260:]
            fixture_path = FIXTURES_DIR / f"cboe_{idx.lower()}_fixture.csv"
            fixture_path.write_text("\n".join(trimmed_lines), encoding="utf-8")
            print(f"  Saved fixture: {fixture_path}")
    except Exception as e:
        print(f"Cboe {idx} ERROR: {e}")
        results[f"cboe_{idx}"] = {"error": str(e)}

# ---------------------------------------------------------
# 2. US Treasury Completed Auctions History
# ---------------------------------------------------------
print("\n--> Probing US Treasury Fiscal Data Auctions...")
# Test 10-Year Note and 3-Month Bill
treasury_tests = [
    {"type": "Note", "term": "10-Year"},
    {"type": "Bill", "term": "3-Month"},
]
for t in treasury_tests:
    sec_type = t["type"]
    sec_term = t["term"]
    url = (
        "https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v1/accounting/od/auctions_query?"
        f"filter=bid_to_cover_ratio:gt:0,security_type:eq:{sec_type},security_term:eq:{sec_term}&sort=-auction_date&page[size]=15"
    )
    try:
        r = requests.get(url, headers=BROWSER_HEADERS, timeout=12)
        print(f"Treasury {sec_type} {sec_term}: {r.status_code}")
        if r.status_code == 200:
            data = r.json()
            rows = data.get("data", [])
            print(f"  Rows returned: {len(rows)}")
            if rows:
                first = rows[0]
                print(f"  Latest auction: date={first.get('auction_date')}, btc={first.get('bid_to_cover_ratio')}, high_yield={first.get('high_yield')}, high_discnt={first.get('high_discnt_rate')}")
            fixture_name = f"treasury_auctions_{sec_type.lower()}_{sec_term.lower().replace('-', '_')}_fixture.json"
            (FIXTURES_DIR / fixture_name).write_text(json.dumps(data, indent=2), encoding="utf-8")
            results[f"treasury_{sec_type}_{sec_term}"] = {"status": 200, "count": len(rows)}
    except Exception as e:
        print(f"Treasury ERROR: {e}")
        results[f"treasury_{sec_type}_{sec_term}"] = {"error": str(e)}

# ---------------------------------------------------------
# 3. SEC EDGAR: Company Facts & Form 4 XML
# ---------------------------------------------------------
print("\n--> Probing SEC EDGAR (NVDA CIK 0001045810)...")
sec_headers = {
    "User-Agent": SEC_USER_AGENT,
    "Accept": "application/json, text/plain, */*",
    "Accept-Encoding": "gzip, deflate",
}

# 3.1 Company Facts
facts_url = "https://data.sec.gov/api/xbrl/companyfacts/CIK0001045810.json"
try:
    time.sleep(0.2)  # Respect SEC rate limits
    rf = requests.get(facts_url, headers=sec_headers, timeout=15)
    print(f"SEC companyfacts: {rf.status_code}, len={len(rf.text)}")
    if rf.status_code == 200:
        facts_data = rf.json()
        us_gaap = facts_data.get("facts", {}).get("us-gaap", {})
        print(f"  Entity: {facts_data.get('entityName')}")
        print(f"  Total us-gaap concepts: {len(us_gaap)}")
        # Check key tags: Revenues, OperatingIncomeLoss, NetIncomeLoss, LongTermDebtNoncurrent, PaymentsForRepurchaseOfCommonStock
        for tag in ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "OperatingIncomeLoss", "NetIncomeLoss"]:
            if tag in us_gaap:
                units = us_gaap[tag].get("units", {})
                first_unit = next(iter(units.values()), [])
                print(f"  Found concept: {tag} ({len(first_unit)} data points)")
        # Save trimmed fixture with target concepts to keep fixture bounded
        target_concepts = [
            "Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax",
            "NetIncomeLoss", "OperatingIncomeLoss", "NetCashProvidedByUsedInOperatingActivities",
            "PaymentsToAcquirePropertyPlantAndEquipment", "LongTermDebtNoncurrent", "LongTermDebt",
            "ResearchAndDevelopmentExpense", "PaymentsForRepurchaseOfCommonStock",
            "CommonStockSharesOutstanding"
        ]
        trimmed_facts = {
            "cik": facts_data.get("cik"),
            "entityName": facts_data.get("entityName"),
            "facts": {
                "us-gaap": {k: v for k, v in us_gaap.items() if k in target_concepts},
                "dei": facts_data.get("facts", {}).get("dei", {})
            }
        }
        (FIXTURES_DIR / "sec_companyfacts_nvda_fixture.json").write_text(json.dumps(trimmed_facts, indent=2), encoding="utf-8")
        print(f"  Saved trimmed facts fixture: {len(trimmed_facts['facts']['us-gaap'])} concepts")
        results["sec_facts"] = {"status": 200, "concepts": len(trimmed_facts['facts']['us-gaap'])}
except Exception as e:
    print(f"SEC facts ERROR: {e}")
    results["sec_facts"] = {"error": str(e)}

# 3.2 Submissions and Form 4 XML
subm_url = "https://data.sec.gov/submissions/CIK0001045810.json"
try:
    time.sleep(0.2)
    rs = requests.get(subm_url, headers=sec_headers, timeout=15)
    print(f"SEC submissions: {rs.status_code}, len={len(rs.text)}")
    if rs.status_code == 200:
        subm_data = rs.json()
        recent = subm_data.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        accessions = recent.get("accessionNumber", [])
        filing_dates = recent.get("filingDate", [])
        primary_docs = recent.get("primaryDocument", [])
        print(f"  Total recent filings: {len(forms)}")
        
        # Find Form 4 filings
        form4_indices = [i for i, f in enumerate(forms) if f in ("4", "4/A")][:5]
        print(f"  Recent Form 4 indices: {form4_indices}")
        
        form4_xml_content = None
        form4_meta = []
        for idx in form4_indices:
            acc = accessions[idx]
            acc_clean = acc.replace("-", "")
            doc = primary_docs[idx]
            f_date = filing_dates[idx]
            form_type = forms[idx]
            form4_meta.append({"form": form_type, "date": f_date, "acc": acc, "doc": doc})
            print(f"  Form 4: date={f_date}, acc={acc}, doc={doc}")
            
            # Fetch the actual XML for the first XML document
            if not form4_xml_content and doc.endswith(".xml"):
                xml_url = f"https://www.sec.gov/Archives/edgar/data/1045810/{acc_clean}/{doc}"
                print(f"  Fetching Form 4 XML: {xml_url}")
                time.sleep(0.2)
                rx = requests.get(xml_url, headers=sec_headers, timeout=12)
                if rx.status_code == 200:
                    form4_xml_content = rx.text
                    print(f"  Got Form 4 XML, len={len(form4_xml_content)}")
                    (FIXTURES_DIR / "sec_form4_nvda_fixture.xml").write_text(form4_xml_content, encoding="utf-8")
        
        # Save trimmed submissions fixture (first 50 filings)
        trimmed_recent = {k: v[:50] for k, v in recent.items() if isinstance(v, list)}
        trimmed_subm = {
            "cik": subm_data.get("cik"),
            "entityName": subm_data.get("entityName"),
            "filings": {"recent": trimmed_recent}
        }
        (FIXTURES_DIR / "sec_submissions_nvda_fixture.json").write_text(json.dumps(trimmed_subm, indent=2), encoding="utf-8")
        results["sec_form4"] = {"status": 200, "form4_count": len(form4_indices), "has_xml": bool(form4_xml_content)}
except Exception as e:
    print(f"SEC submissions ERROR: {e}")
    results["sec_form4"] = {"error": str(e)}

# ---------------------------------------------------------
# 4. News Discovery (Google News RSS Search)
# ---------------------------------------------------------
print("\n--> Probing Google News RSS Search...")
news_url = "https://news.google.com/rss/search?q=(NVDA+NVIDIA)+stock&hl=en-US&gl=US&ceid=US:en"
try:
    rn = requests.get(news_url, headers=BROWSER_HEADERS, timeout=12)
    print(f"Google News RSS: {rn.status_code}, len={len(rn.text)}")
    if rn.status_code == 200:
        root = ET.fromstring(rn.text)
        items = root.findall(".//item")
        print(f"  Articles found: {len(items)}")
        if items:
            first_item = items[0]
            title = first_item.findtext("title", "")
            pub_date = first_item.findtext("pubDate", "")
            source = first_item.findtext("source", "")
            print(f"  Latest article: {title}")
            print(f"  Pub date: {pub_date}, Source: {source}")
        (FIXTURES_DIR / "google_news_nvda_fixture.xml").write_text(rn.text, encoding="utf-8")
        results["google_news"] = {"status": 200, "items": len(items)}
except Exception as e:
    print(f"Google News RSS ERROR: {e}")
    results["google_news"] = {"error": str(e)}

print("\n================ SUMMARY ================")
print(json.dumps(results, indent=2))
