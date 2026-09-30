"""Live contract probe script for Terminal V2 Phase 2 Data Providers.

Probes all 8 external keyless endpoints, records status, shape, headers,
and generates sanitized offline fixtures in tests/fixtures/terminal_v2/.
"""
from datetime import datetime, timedelta
import json
from pathlib import Path
import requests

FIXTURE_DIR = Path("tests/fixtures/terminal_v2")
FIXTURE_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/128.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,application/json,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9,th;q=0.8",
}

results = {}

def probe_finra():
    print("Probing FINRA...")
    now = datetime.utcnow()
    for i in range(8):
        day_str = (now - timedelta(days=i)).strftime("%Y%m%d")
        url = f"https://cdn.finra.org/equity/regsho/daily/CNMSshvol{day_str}.txt"
        try:
            r = requests.get(url, headers=HEADERS, timeout=10)
            if r.status_code == 200 and len(r.text) > 100:
                lines = [l for l in r.text.splitlines() if l.strip()]
                # Keep header, top 50 symbols (including AAPL, NVDA, TSLA if present), and footer
                sample_lines = lines[:20]
                target_syms = {"AAPL", "NVDA", "TSLA", "SPY", "MSFT"}
                for l in lines[20:]:
                    parts = l.split("|")
                    if len(parts) > 1 and parts[1] in target_syms:
                        sample_lines.append(l)
                if lines[-1] not in sample_lines:
                    sample_lines.append(lines[-1])
                fixture_text = "\n".join(sample_lines)
                (FIXTURE_DIR / "finra_short_volume.txt").write_text(fixture_text, encoding="utf-8")
                results["finra"] = {
                    "status": 200,
                    "url": url,
                    "date": day_str,
                    "total_lines": len(lines),
                    "fixture_lines": len(sample_lines),
                }
                print(f"FINRA SUCCESS: {day_str} ({len(lines)} lines)")
                return
        except Exception as e:
            print(f"FINRA day {day_str} failed: {e}")
    results["finra"] = {"status": "failed", "error": "No file found in 7 days"}
    print("FINRA FAILED")

def probe_cboe():
    print("Probing CBOE (AAPL)...")
    url = "https://cdn.cboe.com/api/global/delayed_quotes/options/AAPL.json"
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        if r.status_code == 200:
            data = r.json()
            options = data.get("data", {}).get("options", [])
            # Sanitize fixture: keep first 50 options contracts across a couple expiries
            cur_price = data.get("data", {}).get("current_price")
            iv30 = data.get("data", {}).get("iv30")
            sanitized = {
                "data": {
                    "current_price": cur_price,
                    "iv30": iv30,
                    "options": options[:60],
                }
            }
            (FIXTURE_DIR / "cboe_options_aapl.json").write_text(json.dumps(sanitized, indent=2), encoding="utf-8")
            results["cboe"] = {
                "status": 200,
                "current_price": cur_price,
                "iv30": iv30,
                "total_contracts": len(options),
            }
            print(f"CBOE SUCCESS: {len(options)} contracts, spot={cur_price}")
            return
        results["cboe"] = {"status": r.status_code}
    except Exception as e:
        results["cboe"] = {"status": "error", "error": str(e)}
        print(f"CBOE FAILED: {e}")

def probe_nyfed():
    print("Probing NY Fed Reference Rates...")
    url = "https://markets.newyorkfed.org/api/rates/all/latest.json"
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        if r.status_code == 200:
            data = r.json()
            rates = data.get("refRates", [])
            (FIXTURE_DIR / "nyfed_rates.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
            codes = [item.get("type") for item in rates]
            results["nyfed"] = {
                "status": 200,
                "rates_count": len(rates),
                "codes": codes,
            }
            print(f"NY FED SUCCESS: {codes}")
            return
        results["nyfed"] = {"status": r.status_code}
    except Exception as e:
        results["nyfed"] = {"status": "error", "error": str(e)}
        print(f"NY FED FAILED: {e}")

def probe_treasury():
    print("Probing US Treasury...")
    now = datetime.utcnow()
    # 1. Yield Curve
    month_str = now.strftime("%Y%m")
    yc_url = f"https://home.treasury.gov/resource-center/data-chart-center/interest-rates/pages/xml?data=daily_treasury_yield_curve&field_tdr_date_value_month={month_str}"
    yc_res = "failed"
    try:
        r = requests.get(yc_url, headers=HEADERS, timeout=10)
        if r.status_code == 200 and "<feed" in r.text:
            (FIXTURE_DIR / "treasury_yield_curve.xml").write_text(r.text, encoding="utf-8")
            yc_res = "success"
        else:
            # Fallback to previous month
            prev_m = (now.replace(day=1) - timedelta(days=1)).strftime("%Y%m")
            yc_url = f"https://home.treasury.gov/resource-center/data-chart-center/interest-rates/pages/xml?data=daily_treasury_yield_curve&field_tdr_date_value_month={prev_m}"
            r = requests.get(yc_url, headers=HEADERS, timeout=10)
            if r.status_code == 200 and "<feed" in r.text:
                (FIXTURE_DIR / "treasury_yield_curve.xml").write_text(r.text, encoding="utf-8")
                yc_res = "success_prev_month"
    except Exception as e:
        yc_res = f"error: {e}"

    # 2. Auctions
    auc_url = "https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v1/accounting/od/auctions_query?sort=-auction_date&filter=bid_to_cover_ratio:gt:0&page[size]=10"
    auc_res = "failed"
    try:
        r = requests.get(auc_url, headers=HEADERS, timeout=10)
        if r.status_code == 200:
            (FIXTURE_DIR / "treasury_auctions.json").write_text(r.text, encoding="utf-8")
            auc_res = "success"
    except Exception as e:
        auc_res = f"error: {e}"

    # 3. Debt to Penny
    debt_url = "https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v2/accounting/od/debt_to_penny?sort=-record_date&page[size]=5"
    debt_res = "failed"
    try:
        r = requests.get(debt_url, headers=HEADERS, timeout=10)
        if r.status_code == 200:
            (FIXTURE_DIR / "treasury_debt_to_penny.json").write_text(r.text, encoding="utf-8")
            debt_res = "success"
    except Exception as e:
        debt_res = f"error: {e}"

    results["treasury"] = {
        "yield_curve": yc_res,
        "auctions": auc_res,
        "debt_to_penny": debt_res,
    }
    print(f"TREASURY: YC={yc_res}, Auctions={auc_res}, Debt={debt_res}")

def probe_sec_th():
    print("Probing SEC Thailand...")
    endpoints = {
        "mf_port": "https://dividend.sec.or.th/stat-report/MF_PORT_TH.csv",
        "stat_dept": "https://dividend.sec.or.th/stat-report/STAT_DEPT_TH.csv",
        "offer_debt": "https://dividend.sec.or.th/stat-report/OFFER_DEBT_COR_TH.csv",
    }
    sec_results = {}
    for name, url in endpoints.items():
        try:
            r = requests.get(url, headers=HEADERS, timeout=20)
            if r.status_code == 200 and "<title>Request Rejected</title>" not in r.text:
                filename = f"sec_th_{name}.csv"
                (FIXTURE_DIR / filename).write_text(r.text, encoding="utf-8")
                sec_results[name] = {"status": 200, "size": len(r.text)}
                print(f"SEC TH {name} SUCCESS: {len(r.text)} bytes")
            else:
                sec_results[name] = {"status": r.status_code, "waf_blocked": "<title>Request Rejected" in r.text}
                print(f"SEC TH {name} WAF BLOCKED or failed")
        except Exception as e:
            sec_results[name] = {"status": "error", "error": str(e)}
            print(f"SEC TH {name} error: {e}")
    results["sec_th"] = sec_results

def probe_mof_th():
    print("Probing MOF Thailand Public Debt...")
    url = "https://dataservices.mof.go.th/export/csv/menu5?id=4"
    try:
        r = requests.get(url, headers=HEADERS, timeout=20)
        if r.status_code == 200 and len(r.text) > 100:
            (FIXTURE_DIR / "mof_th_public_debt.csv").write_text(r.text, encoding="utf-8")
            results["mof_th"] = {"status": 200, "size": len(r.text)}
            print(f"MOF TH SUCCESS: {len(r.text)} bytes")
            return
        results["mof_th"] = {"status": r.status_code}
    except Exception as e:
        results["mof_th"] = {"status": "error", "error": str(e)}
        print(f"MOF TH error: {e}")

def probe_polymarket():
    print("Probing Polymarket Gamma...")
    url = "https://gamma-api.polymarket.com/markets?closed=false&limit=10&order=volume&ascending=false"
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        if r.status_code == 200:
            data = r.json()
            (FIXTURE_DIR / "polymarket_markets.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
            results["polymarket"] = {"status": 200, "count": len(data) if isinstance(data, list) else 0}
            print(f"POLYMARKET SUCCESS: {len(data)} markets")
            return
        results["polymarket"] = {"status": r.status_code}
    except Exception as e:
        results["polymarket"] = {"status": "error", "error": str(e)}
        print(f"POLYMARKET error: {e}")

def probe_sosovalue():
    print("Probing SoSoValue ETF Flows...")
    hist_url = "https://api.sosovalue.xyz/openapi/v2/etf/historicalInflowChart"
    metric_url = "https://api.sosovalue.xyz/openapi/v2/etf/currentEtfDataMetrics"
    soso_headers = {**HEADERS, "Content-Type": "application/json"}
    soso_res = {}
    try:
        r_hist = requests.post(hist_url, headers=soso_headers, json={"type": "us-btc-spot"}, timeout=10)
        if r_hist.status_code == 200:
            (FIXTURE_DIR / "sosovalue_history_btc.json").write_text(r_hist.text, encoding="utf-8")
            soso_res["history"] = {"status": 200, "code": r_hist.json().get("code")}
        else:
            soso_res["history"] = {"status": r_hist.status_code}
    except Exception as e:
        soso_res["history"] = {"status": "error", "error": str(e)}

    try:
        r_met = requests.post(metric_url, headers=soso_headers, json={"type": "us-btc-spot"}, timeout=10)
        if r_met.status_code == 200:
            (FIXTURE_DIR / "sosovalue_metrics_btc.json").write_text(r_met.text, encoding="utf-8")
            soso_res["metrics"] = {"status": 200, "code": r_met.json().get("code")}
        else:
            soso_res["metrics"] = {"status": r_met.status_code}
    except Exception as e:
        soso_res["metrics"] = {"status": "error", "error": str(e)}

    results["sosovalue"] = soso_res
    print(f"SOSOVALUE: {soso_res}")

def main():
    probe_finra()
    probe_cboe()
    probe_nyfed()
    probe_treasury()
    probe_sec_th()
    probe_mof_th()
    probe_polymarket()
    probe_sosovalue()
    
    (FIXTURE_DIR / "probe_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("\n--- ALL PROBES COMPLETE ---")
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
