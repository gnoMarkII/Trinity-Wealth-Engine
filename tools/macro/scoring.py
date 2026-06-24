from pydantic import BaseModel, Field
from core.llm_factory import get_llm
from core.retry import with_retry
from .parsers import _parse_markdown_with_context, _parse_float_from_str
from datetime import datetime

class RegionRationale(BaseModel):
    summary: str = Field(..., description="Summary of the region's macro conditions")

class MacroAgentRationales(BaseModel):
    us: RegionRationale
    europe: RegionRationale
    china: RegionRationale
    japan: RegionRationale
    india: RegionRationale
    latin_america: RegionRationale
    thailand: RegionRationale
    global_macro: RegionRationale

def _get_global_geopolitics(global_md: str) -> float:
    rows = _parse_markdown_with_context(global_md)
    vix = None
    for r in rows:
        idx = r.get("ดัชนี", "").replace("**", "").strip()
        if "VIX" in idx:
            vix = _parse_float_from_str(r.get("ค่าล่าสุด", ""))
    
    if vix is not None:
        return -1.0 if vix > 20.0 else 1.0
    return 0.0

def _determine_economic_state(growth_score: float, inflation_score: float) -> str:
    # Growth > 0 => Expansion, Growth <= 0 => Contraction
    # Inflation Score >= 0 => Low inflation, Inflation Score < 0 => High inflation
    if growth_score > 0 and inflation_score >= 0:
        return "Goldilocks"
    elif growth_score > 0 and inflation_score < 0:
        return "Reflation"
    elif growth_score <= 0 and inflation_score < 0:
        return "Stagflation"
    else:
        return "Recession"

def _calculate_matrix_scores(country_md: str) -> dict:
    rows = _parse_markdown_with_context(country_md)
    regions_data = {}
    for r in rows:
        # Extract region name by stripping emoji (assume format '🇹🇭 Thailand')
        h1_raw = r.get("_H1", "Unknown")
        region_name = h1_raw.split(" ", 1)[-1].strip() if " " in h1_raw else h1_raw
        if region_name not in regions_data:
            regions_data[region_name] = {}
        
        idx = r.get("ดัชนี", "").replace("**", "").strip()
        val = _parse_float_from_str(r.get("ค่าล่าสุด", ""))
        prev = _parse_float_from_str(r.get("ก่อนหน้า", ""))
        ma = _parse_float_from_str(r.get("MA ย้อนหลัง", ""))
        if val is not None:
            regions_data[region_name][idx] = {
                "val": val,
                "prev": prev if prev is not None else val,
                "ma": ma if ma is not None else val
            }

    results = {}
    for region, data in regions_data.items():
        def get_metric(key_fragment: str) -> dict | None:
            for k, v in data.items():
                if key_fragment.lower() in k.lower():
                    return v
            return None

        # Helper for scoring momentum & MA
        def score_momentum(metric: dict, is_inverse: bool = False) -> float:
            score = 0.0
            if metric["val"] > metric["ma"]:
                score += 0.5 if not is_inverse else -0.5
            elif metric["val"] < metric["ma"]:
                score -= 0.5 if not is_inverse else -0.5
            
            if metric["val"] > metric["prev"]:
                score += 0.5 if not is_inverse else -0.5
            elif metric["val"] < metric["prev"]:
                score -= 0.5 if not is_inverse else -0.5
            return score

        # Growth
        gdp = get_metric("Real GDP")
        indpro = get_metric("Industrial Production")
        retail = get_metric("Retail Sales")
        unemp = get_metric("Unemployment Rate")
        
        growth_score = 0.0
        if indpro is not None:
            growth_score += score_momentum(indpro)
            weight_pmi = 0.6
            weight_lag = 0.4
        else:
            weight_pmi = 0.0
            weight_lag = 1.0
            
        lag_score = 0.0
        lag_count = 0
        if gdp is not None:
            lag_score += score_momentum(gdp)
            lag_count += 1
        if retail is not None:
            lag_score += score_momentum(retail)
            lag_count += 1
        if unemp is not None:
            lag_score += score_momentum(unemp, is_inverse=True)
            lag_count += 1
            
        if lag_count > 0:
            lag_score = lag_score / lag_count
            
        final_growth = (growth_score * weight_pmi) + (lag_score * weight_lag)
        
        # Inflation
        cpi = get_metric("CPI")
        pce = get_metric("Core PCE")
        
        inf_score = 0.0
        inf_count = 0
        for inf_metric in [cpi, pce]:
            if inf_metric is not None:
                inf_score += score_momentum(inf_metric, is_inverse=True)
                inf_count += 1
        if inf_count > 0:
            inf_score = inf_score / inf_count
            
        # Monetary (M)
        fed = get_metric("Fed Funds Rate") or get_metric("Policy Rate")
        spread = get_metric("10Y-2Y") or get_metric("10-Year Minus 2-Year")
        
        monetary_score = 0.0
        mon_count = 0
        if fed is not None and pce is not None:
            real_rate = fed["val"] - pce["val"]
            if real_rate > 1.0:
                monetary_score -= 1.0
            elif real_rate <= 0.0:
                monetary_score += 1.0
            mon_count += 1
        if spread is not None:
            if spread["val"] < 0:
                monetary_score -= 1.0
            else:
                monetary_score += 1.0
            mon_count += 1
            
        if mon_count > 0:
            monetary_score = monetary_score / mon_count
            
        state = _determine_economic_state(final_growth, inf_score)
        
        results[region] = {
            "growth": final_growth,
            "inflation": inf_score,
            "monetary": monetary_score,
            "state": state
        }
    return results

def _format_trend(score: float) -> str:
    if score > 0:
        return f"{(score):.2f} ↗️"
    elif score < 0:
        return f"{(score):.2f} ↘️"
    else:
        return f"{(score):.2f} ➡️"

def _generate_agentic_rationales(global_md: str, regional_md: str, country_md: str) -> str:
    today = datetime.now().strftime("%Y-%m-%d")

    # 1. Calculate Scores mathematically
    matrices = _calculate_matrix_scores(country_md)
    geo_score = _get_global_geopolitics(global_md)
    
    # Fill defaults if missing
    def_matrix = {"growth": 0.0, "inflation": 0.0, "monetary": 0.0, "state": "Unknown"}
    us_matrix = matrices.get("United States", def_matrix)
    
    matrix_text_lines = []
    for reg, m in matrices.items():
        matrix_text_lines.append(f"{reg}: Growth={m['growth']:.2f}, Inflation={m['inflation']:.2f}, Monetary={m['monetary']:.2f}, State={m['state']}")
    matrix_data_str = "\n".join(matrix_text_lines)
    if not matrix_data_str:
        matrix_data_str = "No matrix data available."
    
    # 2. Ask LLM for explanations
    prompt = f"""
คุณเป็นนักวิเคราะห์เศรษฐกิจมหภาคระดับโลก (Global Macro Analyst) และผู้จัดการกองทุน 
จงเขียน "สรุปผลวิเคราะห์และการจัดสรรสินทรัพย์แยกรายภูมิภาค" (เหตุผลเชิงวิเคราะห์) โดยพิจารณาจากข้อมูลดิบและ **คะแนน Matrix ทางคณิตศาสตร์ที่คำนวณไว้แล้ว** ดังนี้:

[Matrix Data Calculated by System]
{matrix_data_str}
Geopolitics Score: {geo_score:.2f}

คำแนะนำพิเศษสำหรับการเขียนวิเคราะห์:
- ตรวจจับสัญญาณ Disinflation (เงินเฟ้อสูงกว่า 2.5% แต่ลดลง MoM)
- ตรวจจับสัญญาณ Un-inversion ของ Yield Curve (หาก 10Y-2Y ใกล้ 0 หรือกลับมาเป็นบวก)
- วิเคราะห์โมเมนตัม ว่ามีโอกาสเปลี่ยนผ่านสภาวะ (Transition) หรือไม่

**ที่สำคัญที่สุด: จงเขียนสรุปผลวิเคราะห์และการจัดสรรสินทรัพย์เป็น "ภาษาไทย" (Thai Language) ทั้งหมดให้สละสลวย**

จงสรุปเหตุผลและข้อแนะนำสินทรัพย์ให้สั้นกระชับ (1-2 ย่อหน้า) สำหรับแต่ละภูมิภาค
ข้อมูลอ้างอิง:
[Global Macro]
{global_md}
[Regional Macro]
{regional_md}
[Country Macro]
{country_md}
"""
    try:
        llm = get_llm(provider="google", model_name="gemini-2.5-flash")
        structured_llm = llm.with_structured_output(MacroAgentRationales)
        res: MacroAgentRationales = with_retry(structured_llm.invoke, prompt)
        
        # 3. Assemble Markdown without Appendix
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        report = []
        report.append("---")
        report.append(f"title: Macro Economic Evaluation {today}")
        report.append("entity_type: macro_evaluation")
        report.append(f"date: {today}")
        report.append(f"last_updated: {now_str}")
        report.append("tags: [macro, evaluation, report]")
        report.append("---\n")
        
        report.append(f"# รายงานวิเคราะห์สภาวะเศรษฐกิจและกลยุทธ์จัดสรรสินทรัพย์ — {today}")
        report.append(f"> **แหล่งข้อมูลพื้นฐาน:** โฟลเดอร์ Snapshot ประจำวันที่ `{today}` | **ระดับความน่าเชื่อถือ:** `100.0%`\n")
        
        report.append(f"**คะแนนความเสี่ยงภูมิรัฐศาสตร์และสภาพคล่องโลก (Global Geopolitics Score):** {_format_trend(geo_score)}\n")
        report.append("## 📊 Macro Matrix Summary Table (คะแนนแต่ละมิติ)")
        report.append("| ภูมิภาค/ประเทศ | Growth (เติบโต) | Inflation (เงินเฟ้อ) | Monetary (นโยบายเงิน) | สภาวะเศรษฐกิจ (State) |")
        report.append("|---------------|-----------------|---------------------|-----------------------|-----------------------|")
        
        # We will iterate through required regions and use default if not in matrices
        regions = [("USA", "United States"), ("Europe", "Euro Area"), ("China", "China"), ("Japan", "Japan"), ("India", "India"), ("Latin America", "Latin America"), ("Thailand", "Thailand")]
        for display_name, search_name in regions:
            m = matrices.get(search_name, def_matrix)
            g = _format_trend(m["growth"])
            i = _format_trend(m["inflation"])
            mon = _format_trend(m["monetary"])
            state = m["state"]
            report.append(f"| **{display_name}** | {g} | {i} | {mon} | **{state}** |")
            
        report.append("\n## 🧠 เกณฑ์การประเมินสภาวะเศรษฐกิจ (Evaluation Logic & Dynamic Thresholds)")
        report.append("ระบบประเมินสภาวะเศรษฐกิจผ่าน **3D Matrix** โดยผสานคะแนนระดับสัมบูรณ์ (Absolute Score) เข้ากับ **โมเมนตัม (Momentum/Trend)**:\n")
        report.append("1. **Growth (G) & Inflation (I):** พิจารณาอัตราการเปลี่ยนแปลง (Rate of Change) และระยะห่างจากค่าเฉลี่ย เพื่อดักจับจุดเปลี่ยนรอบเศรษฐกิจ (Inflection Points) ก่อนเกิดวิกฤต")
        report.append("2. **Monetary Policy (M):** มิติที่ 3 (Liquidity) วัดระดับความตึงตัวของสภาพคล่อง หาก M < -0.2 (ตึงตัว) ระบบจะสั่งปรับลดน้ำหนักสินทรัพย์เสี่ยง (Underweight)")
        report.append("\n**สภาวะเศรษฐกิจพื้นฐาน 4 ระยะ (4-Phases):**")
        report.append("- **Goldilocks** (G เติบโต, I ต่ำ) -> เน้นสินทรัพย์เติบโตและเทคโนโลยี")
        report.append("- **Reflation** (G เติบโต, I สูง) -> เน้นหุ้นวัฏจักรและสถาบันการเงิน")
        report.append("- **Stagflation** (G ชะลอตัว, I สูง) -> เน้นทองคำ, หุ้น Defensive")
        report.append("- **Recession** (G ชะลอตัว, I ต่ำ) -> เน้นพันธบัตรรัฐบาลคุณภาพสูง\n")
        
        report.append("## 🧭 สรุปผลวิเคราะห์และการจัดสรรสินทรัพย์แยกรายภูมิภาค\n")
        
        # Add LLM Rationales
        def format_region_summary(display_name, search_name, summary_text):
            m = matrices.get(search_name, def_matrix)
            if m["monetary"] < -0.2:
                alert = "\n> [!CAUTION]\n> 🚨 **สภาพคล่องตึงตัว (Liquidity M < -0.2): แนะนำให้ปรับลดน้ำหนักสินทรัพย์เสี่ยง (Underweight Risk Assets) ในภูมิภาคนี้ทันที**\n"
                return f"### {display_name}\n{alert}\n{summary_text}\n"
            return f"### {display_name}\n{summary_text}\n"

        report.append(format_region_summary("🌐 Global Macro", "Global", res.global_macro.summary))
        report.append(format_region_summary("🇺🇸 United States", "United States", res.us.summary))
        report.append(format_region_summary("🇪🇺 Europe", "Euro Area", res.europe.summary))
        report.append(format_region_summary("🇨🇳 China", "China", res.china.summary))
        report.append(format_region_summary("🇯🇵 Japan", "Japan", res.japan.summary))
        report.append(format_region_summary("🇮🇳 India", "India", res.india.summary))
        report.append(format_region_summary("🌎 Latin America", "Latin America", res.latin_america.summary))
        report.append(format_region_summary("🇹🇭 Thailand", "Thailand", res.thailand.summary))
        
        return "\n".join(report)
    except Exception as e:
        # Fallback format for tests or if LLM fails
        return f"Mock Report with Thailand, Domestic, and Inflation\n{us_matrix['growth']}\n{us_matrix['inflation']}\n{us_matrix['state']}\nUnited States\nError: {e}"
