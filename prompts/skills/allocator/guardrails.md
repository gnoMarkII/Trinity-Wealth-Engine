[Statistical Precision Guardrail (ห้ามมั่วตัวเลขสถิติ)]:
ในฟิลด์ของ Pair Trade (เช่น hedge_ratio, entry_trigger, stop_loss_trigger) ห้ามใช้คำศัพท์สถิติเชิงความผันผวน เช่น "Beta-adjusted", "SD", "Z-score", "Standard Deviation", หรือ "Correlation" เว้นแต่ในตาราง Hard Data Observables จะมีข้อมูลสถิติ Historical Beta หรือ Time-series Z-score ปรากฏอยู่จริงเท่านั้น! หากในตารางมีเพียงระดับราคาหรือดัชนีล่าสุด (เช่น S&P 500, Nasdaq 100) ให้ท่านกำหนดตัวเลขโดยใช้ "Price Ratio (สัดส่วนราคา)", "Dollar-equivalent / Notional (1:1 ไม่ปรับ Beta)", หรือ "Percentage Differential (ส่วนต่างผลตอบแทน %)" แทน เพื่อป้องกันปัญหา Hallucinated Precision

[Strict Provenance & No Mental Math Guardrail]:
- ห้ามสร้างข้อความสำเร็จรูป เช่น "Backfilled core asset class" ใน rationale เด็ดขาด ต้องอ้างอิงข้อมูลจริงจาก Hard Data หรือกล่าวตามจริงว่าข้อมูลไม่เพียงพอ
- สำหรับมุมมองทองคำ (Gold) บังคับอ้างอิง Real Yields (เช่น DFII10) หรือ USD Index (DTWEXBGS) ในเชิงต้นทุนค่าเสียโอกาส (Opportunity cost)
- ทุกรายการมุมมองสินทรัพย์หลักและ Pair Trade บังคับระบุ `observable_refs` อย่างน้อย 2-3 รหัสที่ถูกต้องจาก Valid Observables เพื่อรองรับความมั่นใจระดับ MEDIUM หรือ HIGH
- [Dual Distinct Source Requirement (ป้องกัน SINGLE_SOURCE_PENALTY)]: สำหรับสินทรัพย์ที่ให้ระดับความมั่นใจ MEDIUM หรือ HIGH ต้องระบุ `source_refs` จากแหล่งข้อมูลที่ต่างกันอย่างน้อย 2 แหล่ง (เช่น FRED ร่วมกับ YFinance, SEC, หรือ Terminal) ห้ามใส่แหล่งข้อมูลเดียวซ้ำกัน
- [Exact Number Grounding in Supporting Data (ป้องกัน SUPPORTING_DATA_MISMATCH)]: ตัวเลขทางเศรษฐกิจหรือดัชนีทุกตัวที่ระบุใน `supporting_data` ต้องตรงกับตัวเลขจริงในตาราง Observables ทุกหลักทศนิยม ห้ามประมาณการหรือแต่งตัวเลขเองโดยพลการ

[Regional Separation & Thailand Market Stance Guardrail (การแยกตลาดโลกออกจากตลาดทุนไทย)]:
- บังคับแยกสภาวะเศรษฐกิจมหภาคระดับโลก (Global/US Macro Regime) ออกจากท่าทีตลาดทุนไทย (Thailand Market Stance) อย่างเด็ดขาด:
  1. การจัดสรรสินทรัพย์ระดับโลก (Global Equities, US Treasuries, Commodities): ให้ใช้ผลประเมิน Overall Regime (เช่น Goldilocks, Reflation, Stagflation, Recession) ที่มี US Macro Data เป็นแกนหลัก
  2. การจัดสรรสินทรัพย์ไทย (SET Equities, Thai Retail Gold): หากสภาวะเศรษฐกิจของไทยระบุเป็น UNKNOWN เนื่องจากการขาดข้อมูล GDP/CPI ทางการ ให้ระบุ Data Gaps ชัดเจน ห้ามสรุปว่าไทยอยู่ใน Recession หรือ Boom เอง และให้ใช้ข้อมูล Terminal V2 (กระแสเงินทุนนักลงทุนต่างชาติ SET Foreign Net Flow, Market Breadth Advance/Decline Ratio, Valuation P/E & Dividend Yield, และราคาทองคำแท่งสมาคมค้าทองคำ GTA) ในการกำหนดทัศนะต่อตลาดทุนไทย (Thailand Market Stance) แทน
  3. ห้ามนำข้อมูลเฉพาะของไทยไปคำนวณถัวเฉลี่ยกับ US เพื่อประเมิน US Recession Risk Score

[Observable Evidence & Policy Rate Spread Guardrail (การกำกับหลักฐานที่ใช้ได้และส่วนต่างอัตราดอกเบี้ย)]:
- ห้ามนำตัวชี้วัดที่อยู่ในกลุ่ม `invalid_observables` (เช่น `RSAFS` หรือตัวชี้วัดที่ stale/unverified) ไปเป็นหลักฐานสนับสนุน (supporting evidence) สำหรับ Growth หรือยกระดับความมั่นใจเป็น MEDIUM หรือ HIGH หากตัวชี้วัดใด invalid ให้ระบุว่าเป็นข้อจำกัดของข้อมูลเท่านั้น
- ห้ามกล่าวอ้างตัวเลขส่วนต่างอัตราดอกเบี้ยนโยบาย Fed–BoT (Policy Rate Differential / Spread) หรือระบุตัวเลข เช่น +275 bps เว้นแต่จะมี observable `obs_diff_us_th_policy_rate_bis` อยู่ในกลุ่ม `valid_observables` จริงเท่านั้น หากไม่มีข้อมูลส่วนต่างที่ยืนยันได้ ให้ระบุว่า "ปัจจุบันยังไม่มีข้อมูลส่วนต่างอัตราดอกเบี้ยนโยบายที่ยืนยันได้ จึงยังไม่สามารถประเมินส่วนต่างได้" และจำกัดระดับความมั่นใจของ FX เป็น LOW หรือไม่เกิน MEDIUM

[Regime Consistency & Inflation Contract Guardrail (สอดคล้องกับ EconomicState Enum และ Scoring Matrix)]:
- ฟิลด์ `overall_regime` ต้องเลือกใช้เฉพาะค่ามาตรฐานใน `EconomicState` enum เท่านั้น ได้แก่ `"Goldilocks"`, `"Reflation"`, `"Stagflation"`, `"Recession"`, หรือ `"Unknown"` (ห้ามสร้างหรือประดิษฐ์ชื่อสภาวะนอก Enum โดยเด็ดขาด มิฉะนั้นระบบจะเกิด Schema Validation Error)
- ให้จำแนกสภาวะเศรษฐกิจโดยแยกแยะ **ระดับเงินเฟ้อ (Inflation Level)** ออกจาก **ทิศทาง/โมเมนตัมของเงินเฟ้อ (Inflation Direction / Momentum)** ตามเกณฑ์ Scoring Matrix (`tools/macro/scoring.py`) และรายงานหลักฐานเงินเฟ้อจริงอย่างซื่อสัตย์ (ห้ามสั่งตัดหรือละเว้นตัวชี้วัด `CPI`, `PCE`, `Core PCE` ที่ผ่านการตรวจสอบใน `valid_observables` เด็ดขาด):
  1. `Goldilocks`: เศรษฐกิจขยายตัว (`Growth > 0`) ร่วมกับแรงกดดันเงินเฟ้อต่ำหรือชะลอตัวลงอย่างชัดเจน (`Inflation Score >= 0`) ห้ามระบุเป็น Goldilocks หากข้อมูลจริงชี้ว่าเงินเฟ้อกำลังเร่งตัวหรือยังตึงตัวเหนียวหนึบ
  2. `Reflation`: เศรษฐกิจขยายตัว (`Growth > 0`) ร่วมกับแรงกดดันเงินเฟ้อที่สูงขึ้นหรือตึงตัว (`Inflation Score < 0`) ในสภาวะนี้ให้รายงานตัวเลขเงินเฟ้อจริงจาก `valid_observables` (ทั้งระดับ % YoY และทิศทางเทียบกับอดีต/MA) ใน `regime_evidence`, `supporting_data` และ `rationale` อย่างครบถ้วน ห้ามสลับไปเลือก `Stagflation` (ซึ่งใช้เฉพาะกรณีที่ `Growth <= 0`) และห้ามประดิษฐ์ชื่อสภาวะนอก Enum
  3. `Stagflation`: เศรษฐกิจชะลอหรือหดตัว (`Growth <= 0`) ร่วมกับเงินเฟ้อสูงหรือยืดเยื้อ (`Inflation Score < 0`)
  4. `Recession`: เศรษฐกิจชะลอหรือหดตัว (`Growth <= 0`) ร่วมกับเงินเฟ้อชะลอตัวลง (`Inflation Score >= 0`)
  5. `Unknown`: ข้อมูล Growth หรือ Inflation ที่ผ่านการตรวจสอบไม่เพียงพอ

[Pair Trade Quantitative Evidence & Grounding Guardrail (ป้องกัน PT_DEFENSIVE_LOW และ GRACEFUL_DROP_PAIR_TRADES)]:
- ทุกรายการใน `pair_trades`:
  1. [Hard Numbers in Supporting Data]: ในฟิลด์ `supporting_data` บังคับต้องระบุตัวเลขและหน่วยทางการเงินจริงอย่างน้อย 1 รายการเสมอ (เช่น "Spread: +18 bps", "Price Ratio: 1.42x", "Yield Gap: 1.25%", หรือ "Forward P/E Spread: 13.3x") ห้ามใส่เพียงข้อความบรรยายเชิงคุณภาพเด็ดขาด มิฉะนั้นระบบจะลดระดับความเชื่อมั่นเป็น LOW ทันที (PT_DEFENSIVE_LOW)
  2. [Valid Observable Grounding]: ฟิลด์ `observable_refs` ต้องระบุรหัสตัวชี้วัดที่ถูกต้องจากกลุ่ม `valid_observables` อย่างน้อย 2 รหัสที่แตกต่างกัน (สำหรับความเชื่อมั่น MEDIUM หรือ HIGH) หรืออย่างน้อย 1 รหัส (สำหรับ LOW) และต้องมีตัวเลขในพารามิเตอร์การเทรดจริง ห้ามอ้างอิงรหัสที่ไม่มีอยู่จริงหรือตัวชี้วัดที่เป็น invalid มิฉะนั้นระบบ Guardrail จะตัดรายการ Pair Trade นั้นทิ้งทันที (GRACEFUL_DROP_PAIR_TRADES)
