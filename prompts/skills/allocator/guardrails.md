[Statistical Precision Guardrail (ห้ามมั่วตัวเลขสถิติ)]:
ในฟิลด์ของ Pair Trade (เช่น hedge_ratio, entry_trigger, stop_loss_trigger) ห้ามใช้คำศัพท์สถิติเชิงความผันผวน เช่น "Beta-adjusted", "SD", "Z-score", "Standard Deviation", หรือ "Correlation" เว้นแต่ในตาราง Hard Data Observables จะมีข้อมูลสถิติ Historical Beta หรือ Time-series Z-score ปรากฏอยู่จริงเท่านั้น! หากในตารางมีเพียงระดับราคาหรือดัชนีล่าสุด (เช่น S&P 500, Nasdaq 100) ให้ท่านกำหนดตัวเลขโดยใช้ "Price Ratio (สัดส่วนราคา)", "Dollar-equivalent / Notional (1:1 ไม่ปรับ Beta)", หรือ "Percentage Differential (ส่วนต่างผลตอบแทน %)" แทน เพื่อป้องกันปัญหา Hallucinated Precision

[Strict Provenance & No Mental Math Guardrail]:
- ห้ามสร้างข้อความสำเร็จรูป เช่น "Backfilled core asset class" ใน rationale เด็ดขาด ต้องอ้างอิงข้อมูลจริงจาก Hard Data หรือกล่าวตามจริงว่าข้อมูลไม่เพียงพอ
- สำหรับมุมมองทองคำ (Gold) บังคับอ้างอิง Real Yields (เช่น DFII10) หรือ USD Index (DTWEXBGS) ในเชิงต้นทุนค่าเสียโอกาส (Opportunity cost)
- ทุกรายการมุมมองสินทรัพย์หลักและ Pair Trade บังคับระบุ `observable_refs` อย่างน้อย 2-3 รหัสที่ถูกต้องจาก Valid Observables เพื่อรองรับความมั่นใจระดับ MEDIUM หรือ HIGH

[Regional Separation & Thailand Market Stance Guardrail (การแยกตลาดโลกออกจากตลาดทุนไทย)]:
- บังคับแยกสภาวะเศรษฐกิจมหภาคระดับโลก (Global/US Macro Regime) ออกจากท่าทีตลาดทุนไทย (Thailand Market Stance) อย่างเด็ดขาด:
  1. การจัดสรรสินทรัพย์ระดับโลก (Global Equities, US Treasuries, Commodities): ให้ใช้ผลประเมิน Overall Regime (เช่น Goldilocks, Reflation, Stagflation, Recession) ที่มี US Macro Data เป็นแกนหลัก
  2. การจัดสรรสินทรัพย์ไทย (SET Equities, Thai Retail Gold): หากสภาวะเศรษฐกิจของไทยระบุเป็น UNKNOWN เนื่องจากการขาดข้อมูล GDP/CPI ทางการ ให้ระบุ Data Gaps ชัดเจน ห้ามสรุปว่าไทยอยู่ใน Recession หรือ Boom เอง และให้ใช้ข้อมูล Terminal V2 (กระแสเงินทุนนักลงทุนต่างชาติ SET Foreign Net Flow, Market Breadth Advance/Decline Ratio, Valuation P/E & Dividend Yield, และราคาทองคำแท่งสมาคมค้าทองคำ GTA) ในการกำหนดทัศนะต่อตลาดทุนไทย (Thailand Market Stance) แทน
  3. ห้ามนำข้อมูลเฉพาะของไทยไปคำนวณถัวเฉลี่ยกับ US เพื่อประเมิน US Recession Risk Score

[Observable Evidence & Policy Rate Spread Guardrail (การกำกับหลักฐานที่ใช้ได้และส่วนต่างอัตราดอกเบี้ย)]:
- ห้ามนำตัวชี้วัดที่อยู่ในกลุ่ม `invalid_observables` (เช่น `RSAFS` หรือตัวชี้วัดที่ stale/unverified) ไปเป็นหลักฐานสนับสนุน (supporting evidence) สำหรับ Growth หรือยกระดับความมั่นใจเป็น MEDIUM หรือ HIGH หากตัวชี้วัดใด invalid ให้ระบุว่าเป็นข้อจำกัดของข้อมูลเท่านั้น
- ห้ามกล่าวอ้างตัวเลขส่วนต่างอัตราดอกเบี้ยนโยบาย Fed–BoT (Policy Rate Differential / Spread) หรือระบุตัวเลข เช่น +275 bps เว้นแต่จะมี observable `obs_diff_us_th_policy_rate_bis` อยู่ในกลุ่ม `valid_observables` จริงเท่านั้น หากไม่มีข้อมูลส่วนต่างที่ยืนยันได้ ให้ระบุว่า "ปัจจุบันยังไม่มีข้อมูลส่วนต่างอัตราดอกเบี้ยนโยบายที่ยืนยันได้ จึงยังไม่สามารถประเมินส่วนต่างได้" และจำกัดระดับความมั่นใจของ FX เป็น LOW หรือไม่เกิน MEDIUM

