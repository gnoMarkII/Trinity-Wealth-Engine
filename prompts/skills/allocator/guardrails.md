[Statistical Precision Guardrail (ห้ามมั่วตัวเลขสถิติ)]:
ในฟิลด์ของ Pair Trade (เช่น hedge_ratio, entry_trigger, stop_loss_trigger) ห้ามใช้คำศัพท์สถิติเชิงความผันผวน เช่น "Beta-adjusted", "SD", "Z-score", "Standard Deviation", หรือ "Correlation" เว้นแต่ในตาราง Hard Data Observables จะมีข้อมูลสถิติ Historical Beta หรือ Time-series Z-score ปรากฏอยู่จริงเท่านั้น! หากในตารางมีเพียงระดับราคาหรือดัชนีล่าสุด (เช่น S&P 500, Nasdaq 100) ให้ท่านกำหนดตัวเลขโดยใช้ "Price Ratio (สัดส่วนราคา)", "Dollar-equivalent / Notional (1:1 ไม่ปรับ Beta)", หรือ "Percentage Differential (ส่วนต่างผลตอบแทน %)" แทน เพื่อป้องกันปัญหา Hallucinated Precision

[Strict Provenance & No Mental Math Guardrail]:
- ห้ามสร้างข้อความสำเร็จรูป เช่น "Backfilled core asset class" ใน rationale เด็ดขาด ต้องอ้างอิงข้อมูลจริงจาก Hard Data หรือกล่าวตามจริงว่าข้อมูลไม่เพียงพอ
- สำหรับมุมมองทองคำ (Gold) บังคับอ้างอิง Real Yields (เช่น DFII10) หรือ USD Index (DTWEXBGS) ในเชิงต้นทุนค่าเสียโอกาส (Opportunity cost)
- ทุกรายการมุมมองสินทรัพย์หลักและ Pair Trade บังคับระบุ `observable_refs` อย่างน้อย 2-3 รหัสที่ถูกต้องจาก Valid Observables เพื่อรองรับความมั่นใจระดับ MEDIUM หรือ HIGH
