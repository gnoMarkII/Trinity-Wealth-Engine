---
name: custom_dcf_sensitivity
description: สร้างตารางวิเคราะห์ความอ่อนไหวของมูลค่าหุ้น (DCF Sensitivity Matrix) ตาม WACC และ Terminal Growth Rate
---
# Custom DCF Sensitivity Skill

## วัตถุประสงค์
ประเมินมูลค่าหุ้นเชิงลึกตามช่วงอัตราคิดลด (Discount Rate / WACC) และอัตราเติบโตระยะยาว (Terminal Growth)

## ขั้นตอนการทำงาน
1. ดึงรายงานปัจจัยพื้นฐานของหุ้นผ่านเครื่องมือ `get_equity_report_summary(ticker)`
2. ส่งสคริปต์ Python เข้าไปประมวลผลใน `run_sandbox_calculation` ด้วยโมเดล Discounted Cash Flow โดยใช้ตัวเลข `Decimal`
3. สร้างตารางผลลัพธ์ Sensitivity Matrix 3x3 หรือ 5x5
4. สรุป Margin of Safety และโซนราคาเข้าซื้อที่ปลอดภัย

