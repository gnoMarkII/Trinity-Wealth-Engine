# ขั้นร่าง Buckets ตามบทบาทของเงิน (Purpose Buckets)

หน้าที่: ร่างรายการ Buckets 3-5 รายการจากแกนหลักที่ยืนยันแล้ว โดยเชื่อมโยงบทบาทของเงินเข้ากับสัดส่วนจัดสรร

## กติกา
1. ร่าง 3-5 Buckets พร้อมชื่อที่เข้าใจง่าย เช่น "รายได้สม่ำเสมอ", "เติบโตระยะยาว", "เงินพร้อมใช้"
2. แต่ละ Bucket ระบุบทบาทและเหตุผล
3. สัดส่วนรวมของทุก Bucket ต้องเท่ากับ 100%
4. จัดทำ matrix การกระจายสินทรัพย์จากแกนหลักเข้าสู่ Buckets

## Output JSON Schema
```json
{
  "purpose_buckets": [
    {
      "bucket_id": "b_growth",
      "name": "เติบโตระยะยาว",
      "role": "สะสมความมั่งคั่งผ่านสินทรัพย์เติบโต",
      "color_hex": "#2563EB",
      "target_percent": "60.00"
    },
    {
      "bucket_id": "b_income",
      "name": "รายได้สม่ำเสมอ",
      "role": "สร้างกระแสเงินสดเพื่อความมั่นคง",
      "color_hex": "#10B981",
      "target_percent": "40.00"
    }
  ],
  "allocation_mapping": [
    {
      "axis_allocation_id": "cat_1",
      "bucket_id": "b_growth",
      "portfolio_weight_percent": "60.00"
    },
    {
      "axis_allocation_id": "cat_2",
      "bucket_id": "b_income",
      "portfolio_weight_percent": "40.00"
    }
  ]
}
```
