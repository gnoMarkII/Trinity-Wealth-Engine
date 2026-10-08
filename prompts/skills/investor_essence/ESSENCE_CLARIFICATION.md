# ขั้นเติมข้อมูลแก่นแท้ (Clarification Questions)

หน้าที่: สร้างคำถามชี้แจงเพื่อเติมข้อมูลในประเด็นที่ยังไม่ชัดเจน (Unresolved Topics) โดยแยกต่างหากจากการสัมภาษณ์หลัก 10 ข้อ

## กติกา
1. คำถามต้องถามเจาะจงเฉพาะประเด็นที่ยังขาดความชัดเจน
2. ตัวเลือก 4 ข้อ (A, B, C, D) พร้อมสถานะไม่ชี้นำ
3. กำหนด is_clarification = true

## Output JSON Schema
```json
{
  "text": "ข้อความคำถามเพื่อชี้แจงเพิ่มเติม",
  "options": [
    {"key": "A", "text": "ตัวเลือก A"},
    {"key": "B", "text": "ตัวเลือก B"},
    {"key": "C", "text": "ตัวเลือก C"},
    {"key": "D", "text": "ตัวเลือก D"}
  ],
  "coverage_topics": ["หัวข้อที่ต้องการชี้แจง"],
  "evidence_type": "self_report",
  "is_clarification": true
}
```
