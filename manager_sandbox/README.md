# 🛡️ Manager Sandbox (`HERMES_HOME`)

โฟลเดอร์สำหรับแยกพื้นที่ทำงานของ **Trinity AI Advisor (`hermes-agent`)** โดยเฉพาะ:
- `memory/`: Dual-Memory (`USER.md`, `MEMORY.md`, `DREAMS.md`) สำหรับเก็บโปรไฟล์นักลงทุนและประวัติการตัดสินใจ
- `skills/`: สกิลที่ Hermes พัฒนาขึ้นเอง โดยสกิลใหม่จะเข้าตรวจความปลอดภัยที่ `_quarantine/` ก่อน
- `scratch/`: พื้นที่ทดลองรันสคริปต์ Python คำนวณทางการเงินเฉพาะกิจ (ควบคุมด้วย `core/sandbox/ast_guard.py`)
- `audit/`: ประวัติการรันสคริปต์และวิวัฒนาการของระบบบันทึกในรูปแบบ JSONL

