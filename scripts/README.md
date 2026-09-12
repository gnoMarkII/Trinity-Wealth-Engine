# Scripts Directory

โฟลเดอร์นี้รวบรวมสคริปต์สำหรับการปฏิบัติการ ตรวจสอบความพร้อมทางสถาปัตยกรรม (Preflight & Acceptance) และบำรุงรักษา Obsidian Vault V2

## โครงสร้างสคริปต์ปัจจุบัน (Active Scripts)

### 1. การตรวจสอบและทดสอบประจำการหลัก (R9 / R10 Preflight & Acceptance)
* `run_vault_r10_preflight.py` — ตรวจสอบสถานะความพร้อมของ Vault ตามมาตรฐาน R10 (Link Graph, Reader, Cleanup)
* `run_vault_r10_acceptance.py` — Acceptance Suite สำหรับ Milestone R10
* `run_vault_r9_preflight.py` — Guardrail ตรวจสอบ Architecture Invariants ของ R9 (Deny-by-default, SQLite isolation)
* `run_vault_r9_acceptance.py` — Acceptance Suite สำหรับ Milestone R9
* `scan_vault_writers_r9.py` — Deny-by-default Writer Inventory Scanner ตรวจจับ Direct File I/O ที่ผิดกฎสถาปัตยกรรม
* `observe_vault_r9.py` — เฝ้าระวังความเสถียรของ Vault และสรุปสถานะสุขภาพ
* `run_vault_r9_cycle.py` — รัน Acceptance Cycle เต็มรูปแบบ

### 2. การจัดการ Maintenance และ Rebuild
* `rebuild_vault_derived_r9.py` — Rebuild Catalog Database และ Vector Manifests
* `migrate_vault_runtime_layout_r9.py` — จัดการและย้าย Runtime ไว้นอก Vault
* `backup_vault_platform_r9.py` / `restore_vault_platform_r9.py` — สำรองและกู้คืนโครงสร้าง Vault Platform
* `rehearse_vault_r9_runbook.py` — ซักซ้อมกระบวนการ Runbook เพื่อความปลอดภัย

### 3. ชุดสคริปต์ R10 Tooling
* `build_concepts_cleanup_r10.py` — สร้าง Change Set สำหรับคลีนโน้ต Concept ที่ไม่ได้ใช้
* `apply_concepts_cleanup_r10.py` — ดำเนินการย้ายโน้ต Concept ไปยัง Revisions / Archive อย่างปลอดภัย
* `audit_concepts_r10.py` — ตรวจสอบและรายงานสถานะ Concept Notes
* `rollback_concepts_cleanup_r10.py` — กู้คืนโน้ต Concept ที่เคยคลีนกลับคืนสภาพเดิม

### 4. โฟลเดอร์ Archive (Historical Milestone Scripts)
* [scripts/archive/](file:///c:/ChinoDoc/Projects/Claude/invest-agents/scripts/archive/) — รวบรวมสคริปต์ประวัติศาสตร์ในอดีต (R4 – R8) ที่ทำงานเสร็จสิ้นแล้ว เก็บรักษาไว้เป็น Audit Trail โดยมีเอกสารระบุรายละเอียดใน [archive/README.md](file:///c:/ChinoDoc/Projects/Claude/invest-agents/scripts/archive/README.md)
