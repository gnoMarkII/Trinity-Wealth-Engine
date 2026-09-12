# Archived Milestone Scripts (R4 – R8)

โฟลเดอร์นี้รวบรวมสคริปต์ที่ใช้งานเสร็จสิ้นตาม Milestone ในอดีต (R4, R5, R6, R7, R8) เก็บรักษาไว้เป็น Audit Trail และหลักฐานทางสถาปัตยกรรม (Historical Reference) โดยไม่ถูกเรียกใช้ในการทำงานปกติของระบบปัจจุบัน (R9 / R10)

## รายการสคริปต์ที่จัดเก็บใน Archive

### Milestone R4 (Vault V2 Baseline & Quarantine)
* `backfill_vault_v2_contract_r4.py` — สคริปต์ปรับปรุง Metadata Contract ยุคเริ่มต้นของ Vault V2
* `observe_vault_stability_r4.py` — สคริปต์เฝ้าระวังความเสถียรของ Vault ในช่วง R4
* `quarantine_legacy_runtime_r4.py` — สคริปต์แยกไฟล์ Runtime เก่าออกจาก Vault
* `run_vault_v2_acceptance_r4.py` — Acceptance Runner ของ Milestone R4
* `verify_snapshot_restore_r4.py` — ทดสอบการ Snapshot และ Restore โครงสร้าง Vault R4

### Milestone R5 (Catalog & Vector Generation)
* `assemble_r5_evidence.py` — รวบรวมหลักฐานความพร้อมก่อนปิด Milestone R5
* `backfill_r5_provenance_contract.py` — Backfill Contract สำหรับ Provenance
* `benchmark_retrieval_r5.py` — วัดประสิทธิภาพการค้นคืนข้อมูล (Retrieval Benchmark)
* `observe_vault_stability_r5.py` — สคริปต์เฝ้าระวังความเสถียรของ Vault ในช่วง R5
* `quarantine_legacy_catalog_r5.py` — ย้าย Catalog เก่าเข้าสู่ Quarantine
* `quarantine_legacy_vector_manifests_r5.py` — ย้าย Vector Manifest เก่าเข้าสู่ Quarantine
* `quarantine_r5_cleanup.py` — ทำความสะอาดไฟล์ตกค้างจาก R5 Quarantine
* `run_vault_r5_preflight.py` — Preflight Runner ประจำ Milestone R5
* `run_vault_v2_acceptance_r5.py` — Acceptance Runner ประจำ Milestone R5

### Milestone R6 (Multi-App Compatibility & Validation)
* `assemble_multi_app_r6_evidence.py` — รวบรวมหลักฐานความพร้อมของระบบ Multi-App R6
* `run_vault_v2_acceptance_r6.py` — Acceptance Runner ประจำ Milestone R6

### Milestone R7 (Vault Link Audit & Writer Scanners)
* `observe_vault_r7.py` — สคริปต์เฝ้าระวังความสมบูรณ์ของ Vault R7
* `run_vault_r7_preflight.py` — Preflight Runner ประจำ Milestone R7
* `scan_vault_writers_r7.py` — Writer Inventory Scanner ยุค R7 (ถูกแทนที่ด้วย `scan_vault_writers_r9.py`)

### Milestone R8 (Transaction Source & Evidence)
* `assemble_vault_r8_evidence.py` — รวบรวมหลักฐานความพร้อมของ R8
* `run_vault_r8_preflight.py` — Preflight Runner ประจำ Milestone R8 (ถูกแทนที่ด้วย `run_vault_r9_preflight.py`)

---
*หมายเหตุ*: สคริปต์ปัจจุบันที่ใช้งานในการทดสอบและตรวจสอบความปลอดภัยของ Vault ประจำการอยู่ที่โฟลเดอร์หลัก `scripts/` เช่น `run_vault_r9_preflight.py`, `scan_vault_writers_r9.py`, `run_vault_r9_acceptance.py`, และชุดสคริปต์ R10
