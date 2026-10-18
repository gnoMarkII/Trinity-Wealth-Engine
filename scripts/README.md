# Scripts Directory

โฟลเดอร์นี้รวบรวมสคริปต์สำหรับการปฏิบัติการ (Operations), ไปป์ไลน์การคำนวณและประมวลผลข้อมูล (Pipelines), การตรวจสอบความถูกต้องทางสถาปัตยกรรม (Architectural Scanners) และเครื่องมืออำนวยความสะดวกในการพัฒนาของ Trinity Wealth Engine

---

## สคริปต์ประจำการหลัก (Active Scripts)

### 1. 🏛️ Architecture & Boundary Scanners (มี Unit / AST Test ควบคุม)
* `scan_vault_writers_r8.py` — AST scanner ตรวจสอบขอบเขตการเขียน Vault (Deny-by-default write boundary) ควบคุมโดย `tests/architecture/test_vault_write_boundaries.py`
* `scan_vault_writers_r9.py` — ตรวจสอบ Root construction boundaries และ deny-by-default writer inventory ควบคุมโดย `tests/architecture/test_r9_finalization.py`
* `apply_live_file_repairs_r3.py` — สคริปต์ซ่อมแซมและปรับโครงสร้าง Frontmatter/Metadata ของ Vault ควบคุมโดย `tests/tools/archivist/test_live_file_repair_r3.py`
* `audit_macro_page_data.py` — ตรวจสอบ AI lineage integrity และข้อมูล Macro page ควบคุมโดย `tests/tools/macro/test_ai_data_lineage.py`
* `migrate_investor_principles_v2.py` — ตรวจสอบและย้าย Investor Essence principles ควบคุมโดย `tests/unit/investor_essence/test_d9_migration_and_audit.py`

### 2. 📊 Macro Intelligence & Pipeline Tooling
* `run_daily_macro_strategy.py` — สคริปต์รันไปป์ไลน์วิเคราะห์กลยุทธ์ Macro ประจำวัน
* `recommit_macro_strategy.py` — รันและซิงค์ Macro strategy notes เข้าสู่ระบบ
* `verify_macro_data_pipeline.py` — ตรวจสอบความสมบูรณ์และสถานะของ Macro Data Pipeline
* `run_macro_acceptance.py` — Acceptance Suite ตรวจสอบการทำงานแบบ End-to-End ของ Macro
* `run_macro_notebooklm_acceptance.py` — Acceptance Suite ตรวจสอบการทำงานของ NotebookLM Dataset
* `audit_macro_notebooklm_export.py` — เครื่องมือตรวจสอบและสรุปผล NotebookLM Export

### 3. 📈 Sector Rotation & Task Scheduling
* `refresh_sector_rotation.py` — คำนวณและอัปเดตข้อมูล Sector Rotation ประจำวัน (EOD)
* `register_sector_rotation_task.ps1` — สคริปต์ PowerShell สำหรับลงทะเบียน Scheduled Task บน Windows เพื่อรัน `refresh_sector_rotation.py` ทุกวันอัตโนมัติ

### 4. 🛠️ Development & Schema Management
* `generate_golden_manifests.py` — เครื่องมืออัปเดต API Route Contract และ Database Schema Manifests (ดูรายละเอียดใน `AGENTS.md`)
* `export_openapi.py` — ส่งออก OpenAPI Specification สำหรับ Frontend Type Generation (`npm --prefix web run check:types`)
* `seed_test_data.py` — สำหรับ Seed ข้อมูล Portfolio & Transactions ตัวอย่างลงในฐานข้อมูล
* `create_negative_fixtures.py` — เครื่องมือสร้าง Negative Fixtures สำหรับการทดสอบ
* `generate_reference_mappings.py` — สร้างและแมปปิ้งความสัมพันธ์ของ Reference Entities
* `backfill_images.py` — ย้ายและจัดทำ Index รูปภาพในระบบความรู้
* `backfill_youtube_insight_metadata.py` — เติมเต็ม Metadata และ Insights จาก YouTube Transcripts
