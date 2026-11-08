ผลตรวจซ้ำหลังการแก้ไข — 5 พฤศจิกายน 2026

ขอบเขต: ตรวจ working tree ปัจจุบันเทียบกับ 21 ประเด็นใน latest-code-review-plan.md รวมการแก้ไข API/DI/schema/tests และทดลองเส้นทางที่ชุดทดสอบเดิมยังไม่ครอบคลุม รอบนี้ไม่ได้แก้ implementation, schema, manifests หรือ dependencies

ข้อสรุป: แก้ได้หลายจุดและชุดทดสอบที่เกี่ยวข้องผ่านแล้ว แต่ยังไม่สามารถปิดงานได้ มีทั้งประเด็นเดิมที่แก้บางส่วนและข้อผิดพลาดที่พบเพิ่ม

ผลตรวจที่รันจบ

- Backend ที่เกี่ยวข้อง: 273 passed รวม architecture, OpenAPI/state DB contracts, PM review/thesis/analytics และ regression ของโค้ดที่แก้
- Frontend: 123 files / 621 tests passed
- TypeScript typecheck: passed
- git diff --check: passed
- สอง test ที่ล้มในรอบก่อนผ่านแล้ว
- ไม่ได้รัน full backend suite ทั้ง repository ในรอบนี้
- Diagnostic probes ใช้ memory, temporary directories และ in-memory SQLite; ไม่เขียนทดลองกับฐานข้อมูลหรือพอร์ตจริง
- HTTP probes ใช้ router จริงใน FastAPI app แยกและจำลอง require_session ให้คืน None ตรงกับ contract ปัจจุบันของ dependency; ไม่ได้แทน user ด้วย dict แบบที่ tests บางชุดทำ

ข้อผิดพลาดที่พบเพิ่มหรือเกิดชัดเจนหลังการแก้

N01. [P1] Variant Perception error paths คืน 500 เพราะไม่มี import HTTPException และ status
ตำแหน่ง: api/routers/equity/router_estimates.py:4, :25, :32, :45
หลักฐาน HTTP: ไม่ส่ง reported -> 500; reported=bad -> 500; ticker ที่ไม่มี estimate แต่ reported ถูกต้อง -> 500 ทั้งสามควรเป็น controlled validation/not-found response
แผน: import dependencies ที่ใช้ให้ครบ; ใช้ typed finite Decimal และ validation 422 ตามมาตรฐาน; เพิ่ม API tests สำหรับ missing/invalid/unknown และ NaN/Infinity
ตรวจรับ: ไม่มี NameError; missing/invalid เป็น 422 และไม่มีข้อมูลเป็น 404 ตาม contract ที่ตกลง

N02. [P1] Governance และ Catalyst review ใช้ user.get() แต่ authentication dependency คืน None
ตำแหน่ง: api/auth.py:106; api/routers/governance_router.py:36; api/routers/thesis/router_catalyst.py:31; api/routers/pm_review_router.py:82
หลักฐาน: require_session ตรวจ cookie แล้วไม่ได้ return user; valid route-task และ review catalyst ใน HTTP probe คืน 500 จาก None.get() ส่วน PM sign-off ตกไปเชื่อ req.signed_off_by; probe ส่ง FAKE_PM และได้ชื่อนี้ใน sign-off
แผน: แยก authentication guard กับ trusted identity/session provider ให้ชัด และใช้ provider ที่ถูกต้องทุกจุด; bind audit identity จาก session จริง; ทดสอบด้วย cookie/login flow จริงหรือ dependency ที่รักษา return contract เดิม
ตรวจรับ: login แล้วเรียกสอง endpoints ได้; unauthenticated ยัง 401; ชื่อผู้ลงนามไม่ถูก override ด้วย payload โดยไม่มี identity policy ที่ชัดเจน

N03. [P2] Synthesis response มี claim/checksum คนละฉบับกับที่บันทึก
ตำแหน่ง: application/thesis/thesis_synthesis_service.py:297, :311, :335
หลักฐานที่ทดลอง: synthesize dry_run=False แล้วเอา response.claim.checksum_sha256 เทียบ get_active_thesis(ticker).checksum_sha256 ได้ False เพราะสร้าง claim_record เองหนึ่งครั้ง แต่เรียก record_new_thesis เพื่อสร้างและบันทึกอีกครั้งด้วย timestamp/normalization คนละชุด
แผน: ใช้ record ที่ record_new_thesis คืนกลับมาเป็น response โดยตรง; dry-run ใช้ builder เดียวกับ actual write และแยกสถานะให้ชัด
ตรวจรับ: response claim เท่ากับ claim ที่อ่านกลับได้ทุก field รวม as_of_utc/checksum

N04. [P1] Sign-off ที่ปรับตรวจ hash ยังทำ normal review/retry ไม่สำเร็จ
ตำแหน่ง: application/manager/pm_review_service.py:346, :569, :590, :599; tools/thesis/thesis_sqlite_dao.py (insert_signoff); api/db/connection.py (idx_pm_signoff_port_date)
หลักฐาน:
- packet ที่มี breach เดิม build สองครั้งได้ hash ต่างกัน เพราะ scan สร้าง exception UUID/time ใหม่; sign-off ด้วย hash จาก GET ล่าสุดถูกปฏิเสธ Packet SHA-256 mismatch ทั้งที่ข้อมูลตลาดไม่เปลี่ยน
- ลงนามครั้งแรก HTTP 200; retry packet เดิม HTTP 422 เพราะ idempotent branch คืน raw repo dict ที่ไม่มี status ตาม PMSignoffResponseDTO
- เมื่อ state เปลี่ยน packet กลับ REVIEW_REQUIRED ถูกต้อง แต่ลงนามรอบใหม่วัน/พอร์ตเดิมเกิด UNIQUE constraint failed: pm_review_signoffs.portfolio_id, pm_review_signoffs.review_date ซึ่ง router map เป็น 500
แผน: persist immutable packet snapshot และ stable exception identities; read-only GET ไม่สร้าง events ใหม่; คืน response schema เดียวกันทั้ง first/retry; ออกแบบ sign-off history รองรับ packet revision โดยรักษา immutable audit records
ตรวจรับ: unchanged packet hash คงที่และลงนามได้; retry ได้ 200 พร้อม schema เดิม; changed packet ต้อง review ใหม่และลงนามใหม่ได้โดยไม่ลบ audit เก่า
ถ้าต้องเปลี่ยน schema เพื่อ version packet/sign-off ต้องขออนุมัติตาม AGENTS.md ก่อนลงมือ

สถานะทั้ง 21 ประเด็นจากแผนเดิม

| ข้อ | ผลตรวจซ้ำ | สิ่งที่แก้แล้ว / สิ่งที่ยังต้องทำ |
|---|---|---|
| R01 | บางส่วน | ไม่คืน bridge ให้ทุก ticker แล้ว และบังคับส่ง reported แต่ค่าตัวอย่างถูกย้ายไป _AUTHORITATIVE_EARNINGS_DISCLOSURES ใน router พร้อมวันที่เดิม ไม่มี source ingestion/provenance; estimate service ยัง seed GOOGL เป็น APPROVED/DOCUMENTED และ error paths มี N01 |
| R02 | บางส่วน | เอาค่า NAV/กำไรตัวอย่างเดิมออก แต่แทน missing ด้วย zero และ beta=1.0 ทั้งที่ observations=0; ไม่มี benchmark/integrity providers; macro reader อ่าน shape ผิดตามรายละเอียดด้านล่าง |
| R03 | บางส่วน | ปฏิเสธ hash/sample ID ปลอมแล้ว แต่ omit hash และส่ง samples=[] ยังลงนามได้; ยังไม่บังคับครบ required samples/acknowledge transaction; identity ติด N02 และ review/retry ติด N04 |
| R04 | แก้ข้อเดิมแล้ว | signoff repo/schema/request/UI แยก portfolio_id และมี test isolation ผ่าน; ยังต้องจัด packet version/checksum identity ให้ครบใน R03/N04 |
| R05 | แก้ข้อเดิมแล้ว | service/store ใช้ validate_ticker; diagnostic parent traversal ถูกปฏิเสธ |
| R06 | บางส่วน | malformed JSON หยุดและมี backup แล้ว; checksum mismatch ยังถูก drop ออกจาก list ทำให้ append/record ใหม่เขียนทับประวัติส่วนเสียได้ |
| R07 | บางส่วน | FileLock ป้องกัน read-modify-write พร้อมกันแล้ว; stale parent ยังรับได้ และ Markdown sync อยู่นอก lock จึงยังเสี่ยง projection ย้อน version |
| R08 | บางส่วน | เอา default growth/margin/stop-loss เมื่อไม่มีข้อมูลบางส่วนออก และแยก fcf_margin ถูกต้องขึ้น; ยัง self-validate ค่าจาก dictionary เดียวกันแล้วติด DOCUMENTED, custom_stop_loss ไม่ลด tier, ไม่มี primary locator จริง และ no-data synthesis ยังสร้าง positive thesis/coverage ได้ |
| R09 | แก้ข้อเดิมแล้ว | market TH ให้ THB และส่ง portfolio_id ลง batch fallback แล้ว; diagnostic CPALL ได้ THB |
| R10 | ยังเหลือ | production thesis provider ยังไม่ต่อ MarketObservationPort; missing metric ถูกข้าม; GET สร้าง UUID/events ใหม่และไม่แยก unknown จาก healthy |
| R11 | บางส่วน | ตรวจ ticker/period/metric/unit/duplicate และไม่ append unbalanced แล้ว; filter APPROVED และส่ง unit แล้ว แต่ไม่ตรวจ estimate_type ข้าม parent, เลือกแถวสุดท้ายตาม insertion แทนเวลา, cutoff ยัง optional/เปรียบเทียบ string และ route ไม่ส่ง cutoff |
| R12 | บางส่วน | filter future captures เมื่อส่ง cutoff แล้ว และ ID ไม่อิง ordinal; fingerprint ยังไม่รวม new date/classification/content hash จึงยัง reuse review กับ event เปลี่ยนในวันเดียวกันได้ |
| R13 | บางส่วน | routing โหลด artifact กลับตอน startup แล้ว; catalyst/estimate/routine ยัง process-local และไม่มี production ingestion/worker heartbeat wiring ครบ; routing เองยัง overwrite task ID เดิมและใช้ fixed temp |
| R14 | ยังเหลือ | units=0 ยังถูกแทนเป็น 1; mixed-currency fallback/float math และ eligible count จาก quarantined ยังเดิม; missing-price metadata ไม่ครบ |
| R15 | ยังเหลือ | top daily mover ยังใช้ lifetime unrealized P&L; daily P&L ยังต่าง NAV สองแถวโดยไม่ปรับ cash flows/ยืนยันคนละวัน |
| R16 | บางส่วน | offset-aware datetime comparison ใช้ได้แล้ว; invalid timestamp ยัง fallback เป็น string compare และ normalization ตัด fractional seconds; thesis as-of ยังเลือก max version แทน effective time |
| R17 | ยังไม่ผ่าน | เพิ่มเช็ค SQLite header และ hash ไฟล์จริง แต่ยังไม่ query/ตรวจ DB integrity และไม่เทียบ expected hashes กับ state inputs; fake header-only DB ยังได้ 10/10 READY |
| R18 | แก้ domain แล้ว | MNPI shape ไป quarantine และ invalid shape เป็น 422; valid authenticated HTTP path ยังใช้จริงไม่ได้เพราะ N02 |
| R19 | บางส่วน | EarningsCallTab reset ticker และ cancel stale response แล้ว; StructuredThesisCard/MorningDelta ยังไม่มี request identity/cancellation; MorningDelta ยังแสดงและอนุมัติ packet เก่าระหว่างโหลดหรือ failed refresh |
| R20 | แก้ข้อเดิมแล้ว | Hermes ใช้ list_active_theses และลบ get_position_review_service ที่นิยามซ้ำแล้ว |
| R21 | gate ผ่าน / งานบางส่วนค้าง | architecture/OpenAPI tests ผ่านและ storage contract เพิ่ม owner/profile แล้ว; application writers ยังใช้ write_text/fixed temp เอง, strict domain validation ยังไม่ครบ และ tests ยังไม่จับ N01/N02/N04 |

หลักฐานเพิ่มเติมของประเด็นเดิมที่ยังเหลือ

A. [P1 / R06] Checksum mismatch ยังนำไปสู่การสูญหายของ history
ตำแหน่ง: tools/thesis/thesis_file_store.py:116
ทดลองแก้ statement โดยไม่แก้ checksum แล้ว load_claims คืน []; record_new_thesis ทำงานต่อได้ และไฟล์ ledger ใหม่ไม่มี corrupted record เดิม คำว่า Quarantining ใน log ไม่ได้สร้าง durable quarantine สำหรับกรณี checksum เสีย
แก้: ถ้า record ใด integrity ผิด ต้องหยุดอ่านเป็น usable ledger และหยุด append; preserve original bytes/หลักฐานพร้อม explicit integrity error ไม่กรองทิ้งเงียบ
ตรวจรับ: ทั้ง malformed JSON และ checksum mismatch ห้ามเขียน ledger ใหม่ทับ

B. [P1 / R17] Readiness ยังเป็น false assurance
ตำแหน่ง: application/resilience/readiness_audit_service.py:150, :259, :332, :353
temp repository มี .gitignore, DB ที่มีเพียง 16-byte SQLite header และ storage_contract.json={} ได้ 10 PASS / READY_FOR_NAMED_STAGE
แก้: ต่อ read-only DB integrity checker จริง; ตรวจ manifest entries/expected hashes กับ authoritative inputs; ตรวจ packet/human gate ด้วย behavior tests และ execution evidence; missing evidence ต้อง fail/unknown
ตรวจรับ: header-only/truncated/corrupt DB, manifest ไม่มี entries, tampered state, skipped human gate ต้องไม่ให้ READY

C. [P1 / R02] Macro provider ที่ต่อใหม่อ่าน schema ผิด
ตำแหน่ง: api/dependencies.py:769; application/macro/service.py:34; api/schemas/macro.py:158; tools/macro/dashboard.py:112
dashboard จริงมี dashboard_indicators และ overall_regime; _macro_reader อ่าน indicators/strategy แทน จึงคืน indicators=None และ regime=Neutral; PM service คง zero/Low ต่อไป
แก้: adapter map actual dashboard/observable schema ตาม indicator_id/unit/as-of; แยก missing จาก zero และไม่สร้าง low-risk conclusion; ต่อ benchmark/integrity readers จริง
ตรวจรับ: ใช้ fixture รูปแบบ production ที่มีค่าต่างจาก zero และ regime ต่างจาก Neutral ต้องออกตรงทุก field

D. [P1 / R14-R15] Exposure และ Daily Delta ยังผิดเหมือนเดิม
ตำแหน่ง: application/manager/pm_review_service.py:188, :210, :267; api/routers/portfolio/router_advisory.py:96
diagnostic units=0/price=100 ยังคำนวณ exposure=100.0; unrealized_pnl_percent=80 ยังแสดงเป็น Top Daily PnL Mover; endpoint eligible_positions_count ยังเท่าจำนวน quarantine
แก้: ใช้ actual signed units, active status, approved same-currency prices/FX และ Decimal; สร้าง analytics result ที่มี eligible/missing/count ตรงกัน; daily movement ใช้ dated observations และปรับ cash flows
ตรวจรับ: zero units ไม่เพิ่ม exposure, unavailable prices ไม่ถูกเดา, daily mover ไม่อ้าง lifetime return

E. [P1 / R08] Evidence ยังตรวจตัวเอง
ตำแหน่ง: application/thesis/thesis_synthesis_service.py:246, :282
claimed_value สร้างจาก primary_facts แล้วส่ง dictionary เดิมเป็นสิ่งที่ใช้ตรวจ จึงผ่านโดยโครงสร้าง พร้อม source_locator ที่เป็นเพียงชื่อ SEC/Financial Disclosure ที่สร้างขึ้น; diagnostic ใช้ sidecar ไม่มี source metadata และ custom stop-loss ยังได้ DOCUMENTED
แก้: แยก facts ที่ตรวจย้อนเอกสารได้จาก derived policy; claim inference ต้องมี tier ที่เหมาะสม; ไม่ใช้ string locator ทั่วไปแทน source record; no-data ต้อง insufficient evidence ไม่เพิ่ม healthy coverage
ตรวจรับ: ไม่มี source record หรือมี custom derived threshold ต้องไม่กลายเป็น documented fact ทั้ง claim

F. [P1 / R12] Date change ยัง reuse exception ID
ตำแหน่ง: core/thesis/catalyst_queue.py:163; application/thesis/catalyst_service.py (merge reviewed exceptions)
evaluate catalyst เดิมวันเดียวกันก่อน/หลังเปลี่ยน expected date ได้ exception_id เดิม เพราะ key มีเพียง entity/catalyst/source/day; reviewed exception จึงกลบ DATE_CHANGED ได้
แก้: ใช้ immutable event fingerprint ที่รวม material event fields/content revision และ merge เฉพาะ fingerprint ตรงกัน
ตรวจรับ: เปลี่ยน date/classification/content ใน source ID เดิมวันเดียวกัน ต้องเกิด event ใหม่ที่ REVIEW_REQUIRED

G. [P2 / R07-R16] Parent และ cutoff validation ยังไม่เข้มพอ
ตำแหน่ง: tools/thesis/thesis_file_store.py:88; core/provenance.py:43; application/thesis/thesis_service.py:126
สร้าง v1, v2 แล้ว revise โดยอ้าง v1 ยังได้ v3; is_available_as_of('2026-01','2026-02') คืน True แม้ทั้งคู่ไม่ใช่ timestamp ที่ parser ยอมรับ
แก้: optimistic parent ต้องเป็น latest eligible version; reject invalid timestamps; รักษา precision; เรียง effective/available times จริง
ตรวจรับ: stale parent ถูกแจ้ง conflict; malformed timestamp ไม่ได้รับอนุญาตผ่าน cutoff

ลำดับแก้รอบต่อไป

1. แก้ N01/N02 ให้ error paths และ authenticated production routes ใช้งานได้ พร้อม tests ที่ไม่เปลี่ยน auth contract
2. แก้ R03/R10/N04 เป็น flow เดียว: immutable packets, stable events, required approval evidence, retry และ versioned sign-off
3. ปิด R06/R07 ก่อนเขียน thesis เพิ่ม: corruption ต้องหยุด, parent ต้องไม่ stale, projection ต้องไม่ย้อน version
4. ปิด R01/R02/R08/R17: ย้าย fixtures ออกจาก production, map actual macro schema, provenance จริง และตรวจ readiness จาก state/evidence จริง
5. ปิด R11-R16: approved temporal ledger, event fingerprint, durable records, exposure/daily math/cutoff
6. แก้ N03 และ loaders ที่เหลือใน R19; เพิ่ม request-race/retry/negative/restart tests ตาม probes ข้างต้น
7. รัน backend/frontend/typecheck อีกครั้งหลัง implementation และรัน full backend suite ก่อนปิดงาน; อย่าปิดประเด็นด้วยการเปลี่ยน baseline หรือแก้ expected output ให้รับ bug

ไม่ต้องเปลี่ยน schema/dependencies/manifests เพื่อทำ review นี้ หาก implementation รอบถัดไปจำเป็นต้องเปลี่ยนส่วนนั้น ให้ขออนุมัติตาม AGENTS.md ก่อน
