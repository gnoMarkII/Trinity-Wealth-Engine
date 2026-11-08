ผลตรวจโค้ดรอบที่ 3 — 5 พฤศจิกายน 2026

ขอบเขต: working tree ปัจจุบันเทียบกับ review/re-review-results.md และ review/detailed-remediation-plan.md รอบนี้เป็น review และ diagnostic verification ไม่มีการแก้ production code, schema, dependencies หรือ manifests

ข้อสรุป: N01–N03 และข้อผิดพลาดเดิมหลายกรณีแก้แล้ว แต่ยังมีปัญหาที่ต้องแก้ก่อนปิดงาน โดยเฉพาะการอัปเกรดฐานข้อมูลเก่า, immutable sign-off history, integrity error propagation และ readiness ที่ยังให้ผลผ่านผิด

ผลทดสอบที่รันจบ

- Backend ชุดหลักที่เกี่ยวข้อง: 274 passed / 1 failed
- Backend auth, manager-chat และ UnitOfWork เพิ่มเติม: 33 passed
- รวม backend ที่รันจบ: 307 passed / 1 failed
- Frontend: 123 test files / 621 tests passed
- TypeScript typecheck: passed
- git diff --check: passed
- ไม่ได้รัน full backend suite ทั้ง repository ในรอบนี้
- Test ที่ล้ม: tests/api/test_pm_review_router.py::test_sign_off_morning_review_endpoint ที่ line 62 คาด signed_off_by=CHINO_PM จาก payload แต่ implementation ใหม่คืน session:<sid> จาก authenticated cookie
- ข้อนี้ควรแก้ test ให้ตรวจ trusted session identity ตาม behavior ที่ต้องการ ไม่ควรถอย implementation กลับไปเชื่อชื่อจาก payload เพื่อให้ test ผ่าน

สิ่งที่ยืนยันว่าแก้แล้ว

1. N01: Variant Perception ที่ไม่ส่ง reported, ส่งข้อความ, NaN หรือ Infinity คืน 422; unknown estimate คืน 404; ไม่มี NameError/500 แบบเดิม
2. N02: ทดสอบ login จริงใน isolated FastAPI app แล้ว Governance และ Catalyst review คืน 200; PM sign-off ใช้ session identity และไม่ใช้ชื่อปลอมจาก payload
3. N03: synthesis dry_run=False คืน claim ตรงกับ object ที่อ่านกลับจาก thesis store ทุก field รวม checksum
4. R06 กรณีเดิม: แก้ statement โดยไม่แก้ checksumแล้ว append ใหม่ถูกปฏิเสธ และ original bytes ยังอยู่ ไม่ถูกเขียนทับเป็น ledger ใหม่
5. R07 กรณีเดิม: stale parent ถูกปฏิเสธ และ Markdown sync ย้ายอยู่ภายใน FileLock แล้ว
6. N04 บางส่วน: unchanged breach ได้ stable hash ในกรณีที่ทดสอบและลงนามด้วย hash จาก packet ได้; retry ธรรมดาคืน 200 แทน 422 แบบเดิม
7. R12 บางส่วน: เปลี่ยน expected date ของ catalyst ใน source ID เดิมวันเดียวกันได้ exception ID ใหม่แล้ว
8. regression suites ของ auth/manager/UoW และหลายกรณี portfolio isolation/ticker validation/currency ยังผ่าน

ปัญหาที่ต้องแก้ก่อนปิดงาน

G01. [P1] Schema ใหม่อัปเกรดฐานข้อมูลเก่าไม่ครบ และอาจทำให้ request ที่เปิด DB ล้ม

ไฟล์: api/db/connection.py:439, :445, :540
สาเหตุ: _SCHEMA สร้าง unique index บน portfolio_id ก่อนมี migration เพิ่ม column นี้ให้ pm_review_signoffs รุ่นเก่า และโค้ดปัจจุบันไม่มี migration/drop legacy date-only index ที่เคยมีในรอบก่อน

หลักฐานจาก in-memory SQLite สองรูปแบบ:
- ตารางเก่าที่ไม่มี portfolio_id -> init_schema ล้มด้วย OperationalError: no such column: portfolio_id
- ตารางที่มี portfolio_id แต่ยังมี idx_pm_signoff_date UNIQUE(review_date) -> portfolio A ลงนามได้ แต่ portfolio B วันเดียวกันล้มด้วย UNIQUE constraint failed: pm_review_signoffs.review_date

ผลกระทบ: fresh DB tests ผ่านแต่ existing DB ใช้งานไม่ได้หรือเสีย portfolio isolation ตาม constraint เก่า
แผนแก้: ทำ versioned migration ที่เพิ่ม/backfill column ก่อนสร้าง index ใหม่และจัดการ legacy indexes; รักษา rows เก่า; ทดสอบทั้ง pre-portfolio และ intermediate schemas
ตรวจรับ: upgrade ซ้ำได้, row counts/history คงเดิม, A/B ลงนามวันเดียวกันได้ และ rollback/recovery ของ migration มีหลักฐาน
อ้างแผน: WP05/WP11; ต้องอนุมัติ schema/migration ตาม Gate DB-A ก่อนลงมือ

G02. [P1] แก้ re-sign constraint ด้วย UPSERT แต่ลบประวัติ approval เดิม

ไฟล์: tools/thesis/thesis_sqlite_dao.py:223, :241
สาเหตุ: ON CONFLICT(portfolio_id, review_date) DO UPDATE เปลี่ยน signoff_id, signer, timestamp, notes และ packet hash ของ row เดิม

หลักฐาน: ลงนาม packet แรกแล้วเปลี่ยน macro state และลงนามอีกครั้งในวันเดียวกันสำเร็จ แต่ history เหลือ 1 row; signoff_id แรกไม่อยู่แล้ว และ row ปัจจุบันเป็น approval ครั้งที่สอง

ผลกระทบ: error เดิมหาย แต่ immutable audit trail หายไปด้วย ไม่สามารถตรวจว่าผู้ใช้เคยอนุมัติ packet ฉบับแรกอย่างไร
แผนแก้: packet/approval revision model แบบ append-only; unique ต่อ packet identity; retry คืน approval เดิม; state ใหม่สร้าง approval ใหม่; history เก็บทั้งสอง
ตรวจรับ: same-packet retry มี approval เดียว; new packet มี record ใหม่; old ID/hash/notes/identity ยังอ่านได้
อ้างแผน: WP05/WP08/WP11; ห้ามแก้ด้วย overwrite/replace อีกครั้ง

G03. [P1] Thesis store ปฏิเสธข้อมูลเสียแล้ว แต่ PM packet กลืน error และยังอนุมัติได้

ไฟล์: application/manager/pm_review_service.py:346, :352, :375, :558
สาเหตุ: build_morning_packet catch error จาก watchdog/coverage แล้วเก็บเป็น warning log; payload ยังคง exceptions=[] และ quarantine=[] โดยไม่มี blocked/unknown state

หลักฐาน: มี portfolio และ holding จริงใน fixture แต่ thesis checksum เสีย:
- store เองปฏิเสธการอ่าน/เขียนถูกต้อง
- PM packet คืน exceptions=0 และ quarantined=0
- sign-off ของ packet นั้นสำเร็จเป็น SIGNED_OFF

ผลกระทบ: มาตรการ integrity ที่เพิ่มใน store ถูก bypass ด้วย aggregation layer; “อ่านไม่ได้” กลายเป็น “ไม่มีข้อยกเว้น”
แผนแก้: propagate typed integrity/data-quality error; packet BLOCKED หรือ required evaluation UNKNOWN; sign-off ต้องตรวจ required gates; เก็บ incident reference ใน packet
ตรวจรับ: corrupt thesis/failed observer/failed integrity checker ไม่แสดง healthy และลงนามผ่านไม่ได้
อ้างแผน: WP02/WP03/WP05/WP09

G04. [P1] Required evidence ยังข้ามได้ และ retry ส่งคืน verified samples ที่ไม่เคยบันทึก

ไฟล์: application/manager/pm_review_service.py:574, :580, :587–593
สาเหตุ:
- hash ยัง optional และ verified_sample_ids=[] ไม่ถูกตรวจว่า required samples ครบ
- idempotent branch เติม verified_sample_ids จาก request ใหม่เมื่อ repo row ไม่มี field นี้

หลักฐาน: ลงนามครั้งแรกด้วย verified_sample_ids=[] แล้ว retry packet เดิมพร้อม valid sample ID:
- ได้ signoff_id เดิม แต่ response บอกว่า sample นั้น verified แล้ว
- stored notes ยัง None และไม่มี sample evidence นั้นใน durable record
- ตัวอย่างใน fixture: response มี SMP-MOVER-N/A ขณะที่ stored approval ไม่มี verified-samples metadata

ผลกระทบ: response อ้างหลักฐาน approval ที่ไม่ได้บันทึกจริง และ retry เปลี่ยนความหมายของ record เดิมได้
แผนแก้: required hash/sample/exception IDs ตาม immutable packet policy; เก็บ structured evidence; idempotency ตรวจ canonical request hash และคืน stored result เท่านั้น
ตรวจรับ: omit hash/required samples ถูกปฏิเสธ; retry payload เดิมคืนข้อมูลเดิม; payload เปลี่ยนต้อง conflict หรือเป็น explicit review ใหม่
อ้างแผน: WP05

G05. [P1] Stable exception ID ยังทำให้ observed values ใหม่ไม่ถูกบันทึก

ไฟล์: core/thesis/evaluator.py:70–72; application/thesis/thesis_service.py:351, :427–435; tools/thesis/thesis_sqlite_dao.py:146
สาเหตุ: fingerprint มีเพียง ticker/claim/rule/day ไม่รวม source revision หรือ observed fact และ DAO ยัง INSERT อย่างเดียว; duplicate insert ถูก log แล้วข้าม

หลักฐาน: rule เดิมวันเดียวกัน scan ครั้งแรก observed_value=90 USD แล้วครั้งที่สอง=80 USD:
- exception ID เหมือนกัน
- response ครั้งที่สองแสดง 80 USD
- durable exception ยังเก็บ 90 USD

ผลกระทบ: UI/packet กับหลักฐานใน DB ไม่ตรงกัน และ material change อาจ reuse acknowledgement ของ state เก่า
แผนแก้: แยก logical breach กับ immutable event/evaluation revisions; fingerprint รวม material evidence/source revision; duplicate ที่เหมือนเดิม idempotent แต่ value/source ใหม่ต้อง persist ได้
ตรวจรับ: unchanged facts ไม่เพิ่ม event ซ้ำ; changed facts มี revision/evidence ใหม่ที่อ่านกลับตรง response; acknowledgement ผูก exact revision
อ้างแผน: WP03/WP05/WP08

G06. [P1] Readiness ยังให้ 10/10 READY กับไฟล์ที่ SQLite อ่านไม่ได้

ไฟล์: application/resilience/readiness_audit_service.py:149–157, :279–282
สิ่งที่แก้แล้ว: เพิ่มขั้นต่ำขนาด DB และตรวจ structured manifest บางกรณี; empty manifest แบบที่ทดสอบคืน STOP แล้ว
ช่องที่ยังเหลือ:
- DB ที่มี SQLite header และขนาดมากกว่าขั้นต่ำยังถูกถือว่า queryable โดยไม่ได้ query
- nonempty dict ที่ไม่ใช่ frozen state manifest ถูกให้ PASS ว่าเป็น storage contract integrity

หลักฐาน: temp DB มีเพียง header + padding และ storage_contract.json เป็น {arbitrary:true}
- SQLite PRAGMA quick_check ล้มว่า file is not a database
- readiness คืน 10 PASS / READY_FOR_NAMED_STAGE

แผนแก้: read-only DB integrity checker จริง; ต้องตรวจ required schema/query/quick_check; manifest ต้องตรง schema/required authoritative inputs และ expected digests; arbitrary dict ไม่เป็นหลักฐาน integrity
ตรวจรับ: padded header-only/corrupt DB หรือ arbitrary contract ไม่ READY; valid state และหลักฐานจริงเท่านั้นที่ผ่าน
อ้างแผน: WP10

G07. [P1] Macro reader อ่าน shape ใหม่แล้ว แต่ยังไม่ map canonical observable IDs

ไฟล์: api/dependencies.py:770, :791–793; tests/fixtures/macro/field-lineage.json
Canonical IDs ใน pipeline คือ obs_ust_2y_yield, obs_ust_10y_yield และ obs_ofr_financial_stress แต่ reader มองหา yield_spread_10y_2y_bps/yield_spread_10y_2y และ ofr_stress_index/ofr_financial_stress_index

หลักฐาน: fixture ที่ใช้ canonical IDs พร้อมค่า yields/stress จริง:
- overall_regime ถูกอ่านเป็น Risk Off ได้แล้ว
- yield spread และ stress ยังเป็น None
- PM service เก็บ default zero และ Low; beta default ยัง 1.0 ทั้งที่ observations=0
- expected spread ใน diagnostic คำนวณด้วย Decimal จาก fixture ไม่ใช่ตัวเลขตลาดจริง

แผนแก้: adapter map actual IDs/units/clocks; derive spread ใน Python จาก eligible yields เมื่อ source ไม่มี derived field; แยก unavailable จาก zero; ต่อ benchmark/integrity providers จริง
ตรวจรับ: fixture จาก production data contract ให้ค่าครบและถูกหน่วย; ข้อมูลขาดแสดง unavailable ไม่เป็น low-risk/neutral conclusion
อ้างแผน: WP03/WP04/WP09

G08. [P1] ยังมี synthetic/unsupported evidence ใน production path

ไฟล์: api/routers/equity/router_earnings_call.py (_AUTHORITATIVE_EARNINGS_DISCLOSURES); application/equity/estimate_service.py:36; application/thesis/thesis_synthesis_service.py:246–285
หลักฐานจากโค้ดและ diagnostic:
- earnings bridge ยังใช้ constants ที่เรียกว่า authoritative; estimate service ยัง seed GOOGL APPROVED/DOCUMENTED
- synthesis ตรวจ claimed value กับ primary_facts ที่สร้างจาก dictionary เดียวกันพร้อม generic source locator
- sidecar fixture ไม่มี primary source metadata และส่ง custom_stop_loss ยังได้ DOCUMENTED

แผนแก้: แยก test/demo fixtures; อ่าน captured source records พร้อม locator/hash/cutoff; แยก documented facts, tested math และ inferred/policy thresholds; ไม่เพิ่ม monitored coverage ด้วย no-data thesis
ตรวจรับ: ไม่มี source record ไม่กลายเป็น verified/documented และไม่มีค่าตัวอย่างเป็น production fallback
อ้างแผน: WP06/WP07

G09. [P1/P2] Exposure และ daily movement ยังใช้หลักคำนวณผิดบางกรณี

ไฟล์: application/manager/pm_review_service.py:188–218, :267; api/routers/portfolio/router_advisory.py:96
หลักฐาน:
- units=0, price=100 ยังถูกแทน units เป็น 1 และสร้าง gross exposure=100.0 ใน fixture
- unrealized_pnl_percent=80 ยังถูกใช้เป็น Top Daily PnL Mover
- eligible_positions_count ยังนับจำนวน quarantine
- daily P&L ยังต่าง NAV สองแถวโดยไม่มี external-flow/date validation และมี float arithmetic ก่อน Decimal

แผนแก้: actual active signed units, approved same-currency prices/FX, complete eligible/missing result; daily observations และ external-flow-aware P&L ตาม policy
ตรวจรับ: zero holdings ไม่เพิ่ม exposure; count ตรง eligible positions; lifetime return ไม่แทน daily movement; deposit/withdrawal ไม่เป็นกำไร
อ้างแผน: WP04

งานค้างอื่นที่ยืนยันจากโค้ดและยังอยู่ใน detailed plan

| ประเด็น | สถานะล่าสุด | งานที่ยังต้องทำ |
|---|---|---|
| Immutable packet | ยังไม่มี persisted packet/version model; GET ยัง assemble และ scan | WP05: exact snapshots, read-only GET, versioned approvals |
| Required observations | get_thesis_service ยังไม่ต่อ market observer; missing metrics ถูกข้าม | WP03: typed observations และ UNKNOWN/STALE/ERROR ต่อ required rule |
| Estimate temporal policy | approved filter มีแล้ว แต่ cutoff optional/string compare, insertion-order selection และ estimate_type invariant ยังไม่ครบ | WP01/WP06 |
| Catalyst materiality identity | date/classification เปลี่ยน ID แล้ว แต่ content/source revision ต้องผูกกับ human review ให้ครบ | WP03/WP08 |
| Restart durability | catalyst/estimate/routine state ยังอยู่ใน list/dict ของ process; routing persist ดีขึ้นแต่ยังมี fixed temp/overwrite semantics | WP08 |
| Time validation | aware comparison มีแล้ว แต่ invalid timestamp ยัง fallback string compare และ normalization ตัด precision | WP01/WP02/WP06 |
| UI request races | Earnings Bridge ดีขึ้น; StructuredThesisCard/MorningDelta ยังไม่มี cancellation/request identity และ sign-off ยังเปิดได้เมื่อมี old packet ระหว่าง failed refresh | WP09 |
| Strict DTO/evidence validation | บาง enums/financial inputs ยังไม่บังคับ finite/valid semantics ทุก endpoint | WP01/WP07 |
| Final release gate | backend ยังมี stale test หนึ่งข้อและไม่มี full-suite result ของรอบนี้ | WP00/WP11 |

ผลต่อประเด็นเดิมโดยย่อ

- N01/N02/N03: แก้กรณีที่ตรวจพบก่อนหน้าแล้วและยืนยันด้วย real-login/response-readback probes
- N04: stable hash/retry response ดีขึ้น; history/evidence correctness ยังไม่จบตาม G02/G04/G05
- R04: fresh DB isolation ผ่าน แต่ legacy DB ยังผิดตาม G01
- R05/R09/R18/R20: regression cases ที่เกี่ยวข้องยังผ่าน
- R06/R07: direct store corruption/stale-parent/mirror-lock cases แก้แล้ว; integration ต้อง propagate failure ตาม G03 และรักษา durable projection/evidence ตามแผน
- R01/R02/R03/R08/R10/R11/R12/R13/R14/R15/R16/R17/R19/R21: ยังมีงานค้างตามรายการด้านบนและ detailed plan ไม่ควรถือว่าปิดทั้ง WP

ลำดับแก้ที่แนะนำต่อจากรอบนี้

1. ทำ migration draft และ tests ของ old schemas ก่อนใช้กับ existing DB (G01; WP11/Gate DB-A)
2. ออกแบบ append-only packet/signoff revisions แทน UPSERT พร้อม exact stored idempotency response (G02/G04; WP05)
3. propagate required integrity/observation failure เป็น BLOCKED/UNKNOWN และห้าม approve (G03; WP03/WP05)
4. event revisions ต้อง persist source/fact changes และ bind acknowledgement ตรง snapshot (G05; WP03/WP08)
5. readiness อ่าน DB/state evidence จริง ไม่ใช้ header/ขนาด/nonempty-dict เป็น PASS (G06; WP10)
6. ปิด Macro IDs, exposure/daily math และ source/evidence grounding (G07/G08/G09; WP04/WP06/WP07)
7. เพิ่ม migration/history/retry-evidence/corrupt-source integration tests; ปรับ signer test ให้ยืนยัน authenticated session จริง; ปิด UI/durability/temporal tasks แล้วรัน final suites ตาม WP11

หลักฐานและข้อจำกัด

- ใช้ isolated FastAPI routers กับ login cookie จริงสำหรับ HTTP probes โดย mock เฉพาะ data/service boundaries ที่ต้องแยก state
- migration/history/evidence persistence probes ใช้ in-memory SQLite และ adapter/DAO ของ repository จริง
- corruption/readiness probes ใช้ temporary directories และไม่แตะ database/portfolio ของผู้ใช้
- ได้แยก fixture ของ corrupt source ออกจาก fixture ของ clean breach เพื่อไม่ให้ error propagation ทำให้ผล hash test แปลผิด
- ผลทั้งหมดเป็นการตรวจ behavior ปัจจุบัน ไม่มีการอนุมัติ schema/dependency/manifest change หรือการแก้ implementation ในรอบนี้

แผนรายละเอียดที่ใช้ต่อ: review/detailed-remediation-plan.md โดยใช้ผลรอบนี้อัปเดตสถานะตรวจรับของแต่ละ WP
