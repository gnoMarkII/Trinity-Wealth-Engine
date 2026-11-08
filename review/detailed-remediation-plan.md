แผนดำเนินงานแก้ไขฉบับละเอียด — Trinity Wealth Engine

อ้างอิง: ผลตรวจซ้ำ 5 พฤศจิกายน 2026 ใน review/re-review-results.md และแผนเดิม review/latest-code-review-plan.md
สถานะ: แผนสำหรับลงมือทำและตรวจรับ ยังไม่ได้แก้ implementation หรือเปลี่ยนฐานข้อมูล
เป้าหมาย: ปิด N01–N04 และประเด็น R01–R21 ที่ยังเหลือ โดยรักษา portfolio isolation, ticker validation, currency resolution และ Hermes listing ที่แก้ผ่านแล้ว

หลักที่ใช้ตัดสินว่าจบงาน

- ข้อมูลที่ขาดต้องเป็น unavailable/unknown พร้อมเหตุผล ไม่แทนด้วย zero, beta=1 หรือ regime ที่ทำให้เข้าใจว่าผ่านการตรวจแล้ว
- เลขการเงินและ allocation ทุกขั้นคำนวณใน Python ด้วย Decimal; conversion เป็น display value ทำที่ขอบระบบ
- record ที่ checksum เสียต้องหยุดการใช้และการเขียนต่อ โดยรักษา original bytes
- การอนุมัติอ้างถึง immutable packet ที่ผู้ใช้เห็นจริง และเก็บผู้ลงนามจาก trusted session
- retry ไม่เพิ่ม approval/event ซ้ำ; state ใหม่ต้องเป็น revision ใหม่และเก็บประวัติเก่า
- tests ต้องตรวจผลลัพธ์ที่ผิดและ production wiring ด้วย ไม่ยืนยันเพียง response shape หรือค่าจาก fixture

Baseline จากผลตรวจล่าสุด: backend ที่เกี่ยวข้อง 273 passed, frontend 621 passed, typecheck และ git diff --check passed แต่ diagnostic probes ยังพบข้อผิดพลาดจริง ตัวเลขนี้เป็นผลรอบตรวจล่าสุด ไม่ใช่ผลหลัง implementation ตามแผนนี้

ขอบเขตและสิ่งที่คงไว้

1. ใช้ Hexagonal Architecture เดิม: core เป็นกฎ/โมเดล/Decimal math; application ประสานงานผ่าน ports; tools/api.db เป็น adapters; api.dependencies เป็น composition root; routers รับและแปลง HTTP เท่านั้น
2. ใช้ FileLock, SQLite, Pydantic, pytest, Vitest และ dependencies ที่มีอยู่ก่อน ไม่มีแผนเพิ่ม package โดยอัตโนมัติ
3. ใช้ DbUnitOfWork ที่มีอยู่สำหรับ transaction หลาย repository; DAOs ไม่ commit/rollback เอง
4. Vault projection ใช้ knowledge write port และ infrastructure writer/broker ที่มีอยู่; application ไม่ import tools writer หรือทำ filesystem I/O เอง
5. Hermes ยังเป็น Manager เดียว; งานเตรียมข้อมูลและ dispatch ใช้ deterministic Python ไม่เพิ่ม LLM router
6. ไม่แก้ manifest_*.json ด้วยมือ ไม่ปรับ tests ให้ยอมรับข้อผิดพลาด และไม่เปิด legacy corrupted records เป็น usable data

โครงสร้างข้อมูลและพฤติกรรมเป้าหมาย

A. Identity และเวลา
- เพิ่ม AuthenticatedPrincipal ที่แยก trusted subject/session ID จากชื่อแสดงผล โดยเก็บโมเดลใน core และอ่าน cookie ที่ API boundary
- ระบบปัจจุบันใช้ shared password และ signed session ID จึงใช้ session:<sid> เป็น audit subject ได้ ไม่อ้างว่ามี named user/role ที่ระบบยังไม่รองรับ
- req.signed_off_by เป็น display label ได้หากต้องคง compatibility แต่ไม่ใช่ authoritative signer
- คง contract require_session ที่เป็น auth guard; เพิ่ม dependency สำหรับ routes ที่ต้องการ principal แทนสมมติว่า guard คืน dict
- legacy session ที่ไม่มี sid ต้องมีวิธีออก session ใหม่ก่อน mutation ที่ต้อง audit identity; ไม่ใช้ legacy_default_session เป็นผู้อนุมัติร่วมกันโดยเงียบ
- parse ISO-8601 เป็น timezone-aware UTC, รักษา fractional seconds และ reject invalid input ไม่มี string-comparison fallback
- ถ้า endpoint รับ date-only ให้ normalize ด้วยกฎที่ประกาศ; timestamp ที่ไม่มี timezone ต้อง reject หรือใช้ compatibility policy ที่ระบุชัด ไม่ปล่อยให้ timezone เครื่องตัดสิน

B. Observations และ rule evaluation
- observation ต้องมี security identity/market, metric, Decimal value หรือ null, unit, period, available_at, observed_at, source locator และ source/content hash
- rule evaluation แยก PASS, BREACH, UNKNOWN, STALE, ERROR; status ของ exception/acknowledgement เป็นอีกแกนหนึ่ง
- ทุก required rule ต้องมีผลประเมิน; metrics ที่ขาดไม่ถูกข้ามจนกลายเป็น Thesis Intact
- event fingerprint มาจาก rule/catalyst identity, claim version, observed fact, period/unit, source revision/hash และ material event fields
- detected_at, first_seen/last_seen และเวลาที่หน้าเว็บ refresh ไม่ใช้สร้าง logical event identity

C. Packet และ approval
- แต่ละ packet เก็บ packet_id, portfolio_id, review_date, revision, cutoff_utc, schema_version, policy_version, source_versions/fingerprint, payload และ packet_sha256
- packet payload เก็บ decimal เป็น canonical strings และ timestamp เป็น canonical UTC; hash คำนวณจาก payload bytes ที่ freeze แล้ว
- ข้อมูลใน hash ต้องรวม portfolio/cutoff/policy/source identity, review samples, exception IDs และ data-quality state
- ไม่รวม response-view status, signoff_info หรือเวลาที่เปิดหน้าเว็บ; checksum ไม่รวมตัวมันเอง
- GET /today อ่าน published snapshot และ approval ที่สัมพันธ์กัน ไม่เรียก scan/persist หรือ live provider แล้วสร้าง packet ใหม่
- การเตรียม packet แยกเป็น use case; ingestion/scheduler หรือคำสั่งเตรียมใหม่เป็นผู้สร้าง revision
- source_versions เดิมและ semantic payload เดิมต้อง reuse packet เดิม; source state ใหม่สร้าง revision ใหม่ และ old signoff ยังอยู่ใน history
- แสดง prepared_at/cutoff/freshness จริง; packet เก่าไม่ถูกแสดงว่า fresh เพียงเพราะมี GET ใหม่
- sign-off ต้องส่ง packet_id, portfolio_id, packet_sha256, verified_sample_ids และ acknowledged_exception_ids ตาม policy; hash ไม่เป็น optional
- required sample IDs มาจาก snapshot เดิมและต้องตรวจครบ; exception ทุกตัวที่ต้อง acknowledge มีรายการแยกจาก bounded sample 3–5 รายการ เพื่อไม่ตกหล่นเมื่อมี breaches มาก
- packet BLOCKED อนุมัติไม่ได้; optional unavailable fields แสดงตรงไปตรงมา และห้ามใช้สร้าง recommendation ที่ต้องพึ่ง field นั้น
- initial approval และ idempotent retry คืน response DTO เดียวกัน; stale/changed head หรือ reused idempotency key กับ payload คนละชุดคืน conflict

D. State ที่ต้อง durable
- thesis ledger ที่มีอยู่ยังเป็น canonical append-only JSON พร้อม FileLock ในงานแก้รอบแรก เพื่อลดการย้าย source of truth ซ้ำ
- Markdown เป็น projection ที่ rebuild จาก canonical thesis ได้ ไม่เป็น ledger อีกชุดหนึ่ง
- packets, approvals และ exception acknowledgement ใช้ SQLite/UoW เดียวกันเพื่อให้ atomic
- estimate records, catalyst/source revisions, human reviews และ routine execution events ต้องมี durable repositories ไม่พึ่ง singleton memory
- governance routing ใช้ immutable artifacts ต่อ decision revision ผ่าน port; list ต้องอ่าน durable state ไม่ใช่ list ของ process เดิมอย่างเดียว
- source documents/large blobs เก็บเป็น frozen local artifacts; SQLite เก็บ identity, locator และ digest เพื่อไม่ทำซ้ำไฟล์หลักฐานทั้งชุด

ส่วนที่ต้องอนุมัติก่อน implementation

Gate DB-A — จำเป็นสำหรับ immutable packet / approval flow
ข้อเสนอ schema ที่จะจัดทำ migration draft และตรวจร่วมก่อนรัน:
- เพิ่ม pm_review_packets พร้อม unique (portfolio_id, review_date, revision) และ unique semantic source/payload fingerprint ตาม scope เดียวกัน
- เพิ่ม packet_id, signer_subject_id, signer_session_id, verified_sample_ids_json, acknowledged_exception_ids_json, idempotency_key และ request_sha256 ให้ approval records
- เปลี่ยน unique ของ signoff จาก (portfolio_id, review_date) เป็น packet identity สำหรับ record รุ่นใหม่ เพื่อรองรับลงนามหลาย revision ในวันเดียวกัน
- เพิ่ม unique idempotency key ตาม portfolio/actor และตรวจ payload hash ก่อน reuse
- ขยาย exception records ให้มี stable event fingerprint, rule/catalyst reference, source/evidence fingerprint และ payload ที่ reconstruct ได้; acknowledgement ผูกกับ packet/signoff
- existing legacy rows ที่ไม่มี snapshot ต้องรักษาเป็น legacy audit history; ไม่สร้าง payload ปลอมหรือถือว่าอนุมัติ packet ใหม่โดยอัตโนมัติ
- ใช้ nullable/backfill-compatible columns หรือ table rebuild เฉพาะที่ SQLite จำเป็น โดยมี migration test ตรวจ old/new schemas ก่อนเสนอรัน

Gate DB-B — Durable research/routine records
- เสนอ repository tables สำหรับ estimate_records, catalyst revisions, source_captures, catalyst_reviews และ routine_execution_events โดย reuse table เดิมหาก contract รองรับ
- estimate/source values เก็บเป็น TEXT สำหรับ Decimal และ canonical UTC timestamps; identifiers/parent references ต้องตรวจ invariants
- routine events append ตาม execution_id/event sequence; current execution view reconstruct จาก events เพื่อเห็น started/completed/failed/crashed
- cache ใน memory เป็น read cache ได้ แต่ reload/reconcile กับ durable store และไม่เป็น authority

Gate CONTRACT — ทำหลัง API/behavior คงที่
- ขออนุมัติก่อน generate_golden_manifests.py หรือเปลี่ยน contract baseline
- export OpenAPI / regenerate TypeScript types ด้วย scripts ที่มีอยู่; ไม่แก้ generated file หรือ golden fixture เพื่อกลบ bug
- ถ้าจำเป็นต้องเพิ่ม package ให้เสนอเหตุผล/ทางเลือกและขออนุมัติแยกก่อนติดตั้ง

งาน domain, ports, regression tests, HTTP fixes, integrity validation และ UI prototypes ที่ไม่เปลี่ยน schema ทำต่อได้ก่อน Gate DB-A/DB-B ส่วน adapter/migration ที่พึ่ง schema ต้องรออนุมัติจริง แผนนี้ยังไม่ได้ถือว่าอนุมัติแล้ว

ลำดับและ dependencies ของ work packages

| ชุดงาน | เป้าหมาย | ต้องมีมาก่อน | Gate |
|---|---|---|---|
| WP00 | ตรึง regression cases และขอบเขตไฟล์ | ผล review | ไม่มี |
| WP01 | Auth, HTTP และ temporal validation | WP00 | CONTRACT ภายหลัง |
| WP02 | Thesis integrity / concurrency | WP00, temporal contract WP01 | ไม่มี DB change |
| WP03 | Observations, watchdog และ catalyst events | WP01, WP02 | DB-A สำหรับ durable events |
| WP04 | Portfolio/Macro analytics จริง | WP01, observation types WP03 | ไม่มี เว้นแต่ persistence เพิ่ม |
| WP05 | Immutable packets และ atomic sign-off | WP01, WP03, WP04 | DB-A |
| WP06 | Earnings/estimate sources และ ledger policy | WP01, observation/provenance contracts | DB-B สำหรับ persistence |
| WP07 | Grounded synthesis / response identity | WP02, WP03, WP06 | ไม่มีเพิ่ม |
| WP08 | Durable adapters, workers และ projections | Ports จาก WP03/WP05/WP06 | DB-A/DB-B |
| WP09 | UI และ Hermes ตาม final contracts | WP05, WP07, WP08 | CONTRACT ภายหลัง |
| WP10 | Readiness audit จากหลักฐานจริง | WP02–WP08 | ไม่มีเพิ่ม |
| WP11 | Migration rehearsal, contracts และ release gates | ทุก WP ที่เกี่ยวข้อง | CONTRACT / DB gates |

แต่ละ WP ต้องมี changeset ที่ review ได้ พร้อม tests ของ behavior ที่เปลี่ยน ไม่รวม source refactor ที่ไม่ช่วยปิดข้อผิดพลาด


WP00 — ตรึง regression suite และข้อมูลสำหรับตรวจรับ

ไฟล์หลัก:
- tests/api/test_institutional_gems_routers.py, tests/api/test_pm_review_router.py, tests/api/test_earnings_bridge_router.py
- tests/unit/application/test_pm_review_service.py, test_thesis_service.py, test_thesis_synthesis_service.py
- tests/unit/tools/test_thesis_file_store.py, test_thesis_sqlite_dao.py และ tests/api/conftest.py

งานย่อย:
1. บันทึก baseline และแยก user changes ที่มีอยู่จากงานแก้ใหม่ โดยไม่ reset/revert/stash ทับไฟล์ของผู้ใช้
2. แปลง probes N01–N04 และ A–G ใน review เป็น regression tests ที่ล้มด้วย behavior เดิมก่อนเริ่มแก้จุดนั้น
3. API integration tests ใช้ authed_client ที่ login จริง; mock เฉพาะ external provider/I/O boundary ไม่ mock auth guard ให้คืน user dict ที่ production ไม่มี
4. เพิ่ม fixture แยก thesis data/vault, routing audit, readiness output และ SQLite ต่อ test; ตรวจไม่เขียนลง data/, memories/ หรือ review/ ของผู้ใช้
5. ใช้ fake clock, fixed source revisions และ Decimal fixtures ที่มี expected output จาก Python; ไม่พึ่งวันที่เครื่องหรือ provider สด
6. เก็บ tests ของ R04/R05/R09/R18/R20 ที่แก้แล้วเป็น regression guard และตรวจผ่าน production provider อย่างน้อยหนึ่งเส้นทางต่อ feature

ส่งมอบ: test cases อ้าง issue IDs, fixture isolation และรายการ source contracts ที่จะใช้; ไม่เพิ่ม tests ที่เพียง mirror implementation หรือยอมรับค่าผิดเดิม
ตรวจรับ: reproduce 500/error paths, hash churn, duplicate approval, tampered ledger และ fake READY ได้อย่าง deterministic

WP01 — HTTP, trusted principal และ validation primitives

ไฟล์แก้:
- api/auth.py, api/dependencies.py
- api/routers/equity/router_estimates.py, api/routers/governance_router.py
- api/routers/thesis/router_catalyst.py, api/routers/pm_review_router.py, api/routers/thesis_router.py
- core/security.py, core/provenance.py และ schema files ที่เกี่ยวข้อง

งานย่อย:
1. แก้ imports ของ router_estimates ให้ครบ; ย้าย parsing/validation ไป DTO ที่ตรวจ finite Decimal, metric และ period ได้
2. ทำ trusted principal dependency ตาม design ข้างต้น; Governance/Catalyst/PM routes ใช้ principal ที่มี type แน่นอน ไม่อ่าน None.get()
3. รวมการ verify cookie ให้ใช้ policy เดียวกับ auth guard; ไม่รับ role/subject จาก payload
4. ทำ enum ของ operator, evidence tier, work_shape, catalyst status, unit และ period basis; invalid enum ต้องไม่ fallback เป็น DOCUMENTED หรือ routine reading
5. validate ticker/portfolio_id ด้วย primitives กลาง; แยก asset identity ที่รวม market ออกจาก display ticker
6. แก้ parse_utc_timestamp/is_available_as_of/normalize_utc_iso ให้ reject invalid และรักษา precision; แยก date-only parsing จาก timestamp parsing
7. กำหนด domain errors และ map HTTP ให้สม่ำเสมอ: validation 422, missing resource 404, stale/concurrency/idempotency conflict 409, required service unavailable 503; ไม่ส่ง raw filesystem paths/stack traces ให้ client
8. คง MNPI quarantine rule และ single-brain Manager; HTTP request ที่ valid ต้องเดินไป service ได้จริง

Tests ต้องมี:
- reported ขาด/ว่าง/ข้อความ/NaN/Infinity; unknown ticker; valid Decimal; ไม่มี branch ใดเป็น NameError/500
- login จริงแล้ว route-task และ catalyst-review ทำงาน; missing/expired/tampered cookie เป็น 401
- invalid work shape/evidence/operator เป็น 422; typo ไม่ได้รับสิทธิ์เป็น documented fact
- payload signed_off_by/role ไม่เปลี่ยน trusted audit identity
- เวลาเดียวกันใน Z/+00:00/offset เท่ากัน; fractional seconds ก่อน/หลัง cutoff; malformed/naive timestamp ตาม policy
- portfolio/ticker traversal, cross-market symbol ambiguity และ unsupported currency ถูกตรวจที่ boundary

เสร็จเมื่อ: ปิด N01/N02 ที่ HTTP boundary และ primitives พร้อมให้ทุก use case reuse
ข้อควรระวัง: ห้ามเปลี่ยน return type ของ require_session แล้วปล่อย routes อื่นเดาพฤติกรรมเอง ต้องตรวจ call sites ก่อนเปลี่ยน contract

WP02 — Thesis ledger integrity และ optimistic concurrency

ไฟล์แก้:
- tools/thesis/thesis_file_store.py, application/thesis/thesis_service.py, application/thesis/ports.py
- core/thesis/models.py, core/provenance.py
- tools/_atomic_io.py เฉพาะหากต้องเพิ่ม byte-preserving atomic helper
- knowledge projection adapter ที่ใช้ writer port เดิม (ไฟล์ใหม่ที่เสนอ)

งานย่อย:
1. แยก ThesisNotFound, ThesisIntegrityError, ThesisReadError และ ThesisRevisionConflict; not-found เท่านั้นที่ตีความเป็น history ว่าง
2. ถ้ามี checksum mismatch, invalid record structure, duplicate ID/version หรือ broken parent chain ให้ทั้ง ledger ใช้งานไม่ได้ชั่วคราว ไม่กรอง record ทิ้งแล้วเขียนต่อ
3. เก็บ original bytes ใน quarantine ที่ตั้งชื่อด้วย digest/incident identity; atomic copy แบบรักษา bytes; ไม่แก้ต้นฉบับ ไม่เขียน backup ชื่อเดิมทับ และไม่มี read ที่สร้าง backup ซ้ำทุกครั้ง
4. ภายใน per-ticker FileLock ตรวจ current head/expected parent/expected version แล้ว append; initial create อนุญาตเมื่อไม่มี valid prior history เท่านั้น
5. การ revise ที่ใช้ stale parent ต้องแจ้ง conflict; ไม่ auto-rebase thesis ที่มนุษย์ review จากฐานเก่าโดยเงียบ
6. checksum ครอบคลุม canonical identity, version, parent และ substantive fields; record ที่สร้างใหม่ใช้ serialization policy เดียวกัน
7. current/as-of view เลือก revision ที่ eligible ตาม effective/available time ก่อนตัดสิน lifecycle status; latest retired revision ต้องไม่ทำให้ older active recordฟื้นกลับมา
8. JSON canonical write สำเร็จก่อน projection; ส่ง projection command ที่มี expected head/version ให้ knowledge writer และบันทึก pending/retry state ผ่าน infrastructure adapter
9. projection failure ต้องไม่รายงานว่า ledger save ล้มจน clientสร้าง duplicate revision; แจ้ง canonical save สำเร็จและ projection pending อย่างชัดเจน
10. ภายใต้ process race ห้าม Markdown version เก่าเขียนทับ version ใหม่; worker ต้องตรวจ head/fencing ก่อน publish และ rebuild projection ได้จาก ledger
11. ข้อมูล legacy ที่ยังไม่มี temporal provenance ใหม่ ต้องติด limitation ที่ตรงจริง ไม่อ้าง historical availability ที่ไม่มีหลักฐาน

Tests ต้องมี:
- malformed JSON, checksum mismatch, invalid encoding, duplicate/broken chain: write ถูกปฏิเสธและ original bytes เหมือนเดิม
- concurrent creates/revisions ข้าม process: ไม่มี lost update; คนที่แพ้ได้ conflict; version/parent ไม่ผิด
- v1->v2 แล้วส่ง parent=v1: rejected; initial create ซ้ำ: rejected/idempotent ตาม request identity ที่กำหนด
- projection v1 ช้ากว่า v2: final mirror ยังเป็น v2; crash หลัง canonical save: recover projection ได้
- future/backdated/retired revision และ precision cutoff ไม่คืน active thesis ผิดฉบับ

เสร็จเมื่อ: ปิด R06/R07; traversal guard ที่แก้แล้วต้องยังผ่าน และ canonical history ไม่สูญหายแม้ state/projection มีปัญหา

WP03 — Typed observations, complete watchdog evaluation และ stable catalyst events

ไฟล์แก้:
- application/thesis/ports.py, thesis_service.py, catalyst_service.py
- core/thesis/evaluator.py, catalyst_queue.py, models.py
- tools/market/financials/ และ adapters ใหม่สำหรับ observation/catalyst ที่เสนอ
- api/dependencies.py, tools/thesis/thesis_sqlite_dao.py, api/db/adapters.py

งานย่อย:
1. เปลี่ยน observation port ให้คืน typed result ที่มี value/status/unit/period/currency/source clocks; ใช้ wrapper ชั่วคราวสำหรับ call sites เดิมระหว่าง migration
2. ต่อ production provider กับ financial/price/calendar readers ที่มีอยู่; ไม่เรียก network adapter จาก application/core
3. สร้างผลต่อ rule ทุกตัว โดยตรวจ applicability ตาม asset class; NOT_APPLICABLE ต้องมีกฎชัด ไม่ใช้แทน missing data
4. แก้การ merge observed_metrics_by_ticker เพื่อไม่ให้ partial price override ทำให้ fundamentals จาก observer ถูกข้ามทั้งชุด
5. เก็บ threshold, operator, observed value, source revision และ evaluation cutoff ใน result ที่ตรวจกลับได้
6. ใช้ stable event fingerprint; scan ซ้ำกับ source เดิมไม่สร้าง breach ใหม่และไม่รีเซ็ต acknowledgement
7. unknown/stale/error ไม่แสดงเป็น PASS; thesis coverage แยกมีเอกสาร thesis ออกจากมีหลักฐานเพียงพอและ evaluated health
8. Catalyst source selection ใช้ entity/event scope/period และ latest eligible available/captured time; ไม่ใช้ dictionary insertion order หรือ source เก่าที่ match ID มาแทน source ใหม่โดยอัตโนมัติ
9. fingerprint รวม old/new dates, classification, observed fact และ content revision/hash; review ใช้ได้เฉพาะ event fingerprint เดิม
10. แยก pure evaluation ออกจาก persist scan; GET อ่าน persisted evaluation/event snapshot ไม่สร้าง UUID หรือเปลี่ยน clocks
11. persist evaluation/events และ outbox งานเตรียม packet ใน UoW เมื่อ state เปลี่ยน; cache เป็นเพียง optimization

Tests ต้องมี:
- price มีแต่ growth/margin ขาด -> UNKNOWN ที่ระบุ rule; current_price override ไม่ปิด fundamental fetch
- breach เดิม scan ซ้ำ -> event identity เดิม; source/observed value/date เปลี่ยน -> event ใหม่ REVIEW_REQUIRED
- future capture/ผิด period/ผิด entity/invalid source clocks ถูกตัดออก; source order ไม่เปลี่ยนผล
- เปลี่ยนวันที่ใน source ID เดิมวันเดียวกันไม่ reuse acknowledged event เก่า
- GET packet/factor-exposure ไม่เพิ่ม event rows หรือเปลี่ยน state
- restart/worker อีก instance เห็น pending/reviewed state เดียวกันเมื่อใช้ durable adapters

เสร็จเมื่อ: ปิด R10/R12 และให้ WP05 ใช้ชุด evidence ที่ freeze ได้
Gate: domain/ports/tests ทำได้ก่อน; persistence/migration ส่วน events รอ DB-A

WP04 — Portfolio exposure, Daily Delta, Macro และ beta

ไฟล์แก้:
- application/manager/pm_review_service.py (แยก aggregation ออกเป็น use cases/ports ที่มีขอบเขตชัด)
- core/equity_analytics/factor_exposure.py
- api/routers/portfolio/router_advisory.py, api/schemas/portfolio.py, api/dependencies.py
- tools/macro/adapters/strategy_vault_adapter.py หรือ adapter อ่าน dashboard ที่เสนอ
- tools/portfolio/services/performance_service.py, ports และ cash-flow readers ที่เกี่ยวข้อง

งานย่อย:
1. active holding ที่ actual units=0 ไม่มี exposure; missing units ไม่แทนเป็น 1; signed quantity ต้องรักษา long/short direction
2. เลือกราคาใน native currency ของ asset; ไม่มีราคาที่อนุมัติ ณ cutoff ให้ missing ไม่ใช้ avg cost/zero/ราคาคนละ currency เป็น fallback
3. ถ้าต้องรวม THB ใช้ approved USDTHB/FX ที่สอดคล้อง cutoff และเก็บ provenance; ไม่มี FX ให้ unavailable ไม่เดา rate
4. คำนวณด้วย Decimal ตั้งแต่ adapter conversion; ไม่หาร/multiply float แล้วค่อยห่อ Decimal
5. result เดียวคืน eligible_positions_count, missing_price_symbols ทั้งหมด, quarantined holdings และ completeness status; router ไม่คำนวณ count ใหม่จาก quarantine
6. หากใช้ verified subset ต้องติด PARTIAL และเปิดเผย excluded symbols; ไม่แสดงผลรวมว่า complete และไม่สร้าง allocation action ที่พึ่งผลรวมไม่ครบ
7. รักษา definition Long/Gross, abs(Short)/Gross, Net/Gross ตาม domain ที่ประกาศ หากต้องการ exposure ต่อ NAV ให้เพิ่ม field ที่ชื่อชัดและ contract แยก ไม่เปลี่ยน denominator เดิมโดยเงียบ
8. daily mover ใช้ daily price/P&L observation ณสอง cutoffs ที่ระบุ; lifetime unrealized return แสดงเป็นอีก metric และไม่ใช้เลือก daily sample
9. daily P&L = end NAV - start NAV - net external capital flows; แยก deposit/withdrawal จาก internal trades/dividends และใช้รายการธุรกรรมที่ตรวจกลับได้
10. daily_pnl_pct ต้องประกาศ capital-base convention; หากใช้ opening NAV ให้ชื่อ/คำอธิบายเป็น P&L ต่อ opening NAV และไม่อ้างเป็น time-weighted return; ไม่มี required flow/NAV history ให้ unavailable
11. map Macro จาก dashboard_indicators/overall_regime และ observable registry จริงตาม indicator_id/units; delta ต้องมี prior observation ที่เวลาถูกต้อง
12. เอา beta=1, zero stress/spread และ Low/Neutral default ออกจาก absent/error state; optional fields ใช้ null/status/reason
13. ต่อ benchmark reader ที่จับคู่ return dates เดียวกันและ filter cutoff; insufficient observations/zero variance -> unavailable พร้อม reason
14. ต่อ real state-integrity checker ผ่าน port; checker ล้มไม่เงียบและไม่แสดง HEALTHY

Tests ต้องมี:
- zero/archived/short/missing/invalid price; missing FX; mixed currencies; no holdings และ partial subset
- count ของ eligible ถูกต้องแม้มี quarantine; missing symbols ครบทุกตัวและ code vocabulary เดียวกัน
- deposit/withdrawal ไม่เป็นกำไร; snapshots วันเดียวกันไม่เรียก 24h; daily mover แยกจาก lifetime return
- Macro production-shaped fixture มี nonzero values และ regime ต่างจาก Neutral; mapping ต้องออกตรง ไม่พึ่ง shape สมมติ
- beta ข้อมูลไม่พอ/variance zero/future observation -> ไม่มีเลข default
- integrity unavailable/tampered ต้องสะท้อนที่ packet data quality

เสร็จเมื่อ: ปิด R02/R14/R15 และเป็นแหล่งข้อมูลจริงให้ immutable packet


WP05 — Immutable Morning Review และ transactional sign-off

ไฟล์แก้:
- application/manager/pm_review_service.py, application/manager/ports.py
- core/portfolio_advisory/review_sampling.py และโมเดล packet/result ใน core ที่เสนอเพิ่ม
- api/routers/pm_review_router.py, api/dependencies.py
- api/db/connection.py, api/db/uow.py, api/db/adapters.py, tools/thesis/thesis_sqlite_dao.py
- adapters/repository tests ที่เกี่ยวข้อง

งานย่อย:
1. แยก prepare_morning_review ออกจาก read_latest_packet และ sign_off_packet; method เดิมเป็น compatibility wrapper เท่าที่ไม่สร้าง side effects ผิด contract
2. ระหว่าง prepare เก็บ source snapshots/versions และ cutoff ให้ครบก่อนสร้าง packet; ไม่ถือ SQLite write transaction ค้างระหว่าง network/provider calls
3. ตรวจ consistency ของ captured input heads ก่อน publish; ถ้า input เปลี่ยนระหว่างเก็บให้ retry ตาม bounded policy หรือคืน failed/incomplete run ไม่ freeze ภาพผสมคนละ state
4. serialize canonical payload ด้วย schema_version และ policy_version; sort collections ที่ลำดับไม่มีความหมาย แต่รักษาลำดับที่เป็น business semantics; คำนวณ SHA-256 ครั้งเดียวจาก bytes ที่จะบันทึก
5. transaction publish packet, current-head reference และ outbox งานถัดไปพร้อมกัน; same input fingerprint reuse packet เดิม และ concurrent prepare ไม่สร้าง revision ซ้ำ
6. job เตรียมใหม่ผ่าน endpoint/worker ตาม async contract: POST prepare -> 202 + job_id; GET today/history/packet details เป็น read-only
7. GET ที่ยังไม่มี packet คืน 404 หรือ empty-state contract ที่กำหนดชัด; UI เสนอเตรียมข้อมูลได้ ไม่เรียก LLM/สร้าง approval ระหว่าง GET
8. sign-off โหลด exact packet_id ภายใน UoW; ตรวจ portfolio/date/hash/schema/policy/freshness/quality และ expected current head
9. required hash ต้องมีและตรง stored bytes; required sample IDs ต้องครบและไม่รับ ID เกินชุด; acknowledgement IDs ต้องเป็น exceptions ใน snapshot เดียวกันและครบตาม policy
10. validate idempotency key + canonical request hash; key เดิม/payload เดิมคืน original result; key เดิม/payloadต่างกันคืน 409
11. write approval + sample evidence + exception acknowledgement + outbox audit event ใน transaction เดียว; จำลอง failure ระหว่างขั้นแล้วต้อง rollback ทั้งหมด
12. re-sign หลังข้อมูลเปลี่ยนผูก packet revision ใหม่ ไม่ INSERT แข่ง unique portfolio/date เดิม; ประวัติเก่า immutable และแสดงว่าอนุมัติ snapshot ใด
13. response mapping ใช้ typed application result เดียวกันทั้ง create/retry/history; ต้องมี status และ verified samples เสมอ ไม่คืน raw DB row
14. จัด current status จาก latest published packet/approval relationship; historical approval ยัง SIGNED_OFF สำหรับ snapshot เดิม แต่ไม่ได้อนุมัติ revision ใหม่
15. input ingestion ที่เปลี่ยน head แจ้ง deterministic preparation workflow ผ่าน durable outbox; แสดง freshness/cutoff ชัดเจนระหว่างรอ packet ใหม่ ไม่อ้าง live ณเวลาที่เปิดหน้า
16. review sample เก็บ frozen IDs/targets ใน packet; all critical exceptions ที่ต้องรับทราบยังมองเห็นได้แม้ bounded sample เต็มแล้ว

Tests ต้องมี:
- GET ซ้ำไม่เกิด DB/file writes และได้ packet ID/hash เดิม
- unchanged breach ลงนามได้ด้วย hash จาก GET; event detection clock ไม่ทำ hash churn
- omitted hash, forged hash, wrong packet/portfolio, empty/partial/extra samples, unknown exception IDs -> 422/409 ตามชนิด
- first approval และ retry ได้ 200 พร้อม result เหมือนกัน; concurrent retry ได้ audit row เดียว
- changed state -> new revision; old hash rejected เมื่อยื่นกับ new head; new revision ลงนามได้โดยไม่เสีย history
- fault injection หลัง insert approval ก่อน acknowledge/outbox -> ไม่มี partial approval
- cross-portfolio same date, different cutoff/policy/source versions ไม่ใช้ approval แทนกัน
- restart ระหว่าง prepare/publish/sign-off recover ได้โดยไม่สร้าง duplicate packet หรือยืนยันการอนุมัติที่ยังไม่ commit

เสร็จเมื่อ: ปิด R03/N04 และรักษา R04; packet ที่มนุษย์เห็นตรงกับ packet ที่ระบบบันทึกและอนุมัติจริง
Gate: DB-A ต้องได้รับอนุมัติก่อนเปลี่ยน schema/index/migration; API baseline ขออนุมัติหลัง behavior เสถียร

WP06 — Earnings sources, approved estimate ledger และ point-in-time selection

ไฟล์แก้:
- api/routers/equity/router_earnings_call.py, router_estimates.py
- application/equity/estimate_service.py และ earnings-bridge service/ports ที่เสนอเพิ่ม
- core/equity_analytics/earnings_bridge.py, estimate_ledger.py
- tools/market/financials/ และ disclosure/estimate repository adapters ที่เสนอ
- application/macro/vintage_service.py สำหรับแยก synthetic fixtures ออกจาก production provider

งานย่อย:
1. ย้าย _AUTHORITATIVE_EARNINGS_DISCLOSURES และ seed GOOGL/SYN-* ออกจาก production defaults ไป test/demo fixtures ที่ติด SYNTHETIC ชัด
2. GET earnings bridge ใช้ ticker/market/fiscal period/source cutoff เรียก application service; router ไม่ประกอบตัวเลขหรือ drivers
3. อ่าน reported results จาก existing captured filing/financials source พร้อม locator/hash/availability; consensus เป็น frozen pre-release source record และ fund estimate เป็น approved internal record
4. หากไม่มี consensus/fund estimate/source rights หรือข้อมูล historical ให้ unavailable/not found ไม่สร้าง record ทดแทนเพื่อให้หน้าเว็บมีตัวเลข
5. period normalization ต้องรักษา fiscal period ของบริษัท; cutoff ต้องไม่อ้างข้อมูลที่ available ภายหลัง earnings print เป็น prior view
6. ปิด revision invariants ที่เหลือ: estimate_type ต้องตรง parent, parent เป็น latest eligible approved lineage ตาม policy, timestamps ต้อง valid/monotonic ที่กำหนด, unit/metric/security ตรงกัน
7. choose approved latest eligible record ตาม parsed availability/effective time และ stable revision tie-break ไม่ตาม append order; route/use case ส่ง explicit cutoff
8. overrides ต้องเป็น explicit scenario input พร้อม label; ห้าม override ไปปนกับ approved baseline/history
9. missing baseline, zero denominator หรือ nonfinite value มี domain outcome ชัด ไม่สร้าง surprise_pct=0 ที่สื่อว่า inline โดยไม่มีฐานเปรียบเทียบ
10. driver rows มี source/formula locator และหน่วยเดียวกับ bridge; unexplained residual ต้องแสดงเป็น residual/unavailable explanation ไม่แต่ง causal drivers ให้ sum ตรง
11. thesis reconciliation ผูก prior thesis/estimate version IDs จริง; ไม่เปลี่ยน prior belief และไม่ยืนยัน thesis ทั้งฉบับจากแค่ earnings beat
12. production Macro vintage provider ต้องอ่าน captured vintages จริง; fixture service ใช้เฉพาะ test/demo และติด source label ทุกแถว

Tests ต้องมี:
- unknown ticker/period/source -> controlled unavailable; fixture constants ไม่หลุด production provider
- earlier-approved row ที่ append ทีหลังและ pending/future rows ไม่เปลี่ยน historical baseline
- cross estimate_type/metric/unit/security, stale parent, duplicate ID และ unbalanced revision ไม่ถูกใช้งาน
- negative/zero denominator, NaN/Infinity, residual drivers และ mismatched units ไม่ได้ผล “ยืนยันแล้ว” ที่ผิด
- deleting/regenerating revised output ไม่เปลี่ยน prior approved record bytes/hash
- reported/consensus/fund records อยู่คนละ cutoff หรือ source revision ต้องถูกตรวจและเปิดเผย

เสร็จเมื่อ: ปิด R01/R11 และส่วน PIT ของ R16; source metadata ที่ใช้คำนวณย้อนตรวจได้
Gate: durable estimate/source tables รอ DB-B; domain/ports และ read adapters จากแหล่งที่มีอยู่ทำก่อน

WP07 — Evidence tiers, thesis synthesis และ response identity

ไฟล์แก้:
- application/thesis/thesis_synthesis_service.py, thesis_service.py
- core/thesis/claim_validator.py, evidence_taxonomy.py, models.py
- api/routers/thesis_router.py, api/dependencies.py
- tools/hermes_bridge/trinity_mcp_server.py และ prompt เฉพาะส่วนที่ต้องสะท้อน contract ใหม่

งานย่อย:
1. แยก SourceFact, DerivedMetric, PolicyThreshold และ ThesisHypothesis; provenance ต่อข้ออ้าง ไม่ยกระดับทั้ง thesis เพราะตัวเลขหนึ่งตัวมีแหล่ง
2. claim validator ตรวจ fact กับ primary source record ที่มี locator/content hash/period/unit; ไม่สร้าง claimed value จาก dictionary เดียวกับ expected fact แล้วนับเป็นการตรวจ
3. DOCUMENTED ใช้ได้กับข้อเท็จจริงที่ชี้ primary source ได้; DIRECTLY_TESTED ต้องมี calculation inputs/formula/execution locator; custom/derived thresholds เป็น INFERRED/POLICY ตาม taxonomy ที่ตกลง
4. custom_stop_loss ต้องเข้ากฎ tier เช่นเดียวกับ custom growth/margin; ห้ามใช้เพียง current price เป็นเหตุผลว่า narrative ทั้งฉบับ tested
5. stop-loss/growth/margin ที่ derive ตามสูตรต้องมี approved policy_version; ไม่มีข้อมูล/ไม่มี policy ไม่สร้าง floor คงที่หรือ positive company statement โดยอัตโนมัติ
6. financial metric และ unit ตรงกัน เช่น fcf_margin ไม่เป็น operating_margin; ไม่เลือกด้วย truthiness จนค่าศูนย์หาย
7. unsupported calendar/unscheduled catalyst ไม่ถูกอ้างว่า confirmed source; จัดเป็น missing schedule ที่ไม่ผ่าน time-based watchdog อย่างเงียบ
8. แยก “มี thesis document” ออกจาก “มี executable falsifiers และ evidence พร้อมประเมิน”; batch coverage ไม่เพิ่ม monitored coverage ด้วย empty/no-data thesis
9. hostile input ที่ตรวจพบต้องหยุดใช้เป็น instruction/active evidence; เก็บ source ใน quarantine พร้อม security exception ผ่าน adapter ไม่ใช้ cleaned flag เป็นใบอนุญาตให้ persist instructions
10. non-dry-run ใช้ object ที่ record_new_thesis คืนเป็น response ทั้งก้อน; ไม่สร้าง claim ID/time/hash อีกฉบับเพื่อส่งกลับ
11. dry-run ใช้ builder/validation เดียวกันโดยไม่ write; ระบุ provisional identity ให้ชัดและไม่อ้างว่า record ถูกบันทึกแล้ว
12. batch ส่ง portfolio/market context ทุกชั้น และยังแยกหุ้นชื่อคล้ายกันต่างตลาด; errors ราย ticker ต้องทำให้ coverage/result สะท้อน partial failure
13. Hermes tools คืน structured status ที่บอก unknown/inferred/blocked ได้; คง ProgressiveToolRegistry และ wrap_tools_with_harness; ไม่เพิ่ม Manager/LLM router

Tests ต้องมี:
- source metadata ขาด, documented claim mismatch, wrong unit/period, custom threshold -> ไม่ได้ DOCUMENTED โดยไม่มีหลักฐาน
- มีราคาอย่างเดียวหรือไม่มีข้อมูลเลย -> ไม่สร้าง positive verified thesis/healthy monitored coverage
- zero/negative growth/margin ไม่ถูกแทนด้วย default; mixed FCF/operating metrics ไม่ปะปน
- response claim == read-back claim ทุก field รวม timestamp/checksum หลัง write
- dry-run ไม่มี files/rows ใหม่; batch พอร์ต B ไม่อ่าน fallback ของ default
- hostile instruction ไม่กลายเป็น active thesis; log/quarantine evidence ย้อนตรวจได้
- get_active_stock_theses() ไม่ระบุ ticker และเครื่องมือสังเคราะห์ผ่าน production provider ได้โดยรักษา harness

เสร็จเมื่อ: ปิด R08/N03 และรักษา R09/R20; confidence labels สอดคล้องหลักฐานจริง

WP08 — Durable repositories, scheduled workflows และ atomic projections

ไฟล์แก้:
- application/thesis/catalyst_service.py, application/equity/estimate_service.py
- application/resilience/routine_monitor_service.py, application/governance/routing_service.py
- ports ของแต่ละ domain, api/db/adapters.py, api/db/uow.py
- tools/thesis/, tools/archivist/ adapters และ worker/scheduler handler ที่จะ wire ใน composition root
- application/knowledge/write_ports.py และ broker/executor ที่มีอยู่เฉพาะส่วนเชื่อม projection

งานย่อย:
1. เปลี่ยน constructor ของ services ให้รับ repository ports; production provider ต้องส่ง durable adapter และไม่ seed demo data
2. เพิ่ม UoW properties สำหรับ packet/signoff/event/estimate/review repositories แบบ connection-bound; tests บังคับ shared transaction ไม่เปิด connection/commit แยกกลาง use case
3. persist source captures/catalyst revisions/human reviews และ estimate revisions พร้อม parent/fingerprint; review ไม่หายหลัง restart หรือไป worker อื่น
4. ต่อ thesis/calendar/filing capture ingestion จริงเข้าคิว evaluation; ใช้ deterministic handler และ lease/owns_execution fencing ตาม API conventions
5. routine start/completion/failure บันทึก durable execution events; recovery ของ worker ที่ crash สร้าง terminal/missed outcome ตาม timeout ไม่ค้าง IN_PROGRESS ตลอด
6. schedule/cutoff/grace/timezone อยู่ config/policy ชุดเดียว; default morning ใช้เวลาที่ผู้ใช้เห็นใน UI และแปลง UTC ที่ runtime ไม่ hardcodeคนละค่าตาม service
7. missed/late/failed status มีเหตุผลและ last successful cutoff; retry run ID เดิมไม่เพิ่ม run ปลอมและไม่ reset freshness ให้ดูสด
8. routing artifacts เป็น immutable decision revisions; repeated task ID คนละ payload เก็บ revision/conflict ไม่ overwrite recordเดิม
9. ย้าย routing/readiness filesystem I/O ออกจาก application ไป artifact ports/adapters; ใช้ shared atomic helpers และ unique temp
10. Vault notes ผ่าน write port/broker/ArtifactWriter ที่มี revision/concurrency semantics; update storage contract เฉพาะ sink ที่ใช้จริง ไม่ใช้ allowlist เป็นการข้าม writer policy
11. projection/outbox workers ตรวจ lease/fencing ก่อน publish; retry ใช้ semantic artifact key และไม่เพิ่ม duplicate revisions
12. health/readiness อ่าน durable state ไม่ขึ้นกับ memory caches; publish failure ต้องรักษา pending work และ incident evidence

Tests ต้องมี:
- reconstruct state ด้วย service instance ใหม่; read จาก worker/process อีกตัวเห็น approvals/reviews/routines/estimates เดียวกัน
- crash หลัง commit ก่อน notify, หลัง claim lease และระหว่าง projection -> recover โดยไม่ duplicate
- worker lease หมดอายุ/worker เก่ากลับมาทำงาน -> ไม่ publish stale output
- same task ID / changed payload / concurrent writes -> immutable history ถูกต้อง
- app/core ไม่มี adapter imports หรือ filesystem writes; DAO ไม่มี commit/rollback; writer boundary ยังผ่าน

เสร็จเมื่อ: ปิด R13 และ storage ส่วนที่เหลือของ R21; ไม่มี feature ที่ต้องอาศัย process เดิมอยู่ตลอด
Gate: repositories/migrations รอ DB-A/DB-B; ports, deterministic handlers และ adapter-independent tests ทำก่อนได้


WP09 — UI และ Hermes ที่แสดง exact packet และ data quality

ไฟล์แก้:
- web/src/api/client.ts, types.ts และ types.generated.ts ผ่าน generator
- web/src/components/manager/MorningDeltaCard.tsx, ManagerRightPanel.tsx
- web/src/components/equity/StructuredThesisCard.tsx, EarningsCallTab.tsx, EarningsBridgeCard.tsx, VariantPerceptionCard.tsx
- web/src/pages/ManagerAdvisor.tsx, tools/hermes_bridge/ และ tests ของ components/tools ที่เกี่ยวข้อง

งานย่อย:
1. DTO แยก numeric value ที่ unavailable ออกจาก zero และมี reason/as-of/source reference; format helper ไม่เรียก toFixed/toLocaleString บน null/undefined
2. แสดง packet_id/revision/cutoff/prepared_at/freshness/quality และสถานะ approval ที่ตรง snapshot; ใช้ wording “ไม่มีข้อมูล” สำหรับ UNKNOWN ไม่ใช้ “Thesis Intact”
3. cache/request identity รวม portfolio/ticker/period/packet ID; reset data, selections, notes และ modal state เมื่อ identity เปลี่ยน
4. ใช้ AbortController หรือ request generation guard ใน StructuredThesisCard/MorningDelta; response เก่าหรือ request ยกเลิกห้ามเขียน state ของ identity ใหม่
5. ระหว่าง initial load/refresh error/stale/blocked packet ปิด sign-off; ถ้ายังแสดง snapshot เก่าเพื่ออ่านต้องติดสถานะและไม่มี active approval control
6. modal ยึด packet ID/hash และ required sample IDs เดิม; ถ้า head เปลี่ยนให้แจ้ง reload/review ใหม่ ไม่โยก selections ไป snapshot ใหม่อัตโนมัติ
7. ส่ง trusted request fields ตาม sign-off contract; ไม่ให้ client label เป็น authoritative signer; จัดการ 409/422/503 แยกและคงข้อมูลที่ผู้ใช้ต้องตรวจได้
8. verify-all ช่วยเลือกได้เฉพาะ required IDs ที่แสดง แต่ server ยังบังคับ policy; exception acknowledgement แยกจาก bounded samples
9. เตรียม packet ผ่าน explicit prepare action/job status; หน้าอ่านข้อมูลไม่ trigger research หรือ synthesis โดยปริยาย
10. Hermes อ่าน packet/evidence exact IDs และ structured missing/inferred state; หลักฐาน badges เป็นการนำเสนอ server provenance ไม่ใช่การยืนยันข้อเท็จจริงจากข้อความที่ LLM แต่งเอง
11. รักษา tab keep-alive และ ticker resets ที่แก้แล้ว โดยผูก freshness กับ data identity แทน lifecycle ของ tab อย่างเดียว

Tests ต้องมี:
- ticker/portfolio A->B แล้ว response A กลับช้า: ไม่แสดง A และไม่ลงนามจาก A
- refresh ล้มขณะมี old packet: เห็น error/stale และ sign-off disabled
- optional null metrics ไม่ crash และไม่แสดงเป็น 0/neutral
- modal เก่าพบ new revision: ต้อง review ใหม่; selections ไม่ถูก carry over
- first sign-off/retry/409 stale/422 missing evidence และ async prepare states
- Hermes summaries ใช้ evidence IDs ที่ service ส่งและไม่แปลง unavailable เป็นเลขสมมติ

เสร็จเมื่อ: ปิด R19 และ UI assumptions ใน R02/R03; รักษา R09/R20 ที่แก้ผ่านแล้ว

WP10 — Readiness audit ที่พิสูจน์ได้จาก durable evidence

ไฟล์แก้:
- application/resilience/readiness_audit_service.py
- core/resilience/readiness_audit.py, core/portfolio_advisory/state_integrity.py
- scripts/institutional_readiness_audit.py
- read-only database/state/evidence adapters ที่เสนอ และ API/CLI composition wiring

งานย่อย:
1. application รับ ReadinessEvidencePort/ArtifactOutputPort; core ประเมิน supplied evidence โดยไม่อ่านไฟล์/SQLite เอง
2. Q1 ใช้ configured DB path จริงและ read-only SQLite connection; ตรวจ query/schema/version/quick_check แทนเช็ค header
3. Q2 ใช้ evidence ของ data/environment approval ที่ระบบมี; .gitignore เป็นเพียงหนึ่ง technical control ไม่ถือว่าแสดง data rights approval ทั้งหมด
4. Q3 ตรวจ clocks/source availability ของ records ที่ใช้จริง พร้อม cutoff และ provenance coverage ไม่ทดสอบเพียงตัวอย่าง hardcoded
5. Q4 ตรวจผล calculation records/fixtures ที่ผูก engine version และ input digest รวม missing/invalid cases; ไม่อ้างว่าทุก financial path ผ่านเพราะสูตรหนึ่งทำงาน
6. Q5 ใช้ trusted frozen manifest ที่มี required inputs; เทียบทุก expected digest กับ actual bytes และตรวจ missing/unexpected/path traversal; empty manifest ต้องไม่ PASS
7. Q6 ใช้ seeded failure/recovery test evidence ที่รันจบและอ้าง artifact/run ID ไม่ใช่เพียง classify error string ได้
8. Q7 ใช้ positive/negative claim tie-out evidence ที่อ้าง primary records/formulas; ต้องมี mismatch/unknown cases ที่ถูก reject
9. Q8 สร้าง packet/research view จาก durable inputs ด้วย instance ใหม่โดยไม่ใช้ chat/memory และเทียบ identity/hash ตาม canonical policy
10. Q9 ตรวจ API/service approval gate, required samples, idempotency และ transaction behavior จริง ไม่ค้นคำ AUTONOMOUS_EXECUTE ใน enum
11. Q10 ตรวจ incident/security/quarantine artifact ที่ persist และอ่านกลับได้ ไม่ถือว่ารายการ exception ในหน่วยความจำเป็น preserved evidence
12. ทุก question ส่ง verdict, checked_at, source/run IDs, input hash, actual checks และ reason; evidence ขาด/เก่าหรือยังไม่รัน = FAIL/UNKNOWN ตาม gate
13. automated report มี evaluator/run ID และแยกจาก human sign-off; ชื่อ signatory ใน CLI ไม่ถือว่า PM ลงนามจริงหากไม่มี approval record
14. decision READY เกิดเมื่อ required controls ผ่านด้วย current evidence ครบ; reportเขียนผ่าน artifact port แบบ immutable revision และเก็บประวัติ

Tests ต้องมี:
- header-only/truncated/corrupt DB, wrong configured path, broken schema และ failed quick_check -> ไม่ READY
- empty/missing/tampered manifest/input; manifest digest คำนวณได้แต่ expected bytes ไม่ตรง -> FAIL
- no execution evidence, stale tests, bypass sample gate และ evidence store read failure -> ไม่ READY
- valid end-to-end evidence fixture -> READY; tamper ทีละ required artifact แล้วผลต้องเปลี่ยนอย่างมีเหตุผล
- reconstruct หลัง restart ให้ audit verdict/evidence identity เดิมเมื่อ inputs เดิม ไม่พึ่ง wall-clock ที่ไม่ได้ freeze

เสร็จเมื่อ: ปิด R17 และย้าย readiness I/O ตาม R21; ไม่มี 10/10 ที่เกิดจาก placeholders/import/header checks

WP11 — Migration rehearsal, contracts และ final verification

ไฟล์หลัก:
- api/db/connection.py และ migration entry point ที่ผ่านการอนุมัติ
- tests/unit/db/, tests/unit/tools/, tests/api/, tests/architecture/
- scripts/generate_golden_manifests.py, scripts/export_openapi.py
- web/scripts/check_types_drift.mjs และ generated types

ก่อนใช้ schema ใหม่:
1. สร้าง migration draft ที่ review ได้ มี field definitions, indexes, foreign keys, old->new mapping และ failure/rollback behavior; ยังไม่รันกับฐานข้อมูลผู้ใช้ก่อน Gate DB-A/DB-B
2. inventory row counts/indexes/schema version จาก configured DB แบบ read-only และสร้าง consistent backup ด้วย SQLite backup API ที่รวม WAL state ได้ ไม่ใช้ file copy ที่อาจพลาด WAL
3. ซ้อมบนสำเนาและ fixtures ของ old schema ทั้งก่อน/หลัง portfolio_id; ตรวจ migrations idempotent และ failure ระหว่างขั้น rollback ได้
4. legacy signoffs/UUID exceptions เก็บ audit history เดิม; rows ที่ไม่มี source/snapshot identity ไม่ถูกยกระดับเป็น verified records อัตโนมัติ
5. ก่อน drop unique portfolio/date เดิม ตรวจการเก็บ legacy history แล้วสร้าง constraints ของ packet/revision/idempotency ใหม่ใน transaction ที่ควบคุม
6. ตรวจ row counts, Decimal text, timestamps, parent chains, hashes, foreign_key_check และ integrity_check หลัง migration
7. ทดสอบ import/startup เป็น read-only ต่อ schema ที่พร้อมแล้ว; migration ที่ rebuild table ต้องเป็น explicit controlled action ไม่เกิดจากการเปิดหน้าเว็บ
8. rollback plan ใช้ verified backup และเก็บ migration journal/ข้อมูลใหม่ที่เกิดหลัง backup ก่อน rollback; ไม่ใช้ downgrade ที่ลบทิ้ง approval ใหม่โดยเงียบ

Contract review หลัง behavior คงที่:
- จัดทำ endpoint/DTO/error-code diff ให้ตรวจ ก่อนขออนุมัติ baseline regeneration
- กำหนด OpenAPI ของ prepare (202/job_id), packet details/latest, required signoff fields, typed data quality และ conflicts
- หลังอนุมัติ ใช้ generator ของ repository เท่านั้น และตรวจว่า generated diff สะท้อน final API ไม่ใช่ข้อผิดพลาดที่ยังเปิดอยู่
- TypeScript client/types ต้องตรง OpenAPI; preserve compatibility wrappers เฉพาะที่ไม่ทำให้ packet/approval ambiguity

คำสั่ง targeted verification ระหว่างแก้ (เลือกตาม changeset):
- .venv\Scripts\pytest -o addopts= tests/api/test_institutional_gems_routers.py tests/api/test_pm_review_router.py tests/api/test_earnings_bridge_router.py -v
- .venv\Scripts\pytest -o addopts= tests/unit/application/test_pm_review_service.py tests/unit/application/test_thesis_service.py tests/unit/application/test_thesis_synthesis_service.py tests/unit/tools tests/unit/core tests/unit/thesis -v
- .venv\Scripts\pytest -o addopts= tests/unit/db tests/architecture tests/api/test_openapi_contract.py tests/unit/test_state_db_contract.py -v
- npm --prefix web test -- --run
- npm --prefix web run typecheck

หลังได้รับอนุมัติ baseline/types ตาม gate:
- .venv\Scripts\python scripts/generate_golden_manifests.py
- npm --prefix web run gen:types
- npm --prefix web run check:types
- ตรวจ generated diffs และ git diff --check

Final gates ก่อนถือว่าปิดงาน:
- .venv\Scripts\pytest tests/ -v
- npm --prefix web test -- --run
- npm --prefix web run typecheck
- git diff --check
- ทดสอบ migration/restart/concurrency/fault injection และ production-provider integration ที่ระบุในแต่ละ WP
- หากมี external integration ที่รันไม่ได้ ให้รายงานชื่อ/เหตุผลและแยกจาก passing suite; ห้ามสรุปทั้งระบบผ่านโดยอ้างเฉพาะ targeted tests
- การ broaden/repeat tests ทำเมื่อ changeset/failure/dependency ใหม่มีเหตุผล; หลัง checks ครบและไม่มีการเปลี่ยนเพิ่มให้ส่งมอบ ไม่วนรันทดสอบเดิมโดยไม่จำเป็น

เสร็จเมื่อ: relevant และ full gates ผ่าน, evidence/repro cases ที่เคยพบผิดผ่านทั้งหมด และไม่มี issue ที่ปิดด้วย mock/allowlist/baseline อย่างเดียว


ข้อเสนอ schema สำหรับตรวจรับ Gate DB-A (ยังไม่ใช่ migration ที่รันแล้ว)

| Resource | Fields / constraints ที่ต้องอยู่ใน draft | เหตุผล |
|---|---|---|
| pm_review_packets (ใหม่) | packet_id PK; portfolio_id; review_date; revision>0; cutoff_utc; schema_version; policy_version; source_fingerprint; packet_sha256; canonical_payload_json; quality; prepared_at_utc; unique(portfolio_id,review_date,revision); scoped semantic/hash uniqueness | freeze exact data และ reuse packet เมื่อไม่มี semantic change |
| pm_review_signoffs (ขยายของเดิม) | packet_id FK; signer_subject_id/session_id; verified_sample_ids_json; acknowledged_exception_ids_json; idempotency_key; request_sha256; signed_off_at_utc; preserve old signoff_id/date/notes | identity/evidence/retry ที่ตรวจกลับได้และรักษา old rows |
| signoff indexes/checks | unique packet_id สำหรับ approvals รุ่นใหม่; unique(portfolio_id,signer_subject_id,idempotency_key) เมื่อ key มีค่า; packet-scoped checks; legacy rows แยกด้วย packet_id null และ legacy audit status | เลิกจำกัด approval แค่วันละหนึ่งครั้งต่อพอร์ต โดยไม่สูญเสีย audit เดิม |
| pm_review_acknowledgements (ใหม่) | signoff_id + exception_id เป็น unique pair; event_fingerprint; portfolio/packet context ผ่าน signoff FK; acknowledged_at/notes | acknowledgement ของ portfolio A ไม่ทำให้ portfolio B ถือว่ารับทราบ source event ร่วมกันแล้ว |
| thesis_exceptions (ขยาย) | stable fingerprint, event type, rule/catalyst/claim version references, source revision, canonical event payload; unique logical fingerprint ตาม event scope | deduplicate repeated scan และรักษา material event changes |

ข้อกำหนดเพิ่ม:
- legacy rows ที่ identity/snapshot ไม่ครบแสดงเป็น legacy/unverified audit evidence; ห้ามเติม fabricated source/hash/user เพื่อให้ constraint ผ่าน
- payload bytes ใน packet immutable; metadata ที่เปลี่ยนตาม view/status แยกจาก payload
- event fact ใช้ร่วมข้ามพอร์ตได้ แต่ human acknowledgement ต้องผูก approval/portfolio; ไม่ใช้ global ACKNOWLEDGED flag เป็นสิทธิ์ของทุกพอร์ต
- first approval/retry/another actor ต้องคืน original approval identity ที่ตรงจริง หรือ conflict ตาม policy; ไม่สร้างลายเซ็นใหม่แทนคนเดิมโดยเงียบ
- foreign keys ต้องเปิดและมี tests บน connection จริง; constraints ใน DB เสริม application validation ไม่ใช้แทนกัน
- schema version/migration state ต้องระบุ explicit; raw DAOs ไม่เป็นเจ้าของ transaction
- DB-B มี draft field definitions ของแต่ละ repository และ recovery semantics ก่อน approval เช่นเดียวกัน ไม่มีการเพิ่มตารางตามชื่อในเอกสารนี้โดยอัตโนมัติ

ชุด regression ที่ต้องมีอย่างน้อย (ชื่อ test เป็นข้อเสนอสำหรับ implementation)

| Case | ไฟล์ทดสอบหลัก | ผลตรวจรับ |
|---|---|---|
| test_variant_missing_or_invalid_reported_returns_422 | tests/api/test_institutional_gems_routers.py | missing/bad/NaN/Infinity ไม่เป็น 500 |
| test_variant_unknown_estimate_returns_404 | ไฟล์ API เดียวกัน | ไม่มี baseline ไม่สร้างตัวเลขแทน |
| test_real_login_principal_reaches_mutation_routes | tests/api/test_institutional_gems_routers.py | auth guard เดิมคืน None ก็ไม่ทำ route crash |
| test_client_label_cannot_change_signer_subject | tests/api/test_pm_review_router.py | audit subject มาจาก trusted session |
| test_checksum_corruption_blocks_append_without_data_loss | tests/unit/tools/test_thesis_file_store.py | bytes เดิมคงอยู่และ append หยุด |
| test_stale_parent_and_competing_revision_conflict | tests/unit/tools/test_thesis_file_store.py | ไม่ append revision จาก head เก่า |
| test_projection_cannot_regress_after_slow_worker | tests/unit/tools/test_thesis_file_store.py หรือ adapter tests | v1 ช้ากว่า v2 ไม่เขียนทับ v2 |
| test_missing_fundamentals_are_unknown | tests/unit/application/test_thesis_service.py | missing ไม่เป็น healthy |
| test_partial_price_override_keeps_fundamental_observations | tests/unit/application/test_thesis_service.py | price override ไม่กลบ observer ทั้งชุด |
| test_material_catalyst_change_has_new_event_identity | tests/unit/core/test_catalyst_queue.py | date/classification/content เปลี่ยนไม่ reuse review |
| test_packet_get_is_read_only_and_hash_stable | tests/api/test_pm_review_router.py | GET ซ้ำไม่เขียน row/file หรือเปลี่ยน hash |
| test_signoff_retry_returns_same_complete_response | tests/api/test_pm_review_router.py | retry 200/schema เดิม ไม่มี duplicate |
| test_changed_packet_can_be_reviewed_as_new_revision | tests/api/test_pm_review_router.py | signoff ใหม่ไม่ติด unique portfolio/date |
| test_signoff_rolls_back_with_acknowledgement_failure | tests/unit/db/test_db_uow.py | approval/ack/outbox atomic |
| test_required_evidence_and_cross_portfolio_are_enforced | tests/api/test_pm_review_router.py | missing samples/hash/acks และ wrong scope ถูกปฏิเสธ |
| test_macro_reader_uses_production_dashboard_shape | tests/unit/application/test_pm_review_service.py หรือ new adapter test | nonzero values/regime จริงไม่หายเป็น default |
| test_zero_units_and_missing_fx_do_not_create_exposure | tests/unit/core/test_factor_exposure.py และ service tests | ไม่มี phantom holding/fallback currency |
| test_daily_pnl_excludes_external_flows | core analytics/service tests ที่รับผิดชอบ daily metrics | external capital flow ไม่เป็น profit |
| test_estimate_selection_is_approved_temporal_and_type_safe | tests/unit/core/test_estimate_ledger.py + service tests | no insertion-order/look-ahead/type-crossing |
| test_synthesis_response_equals_persisted_claim | tests/unit/application/test_thesis_synthesis_service.py | response/read-back ตรงทุก field |
| test_unsupported_evidence_cannot_be_documented | tests/unit/application/test_thesis_synthesis_service.py | source/threshold inference ไม่ได้ tier เกินจริง |
| test_restart_reconstructs_reviews_estimates_and_runs | repository/service integration tests ใหม่ | ไม่พึ่ง memory ของ process แรก |
| test_ui_discards_stale_response_and_blocks_old_packet_signoff | component tests ที่เกี่ยวข้อง | no response race / stale approval |
| test_readiness_rejects_header_only_db_and_empty_manifest | tests/unit/core/test_readiness_audit.py | fake setup ไม่ได้ READY |
| test_legacy_migration_preserves_history_and_is_idempotent | tests/unit/db/ และ adapter tests ใหม่ | old/new schema upgrade ถูกต้อง ไม่ fabricate snapshot |
| test_lower_layers_and_new_domains_keep_architecture_boundaries | tests/architecture/test_dependency_rules.py และ writer tests | ครอบคลุม core/thesis, core/macro, core/governance, core/resilience และ adapters ที่เพิ่มใหม่ด้วย |

ข้อควรทำใน tests:
- แยก API response tests, domain tests และ adapter transaction tests; test ที่ใช้ mock service ทั้งก้อนต้องไม่เป็นหลักฐานว่า production wiring ใช้ได้
- concurrency/fault tests ใช้ barriers/controlled hooks ที่ไม่ติด deadlock ภายใต้ lock ใหม่; มี timeout และไม่อาศัย sleep/ความเร็วเครื่อง
- snapshot/hash tests freeze semantic inputs และ clocks ที่เป็นข้อมูลจริง; ไม่ freeze volatile UUID เพื่อซ่อน design bug
- finite Decimal/rounding expected values มาจาก Python fixture/tool และ policy ที่ประกาศ ไม่คำนวณด้วย LLM
- full suite ต้องรักษา test isolation ทุก output path รวม review/routing ซึ่งก่อนหน้านี้ tests เขียนผ่าน production defaults ได้

Traceability เพื่อปิดทุกข้อ

| Issue | ชุดงานหลัก | หลักฐานที่จะใช้ปิด |
|---|---|---|
| N01 | WP01 | error-path API tests ผ่านและไม่มี NameError |
| N02 | WP01, WP05 | real-login mutations + trusted signer audit |
| N03 | WP07 | response/read-back equality |
| N04 | WP03, WP05, WP08 | stable packet, successful retry/new revision, atomic state |
| R01 | WP06 | production source records/availability และ no fixture fallback |
| R02 | WP04, WP09 | real Macro/beta/integrity wiring และ unavailable UI |
| R03 | WP01, WP05, WP09 | exact packet/evidence/identity enforcement |
| R04 | WP05, WP11 | existing portfolio isolation regression ยังผ่าน |
| R05 | WP01, WP02 | traversal/identifier boundary regression ยังผ่าน |
| R06 | WP02 | corrupted history ไม่ถูกใช้หรือเขียนทับ |
| R07 | WP02, WP08 | optimistic concurrency + monotonic projection |
| R08 | WP06, WP07 | primary evidence vs inferred policy แยกจริง |
| R09 | WP07, WP09 | THB/market/portfolio context regression ยังผ่าน |
| R10 | WP03, WP05 | complete rule statuses และ read-only GET |
| R11 | WP01, WP06, WP08 | approved temporal durable estimate ledger |
| R12 | WP03, WP08 | stable material-event identity + scoped review |
| R13 | WP08 | restart/cross-worker/recovery tests |
| R14 | WP04 | complete exposure/count/currency assertions |
| R15 | WP04 | dated movers + external-flow-aware P&L |
| R16 | WP01, WP02, WP06 | strict UTC/precision/effective availability tests |
| R17 | WP10 | negative readiness evidence cases ไม่ READY |
| R18 | WP01 | MNPI/invalid shape policy และ authenticated HTTP path |
| R19 | WP09 | response-race/stale-modal/error-refresh tests |
| R20 | WP07, WP09 | Hermes listing/production provider/harness regression |
| R21 | WP08, WP11 | architecture/storage/contracts/types/final suites |

แผนแบ่ง changesets สำหรับ review

1. Boundary fixes + regression characterization: WP00/WP01 — ทำให้ API/auth/time contracts ถูกต้องก่อน
2. Thesis safety: WP02 — corruption/concurrency/projection interfaces พร้อม tests
3. Observation/event domain และ analytics: WP03/WP04 — domain/ports/provider mapping ที่ไม่รอ migration
4. เสนอ DB-A/DB-B migration drafts พร้อม rehearsal results เพื่อขออนุมัติ; ทำงานส่วนที่ไม่ขึ้นกับ schema ต่อระหว่างรอได้
5. Packet approval flow: WP05 + adapters ส่วนจำเป็นของ WP08 — ส่งมอบ end-to-end prepare/read/review/retry/history
6. Source/ledger/synthesis: WP06/WP07 + durable repositories WP08 — ข้อมูลจริง/evidence tiers/response identity
7. UI/Hermes/worker integration: WP08/WP09 — production wiring/freshness/restart/races
8. Readiness + final contracts/gates: WP10/WP11 — ตรวจทั้ง positive และ negative behaviors ก่อน release review

การประมาณกำลังงานควรทำหลัง WP00 และ migration draft เพราะความพร้อมของ historical source records และ legacy state มีผลต่อขนาดงาน ไม่กำหนดเวลาเหมารวมที่ไม่มีหลักฐาน

Definition of Done ที่ต้องตรวจครบ

- [ ] N01–N04 และ R01–R21 มีผลตรวจรับ/issue mapping ครบ; ข้อที่แก้แล้วไม่ regress
- [ ] ไม่มี 500 จาก normal validation/auth flow และไม่มี user identity ที่เกิดจาก payload อย่างเดียว
- [ ] ไม่มี synthetic financial data หลุด production เป็น DOCUMENTED/verified โดยไม่มีแหล่ง
- [ ] corrupted thesis ไม่ถูกนำไปใช้งานหรือ overwrite; concurrent history/projection ไม่สูญหาย
- [ ] unchanged packet stable; new state เป็น new revision; signoff/retry/history/acknowledgement ตรง exact packet และ portfolio
- [ ] missing/stale/error ไม่กลายเป็น healthy/zero/neutral; UI แสดงข้อจำกัดและบล็อกการอนุมัติที่ไม่ผ่าน required policy
- [ ] estimates/catalysts/routines/reviews reconstruct หลัง restart และผ่าน worker fencing/recovery
- [ ] readiness ไม่มี PASS จาก import/header/hash-length checks เพียงอย่างเดียว
- [ ] approved migration rehearsal และ backup/recovery หลักฐานครบ ก่อนแตะ state ของผู้ใช้
- [ ] baseline/types อัปเดตหลังอนุมัติและตรง API final behavior
- [ ] relevant tests + full backend + frontend + typecheck + architecture + diff checks ผ่าน; unavailable external checks ระบุแยกชัด
- [ ] ส่งมอบ changelog ว่าแก้อะไร, tests ที่รัน, migration status, known limitations และไฟล์หลักฐาน โดยไม่ auto-deploy/merge/publish จากงานวางแผนนี้

ผลส่งมอบของคำขอปัจจุบัน: เอกสารแผนฉบับนี้เท่านั้น ไม่มีการอนุมัติหรือดำเนินการเปลี่ยน schema, dependencies, contract baselines หรือ implementation รวมอยู่ด้วย
