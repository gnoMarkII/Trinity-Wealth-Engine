แผนแก้ไขจากการตรวจโค้ดล่าสุด — 5 พฤศจิกายน 2026

ขอบเขต: working tree เทียบ HEAD fea7fbb รวมไฟล์ใหม่ที่ยังไม่ tracked ใน API, application, core, tools, frontend และ tests ที่เกี่ยวข้อง การตรวจนี้ยังไม่ได้แก้ production code, schema, dependencies หรือ golden manifests

ข้อสรุป: ยังไม่ผ่านเกณฑ์รวมงาน พบปัญหาการใช้ข้อมูลจริง การอนุมัติ ความสมบูรณ์ของ ledger และการเชื่อมต่อบริการ แผนนี้เรียงเป็น 21 ประเด็น พร้อมเกณฑ์ตรวจรับ

หลักฐานการทดสอบ

- Frontend: 123 test files / 621 tests ผ่าน; TypeScript typecheck ผ่าน
- Backend ชุดใหม่และ architecture: 215 ผ่าน / 1 ไม่ผ่าน
- Backend ชุด regression ที่เกี่ยวข้องและ OpenAPI: 48 ผ่าน / 1 ไม่ผ่าน
- รวม backend ที่รันจบ: 263 ผ่าน / 2 ไม่ผ่าน
- Test ที่ล้ม: tests/architecture/test_vault_write_boundaries.py::test_writer_inventory_has_no_unresolved_production_intent และ tests/api/test_openapi_contract.py::test_openapi_paths_and_operations
- Full backend suite รอบแรกถูกระบบรีสตาร์ตขัดจังหวะ จึงยังสรุปว่าทั้ง repository ผ่านไม่ได้
- ยืนยันหลายข้อด้วย diagnostic snippets ในหน่วยความจำและ temporary directories ไม่ทดลองเขียนข้อมูลเสียกับ portfolio/database จริง
- git diff --check พบ blank line ที่ EOF ในไฟล์ที่แก้เดิม 7 จุด ให้เก็บเป็นงาน cleanup หลังแก้สาระสำคัญ

รายการปัญหาและงานแก้ไข

1. [R01 / P1] Earnings Bridge และ Variant Perception ใช้ตัวเลขคงที่เป็นผลประกอบการจริง
   ตำแหน่ง: api/routers/equity/router_earnings_call.py:302; api/routers/equity/router_estimates.py:24; application/equity/estimate_service.py:36
   หลักฐาน: bridge มีข้อมูลเพียงสองชุดตาม suffix .BK และวันที่คงที่; variant endpoint ใส่ reported เองเมื่อไม่ส่งค่า; ledger เริ่มด้วยตัวอย่าง GOOGL ที่ติด DOCUMENTED โดยไม่มี source record ประกอบ เปลี่ยน ticker/period ก็ยังใช้ตัวเลขเดิมได้
   แก้: ใช้ application service ผ่าน ports อ่าน actual report/frozen consensus ตาม ticker, period, cutoff; แยก fixtures ออกจาก production; ขาดข้อมูลให้ unavailable พร้อมเหตุผล
   ตรวจรับ: หุ้น/งวดต่างกันใช้คนละ source record; ไม่มีข้อมูลต้องไม่คืนตัวเลขหรือคำตัดสินยืนยัน thesis จากค่าตัวอย่าง

2. [R02 / P1] Morning Review ผสมข้อมูลจริงกับ default ที่แต่งไว้
   ตำแหน่ง: application/manager/pm_review_service.py:147, :291; api/dependencies.py:771
   หลักฐานที่ทดลอง: service ไม่มี readers ยังคืน NAV 1250000.0, beta 1.05 และ Late Cycle Expansion; production provider ไม่ส่ง macro_reader, benchmark_return_reader หรือ state_integrity_checker; reader ล้มแล้วเก็บค่าตัวอย่างเดิม
   แก้: เอา financial defaults ออก; ต่อ readers จริง; ใช้สถานะ missing/stale/error และ provenance ต่อ field; ปรับ DTO/UI รองรับ unavailable
   ตรวจรับ: empty portfolio, failed reader, missing benchmark/macro ไม่แสดงค่าตัวอย่าง; integrity check ที่รันไม่ได้ต้องแสดงข้อผิดพลาด

3. [R03 / P1] Sign-off ไม่ตรวจ packet, samples และตัวผู้ลงนาม
   ตำแหน่ง: application/manager/pm_review_service.py:533, :550; api/routers/pm_review_router.py:88
   หลักฐานที่ทดลอง: hash arbitrary-invalid-hash กับ sample ID ที่ไม่มีจริงยังได้รับ SIGNED_OFF; เมื่อส่ง hash จะบันทึก exception count เป็น 0; GET ถัดไปเช็คเพียงว่ามี signoff ของวันนั้น โดยไม่เทียบ checksum กับ packet ปัจจุบัน
   แก้: เก็บ immutable packet snapshot; ตรวจ hash/required sample IDs ฝั่ง server; ใช้ authenticated identity; acknowledge เฉพาะ exceptions ที่ review จริงใน transaction เดียว; แยก packet ที่เปลี่ยนจาก packet ที่อนุมัติแล้ว; ทำ retries ให้ idempotent
   ตรวจรับ: hash ปลอม, stale packet, sample ขาด/ปลอม และผู้ลงนามปลอมถูกปฏิเสธ; packet เปลี่ยนกลับ REVIEW_REQUIRED; retry ไม่เป็น 500 จาก unique constraint

4. [R04 / P1] Sign-off ของพอร์ตหนึ่งทำให้อีกพอร์ตถูกอนุมัติด้วย
   ตำแหน่ง: api/db/connection.py:430; application/manager/ports.py (PMReviewSignoffRepoPort); application/manager/pm_review_service.py:519; web/src/components/manager/MorningDeltaCard.tsx:76
   หลักฐานที่ทดลอง: ลงนามแล้ว build packet ของ portfolio_id อื่นในวันเดียวกันได้ SIGNED_OFF; repo key มีเพียง review_date; sign-off request ไม่มี portfolio_id
   แก้: เพิ่ม portfolio identity ให้ packet, checksum, request, repo query, uniqueness และ history; scope ต่อ portfolio/packet version
   ตรวจรับ: อนุมัติ A แล้ว B ยัง REVIEW_REQUIRED; history และ retries แยกกัน
   ก่อน implementation: ขออนุมัติ schema/migration ตาม AGENTS.md ก่อนทำส่วนฐานข้อมูล

5. [R05 / P1] Ticker พา path ออกนอก thesis store ได้
   ตำแหน่ง: tools/thesis/thesis_file_store.py:44; application/thesis/thesis_service.py:61; api/routers/thesis_router.py:121
   หลักฐานที่ทดลอง: ticker ที่มี Windows backslash และ parent traversal ทำให้ resolved JSON path อยู่นอก data_dir; จุดเข้าใหม่เพียง strip/upper โดยไม่ใช้ validate_ticker
   แก้: validation กลางที่ service และ adapter boundary; ตรวจ resolved path อยู่ภายใน root ทั้ง JSON/Markdown; reject drive/absolute/separators/parent segments ก่อน I/O
   ตรวจรับ: read/write ทั้ง slash, backslash, encoded separators และ drive paths ไม่แตะไฟล์นอก root

6. [R06 / P1] Thesis store ใช้ข้อมูล checksum เสีย และเขียนทับประวัติที่อ่านไม่ได้
   ตำแหน่ง: tools/thesis/thesis_file_store.py:72
   หลักฐานที่ทดลอง: แก้ claim_statement โดยไม่แก้ checksum แล้วยังได้ active thesis ที่ verify_integrity() เป็น False; malformed JSON ถูกอ่านเป็น [] แล้ว record_new_thesis เขียน ledger ใหม่ทับได้
   แก้: แยก not-found ออกจาก corruption/read error; quarantine record ที่ไม่ผ่าน integrity; ห้าม append/overwrite เมื่อ history เสีย; เก็บหลักฐานสำหรับ recovery; ตรวจ identifier และ linkage ใน integrity policy
   ตรวจรับ: mismatch/malformed JSON หยุดการใช้และการเขียนทับ โดยรักษาไฟล์เดิมไว้

7. [R07 / P1] Concurrent append ทำประวัติหายแม้ใช้ atomic replace
   ตำแหน่ง: tools/thesis/thesis_file_store.py:52; application/thesis/thesis_service.py:102
   หลักฐานที่ทดลอง: สอง save_claim สำเร็จพร้อมกัน แต่เหลือเพียงหนึ่ง claim; read-modify-write ไม่มี lock; version/parent ถูกเลือกนอก transaction
   แก้: locking ข้าม process ด้วย dependency ที่มีอยู่ หรือ transactional repository; ทำ load/version/parent/duplicate validation/append ใน critical section เดียว; ตรวจ optimistic parent/version และ parent ticker
   ตรวจรับ: concurrent create/revise ไม่หาย ไม่ซ้ำ version ไม่อ้าง parent ผิด ticker; conflict รายงานชัดและไม่ทำลายประวัติ

8. [R08 / P1] Thesis synthesis สร้างข้ออ้างและระดับหลักฐานเกินข้อมูล
   ตำแหน่ง: application/thesis/thesis_synthesis_service.py:113, :143, :161, :200, :214, :252; api/routers/thesis_router.py:322
   หลักฐาน: ไม่มีราคาก็สร้าง stop loss คงที่; growth/margin ใช้ default; catalyst ทุกหุ้นวันที่ 2026-12-15; ใช้ FCF margin เป็น operating_margin; ตรวจค่าที่อนุมานเองกับ sidecar ด้วย tolerance 5000 bps แล้วให้ DOCUMENTED; มีราคาอย่างเดียวก็อาจให้ DIRECTLY_TESTED; evidence enum สะกดผิดกลับตกเป็น DOCUMENTED
   แก้: แยก fact/assumption/PM policy; ไม่สร้าง threshold/date เมื่อขาดหลักฐาน; map metric/unit ให้ตรง; เก็บ source locator/hash/cutoff ต่อข้ออ้าง; ใช้ Decimal ตั้งแต่ต้น; derived/custom hypotheses จัดเป็น INFERRED ตามส่วนที่ใช้; reject invalid evidence enums
   ตรวจรับ: missing source ไม่กลายเป็น documented/tested claim; FCF ไม่ถูกตรวจแทน operating margin; วัน catalyst มาจาก calendar จริงหรือแสดง missing

9. [R09 / P1] สกุลเงินหุ้นไทยและ portfolio context ใน batch synthesis ผิด
   ตำแหน่ง: application/thesis/thesis_synthesis_service.py:84, :92, :329
   หลักฐานที่ทดลอง: sidecar market=TH ของ CPALL ที่มีราคาอยู่แล้วให้ falsifier unit=USD; is_us ใช้ OR กับการไม่มี .BK; batch รับ portfolio_id แต่ fallback price ของแต่ละ ticker อ่าน default เสมอ
   แก้: normalize market/native_currency จาก metadata; ใช้ canonical identity ที่ไม่ทำให้หุ้นต่างตลาดชนกัน; ส่ง portfolio_id/holding context ตลอด flow
   ตรวจรับ: CPALL และ CPALL.BK ที่ระบุ TH ใช้ THB; batch B อ่านราคาของ B; ไม่ใช้ coverage ของหุ้นชื่อคล้ายกันต่างตลาดแทนกัน

10. [R10 / P1] Watchdog ไม่แยก not evaluated จาก healthy และ refresh สร้าง exceptions ซ้ำ
    ตำแหน่ง: api/dependencies.py:730; application/thesis/thesis_service.py:182; core/thesis/evaluator.py:170; application/manager/pm_review_service.py:346
    หลักฐาน: provider ไม่มี MarketObservationPort; morning packet ส่งเพียง current_price ทำให้ growth/margin ไม่ถูกประเมิน; missing metric ถูกข้ามแต่ UI แสดง Thesis Intact; GET morning/factor exposure เรียก scan ที่สร้าง UUID ใหม่และ insert ซ้ำได้
    แก้: ต่อ observation provider พร้อม metric/unit/period/cutoff; แสดง evaluated/unknown/stale ต่อ rule; validate operator/finite Decimal; แยก scan/persist ออกจาก read-only packet assembly; ใช้ stable breach identity/idempotency และโหลด acknowledgement จาก store
    ตรวจรับ: metric ขาดไม่สรุปว่าปกติ; breach เดิมไม่เพิ่ม record ทุก refresh; event ใหม่เปิด exception ใหม่; bad operator/NaN ถูก reject

11. [R11 / P1] Estimate revision ไม่บังคับ ledger invariants
    ตำแหน่ง: core/equity_analytics/estimate_ledger.py:220; application/equity/estimate_service.py:98
    หลักฐานที่ทดลอง: revision คนละ ticker/period/metric และ driver sum ไม่ตรงยัง append ได้; REVIEW_REQUIRED row ท้ายสุดถูกใช้แทน APPROVED baseline; service ไม่ส่ง unit เข้า calculation จึงใช้ USD_M เสมอ
    แก้: ตรวจ identity/type/unit/unique ID/parent chain; reject หรือแยก unbalanced draft ออกจาก usable ledger; เลือก approved/available-before-cutoff ตามเวลา; ไม่แทน missing consensus ด้วย zero หรือ missing fund thesis ด้วย consensus โดยเงียบ
    ตรวจรับ: cross-series/unbalanced/duplicate/pending/future revisions ไม่เปลี่ยน approved view; EPS/margin/revenue รักษาหน่วยถูกต้อง

12. [R12 / P1] Catalyst acknowledgement เก่ากลบ event ใหม่ และ scan ใช้ future captures
    ตำแหน่ง: application/thesis/catalyst_service.py:101; core/thesis/catalyst_queue.py:163, :258
    หลักฐานที่ทดลอง: review NO_CHANGE แล้วเพิ่ม source ใหม่ยังคืน NO_CHANGE/ACKNOWLEDGED/source เก่า เพราะ ID อ้าง entity+ลำดับเดิม; capture เดือนถัดไปถูกจัด NEW_SOURCE ใน cutoff ก่อน capture
    แก้: identity จาก catalyst/event revision/source fingerprint; merge assessment เฉพาะ event เดิม; filter captured/available time ก่อน cutoff และเลือก source eligible ล่าสุดตามเวลา
    ตรวจรับ: source/date/classification ใหม่กลับ REVIEW_REQUIRED; event เดิมรักษา assessment; future capture ไม่เปลี่ยน historical packet

13. [R13 / P1] State สำคัญอยู่ใน memory และ routine monitoring ยังไม่ต่อ workflow จริง
    ตำแหน่ง: application/thesis/catalyst_service.py:28; application/equity/estimate_service.py:29; application/resilience/routine_monitor_service.py:26; application/governance/routing_service.py:30
    หลักฐาน: reviews/captures/estimates/heartbeats อยู่ใน process-local list/dict; routing history ไม่โหลดไฟล์กลับ; register_catalyst/register_capture ยังไม่มี production call site เชื่อม ingestion/theses; heartbeat มีเพียง manual endpoint
    แก้: persistence ports/adapters ที่อ่านกลับได้; ต่อ calendar/filing/thesis ingress และ workers; บันทึก heartbeat start/completion/failure; ใช้ cutoff/timezone เดียวกันแทน 06:00 UTC ใน service กับ 08:30 ไทยใน UI; timeout งานค้าง
    ตรวจรับ: restart/อีก worker อ่าน state เดียวกัน; catalyst ใหม่เข้าคิวจริง; routine สำเร็จ/ล้ม/ขาดตรง execution จริงโดยไม่ต้องส่ง heartbeat ด้วยมือ

14. [R14 / P1] Exposure มี phantom holdings และ count/missing-price metadata ผิด
    ตำแหน่ง: application/manager/pm_review_service.py:203; api/routers/portfolio/router_advisory.py:84; core/equity_analytics/factor_exposure.py:198
    หลักฐานที่ทดลอง: units=0, price=100 ถูกแทน units เป็น 1 จึงมี gross exposure=100.0; fallback อาจนำ USD มารวม THB; missing hard-fail เก็บเพียงตัวแรกแล้วบาง field คง default; endpoint นับ eligible จาก quarantined และ filter รับ code เดียวทั้งที่อีกเส้นใช้ MISSING_AS_OF_PRICE; core count รวม positions ที่ skip
    แก้: กรองตาม status/actual signed units; ไม่เดาหน่วย/ราคา; currency conversion ด้วย approved FX ณ cutoff; ใช้ Decimal ไม่หาร float ก่อน; ส่ง eligible/missing ทั้งหมดจาก result เดียวและรวม guardrail vocabulary; ระบุ partial exposure หรือหยุดตาม policy
    ตรวจรับ: zero/archived/missing/short/mixed currency มีมูลค่าและ count ที่ตรวจกลับได้; missing_symbols ครบ; ไม่มี default exposure เมื่อคำนวณล้ม

15. [R15 / P2] Daily movement ใช้ lifetime return และ NAV delta ปะปน P&L
    ตำแหน่ง: application/manager/pm_review_service.py:188, :267; core/portfolio_advisory/review_sampling.py:150
    หลักฐานที่ทดลอง: unrealized_pnl_percent=80 กลายเป็น Top Daily PnL Mover; daily_pnl ใช้สอง NAV แถวสุดท้าย ไม่ยืนยันคนละวันและไม่ปรับ deposits/withdrawals
    แก้: ใช้ dated daily observations; คำนวณ daily return/P&L ด้วย Decimal และปรับ cash flows; ข้อมูลไม่พอให้ unavailable; sampling รับ daily movers ที่ตรวจแล้ว
    ตรวจรับ: lifetime gain ไม่แทน daily move; cash flow ไม่กลายเป็นกำไร; snapshot วันเดียวกันไม่อ้างเป็น 24h

16. [R16 / P2] Point-in-time helpers เปรียบเทียบ timestamp เป็นข้อความ
    ตำแหน่ง: core/provenance.py:32, :54; application/thesis/thesis_service.py:138
    หลักฐานที่ทดลอง: available=2026-11-05T08:00:00+07:00 กับ cutoff=2026-11-05T02:00:00Z ให้ False แต่ aware datetime ให้ True; Z/+00:00/fractional seconds ก็เรียงผิดได้
    แก้: parse/validate/normalize UTC ที่ boundary; ใช้ aware datetime; นิยาม revised-time และ active-as-of policy; ไม่ใช้ version แทน effective time เมื่อรับ backdated input
    ตรวจรับ: timezone representations เวลาเดียวกันเท่ากัน; offset/fractional/backdated/future cases ถูกต้อง

17. [R17 / P1] Readiness Audit รับรองความพร้อมโดยไม่ตรวจ state จริง
    ตำแหน่ง: application/resilience/readiness_audit_service.py:131, :218, :296, :317
    หลักฐานที่ทดลอง: temp repository มีเพียง .gitignore และไฟล์ชื่อ DB ที่เป็นข้อความเสีย ได้ 10 PASS และ READY_FOR_NAMED_STAGE; Q1 เช็ค exists; Q5 เช็คความยาว hash; Q8 เช็ค import; Q9 PASS ตายตัว
    แก้: read-only database adapter ตาม configuration จริง; verify manifest กับไฟล์จริง; ใช้ execution evidence ของ integration checks/human gate/security log; ขาดหลักฐานต้อง fail/unknown; แยก automated audit จาก human sign-off
    ตรวจรับ: corrupt DB, missing/tampered inputs, bypassed human gate และ evidence log หาย ไม่ให้ READY; output ต้องอ้างผลที่ตรวจกลับได้

18. [R18 / P2] Governance ส่ง MNPI work shape และ invalid shape ไป LLM ได้
    ตำแหน่ง: core/governance/task_router.py:105, :157; api/routers/governance_router.py:21
    หลักฐานที่ทดลอง: MNPI_OR_INCIDENT_REVIEW กับ default context ให้ LLM_RESEARCH_SYNTHESIS; invalid work_shape ถูกเปลี่ยนเป็น routine reading โดยเงียบ
    แก้: sensitive shape ไป quarantine/human policy review; ใช้ DTO enum และ reject invalid เป็น 422; ตรวจ policy context จาก trusted source ก่อน dispatch
    ตรวจรับ: sensitive task ไม่ไป generic LLM เพราะลืม flag; typo ไม่เปลี่ยน work type เงียบ

19. [R19 / P2] UI loaders ไม่ป้องกันข้อมูลค้างเมื่อ props เปลี่ยนหรือ response สลับลำดับ
    ตำแหน่ง: web/src/components/equity/EarningsCallTab.tsx:151; web/src/components/equity/StructuredThesisCard.tsx:54; web/src/components/manager/MorningDeltaCard.tsx:33
    หลักฐานจากโค้ด: bridge โหลดเมื่อ bridgeData ว่างเท่านั้น จึงไม่โหลด ticker ใหม่หาก instance เดิมรับ prop ใหม่; ไม่มี cancellation/request identity; MorningDelta เก็บ packet และ verified samples เดิมขณะเปลี่ยน portfolio
    แก้: key cache ตาม identity; reset state/selections เมื่อเปลี่ยน ticker/period/portfolio; AbortController/request-generation guard; ห้าม sign-off ระหว่าง loading/stale/error และแสดง refresh error
    ตรวจรับ: rerender A->B และ response A กลับทีหลังไม่แสดงข้อมูล A; เปลี่ยน portfolio ไม่อนุมัติ packet/sample เก่า

20. [R20 / P2] Hermes อ่าน theses ทั้งหมดไม่ได้
    ตำแหน่ง: api/dependencies.py:1194
    หลักฐาน: provider เรียก list_all_theses() ที่ไม่มีใน ThesisApplicationService; method จริงคือ list_active_theses(); diagnostic ยืนยัน missing method จึงคืน error เมื่อ tool ไม่ระบุ ticker
    แก้: เรียก port/method ถูกต้อง; กำหนด schema สำหรับ active claims; integration test ผ่าน production provider แบบไม่มี ticker; cleanup get_position_review_service ที่นิยามซ้ำใน DI
    ตรวจรับ: get_active_stock_theses() คืน claims ทั้งหมดหรือ empty state ได้จริงโดยไม่ mock service ทิ้ง

21. [R21 / P1 release gate] Storage boundary และ API contract ยังไม่ผ่าน
    ตำแหน่ง: tools/thesis/thesis_file_store.py:67, :170; application/governance/routing_service.py:70; application/resilience/readiness_audit_service.py:351; api/main.py:168
    หลักฐาน: writer scanner พบ unresolved สองจุดใน thesis_file_store; OpenAPI พบ route ใหม่ 21 paths นอก baseline; application writers ใช้ write_text/fixed temp เอง แทน shared _atomic_io และ persistence ports
    แก้: ต่อ writes เข้า storage contract/broker พร้อม owner/profile ตามนโยบายจริง; persistence อยู่ใน adapters; ใช้ shared atomic writer/unique temp/locks; strict DTO validation และ domain status 422; ทบทวน auth/contracts แล้ว sync generated TypeScript types
    ตรวจรับ: architecture ไม่มี unresolved; contracts/types ตรง final behavior; invalid Decimal/date/operator/evidence/work shape ได้ validation error; ไม่เพิ่ม allowlist เพื่อซ่อน bypass
    ก่อน baseline change: ขออนุมัติตาม AGENTS.md แล้วใช้ generate_golden_manifests.py เท่านั้น ห้ามแก้ manifest_*.json ด้วยมือ

ลำดับดำเนินงาน

- ระยะ A: หยุดข้อมูลผิดและรักษาหลักฐาน — R01, R02, R05, R06, R08, R09, R17
- ระยะ B: ทำ state/approval ถูกต้อง — R03, R04, R07, R10, R11, R12, R13
- ระยะ C: แก้ analytics/policy — R14, R15, R16, R18
- ระยะ D: เชื่อม UI/Hermes และปิด gates — R19, R20, R21
- เพิ่ม negative/concurrency/restart/cross-portfolio tests ตามข้อผิดพลาดที่ยืนยันแล้ว และ integration tests ผ่าน production providers
- ใช้ dependencies เดิมก่อน; ต้องขออนุมัติก่อนเพิ่ม Python/npm package
- การตรวจครั้งนี้ไม่ได้อนุมัติ schema changes หรือ manifest regeneration; ขอเฉพาะก่อนลงมือเปลี่ยนส่วนนั้น

คำสั่งปิดงานหลัง implementation

- .venv\Scripts\pytest tests/ -v
- npm --prefix web test -- --run
- npm --prefix web run typecheck
- architecture, API contract, relevant backend regression และ frontend gates ต้องผ่านทั้งหมด
- ถ้า full suite ติด external provider/environment ให้แยกสิ่งที่ทดสอบไม่ได้ออกจากผลที่รันจบจริง

ผลส่งมอบของรอบนี้คือแผนแก้ไข ยังไม่มีการแก้ implementation
