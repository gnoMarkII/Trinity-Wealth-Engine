# ผลตรวจซ้ำรอบที่ 4 — 2026-11-06

ตรวจ working tree ล่าสุดบน HEAD `fea7fbb` เทียบข้อค้างรอบที่ 3 และแผน `review/detailed-remediation-plan.md` ผลคือ **ยังมีจุดต้องแก้** แม้ชุดทดสอบเดิมผ่านทั้งหมดที่รันในรอบนี้

การตรวจรอบนี้ไม่ได้แก้ production code, เปลี่ยน schema, ติดตั้ง dependency หรือ regenerate manifests ใช้ฐานข้อมูล/ไฟล์ชั่วคราวสำหรับ probes และลบหลังตรวจเสร็จ

## จุดที่ยังต้องแก้ เรียงตามความสำคัญ

### J01 — P1: ลงนามได้โดยไม่ส่ง checksum และไม่ยืนยันตัวอย่าง

ตำแหน่ง: `api/routers/pm_review_router.py:20`, `application/manager/pm_review_service.py:651`, `:657`

DTO ยังอนุญาต `packet_sha256=None` และ `verified_sample_ids=[]` ส่วน service ตรวจเฉพาะเมื่อค่าที่ส่งมาเป็น truthy จึงข้ามทั้งสอง gate ได้

ยืนยันด้วย service + SQLite adapter จริง: packet มี 1 review sample และ 1 thesis exception แต่เรียก sign_off โดยไม่ส่ง checksum และส่ง samples ว่าง ได้ `SIGNED_OFF` และบันทึก samples ว่างจริง

แนวแก้: บังคับ client ส่ง hash ของ packet ที่เปิดตรวจ ตรวจ required samples ตาม policy ฝั่ง server และผูก evidence กับ packet revision ที่อนุมัติ หาก packet ไม่มี samples ต้องแยกกรณีที่ policy อนุญาตไว้อย่างชัดเจน ตรวจรับด้วย API tests สำหรับ missing/null/empty hash, samples ไม่ครบ และ stale packet อ้างอิง WP05

### J02 — P1: ลงนามพร้อมกันเขียนทับผู้อนุมัติ และตอบ ID ที่ไม่มีในประวัติ

ตำแหน่ง: `application/manager/pm_review_service.py:665`, `:684`, `:707`; `tools/thesis/thesis_sqlite_dao.py:247`

การตรวจ existing กับการ save แยก transaction ขณะที่ DAO ใช้ `ON CONFLICT(portfolio_id, packet_sha256) DO UPDATE` แก้ signer/time/notes/evidence ของแถวเดิม แต่ไม่เปลี่ยน signoff_id

ยืนยันด้วยสอง threads และ SQLite adapter แบบเปิด connection จริง โดย synchronize ให้ทั้งสองอ่านว่าไม่มี approval ก่อน save: ทั้งสองตอบสำเร็จด้วยคนละ ID แต่ DB เหลือหนึ่งแถว ID ของ reviewer-A กลับมี signer เป็น reviewer-B และ ID ที่ตอบให้ reviewer-B ค้นไม่พบ

แนวแก้: ทำ create-or-return-existing ภายใน transaction เดียว ใช้ conflict แบบไม่แก้ audit record เดิม แล้วคืนแถว canonical ที่บันทึกจริง ให้ retry/concurrent request ได้ ID และ evidence เดียวกัน ทดสอบพร้อมกันหลาย connection; packet ต่าง revision ต้องยังเก็บประวัติครบ อ้างอิง WP05

### J03 — P1: Readiness audit ยังให้ READY โดยไม่มีหลักฐาน state ที่เพียงพอ

ตำแหน่ง: `application/resilience/readiness_audit_service.py:146`, `:297`, `:378`; `scripts/institutional_readiness_audit.py:41`

มีสองกรณีที่ยืนยันซ้ำแล้ว:

- Default service ให้ 10/10 PASS และ `READY_FOR_NAMED_STAGE` กับไฟล์ 4096 bytes ที่มี header/page-size/B-tree flag ตามที่โค้ดตรวจ แต่ `sqlite3 PRAGMA quick_check` เปิดแล้วได้ `DatabaseError: file is not a database`
- แม้ใช้ integrity checker ของ CLI จริง ฐาน SQLite ที่มีเพียงตาราง `unrelated` กับ storage_contract ของ repo และไม่มี protected state ก็ยังได้ 10/10 READY เพราะ checker ตรวจแค่มีอย่างน้อยหนึ่งตาราง และ Q5 ให้ PASS จากรูปแบบ storage configuration โดยไม่ตรวจ checksum ของ protected files

CLI แก้กรณีไฟล์เสียให้ตรวจ quick_check แล้ว แต่ยังไม่ตรวจ required schema/state ส่วน Q8 ยัง PASS เพียง import class สำเร็จ จึงยังไม่พิสูจน์การ reconstruct packet จริง

แนวแก้: ใช้ integrity-check port ที่ตรวจจริงทุก entry point ตรวจ required schema/version แบบ read-only; Q5 ต้อง verify frozen manifest กับไฟล์จริงและไม่แทนด้วย write-policy config; Q8/Q9 ต้องใช้ผล rebuild/negative gate ที่ตรวจซ้ำได้ ถ้าหลักฐานขาดให้ FAIL/CONDITIONAL ตรวจรับด้วย corrupted DB, unrelated-schema DB, missing/tampered protected files และ reconstruct failure อ้างอิง WP10

### J04 — P1: Watchdog ข้าม fundamental rules แม้มี observer

ตำแหน่ง: `api/dependencies.py:716`; `application/thesis/thesis_service.py:369`; `application/manager/pm_review_service.py:362`

Production provider ยังไม่ต่อ market_observer และใน service ใช้ elif: เมื่อ PM ส่ง current_price ใน observed_metrics_by_ticker แล้ว จะไม่อ่าน metrics อื่นจาก observer

ยืนยันด้วย thesis ที่มี price rule และ revenue-growth rule: observer ให้ growth ต่ำกว่า floor แต่ PM packet ได้ 0 exceptions และไม่มีการเรียก observer เลย ส่วน direct watchdog scan ที่ไม่ส่ง price override ตรวจพบ revenue-growth breach เดียวกัน

แนวแก้: เติมข้อมูลแยกตาม required metric; override เฉพาะ metric ที่ให้มาและอ่านค่าที่ยังขาดผ่าน port ต่อ observation adapter ใน production พร้อม source/unit/as-of metadata และสถานะ UNKNOWN/STALE/ERROR สำหรับค่าที่ประเมินไม่ได้ ห้ามสรุปว่าไม่มี breach จากการไม่มีข้อมูล ตรวจรับทั้ง direct scan และ PM packet integration อ้างอิง WP03

### J05 — P1/P2: ค่า unavailable จาก API ทำให้ Morning Review พัง หรือแสดงเป็นค่าศูนย์

ตำแหน่ง: `api/dependencies.py:813`; `web/src/components/manager/MorningDeltaCard.tsx:343`, `:349`, `:394`

Macro reader ส่ง null เมื่อไม่มีข้อมูล แต่ UI เรียก `yield_spread_10y_2y_bps.toFixed(1)` โดยไม่ตรวจ null จึง crash จริงใน React render probe นอกจากนี้ backend แก้ beta ที่ไม่มี observations เป็น None แล้ว แต่ UI ตรวจเพียง !== undefined และใช้ Number(null) จึงแสดง beta เป็น 0.00

แนวแก้: ให้ TS DTO สอดคล้องกับ nullable API fields; ตรวจ null และ finite number ทุกจุดก่อน format แสดง unavailable โดยไม่แปลงเป็นศูนย์ เพิ่ม component tests จาก response ที่ไม่มี macro/beta จริง อ้างอิง WP04/WP06

### J06 — P1: ยังให้สถานะ verified/DOCUMENTED โดยไม่มี independent source verification

ตำแหน่ง: `api/routers/equity/router_earnings_call.py:296`, `:380`; `application/thesis/thesis_synthesis_service.py:234`, `:275`, `:286`

1. Earnings bridge ยังอ่าน actual/consensus/drivers จาก `_AUTHORITATIVE_EARNINGS_DISCLOSURES` ที่เขียนตัวเลขไว้ใน router และใช้ as_of คงที่ โดยไม่มี source record ให้ตรวจย้อนกลับ การเปลี่ยนชื่อเป็น authoritative ไม่ได้เพิ่มหลักฐานการ verify
2. Thesis synthesis สร้างทั้ง claimed values และ primary_facts จาก sidecar ชุดเดียวกัน แล้วเทียบค่ากับตัวเอง และสร้าง source_locator ข้อความทั่วไป ยืนยันด้วย sidecar สมมติที่ไม่มี filing locator/hash/source timestamps ก็ยังได้ `DOCUMENTED`

แนวแก้: แยก fixture/demo ออกจาก production provider; อ่าน actual/consensus ตาม ticker/period/source revision ที่เก็บหลักฐานได้ อย่าระบุว่า verified หากยังไม่มี source; synthesis ต้องตรวจ claim กับ primary evidence ที่เป็นอิสระจาก claim และลด tier เมื่อหลักฐานไม่ครบ ตรวจรับด้วย sidecar ไม่มีแหล่งอ้างอิง/แหล่งไม่ตรง/ข้อมูลต่าง revision/period ไม่พร้อม อ้างอิง WP04/WP07

หมายเหตุ: รอบนี้ไม่อ้างว่าตัวเลขบริษัทใดผิดจากเอกสารภายนอก ประเด็นที่ยืนยันคือ code path ไม่มีหลักฐานรองรับสถานะที่แสดง

### J07 — P2: Daily P&L ยังเป็นผลต่าง snapshot สองแถวโดยไม่ตรวจวันและ cash flows

ตำแหน่ง: `application/manager/pm_review_service.py:298`; `web/src/components/manager/MorningDeltaCard.tsx:319`

Probe ใช้ snapshots วันที่ 2026-10-01 และ 2026-11-01 แล้วขอ review วันที่ 2026-11-06 ระบบยังแสดงผลต่างเป็น daily P&L 50.0 / 50.0% ทั้งที่ไม่มีข้อมูลของวัน review สูตรยังไม่แยกเงินฝาก/ถอนออกจาก NAV change และใช้ float กับ financial calculations

Backend เพิ่ม `is_lifetime_fallback` ให้ top_mover แล้ว แต่ UI ไม่ใช้ flag จึงยังแสดง unrealized return ตั้งแต่ซื้อเป็น Top Mover ของ daily packet

แนวแก้: กำหนด as-of/date window ตรวจความใหม่และลำดับ snapshots; หากข้อมูลไม่พอให้ unavailable; ใช้ net external flows คำนวณผลการลงทุนอย่างถูกนิยามผ่าน pure Decimal calculation; ไม่แสดง lifetime return ในนาม daily movement ตรวจรับ stale/nonconsecutive snapshots, cash deposit/withdraw, zero return และ precision boundaries อ้างอิง WP06

### J08 — P2: Factor exposure นับราคาศูนย์ว่า eligible และส่งรายชื่อ missing prices ไม่ครบ

ตำแหน่ง: `application/manager/pm_review_service.py:258`, `:271`, `:519`; `api/routers/portfolio/router_advisory.py:83`

Probe สอง holdings ที่มี units แต่ไม่มีราคา และ market_value_thb=0 ได้ eligible_positions_count=2 ทั้งที่ควรไม่มี verified position ทั้งสองอยู่ใน quarantine แต่คนแรกใช้ `MISSING_PRICE_EXCEPTION` คนที่สองใช้ `MISSING_AS_OF_PRICE` ทำให้ endpoint missing_price_symbols ส่งเพียงชื่อแรก

แนวแก้: จำแนกทุก position ก่อนคำนวณ ให้ eligible ใช้ราคา finite > 0 และ units ที่ใช้คำนวณได้; collect missing symbols ทุกตัวด้วย code contract เดียว แล้วคำนวณ verified subset ตรวจรับหลาย missing holdings, zero/negative price, zero units และ positions ที่ผ่านจริง อ้างอิง WP06

### J09 — P2: Response เก่าเขียนทับ ticker/portfolio ที่เลือกใหม่

ตำแหน่ง: `web/src/components/equity/StructuredThesisCard.tsx:54`; `web/src/components/manager/MorningDeltaCard.tsx:32`, `:43`

ทั้งสอง component ไม่มี cancellation/request-generation/context check เมื่อ props เปลี่ยน ยืนยันด้วย deferred requests A/B: ให้ B ตอบก่อนแล้ว A ตอบทีหลัง UI กลับไปแสดง thesis/NAV ของ A ทั้งที่ props ปัจจุบันเป็น B

แนวแก้: abort หรือ ignore response ที่ไม่ตรง request/context ปัจจุบัน รวม error/finally; reset packet/claim และ review selections ตาม context; ป้องกัน action บนข้อมูลเก่าระหว่างโหลด ตรวจรับ response สลับลำดับ การเปลี่ยน ticker/portfolio และ unmount อ้างอิง WP09

### J10 — P2: Routine/catalyst/estimate audit state ยังหายเมื่อสร้าง process ใหม่

ตำแหน่ง: `application/resilience/routine_monitor_service.py:29`; `application/thesis/catalyst_service.py:29`, `:138`; `application/equity/estimate_service.py:30`, `:152`; `api/dependencies.py:1283`

Routine history, catalyst assessments และ estimate revisions ยังเก็บใน list/dict ของ service ที่ cache ต่อ process ยืนยัน routine heartbeat บันทึกแล้วมี history 1 รายการ แต่ service instance ใหม่มี 0 รายการ จึงไม่รองรับ restart หรือ workers หลาย process ให้เห็น audit state เดียวกัน

แนวแก้: ใช้ durable repository ports และ append-only records พร้อม source/reviewer/version identity; ทดสอบ reload/new-process และ concurrent workers งานออกแบบ/เปลี่ยน SQLite schema ต้องขออนุมัติตาม AGENTS.md ก่อนทำ อ้างอิง WP08

### J11 — P2: Macro mapping ทำค่าศูนย์หาย

ตำแหน่ง: `api/dependencies.py:794`, `:801`

Mapping nonzero canonical IDs แก้แล้ว แต่ใช้ `or` เลือกค่าทำให้ OFR stress ที่เท่ากับ 0 กลายเป็น None และ direct yield spread 0 กลายเป็น None เช่นกัน ยืนยันผ่าน closure ที่ production provider สร้างจริง

แนวแก้: เลือก alias แรกที่ค่า is not None แทน truthiness รักษา valid zero และไม่ fallback ไปค่าเก่า/คนละหน่วย ตรวจรับ 0, negative, missing และ canonical IDs โดยยืนยัน units อ้างอิง WP04

## ข้อที่ตรวจแล้วว่าแก้ได้จริงในรอบนี้

- Legacy DB ทั้งแบบไม่มี portfolio_id และแบบมี unique index review_date เดิม init_schema ผ่าน และบันทึกสาม signoffs ข้าม portfolio/packet revision ได้
- Approval คนละ packet hash ในวันเดียวกันไม่ลบแถวเก่าแล้ว ส่วน concurrent same-hash ยังเหลือ J02
- Corrupt thesis ทำให้ packet เป็น BLOCKED และ sign_off ปฏิเสธแล้ว
- Serial retry คืน verified samples ที่ persist จริง ไม่เติมหลักฐานจาก request ใหม่; มี structured JSON persistence แล้ว ส่วน required evidence ยังเหลือ J01
- Falsifier observed value เปลี่ยนในวันเดียวกันได้ exception_id ใหม่; หลัง acknowledge ค่าเดิม ค่าที่เปลี่ยนกลับมาเป็น pending ได้
- Canonical macro IDs และ Decimal yield-spread derivation ทำงานกับ nonzero inputs; ส่วน zero ยังเหลือ J11
- Custom stop loss เปลี่ยน tier เป็น INFERRED แล้ว และ estimate seed เปลี่ยน evidence_label เป็น SYNTHETIC แล้ว
- Zero units ไม่ถูกแทนด้วย 1 และ backend beta ที่ไม่มี observations ไม่ถูกกำหนดเป็น 1 แล้ว ส่วนการแสดงผล/valid-price count ยังเหลือ J05/J08
- PM router test ที่ค้างจาก principal contract ในรอบก่อน ผ่านแล้ว

## ผลการตรวจ

- Backend suites ที่เกี่ยวข้อง: **310 passed**, 3 deprecation warnings รวม architecture, auth, PM/thesis/macro/earnings routers, application/core/thesis/tools/db และ portfolio guardrails/integrity ที่เกี่ยวข้อง เป็น targeted run ด้วย `-o addopts=` ไม่ใช่ full backend suite หรือ full coverage gate
- Frontend: `npm --prefix web test -- --run` — **123 files / 621 tests passed**
- TypeScript: `npm --prefix web run typecheck` — **ผ่าน**
- Additional Python probes: legacy migrations, revision history, evidence bypass/retry, concurrent signoff, corrupt-thesis propagation, changed breach, missing metrics, unsourced synthesis, readiness default/CLI-checker, stale daily data, missing-price counts, macro zero และ routine instance reload
- Additional React diagnostic probes: **4 passed** เพื่อยืนยันพฤติกรรมที่ผิด ได้แก่ null beta กลายเป็น 0, null macro ทำให้ render throw และ stale response เขียนทับทั้ง ticker/portfolio ไม่ใช่การรับรองว่า behavior เหล่านี้ถูกต้อง ไฟล์ probe ชั่วคราวถูกลบหลังตรวจ
- ไม่ได้ regenerate manifests หรือเปลี่ยน contract baselines

ลำดับแก้ที่เสนอ: J01/J02/J04/J05 เพื่อปิดการอนุมัติและการแสดงข้อมูลที่ไม่ครบ ตามด้วย J03/J06 เพื่อให้สถานะ readiness/evidence มีหลักฐานจริง แล้วปิด J07–J11 พร้อม regression cases ข้างต้น แผนเดิมยังใช้ได้ตาม WP ที่อ้างอิง แต่ไม่ควรถือว่าปิดทั้ง work package จาก test gate ที่ผ่านเพียงอย่างเดียว
