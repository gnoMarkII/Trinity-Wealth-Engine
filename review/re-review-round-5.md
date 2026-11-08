# ผลตรวจซ้ำรอบที่ 5 — 2026-11-06

ตรวจ working tree ล่าสุดบน HEAD `fea7fbb` เทียบ J01–J11 ใน `review/re-review-round-4.md` และลองกรณีที่เคยผิดผ่าน service/adapters จริง ผล: **ยังเหลือจุดต้องแก้ 9 กลุ่ม** มีทั้งข้อค้างที่แก้ได้บางส่วนและ backend test gate ที่ยังไม่ผ่าน

รอบนี้ไม่ได้แก้ production code, schema, dependencies หรือ contract baselines ใช้ไฟล์/SQLite ชั่วคราวสำหรับ probes และลบหลังตรวจแล้ว

## K01 — P1: ยังข้ามการตรวจตัวอย่าง CRITICAL แล้วลงนามทั้ง packet ได้

ตำแหน่ง: `application/manager/pm_review_service.py:696`–`:706`, `:734`; `web/src/components/manager/MorningDeltaCard.tsx:880`

แก้ missing checksum และ empty samples แล้ว แต่เงื่อนไขปัจจุบันยอมรับเพียง sample ID ใดก็ได้ที่มีอยู่ ไม่ตรวจว่าครบ required/critical samples

Probe ด้วย SQLite repository จริง: packet มี 3 samples รวม `SMP-BREACH-AAA` ที่ urgency=CRITICAL จาก price falsifier breach แต่ส่งเพียง `SMP-MOVER-N/A` ได้ `SIGNED_OFF` พร้อม acknowledged_exceptions_count=1 ทั้งที่ไม่ได้ยืนยันตัวอย่าง breach นั้น UI ก็ยังอนุญาตกด confirm โดยตรวจเพียง isSubmitting

แนวแก้:
- กำหนด required sample set ฝั่ง server อย่างน้อยครอบคลุม CRITICAL ตาม policy และตรวจ set inclusion แทนแค่ membership ของรายการที่ส่งมา
- ผูก acknowledgement count กับ evidence ที่ตรวจจริง หาก policy อนุญาต partial review ต้องไม่รับรอง exceptions ที่ไม่ได้ตรวจ
- ให้ UI แสดงรายการที่ยังจำเป็นต้องตรวจและปิด confirm จนผ่านเงื่อนไขเดียวกัน
- ตรวจรับ: benign-only sample, ข้าม critical หนึ่งตัว, duplicate IDs, invalid IDs, no-sample packet และครบทุก required item อ้างอิง WP05

## K02 — P1: Market observer ต่อแล้ว แต่ยังอ่าน field ไม่ตรง production sidecar

ตำแหน่ง: `tools/thesis/market_observation_adapter.py:35`–`:48`; `tools/market/adapters/equity_vault_query_adapter.py:210`; `application/thesis/thesis_service.py:375`; `core/thesis/evaluator.py:188`

Production sidecar ใช้ `quant_signals` และ `revenue_growth_yoy_pct` แต่ adapter ใหม่อ่าน `quantitative_signals` และ aliases ที่ไม่มี revenue_growth_yoy_pct อีกทั้งการเลือก alias ด้วย or ทำค่าศูนย์หาย

ยืนยันด้วย EquitySidecarAdapter จริงอ่าน latest.json ใน vault ชั่วคราวที่มี revenue growth และ operating margin เท่ากับศูนย์: observer คืน None ทั้ง growth/margin/price และ PM packet ได้ 0 exceptions เมื่อเปลี่ยนเป็น observer ที่ส่ง metrics ชุดเดิมได้ถูกต้อง ตรวจพบ growth และ operating-margin breaches 2 รายการ การ merge price override กับ observer แก้แล้ว แต่ data contract ยังทำให้ production path ข้ามกฎ

แนวแก้:
- Map ตาม MicroQuantOutput/QuantSignals จริง รวม revenue_growth_yoy_pct, operating_margin_pct และ raw_analysis_price; เลือก alias ด้วย is not None เพื่อรักษาศูนย์
- ให้ required rule ที่ไม่มี observation แสดง UNKNOWN/STALE/ERROR แทนการเงียบหายไปจากผลประเมิน และกำหนด sign-off policy สำหรับข้อมูลขาด
- ใช้ typed observation ที่มี unit/source/as-of metadata
- ตรวจรับผ่าน actual sidecar adapter -> observer -> PM packet ด้วย zero, negative, missing และ stale values อ้างอิง WP03

## K03 — P1: Readiness ยังให้ 10/10 READY กับ DB ที่เปิดไม่ได้หรือ schema ใช้งานไม่ได้

ตำแหน่ง: `application/resilience/readiness_audit_service.py:152`–`:195`, `:411`–`:422`; `scripts/institutional_readiness_audit.py:43`

Q5 แก้แล้ว: ไม่ใช้ storage_contract เป็นหลักฐาน state checksum อีก และ CLI ตรวจ required table names เพิ่มแล้ว แต่ยังยืนยันความพร้อมผิดได้สองทาง:

1. Default service: ไฟล์ 512 bytes ที่มี SQLite header, page size, num_cells=1 และ content offset ผ่านเงื่อนไขใหม่ ได้ Q1 PASS และ 10/10 READY เมื่อมี frozen manifest ที่ valid ทั้งที่ SQLite quick_check เปิดแล้วได้ `DatabaseError: file is not a database`
2. ใช้ checker ของ CLI จริง: SQLite มีตาราง jobs/kanban_cards/thesis_claims/pm_review_signoffs แต่แต่ละตารางมีเพียงคอลัมน์ id และมี manifest hash ของ DB ถูกต้อง checker คืน True และ audit ได้ 10/10 READY แต่ query `SELECT packet_sha256 FROM pm_review_signoffs` ล้มด้วย `no such column: packet_sha256`

Q8 เปลี่ยนจาก import-only เป็น build แล้ว แต่สร้าง service ที่ไม่มี portfolio/thesis readers และตรวจเพียง hash ยาว 64 จึงยังไม่พิสูจน์การ rebuild จาก authoritative state จริง

แนวแก้:
- ให้ทุก entry point ใช้ read-only SQLite integrity adapter จริง หากไม่มี checker ให้ fail แทนการใช้ header เป็น health verdict
- ตรวจ required schema/projections/version ด้วย queries ที่ระบบต้องใช้ ไม่ตรวจเพียงชื่อ table
- Q8 ต้อง reconstruct จาก persisted/frozen inputs จริง และตรวจ output/evidence ที่คาดหวัง
- ตรวจรับ malformed DB ที่มี header/cells เหมือนจริง, correct table names แต่ wrong columns และ packet rebuild ที่ขาด state อ้างอิง WP10

## K04 — P1: เพิ่ม locator แล้วแต่ยังไม่ได้ตรวจ primary evidence อย่างอิสระ

ตำแหน่ง: `application/thesis/thesis_synthesis_service.py:234`–`:304`; `application/earnings_call/disclosures.py:9`–`:88`; `api/routers/equity/router_earnings_call.py:309`

Thesis ที่ไม่ใส่ locator ลดเป็น INFERRED แล้ว แต่ has_independent_source ตรวจเพียง substring เช่น 10-q/sec.gov/edgar ส่วน claimed values และ primary_facts ยังมาจาก sidecar ชุดเดียวกัน

Probe sidecar สมมติไม่มีเอกสารหรือ facts ภายนอก: ไม่ส่ง locator ได้ INFERRED แต่ใส่ข้อความ `this is not a 10-q; no primary facts were supplied` กลับได้ DOCUMENTED เพราะผ่าน substring และเทียบค่ากับตัวเอง

Earnings numbers ย้ายออกจาก router แล้ว แต่ยังเป็น constants ใน VERIFIED_EARNINGS_DISCLOSURES การใส่ source_filing และลิงก์ company financial-statement/EDGAR browse page ยังไม่ได้สร้าง source snapshot ที่อ่านตรวจ actual, consensus และ drivers ของ period นั้นได้ โดยเฉพาะ consensus ต้องมีแหล่งของตัวเอง

แนวแก้:
- ใช้ independent source adapter/record ที่ตรวจ ticker, period, source revision, units และ content hash ได้จริง ไม่ยกระดับ tier จากข้อความ locator
- เทียบ claims กับ facts ที่ extract จาก source record ที่ freeze แล้ว หากยังไม่มีหลักฐานให้คง INFERRED/UNVERIFIED
- แยก curated/demo constants และอย่าระบุว่า verified จนมี source evidence ครบ
- ตรวจรับ locator มี keyword แต่ไม่มีเอกสาร, ticker/period ไม่ตรง, values ต่างจาก source และ source ที่ valid จริง อ้างอิง WP04/WP07

ข้อสังเกตรอบนี้เป็นเรื่องกลไกการ verify ในโค้ด ไม่ได้อ้างว่าตรวจยืนยันตัวเลขบริษัทกับเว็บภายนอกแล้ว

## K05 — P1: Exposure fallback นำราคา USD ไปรวมกับมูลค่า THB

ตำแหน่ง: `application/manager/pm_review_service.py:243`–`:263`

เมื่อ market_value_thb เป็นศูนย์/ใช้ไม่ได้ จะ fallback ไป current_price_thb/current_price/current_price_usd โดยไม่แปลง currency ก่อนส่งเข้าเครื่องคำนวณ exposure

Probe ด้วย Holding model จริง: 2 units, ราคา USD 100, fx_rate=35, current_price_thb=None, market_value_thb=0 ได้ gross_market_value=200 และไม่มี quarantine ทั้งที่ค่า THB จาก Decimal calculation คือ 7000.000 หากใช้ FX ไม่ได้ต้องแสดง unavailable/quarantine ไม่ควรรวมหน่วยปนกัน

แนวแก้:
- Normalize position value เป็น base currency ผ่าน approved as-of FX ก่อนรวมทุกครั้ง และตรวจ currency/FX provenance
- หากไม่มีมูลค่า THB หรือ FX ที่เชื่อถือได้ให้ quarantine position
- ใช้ Decimal ตั้งแต่ derive price/value หลีกเลี่ยง float division ที่บรรทัด 247
- ตรวจรับ mixed USD/THB holdings, missing/stale FX, cached market_value เป็นศูนย์ และ long/short อ้างอิง WP06 และประเด็น G09 เดิม

## K06 — P2: Daily P&L ยังมี look-ahead, cash-flow wiring ไม่ครบ และข้อมูลขาดกลายเป็นศูนย์

ตำแหน่ง: `application/manager/pm_review_service.py:150`, `:327`–`:344`; `tools/portfolio/services/performance_service.py:59`–`:68`

แก้ใช้ Decimal และอ่าน net flow ใน consumer แล้ว แต่ยังเหลือ:

- ใช้ abs(target_date - last_date) <= 4 จึงยอมให้ snapshot หลัง review date ผ่าน Probe review 2026-11-06 กับ snapshots 2026-11-06/2026-11-07 ได้ daily P&L 50.0 / 50.0% จากข้อมูลอนาคต
- Consumer อ่าน Net_Cash_Flow/net_external_flow แต่ production snapshot producer ไม่ส่ง field เหล่านี้ Probe รูปแบบข้อมูล production: NAV 100 -> 150 โดยเงินสดเพิ่มจากเงินฝาก 50 ได้ P&L 50.0 ทั้งที่ investment gain ที่คำนวณด้วย Decimal เป็น 0 เมื่อฉีด Net_Cash_Flow=50 เพิ่มเองจึงได้ 0.0
- Probe stale snapshots ที่ถูกปฏิเสธแล้ว ยังได้ daily_pnl_value/daily_pnl_pct=0.0 จาก defaults ผู้ใช้แยกข้อมูลขาดจากผลตอบแทนศูนย์ไม่ได้

แนวแก้:
- Filter snapshots ตาม as-of/cutoff ก่อนเลือกคู่ ห้าม last_date > review_date และกำหนด trading-day window ชัดเจน
- ต่อ net external flows จาก ledger/adapter จริงให้ตรงช่วงเวลา หรือส่งผ่าน production performance contract อย่างครบถ้วน
- ใช้ nullable/unavailable status เมื่อคำนวณไม่ได้ และปรับ UI/types รองรับ
- ตรวจรับ future dates, missing/stale snapshots, deposits/withdrawals และผลตอบแทนศูนย์จริง อ้างอิง WP06

## K07 — P2: Durable storage ยังไม่ถูกต่อใน production และหลาย instance เขียนทับกันได้

ตำแหน่ง: `api/dependencies.py:1295`–`:1320`; `application/resilience/routine_monitor_service.py:45`–`:99`; `application/thesis/catalyst_service.py:44`–`:61`; `application/equity/estimate_service.py:45`–`:59`

เพิ่ม storage_path และ serial restart tests แล้ว แต่ production providers ยังสร้างทั้งสาม service โดยไม่ส่ง path ทำให้ _storage_path=None การตั้ง AUDIT_STATE_DIR ก็ยังไม่ถูกนำมาใช้

Probe ผ่าน dependency providers จริง: ทั้งสาม path เป็น None; บันทึก routine heartbeat แล้ว recreate provider instance ได้ history 0 รายการ

แม้ส่ง storage_path: สร้าง routine instances A/B ก่อนเริ่มเขียน แล้ว A บันทึก WORKER-A ตามด้วย B บันทึก WORKER-B เมื่อ reload เหลือเพียง WORKER-B เพราะแต่ละ instance flush ทั้ง in-memory list ทับไฟล์ โดยไม่มี reload/merge ภายใต้ transaction/lock นอกจากนี้ save helpers กลืน write errors เป็น warning จึงยังมีทางตอบรับ mutation ทั้งที่ไม่ durable

แนวแก้:
- ต่อ durable repository/config ใน production providers และมี integration test ผ่าน providers/API ไม่ทดสอบแค่ constructor ที่ส่ง path เอง
- ย้าย persistence เข้า adapter/port ใช้ shared atomic I/O ตาม AGENTS.md พร้อม transaction/lock และอ่าน state ล่าสุดก่อน append
- ให้ write failure propagate และรักษา cache/state ให้สอดคล้องกับผล commit; ตรวจ corruption อย่างชัดเจน
- ตรวจรับ restart, two-instance interleaving, concurrent workers และ failed writes อ้างอิง WP08 งานเปลี่ยน DB schema/contract ต้องผ่าน approval gate เดิมก่อนทำ

## K08 — P2: Morning card ยังแสดง portfolio เก่าระหว่างโหลด และยังแพ้ response สลับ A -> B -> A

ตำแหน่ง: `web/src/components/manager/MorningDeltaCard.tsx:31`–`:64`, `:78`–`:89`

การสลับ ticker ใน StructuredThesisCard แก้ cancellation แล้ว และ MorningDeltaCard ป้องกัน A -> B แบบง่ายด้วย currentPortfolioRef แล้ว แต่ยังไม่ใช้ request generation:

- Diagnostic React test: แสดง A แล้วเลือก B ระหว่างรอ B ยังเห็น NAV ของ A และยังมี actions ของ packet เดิม ในขณะที่ handler ใช้ portfolioId ปัจจุบัน
- เริ่ม A1 -> B -> A2 ให้ A2 ตอบก่อนแล้ว A1 ตอบทีหลัง UI กลับแสดง NAV จาก A1 เพราะทั้งสอง request มี portfolioId=A เหมือนกัน

แนวแก้:
- ใช้ monotonic request ID/AbortController ต่อ request ไม่ตรวจเพียง portfolio ID
- Clear หรือแสดงสถานะ stale/loading ที่ป้องกัน actions ของ packet เก่าเมื่อ context เปลี่ยน รวม cleanup/unmount
- ตรวจรับ A -> B, A -> B -> A, concurrent refresh และการเปลี่ยน initialPacket ระหว่าง request อ้างอิง WP09

## K09 — P2: Backend test gate ยังไม่ผ่าน เพราะ guardrail code contract ใน test ไม่ตรงโค้ด

ตำแหน่ง: `tests/unit/test_portfolio_advisory_guardrails.py:222`; `core/portfolio_advisory/guardrails.py:403`

ผลรันปัจจุบัน: 318 passed, 1 failed
`test_check_missing_price_policy` คาดหวัง MISSING_AS_OF_PRICE แต่ implementation คืน MISSING_PRICE_EXCEPTION

การรวม missing-price codes ทำให้รายชื่อ missing symbols ถูกต้องใน probe แล้ว ไม่ได้สรุปว่าต้อง revert implementation กลับ ให้ตกลง canonical code และปรับ test/consumers/contract ที่เกี่ยวข้องให้ตรงกัน แล้วรัน gate ซ้ำ โดยไม่แก้ golden manifests ด้วยมือ

## ผลปิดข้อค้างรอบที่ 4

| ข้อเดิม | ผลรอบนี้ |
|---|---|
| J01 | Missing checksum/empty samples ปฏิเสธแล้ว แต่ยังข้าม CRITICAL ได้ — K01 |
| J02 | ผ่าน concurrent SQLite probe: สองคำขอคืน signoff ID/signer เดียวกันและตรงแถวที่เก็บจริง ประวัติไม่ถูก overwrite |
| J03 | Q5 ไม่ใช้ storage_contract แทน manifest แล้ว; CLI ตรวจ table names เพิ่มแล้ว แต่ยังมี false READY — K03 |
| J04 | Provider ต่อ observer และ merge missing metrics หลัง override แล้ว แต่ production field mapping ยังผิด — K02 |
| J05 | ผ่าน React probe: macro null ไม่ crash และ beta null แสดง N/A |
| J06 | ไม่มี locator ลดเป็น INFERRED แล้ว แต่ keyword locator/self-comparison ยังได้ DOCUMENTED — K04 |
| J07 | Decimal, date filtering และ lifetime label ดีขึ้น ยังเหลือ production cash flows/as-of/unavailable — K06 |
| J08 | สอง missing holdings ได้ eligible=0 และ missing_price_symbols ครบ AAA/BBB แล้ว; currency fallback ยังต้องแก้ — K05 |
| J09 | Ticker stale response ถูก ignore แล้ว; morning portfolio request generation ยังไม่ครบ — K08 |
| J10 | Explicit storage_path serial restart tests ผ่าน แต่ production wiring/multi-instance ยังไม่ครบ — K07 |
| J11 | ผ่าน production-reader closure probe: OFR=0 และ spread=0 คงเป็น 0.0 แล้ว |

## ผลทดสอบและขอบเขต

- Backend ที่เกี่ยวข้อง: **318 passed / 1 failed**, รวม architecture, auth, PM/thesis/macro/earnings routes, OpenAPI, application services, core/thesis/tools/db, durable audit tests และ portfolio integrity/guardrails ใช้ targeted run ด้วย `-o addopts=` ไม่ใช่ full backend/coverage gate
- Frontend: `npm --prefix web test -- --run` — **123 files / 621 tests passed**
- Typecheck: `npm --prefix web run typecheck` — **ผ่าน**
- Additional React probes: **4 passed** แยกเป็นยืนยันสองกรณีที่แก้แล้ว (null/ticker cancellation) และสองกรณีที่ยังผิด (stale portfolio display/A-B-A overwrite) การที่ diagnostic probe ผ่านไม่ได้หมายความว่า behavior ที่ผิดถูกแก้แล้ว
- Python probes ใช้ actual SQLite signoff adapter, actual file thesis store, actual sidecar/market-observation adapters, actual dependency providers และ CLI integrity checker; financial expected values คำนวณด้วย Decimal
- ลบไฟล์ตรวจ UI และ temporary directories แล้ว ไม่ได้แก้ production code หรือ regenerate manifests

ลำดับแก้ที่เสนอ: K01/K02 ปิด approval ที่ขาด evidence และ watchdog ที่อ่านข้อมูลไม่ครบ, K03/K04 ทำให้ readiness/evidence verdict มีหลักฐานจริง, K05/K06 แก้หน่วยและช่วงเวลาของผลคำนวณ, K07/K08 ปิด persistence/context races แล้วปิด K09 ด้วย test gate ที่สอดคล้องกับ canonical contract
