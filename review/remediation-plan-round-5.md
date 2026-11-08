# แผนแก้ไขละเอียดเพื่อปิด K01–K09 — รอบที่ 5

วันที่: 2026-11-06  
สถานะ: แผนพร้อมนำไปทำงาน ยังไม่ได้ดำเนินการแก้ production code หรืออนุมัติ migration/contract changes  
ฐานอ้างอิง: HEAD `fea7fbb8efee190c5e0e6a735c6f2a9315d6cbc6` และ source hashes ที่บันทึกใน [acceptance matrix](remediation-acceptance-round-5.json)  
รายงานปัญหา: [re-review-round-5.md](re-review-round-5.md)  
ข้อกำหนดภาพรวมเดิม: [detailed-remediation-plan.md](detailed-remediation-plan.md)

ใช้เอกสารนี้และ acceptance matrix เป็นชุดตรวจรับหลักสำหรับข้อค้าง K01–K09 ส่วนแผนเดิมเป็นข้อกำหนดสถาปัตยกรรมและ approval gates ที่ยังใช้ร่วมกัน

## 1. เป้าหมายและวิธีปิดงาน

ปิด K01–K09 ด้วยการแก้ตลอดเส้นทาง API -> application -> domain -> adapters/storage -> read-back/UI และเก็บหลักฐานว่าพฤติกรรมที่ผิดเดิมถูกปฏิเสธหรือคำนวณถูกจริง

แผนนี้มี 98 กรณีตรวจรับ: K01–K09 รวม 89 กรณี และ RG 9 กรณีที่ป้องกันข้อที่แก้แล้วกลับมาผิด กรณีเหล่านี้เป็นรายการที่จะ implement/เชื่อมกับ test node จริง ไม่ใช่การอ้างว่ามี automated tests ครบแล้ว ทุก case ใน JSON เริ่มที่ planned/not_run

ข้อหนึ่งจะเปลี่ยนเป็น CLOSED เมื่อครบทั้งหมด:

1. กรณีที่เคยผิดมี test ที่ assert พฤติกรรมที่ต้องการ และแสดงให้เห็นว่า fail ก่อนแก้หรือมี reproduction log ที่ผูกกับ source revision ชัดเจน
2. Domain/service tests ผ่าน พร้อม integration test ผ่าน production providers/adapters ที่เกี่ยวข้อง
3. API/UI ใช้ final contract เดียวกัน รวม unavailable/error states และ actor/source/portfolio context
4. ทดสอบ restart/concurrency/failed write ถ้างานนั้นเกี่ยวกับ state
5. Read-back ของข้อมูลที่ commit ตรงกับ response และ source evidence
6. Case matrix ระบุ test node, source revision/hash, คำสั่ง, exit code และหลักฐานผลรันจริง
7. Required backend/frontend/typecheck/architecture gates ผ่าน และ approval gates ที่จำเป็นได้รับอนุมัติจริง

การเพิ่ม method, constructor argument, URL, alias หรือ test ที่ใช้เฉพาะ mock ยังไม่พอสำหรับปิดข้อบกพร่องที่อยู่ใน production wiring

ผล baseline ที่ยืนยันซ้ำระหว่างจัดทำแผน:

| Gate | ผล |
|---|---|
| Targeted backend suites ที่ใช้ใน review รอบ 5 | 318 passed / 1 failed |
| Failure ที่เหลือ | test_check_missing_price_policy: test คาด MISSING_AS_OF_PRICE แต่ producer คืน MISSING_PRICE_EXCEPTION |
| Frontend Vitest | 123 files / 621 tests passed |
| TypeScript | ผ่าน |
| ขอบเขต coverage | Backend run ใช้ -o addopts= เป็น targeted verification; final gate ต้องใช้ configuration ปกติของโครงการ |

## 2. ขอบเขตและข้อกำหนดที่ตรึงก่อนแก้

- ใช้ K01–K09 เป็น issue IDs คงที่ ทุก changeset อ้าง case IDs ในแผนนี้
- เก็บผลตรวจเดิมเป็นประวัติ ไม่เปลี่ยนชื่อปัญหาเดิมเพื่อให้ดูเหมือนปิดแล้ว
- ข้อกำหนดของแผนเดิมยังใช้ โดยเฉพาะ immutable packet, required evidence, trusted principal, typed observations และ durable audit records
- หากพบกรณีใหม่ ให้มี reproduction/ผลกระทบ/ความเชื่อมโยงก่อนเพิ่มงาน จัดสถานะให้ชัดว่ากระทบ K ใดหรือเป็นข้อใหม่
- การทดสอบ network ใช้ frozen source artifacts/transport fixtures ที่ควบคุมได้ การตรวจว่าข้อมูล production พร้อมใช้งานต้องมี actual source records ที่ตรวจย้อนกลับได้
- การอ่านหรือเตรียมแผนยังไม่ใช่การอนุมัติเปลี่ยน schema, เพิ่ม dependencies หรือ regenerate contract baselines

การจับคู่กับ WP เดิมยึดชื่อจาก detailed-remediation-plan.md โดย WP04 คือ analytics และ WP06 คือ earnings/estimates:

| ปัญหา | งานหลัก | WP เดิม | หลักฐานปิดงาน |
|---|---|---|---|
| K01 | required samples/acknowledgements + immutable approval | WP05, WP09 | ข้าม critical ไม่ได้; exact snapshot และ canonical audit |
| K02 | sidecar mapping + complete rule evaluation | WP03 | actual adapter chain ให้ค่าศูนย์/metrics/status ถูก |
| K03 | real DB health + actual state rebuild | WP10, WP11 | corrupt/wrong-schema DB ไม่ READY |
| K04 | independent primary facts + honest evidence tiers | WP06, WP07 | locator อย่างเดียวไม่ยกระดับ tier |
| K05 | base-currency valuation | WP04 | USD/THB/short/FX คำนวณตรง Decimal oracle |
| K06 | as-of snapshots + external flows | WP04 | actual producer/storage/readers ให้ P&L ถูก |
| K07 | durable repositories และ production providers | WP08 | restart/multi-worker/failed writes ผ่าน |
| K08 | UI context + request generation | WP09 | stale response/action ข้าม context ไม่ได้ |
| K09 | canonical warning/tests/contracts | WP11 | required gates exit 0 และ generated contracts ถูก |

## 3. ลำดับส่งมอบและ dependencies

| ชุดงาน | เนื้อหา | ต้องมีมาก่อน | Gate ที่เกี่ยวข้อง | สถานะเริ่มต้น |
|---|---|---|---|---|
| W00 | baseline, regression inventory, canonical warning | รายงาน K01–K09 | ไม่มี | PLANNED |
| W01 | domain/ports/contracts และ migration drafts | W00 | เตรียม DB-A/DB-B/CONTRACT | PLANNED |
| W02 | observations และ complete watchdog | W01 | DB-A ถ้าเก็บ evaluation/evidence เพิ่ม | PLANNED |
| W03 | currency normalization และ P&L ทั้ง producer/readback | W01 | layout/contract ตาม draft | PLANNED |
| W04 | primary sources, synthesis, earnings evidence | W01 | DB-B สำหรับ source records | PLANNED |
| W05 | durable repositories, providers, legacy import | W01 | DB-A/DB-B | PLANNED |
| W06 | required evidence และ transactional approval | W02/W03/W04/W05 | DB-A | PLANNED |
| W07 | UIตาม final contracts และ request generations | W06 และ nullable results ของ W02/W03 | CONTRACT ภายหลัง | PLANNED |
| W08 | readiness จาก state จริงและ negative gate proofs | W02–W07 | ไม่มี schema change เพิ่มจาก audit | PLANNED |
| W09 | migration rehearsal, contract generation, final gates | ทุกชุดงานที่เกี่ยวข้อง | CONTRACT | PLANNED |

ลำดับลงมือ: W00 -> W01 -> W02/W03/W04 -> W05 -> W06 -> W07 -> W08 -> W09 โดย W05 draft/adapter work ที่ไม่ติด schema ทำระหว่าง W02–W04 ได้ การปิด K01 เป็นผลรวมหลัง inputs และ persistence พร้อม

ช่วงลดความเสี่ยงระหว่างทำงาน: เพิ่ม required-evidence validation ฝั่ง service ได้ก่อน migration โดยใช้ packet/policy ปัจจุบัน แต่คงสถานะ K01 เป็น IN_PROGRESS จน exact snapshot, persistence, API และ UI ผ่านครบ

## 4. W00 — ตรึง regression และ baseline ก่อนแก้

ไฟล์หลัก: report/matrix นี้, tests/unit/application/test_pm_review_service.py, tests/unit/test_portfolio_advisory_guardrails.py และ test files ที่ระบุในแต่ละ K

ขั้นตอน:

1. บันทึก HEAD, dirty-file list และ source SHA-256 ของไฟล์ใน scope โดยไม่ overwrite งานเดิมของผู้ใช้
2. แปลง Python/React diagnostic probes จาก report เป็น tests ที่ assert ผลที่ต้องการ ไม่เขียน test ที่คาดหวังพฤติกรรมผิดแล้วนับว่าผ่าน
3. กำหนด deterministic fixtures: fixed UTC cutoff, temporary vault/DB, Decimal inputs, source body/hash/period และ actor/portfolio แยกกัน
4. ตัด conditional test ที่ทำให้กรณีสำคัญไม่ถูกรัน เช่น if packet has samples: ให้ fixture assert ว่ามี CRITICAL/required items จริงก่อนทดสอบ gate
5. กำหนด canonical new-write warning เป็น MISSING_PRICE_EXCEPTION ตาม producer ปัจจุบัน; legacy MISSING_AS_OF_PRICE ยอมรับที่ read boundary เพื่อ compatibility
6. ปรับ test K09 หลังพิสูจน์ behavior แล้ว ตรวจ factor-exposure endpoint ว่ายังส่ง missing symbols ครบและ eligible ถูก
7. ใช้ RG-T01–RG-T09 รักษา behavior ที่ปิดแล้ว รวม concurrent signoff, legacy DB, corrupt thesis, response/read-back identity, macro zero และ nullable UI

ผลส่งมอบ: regression cases ที่ reproduce ปัญหา, baseline log, scope/hash snapshot และ canonical-code decision  
ตรวจรับหลัก: K09-T01–T03, RG-T01–T09  
ห้ามใช้การลบ assertions, skip/xfail หรือแก้ golden manifest ด้วยมือเป็นวิธีทำให้ baseline ดู green

## 5. W01 — ตรึง policy และ interfaces ก่อนแก้หลายไฟล์

### 5.1 Approval และ sample policy

- ทุก sample ที่ระบบประกาศ required ต้องตรวจครบ ในรุ่นแรกให้ selected review samples จริงทั้งหมดเป็น required เพื่อให้กฎตรวจรับชัด
- Critical exceptions ที่ต้อง acknowledge มีชุด IDs แยกจาก bounded sample list การจำกัด samples 3–5 รายการไม่จำกัดจำนวน critical acknowledgements
- required set มาจาก frozen server snapshot; client ส่งเฉพาะ evidence IDs ที่อยู่ใน scope นั้น
- ไม่มี synthetic sample เช่น SMP-MOVER-N/A หากไม่มี mover หรือ holdings จริง ให้เป็น absent/unavailable และนับเฉพาะ valid items
- Empty portfolio ที่ state ครบใช้ no-sample policy ได้; portfolio ที่มี holdings แต่ประเมิน required rules ไม่ได้เป็น BLOCKED
- signer มาจาก authenticated principal ของ require_session; body signed_off_by ไม่เปลี่ยนผู้อนุมัติ
- Acknowledgement count นับ distinct exception IDs ที่ผูกกับ approval จริง ไม่ใช้ len(all_exceptions) แทนการตรวจ evidence
- Same packet concurrent requests คืน canonical stored approval; different packet revision เก็บ history แยก

### 5.2 Domain results ที่ต้องเพิ่มหรือปรับ

ชื่อใหม่ต่อไปนี้เป็นข้อเสนอสำหรับ implementation; ก่อนเพิ่มให้ตรวจ reuse type ที่มีอยู่แล้ว:

| Type/port | ข้อมูลจำเป็น | ผู้ใช้งาน |
|---|---|---|
| MetricObservation / MarketObservationPort | ticker/metric, Decimal value หรือ None, unit/currency, fiscal period, source_id/revision/hash, clocks, status/reason | watchdog, PM, synthesis |
| RuleEvaluation | rule/claim IDs, observation refs, BREACHED/NOT_BREACHED/NOT_EVALUABLE และเหตุผล | packet, human gate |
| PositionValuation | signed units, base THB value, price/FX refs, status/reason | exposure |
| PerformanceWindow | portfolio/cutoff, prev/current snapshot refs, external-flow summary, coverage status | daily P&L |
| VerifiedFact / PrimaryEvidencePort | entity/period/metric/unit/value, content hash, exact locator/span, availability clocks, verification result | earnings/synthesis |
| Audit repositories | append/get/read-back/list, idempotency/conflict semantics, transaction boundary | routine/catalyst/estimate |
| StateDbHealthPort / PacketRebuildPort | configured source paths, schema projections/version, read-only health, rebuild proof | readiness |

Domain models ใช้ dataclass/Enum/Decimal และ standard library; core ไม่ import frameworks, adapters, SQLite หรือ API

ทุก required metric ต้องมี evaluation row แม้ value ขาด ค่า zero/negative ที่ valid สำหรับ metric นั้นยังเป็นข้อมูลจริง เช่น growth=0; stock price <=0 เป็น invalid ตาม price policy

UTC timestamps ต้อง validate ก่อนเทียบ ใช้ TemporalClocks/is_available_as_of เฉพาะกับข้อมูลที่ผ่าน validation แล้ว ไม่ใช้ string ordering เป็นทางผ่านสำหรับ timestamp เสีย

### 5.3 Ports/composition และ compatibility

- ขยาย application/thesis/ports.py ให้ส่ง typed observations/cutoff ได้ โดย compatibility method get_financial_metric ใช้เฉพาะผู้เรียกเก่าระหว่างย้าย ไม่เป็นช่องทางให้ PM ละเลย status
- application/manager/ports.py ต้องมี approval/snapshot contract ที่ตรง adapter จริง และแก้ typing imports ให้ครบในไฟล์ที่เปลี่ยน
- เพิ่ม ports ใน application/resilience/ และ application/earnings_call/ สำหรับ health/rebuild/evidence โดย application ไม่ import concrete storage adapters
- api/dependencies.py เป็น composition root ที่ส่ง configured repositories/paths/cutoffs จริง CLI ต้อง reuse wiring เดียวกันผ่าน adapter factory ที่ไม่ทำ lower-layer import api.*
- HTTP: authenticated synchronous success 200, prepare job 202 พร้อม job_id ถ้าเป็นงาน background, missing resource/source 404, invalid evidence/domain input 422, required storage/provider unavailable 503 ตาม API guidelines
- เสนอ prepare_morning_review use case แยกจาก GET และใช้ deterministic dispatcher/worker เดิม การเพิ่ม REST prepare endpoint ถ้าจำเป็นรวมใน final CONTRACT review

ผลส่งมอบ: interface signatures, data/status examples, policy_version, compatibility map และ migration drafts ที่ review ได้  
เกณฑ์ปิด W01: producer/service/adapter/DTO/UI มี schema เดียวกัน และการเปลี่ยน public/layout contracts ถูกระบุใน approval package ครบ

## 6. W02 / K02 — Observation และ watchdog ที่ประเมินครบจริง

ไฟล์ที่แก้:

- tools/thesis/market_observation_adapter.py
- application/thesis/ports.py และ thesis_service.py
- core/thesis/evaluator.py และ proposed core/thesis/observations.py
- api/dependencies.py
- schemas/micro_quant_schemas.py เฉพาะเมื่อมีเหตุผลต้องเพิ่ม typed source field จริง

Mapping ที่ต้องตรวจจาก production contract:

| Rule metric | Source ที่รองรับจริง | Unit/ข้อกำหนด |
|---|---|---|
| revenue_growth_yoy | quant_signals.revenue_growth_yoy_pct | PERCENT; zero/negative valid |
| fcf_margin | quant_signals.fcf_margin_pct | PERCENT |
| current_price | raw_analysis_price_str ก่อน raw_analysis_price เมื่อมีทั้งคู่ | native currency ตาม market; ต้องมี price as-of |
| operating_margin/gross_margin | verified fundamentals/source field ที่มีอยู่จริงและประกาศ mapping | ถ้า QuantSignals ไม่มี field ให้ NOT_EVALUABLE ไม่สร้างค่าเอง |
| catalyst date | actual event/source record และ exact event identity | scheduled vs observed dates แยกกัน |

ขั้นตอน:

1. อ่าน/validate sidecar snapshot หนึ่ง version ต่อ ticker/scan แล้ว extract metrics จาก snapshot เดียว เพื่อไม่ mix facts ระหว่างอ่านหลายรอบ
2. รองรับ quant_signals เป็น canonical shape; legacy quantitative_signals ต้องผ่าน versioned mapping ที่ชัดเจน ไม่ fallback สุ่มเมื่อ aliases ขัดกัน
3. เปลี่ยน or-based alias selection เป็น first non-None พร้อมตรวจ non-finite/units/period/source identity
4. รักษา price override ที่ถูก scope และเติม metrics ที่ยังขาดจาก observer เป็นราย metric
5. เปลี่ยน evaluator ให้คืน NOT_EVALUABLE สำหรับ missing/stale/error ไม่เงียบข้าม rule; required evaluation failure ทำให้ packet BLOCKED
6. ส่ง completeness/reasons ผ่าน DTO และข้อมูลที่ freeze ใน packet/hash; optional observations ที่ไม่ใช้ใน policy แสดง unavailable ได้
7. ทดสอบผ่าน EquitySidecarAdapter จริงและ providers จริงกับ temp vault ไม่ override service/adapter ตัวที่ต้องพิสูจน์

ตรวจรับ: K02-T01–T10 และ K01-T07  
เสร็จเมื่อ: case zero-growth/zero-margin ตรวจพบ breach หรือระบุ unsupported ตาม actual schema อย่างถูกต้อง; ไม่มีเส้นทางที่ missing metric ถูกนับว่า rule ผ่าน

## 7. W03 / K05 — Normalize currency และคำนวณ exposure ด้วย Decimal

ไฟล์หลัก: application/manager/pm_review_service.py, core/equity_analytics/factor_exposure.py, core/portfolio_advisory/guardrails.py, proposed core/portfolio_advisory/valuation.py และ api/routers/portfolio/router_advisory.py

ขั้นตอน:

1. แยก pure valuation helper ออกจาก packet assembly ให้ input ระบุ currency และ as-of ของ market value/price/FX ชัดเจน
2. ใช้ trusted market_value_thb ที่สอดคล้องกับ source state หรือคำนวณ signed units * native price * native-to-THB rate
3. Native THB ใช้ rate=1 จาก currency identity; USD ต้องมี approved USD->THB FX และ availability <= cutoff
4. หากราคา/FX ไม่ครบหรือ source revisions ขัดกัน ให้ quarantine ทุก position ที่ผิด พร้อม reason; ไม่มี raw USD fallback ใน aggregate THB
5. Zero units ไม่เพิ่ม exposure/count; invalid shares/price/FX ไม่ถูกเปลี่ยนเป็น zero หรือ silently continue
6. คำนวณ Decimal ตั้งแต่ parse, divide, multiply, sum จน rounding ที่ defined boundary; ใช้ decimal strings ใน frozen payload
7. Eligible count นับเฉพาะ valuations ที่ผ่าน; API ส่ง missing symbols ครบ และ partial-quality state ถ้าคำนวณจาก verified subset
8. เปอร์เซ็นต์ใช้ denominator ตาม contract ที่ประกาศเดียวกัน และตรวจ short sign convention

Expected values ต่อไปนี้คำนวณด้วย Python Decimal ระหว่างทำแผน:

| Fixture | THB value |
|---|---|
| USD long: units=2, price=100, FX=35 | 7000 |
| USD short: units=-1, price=50, FX=35 | -1750 |
| THB long: units=3, price=200 | 600 |
| Combined gross | 9350 |
| Combined net | 5850 |

ตรวจรับ: K05-T01–T10 และ RG-T08  
เสร็จเมื่อ: actual Holding model + provider/API integration ไม่ให้ gross=200 THB สำหรับ fixture USD ที่ควรเป็น 7000 THB และไม่มีตำแหน่ง invalid หายจาก quarantine/count

## 8. W03 / K06 — Daily P&L ต้องต่อครบจาก cash-flow producer ถึง UI

ไฟล์ที่ต้อง trace และแก้ตาม final contract:

- tools/portfolio/services/cash_flow_service.py
- tools/portfolio/domain/mutation.py และ structured journal/event models ที่ใช้จริง
- tools/portfolio/transactional_repository.py และ transaction_store.py
- tools/portfolio/ports/performance_port.py และ proposed ExternalCashFlowPort
- tools/portfolio/services/performance_service.py
- tools/portfolio/adapters/markdown/performance_adapter.py และ paths.py
- application/manager/pm_review_service.py และ api/dependencies.py
- web/src/api/types.ts และ MorningDeltaCard.tsx

เหตุผลที่ต้องแก้ตลอดเส้นทาง: ปัจจุบัน cash_flow_recorded มี metadata เพียง action/currency โดยจำนวนเงินอยู่ในข้อความ และ CSV writer สร้าง row_list ตาม header ตายตัว การเพิ่ม Net_Cash_Flow ใน consumer อย่างเดียวจึงไม่ทำให้ production มีข้อมูลครบ

### 8.1 Canonical external-flow event

1. เก็บ event_id/portfolio_id/action, signed Decimal amount_native, currency, occurred/available timestamps, native-to-THB FX/ref และ amount_thb พร้อม source/receipt identity
2. ใช้ canonical transaction stream เดิมเป็น authority โดยใส่ versioned structured flow payload ผูกกับ state mutation/sequence เดียวกัน ผ่าน port/UoW
3. Amount/flow event และ cash state ต้อง commit/rollback ร่วมกัน retry ใช้ event/command identity เดิมไม่เกิด double counting
4. Cash balance delta ไม่ใช่ net external flow; trade, dividend/income, fee และ FX conversion ภายใน portfolio ต้อง classify ตาม accounting policy
5. Legacy note-only events ที่ขาด amount/FX/coverage ใช้สถานะ MISSING_FLOW_HISTORY ห้ามตีความว่า flow=0; ถ้าจะ backfill ต้องมี source/reconciliation ที่ตรวจได้และเก็บ legacy provenance
6. หาก existing transaction payload/adapter ต้องขยาย layout ให้ทำ migration/compatibility draft และผ่าน gate ก่อนใช้กับข้อมูลผู้ใช้

### 8.2 Snapshot/window contract

- ระบุ portfolio, snapshot_id/revision, cutoff_utc, source state sequence/hash, NAV THB, availability time และ flow-coverage status
- Production reader ต้องรับ requested cutoff/range แทน days ที่ผูกกับเวลาปัจจุบันอย่างเดียว
- เลือก snapshots ที่ available <= cutoff และ date ไม่อยู่หลัง review date; เรียง/จัดการ duplicate/revisions ตาม source identity
- กำหนด daily/trading-day window ใน policy; ข้อมูลหลายวันต้องแสดง interval ตามจริงหรือ unavailable หากไม่ใช่ daily interval ที่ policy ยอมรับ
- Join structured external flows ในช่วง (previous_cutoff, current_cutoff] ผ่าน actual store reader และบันทึก contributing flow IDs
- Snapshot producer/writer/header/reader ต้อง round-trip metadata ที่จำเป็นครบ หากใช้ CSV v2 เพิ่ม fields หลัง header เดิมอย่าง versioned และมี old-reader/new-reader rehearsal
- Net_Cash_Flow/net_external_flow ที่รองรับ compatibility ต้อง derive จาก canonical event set เดียวกัน ไม่เป็นแหล่งข้อมูลคนละชุด
- ไม่มีคู่ snapshot หรือไม่ทราบ flow coverage ให้ values=null, quality=UNAVAILABLE พร้อม reason; แยกจาก AVAILABLE + 0.00

### 8.3 สูตรและการแสดงผล

    investment_gain_thb = end_nav_thb - start_nav_thb - net_external_flow_thb
    cash_adjusted_return_pct = investment_gain_thb / start_nav_thb * Decimal("100")

start_nav <=0 หรือ inputs ไม่ครบให้ unavailable; สูตรนี้เป็น cash-adjusted change สำหรับช่วง snapshot ที่ประกาศ ไม่ระบุว่าเป็น time-weighted return โดยไม่มีวิธีคำนวณรองรับ

| Fixture | Gain | Return % |
|---|---|---|
| NAV 100 -> 150, deposit 50 | 0 | 0.00 |
| NAV 100 -> 70, withdrawal 30 | 0 | 0.00 |
| NAV 100 -> 160, deposit 50 | 10 | 10.00 |
| NAV unchanged และยืนยัน flow=0 | 0 | 0.00 |

ตรวจรับ: K06-T01–T12 ผ่าน manage_cash_flow -> commit -> snapshot persistence -> production reader -> API ไม่ฉีด Net_Cash_Flow เฉพาะใน mock  
เสร็จเมื่อ: future/stale/unknown-flow ไม่กลายเป็น performance ศูนย์หรือใช้ข้อมูลอนาคต และ real deposit/withdraw ไม่ถูกนับเป็นผลการลงทุน

## 9. W04 / K04 — Primary facts และ tier ที่มีหลักฐานจริง

ไฟล์หลัก: application/thesis/thesis_synthesis_service.py, core/thesis/claim_validator.py, evidence_taxonomy.py, application/earnings_call/disclosures.py, application/earnings_call/ports.py, router_earnings_call.py และ proposed tools/content/earnings_call/adapters/disclosure_source_adapter.py

### 9.1 Evidence records และ ingestion boundary

- SourceArtifact เก็บ source_id/revision, issuer/ticker, document type/period, exact locator, content hash, clocks, rights/approval reference และ local immutable artifact ref
- VerifiedFact เก็บ metric/unit/Decimal value, period, exact source span/table cell และ reference ไป source artifact ที่ผ่าน hash/integrity verification
- Consensus เป็น source คนละชนิดจาก actual filing ต้องมี provider/as-of/frozen revision ของตัวเอง
- Derived driver/model assumption แยกจาก reported fact การ reconcile sum ได้พิสูจน์การคำนวณ ไม่พิสูจน์ว่าคำอธิบาย driver เป็น disclosure จริง
- Ingestion adapter เท่านั้นที่ resolve/อ่าน source documents; synthesis/HTTP ไม่เปิด URL ที่ client ส่งมาหรือใช้ substring เป็น authority
- Source state/version ที่ incomplete หรือ hash ไม่ตรงเป็น unavailable/integrity error พร้อม reason; ไม่แทนด้วย constants ของหุ้นจริง

### 9.2 เปลี่ยน validation และ tiers

1. แยก claimed values ออกจาก primary facts ที่ loader อ่านจาก independent record; ห้ามสร้างสองฝั่งจาก quant dict เดียวแล้วเทียบกัน
2. ตรวจ ticker/period/unit/cutoff/source revision ก่อน tolerance/value matching
3. DOCUMENTED ใช้กับ factual assertion ที่ source สนับสนุนจริง; locator/accession/URL เป็น reference สำหรับค้น record ไม่ใช่ proof ในตัวเอง
4. Analyst hypothesis, automatic stop-loss/growth/margin heuristic และ forward narrative คง INFERRED แม้ baseline factual metric จะ DOCUMENTED
5. Factual claim ที่ source ไม่รองรับเป็น UNVERIFIED/CITATION_MISMATCH; ไม่อัปเป็น DIRECTLY_TESTED เพราะมี current_price หรือเพราะรัน Python สำเร็จ
6. Calculation trace ที่มี frozen inputs อาจเป็น DIRECTLY_TESTED สำหรับคำกล่าวเรื่องผลคำนวณนั้น พร้อม provenance ของ inputs
7. API/claim response และ persisted record/read-back ต้องมี tier/source refs ตรงกันตาม schema ที่อนุมัติ

### 9.3 Earnings production path

- เปลี่ยน get_verified_earnings_disclosure ให้ใช้ port/repository ที่โหลด verified record จริง และ inject ผ่าน composition root
- เลือก actual + consensus + driver assumptions ตาม ticker/period/as-of อย่าง explicit; ไม่มี global as_of fallback คงที่
- เมื่อไม่มี verified actual/consensus ให้ endpoint ตอบ unavailable/404 ตาม contract ไม่ส่งตัวเลข hardcoded ในนาม verified
- Fixture/demo records อยู่ใน test/demo scope และไม่ถูก production provider โหลดเป็น actuals ของ ticker จริง
- ชุดข้อมูล production ที่ยังไม่มี primary artifacts ให้บันทึก DATA_UNAVAILABLE ใน readiness/feature status; การปิด bug เรื่อง false verification ต้องพิสูจน์ fail-closed ส่วนการเปิดใช้งาน dataset ต้องมี positive real-source evidence ก่อนนับว่าพร้อม

ตรวจรับ: K04-T01–T11 และ RG-T05; positive source verification ใช้ frozen source ที่อ่านตรวจได้ ส่วน synthetic fixtures ใช้ทดสอบ workflow/negative boundaries พร้อม provenance ว่าเป็น fixture  
เสร็จเมื่อ: ข้อความ "not a 10-q" หรือ URL มี sec.gov ใน path ไม่ทำให้เกิด DOCUMENTED และ figures/consensus/drivers ทุกตัวที่ระบุ verified อ่านย้อนถึง artifact/version ได้

## 10. W05 / K07 — Durable state ที่ production ใช้จริง

ไฟล์หลัก: api/dependencies.py, api/db/uow.py, application/resilience/routine_monitor_service.py, application/thesis/catalyst_service.py, application/equity/estimate_service.py และ repository adapters/DAOs ที่เสนอเพิ่ม

แนวทางหลัก: ใช้ durable SQLite repositories ผ่าน UoW เดิมสำหรับ metadata/events; ใช้ immutable file artifacts สำหรับ body ใหญ่และ _atomic_io สำหรับ file projections/cache ที่จำเป็น

ขั้นตอน:

1. เพิ่ม repository ports สำหรับ routine execution events, catalyst/capture/assessment revisions และ estimate ledger
2. ย้าย filesystem persistence helpers ออกจาก application services; DAOs ไม่ commit/rollback เอง
3. Provider ทั้งสามส่ง configured repository จริง ค่าตั้งต้นต้องสอดคล้อง WEBUI_STATE_DB_PATH/approved configuration
4. AUDIT_STATE_DIR ใช้เป็น legacy-import/artifact location ตาม config ที่ระบุ ไม่ใช่ค่าที่ตั้งแล้วถูก ignore เงียบ ๆ
5. Mutation อ่าน current durable version และ append ภายใต้ transaction เดียว; ใช้ event ID/idempotency key + payload hash และ optimistic version ตาม domain
6. Response มาจาก canonical row ที่ commit แล้ว Cache เป็น read cache ที่ invalidate/reconcile ตาม durable version
7. Write/commit failure ต้อง propagate และ rollback; ไม่ตอบ success จาก in-memory object ที่ยังไม่ durable
8. Read corruption ต้อง repair-required/preserve evidence ไม่ seed ค่าเดิมแล้ว overwrite
9. Human assessments ผูก principal, portfolio/claim/catalyst, exact source/fingerprint และ revision; source facts ใหม่ไม่ reuse acknowledgement เก่า
10. Legacy JSON import เป็น idempotent migration command: checksum input, validate types/IDs/times, compare duplicate payloads, preserve actor/time, log imported mapping และคงต้นฉบับ
11. Source artifacts ใช้ content-addressed immutable refs: publish artifact ที่ validate แล้วก่อน commit reference; DB rollback ไม่ทำให้ reference ชี้ไฟล์ที่ยังไม่พร้อม
12. Worker notification/outbox อยู่ transaction เดียวกับ state mutations ตาม API guidelines; deterministic scheduler/dispatch ไม่เพิ่ม manager LLM อีกตัว

Schema draft ที่ต้องเสนอใน DB-B:

| Record family | Identity/uniqueness | Fields/invariants สำคัญ |
|---|---|---|
| routine_execution_events | execution_id + event_sequence | scheduled/start/completed/error, UTC clocks, terminal transition, artifact refs |
| catalyst revisions | catalyst_id + version | entity/claim/portfolio scope, expected/observed dates, parent revision |
| source captures | source_id + content_hash/revision | actual locator/body ref/clocks/period, immutable content |
| catalyst assessments | assessment_id + scoped exception fingerprint | reviewer subject/session, source revision, assessment text/time |
| estimate records | estimate_id + prior_record_id | ticker/period/metric/type/unit/Decimal text, approval/source as-of |
| import receipts | source_digest + record identity | repeat-safe migration, legacy-to-new mapping, conflicts |

Table names/fields เป็น draft สำหรับ review ต้องตรวจ reuse tables ที่มีอยู่ก่อนเขียน migration ไม่เพิ่ม schema ตามตารางในเอกสารนี้โดยอัตโนมัติ

ตรวจรับ: K07-T01–T12  
เสร็จเมื่อ: APIs/providers จริงผ่าน fresh-process readback และ two-instance interleaving เก็บทั้ง A และ B; optional storage_path tests อย่างเดียวไม่ถือว่าปิด K07

## 11. W06 / K01 — Exact packet และ approval transaction

ไฟล์หลัก: application/manager/pm_review_service.py, application/manager/ports.py, core/portfolio_advisory/review_sampling.py, api/routers/pm_review_router.py, api/db/adapters.py, tools/thesis/thesis_sqlite_dao.py และ packet/approval repositories ที่เพิ่มตาม DB-A

### 11.1 Prepared immutable snapshot

- Freeze portfolio_id, review_date, revision, cutoff, source refs/versions, rule evaluations, required samples, critical exception IDs, quality/block reasons และ policy_version
- Canonical numeric values เป็น decimal strings; clocks เป็น validated UTC; checksum คำนวณจาก canonical frozen payload
- ไม่รวม signoff_info, response-view status, เวลาที่เปิด GET หรือ checksum ตัวมันเองใน hash
- Semantic/source state เดิม reuse revision เดิม; input/policy/source revision ใหม่สร้าง packet ใหม่
- GET today/history อ่าน published snapshot กับ approval; preparation/scan/persist เป็น use case ที่ scheduler/prepare job เรียก
- Readiness rebuild ใช้ input refs ของ snapshot จริง ไม่ใช้ service ที่ไม่มี providers

### 11.2 Approval request และ validation

1. Request ระบุ portfolio_id, packet_id/revision, packet_sha256, verified_sample_ids, acknowledged_exception_ids และ idempotency key ตาม final DTO
2. Authenticate principal แล้วโหลด exact stored snapshot; ตรวจ scope/hash/policy/current eligible revision
3. ปฏิเสธ BLOCKED, missing/forged/stale hash, cross-portfolio/revision IDs และ missing required samples/critical acknowledgements
4. Deduplicate evidence IDs เป็น deterministic unique sets ก่อน validation/count และ persist ตาม contract และสร้าง request digest จาก normalized payload
5. Transaction ทำ validate-current-head + create-or-return-canonical approval + acknowledgement events/receipt ที่เกี่ยวข้องร่วมกัน
6. ใช้ existing uniqueness semantics ที่รักษา same-packet canonical row และ append history ของ packet revision ใหม่; conflict ไม่แก้ signer/time/notes/evidence ของแถวเก่า
7. Signer, actor/session, server time และ policy/source identity บันทึกจาก trusted boundary; response คือ persisted canonical result
8. Read-back/history/retry คืน evidence เดิม ไม่ reconstruct verified IDs จากข้อความ notes เป็น authority

### 11.3 DB-A draft และ legacy

- เสนอ immutable packet table/record และ composite scope keys/indexes ตาม schema draft เดิม
- Approval เก็บ reference ไป packet, normalized verified/acknowledged IDs, request digest/receipt และ trusted principal metadata
- Legacy approvals ที่มีเพียง hash/notes รักษาเป็น legacy audit ไม่ fabricate packet payload หรือใหม่ให้ดู verified
- เตรียม old/new schema rehearsal ก่อนขออนุมัติ actual migration; preserve portfolio isolation และ existing history

ตรวจรับ: K01-T01–T11, K02-T09, K07-T04–T05, RG-T01–T04  
เสร็จเมื่อ: benign-only sample ถูกปฏิเสธจริงผ่าน API; exact snapshot/evidence/read-back/concurrent/restart ครบ และ required evidence เกิน sample cap ยังถูกตรวจทุกตัว

## 12. W07 / K08 — UI ผูก context กับ request generation

ไฟล์หลัก: web/src/components/manager/MorningDeltaCard.tsx/.test.tsx, StructuredThesisCard.tsx/.test.tsx, web/src/api/client.ts, types.ts/generated types และ ManagerAdvisor integration tests

ขั้นตอน:

1. ใช้ monotonic generation ต่อ request พร้อม captured portfolio/packet context; cleanup หรือ AbortController ทำให้ older generations ใช้งานไม่ได้
2. Success/catch/finally ทุกจุดตรวจ generation ปัจจุบัน ไม่ตรวจเพียง portfolioId เพราะ A -> B -> A มี ID ซ้ำ
3. เปลี่ยน portfolio แล้ว clear packet หรือแสดง loading/stale boundary ที่ปิด actions จนข้อมูล context ใหม่พร้อม
4. InitialPacket เปลี่ยน, manual refresh, unmount และ sign-off follow-up ต้อง invalidate generation ที่ไม่เกี่ยวข้อง
5. Modal/evidence selections ผูก packet_id/hash/policy; เปลี่ยน packet/portfolio ให้ invalidate/reset ก่อนส่ง mutation
6. ส่ง exact displayed snapshot ใน sign-off และปิด confirm จน required evidence ครบ
7. Render macro/beta/P&L null เป็น unavailable/N/A; valid zero แสดง zero; lifetime mover ใช้ label ที่แก้แล้ว
8. Server-side validation ยังเป็น authority; UI guard ช่วยป้องกันการกดผิด context
9. รักษา ticker cancellation ที่แก้แล้ว และใช้ deferred promises + act สำหรับ tests แทน time-based sleeps

ตรวจรับ: K08-T01–T08, K01-T11, K06-T12 และ RG-T08  
เสร็จเมื่อ: A -> B -> A และ same-portfolio concurrent refresh ไม่ทำให้ response เก่า overwrite; ผู้ใช้กดอนุมัติ packet เก่าภายใต้ portfolio ใหม่ไม่ได้

## 13. W08 / K03 — Readiness ที่ตรวจ state และ rebuild จริง

ไฟล์หลัก: application/resilience/readiness_audit_service.py, scripts/institutional_readiness_audit.py, proposed application/resilience/ports.py, proposed tools/resilience/sqlite_integrity_adapter.py และ packet/source repository factories

ขั้นตอน:

1. เอา header/page-size/cell inspection ออกจากเงื่อนไขที่ให้ PASS ใช้เป็น diagnostic ได้ แต่ไม่มี actual integrity checker ให้ FAIL/CONDITIONAL
2. Read-only adapter ใช้ configured DB path เดียวกับ production, SQLite health query และ canonical required projections/columns/index/schema version
3. Health check ไม่เรียก init_schema, migrate, ALTER, CREATE หรือ repair ระหว่าง audit
4. CLI และ service ใช้ checker/factory เดียวกัน ตรวจ required projections ด้วย queries ที่ระบบใช้งานได้จริง ไม่เพียงชื่อ table
5. Q5 ตรวจ frozen manifest/input artifacts ของ named scope พร้อม coverage/integrity; configuration JSON หรือ arbitrary unchanged file ไม่แทน authoritative inputs ทั้งชุด
6. Q8 ใช้ PacketRebuildPort โหลด frozen inputs/versions แล้ว reconstruct payload/hash เทียบ snapshot reference ใน fresh process โดยไม่มี chat transcript
7. Q9 ใช้ positive/negative human-gate evidence จาก actual service/transaction paths ไม่ตรวจเพียง enum ว่าไม่มีชื่อ AUTONOMOUS
8. ทุก question เก็บ evidence ref/hash/config scope/checker version/check time; ไม่สรุป production readiness จาก importability หรือ toy calculation อย่างเดียว
9. Overall READY เฉพาะ named stage ที่มี required proofs ครบ; unknown/missing/failed proof ไม่เป็น PASS
10. Write readiness decision artifact ผ่าน _atomic_io; report identity/hash ครอบคลุมผลประเมิน/เหตุผล/evidence ที่แสดงจริงตาม final contract

ตรวจรับ: K03-T01–T10 รวม fake DB ที่มี cells เหมือนจริง และ DB ที่มีชื่อ tables ครบแต่แต่ละตารางมี id คอลัมน์เดียว  
เสร็จเมื่อ: ทั้ง default service และ CLI ไม่คืน READY กับ state ที่ actual queries/rebuild ใช้งานไม่ได้

## 14. Approval packages และ migration rehearsal

อ้างอิงข้อ Ask First ใน AGENTS.md: การเปลี่ยน SQLite tables/migrations/layout, dependencies และ contract baselines ต้องได้รับอนุมัติก่อนลงมือ ส่วนการทำ draft และ rehearsal บน temporary fixtures ทำได้เพื่อให้มีงานที่ review ได้ก่อนถามอนุมัติ

| Gate | สิ่งที่ต้องพร้อมให้อนุมัติ | อนุญาตหลังผ่าน |
|---|---|---|
| DB-A | exact DDL/index/FK changes สำหรับ packets/approvals/evaluations, old->new mapping, legacy policy, rollback/rehearsal logs | implementation และ migration ตาม scope ที่อนุมัติ |
| DB-B | durable events/source/estimate repositories, importer conflict rules, recovery/concurrency tests | repository schema และ import ตาม draft |
| STORAGE/CONTRACT | flow/snapshot JSON/CSV versioning, DTO/statuses, affected readers/types, compatibility examples | เปลี่ยน layout/public contract ตามรายการ |
| GOLDEN | final OpenAPI/state exports diff และผล behavior tests | generator/baseline update หลังอนุมัติ |
| DEP | package ที่จำเป็น เหตุผลและทางเลือกจากของเดิม | เพิ่ม dependency เฉพาะที่อนุมัติ; แผนนี้ตั้งต้นใช้ของที่มีอยู่ |

Migration rehearsal ที่ต้องทำก่อนใช้กับข้อมูลจริง:

1. Inventory schema/version, row counts, portfolio IDs, duplicate/conflicting record IDs, existing indexes และ legacy source files
2. สร้าง fixtures ครอบคลุม fresh DB, no-portfolio legacy, unique-date legacy, current portfolio/hash DB และ corrupted/conflicting import
3. Dry-run แสดง mapping/constraints/conflicts โดยไม่เขียนต้นทาง และตรวจว่าค่า legacy ที่ไม่รู้ยังเป็น unknown
4. ใช้ transaction/UoW สำหรับ DB changes; DAO ไม่มี commit/rollback เอง
5. ทำ consistent SQLite backup/restore rehearsal ที่คำนึงถึง WAL; กำหนด write fencing/maintenance window ใน actual migration runbook
6. Verify counts/hash/identity, read existing history, create new revision, concurrent retry และ restart
7. ทดสอบ failure ระหว่างขั้นและ rollback; projection/artifact failure ไม่ publish canonical state ครึ่งเดียว
8. รักษา original JSON/artifacts และ import receipts; ห้ามลบต้นฉบับก่อน verify/approval
9. ขออนุมัติด้วย concrete draft/diff/log แล้วจึงทำ actual schema/layout migration

คำขอนี้ครอบคลุมการจัดทำแผน การอนุมัติ implementation ใช้ draft และผล rehearsal ตาม gates ด้านบน

## 15. W09 / K09 — Contract generation และ final gates

ลำดับตรวจ:

1. ทุก K มี case-to-test mapping และ test assert desired behavior; ทุก required case ผ่านโดยไม่มี skip/xfail หลบเหตุขัดข้อง
2. Production-provider integration suite ผ่านกับ temp vault/DB และ real session authentication สำหรับ mutations
3. Cross-process/restart/concurrency/import/failed-write cases ผ่าน
4. Schema/API/type review เสถียรและได้รับ approval ที่เกี่ยวข้อง
5. หลัง GOLDEN gate จึงใช้ scripts/generate_golden_manifests.py; generated TS/OpenAPI ใช้ scripts ที่ repo มีอยู่ ไม่แก้ไฟล์ generated/golden ด้วยมือ
6. รัน full required commands ด้วย configuration ปกติ เก็บ output/exit code/source revision และตรวจ RG คงผ่าน
7. Reviewer อ่าน diff ตาม K01–K09 และลอง failed scenarios เดิมครบ ไม่ปิดงานเพียงเพราะ test totals สูงขึ้น

คำสั่ง final ที่ต้องใช้จาก AGENTS.md:

    .venv\Scripts\pytest tests/ -v
    npm --prefix web test -- --run
    npm --prefix web run typecheck

คำสั่งหลังอนุมัติ GOLDEN เท่านั้น:

    .venv\Scripts\python scripts/generate_golden_manifests.py

ระหว่างพัฒนาให้รัน suites ของชุดงานก่อน แล้วรันรวมเมื่อ inputs/contracts คงที่ ไม่เปลี่ยน -o addopts= เพื่อหลบ coverage/required configuration ใน final gate หากมี failure ต้องรายงาน test node/สาเหตุ/ผลกระทบ และแก้หรือมี explicit scope decision ก่อนเรียกว่า release-ready

## 16. วิธีบันทึกสถานะเพื่อให้การตรวจรอบต่อไปปิดได้จริง

สถานะต่อ K: PLANNED -> REPRODUCED -> IMPLEMENTED -> INTEGRATED -> VERIFIED -> CLOSED

- REPRODUCED ต้องมี failing desired-behavior test หรือ reproduction log ที่ผูก baseline hash
- IMPLEMENTED หมายถึง patch พร้อม ยังไม่เท่ากับปิด
- INTEGRATED ต้องผ่าน actual production providers + storage/read-back
- VERIFIED ต้องผ่าน failure/concurrency/restart/UI cases ที่เกี่ยวข้องและ final contracts
- CLOSED ต้องมี reviewer evidence และ required gates ผ่าน
- ถ้ารอ schema/source availability ให้ระบุ WAITING_APPROVAL หรือ DATA_UNAVAILABLE พร้อม dependency ไม่ใช้ CLOSED แทน
- บันทึก evidence ใน JSON matrix ต่อ case: actual test nodes, command, commit/source hash, run time, result, log/artifact path; source/contract เปลี่ยนหลังรันต้องรันทดสอบส่วนที่ได้รับผลกระทบอีกครั้ง
- Review report รอบถัดไปใช้ K IDs เดิม แสดง closed/remaining/new-with-reproduction เพื่อให้เห็นความคืบหน้าจริง

หลักฐานส่งมอบต่อ K:

| รายการ | ต้องมี |
|---|---|
| Implementation | diff/files และ decision ที่เปลี่ยน behavior |
| Regression | case IDs + failing-before / passing-after evidence |
| Wiring | actual API/provider/adapter/storage path และ config ที่ทดสอบ |
| Persistence | response/read-back/restart/concurrent proof ถ้าเกี่ยวข้อง |
| Contract | schema/type/status/policy versions และ approval ref |
| Review | remaining limitations, reviewer result และ source revision |

## 17. การแยก environment และป้องกันผลทดสอบคลาดเคลื่อน

- ใช้ temporary workspace/vault/DB และ existing tests/conftest.py isolation; ห้ามให้ test เขียน user database/vault จริง
- Clear/restore dependency caches และ env configuration ใน fixtures เพื่อทดสอบ provider binding ตาม path ที่ตั้งจริง
- Service/adapter ตัวที่ต้องตรวจ integration ใช้ของจริง Override ได้ที่ external transport/source input boundary โดยอธิบายไว้ ไม่ override ทั้ง use case ให้คืนผลคงที่
- SQLite concurrency ใช้คนละ connection/process พร้อม barriers/events; helpers ต้อง spawn-safe บน Windows และปิด connections ก่อน cleanup
- UI race ใช้ deferred responses ควบคุมลำดับ รวม success/error/finally; ไม่พึ่ง sleeps เพื่อให้ test บังเอิญผ่าน
- Files ใช้ _atomic_io; DB writes ผ่าน UoW; clocks และ Decimal inputs ตรึงใน fixtures
- Assert ว่ามี CRITICAL/required inputs ตาม scenario ก่อนทำ mutation และตรวจว่า rejected request ไม่มี durable side effects
- Network unavailable/source absent เป็น negative cases ไม่เปลี่ยนเป็น synthetic verified success

## 18. รายการตรวจรับแบบคงที่

แต่ละ K มี target test files อยู่ใน JSON matrix และในชุดงานข้างต้น ตารางนี้เป็น behavioral inventory บาง case ใช้ parameterized tests หลาย inputs หรือหลายระดับได้ จำนวน 98 คือกรณีตรวจรับ ไม่ใช่คำอ้างจำนวน test functions ที่มีอยู่แล้ว


### K01 — Required samples และ immutable approval (11 กรณี)

| Case ID | ระดับ | Scenario | Expected result |
|---|---|---|---|
| K01-T01 | api | packet มี CRITICAL breach แต่ส่งเฉพาะ benign/N/A sample | 422; ไม่มี signoff หรือ acknowledgement ใหม่ |
| K01-T02 | api | ข้าม required sample หนึ่งรายการจาก snapshot ที่เปิดตรวจ | 422 พร้อมรายการ required IDs ที่ยังขาด |
| K01-T03 | api | ส่งครบ required samples และ acknowledge critical exceptions ทุกตัว | 200; read-back มี signer/hash/evidence ตรง canonical record |
| K01-T04 | api | missing/empty/forged/stale packet checksum | 422; ไม่เปลี่ยน audit state |
| K01-T05 | api | ส่ง invalid ID, duplicate ID หรือ ID จากอีก portfolio/revision | invalid/cross-scope IDs ถูกปฏิเสธ; duplicates ถูก deduplicate ก่อนตรวจ required set/count และ persist เป็น unique canonical IDs |
| K01-T06 | integration | critical exceptions มากกว่า bounded sample cap | critical ทุกตัวต้องมี acknowledgement แม้ไม่ได้อยู่ใน samples |
| K01-T07 | api | portfolio มี holdings แต่ evaluator ข้อมูล required ไม่ครบ | packet BLOCKED และ sign-off ไม่สำเร็จ |
| K01-T08 | integration | empty portfolio และข้อมูลครบ โดยไม่มี samples จริง | policy ให้ review ได้โดยไม่สร้าง sample N/A หรือ acknowledgement ปลอม |
| K01-T09 | integration | concurrent same-packet approvals จากคนละ SQLite connection | คืน canonical ID/signer/evidence เดียวกัน; แถวเดิมไม่เปลี่ยน |
| K01-T10 | integration | อ่าน approved snapshot อีกครั้งหลัง restart และ source ใหม่เข้า | snapshot เดิม reconstruct ได้; source ใหม่เป็น revision ใหม่และต้อง re-review |
| K01-T11 | ui | ยืนยันเพียงบาง required items หรือกำลังเปลี่ยน portfolio | ปุ่ม confirm ใช้ไม่ได้ และ request ส่ง exact displayed packet ID/hash/evidence |

### K02 — Observation mapping และ complete evaluation (10 กรณี)

| Case ID | ระดับ | Scenario | Expected result |
|---|---|---|---|
| K02-T01 | adapter | อ่าน sidecar แบบ MicroQuantOutput ผ่าน EquitySidecarAdapter จริง | revenue_growth_yoy_pct/fcf_margin_pct/raw_analysis_price map ถูก metric/unit |
| K02-T02 | adapter | growth หรือ margin เท่ากับ 0 หรือเป็นค่าติดลบ | รักษา valid zero/negative; ไม่เปลี่ยนเป็น None จาก alias truthiness |
| K02-T03 | domain | metric name ไม่รองรับ หรือ operating/gross margin ไม่มี field จริงใน source | explicit MISSING/UNSUPPORTED; ไม่สร้างค่าแทน |
| K02-T04 | integration | PM ส่ง price override พร้อมมี fundamental rules | อ่าน metrics ที่ยังขาดจาก observer และตรวจพบทุก breach |
| K02-T05 | domain | override กับ sidecar มี metric เดียวกันแต่ต่าง revision/unit | เลือกตาม policy/source identity ที่ประกาศ; unit conflict เป็น INVALID |
| K02-T06 | adapter | NaN/Infinity/string เสีย/alias ให้ค่าขัดกัน | INVALID พร้อมเหตุผล; JSON ไม่ปล่อย non-finite numeric |
| K02-T07 | integration | source stale, available_time หลัง cutoff หรือไม่มี provenance ที่จำเป็น | STALE/NOT_AVAILABLE_AS_OF/MISSING_METADATA; ไม่สรุป NO_BREACH |
| K02-T08 | integration | provider read error หรือ sidecar เสีย | ERROR/repair-required; required rule ทำให้ packet BLOCKED |
| K02-T09 | integration | required rule ไม่มี observation ขณะที่ rules อื่นผ่าน | ผลประเมินครบทุก rule; completeness false และอนุมัติไม่ได้ |
| K02-T10 | api | production get_thesis_service/get_pm_review_service ใช้ temporary configured vault | อ่าน actual adapter chain; API แสดง breaches/statuses ตรง frozen input |

### K03 — Readiness จาก DB/state/rebuild จริง (10 กรณี)

| Case ID | ระดับ | Scenario | Expected result |
|---|---|---|---|
| K03-T01 | adapter | fake SQLite header/page-size/cells/content-offset ที่ดูสมจริง | Q1 FAIL จาก SQLite read-only query; ไม่ READY |
| K03-T02 | adapter | มีชื่อ canonical tables ครบแต่ไม่มี required columns | Q1 FAIL พร้อม missing schema/projection; ไม่ READY |
| K03-T03 | adapter | DB missing/truncated/corrupt หรือเปิดไม่ได้ | health เป็น FAIL; ไม่มีการ initialize/ALTER/repair ระหว่าง audit |
| K03-T04 | integration | schema/column/index version ของ legacy ที่ระบบไม่รองรับ | รายงาน remediation ที่ชัดเจน; ไม่ถือว่าพร้อมโดยอัตโนมัติ |
| K03-T05 | integration | ไม่มี checker/configuration หรือ authoritative path ผิด | FAIL/CONDITIONAL; ไม่ใช้ header heuristic เป็น PASS |
| K03-T06 | integration | manifest ว่าง/ไม่ครบ/file missing/file tampered/storage_contract อย่างเดียว | Q5 FAIL; หลักฐานต้องเป็น frozen inputs ที่ scoped จริง |
| K03-T07 | integration | rebuild packet จาก actual persisted adapters ใน fresh process | payload/hash/source refs ตรง snapshot ที่คาดหวัง โดยไม่ใช้ chat history |
| K03-T08 | integration | ใช้ empty/mock-free-input builder แต่ production state เสีย | Q8 FAIL; hash ยาว 64 ตัวอย่างเดียวไม่เป็น proof |
| K03-T09 | integration | negative human gate หรือ required validator failure | Q9/validator proof FAIL และ overall ไม่ READY |
| K03-T10 | integration | valid DB/projections + valid manifest + successful rebuild + gate proofs | READY เฉพาะ named stage/scope ที่ตรวจครบ; audit ไม่เปลี่ยน source bytes/schema |

### K04 — Independent primary evidence และ evidence tiers (11 กรณี)

| Case ID | ระดับ | Scenario | Expected result |
|---|---|---|---|
| K04-T01 | application | ไม่มี primary artifact/record แต่มี sidecar metrics | INFERRED/UNVERIFIED ตามชนิด claim; ไม่ DOCUMENTED |
| K04-T02 | application | locator มี keyword 10-q/sec.gov แต่ไม่มี verified source body | ไม่ยกระดับ tier; ไม่มี self-validation |
| K04-T03 | adapter | source body hash ไม่ตรง snapshot metadata หรือถูกแก้ไข | INTEGRITY_ERROR/UNVERIFIED; ไม่ใช้ figures |
| K04-T04 | domain | source ตรง ticker แต่คนละ fiscal period/unit | CITATION_MISMATCH/INVALID; ไม่อาศัยเลขที่บังเอิญเท่ากัน |
| K04-T05 | domain | claimed value ต่างจาก independent primary fact | CITATION_MISMATCH; ไม่ fallback เป็น DIRECTLY_TESTED เพราะมีราคา |
| K04-T06 | integration | source ที่ตรวจได้ตรง ticker/period/metric/unit/cutoff และ facts match | factual assertion ได้ DOCUMENTED พร้อม exact artifact/locator/hash |
| K04-T07 | domain | heuristic stop-loss/growth floor/forward narrative อ้างอิง baseline fact ที่ verified | derived/forecast claim คง INFERRED; factual provenance แยกจาก inference |
| K04-T08 | domain | user custom stop/growth/margin หรือ prompt-injection content | custom inference และ sanitization policy เดิมยังทำงาน |
| K04-T09 | api | earnings request ไม่พบ verified actual หรือ consensus ของ period | ตอบ unavailable/404 ตาม final contract; ไม่คืน hardcoded defaults |
| K04-T10 | integration | consensus เพิ่งเผยแพร่หลัง earnings/cutoff หรือคนละ revision | ไม่ใช้ข้อมูลอนาคต; ระบุ missing expected baseline |
| K04-T11 | integration | bridge โหลด actual/consensus/drivers ที่ frozen และมี provenance ครบ | driver sum/units/period/source refs ตรวจได้; calculation trace แยกจาก source verification |

### K05 — THB valuation และ exact exposure (10 กรณี)

| Case ID | ระดับ | Scenario | Expected result |
|---|---|---|---|
| K05-T01 | domain | USD holding: 2 units * USD 100 * THB/USD 35, market_value_thb=0 | มูลค่า THB=7000; ไม่มี USD ปนกับ THB |
| K05-T02 | domain | long USD + short USD + long THB ใน fixture เดียว | gross THB=9350; net THB=5850; signs ถูกต้อง |
| K05-T03 | integration | native price ใช้ได้ แต่ FX missing/stale/future/invalid | quarantine/unavailable position; ไม่เดา rate หรือใช้ USD เป็น THB |
| K05-T04 | domain | zero units หรือ holding ที่ปิดสถานะแล้ว | exposure/count ไม่เพิ่ม; ไม่แทน units ด้วย 1 |
| K05-T05 | domain | shares/price/FX เป็น non-finite หรือ stock price <= 0 | INVALID/quarantine พร้อมชื่อทุก position ที่ผิด |
| K05-T06 | integration | มี missing-price holdings หลายตัว | eligible count นับเฉพาะ valid; missing_price_symbols ครบทุกตัว |
| K05-T07 | domain | price string ความละเอียดสูงและ fractional units | Decimal ตั้งแต่ input ถึง sum; rounding เฉพาะ boundary ตาม policy |
| K05-T08 | integration | current_price_thb กับ native*FX ไม่สอดคล้องหรือคนละ as-of | ใช้ approved coherent source/version หรือ quarantine; ไม่ mix aliases |
| K05-T09 | api | portfolio ที่มี cash กับหลาย currencies และค่าที่ quarantine | API totals/counts/currency/partial-quality ตรง domain ไม่เติม 100% default |
| K05-T10 | domain | long/short input conventions และ exposure percentage denominator | คำนวณตาม contract เดียวกัน; ไม่ double-negate short หรือเปลี่ยน denominator เงียบ ๆ |

### K06 — As-of NAV และ external-flow-aware P&L (12 กรณี)

| Case ID | ระดับ | Scenario | Expected result |
|---|---|---|---|
| K06-T01 | domain | snapshot หลัง review cutoff แม้อยู่ห่างไม่เกิน 4 วัน | exclude future snapshot; ไม่มี look-ahead |
| K06-T02 | adapter | historical review date อยู่นอก days ที่นับจากเวลาปัจจุบัน | query/filter ใช้ requested cutoff ไม่ใช้ wall-clock แทน |
| K06-T03 | integration | deposit ผ่าน actual cash-flow service -> canonical store -> snapshot reader -> PM | gain=0; return=0.00% โดยไม่ฉีด field เฉพาะใน mock |
| K06-T04 | integration | withdraw ผ่าน producer/storage/readback จริง | gain=0; return=0.00% |
| K06-T05 | domain | NAV 100 -> 160 และ external deposit 50 | gain=10; return=10.00% |
| K06-T06 | domain | ไม่มี movement และยืนยัน external flows เป็น 0 จริง | AVAILABLE; gain=0; return=0.00% |
| K06-T07 | integration | ไม่มีคู่ snapshot/ข้อมูล stale/flow history ไม่ครบ/legacy note-only flow | UNAVAILABLE พร้อม reason; values เป็น null ไม่ใช่ 0 |
| K06-T08 | integration | trade, dividend, fee หรือ FX conversion ภายใน portfolio | classify external/internal ถูก; ไม่หัก capital flow ซ้ำหรือใช้ cash-balance delta แทน |
| K06-T09 | integration | USD deposit/withdraw มี event FX ที่ cutoff ยืนยันได้ | THB flow ใช้ Decimal และ source FX ของ event; missing FX ทำให้ unavailable |
| K06-T10 | adapter | CSV/typed snapshot schema v1/v2, unsorted/duplicate/revised rows | round-trip fields ที่จำเป็นครบ; legacy incomplete ถูกระบุและ revision selection ถูก |
| K06-T11 | integration | crash/retry ระหว่าง cash-state mutation กับ flow persistence | ทั้ง balance และ external-flow event commit/rollback ร่วมกัน; retry ไม่บันทึกซ้ำ |
| K06-T12 | ui | P&L null, true zero, stale status และ lifetime-only mover | แสดง unavailable/0.00/label ได้ถูกความหมาย; ไม่มี null.toFixed และไม่เรียก lifetime ว่า daily |

### K07 — Durable audit state ผ่าน production wiring (12 กรณี)

| Case ID | ระดับ | Scenario | Expected result |
|---|---|---|---|
| K07-T01 | integration | ตั้ง temporary DB/AUDIT_STATE_DIR แล้วใช้ get_* providers จริง | ทุก service ใช้ durable repository/path ที่ตั้งไว้; ไม่ใช่ None |
| K07-T02 | integration | API heartbeat/catalyst assessment/estimate revision แล้วสร้าง fresh process | read-back ตรง record/actor/source revision ที่ commit |
| K07-T03 | integration | สอง instances อ่าน state เดิมก่อน A และ B append คนละ event | reload มีทั้ง A และ B; ไม่มี lost update |
| K07-T04 | integration | same event ID/request ID retry payload เดิม | คืน canonical record เดียว; จำนวน records ไม่เพิ่ม |
| K07-T05 | integration | same event ID แต่ payload ต่าง หรือ stale parent/revision | reject conflict ตาม contract; ไม่ overwrite history |
| K07-T06 | adapter | disk/DB write/commit failure หลัง validate ก่อนตอบรับ | propagate failure; ไม่มี success/acknowledgement และ cache ไม่แซง durable state |
| K07-T07 | adapter | corrupted/truncated durable state | repair-required; ไม่กลับไป seed ค่าเริ่มต้นแล้วเขียนทับหลักฐาน |
| K07-T08 | integration | import legacy JSON หลายครั้งและมี duplicate IDs ที่ payload ไม่ตรง | dry-run/integrity report; same-payload import idempotent; conflict หยุดโดยไม่เดาข้อมูล |
| K07-T09 | migration | upgrade old schema/JSON แล้ว restart ขณะมีประวัติ approvals | รักษา ID/author/time/hash/ack scope; ไม่ fabricate legacy evidence |
| K07-T10 | integration | worker crash/retry และเปลี่ยน source/exception fingerprint | event/assessment ผูก revision ถูก; acknowledgement เก่าไม่ใช้กับ facts ใหม่ |
| K07-T11 | architecture | application services ไม่ทำ file persistence เองและ DAOs ไม่ commit | ports/adapters/UoW isolation ผ่าน; file artifacts ใช้ _atomic_io |
| K07-T12 | integration | environment/cache isolation และ rollback หลัง failed migration | provider cache ไม่ชี้ DB เก่า; failed transaction ไม่ publish state ครึ่งเดียว |

### K08 — Current portfolio/request generation ใน UI (8 กรณี)

| Case ID | ระดับ | Scenario | Expected result |
|---|---|---|---|
| K08-T01 | ui | A loaded แล้วเปลี่ยนเป็น B ที่ request ยัง pending | ไม่แสดงข้อมูล A ในฐานะ B; actions ของ packet เก่าใช้งานไม่ได้ |
| K08-T02 | ui | A1 -> B -> A2 โดย A2 ตอบก่อน A1 | เฉพาะ generation ล่าสุดเปลี่ยน data/error/loading ได้ |
| K08-T03 | ui | refresh สองครั้งใน portfolio เดียวกันและ response สลับลำดับ | newer request ชนะ; sample selections ผูก packet revision |
| K08-T04 | ui | request เก่า error/finally หลัง request ใหม่สำเร็จ | ไม่เขียนทับ current error/loading หรือซ่อนข้อมูลใหม่ |
| K08-T05 | ui | initialPacket เปลี่ยนขณะ request เดิมยังค้าง | initial context ใหม่ทำให้ generation เก่าเป็น invalid |
| K08-T06 | ui | เปลี่ยน portfolio/unmount ระหว่าง sign-off หรือ refresh | ไม่มี post-unmount stale update และไม่ส่ง follow-up ผิด portfolio |
| K08-T07 | ui | เปิด sign-off modal แล้ว portfolio หรือ packet hash เปลี่ยน | ปิด/reset หรือ invalidate modal/evidence; ส่ง exact current snapshot เท่านั้น |
| K08-T08 | ui | ticker A -> B response สลับ และ macro/beta/P&L nullable | cancellation ที่แก้แล้วคงอยู่; unknown ไม่แสดงเป็นตัวเลขศูนย์ |

### K09 — Canonical guardrail และ final test gates (5 กรณี)

| Case ID | ระดับ | Scenario | Expected result |
|---|---|---|---|
| K09-T01 | unit | new missing-price warning จาก check_missing_price_policy | canonical MISSING_PRICE_EXCEPTION; tests ตรง semantics ของ producer |
| K09-T02 | adapter | legacy stored warning ใช้ MISSING_AS_OF_PRICE | boundary compatibility map/read ได้; new records ไม่สลับ codes |
| K09-T03 | api | factor-exposure warning หลาย holding ผ่าน canonical/legacy data | missing symbols ครบและ eligible count ถูก ไม่แก้ test เพื่อกลบข้อมูลผิด |
| K09-T04 | system | full relevant backend + frontend + typecheck และ architecture | exit 0 ทุก required gate; ไม่มี skipped/xfail ที่ซ่อน K01-K09 |
| K09-T05 | contract | approved schema/API/type changes และ generated manifests | generator ผลตรง final contracts; ไม่ edit golden fixtures ด้วยมือ |

### RG — พฤติกรรมที่ปิดแล้วและต้องคงไว้ (9 กรณี)

| Case ID | ระดับ | Scenario | Expected result |
|---|---|---|---|
| RG-T01 | api | real session principal และ forged signed_off_by ใน body | signer มาจาก authenticated principal |
| RG-T02 | integration | same packet concurrent retry และ different packet revision ในวันเดียวกัน | same packet canonical; different revision เก็บ history ครบ |
| RG-T03 | integration | legacy DB ไม่มี portfolio_id หรือมี unique date index เดิม | migration rehearsal ผ่านและ portfolio isolation คงอยู่ |
| RG-T04 | integration | corrupt thesis/stale parent/concurrent thesis writers | preserve bytes/history, reject invalid write และ packet BLOCKED |
| RG-T05 | integration | thesis synthesis response เทียบ read-back persisted claim | ID/version/checksum/payload ตรงกัน |
| RG-T06 | domain | observed fact เปลี่ยนหลัง acknowledge ค่าเก่า | new event fingerprint และ pending review ของ facts ใหม่ |
| RG-T07 | integration | canonical Macro IDs, OFR=0 และ spread=0 | รักษา valid zero และ units/as-of |
| RG-T08 | ui | null macro/beta, zero units และหลาย missing positions | ไม่มี crash/fake beta; eligible/missing list ตรงข้อมูลจริง |
| RG-T09 | architecture | single-brain Hermes, dependency isolation และ tool harness | ข้อกำหนดเดิมผ่าน ไม่เพิ่ม LLM router หรือ service-locator workaround |
