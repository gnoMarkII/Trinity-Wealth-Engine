# แผนตรวจสอบ Sector Rotation สำหรับ Dashboard และ Macro AI

วันที่: 3 ตุลาคม 2026  
สถานะ: วางแผนตรวจสอบ ยังไม่ได้รัน acceptance suite ตามเอกสารนี้  
ขอบเขต: working tree ปัจจุบันของ `invest-agents` และ Windows Task Scheduler บนเครื่องนี้

เอกสารอ้างอิงใน workspace: [แผน implementation](../docs/macro-sector-rotation-implementation-plan.md), [แผน remediation และ RC matrix](../docs/macro-sector-rotation-remediation-plan.md), [EOD runbook](sector-rotation-eod.md)

## 1. ผลลัพธ์ที่ต้องพิสูจน์

ตรวจให้ครบสามเส้นทาง: provider → snapshot → dashboard, snapshot → Macro Quant/Economist/Allocator และรายงาน AI → evidence ที่เปิดย้อนกลับได้ การมีโค้ดหรือ unit tests ผ่านบางกรณียังไม่เท่ากับการผ่าน acceptance ทั้งเส้นทาง

เมื่อจบงานต้องมี:

1. ผลตรวจ AC-01–32 และ RC-01–26 พร้อมหลักฐานรายกรณีและรายการข้อบกพร่องที่แก้แล้ว
2. HTTP/API และ dashboard ที่ใช้ข้อมูล revision เดียวกัน รวม authentication, mobile และ archive
3. รายงาน Macro AI จริงใน environment shadow ที่ตรวจ snapshot ID, metric refs, วันที่และหน่วยได้
4. ผลจาก EOD scheduler อย่างน้อย 5 completed US sessions ที่แตกต่างกัน รวมอย่างน้อย 1 weekly close
5. ผลซ้อม rollback ที่ยังเปิดรายงานและ canonical evidence เดิมได้
6. completion report ระบุผลที่ผ่าน สิ่งที่ยังไม่ผ่าน และข้อจำกัดของหลักฐาน

DATA เปิดและ AI ปิดใน environment ผู้ใช้ระหว่างตรวจ การเปิด AI สำหรับ shadow ให้ตั้งเฉพาะ process/environment ทดสอบ การเปิดให้ผู้ใช้เกิดหลัง Gate B ตามแผนเดิม

## 2. หลักฐานตั้งต้นและช่องว่าง

ผล tests ในตารางนี้มาจากบันทึกการตรวจเดิมวันที่ 3 ตุลาคม 2026 ไม่ได้รันซ้ำในการสร้างแผนนี้

| สิ่งที่มีหลักฐานแล้ว | ขอบเขตของหลักฐาน | สิ่งที่ต้องตรวจเพิ่ม |
| --- | --- | --- |
| Sector unit tests 25 ผ่าน | calculations, cache, evidence, binding, stale claims, provider deadline | calendar จริง, boundary ทั้งชุด, หลาย process, crash/restart และ agent graph |
| API route tests 4 ผ่าน | เรียกฟังก์ชัน route ด้วย fake service | HTTP validation, authentication, lifespan, service/adapters จริง |
| EOD script tests 6 ผ่าน | stale limit, scope label และ append log | exit/result ทุกกรณี, task retry, missed start และ log/receipt correlation |
| Report formatter tests 15 ผ่าน | refs, canonical commit และ retry ที่มี fixtures | graph handoff, latest ordering หลายรายงาน และ projection failure |
| Frontend 68 files / 354 tests และ build ผ่าน | รวม cold-start/timeframe/refresh ของ sector component | ข้อมูลจริง, archive navigation, race, missing values, keyboard และ mobile |
| Live smoke ใน scratch | 11/11 sectors, fresh วันที่ 2026-10-02, archive v3 commit/recovery และ cache reuse | adjusted-history semantics และ HTTP/browser/AI จริง |
| Windows task ติดตั้งแล้ว | อังคาร–เสาร์ 10:15 น., interactive logon, limited privilege, retry 2 ครั้ง | ตรวจสถานะและประวัติล่าสุดจาก Task Scheduler |
| Log รอบ configured วันที่ 2026-10-03 | เริ่ม 10:15:23 และจบ 10:15:38 เวลาไทย; `status=ok`, `freshness=fresh`, `missing_sessions=0`, 11/11 sectors | เทียบ `LastRunTime`/`LastTaskResult` และ committed receipt ก่อนนับเป็น scheduler acceptance |

Log configured รอบข้างต้นอยู่ใน `logs/sector_rotation_eod.jsonl` และอ้าง snapshot `sr_ab9a2e1ec7c0066bb060347cdd68523a` ขณะที่ scratch smoke อ้าง `sr_244840f4b12de95b9903da06dea186ff` แม้ as-of date เท่ากัน ต้องเทียบ `input_digest`, session grid และ versions เพื่ออธิบาย revision; ไม่ใช้วันที่อย่างเดียวตัดสินว่า snapshot ต้องมี ID เดียวกัน

Versions ที่ตรวจจากโค้ดปัจจุบัน:

| Contract | ค่า |
| --- | --- |
| Snapshot schema | `sector-rotation-snapshot-v2` |
| Formula | `relative-rotation-v2` |
| Calendar | `us-equity-sessions-v1` |
| Transition rule | `confirmed-after-two-bars-v1` |
| Canonical evidence archive | `sector-rotation-evidence-v3`; มี legacy read path |
| Price basis | `auto_adjusted_close` |
| Universe / benchmark | XLK, XLC, XLY, XLF, XLI, XLB, XLE, XLV, XLP, XLU, XLRE / SPY |

## 3. ลำดับงานและผู้รับผิดชอบ

ผู้ดำเนินการหลักคือผู้พัฒนาโครงการ ผู้ใช้ช่วย login สำหรับ browser smoke เมื่อจำเป็น และให้ความเห็นต่อความชัดเจนของรายงานภาษาไทย

| งาน | ระยะเวลาโดยประมาณ | ต้องเสร็จก่อน | ผลส่งมอบ |
| --- | --- | --- | --- |
| V00 เตรียม baseline และ isolation | 2 ชั่วโมง | — | working-tree manifest, fixture/runtime scope และ evidence index |
| V01 สูตร ปฏิทินและ freshness | 4–6 ชั่วโมง | V00 | independent expected results และ boundary cases |
| V02 Archive, integrity และ flags | 4–8 ชั่วโมง | V00 | receipts, corruption/recovery/compatibility/flag results |
| V03 Concurrency, crash และ run binding | 6–10 ชั่วโมง | V01, V02 | process traces, provider counts และ checkpoint/retry evidence |
| V04 AI handoff และ report publication | 6–10 ชั่วโมง | V03 | captured contexts, resolved refs และ report commit/order results |
| V05 HTTP/API และ dashboard | 4–6 ชั่วโมง | V01, V02; archive/AI view หลัง V04 | HTTP results, browser evidence และ latency measurements |
| V06 Live provider และ AI shadow | 3–5 ชั่วโมง | deterministic cases ของ V01–V05 ผ่าน | provider captures และรายงาน AI จริงอย่างน้อยหนึ่งฉบับ |
| V07 Scheduler และ 5-session shadow | 15–30 นาทีต่อรอบ + 5 US sessions | V00; เริ่มสังเกต task ได้ทันที | task/log/receipt records และ weekly-close evidence |
| V08 Rollback และ completion report | 2–4 ชั่วโมง | V02–V07 | rollback records, AC/RC matrix และ Gate B decision |

รวมเวลาทำงานประมาณ 4–7 วันทำงาน ขึ้นกับข้อบกพร่องที่พบ และต้องรอข้อมูลอย่างน้อย 5 US sessions งาน shadow ทำควบคู่กับการตรวจ deterministic ได้ แต่ไม่นับการ rerun วันเดิมเป็น session เพิ่ม

ลำดับหลัก: V00 → V01/V02 → V03 → V04/V05 → V06 → V08 โดย V07 เก็บข้อมูลควบคู่กัน หากพบข้อบกพร่อง ให้แก้และรันกรณีที่ได้รับผลกระทบก่อนปิดรายการนั้น

## 4. วิธีตรวจรายงาน

### V00 — Baseline และ isolation

- บันทึก commit ปัจจุบัน, `git status`, hash ของไฟล์ที่แก้/ยัง untracked, dependency versions และ effective flags โดยไม่เปิดเผย secrets การมี working tree ที่ยังไม่ commit ต้องใช้ file hashes ผูกผลตรวจกับโค้ดที่ตรวจจริง
- สร้าง vault, runtime, broker DB, checkpoint, WebUI DB และ news store แยกต่อชุด integration และกำหนดก่อน import application modules
- ใช้ `OBSIDIAN_VAULT_PATH`, `INVEST_VAULT_RUNTIME_BASE`, `CHECKPOINT_DB_PATH`, `WEBUI_STATE_DB_PATH`, `NEWS_FUNNEL_STORE_PATH` และ test authentication configuration ที่ชี้ environment ทดสอบ ตรวจให้ legacy runtime variable ไม่ขัดกับ canonical variable
- ปิด workers/scheduler ที่ไม่เกี่ยวข้อง ตรวจค่า effective หลัง fixtures และ lifespan โหลดแล้ว เพราะ root `tests/conftest.py` ตั้ง environment เอง รวม `ENABLE_JOB_WORKERS=true` งาน graph ที่ต้องใช้ worker ให้เปิดเฉพาะ worker ทดสอบ
- ตรวจ module-level paths และ singleton/cache ที่เคย import แล้วให้ตรง test environment รวม composition root ที่ใช้ `lru_cache` ปิด services/executors/SQLite handles เมื่อจบ
- เปิด root isolation guard ใน integration acceptance และบันทึก hash ก่อน/หลังของ `data/` และ `memories/` การตรวจ local ที่ใช้ `--confcutdir` เป็นหลักฐานเฉพาะชุดนั้น หาก guard พบ leakage ให้แก้ fixture/import path ก่อนรันซ้ำ
- รัน integration suites ตามลำดับจนพิสูจน์ว่าไม่ใช้ global temp directory ร่วมกัน การใช้หลาย child processes ใน V03 ให้ใช้ runtime ร่วมเฉพาะ fixture ของกรณีนั้น
- ตรวจ dependency rules และ vault writer inventory ที่เกี่ยวข้องกับ service/domain/adapters ใหม่ โดยไม่เพิ่ม allowlist เพียงเพื่อให้ tests ผ่าน

**ผ่านเมื่อ:** ยืนยัน scopes ได้ก่อนเริ่ม writes, isolation guard ไม่พบการแก้ production data และ reproducible baseline ระบุได้ว่าโค้ดชุดใดถูกตรวจ

### V01 — สูตร ปฏิทิน คุณภาพข้อมูลและ freshness

- ใช้ reference calculation ที่คำนวณจาก fixture โดยไม่เรียก production calculation function เป็น expected result ตรวจ absolute %, excess pp และ relative % แยกกันทุก horizon: 1W/1M/3M/6M/YTD/1Y
- ตรวจ +10% เทียบ SPY +5% และ −5% เทียบ SPY −10% รวมเครื่องหมาย/หน่วยบน UI และ AI ใช้ tolerance ตามการปัดค่าจริง เช่น return ที่แสดง 6 decimals เทียบด้วย absolute tolerance `1e-6` และเก็บค่าก่อนปัด
- ตรวจ rolling warm-up: weekly 26/27 bars, daily 90/91 bars, constant/near-constant ratio และเกณฑ์ variance; fixture สำหรับ calendar ต้องใช้วันที่จริง แยกจาก fixture ทดสอบสมการ
- ตรวจ holiday, observed holiday, DST ก่อน/หลังเปลี่ยนเวลา, regular/early close ก่อน/ตรง/หลังเวลาปิด และกรณีข้ามปี เทียบ session fixture กับปฏิทินตลาดที่เผยแพร่จริง ไม่ใช้ expected จาก resolver เดียวกับที่ทดสอบ
- ตรวจ weekly close ของสัปดาห์ปกติ, ศุกร์หยุด, ข้อมูลวันจันทร์–พฤหัสฯ ของสัปดาห์ที่ยังไม่จบ และ weekly close ที่ขาด
- ตรวจ missing start/end horizon, YTD close ปีที่แล้วขาด, gap กลางช่วง, benchmark ขาด, ETF ขาด, NaN/Inf/ราคาไม่บวก, duplicate dates และข้อมูลหลัง cutoff; ห้ามเลื่อน endpoint หรือบีบ session grid เพื่อให้ครบ
- ตรวจ rank/breadth ด้วย valid denominator จริง, partial coverage และ broad-market conclusion เป็น insufficient data เมื่อข้อมูลไม่พอ
- ตรวจ transitions A→B→B, A→B→A, A→missing→B, changed/confirmed dates และ event ID เมื่อเพิ่มวันหรือแก้ข้อมูลย้อนหลัง
- ตรวจ dashboard freshness ที่ขาด 0/1/3/4 sessions และ EOD lag 0/1/>limit รวม max-stale configuration ที่ไม่ถูกต้อง
- แยก EOD acceptance, dashboard fallback และ AI eligibility: EOD อนุญาต lag ตาม limit; dashboard มี last-good policy; `prepare_for_analysis` ต้องยังปฏิเสธ snapshot stale ตรวจ daily claim stale กับ weekly metric ที่ยัง valid ตาม eligibility ของ binding

**ผ่านเมื่อ:** ทุก boundary ตรง contract, unavailable เป็น null+reason, ไม่สร้างค่าหรือวันที่แทนข้อมูลที่ขาด และ independent expected results ตรง snapshot/DTO/context

### V02 — Canonical evidence, integrity, compatibility และ flags

- ใช้ real broker/evidence/snapshot adapters กับ fixture vault/runtime ตรวจ accepted/pending เทียบ committed, duplicate reuse, conflict, receipt/artifact checksum และ latest pointer
- ตรวจ archive v3 สร้าง snapshot คืนจาก normalized prices + expected sessions + identity ได้ตรงทุก field การแก้ input/grid/identity หรือ artifact หลัง verify ครั้งแรกต้องถูกตรวจพบ
- ตรวจ 5-year history จริงและ long-history fixture 1,400 sessions; วัด body bytes เทียบ limit `4_500_000` และทดสอบ over-size rejection โดย latest/receipt เดิมยังอยู่
- ตรวจ legacy archive v1/v2 และ old Macro report ที่ไม่มี sector fields ด้วย fixture ตาม contract เดิม ต้องเปิดอ่านได้โดยไม่ย้ายไป latest และไม่มี destructive rewrite
- ซ้อม cache loss ใน scratch โดยใช้ snapshot store ว่างและรักษา canonical artifact กับ broker receipt/runtime เดิม แล้วเปิด archived snapshot/history/report ตรวจ digest/ID/metrics เหมือนเดิม การสูญเสีย broker receipt ทั้งหมดเป็น disaster-recovery อีกขอบเขตหนึ่ง ต้องระบุแยก
- ตรวจ cache/runtime tampering หลัง read สำเร็จ, canonical artifact เสีย, ID ไม่พบ และ rollback formula version; ห้ามเสิร์ฟ facts ที่ตรวจ integrity ไม่ผ่าน
- ตรวจ flags ทั้งสี่คู่ใน fresh process: DATA/AI = on/off, on/on, off/off, off/on ตรวจว่า DATA off ไม่ผลิต latest/refresh, AI off ไม่สร้าง sector context/analysis ใหม่, AI on+DATA off ไม่ bypass DATA และ authorized archive/report reads ยังทำงาน

**ผ่านเมื่อ:** ชุด canonical artifacts ที่ใช้สำเร็จ committed และตรวจ integrity ได้, archive recovery ตรง revision, legacy reads ผ่าน และ flags ทำงานตรง serving policy

### V03 — Concurrency, crash/restart และ run binding

- ทดสอบด้วยอย่างน้อยสอง process ที่ใช้ fixture runtime เดียวกัน: cold GET, force refresh, lease หมดอายุ และ provider timeout นับ fetch/attempt/receipt เพื่อยืนยันการ coalesce ต่อ key และไม่ผสม generation
- แยก assertion ของคำขอซ้อนที่ควร coalesce ออกจาก force refresh ครั้งใหม่หลังงานจบ ซึ่งอาจเป็น attempt ใหม่ตาม policy
- ใส่ failure points ก่อน selection, หลัง selection ก่อน publish, broker accepted ก่อน committed, หลัง evidence commit ก่อน cache/latest update, หลัง pin ก่อน LLM และหลัง report commit ก่อน projection
- Restart ด้วย job/task/run identity เดิม ตรวจ selection/command/payload/binding เดิม แม้ latest เปลี่ยน; new job/thread reuse/task ถัดไป/replan ต้อง reset หรือรักษา binding ให้ตรงกรณี
- ทดสอบ two attempts ของ run เดียวให้ได้ winner เดียวและ lock มี timeout ตรวจ stale lease ถูกกู้คืน และไม่มี deadlock/background task ค้าง
- Provider fake ที่ไม่ตอบต้องทำให้ batch ส่งสถานะ timeout ภายใน deadline; ตรวจ process/executor shutdown ด้วย เพราะการ return จาก function อย่างเดียวไม่พิสูจน์ว่า process ออกได้

**ผ่านเมื่อ:** trace พิสูจน์ run เดิมไม่ repin, publish/commit ไม่ซ้ำแบบเปลี่ยน payload, latest ชี้ artifact set ครบ และทุก wait มีขอบเขตตาม configuration

### V04 — AI context, validation และ report publication

- รัน Manager/job flow และ CLI ด้วย fake LLM + real service/adapters; capture handoff ของ ingest/Quant/Economist/Allocator แล้วเทียบ snapshot ID, facts, dates, quality และ versions
- ยืนยัน Economist prompt ใช้ input จริงและ Quant facts หลัง LLM ถูกผูกกลับจาก Python; sector derived evidence ไม่เพิ่ม hard-data support/confidence ของ CPI/GDP
- ใส่ LLM outputs ที่ ref ผิด, snapshot ผิด, ticker/horizon ผิด, value/sign/unit ผิด, stale fact, unsupported transition และ future threshold ที่แอบอ้างเป็น observed fact ตรวจ reject/scoped retry/typed unavailable
- ตรวจ valid partial sector claims และ broad-market insufficient-data; ทำให้ LLM ล้มเพื่อพิสูจน์ dashboard data path ยังใช้ได้
- Canonical report commit ล้มต้องไม่ส่ง success/latest; projection ล้มหลัง commit ต้องเปิด canonical archive ได้และมี warning ตรงสาเหตุ
- สร้างรายงานสองรอบวันเดียวให้งานเก่าจบทีหลัง ตรวจ report IDs แยก, latest ไม่ย้อนกลับ และ Markdown/JSON/API/source refs ตรง canonical body
- เปิด archived report ที่ใช้ snapshot เก่า ในขณะที่ latest เป็น snapshot ใหม่; ตรวจ as-of ของ sector กับวันรายงานและ source drawer

**ผ่านเมื่อ:** refs ทุกตัว resolve ได้จาก pinned revision, invalid/stale facts ไม่ผ่าน validation และ success เกิดหลัง canonical commit

### V05 — HTTP/API, dashboard และ performance

HTTP tests ต้องผ่าน router และ `require_session` จริง ไม่ใช้ direct route call หรือ override authentication เป็นหลักฐาน auth:

| Endpoint | กรณีที่ต้องตรวจ |
| --- | --- |
| `GET /api/macro/sector-rotation/latest` | weekly/daily, default tail, warm, cold 202, stale fallback, failed refresh, DATA disabled |
| `POST /api/macro/sector-rotation/refresh` | force/non-force, request ซ้อน, deadline/error และ capability disabled |
| `GET /api/macro/sector-rotation/snapshots/{snapshot_id}` | latest/old revision, missing 404, integrity failure 503, archive read เมื่อ DATA ปิด |
| `GET /api/macro/sector-rotation/history` | snapshot_id, weekly/daily, 3m/6m/1y/2y, missing/integrity และ invalid parameters |
| Macro dashboard/report API | รายงานใหม่/เก่า, sector unavailable และ canonical refs |

- ทุก sector endpoint ตรวจ session ถูกต้อง/ไม่มี session/หมดอายุ รวม DTO/422/status/content type และ error body ตาม auth/API contract จริง
- Automated HTTP auth ใช้ test credentials/session ของ environment ทดสอบ ส่วน browser smoke ใช้ session ที่ผู้ใช้ login แล้ว ไม่เก็บ password/cookie ในหลักฐาน
- Component tests ใช้ delayed responses เพื่อสลับ daily/weekly/history อย่างรวดเร็ว; response เก่าต้องไม่ทับใหม่ snapshot/map/table/detail ต้องเป็น revision/timeframe ที่เลือกเดียวกัน
- Browser desktop 1440×900 และ mobile 390×844: loading/polling/retry, freshness/coverage, map/tails, sector selection, returns/units, rebased line, timeline และ archive navigation
- ตรวจ keyboard/tab/focus, label/legend ที่ไม่อาศัยสีอย่างเดียว, null ไม่เป็น 0, graph ไม่ลากข้าม gap, mobile table/controls ไม่ทับกัน และ report source drawer เปิด evidence ที่ถูกต้อง
- ใช้ live HTTP กับ snapshot จริงหลัง V06 และทดสอบ Macro/Portfolio sections ที่ได้รับผลกระทบ รวม old report และ AI 404/LLM unavailable
- วัด warm authenticated latest: warm-up 3 requests แล้วอย่างน้อย 100 requests บันทึก p50/p95/max และ response bytes; วัด cold 202 อย่างน้อย 5 ครั้งใน store ว่างแยกกัน
- Engineering targets จากแผนเดิม: warm-cache p95 < 500 ms, cold 202 < 1 s, latest map+table JSON payload < 200 KiB ที่ default tail วัด actual serialized response ทั้งชุด พร้อมระบุ compression/configuration; ห้ามนับ full history เป็น default payload หรืออ้างผ่านโดยไม่มี measurement

**ผ่านเมื่อ:** HTTP contract/auth ผ่าน, UX หลัก desktop/mobile ใช้งานได้, ไม่มี response race และ performance targets มีผลวัด หากเกิน target ให้บันทึก defect/แนวแก้ก่อนสรุป gate

### V06 — Live adjusted history และ Macro AI shadow

- ใช้ isolated vault/runtime และ provider เดียวกับ production เก็บ ticker/date/provider/package version/request settings และ digest ของ sanitized captured input
- เลือก dividend event ของ sector/SPY และ split event ที่ provider รายงานจริงอย่างน้อยหนึ่งกรณี หาก universe ไม่มี split ในช่วงที่ใช้ ให้ใช้ historical split ที่ยืนยันได้เพื่อตรวจ shared adapter และระบุขอบเขตนั้นชัดเจน
- Fetch `auto_adjust=True/False` ในช่วง event เดียวกันจาก provider เดียวกัน ตรวจ adjusted Close เทียบ Adj Close ตาม precision ของ provider, dividend/split adjustment, timezone, cutoff และไม่มี double adjustment ห้ามสรุป semantics จาก quote ล่าสุดอย่างเดียว
- Rebuild snapshot จาก captured prices และคำนวณ horizon/rotation ตัวอย่างแยกต่างหาก เทียบ actual DTO/UI; data revision ใหม่ต้องได้ ID ใหม่และ archive เก่ายังเปิดได้
- หลัง deterministic AI cases ผ่าน เปิด AI เฉพาะ shadow process และสร้าง Macro report จริงอย่างน้อยหนึ่งฉบับภาษาไทย เก็บ run/task/snapshot/report IDs, resolved refs และ canonical committed receipt
- ตรวจรายงานอธิบาย relative strength ร่วมกับ macro โดยมีหลักฐาน macro ของตนเอง ข้อมูล sector ไม่ถูกใช้ยืนยัน GDP/CPI โดยลำพัง และประโยคที่อ้างตัวเลขมีวัน/หน่วย/ref ตรง snapshot
- Retry/resume shadow run เดิมหลัง latest update และเปิด archived report บน dashboard ตรวจ lineage เดิม รวมค่าใช้จ่าย/เวลา/validation results โดยไม่แนบ credentials หรือ raw private prompts

**ผ่านเมื่อ:** adjusted-price semantics มี captured evidence, live HTTP/UI numbers ตรง input และรายงานจริงมี refs ที่เปิดได้ การผ่าน live report ฉบับเดียวเสริม deterministic rejection cases

### V07 — Windows scheduler และ shadow observations

1. ตรวจ task `InvestAgents-SectorRotation-EOD`: absolute Python/script/WorkingDirectory ของ project นี้, local timezone `SE Asia Standard Time`, อังคาร–เสาร์ 10:15 น., interactive logon, limited privilege, StartWhenAvailable, IgnoreNew, execution limit 15 นาทีและ retry 2 ครั้งทุก 15 นาที
2. เทียบ `LastRunTime`, `LastTaskResult`, JSONL configured record และ receipt/archive ของรอบ 3 ต.ค. 10:15 น. ก่อนนับเป็น production observation; `vault_scope=configured` เป็น scope label ต้องตรวจ resolved target ภายในเครื่องด้วย ไม่ใช่หลักฐานว่า path ถูกต้องเพียงอย่างเดียว
3. เก็บ scheduled runs อย่างน้อย 5 completed US sessions ที่ไม่ซ้ำและ 1 weekly close แนะนำเก็บเต็มสัปดาห์ US 5–9 ต.ค. ซึ่งมีรอบเช้าไทย 6–10 ต.ค. 10:15 น. หลังยืนยันปฏิทินจริง รอบ 2 ต.ค. เป็น candidate เพิ่มได้เมื่อ correlation ผ่าน
4. ตรวจหลัง 10:15 และติดตาม stale/failed ภายใน 11:00 น. เวลาไทย บันทึก missed run/logon/sleep/network/provider lag ตามที่เกิดจริง ไม่จำเป็นต้องจำลองรอวันหยุดหรือ DST ด้วย task จริง
5. `status=stale` ภายใน EOD limit คืน exit 0 จึงไม่กระตุ้น scheduler failure retry ต้องอ่าน freshness/missing sessions ด้วย หากยัง stale ให้ติดตามหรือ rerun เมื่อ provider พร้อมตาม runbook; ไม่นับ stale-only round เป็น fresh-data acceptance ของ expected session นั้น
6. Failed/>limit คืน nonzero; ทดสอบ retry, overlap และ missed start ใน task ชั่วคราวที่ชี้ scratch script/runtime ชื่อแยก รักษา production task ระหว่าง failure injection และลบเฉพาะ task ทดสอบเมื่อจบ
7. ครบ 5 session records ต้องมี expected/observed dates, missing sessions, benchmark/sector coverage, snapshot/digest/versions, task result, duration, receipt status และข้อสรุปของผู้ตรวจ
8. อย่างน้อยหนึ่ง weekly-close record ต้องยืนยัน latest completed weekly session และ map/summary ที่ตรงกัน; การมี log วันศุกร์อย่างเดียวไม่พิสูจน์ weekly calculation
9. Failure/recovery และ cache recovery ต้องมีหลักฐานจาก deterministic tests/temporary task หาก production ไม่เกิด failure ให้ระบุว่ารอบ production ปกติและแยก synthetic evidence

คำสั่งอ่านสถานะสำหรับใช้ตอนดำเนินแผน:

```powershell
Get-ScheduledTask -TaskName InvestAgents-SectorRotation-EOD
Get-ScheduledTaskInfo -TaskName InvestAgents-SectorRotation-EOD
Get-Content .\logs\sector_rotation_eod.jsonl -Encoding UTF8 -Tail 20
```

**ผ่านเมื่อ:** task/log/receipt สอดคล้อง, ได้ข้อมูลครบตาม 5 distinct sessions + weekly close และ stale/failed/missed run มีการตรวจจับกับวิธี recovery ที่พิสูจน์แล้ว บันทึกข้อจำกัด interactive logon ใน completion report

### V08 — Rollback และ completion

- ซ้อม AI off กับ DATA on ใน isolated deployment process: latest/dashboard ยังใช้ได้, new AI sector context unavailable, archived reports/evidence ยังเปิดได้
- ซ้อม DATA off กับ archived reads และ Macro/Portfolio ส่วนอื่น; new refresh ต้องถูกปิด, UI capability ตรง response
- ซ้อม provider failure/last-good หมด freshness limit และ formula version ที่ไม่ current; ห้าม republish invalid revision เป็น latest หรือเปลี่ยน observed date
- Restart process และตรวจ flags effective จริง รวม persisted bindings/reports การ rollback ไม่ลบ pinned evidence, broker receipts หรือ revisions
- เติม result ของทุก AC/RC พร้อม artifact link; รายการย่อยที่ยังไม่ตรวจทำให้ทั้งกรณีเป็น PARTIAL ห้ามใช้จำนวน tests ผ่านแทนผล acceptance
- จัดทำ completion report และอัปเดต SR-31–34/Gate status ตามหลักฐานจริง; เปิด AI ให้ผู้ใช้หลัง Gate B ผ่านตาม rollout plan

**ผ่านเมื่อ:** rollback observations ครบ, canonical history ยังอ่านได้และไม่มี acceptance ที่ไม่มีหลักฐานถูก mark PASS

## 5. AC → RC → งานตรวจสอบ

สถานะตั้งต้น: `PARTIAL` หมายถึงมี tests/หลักฐานบางส่วน แต่ยังไม่ครบทุกเงื่อนไขของ AC; `PLANNED` หมายถึงยังไม่มีหลักฐานตรงกรณีที่ตรวจยืนยันใน baseline นี้ ทั้งสองสถานะยังไม่ใช่ PASS รายการ RC เป็น coverage ที่เกี่ยวข้อง ต้องตรวจ subcases ของ AC เพิ่มเมื่อ RC ไม่ครอบคลุมทั้งหมด

| AC | RC ที่เกี่ยวข้อง | งาน | สิ่งที่ต้องพิสูจน์ / สถานะตั้งต้น |
| --- | --- | --- | --- |
| 01 | 05–07 | V01, V05 | absolute/excess/relative values และหน่วย; PARTIAL (numeric unit test ผ่าน) |
| 02 | 26 | V01, V04, V05 | ขาดทุน absolute แต่ relative บวกและ UI/AI อธิบายตรง; PARTIAL |
| 03 | 09 | V01 | constant + near-constant ไม่ fabricate quadrant; PARTIAL |
| 04 | 08–09 | V01 | 26/27 weekly, 90/91 daily และ valid-date/gap boundaries; PARTIAL |
| 05 | 08 (เสริม calendar cases) | V01, V07 | holiday/DST/early-close/cutoff รวมข้ามปี; PLANNED |
| 06 | 08 | V01, V07 | holiday Friday/unfinished week/missing weekly close; PLANNED |
| 07 | 06, 25 | V01, V05 | missing ETF/benchmark, null+reason และ denominator; PARTIAL |
| 08 | — (เพิ่ม cutoff case) | V01 | future input ไม่เปลี่ยน pre-cutoff facts; PLANNED |
| 09 | 15 | V02, V06 | adjustment revision เป็น ID ใหม่ archive เก่าไม่เปลี่ยน; PLANNED |
| 10 | 04 | V03 | multi-process/coalescing/generation/latest atomicity; PLANNED |
| 11 | 02, 12, 17, 21 | V02, V03 | failure ไม่มี invalid pointer/evidence และ last-good policy; PARTIAL |
| 12 | 16–19 | V03, V04 | retry/resume ยัง pinned แม้ refresh latest; PARTIAL |
| 13 | 20, 26 | V04 | LLM ไม่แก้ authoritative sector facts; PARTIAL |
| 14 | 20, 26 | V04, V06 | capture actual Economist prompt/context/dates/quality; PLANNED |
| 15 | 10–11, 26 | V04 | invalid refs/numbers, bounded retry/unavailable; PARTIAL |
| 16 | 26 | V04, V06 | sector-only ไม่เพิ่ม macro hard-data support; PLANNED |
| 17 | 25–26 | V01, V04 | valid partial claims แต่ broad conclusion insufficient; PARTIAL |
| 18 | 23–24, 26 | V04, V05 | AI failure ไม่ทำให้ data map/table ล้ม; PARTIAL |
| 19 | 23 | V05 | stale response, shared selection และ keyboard; PARTIAL |
| 20 | 24 | V02, V05 | old report API/formatter/UI compatibility; PLANNED |
| 21 | 01, 22–23 | V02, V04, V05 | archived revision กับ AI refs และสอง as-of dates; PARTIAL |
| 22 | 23, 26 (เสริม drawer case) | V04, V05, V06 | source drawer เปิด canonical sector evidence; PLANNED |
| 23 | 14–15, 17 | V02 | duplicate/conflict/over-size/receipt states; PARTIAL |
| 24 | 26 | V04, V06 | Web/CLI graph policy/versions/binding เดียวกัน; PLANNED |
| 25 | 03, 14, 25 | V01, V02 | range/tail/diagnostics ไม่เปลี่ยน input identity; PARTIAL |
| 26 | 13, 15 | V01, V02 | confirmation + stable semantic ID + revision immutability; PARTIAL |
| 27 | 18–19 | V03, V04 | new job/task reset, same-task retry/replan reuse; PARTIAL |
| 28 | 22 | V04 | same-day IDs แยก งานเก่าจบทีหลัง latest ไม่ย้อน; PARTIAL |
| 29 | 17, 21 | V02–V04 | accepted/committed และ crash retry exact command; PARTIAL |
| 30 | 01–02, 24 | V02, V05 | cache loss recovery/checksum ไม่เปลี่ยน revision; PARTIAL (scratch recovery ผ่าน) |
| 31 | 03–04, 12, 23 | V03, V05 | cold HTTP 202/polling/coalescing/typed timeout; PARTIAL |
| 32 | 10–11, 26 | V04 | sign/unit/horizon/threshold ไม่กลายเป็น false fact; PARTIAL |

RC coverage สำหรับตรวจไม่ให้ตกหล่น:

| RC | ชุดตรวจหลัก |
| --- | --- |
| 01–03 | V02 lookup/recovery/integrity และ fresh provider-fetch count |
| 04 | V03 process contention และ V05 cold/force HTTP |
| 05–09 | V01 endpoints/YTD/gaps/warm-up/calendar |
| 10–12 | V01 freshness และ V04 claim eligibility |
| 13–15 | V01 events/identity และ V02 immutable evidence/revisions |
| 16–19 | V03 retry/crash/checkpoint/job identity |
| 20–22 | V04 handoff/report commit/latest ordering |
| 23 | V05 UI/API response ordering/archive/null |
| 24 | V02/V08 flags/legacy/recovery |
| 25 | V01/V05 history slices/summary/gaps |
| 26 | V04/V06 Web/CLI, claim validation และ evidence family |

## 6. ชุดทดสอบที่จะใช้และที่ต้องเพิ่ม

ใช้ชุดเดิมเมื่อยังตรงโจทย์ เพิ่ม tests เฉพาะ acceptance ที่ขาดหรือ defect ที่พบ:

| ชุด | สถานะ/งานที่ต้องทำ |
| --- | --- |
| `tests/unit/macro/sector_rotation/` | มีแล้ว; เพิ่ม calendar, endpoint/YTD, near-constant, freshness boundaries และ event identity ที่ขาด |
| `tests/unit/scripts/test_refresh_sector_rotation.py` | มีแล้ว; เพิ่ม end-to-end main exit/status/failure log และ stale-no-retry semantics |
| `tests/tools/macro/test_report_formatter.py` | มีแล้ว; เติม ordering/projection cases ที่ยังพิสูจน์ไม่ครบ |
| `tests/api/test_sector_rotation_http.py` | เสนอเพิ่ม; real HTTP/session/DTO/error/flag contracts |
| `tests/integration/sector_rotation/` | เสนอเพิ่ม; isolated real adapters, two processes, crash, graph handoff และ report lineage |
| `web/src/components/macro/cockpit/SectorRotationDashboard.test.tsx` | มีแล้ว; เพิ่ม response race, complete/partial data, archive/shared selection/null rendering |
| `tests/architecture/test_dependency_rules.py` และ `test_vault_write_boundaries.py` | มีแล้ว; ตรวจขอบเขต imports/canonical writes ของ sector paths |

คำสั่ง baseline ที่เคยใช้ สำหรับรันเฉพาะเมื่อโค้ดเปลี่ยนหรือกรณีที่ยังค้างต้องการยืนยัน:

```powershell
.\.venv\Scripts\python.exe -m pytest --confcutdir=tests/unit/macro/sector_rotation -o addopts= tests/unit/macro/sector_rotation -q
.\.venv\Scripts\python.exe -m pytest --confcutdir=tests/unit/api -o addopts= tests/unit/api/test_sector_rotation_routes.py -q
.\.venv\Scripts\python.exe -m pytest --confcutdir=tests/unit/scripts -o addopts= tests/unit/scripts/test_refresh_sector_rotation.py -q
.\.venv\Scripts\python.exe -m pytest --confcutdir=tests/tools/macro -o addopts= tests/tools/macro/test_report_formatter.py -q
npm --prefix web run test -- src/components/macro/cockpit/SectorRotationDashboard.test.tsx
```

หลัง V00 isolation ผ่านและสร้างชุดที่เสนอแล้ว ให้รัน HTTP/integration/architecture พร้อม root guard ตาม path ที่สร้างจริง ไม่ใช้ `--confcutdir` ตัด guard สำหรับ integration acceptance รัน frontend tests ที่ได้รับผลกระทบและ `check:types`/build เมื่อ schema/UI ลงตัว; ขยาย suite เมื่อมี changes/failures ที่สมควรตรวจเพิ่ม

## 7. รูปแบบหลักฐานและสถานะ

เก็บ artifacts ที่ `scratch/sector-rotation-verification-<run-id>/` โดยมี `baseline.json`, `acceptance-results.json`, `http-results.json`, `performance.json`, `shadow-records.jsonl`, process traces, sanitized receipts และ browser captures เก็บ summary ที่ review ได้ใน tracked Markdown; scratch/logs เป็น local artifacts ต้องนำหลักฐานที่จำเป็นไปไว้ในระบบจัดเก็บที่ใช้ส่งมอบจริงก่อนปิดงาน

Acceptance record ทุกกรณีมี: AC/RC IDs, scenario/subcase, code hashes/version/configuration, fixture/provider capture digest, expected/actual, command/time/environment scope, artifact links, status และ defect ID หากล้มเหลว

Shadow record ต้องมี: US expected session, Thai task/run time, actual as-of, task result, status/freshness/missing sessions, 11-sector/benchmark coverage, snapshot/input digest, formula/calendar/config, duration, committed receipt, weekly close และผู้ตรวจ `finished_at` มีใน success log ปัจจุบัน; failed attempt duration/receipt/digest ต้องเก็บเพิ่มจาก trace/Task Scheduler หรือ verification harness

ใช้สถานะ `PLANNED`, `PARTIAL`, `PASS`, `FAIL`, `BLOCKED` โดย BLOCKED ระบุ dependency ที่ขาด เช่น session/login/provider/เวลาที่ต้องรอ และงานที่ยังทำต่อได้ Test count หรือการมี field ใน output เพียงอย่างเดียวไม่ทำให้ acceptance เป็น PASS

## 8. เกณฑ์ gate และการจัดการข้อบกพร่อง

**Gate A:** ปิด regression issues เดิม SR-F01–08 ด้วยหลักฐานที่เกี่ยวข้อง รวม deterministic math/integrity/binding/report cases ไม่มี P0/P1 ค้าง และ isolation/architecture checks ของเส้นทางที่แก้ผ่าน หลักฐานเดิมที่มีบางส่วนต้องเติม subcases ก่อนอ้างว่า Gate A ครบ

**Gate B:** AC-01–32 และ RC-01–26 มีผลครบ, live provider/HTTP/browser/AI checks ผ่าน, ได้ shadow อย่างน้อย 5 completed US sessions + 1 weekly close, operational recovery/rollback ผ่าน และ completion report ผูกกับโค้ดที่ตรวจจริง จึงเปิด AI ให้ผู้ใช้และปิด SR-31–34 ตามแผนเดิม

ลำดับแก้ข้อบกพร่อง:

- P0: authentication bypass, production data leakage หรือ canonical evidence ถูกเขียนทับ/เสียหาย ให้หยุดเส้นทางที่เกี่ยวข้องและแก้ก่อนทำ writes เพิ่ม
- P1: สูตร/วันที่/หน่วยผิด, stale fact ผ่าน AI, run repin, success ก่อน commit, archive lineage ผิด, refresh/process ค้าง หรือ rollback อ่านหลักฐานไม่ได้ ต้องแก้ก่อน release
- P2: UX/performance ที่ไม่ผ่าน requirement ต้องมีผลตรวจและแนวแก้ ห้ามลดเกณฑ์หรือ mark PASS โดยไม่มีหลักฐาน

เมื่อแก้แล้ว ให้รัน failing case และชุดที่ได้รับผลกระทบ บันทึกผลใหม่แยกจากผลเดิม หาก change เปลี่ยน formula/calendar/input identity ต้องเก็บ shadow ของ revision ใหม่ให้เพียงพอที่จะพิสูจน์ revision ที่จะ release

Checklist จบงาน:

- [x] V00 baseline/isolation/architecture (PASSED)
- [x] V01 calculation/calendar/freshness (PASSED)
- [x] V02 evidence/recovery/compatibility/flags (PASSED)
- [x] V03 concurrency/crash/run identity (PASSED)
- [x] V04 AI context/validation/report publication (PASSED)
- [x] V05 HTTP/auth/dashboard/performance (PASSED)
- [x] V06 live adjustment semantics/AI report (PASSED)
- [ ] V07 scheduler correlation/5 sessions/weekly close (PARTIAL - 1/5 sessions completed, active observation)
- [x] V08 rollback/complete AC–RC evidence/Gate B (Rollback rehearsal PASSED, Gate A PASSED)

## 9. สิ่งที่ดำเนินการในการสร้างแผนครั้งนี้

อ่านแผนเดิม, test case inventory, API/contracts, EOD script/task registration script และ JSONL logs เพื่อสร้างแผนนี้ พบ configured success log รอบ 10:15 น. แต่ยังไม่ได้ query Task Scheduler หรืออ่าน canonical receipt ของรอบนั้น ไม่มีการรัน tests/LLM/browser, เปลี่ยน flags หรือติดตั้ง task เพิ่มในการวางแผนครั้งนี้

เอกสารหลักเก็บใน `scripts/` เพื่อให้ส่งไปกับ repository ได้ เพราะ `.gitignore` ปัจจุบัน ignore `docs/` ทั้งโฟลเดอร์ เอกสารอ้างอิงใน `docs/` จึงเป็น local references ที่ต้องจัดการก่อนส่งมอบใน PR
