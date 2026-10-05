# แผนตรวจรับ Macro: แหล่งข้อมูล → AI → รายงาน → Dashboard

วันที่จัดทำ: 4 ตุลาคม 2026  
สถานะ: วางแผนจากโค้ดและหลักฐานปัจจุบัน — ยังไม่ได้ดำเนินการตรวจรับตามแผนนี้  
เป้าหมาย: ปิดข้อบกพร่องและเตรียมหลักฐานให้ครบก่อนตรวจรับรอบเดียว โดยตรวจทั้งความถูกต้องของข้อมูล ความครบถ้วนที่ส่งเข้า AI และการแสดงผลบนหน้าเว็บจริง

## 1. ความหมายของ “ผ่านในทีเดียว”

ทำงานเตรียมและแก้ไขทั้งหมดก่อนเริ่มรอบตรวจรับ: ตรวจ dependency, เติมสัญญาข้อมูล, แก้ audit ที่อาจผ่านผิด, เพิ่มกรณีทดสอบที่จำเป็น และซ้อมด้วยข้อมูลตรึงชุดเดียวกัน จากนั้นจึงรัน AI จริงหนึ่งรอบและตรวจหน้าเว็บกับรายงานที่ได้จากรอบนั้น

รอบตรวจรับต้องใช้ code/config/input revision ที่ระบุแน่นอน ไม่แก้โค้ดหรือเปลี่ยน input ระหว่างทาง หากพบข้อผิดพลาดที่ต้องแก้ ให้ปิดรอบนั้นเป็น FAIL/BLOCKED เก็บหลักฐานไว้ และเปิดรอบใหม่หลังแก้พร้อมวิเคราะห์ผลกระทบ ห้ามแก้แล้วนำผลก่อนและหลังมารวมเป็น PASS รอบเดียว

เป้าหมายนี้ลดการวนรัน AI และการตรวจซ้ำที่ไม่จำเป็น แต่ไม่รับประกันว่าแหล่งข้อมูลภายนอกจะพร้อมตลอดเวลา การขาด provider, browser หรือ model ที่จำเป็นต้องปรากฏเป็น BLOCKED ห้ามแทนด้วยผล unit test หรือ mock แล้วสรุปว่าผ่านครบ

## 2. ขอบเขตและระดับการรับรอง

| ระดับ | สิ่งที่ต้องพิสูจน์ | เงื่อนไขสรุปผล |
| --- | --- | --- |
| A — ข้อมูลและการคำนวณ | แหล่งจริง, วัน/หน่วย/ประวัติ, การแปลงข้อมูล, freshness, scoring, lineage และ API | ผ่าน G0–G4 และกรณีที่เกี่ยวข้องครบ |
| B — Macro ครบเส้นทาง | ระดับ A + model invocation จริง + publication + browser จริงครบทุก tab/archive/error state | ผ่าน G0–G8 และ MC-01–MC-72 ครบ เป็นเป้าหมายหลักของแผนนี้ |
| C — การทำงานต่อเนื่องของ Sector/EOD | หลักฐาน scheduler หลาย session, weekly close และ rollback ตามแผน Sector เดิม | ตรวจแยกตามระยะเวลาจริง ห้ามอ้างว่าผ่านจากการรันวันเดียว |

หน้าเว็บในขอบเขตคือ `/macro` และ tabs `ai`, `us`, `th`, `cross-border`; รวม dashboard summary, detail/drawer, indicator history, sector rotation, crypto liquidity, การเลือก archived report และการสั่งอัปเดตบทวิเคราะห์

ไม่ถือว่าค่าในรายงานเก่าต้องเปลี่ยนตาม provider ล่าสุด: รายงานเก่าต้องคง input เดิม ส่วน widget ที่เป็นข้อมูลตลาดล่าสุดต้องระบุเวลา/ที่มาของตน หากนำ fallback จากรายงานมาใช้ต้องแสดงว่าเป็น fallback และบอกวันที่รายงาน

## 3. หลักฐานเริ่มต้นและช่องว่างที่ต้องปิด

อ้างอิง [audit วันที่ 4 ตุลาคม](../docs/macro-audit-2026-10-04.md), [ผลตรวจ API เดิม](../tests/audit_macro_page_result.json) และ [ผลดึงข้อมูลจริงหลังแก้](../tests/audit_macro_pipeline_result.json) ผลเหล่านี้เป็น baseline เท่านั้น ไม่ใช่ผลตรวจรับแผนนี้

| หลักฐานที่มี | สิ่งที่พิสูจน์ได้ | ช่องว่าง |
| --- | --- | --- |
| 14 endpoints และ 20 indicator series ตอบสำเร็จ | เส้นทาง API ที่ audit เรียกใช้งานตอบกลับได้ | ยังไม่ได้เทียบทุกค่ากับต้นทาง ตรวจทุก schema/status หรือแสดงผลใน browser จริง |
| รายงานเดิมมี observables 107 รายการ ตรงกับ snapshot ที่ audit พบ | comparison เดิมไม่พบ missing/value/unit/date/validity mismatch | snapshot รายวันอาจถูกเขียนทับ; ต้องผูก revision/hash และเทียบ metadata/status/provenance ด้วย |
| ingest ใหม่มี 126 รายการ: valid 115 / invalid 11 | extraction ของ ingest และ adapter ที่ script เรียกทำงานกับแหล่งจริง | script ยังไม่ครอบคลุม `evaluate_macro_matrix` ทั้งหมด, scoring, sector และ model invocation จริง |
| backend 193 passed / 1 skipped | กลุ่ม tests ที่เลือกผ่าน; live LLM ถูก skip | ไม่ใช่ผล full suite หรือ live AI ใหม่ |
| frontend 74 passed และ typecheck/build ผ่าน | กลุ่ม component tests ที่เลือกและ build ใช้งานได้ | ยังไม่ได้เปิด browser และยังไม่ใช่ full frontend suite |
| Yahoo 28 รายการมีวันสังเกตจริง | ปัญหาวัน observation ที่แก้มีหลักฐาน extraction | ต้องเทียบราคาก่อนหน้า หน่วย corporate actions และจำนวนที่คาดหวังจาก config |

ข้อบกพร่อง/ความเสี่ยงที่ต้องจัดการก่อนรับรอง:

1. `scripts/audit_macro_page_data.py` จบด้วย exit code 0 แม้ผลตรวจมีปัญหา และเมื่อไม่พบ snapshot ยังอาจไม่มีรายการ missing ให้เห็น ต้องทำให้ fail ได้จริง
2. หยุดเทียบ report กับไฟล์รายวันด้วยวันที่อย่างเดียว ต้องระบุ immutable snapshot/revision และ checksum ที่สร้างรายงานนั้น
3. `scripts/verify_macro_data_pipeline.py` ตรวจเพียงบางเส้นทาง; ต้องขยาย coverage ถึง tool evaluation, derived metrics, scoring และ input จริงของทุก agent
4. Thai GDP/CPI/MPI อ่านจากไฟล์ local ที่ระบุ verified; ต้องตรวจเอกสารต้นทางและประวัติ ไม่ถือว่า flag verified เป็นหลักฐานในตัวเอง
5. Thai public debt ที่ baseline พบเป็น period 2026-02-28 และ FRED บาง series ต่างประเทศเก่ามาก ต้องได้ข้อมูลที่ใช้ได้จากต้นทางที่นิยามตรงกัน หรือประกาศ unavailable/stale พร้อมกันออกจากการประเมินตามสัญญา
6. ต้องตรวจทั้ง registry และ assessment ที่ฝังมาแล้ว เช่น `fiscal_health`/`thai_yield_curve`; การกรอง registry อย่างเดียวไม่พอ หาก UI อ่านค่าจาก assessment อีกทาง
7. Crypto มีหลายแหล่งและหลายวัน ต้องตรวจ freshness ราย component; วัน BTC benchmark ไม่ควรทำให้ stablecoin/ETF ที่เก่าดูใหม่
8. หน้า Macro ต้องไม่ค้างเพราะ provider เดียว ไม่มี method แล้วถูกนับเป็น fulfilled หรือ response เก่าทับ response ใหม่
9. ต้องแยก “วันที่ fetch”, “effective/observation date”, “period”, “วันที่ AI วิเคราะห์” และ “วันที่ report commit” บนข้อมูลและหน้าจอ
10. ต้องมี browser ที่เชื่อมต่อได้และ model credentials ที่ใช้ได้ก่อนเปิดรอบตรวจรับระดับ B; รอบ audit ก่อนหน้าไม่มี browser จึงยังยืนยันภาพจริงไม่ได้

จำนวน 107/126/115/11 และจำนวน test เดิมห้ามนำไปเป็นค่าคาดหวังถาวร ให้สร้าง expected set จาก source contract และ config ของ revision ที่รับรอง

## 4. Gate ที่ต้องผ่านตามลำดับ

| Gate | งาน | เงื่อนไขออกจาก Gate | หลักฐาน |
| --- | --- | --- | --- |
| G0 | เตรียม environment และตรึง scope/config | dependency/model/browser/API/auth พร้อม; รู้ expected coverage และนโยบาย degraded data | manifest, preflight, source contract |
| G1 | แก้ช่องว่างของตัวตรวจและ lineage | ตัวตรวจจับ failure ได้; immutable binding/coverage/status/metadata ครบ | negative tests, checksum checks |
| G2 | รับรอง source และ ingest | required sources ใช้ได้ หรือผ่าน degraded policy ที่กำหนดก่อนรัน; primary evidence ครบ | raw payloads, source comparison, freshness matrix |
| G3 | คำนวณและสร้าง QuantScore | สูตร/unit/date/eligibility/score ตรง fixture และข้อมูลตรึง; ไม่เกิด data loss | calculation reconciliation, observables diff |
| G4 | regression และ rehearsal | backend/frontend/schema/build ผ่าน; graph แบบ mock ตรึงครบ; API พร้อม | test logs, rehearsal handoff receipts |
| G5 | AI จริงหนึ่งรอบ | ทุก agent ได้ input ตาม contract; output/citations/commit ผ่าน | invocation receipts, sanitized prompts, report |
| G6 | HTTP และ archive กับรายงานจริง | identity/values/status/refs ตรง evidence; latest/archive ไม่ปะปน | API payloads, reconciliation |
| G7 | browser จริง | ทุก tab/detail/history/refresh/error/mobile ตรวจด้วยข้อมูลเดียวกับ G6 | screenshots, UI assertion results |
| G8 | failure recovery และสรุปตรวจรับ | กรณี failure ผ่าน, ไม่มีผลค้าง, ทุก case มีหลักฐาน/สถานะ, ตรวจ drift รอบสุดท้าย | failure matrix, final result, manifest hashes |

ก่อน G5 ต้องผ่าน G0–G4 ทั้งหมด รวมทั้ง fixture สำหรับ browser failure/race scenarios การรัน G5 แล้วค่อยค้นว่าตัวตรวจหรือ browser ใช้ไม่ได้ผิดลำดับของแผนนี้

## 5. Environment และหลักฐานที่ต้องตรึง

### 5.1 แยกพื้นที่ทดสอบจากข้อมูลที่ใช้งานอยู่

- เก็บ working tree เดิมทั้งหมด: บันทึก git HEAD, staged/unstaged diff digest และ hashes ของไฟล์ untracked ที่เกี่ยวข้อง ไม่ reset/clean/stash งานผู้ใช้โดยอัตโนมัติ
- ใช้พื้นที่ shadow ภายใน workspace สำหรับ vault, state DB, checkpoint DB, evidence และ outputs ของรอบทดสอบ สำรองรายงานเดิมและบันทึก hashes ก่อนเริ่ม
- ตรวจ path ที่ resolve จริงทั้งหมด โดยเฉพาะไฟล์ `data/macro/thailand/official_hard_data.json` และ snapshot/cache ที่อ้าง relative path; หาก env override ไม่ครอบคลุม ให้เพิ่มการตั้งค่า path หรือใช้ working copy ที่แยกก่อนรัน
- ตั้ง `OBSIDIAN_VAULT_PATH`, `WEBUI_STATE_DB_PATH`, `CHECKPOINT_DB_PATH`, `NEWS_FUNNEL_STORE_PATH` ให้ชี้ shadow ตาม config ที่ระบบรองรับจริง
- ปิด scheduler/background processes ที่อาจเขียน input ระหว่างตรวจ ใน shadow ใช้ `SCHEDULER_ENABLED=false` และ `ENABLE_BACKGROUND_WORKERS=false`; เปิด job workers เฉพาะเมื่อกรณี queue ต้องใช้ และตรวจว่า process อ่านค่า config จริง
- รัน pytest และ live pipeline ตามลำดับ หลีกเลี่ยงให้ live process เขียน protected data ระหว่างชุดทดสอบที่ตรวจ data leakage
- Live acceptance ใช้ `ALLOW_MOCK_MACRO_INGEST=false` และไม่ใช้ offline evaluation เพื่อแทน model จริง; offline/mock ใช้ได้เฉพาะ rehearsal ที่ระบุชนิดหลักฐานชัดเจน
- ปิด notification/Discord dispatch ใน shadow ผ่านตัวเลือกที่ระบบรองรับ และตรวจว่าไม่มีการส่งออกจากกรณีสร้างงานหรือ retry
- ไม่เก็บ API keys, cookies, session tokens หรือ secrets ลง prompt logs/manifest; เก็บเพียงชื่อ config และการตรวจว่าพร้อมใช้งาน

### 5.2 รูปแบบ evidence bundle

สร้าง `tests/artifacts/macro-acceptance/<acceptance_id>/` ตอนลงมือทำ ตรวจว่า path นี้เหมาะกับการเก็บ artifact ของ repo และไม่เผยข้อมูลส่วนตัวจาก portfolio โดยไม่จำเป็น

```text
manifest.json
source-contract.json
field-lineage.json
raw/                     # payloads และเอกสารต้นทางพร้อม checksum
snapshots/               # immutable copies พร้อม revision
reconciliation/          # coverage/value/date/unit/metadata/score diffs
tests/                   # commands, exit codes, duration, junit/results
ai/                      # invocation receipts และ prompt/output ที่ตัด secrets แล้ว
api/                     # HTTP status, schema, report/indicator/sector payloads
ui/                      # screenshots และผล assertion รายกรณี
failures/                # injection, recovery, publication/queue receipts
result.json
completion.md
```

`manifest.json` ต้องมี acceptance ID, เวลาทั้ง UTC/Asia-Bangkok, code/config/formula/schema versions, dependency lock digests, timezone/calendar version, source contract digest, raw/snapshot/input hashes, macro run/task/report IDs และขอบเขตที่รับรอง

เก็บข้อมูลฉบับเต็มที่จำเป็นต่อการตรวจ ไม่ใช้ sample 5 แถวเป็นหลักฐานการคำนวณทุกประเทศ/tenor/sector ห้ามให้ report body เพียงไฟล์เดียวแทน input/output ของทุก stage

### 5.3 สถานะมาตรฐานและ exit code

| สถานะ | ความหมาย | นับว่ารับรองผ่านหรือไม่ |
| --- | --- | --- |
| PASS | assertion ครบและมีหลักฐานของ revision ที่รับรอง | ใช่ |
| FAIL | ข้อมูล/พฤติกรรมขัด acceptance criterion | ไม่ |
| BLOCKED | ขาด dependency/input/browser/model ทำให้พิสูจน์ไม่ได้ | ไม่ |
| NOT_RUN | ยังไม่ได้ตรวจ | ไม่ |
| N/A | อยู่นอก scope ที่ตรึงไว้ก่อนรัน มีเหตุผลและผู้รับผิดชอบ | ไม่ใช้แทนกรณีบังคับ MC-01–MC-72 |

มาตรฐาน exit code ที่ต้องเพิ่มให้ตัวตรวจ: `0` = ทุกกรณีบังคับ PASS, `1` = พบ FAIL, `2` = BLOCKED/ตรวจไม่ครบ, `3` = ตัวตรวจเองผิดพลาด บันทึก partial evidence แม้ fail; ไม่จับ exception แล้วคืน 0

## 6. สัญญาข้อมูลและ field lineage

### 6.1 รายการที่ต้องบันทึกต่อ field/observable

ขยาย [data lineage matrix เดิม](../tests/fixtures/macro/data_lineage_matrix.json) หรือสร้าง contract เพิ่ม โดยต้อง map ทุก field ที่ UI ใช้และทุก input ของการคำนวณ ไม่ตรวจเฉพาะ observable IDs ที่เลือกไว้ใน script

| กลุ่มข้อมูลใน contract | รายละเอียดขั้นต่ำ |
| --- | --- |
| Identity | field path, observable ID/alias, series/ticker ID, provider, source endpoint/artifact, contract version |
| Value | raw value/unit, normalized value/unit, transform/formula version, rounding เฉพาะตอนแสดงผล |
| Time | observation/effective date, period start/end, publication date เมื่อมี, fetched_at, timezone และ release/calendar semantics |
| History | จำนวน period ที่สูตรต้องใช้, missing/duplicate periods, revisions, adjustment policy |
| Quality | status, is_valid, freshness policy/reason, source authority, partial/coverage และข้อจำกัด |
| Dependency | required/optional, scoring role, derivative input IDs, eligibility และนโยบาย degraded data |
| Consumers | Quant/Economist/Allocator/report/API/UI field และเหตุผลหาก field ใช้เฉพาะ UI |
| Evidence | immutable snapshot ID, input digest, primary-source evidence checksum และ report/run binding |

“ส่งครบให้ AI” หมายถึงทุก input ที่กำหนดให้ใช้วิเคราะห์ถูกส่งครบ พร้อมวัน หน่วย สถานะและ lineage; field ที่มีเพื่อ UI/diagnostic เท่านั้นต้องระบุว่าไม่ใช่ AI input และอธิบายเหตุผล ห้ามปล่อย field หลุดโดยไม่มี mapping

### 6.2 กลุ่ม source ที่ต้องตรวจทั้งหมด

| Source family | สิ่งที่ต้องตรวจ | สิ่งที่ต้องระวัง |
| --- | --- | --- |
| Yahoo global/regional/Thai tickers | expected tickers จาก `ticker_config.py`, last/previous bar/date, FX/index/ETF/commodity unit | exception ราย ticker อาจถูกซ่อน; markdown ไม่ว่างไม่ได้แปลว่าครบ; วันหยุดตลาดและ futures/spot ต้องนิยามชัด |
| US FRED growth/inflation/labor/liquidity | series IDs, frequency, SA/NSA, raw units, history, transformations และ release dates | quarterly label ต้นไตรมาสไม่ใช่ publication date; discontinued series ห้ามถือว่ายังสด |
| Country FRED | expected countries/series, equivalent definition และ replacement mapping | ห้ามเปลี่ยน GDP level เป็น growth หรือ policy rate เป็น series คนละความหมายเพื่อให้ coverage ผ่าน |
| Treasury yields | ทุก tenor ที่ API ส่ง, วันต้นทาง, percent และ spreads | ไม่เลือกเฉพาะ 2Y/10Y; bps = percentage-point difference × 100 |
| Treasury auctions | Note/Bill โดยเฉพาะ 10Y/13W, auction date, bid-to-cover, history/baseline | ห้ามใช้วัน fetch แทน auction date; missing baseline ต้องเห็น |
| CBOE volatility | equity/metals/oil รวม GVZ/VXSLV/OVX, percentile, window/date | field หนึ่งมีข้อมูลไม่ควรทำให้อีก field ที่หายถูกนับว่าครบ |
| CFTC COT | report date, positions/change, categories, contracts/unit | reporting lag และ market definitions ต้องคงเดิม |
| OFR financial stress | aggregate/categories, dates, missing category | category metadata ต้องไม่หายระหว่าง builder → AI |
| BIS policy rates | ทุก country, fractional values, monthly effective period และ US/TH spread | fetched date ไม่ใช่ effective date; Fed midpoint ต้องไม่ถูกปัดก่อนคำนวณ |
| Thai GDP | NESDC evidence, real/nominal/YoY/QoQ definition, quarterly history | flag verified ใน local file ไม่แทนเอกสารต้นทาง; ต้องมี history ตามสูตร |
| Thai headline/core CPI | หน่วยงานทางการที่ใช้จริง, index vs YoY, monthly period และ history | ห้ามผสม headline/core หรือ SA/NSA; previous/YoY ต้องมี period ที่ถูกต้อง |
| Thai MPI | OIE evidence, index/base year, growth/period/history | ห้ามใช้วันที่แก้ไฟล์เป็น observation date |
| Thai fiscal debt | MOF/PDMO period, outstanding debt, government debt, debt/GDP, units | payload ล่าสุดที่ fetch ได้อาจยังมี period เก่า; กรองทั้ง registry และ fiscal assessment |
| Thai yields/flows/valuation/breadth | ThaiBMA/SET/SEC/แหล่งที่ adapter ใช้จริง, tenors, daily flows, PE, counts | net/gross, million/billion THB, zero/null, settlement/reporting date และ fallback ต้องไม่สลับ |
| Crypto benchmark | BTC price, gold benchmark, ratio formula/currency/date | denominator futures/spot ต้องมีชื่อถูกต้อง; BTC price ไม่ควรหายเพราะ ratio ใช้ไม่ได้ |
| Stablecoin liquidity | supply, 7D/30D growth, history, top constituents/coverage | `is_partial` และ component dates ต้องมี; aggregate BTC date ไม่ทำให้ supply ใหม่ขึ้น |
| Crypto ETF flows | net flow, date, optional status/source | ค่า 0 ใช้ได้; provider ล้มเหลวต้องไม่กลายเป็น 0 หรือ Neutral ที่ไม่มีคำอธิบาย |
| Sector rotation | 11 sectors + SPY, adjusted prices, completed sessions, returns, ranks/relative strength, evidence binding | daily/weekly ต้องตรวจคนละ revision ที่ผูกไว้; ETF price returns ไม่ใช่ fund flows |
| News/YouTube/context ที่ agent ใช้ | source/title/time/extraction status, ข้อจำกัดและ input selection | ต้องมี lineage ของ context ที่มีผลต่อ narrative; headline ไม่ใช้แทน quantitative evidence |

### 6.3 ความสดและนโยบายข้อมูลขาด

1. แยก freshness ราย source/field และใช้ period/calendar ตามความถี่; ค่า daily, weekly, monthly, quarterly, annual/event ไม่ใช้เกณฑ์เดียวกัน
2. ทดสอบ cutoff ที่ `limit-1`, `limit`, `limit+1`, missing date, future date, weekend/holiday และ quarterly period end พร้อมเก็บ original label
3. ระบุว่า timestamp ที่ไม่รู้ต้องเป็น unknown ห้ามเติมวันที่วันนี้หรือ epoch แล้วตีความว่าเป็นข้อมูลจริง
4. required scoring dependency ขาด → dimension เป็น Unknown/unavailable พร้อมเหตุผล; ห้ามใช้ neutral/0/default เพื่อสร้างคะแนนดูครบ
5. optional source ขาด → แสดง degraded state เฉพาะ component พร้อมเหตุผล ไม่ทำให้แหล่งอื่นที่ใช้ได้หาย และไม่เพิ่ม confidence เหมือนข้อมูลครบ
6. stale/invalid fields เก็บได้เพื่อ audit แต่ห้ามเข้าสูตรหรือถูก AI อ้างเป็นค่าปัจจุบัน; คำอธิบายช่องว่างอ้าง ID ได้โดยบอกว่าใช้ประเมินไม่ได้
7. กำหนด required/optional และ degraded policy ก่อน G2 ห้ามลด required เป็น optional หลังเห็นว่า provider ล้มเหลวเพื่อให้รอบนั้นผ่าน
8. ชุด live ที่ใช้รับรองต้องไม่มี unexpected gap; gap ที่อนุญาตต้องตรง policy ที่ตรึงไว้ และ UI/AI/scoring ต้องแสดงผลตามนั้นครบ

## 7. งานเตรียมและแก้ไขก่อนตรวจรับ

| งาน | จุดดำเนินการหลัก | ผลส่งมอบ/Definition of Done |
| --- | --- | --- |
| W01: coverage contract | ticker config, lineage matrix, API schemas, UI consumers | expected IDs/fields ครบ; required/optional/AI-used ชัด; duplicate/extra/missing ตรวจได้ |
| W02: strict audit | `scripts/audit_macro_page_data.py`, `scripts/verify_macro_data_pipeline.py` | ไม่พบ snapshot/endpoint/series/field → FAIL/BLOCKED; strict exit codes; full payload comparison |
| W03: immutable run evidence | manager, ingest, formatter, run identity และ strategy vault | raw/snapshot/quant/prompt/report ผูก run ID/revision/hash; snapshot วันเดียวกันต่าง run ไม่ทับหลักฐาน |
| W04: source remediation | Thai official adapters/local data, MOF, country series และ crypto | primary evidence/history ครบ; replacement definition ตรง; freshness ราย component; gap policy ถูกต้อง |
| W05: normalization/score guards | evaluation, observables, scoring, schemas | status/is_valid สอดคล้อง; duplicate ID fail; finite values; history/units/formulas/eligibility ตรวจได้ |
| W06: agent handoff capture | Quant, Economist, Allocator, manager | actual tool output/input/output receipts; compare full payload; budget/context-size preflight |
| W07: publication/job contract | daily runner, job runner, report formatter/archive | terminal success ต้องมี committed report; bounded/idempotent retry; failed job ไม่แสดงสำเร็จ |
| W08: dashboard resilience | Macro page, Thai/US/cross-border/crypto/rates/sector/drawers | finite timeout, partial updates, missing methods, race/unmount, stale/fallback/null/zero และ dates ถูกต้อง |
| W09: regression/failure fixtures | backend/frontend/API/integration tests | ทดสอบเฉพาะจุดที่พิสูจน์ invariant/failure จริง ไม่เพิ่ม test ที่ทำซ้ำ implementation เฉย ๆ |
| W10: acceptance orchestration | strict runner/result writer และ CI | รวมผลเป็น artifact เดียว; มี coverage summary; NOT_RUN/BLOCKED ไม่ผ่าน; drift guard |
| W11: browser/model readiness | shadow API/web, auth, connected browser, model config | เปิดหน้า/ล็อกอินได้; model preflight พร้อม; planned screenshots/fixtures พร้อมก่อน G5 |
| W12: rehearsal | graph mock + pinned raw inputs + API/UI fixtures | MC cases ที่ไม่ต้อง live AI ซ้อมผ่านครบ; ประมาณ token/time จาก input จริงและกำหนด deadlines |

ลำดับ dependencies: W01 → W02/W03 → W04/W05 → W06/W07/W08 → W09/W10/W11 → W12 → G5 การอ่าน/แก้คนละ subsystem ทำขนานได้ แต่การ freeze input, commit, model invocation และตรวจ identity ต้องเป็นลำดับ

ต้องตรวจ `scripts/run_daily_macro_strategy.py` เพิ่ม: ข้อความ “Pipeline complete” หรือ exit 0 ไม่เพียงพอ ต้อง assert ว่า nodes ที่บังคับทำงานครบและมี commit receipt จริง; การตรวจ snapshot ต้องอ่าน directory layout จริงที่มี year/month ไม่ glob เฉพาะ top-level

## 8. Acceptance matrix: 72 กรณีบังคับ

ทุกกรณีต้องมี `case_id`, status, acceptance ID, start/end, input hashes, expected/actual, evidence paths และ defect reference เมื่อไม่ผ่าน ผลเริ่มต้นทั้งหมดคือ NOT_RUN

### 8.1 Environment และเครื่องมือตรวจ

| ID | วิธีตรวจ | เกณฑ์ผ่าน | หลักฐาน |
| --- | --- | --- | --- |
| MC-01 | inventory code/config/dependencies/working tree | ระบุ revision รวม untracked; path/config จริงตรง manifest | manifest + hashes |
| MC-02 | resolve shadow paths และตรวจ hashes ข้อมูลเดิมก่อน/หลัง | ไม่มี production data/report ถูกเปลี่ยนโดยการทดสอบ | isolation/diff report |
| MC-03 | ตรวจ auth/API/web/model/browser และ worker readiness | ใช้งานจริงได้ครบก่อนเริ่ม live AI; missing dependency เป็น BLOCKED | preflight receipts |
| MC-04 | ทำให้ audit พบ endpoint error/invalid payload | status และ exit code ไม่ใช่ PASS/0; บันทึก partial evidence | negative audit results |
| MC-05 | ทำให้ไม่พบ snapshot/coverage contract | comparison ไม่สร้าง empty-success; FAIL/BLOCKED พร้อมเหตุผล | negative binding tests |
| MC-06 | ให้ observable ID ซ้ำและ metadata ขัดกัน | ไม่ dedup เงียบ; reject หรือ alias ตาม contract ที่ตรวจได้ | duplicate fixture diff |
| MC-07 | เปลี่ยน snapshot/config/code หลัง freeze | drift guard ตรวจพบและปิดรอบ; checksum mismatch ไม่ถูกยอมรับ | mutation detection |
| MC-08 | ตรวจ case/result aggregator มี NOT_RUN/BLOCKED | ไม่สรุป all-pass และไม่คืน 0 หากกรณีบังคับยังไม่ครบ | aggregator tests |

### 8.2 Source และ ingest

| ID | วิธีตรวจ | เกณฑ์ผ่าน | หลักฐาน |
| --- | --- | --- | --- |
| MC-09 | fetch expected Yahoo tickers ทั้งหมด | expected coverage ตรง config; partial error มีชื่อ ticker/เหตุผล | raw history + coverage |
| MC-10 | เทียบ last/previous/observed date กับ bar ต้นทาง | มาจาก bar ถูกต้อง; ไม่มีการอ่าน percent change เป็น previous | reconciliation |
| MC-11 | ตรวจ FRED ทุก required series/history | series/frequency/unit/period/transform ตรง contract; gap/stopped series เห็นชัด | raw series + matrix |
| MC-12 | ตรวจ Thai GDP/CPI/core CPI/MPI กับเอกสารทางการ | raw/normalized/period/history ตรง; verified flag มีหลักฐานรองรับ | primary artifacts + hashes |
| MC-13 | ตรวจ Thai debt/yields/flow/PE/breadth | ทุกค่า/date/unit ตรงต้นทาง; stale debt ไม่ถูกแสดงเป็น current | raw comparison |
| MC-14 | ตรวจ Treasury/BIS/CBOE/COT/OFR | tenors/countries/auctions/vol/categories ครบตาม payload ไม่เลือกเพียงตัวอย่าง | full field reconciliation |
| MC-15 | ตรวจ crypto ทุก component และ history | supply/growth/BTC/gold/ETF มีวันที่/coverage ของตน; zero/null ถูกต้อง | component lineage |
| MC-16 | ตรวจ sector daily/weekly จาก raw adjusted bars | 11 sectors + benchmark ครบ; completed sessions และ snapshot digest ตรง | sector evidence |

### 8.3 เวลา หน่วย และการคำนวณ

| ID | วิธีตรวจ | เกณฑ์ผ่าน | หลักฐาน |
| --- | --- | --- | --- |
| MC-17 | ทดสอบ freshness boundaries ทุก frequency | cutoff/missing/future/holiday/quarter end ตรง policy ที่ตรึง | parameterized results |
| MC-18 | เทียบ raw → normalized ทุก unit family | currency/percent/bps/million/billion/contracts/index ถูกต้อง; finite | unit matrix |
| MC-19 | คำนวณ growth/returns/spreads/ratios ซ้ำจาก raw | ค่าตรงสูตร; prev/history ถูกต้อง; denominator 0 เป็น unavailable | independent calculation diff |
| MC-20 | ทดสอบ precision เช่น Fed midpoint/US-TH spread | ไม่ปัดก่อนคำนวณ; display rounding ตรง contract | raw/score/display diff |
| MC-21 | ให้ history ไม่ครบ/มี period ซ้ำ/revised | formula ไม่ใช้ history ผิด; version/revision และ Unknown reason ชัด | history fixtures |
| MC-22 | ให้ status กับ is_valid ขัดกัน/NaN/infinity | rejected/invalid ตาม schema; downstream eligibility สอดคล้อง | eligibility tests |
| MC-23 | ทดสอบ threshold ก่อน/ตรง/หลัง และ weight/missing dimension | คะแนน/state/confidence ถูกต้อง; missing ไม่สร้าง neutral score | scoring fixture diff |
| MC-24 | รัน `evaluate_macro_matrix` ด้วย input ตรึง | full QuantScore + derived/valuation/risk/sector ตรง expected contract | tool output + coverage |

### 8.4 AI handoff และการอ้างหลักฐาน

| ID | วิธีตรวจ | เกณฑ์ผ่าน | หลักฐาน |
| --- | --- | --- | --- |
| MC-25 | จับ actual tool result และ post-Quant payload | full canonical fields/metadata/IDs ไม่หายหรือถูก rewrite | full structural diff |
| MC-26 | จับ input จริงของ Economist | ได้ QuantScore/registry/context/sector ที่กำหนดครบ ไม่ใช่เพียง test stub | invocation input receipt |
| MC-27 | จับ input จริงของ Allocator | ได้ full registry และ eligibility ที่สอดคล้อง; condensed copies ไม่ขัดกัน | input/registry diff |
| MC-28 | ตรวจ input size/token budget ก่อนและระหว่าง invocation | ไม่มี truncation; schema/prompt versions ถูกต้อง; retries/usage บันทึกครบ | model receipts + usage |
| MC-29 | ตรวจ live output ของทุก agent กับ pinned evidence | numeric statements/unit/date/metric refs ตรง evidence; ไม่มี fabricated current values | claim reconciliation |
| MC-30 | ป้อน invalid/stale/optional gap ใน rehearsal | AI/score/confidence ระบุข้อจำกัด; ไม่อ้าง invalid เป็น current evidence | guarded outputs |
| MC-31 | ป้อน schema-invalid/unknown ref ใน output fixture | validation reject/repair แบบ bounded; ไม่ publish invalid report | validation receipts |
| MC-32 | รัน model จริงครบ graph หนึ่งรอบ | Quant/Economist/Allocator ที่บังคับทำงานจริง; task/run binding ครบ; ไม่ใช้ mock แทน | invocation chain + run IDs |

### 8.5 Publication และ identity

| ID | วิธีตรวจ | เกณฑ์ผ่าน | หลักฐาน |
| --- | --- | --- | --- |
| MC-33 | commit report จาก live run | terminal state success หลัง canonical report/evidence commit สำเร็จ | commit receipt |
| MC-34 | เทียบ report registry กับ input registry | IDs/value/unit/date/status/metadata/provenance/derivative refs ตรงครบ | full canonical diff |
| MC-35 | สร้าง 2 runs วันเดียวกันใน fixture | revision/hash แยก; report A ยังคง evidence A หลัง run B | same-day revision tests |
| MC-36 | latest ordering และ archive navigation | latest ตามเกณฑ์ที่กำหนด; archived IDs ไม่ถูกเขียนทับ | report index + IDs |
| MC-37 | retry task/commit receipt เดิม | idempotent; ไม่เกิด report/job/evidence ซ้ำหรือ side effect ซ้ำ | retry receipts |
| MC-38 | ให้ report/sector evidence digest ขัดกัน | เปิดข้อมูลปะปนไม่ได้; error ชัด ไม่มี fallback ข้าม report เงียบ ๆ | integrity failure results |
| MC-39 | crash ระหว่าง canonical/projection write | recover ได้ตามสัญญา; ไม่มี latest ชี้รายงานที่ยังไม่สมบูรณ์ | recovery logs + hashes |
| MC-40 | เปิด legacy/archive หลัง rollback ใน shadow | รายงานเดิมอ่านได้ หรือแสดง compatibility state ที่กำหนด; evidence ไม่เปลี่ยน | rollback check |

### 8.6 HTTP/API

| ID | วิธีตรวจ | เกณฑ์ผ่าน | หลักฐาน |
| --- | --- | --- | --- |
| MC-41 | session ที่ถูกต้อง/ไม่มี session/หมดอายุ | auth status และ UI behavior ตรง route contract; ไม่มี unauthorized data leak | HTTP/auth results |
| MC-42 | GET macro dashboard/report live และ archive | report/run/registry/assessment/summary ตรง canonical evidence | full API diff |
| MC-43 | GET indicator ทุก series และ ranges ที่รองรับ | period/order/unit/value/limit ตรง source; missing/invalid query ชัด | series reconciliation |
| MC-44 | GET/refresh sector daily/weekly pending/fail/ready | 200/202/503 และ schema ตรง state; wrong query 422 ตาม contract | state/status matrix |
| MC-45 | ตรวจ provider endpoints ทุก consumer ของ Macro | every field ตรง contract; empty HTTP 200 ไม่ผ่านเพราะ status อย่างเดียว | endpoint coverage matrix |
| MC-46 | ตรวจ null/zero/stale/partial/timeout/error payloads | schema และ freshness ไม่ขัดกัน; ไม่มี today/default/Neutral ปิดบัง gap | payload fixtures |
| MC-47 | ตรวจ OpenAPI/type drift และ frontend client | routes/types/optional methods/status unions ตรง backend | generated schema diff |
| MC-48 | สั่งงานผ่าน API/job และตรวจ completion | card/job IDs ถูกต้อง; complete ต่อเมื่อ report commit; failure/retry ตรงสถานะ | job/report correlation |

### 8.7 Browser: ข้อมูลปกติและรายงาน

| ID | วิธีตรวจ | เกณฑ์ผ่าน | หลักฐาน |
| --- | --- | --- | --- |
| MC-49 | เปิด `/macro?tab=ai` กับ live report ที่รับรอง | briefing/stance/confidence/date/limitations ตรง report; ไม่สลับ provider timestamp | screenshot + assertions |
| MC-50 | เปิด US tab และ detail/drawers | score/rates/curve/auctions/vol/COT/OFR/series ตรง API/evidence | field-by-field UI diff |
| MC-51 | เปิด TH tab และ detail/drawers | GDP/CPI/MPI/rates/fiscal/yields/flow/valuation/breadth ครบ; invalid assessment ไม่หลุดแสดง current | UI diff + freshness |
| MC-52 | เปิด cross-border tab | FX/capital flow/US-TH spread/crypto ตรงข้อมูลและหน่วย; zero ไม่กลายเป็น missing | UI assertions |
| MC-53 | เปิด sector daily/weekly/panels/refs | snapshot identity/timeframe/ranks/returns/AI claims ตรง pinned evidence | screenshots + ID diff |
| MC-54 | เปิด citations/indicator ranges จากรายงาน | refs ไป evidence/series ถูก ID/period/report; ไม่มี unknown ref | navigation results |
| MC-55 | เลือก archive แล้วกลับ latest | AI analysis/evidence/archive date เปลี่ยนตรง report; live widgets ระบุเวลาแยก | report selection results |
| MC-56 | กดอัปเดตบทวิเคราะห์และติดตาม job | กดซ้ำไม่สร้างงานซ้ำ; pending/failure/success ชัด; โหลด report ใหม่หลัง commit | UI/job/receipt chain |

### 8.8 Browser: ความทนทานและการใช้งาน

| ID | วิธีตรวจ | เกณฑ์ผ่าน | หลักฐาน |
| --- | --- | --- | --- |
| MC-57 | ไม่มี report/ไม่มี provider data/empty data | empty state ถูกต้อง; default tab ไม่ค้าง; ไม่มีค่า fallback สมมติ | empty-state screenshots |
| MC-58 | provider เดียว timeout/500/วิธี client ไม่มี | finite deadline; ข้อมูลอื่นแสดงได้; failure ชื่อชัด; ไม่ stamp ว่าทุกแหล่งอัปเดตแล้ว | partial-failure assertions |
| MC-59 | response เก่ามาช้ากว่ารอบใหม่/สลับ report เร็ว/unmount | response เก่าไม่ทับข้อมูลใหม่; ไม่มี state leak หรือ stuck loading | race test + browser trace |
| MC-60 | stale/partial/fallback จากรายงานเก่า | badge/source/date/reason มองเห็น; ไม่ปะปนค่าปัจจุบันกับ fallback เงียบ ๆ | provenance screenshots |
| MC-61 | ใส่ 0/null/negative/fractional/large values | dash/zero/sign/decimals/unit ถูกต้องทุก widget ที่ใช้ field เหล่านี้ | display matrix |
| MC-62 | refresh/login expiry/network reconnect | old data ระบุเวลา; error เคลียร์หลังสำเร็จ; retry ไม่สร้างงานซ้ำ | recovery assertions |
| MC-63 | ตรวจ desktop 1440×900/mobile 390×844 และ keyboard | tabs/cards/drawers/tables ใช้งานได้; ไม่มี clipping บังค่า/หน่วย/สถานะ | viewport screenshots |
| MC-64 | reload/deep link/back-forward และ runtime errors | selected report/tab ถูกต้อง; ไม่มี unhandled error; console error ที่เกี่ยวข้องเป็นศูนย์ | navigation/error log |

### 8.9 Regression, recovery และผลตรวจรวม

| ID | วิธีตรวจ | เกณฑ์ผ่าน | หลักฐาน |
| --- | --- | --- | --- |
| MC-65 | backend relevant suite + API/architecture regressions | tests ผ่าน; skipped live ถูกจัดเป็นหลักฐานคนละประเภทและตรวจจริงใน MC-32 | command/exit/test logs |
| MC-66 | frontend full suite/lint/typecheck/types/build | ผ่านทั้งหมด; ไม่มีเลือกตัด suite ที่ fail ออกโดยไม่แก้ | command logs |
| MC-67 | provider 429/outage/timeout/malformed/partial | retry/deadline bounded; unavailable/degraded ไม่สร้างคะแนนหรือค่าปลอม | injection matrix |
| MC-68 | model timeout/invalid output/queue retry | failure ไม่ publish success; bounded retry มี receipts; recovery idempotent | model/job failures |
| MC-69 | store write failure/evidence missing/corrupt/cache stale | fail closed สำหรับ integrity; ไม่เผย latest ที่ผิด; recover ตามสัญญา | persistence failures |
| MC-70 | ตรวจ protected files/notification/processes หลังจบ | data เดิม hash ไม่เปลี่ยน; ไม่มี notification ไม่ตั้งใจ; shadow processes เก็บกวาดตามแผน | before/after receipts |
| MC-71 | ตรวจ coverage และ drift รอบสุดท้าย | MC-01–72 มีหลักฐาน; manifest/input/report/UI revision ตรง; ไม่มี unexpected gap | final reconciliation |
| MC-72 | สร้าง completion report และเครื่องมืออ่านผล | สรุป scope A/B/C ถูกต้อง; PASS เฉพาะเมื่อทุกกรณีบังคับผ่าน; exit code สอดคล้อง | result.json + completion.md |

## 9. สูตรและ reconciliation ที่ต้องทำแบบอิสระ

ตัวตรวจต้องคำนวณจาก raw input โดยใช้วิธีที่ไม่เรียก production helper เดิมมาทำ expected result ทั้งหมด มิฉะนั้น implementation ผิดกับ test อาจผิดเหมือนกัน

| รายการ | วิธีตรวจ |
| --- | --- |
| Price change/return | เทียบ bar IDs และ `(current / previous - 1)` ตาม percent/unit contract; missing previous ไม่ใช้ 0 |
| YoY/QoQ/rolling history | เทียบ exact periods ตาม series definition; ไม่เทียบ monthly row ข้างกันแล้วเรียก YoY |
| Rate spread/yield curve | คง raw precision และ percent units; convert bps หนเดียว; เทียบ tenor/date |
| BTC/gold ratio | เทียบ underlying symbols/currency/units/date alignment; denominator invalid → unavailable |
| Stablecoin 7D/30D growth | เทียบ supply history ที่ window ตรง; missing constituent/history มี coverage reason |
| Sector returns/relative strength/rank | คำนวณจาก adjusted bar grid เดียวกัน; benchmark/date/window/rank ties ตาม contract |
| Dimension score/state/confidence | ใช้ fixture ที่ boundary และ expected contribution ต่อ input; ตรวจการกัน invalid/stale |
| Display | คำนวณ expected text จาก contract rounding/unit; เทียบค่าที่ UI render กับ API revision ที่บันทึก |

ค่า raw/identity/date/status/string ต้องตรงตาม contract; floating point ใช้ tolerance ที่กำหนดต่อสูตรและ magnitude พร้อมเหตุผล ห้ามตั้ง tolerance กว้างเท่าความต่างที่พบเพื่อให้ผ่าน Display tolerance ต้องไม่เกินการปัดที่แสดงจริง

## 10. Failure injection และ recovery

ซ้อมทั้งหมดก่อน live AI ด้วย fixtures/proxy/test adapter ที่แยกจาก production จากนั้นตรวจตัวอย่าง browser จริงใน shadow ที่เตรียมไว้ ห้ามทำให้ provider หรือข้อมูลจริงของผู้ใช้เสียเพื่อทดสอบ

| Failure family | Variants ที่ต้องครอบคลุม | Expected behavior |
| --- | --- | --- |
| Provider | timeout, 429, 500, malformed, HTTP 200 empty, one field missing, stopped series | มี deadline; bounded retry; ชื่อ provider/field/reason ชัด; partial data ใช้ตาม policy |
| Dates/history | missing/future/stale/cutoff/quarter label/holiday/duplicate/revised | ไม่ใช้ fetch date แทน observation; ไม่เกิด look-ahead; history ถูกต้องหรือ Unknown |
| Values/schema | null, 0, negative, fractional, NaN, infinity, wrong unit, conflicting status | preserve zero/precision; reject invalid; score/AI/UI สอดคล้อง |
| AI | context overflow, timeout, schema invalid, unknown ref, fabricated number, stale citation | preflight/validation จับได้; ไม่ publish success; retries บันทึกและ bounded |
| Queue | double click, duplicate message, worker restart, out-of-order completion | job/run correlation ชัด; idempotent; latest ordering ไม่ผิด |
| Persistence | disk/write error, crash stages, missing digest, corrupt evidence, projection failure | canonical integrity คงอยู่; latest ไม่ชี้ incomplete; recover ตรวจ hash |
| Frontend | API method absent, one slow request, stale response race, unmount, expired login | partial render/timeout/error/retry ถูกต้อง; no state contamination |
| Archive/rollback | schema legacy, same-day revisions, stale latest cache, fallback refs | report/evidence เดิม immutable; compatibility state ถูกต้อง |

กำหนด timeout/retry/overall deadlines ใน G0 จาก config และ rehearsal จริง: ทุก operation ต้องมีขอบเขตเวลา, ทุก retry ต้องมีจำนวนสูงสุดและ backoff, graph retry ต้องไม่สร้าง publication/notification ซ้ำ แยก provider timeout, request timeout, model timeout และ job timeout ไม่ใช้ timeout ของ UI มาอ้างว่า backend หยุดแล้ว

## 11. คำสั่งตรวจที่มีอยู่จริง

รันแต่ละคำสั่งแยกกัน เก็บ cwd, command, environment profile ที่ตัด secrets, exit code และ log ใช้ dependency ที่ติดตั้ง/lock ไว้แล้ว ไม่ upgrade packages ระหว่างรอบรับรอง

### 11.1 Backend focused regression — cwd = repo root

```powershell
.\.venv\Scripts\python.exe -m pytest tests/tools/macro tests/unit/macro tests/unit/market/test_macro_publication_calendar_freshness.py tests/unit/application/test_macro_card_service.py tests/unit/terminal_v2/test_crypto_liquidity_adapter.py tests/unit/terminal_v2/test_api_endpoints.py tests/api/test_sector_rotation_http.py tests/unit/api/test_sector_rotation_routes.py tests/integration/test_macro.py -q -o addopts=
```

จากนั้นตรวจ regression ที่กว้างขึ้นตามส่วนที่เปลี่ยน โดยแยก live tests ก่อนรัน:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/api tests/architecture -q -o addopts=
```

Inventory `tests/integration/sector_rotation/` และ `tests/unit/scripts/` แล้วเพิ่มรายการ deterministic tests ที่เกี่ยวข้องลง command manifest ก่อน freeze อย่ารันเหมารวมโดยถือว่าทุก integration เป็น offline: มี test ที่เรียก live yfinance และชื่อ “AI shadow” บางกรณียังใช้ synthetic inputs จึงต้องจำแนกจากพฤติกรรมจริง

`tests/integration/test_macro.py` มี live LLM case ที่ skip อยู่ ผล suite ผ่านไม่แทน MC-32 ให้ใช้ shadow graph invocation จริงพร้อม input/output receipts เพื่อรับรอง โดยไม่เปิด skip แบบเหมารวมจนเรียก AI หลายครั้ง

### 11.2 Frontend — cwd = `web`

```powershell
npm run lint
npm run typecheck
npm run check:types
npm run test
npm run build
```

`check:types` export OpenAPI ลงไฟล์: ตรวจ diff ว่าสอดคล้องกับ schema change ที่ตั้งใจไว้และ freeze หลังตรวจเรียบร้อย ต้องรวม drawers/panels/content references และ tests ทั้งหมด มิใช่เฉพาะชุด 74 tests เดิม

### 11.3 Live source/API diagnostics — cwd = repo root

```powershell
.\.venv\Scripts\python.exe scripts/verify_macro_data_pipeline.py
.\.venv\Scripts\python.exe scripts/audit_macro_page_data.py
```

สองคำสั่งนี้มีอยู่จริง แต่ปัจจุบันยังไม่ใช่ strict acceptance runner ต้องผ่าน W02/W03/W05 ก่อนนำผลไปรับรอง Copy outputs เข้า acceptance bundle ทันทีเพื่อไม่ให้รอบถัดไปเขียนทับ หลักฐานต้องบอกว่าคำสั่งไหนเรียก source จริงและคำสั่งไหนเรียก AI

### 11.4 Live graph และ browser

ใช้ daily runner หรือ dispatch ผ่าน shadow API ตาม entry point ที่ตรวจพร้อมแล้ว ผูก `macro_run_id`/task/report identity และ capture actual invocations ต้องตรวจ configuration และ resolved paths ของ `scripts/run_daily_macro_strategy.py` ก่อนใช้ ห้ามถือว่า script มี flags `--shadow`/`--acceptance-id` หากยังไม่ได้ implement

ใช้ browser tooling ที่เชื่อมต่อและได้รับอนุญาตจริง ตรวจทั้ง screenshots และ rendered values จาก API/report revision ที่บันทึก หาก browser ยังไม่พร้อมให้หยุดก่อน G5 เป็น BLOCKED ไม่ใช้ jsdom แทน browser acceptance

strict orchestration/result writer ใน W10 เป็นงานที่ต้องเพิ่ม แผนนี้ไม่ได้อ้างว่ามีคำสั่ง acceptance ใหม่พร้อมใช้งานแล้ว

## 12. ขั้นตอนตรวจรับรอบเดียว

1. ตรวจทุกงาน W01–W12 ว่ามีผลส่งมอบจริงและ rehearsal ผ่าน ปิด defect ที่มีผลต่อเกณฑ์บังคับทั้งหมด
2. สร้าง acceptance ID; freeze code/config/source contract/calendar/formulas/dependencies และ hashes ของข้อมูลเดิม
3. รัน G0/G1 negative preflight ตรวจเครื่องมือตรวจเอง โดยไม่เริ่ม AI
4. ดึง source จริงชุดเดียว เก็บ raw artifacts และสร้าง immutable input/snapshots; เปรียบเทียบ primary sources และ coverage ตาม G2
5. รัน full evaluation/score จากชุดนั้น ตรวจสูตร/eligibility/lineage ตาม G3; แยก current provider snapshot กับ report input snapshot อย่างชัดเจน
6. รัน regression/rehearsal ตาม G4 ไม่มี live process เขียนข้อมูลระหว่าง pytest; ตรวจ browser/model readiness ซ้ำแบบเบาเพื่อไม่ให้เสีย live run
7. เรียก graph/model จริงหนึ่งรอบตาม G5 โดยใช้ frozen source inputs; stage ที่จำเป็นต้องดึงเพิ่มต้องบันทึกเป็น input revision ใน manifest ก่อน handoff และตรวจ coverage/hash ของข้อมูลนั้น
8. เก็บ tool/agent/prompt/output/usage/validation/commit receipts ทุก stage ตรวจ canonical diff และ citations ก่อนยอมให้ผลเป็น committed success
9. เรียก HTTP บนรายงานที่ commit จริง ตรวจ G6 จาก full payloads ไม่ใช่ HTTP status อย่างเดียว
10. เปิด browser ตรวจ G7 normal/archive/detail/mobile กับรายงานเดียวกัน สำหรับ live market widgets ตรึง API responses ที่ capture ไว้ให้มี identity ตรง หรือ capture revision ใหม่แล้วเทียบ ณ เวลานั้น ห้ามบังคับค่าตลาดที่เปลี่ยนแล้วให้เท่ารายงานเก่า
11. ตรวจ G8 failure cases ด้วย shadow fixtures ที่เตรียมไว้ ไม่เรียก live AI ใหม่สำหรับ failure ที่พิสูจน์ด้วย injection ได้ และตรวจ failure ของ model/API boundaries ตาม receipts
12. ตรวจ hashes/drift/protected data/artifact completeness อีกรอบ สร้าง `result.json` และ `completion.md`; หากมี FAIL/BLOCKED/NOT_RUN ในกรณีบังคับสรุปยังไม่ผ่านระดับ B

ถ้ามี source update ระหว่างรอบ ไม่ refresh frozen report inputs เงียบ ๆ ให้เก็บเป็น revision ใหม่หรือ current-widget evidence แยก หากหมด freshness window ก่อนจบรอบ ต้องให้ตัวตรวจระบุว่ารับรอง ณ เวลาใด และ refresh/re-run เฉพาะขอบเขตที่เปลี่ยนในรอบใหม่

## 13. เกณฑ์จบงานและรายงานที่ส่งมอบ

Definition of Done สำหรับระดับ B:

- [ ] MC-01–MC-72 PASS ทั้งหมดและมีหลักฐานของ acceptance ID เดียวกัน
- [ ] ทุก UI field/AI dependency มี lineage mapping; ไม่มี unexpected missing/duplicate/invalid field
- [ ] required data พร้อม หรือ degraded behavior ตรง policy ที่กำหนดไว้ก่อนรัน; ไม่มีค่าปลอมเพื่อเติมช่องว่าง
- [ ] actual Quant/Economist/Allocator invocation และ committed report ถูกตรวจ; ไม่มี truncation หรือ invalid current citation
- [ ] full backend ที่เกี่ยวข้อง/frontend/schema/build ผ่าน และ live/model/browser evidence แยกจาก mock ชัดเจน
- [ ] latest/archive/evidence identity ตรง; ข้อมูลเดิมไม่ถูกเปลี่ยน; failure/retry/rollback ไม่ทำให้ false success
- [ ] ไม่มี code/config/input drift ระหว่างผลที่นำมารับรอง และไม่มี defect ค้างที่ละเมิด acceptance criterion
- [ ] completion report บอกผล A/B/C แยกกัน พร้อมข้อจำกัดจริง ไม่อ้างว่า EOD หลาย session ผ่านจากวันเดียว

Completion report ต้องมี: acceptance ID/revision, scope, gates/case counts, source coverage/freshness, AI run/task/report IDs, invocation/usage summary, API/UI reconciliation, failure recovery results, artifacts, defect list และสิ่งที่ยังไม่ได้รับรอง

ตัวอย่างข้อความผลลัพธ์ที่ใช้ได้: “Macro ระดับ B ผ่าน 72/72 กรณีสำหรับ revision … และ report …; การรับรอง EOD ต่อเนื่องระดับ C ยังรอ session ตามแผน Sector” ใช้ข้อความนี้เฉพาะเมื่อหลักฐานจริงครบตามนั้น

## 14. ความสัมพันธ์กับแผนเดิมและ CI

- [แผน Sector Rotation](sector-rotation-verification-plan.md): เก็บ AC/RC และเกณฑ์ completed US sessions/weekly close/rollback เดิมไว้เป็นระดับ C; สำหรับ Macro ระดับ B ต้องตรวจ sector data, AI binding และ UI ในรอบนี้ด้วย
- [แผนแก้ช่องว่างข้อมูลไทย](thailand-macro-data-gap-remediation-plan.md): นำ source definitions/TH criteria มา map กับ MC-12/13/17/21/23/30/51/60; checkbox เดิมไม่ใช่หลักฐานรับรอง revision ปัจจุบัน ต้องเปิด evidence ตรวจซ้ำ
- [Audit เดิม](../docs/macro-audit-2026-10-04.md): ใช้เป็น baseline และ defect history ไม่ใช้แทน live AI/browser ของรอบใหม่
- CI ต้องเพิ่ม Macro-focused invariants/lineage/strict audit tests ที่ยังไม่ได้อยู่ใน jobs ปัจจุบัน และ publish sanitized test artifacts เมื่อ fail
- CI deterministic แยกจาก live provider/model/browser acceptance เพื่อไม่เรียก model มีค่าใช้จ่ายทุก push; แต่การแยกนี้ไม่ทำให้ live cases ถูกนับ PASS โดยอัตโนมัติ

## 15. Checklist สำหรับเริ่มลงมือ

- [ ] ตั้งผู้รับผิดชอบ Data/Backend, AI/Publication, Frontend/UI และผู้รวบรวมผล จะเป็นผู้ดำเนินการคนเดียวก็ได้
- [ ] ทำ W01–W03 ก่อน เพื่อให้ทุกผลตรวจหลังจากนั้นเชื่อถือได้
- [ ] ปิด Thai source/history, MOF/country series policy และ crypto component dates ใน W04
- [ ] ตรวจ full evaluation/score/actual handoff/publication และ frontend failures ใน W05–W08
- [ ] ทำ full regression, negative tests และ acceptance aggregator ใน W09–W10
- [ ] เตรียม browser/model/shadow workers/notification isolation ใน W11
- [ ] ซ้อม W12 และตรวจ input size/deadlines; freeze แล้วจึงเปิดรอบตรวจรับ

เอกสารนี้จัดทำแผนและเกณฑ์ตรวจเท่านั้น การสร้างไฟล์นี้ไม่ได้รัน regression ใหม่ ไม่ได้เปิด browser และไม่ได้เรียก model หรือสร้างรายงาน AI ใหม่
