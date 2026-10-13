# แผนแก้ไข Portfolio Principles แบบละเอียด

วันที่: 10 ตุลาคม 2026  
สถานะ: แผนงานพร้อมนำไปลงมือ ยังไม่ได้ดำเนินการแก้โค้ดผลิตภัณฑ์  
อ้างอิง: [ผลรีวิว](REVIEW.md), [หลักฐาน backend](backend_evidence.json), เคส React ใน `web/scratch/principles-review/`

## 1. ผลลัพธ์ที่ต้องได้

ผู้ใช้ทำ Essence → Financial Context → Axis → Buckets → Preview → Apply ได้ต่อเนื่อง โดยข้อมูลที่ยืนยันแต่ละขั้นถูกส่งครบ มีที่มา ตรวจสอบความสอดคล้องได้ และเห็นการใช้ buckets ในหน้าดูแลพอร์ตจริงหลัง reload

งานถือว่าเสร็จเมื่อ:

1. ทุก mutation เกิดกับพอร์ตที่ผู้ใช้เลือก และผล request เก่าไม่ทับพอร์ตใหม่
2. ขั้นถัดไปใช้ฉบับที่ยืนยันจริง พร้อม revision/hash; การเริ่มสัมภาษณ์หรือแก้ draft ใหม่ไม่เปลี่ยนแผนใช้งานเดิม
3. ข้อมูลไม่ทราบ/ไม่มี/ศูนย์มีความหมายต่างกัน และข้อมูลที่ยังไม่พอไม่ถูกแทนด้วยตัวเลขตายตัว
4. คำถามและข้อสรุปมีหลักฐานจริงครบตามเรื่องที่จำเป็นต่อการวางแผน
5. Axis, matrix, buckets, เงินเข้า และ remapping ตรวจด้วยกฎเดียวกันทั้ง UI/API/Apply
6. แต่ละ bucket แสดงเป้าเงิน วันใช้เงิน เงินที่มี ส่วนที่ขาด แผนเติมเงิน สินทรัพย์ที่ใช้ได้ และเงื่อนไขใช้เงินได้
7. Apply บันทึก targets, assignments, policy, provenance และ durable receipt โดย retry ไม่เกิดการบันทึกซ้ำ
8. ระบบแสดงสถานะบันทึก/เผยแพร่/นำไปใช้ตามจริง และกู้ต่อได้เมื่อ storage หรือ projection บางส่วนล้มเหลว
9. ข้อมูลเก่าเปิดอ่านได้ และข้อมูลที่กู้หลักฐานการยืนยันไม่ได้ถูกระบุให้ทบทวน
10. ผ่าน integration tests ที่ใช้ production adapters กับ temporary storage และตรวจ flow ผ่าน browser จริง

## 2. ขอบเขตและแนวทางออกแบบ

- ใช้ React/FastAPI/domain services/SQLite/PortfolioTransactionStore ที่มีอยู่ ยึดโครงสร้าง ports และ adapters เดิม
- Essence เป็นข้อมูลระดับ workspace; context/axis/bucket plan เป็นข้อมูลระดับ portfolio
- แผนยังเป็นเครื่องมือจัดหมวดและช่วยตัดสินใจ การแสดง DCA/withdrawal เป็นจำนวนเงินและรายการปฏิบัติที่ผู้ใช้ตรวจได้
- ใช้ THB เป็นฐานคำนวณตามพอร์ตปัจจุบัน และเก็บ currency/FX/as-of ให้ชัดเมื่อต้นทางเป็นเงินสกุลอื่น
- การแก้ครั้งนี้ครอบคลุมหน้า Overview/Holdings/Goals เท่าที่จำเป็นต่อการใช้ policy หลัง Apply
- ทำงานบน working tree ปัจจุบันซึ่งมีงานเดิมหลายไฟล์ ตรวจและรักษาการเปลี่ยนแปลงเดิมในแต่ละชุดงาน
- แยก backend correctness ออกจากคุณภาพข้อความ AI: ใช้ deterministic tests ตรวจ contract/กฎ และใช้ข้อมูลสังเคราะห์ตรวจคุณภาพ output

### 2.1 แหล่งข้อมูลหลัก

| ข้อมูล | แหล่งหลักที่เลือก | หลักการใช้งาน |
| --- | --- | --- |
| คำถาม/คำตอบ/summary draft | investor runtime repositories | แก้ไขได้ตาม revision และ branch |
| Essence/Axis ฉบับยืนยัน | immutable confirmed snapshots ใน investor runtime | เก็บทุกฉบับ; current pointer ชี้ฉบับที่ยืนยันล่าสุด |
| Knowledge note | Vault ผ่าน typed write broker | เป็น publication ของ snapshot; สถานะเผยแพร่แยกจากการยืนยัน |
| Financial context | context snapshot ID ที่ระบุชัด | อ่านตาม ref ได้; เก็บข้อมูลต้นทาง/เวลาที่รายงาน |
| Goals | goal_id ที่คงเดิมข้ามการแก้ชื่อ | มี funding scope ของพอร์ตแยกจากภาพรวมผู้ลงทุน |
| Holdings/NAV/assignments | PortfolioTransactionStore | เป็นสถานะปฏิบัติจริง ใช้ checkpoint ที่อ่านพร้อมกัน |
| Applied bucket policy | policy snapshot ใน PortfolioState | อ่านได้จาก operational portfolio โดยไม่ต้องอาศัย latest draft |
| Apply receipt | receipt ใน transaction เดียวกับสถานะพอร์ต | ใช้กู้ investor runtime และตอบ retry อย่างถูกต้อง |

ห้ามแสดง `vault:` ref ว่าเป็น artifact จริงก่อน broker คืน note_id/revision_id/content_hash/artifact_set_hash จริง ให้แยก `confirmation_ref` กับ `published_artifact_ref` ซึ่งอาจยังไม่มี

### 2.2 สถานะที่ต้องแยกกัน

| เรื่อง | สถานะที่ต้องรองรับ |
| --- | --- |
| สัมภาษณ์/สรุป | interviewing, summary_pending, summarizing, summary_failed, review, confirmed |
| ฉบับนโยบาย | draft, confirmed; มี stale reasons แยกจากสถานะการยืนยัน |
| Publication | pending, published, failed; retry ได้ |
| Bucket plan | draft, applying, applied; validation issues และ stale reasons แยกกัน |
| Apply outcome | canonical committed/duplicate; projection ready/pending; runtime reconciliation pending/complete |

`confirmed` ไม่เท่ากับ `published`, มี confirmed เก่าไม่แปลว่า draft ใหม่ถูกยืนยัน และ validation ผ่านไม่แปลว่า portfolio checkpoint ยังตรง

## 3. แบ่งงานและลำดับ dependency

| รหัส | งาน | ความสำคัญ | ทำหลัง |
| --- | --- | --- | --- |
| WP00 | เปลี่ยนหลักฐานบั๊กเป็น regression tests | P1 | เริ่มได้ทันที |
| WP01 | แยก state ตามพอร์ตและแก้ failure propagation | P1 | WP00 |
| WP02 | กำหนด DTO/validation issue/ref/hash contracts | P1 | WP00 |
| WP03 | เก็บ confirmed snapshots และ lineage จริง | P1 | WP02 |
| WP04 | แก้ publication adapter และสถานะ broker | P1 | WP02, WP03 |
| WP05 | แก้ adaptive interview, evidence, clarification, summary retry | P1 | WP02, WP03 |
| WP06 | Financial context, readiness, goals และ funding scope | P1 | WP02 |
| WP07 | Axis validation และการใช้ฉบับที่ยืนยัน | P1 | WP03, WP05, WP06 |
| WP08 | แก้ holdings snapshot และ input สำหรับ bucket generator | P1 | WP03, WP06, WP07 |
| WP09 | โมเดล practical buckets และการคำนวณ funding | P1 | WP06, WP07, WP08 |
| WP10 | Shared validator, allocation matrix และ assignments | P1 | WP07, WP09 |
| WP11 | Preview ที่ผูก revision/hash/checkpoint และ rebase | P1 | WP01, WP03, WP10 |
| WP12 | Persist policy, durable apply receipt และ recovery | P1 | WP03, WP09, WP10, WP11 |
| WP13 | หน้าใช้งานจริง, presets, draft/applied/stale UX | P1/P2 | WP01, WP05–WP12 |
| WP14 | ย้ายข้อมูลเก่าและตรวจ compatibility | P1 | ออกแบบเริ่ม WP02; ทำเสร็จหลัง WP03, WP06, WP09, WP12 |
| WP15 | Integration/browser/AI quality verification | P1 | เพิ่ม tests ตลอดงาน; gate สุดท้ายหลัง WP01–WP14 |

แต่ละ WP ต้องส่ง code + tests + migration ที่เกี่ยวข้อง + ผลตรวจรับในชุดเดียวกัน ไม่สะสมการทดสอบไว้ตอนจบ

## 4. รายละเอียดแต่ละงาน

### WP00 — สร้างฐาน regression ที่ตรวจพฤติกรรมถูกต้อง

**ไฟล์:** `tests/unit/investor_essence/`, `tests/api/test_investor_essence.py`, `tests/unit/db/test_investor_essence_repository.py`, tests ของ components/hooks ใน `web/src/`

**งานย่อย:**

1. ย้ายเคสจำลองจากรายงานมาเป็น tests ใน normal suite โดยกลับ assertion เป็นพฤติกรรมที่ถูกต้อง
2. สร้าง fixture พอร์ต A/B, หลาย session, หลาย draft และหลาย confirmed revisions ให้แยกกันจริง
3. สร้าง controllable promises สำหรับทดสอบ response มาถึงผิดลำดับใน React
4. เพิ่ม spy generator ที่เก็บ input เพื่อตรวจข้อความ choice, claims, context และ holdings ที่ส่งเข้า AI
5. ตั้ง temporary SQLite, transaction store และ Vault สำหรับ tests ของ production adapters
6. บันทึกว่า test ไหนคาดว่าล้มเหลวก่อนแก้ และผูกกับ WP เจ้าของงาน; ชุดงานที่เสร็จต้องเปลี่ยน tests ของตนเป็นผ่าน

**ตรวจรับ:** มี test ที่จับ wrong-portfolio mutation, claims ว่าง, context หาย, holdings=0, invalid matrix/DCA, orphan assignments, broker contract และ apply recovery ได้ ไม่ใช้ passing characterization tests เป็นหลักฐานว่าฟีเจอร์ถูกต้อง

### WP01 — ป้องกันการแก้ผิดพอร์ตและหยุด flow เมื่อขั้นก่อนหน้าล้มเหลว

**ไฟล์:** `web/src/hooks/useInvestorPrinciples.ts`, `web/src/pages/Portfolio.tsx`, `InvestorPrinciplesTab.tsx`, `FinancialContextForm.tsx`, `InvestmentAxisEditor.tsx`, `EssenceBucketBuilder.tsx`

**งานย่อย:**

1. แยก workspace state ออกจาก portfolio state; reset context/axis/plan/preview/receipt/substep เมื่อ portfolioId เปลี่ยน
2. ใช้ request generation หรือ AbortController ให้ผล request เก่าถูกทิ้งเมื่อพอร์ตเปลี่ยน/unmount
3. เพิ่ม guard ก่อน update/confirm/preview/apply ว่า draft.portfolio_id เท่ากับพอร์ตที่เลือก
4. ให้ mutation คืน DTO ที่บันทึกสำเร็จและ throw typed error เมื่อไม่สำเร็จ; hook ตั้ง error ให้ UI แล้วส่ง error ต่อ
5. `save → confirm` ใช้ draft_id/revision จากผล save โดยตรง ไม่อ่าน closure เก่าหรือ fallback สร้าง draft โดยเงียบ
6. `save → preview → review` เปลี่ยนหน้าหลังทั้งสองงานสำเร็จ; เก็บ edits ไว้ถ้า save/preview ล้มเหลว
7. แยก loading ของแต่ละ operation ป้องกัน double-click และคำตอบเก่าทับ action ใหม่
8. clear preview/receipt เมื่อแก้หรือสร้างแผนใหม่; restore applied state จาก server เมื่อ reload

**ตรวจรับ:** สลับ A→B ระหว่างโหลด/บันทึกไม่ส่ง mutation ผิดพอร์ต; save failure ไม่เรียกขั้นถัดไป; failure ไม่ล้างข้อมูลที่ผู้ใช้พิมพ์; receipt ของแผนเก่าไม่แสดงบนแผนใหม่

### WP02 — ทำ contracts ให้ชัดและไม่เกิด schema drift

**ไฟล์:** `core/investor_essence/models.py`, `application/investor_essence/dto.py`, `ports.py`, `api/schemas/investor_essence.py`, router, `web/src/api/client.ts`, `types.ts`, `types.generated.ts`

**งานย่อย:**

1. กำหนด Typed DTO สำหรับ ConfirmedEssence, ConfirmedAxis, FinancialFact, GoalFundingLink, HoldingPlanningSnapshot, BucketPolicy, Preview และ ApplyRequest
2. ใช้ ValidationIssue เดียว: `code`, `severity`, `field_path`, `message`, `details`; map ข้อความใน UI จาก fields เดียวกับ API
3. ใช้ Decimal ใน domain และ decimal strings ใน JSON; normalize 2 ตำแหน่งสำหรับเงินและ % โดยแสดงฐาน/unit ชัด
4. ใช้ tolerance 0.01 จุดเปอร์เซ็นต์ทั้ง server/client; ฝั่ง UI ใช้หน่วยร้อยส่วนหรือ decimal utility เพื่อลด float discrepancy
5. กำหนด SourceRef แบบแยกชนิด confirmed snapshot, context snapshot, goal revision และ published Vault artifact
6. กำหนด hash จาก semantic payload ที่ canonicalize แล้ว: allocation, rules, mapping, remapping และ source refs ต้องรวม; metadata เวลา request ไม่ทำให้ semantic hash เปลี่ยนโดยไม่มีเหตุ
7. เพิ่ม schema_version สำหรับ payload ใหม่ และ compatibility serializer/parser สำหรับ legacy fields; alias ที่ส่งมาขัดกันคืน 422
8. สร้าง frontend type aliases จาก OpenAPI สำหรับส่วนนี้ ลด handwritten `any` และ unsafe casts
9. เพิ่ม expected revision/source refs ใน confirm/apply และ structured 409/422 ที่บอกเหตุและสิ่งที่ต้องทบทวนได้

**ตรวจรับ:** generated types ตรง API; readiness message แสดงถูก; Decimal ไม่มี NaN/Infinity; contract round-trip รักษา fields ครบ; client/server ตัดสินเปอร์เซ็นต์เหมือนกัน

### WP03 — ใช้ immutable confirmed snapshots เป็นฐานทุกขั้น

**ไฟล์:** confirmation service, investment axis service, bucket planning service, domain claims/axis, `api/db/connection.py`, `api/db/repositories/investor_essence_repository.py`, fakes

**งานย่อย:**

1. เพิ่ม `investor_confirmed_snapshots` เก็บทุก confirmation ตาม ID พร้อม kind/scope/source_id/source_revision/content_hash/snapshot_json/confirmed_at
2. snapshot Essence เก็บ statement, final accepted claims, text revisions, source kinds, fit ratings, evidence, unresolved topics และ coverage
3. เก็บ snapshot + receipt + CAS current pointer ใน runtime transaction เดียว; accepted IDs ของ batch เป็นรายการจริงที่ downstream อ่าน
4. snapshot Axis เก็บทั้ง 8 หัวข้อ, allocation_basis/rows, numeric origins/assumptions, essence/context/goal refs และ coverage/readiness decisions
5. ให้ create Axis อ่าน Essence ตาม confirmation_ref; create Buckets อ่าน Axis ตาม confirmation_ref พร้อม Essence/Context refs ที่ Axis ใช้จริง
6. เพิ่ม get snapshot by ID/ref และอ่าน context by snapshot ID เพื่อไม่พึ่ง latest/current เมื่อต้องอ่านฉบับเก่า
7. content_hash ของ Axis รวม allocation และ policy ทั้งชุด; เปลี่ยน allocation/rules ต้องได้ hash ใหม่
8. เก็บ source refs แบบจริง และ stale reasons เมื่อ current sources เปลี่ยน โดยไม่เขียนทับ snapshot เดิม
9. ยกเลิก read-time recovery ที่หยิบ latest draft มาสวมเป็น confirmed snapshot; ส่ง legacy_unverified หากกู้จากหลักฐาน exact version ไม่ได้

**ตรวจรับ:** confirm 2 claims แล้ว Axis ได้ 2 claims ตรงข้อความสุดท้าย; session ใหม่ไม่เปลี่ยน downstream; draft ใหม่ไม่ทับ confirmed; อ่านฉบับเก่าได้; CAS conflict ไม่เกิด partial pointer/snapshot

### WP04 — แก้ Knowledge publication ให้ตรง production broker

**ไฟล์:** `api/dependencies.py`, investor ports/services, adapter ใหม่ใน `tools/investor_essence/adapters/`, `schemas/vault/entities/investor-essence-v1.json`, `investment-axis-v1.json`

**งานย่อย:**

1. สร้าง adapter ที่แปลง confirmed snapshot เป็น `application.knowledge.write_models.KnowledgeWriteCommand`
2. ใช้ payload `{metadata, body, ...}` ตาม contract ของ broker/note writer และใส่ metadata ที่ entity schemas บังคับครบ
3. อย่าส่ง claims/basic_policy เป็น keys ระดับผิดโครงสร้าง; body และ metadata ต้องตรง snapshot/hash เดียวกัน
4. เก็บ publication intent ใน runtime transaction ที่ยืนยันก่อน แล้ว dispatch หลัง commit; เพิ่มสถานะ pending/failed/published และ retry
5. เก็บ ArtifactRef จาก broker receipt จริง; ตรวจ success/status และ committed identity ครบ
6. หากเผยแพร่ล้มเหลว ให้ยืนยันใน runtime คงอยู่ แต่ response/UI บอก publication pending/failed ตามจริง
7. ใช้ idempotency fingerprint จาก confirmation ID + publication payload hash; retry ไม่สร้าง note/revision ซ้ำ

**ตรวจรับ:** production broker รับ typed command, ผ่าน schema validation และคืน ref จริง; broker reject/timeout ไม่ถูกแสดงเป็น published; retry สำเร็จจาก snapshot เดิม

### WP05 — ทำ Essence ให้ลึกและ resume ได้

**ไฟล์:** interview/claim review services, `core/investor_essence/interview.py`, claims, LLM adapter, interview/summary/clarification prompts, router, `EssenceInterview.tsx`, `EssenceResult.tsx`, hook, workflow worker

**งานย่อย:**

1. สร้าง shared evidence builder ให้ history ของ next question/summary/clarification ใช้โครงเดียวกัน
2. ส่ง question text, options, selected option text, free text, answer_id, evidence_type และ question/answer revisions; choice+คำอธิบายต้องรักษาทั้งสองอย่าง
3. นับ coverage จากคำตอบที่มีเนื้อหาและ topic ที่รองรับ ไม่ถือ unsure/skipped ว่าครอบคลุมแล้ว
4. วาง coverage ของ 6 pillars ระหว่าง 10 คำถามหลัก และตรวจ priority/trade-off, ประสบการณ์จริง, เวลา/สภาพคล่อง, วินัยที่ทำได้และข้อจำกัด
5. เมื่อยังขาดเรื่องสำคัญ ให้ clarification ต่อเฉพาะจุดหลัง 10 ข้อ ไม่สรุปว่าครบเพราะจำนวนคำถาม
6. ตรวจ evidence_refs ว่าอยู่ใน active branch, อ้าง answer จริง, quote ตรง selected text/free text; ข้ออ้างที่ไม่มีหลักฐานเป็น unresolved ไม่ใช่ accepted fact
7. partial claim ต้องแก้หรือ exclude ก่อนยืนยัน; source_kind AI_INFERRED แสดงตรง enum จริงใน UI
8. Q10 บันทึกแล้วเข้าสู่ summary_pending; summary error แสดง retry ได้โดยไม่ตอบใหม่และไม่ทิ้งคำตอบ
9. refresh/resume ดึง operation/summary status กลับมา; pause บอกสถานะบันทึกที่จริงและทำต่อได้
10. ใช้ existing worker/operation infrastructure เมื่อเหมาะสม พร้อม frozen input, resource_revision และ fencing; LLM call อยู่นอก transaction ที่ถือ write lock แล้ว CAS ผลกลับ
11. provider transient failure มี bounded retry; model schema/validation failure แสดงให้แก้ ไม่ retry แบบไม่จำกัด

**ตรวจรับ:** Q2 รู้ว่าผู้ใช้เลือกอะไร; unsure ไม่ถูกสร้างเป็นข้อเท็จจริง; 6 pillars มีสถานะ covered/needs_review/not_ready; summary fail/reload/retry เก็บคำตอบครบ; output เก่าที่ revision ไม่ตรงไม่ถูกใช้

### WP06 — Financial context และ goals ที่ใช้คำนวณได้จริง

**ไฟล์:** financial context domain/service/form, DTO/schema/repository, `tools/portfolio/domain/models.py`, goal service/goals adapters, `tools/portfolio/goals.py`, portfolio schemas/routes, `PortfolioGoalsTab.tsx`

**งานย่อย:**

1. แต่ละ fact มี status known/none/unknown พร้อม numeric value หรือ range, unit, source และ as-of; known 0 ไม่กลายเป็น unknown
2. Form แยก horizon value/range, amount และหน่วย; parse comma/ช่องว่างอย่างชัดเจน ค่าผิดคืน 422 ระบุ field แทน Decimal exception
3. เก็บและส่งต่อ target use amount/date, emergency amount/months, obligations, withdrawal amount/frequency, experience และ unknowns ครบ
4. เพิ่ม monthly investable contribution, essential monthly expense, existing reserves outside portfolio และ funding scope ของเงินสำรอง
5. readiness แยกความพร้อมร่างเชิงคุณภาพจากความพร้อมยืนยันแผนตัวเลข; unknown horizon/liquidity/funding ที่จำเป็นต้องเป็น blocking ต่อการคำนวณส่วนนั้น
6. เพิ่ม stable goal_id และ revision/semantic hash; rename goal ไม่เปลี่ยน ID และ goal ชื่อซ้ำต่างพอร์ตไม่ชน
7. goal links แยก goal ภาพรวมจากจำนวนที่พอร์ตนี้รับผิดชอบ; goals ทุกพอร์ตใช้เป็นบริบท แต่ไม่ fund ซ้ำอัตโนมัติ
8. แยก stock targets เช่นเงินก้อน/NAV จาก flow targets เช่นรายได้ต่อปี; aggregate NAV goal ไม่ถูกบวกซ้ำกับ buckets ที่เป็นส่วนประกอบ
9. เพิ่ม priority, deadline flexibility, amount flexibility และ currency/FX basis เท่าที่จำเป็นต่อ funding plan
10. สร้าง normalized planning context ที่ generator ทั้ง Axis/Buckets ใช้ร่วมกัน พร้อม readiness issues และ refs

**ตรวจรับ:** ข้อมูลครบทุก field หลัง round-trip; none/unknown/0 ไม่ปะปน; goal links คงเดิมหลัง rename; เป้าหมายคนละพอร์ตไม่ถูกรวมเป็นภาระพอร์ตนี้; แผนรายได้ต่อปีไม่ถูกคำนวณเป็นเงินก้อนโดยเงียบ

### WP07 — Axis ที่ตรวจความหมายของตัวเลขและสถานะได้

**ไฟล์:** `core/investor_essence/investment_axis.py`, axis service/editor, `INVESTMENT_AXIS.md`, LLM adapter, confirmed DTO/schema

**งานย่อย:**

1. ครบ 8 หัวข้อและ non-actions อย่างน้อย 3 ข้อที่มีเนื้อหาและไม่ซ้ำ; รายการที่มีแต่ช่องว่างไม่ถือว่าครบ
2. ตรวจ risk value finite และอยู่ในช่วงของ unit; percent 0–100; มี calculation_basis ที่รองรับและเข้าใจตรงกัน
3. เลิก fallback MDD=15/loss=2; เสนอโดย AI ต้องระบุ origin/assumptions/source และผู้ใช้ยืนยันอย่างเห็นค่า
4. เปลี่ยนค่า origin เป็น USER_INPUT เฉพาะตัวเลขที่ผู้ใช้แก้; ผู้ใช้ยอมรับข้อเสนอ AI ให้คง provenance พร้อม confirmation flag/revision
5. รองรับข้อจำกัดที่ไม่เกี่ยวข้อง เช่น per-trade loss สำหรับวิธีที่ไม่มี trade sizing เป็น explicit not_applicable พร้อมเหตุผล แทนเติมเลขเพื่อผ่าน checklist
6. ตรวจ allocation rows ไม่ติดลบ, IDs unique, sum 100±0.01; categories ใช้ stable ID และตรง asset taxonomy ที่จะใช้ใน matrix
7. ระบุ rebalance frequency + threshold + threshold unit เป็น structured rule เช่นจุดเปอร์เซ็นต์ ไม่ใช้ข้อความเดียวเป็นกฎที่คำนวณไม่ได้
8. ตรวจ goal/context readiness และ unresolved items ที่มีผลต่อ numeric policy ก่อน confirm
9. Editor read-only อ่าน confirmed snapshot; edit mode อ่าน draft; แสดง edits pending และ stale reasons พร้อม source revision
10. confirm รับ expected draft revision และ source refs; ข้อมูลต้นทางเปลี่ยนระหว่าง save/confirm คืน 409 ให้ทบทวน

**ตรวจรับ:** invalid number/unit/basis ถูกปฏิเสธ; no-default numerics; draft ไม่แสดงว่าใช้งานจริง; bucket generator ได้ allocation จาก Axis ที่แสดงว่า confirmed; การยืนยันไม่รับ revision เก่า

### WP08 — แก้ snapshot และ input ที่ส่งไปออกแบบ buckets

**ไฟล์:** `portfolio_planning_adapter.py`, bucket service, DTO/ports, LLM adapter, `BUCKET_PROPOSAL.md`

**งานย่อย:**

1. ใช้ HoldingPlanningSnapshot ชุดเดียว: holding identity, symbol, asset_type, market_value_thb, weight_percent, existing bucket_id และ currency
2. อ่าน NAV/holdings/targets/checkpoint จาก state เดียวที่ consistent; ห้ามโหลด state กับ checkpoint คนละเวลาโดยไม่มี consistency guarantee
3. คำนวณ weight = market_value/NAV ×100; NAV=0 แสดงสภาวะ empty ไม่หารศูนย์หรือแต่ง actual weights
4. ส่ง holdings ครบ หรือ aggregate อย่างมี coverage/count/total พร้อมรายการที่ต้อง remap ครบ; ห้ามตัดเหลือ 10 รายการแบบเงียบ
5. ส่ง confirmed Axis รวม allocation_basis/rows/risk/non-actions, confirmed Essence claims, exact financial context และ scoped goals
6. generator port รองรับ input ใหม่ครบและ fakes ตรวจทุก field; malformed AI output ผ่าน proposal validator ก่อนบันทึก
7. ระบุ as-of/FX และ source refs ใน draft; เปลี่ยนต้นทางระหว่าง generate ให้ mark superseded/stale แทนถือว่าพร้อมใช้

**ตรวจรับ:** 900,000 จาก NAV 1,000,000 ได้ 90%; portfolio >10 holdings มี coverage ครบ; ค่า NAV/weight/asset type ถูกต้อง; source lineage ไม่เป็น latest/hash placeholders

### WP09 — สร้าง BucketPolicy และ funding calculation ที่ practical

**ไฟล์:** domain models/DTO/schema, bucket service, module ใหม่ `core/investor_essence/bucket_policy.py` และ `funding.py`, LLM adapter/prompt, portfolio domain models

**ข้อมูลที่ต้องมีต่อ bucket:**

| ส่วน | Fields ที่ต้องเพิ่ม/ปรับ |
| --- | --- |
| Identity | bucket_id, name, purpose, role, schema_version |
| Goals | goal links ที่มี goal_id/revision/funding_scope/funding_amount หรือ explicit purpose ที่ยังไม่ผูก goal |
| Targets | target_percent, target_amount, minimum_amount, calculation_basis, currency |
| Time | need_by_date/horizon, liquidity_requirement, withdrawal schedule |
| Inflow | route mode, route_percent, priority, when_full_redirect_bucket_id |
| Outflow | withdrawal order/condition/limit, refill rule |
| Assets | allowed/excluded category IDs + ข้อความอธิบาย, source axis allocations |
| Rules | structured rebalance/risk/concentration rules + narrative notes |
| Provenance | essence/axis/context/goal refs, source hash, assumptions, confirmed revision |

current value, gap, estimated monthly contribution และ funding progress เป็น derived view จาก policy + state ปัจจุบัน ไม่เก็บเป็นยอดจริงค้างอยู่ใน policy

**งานย่อย:**

1. รองรับ target แบบเงินก้อน/floor และแบบ % ตามบทบาท ไม่บังคับทุกเป้าหมายเป็น %NAV อย่างเดียว
2. reserve requirement คำนวณจาก expense × months ที่ผู้ใช้ยืนยัน แล้วหัก reserve นอกพอร์ตตาม scope; ถ้ามี amount ที่ยืนยันแล้วให้แสดงฐานและความต่าง ไม่เลือกแทนโดยเงียบ
3. `gap=max(0,target-current)`; แยกยอดที่มีจริงจากยอดที่ตั้งเป้าและยอดที่ allocate บนกระดาษ
4. เงินเข้า fixed routing: amount_i = monthly_budget × route_i/100; route รวม 100 หาก mode นี้ หรือรองรับ priority fill ที่มี remainder destination ชัด
5. เงินเข้า priority fill ใช้ gap/floor/priority/เงินที่มีจริง; เต็มเป้าแล้ว redirect; เงินไม่พอแสดง unmet requirement ตามลำดับ
6. คำนวณ baseline contribution = gap/months_remaining แบบไม่สมมติผลตอบแทน; deadline ผ่านหรือข้อมูลไม่ครบแสดงสถานะที่ตรงเหตุ ไม่แสดง feasibility ว่าเป็นไปได้โดยไม่มีข้อมูล
7. scenario ที่มีสมมติฐานผลตอบแทนแยกเป็น optional view พร้อม assumption; ไม่ใช้เป็น gate ว่า goal สำเร็จแน่นอน
8. residual allocation ที่ไม่รู้เป้าหมายใช้เงินให้แสดง flexible/uncommitted แทนผูก goal ปลอม
9. จำนวน buckets ตามบทบาทจริง; เริ่มแนะนำจำนวนที่ดูแลได้ เช่นไม่เกิน 5 ใบ แต่ไม่บังคับว่าทุกคนต้องมี emergency/income/growth/tactical ครบ
10. free-text rules ใช้เพื่ออธิบาย ส่วนกฎที่ต้อง validate/calculate ต้องเป็น structured fields; AI ไม่กำหนดผลคำนวณแทน deterministic engine

**ตรวจรับ:** ผู้ใช้เห็นเงินที่ขาดและจำนวนที่ควรจัดจาก budget ที่ระบุ; routing ไม่เกินเงินที่มี; เงินสำรองนอกพอร์ตไม่ถูกนับซ้ำ; goals ที่ไม่มี deadline/budget ไม่ถูกเติมขึ้นเอง

### WP10 — ตรวจ matrix และจัด holdings โดยไม่ล้างข้อมูลเงียบ

**ไฟล์:** `core/investor_essence/validation.py`, module ใหม่ `bucket_validation.py`, bucket service, portfolio adapter/calculations, bucket builder/remapping UI

**งานย่อย:**

1. Shared validator ตรวจ unique IDs, numeric ranges, target sum, valid refs, funding scopes และ structured policy constraints
2. Matrix ใช้ portfolio weights: cell รวม 100, row sums ตรง Axis target, column sums ตรง bucket target; category/bucket IDs ต้องมีจริง
3. ตรวจ allowed asset compatibility/exclude constraints จาก stable category IDs ไม่ตัดสินจาก substring ของข้อความ
4. แยก target matrix จาก actual holdings: holdings ปัจจุบันอาจเบี่ยงเบนจาก targets ได้โดยไม่ถือเป็น matrix validation error
5. แก้ target/add/remove bucket แล้ว rebuild/review matrix; หาก constraints ทำให้จัดไม่ได้ แจ้งให้ปรับ Axis หรือ bucket ก่อน แทนเปลี่ยน Axis ที่ยืนยันแล้วอัตโนมัติ
6. ใช้ holding-level assignment โดย stable holding identity; existing bucket remap เป็น bulk suggestion ที่ตรวจได้ ไม่ใช่การตัดสินตามชื่อภาษาไทยเทียบ ID
7. หนึ่ง holding อยู่หนึ่ง purpose bucket ในขอบเขตครั้งนี้; target matrix อาจ N:N แต่การ split holding ข้ามหลาย purpose ต้องมีโมเดลรองรับก่อน ไม่จำลองว่า split แล้ว
8. Unassigned เป็นตัวเลือก explicit พร้อมจำนวน/มูลค่า; require acknowledgement ถ้า Apply จะทำ holding ที่เคยมีหมวดกลายเป็น Unassigned
9. reject nonexistent destination และ assignment ที่เข้าหมวดสินทรัพย์ไม่อนุญาต; unknown asset classification ต้องทบทวน
10. allocation view จัด orphan bucket IDs ให้อยู่ในกลุ่มต้องจัดหมวดใหม่ และรวม value ครบ NAV
11. invalid draft บันทึกเพื่อแก้ต่อได้โดย is_valid=false แต่ Preview/Apply ต้องแสดง blocking issues

**ตรวจรับ:** 90/10 กับ matrix 70/30 ไม่ผ่าน; DCA 180% ไม่ผ่าน; preset เปลี่ยน IDs แล้วไม่มี dangling refs; ไม่มี holdings/value หายจาก allocation view

### WP11 — Preview และยืนยันแผนเดียวกับที่จะ Apply

**ไฟล์:** Preview/Apply DTO/schema/router, bucket service, portfolio adapter, `BucketApplyReview.tsx`, hook

**งานย่อย:**

1. Preview คืน preview_id, draft_id/revision, semantic payload_hash, portfolio checkpoint sequence/hash, source refs และ validation results
2. before/after มี targets, policy, holding assignments, Unassigned values และผล derived funding ที่เปลี่ยน
3. ตรวจ draft sources/checkpoint กับสถานะปัจจุบัน; หาก stale แสดงเหตุที่เจาะจง
4. เพิ่ม refresh/rebase ที่รักษา user edits: เปลี่ยน checkpoint และ actuals, ตรวจ newly added/deleted holdings, invalidate assignments ที่ใช้ไม่ได้ แล้วสร้าง preview ใหม่
5. ถ้า Essence/Axis/Context/Goals เปลี่ยน ให้ผู้ใช้ทบทวนหรือสร้าง draft revision ใหม่จาก sources ที่เลือก; ไม่แอบแทน refs
6. Apply request ส่ง preview_id/expected draft revision/payload hash/checkpoint/source refs/idempotency key; server revalidate ทุกอย่างใต้ lock ที่จำเป็น
7. ปุ่ม Apply ใช้ได้เมื่อมี preview ตรงแผน ไม่มี blocking issues และผู้ใช้ตรวจ revision นี้แล้ว
8. checkbox acknowledgement reset เมื่อ preview/draft/payload เปลี่ยน; 409 ส่งกลับไป review แบบรักษา edits

**ตรวจรับ:** ไม่มี preview/preview คนละ revision/พอร์ตเปลี่ยนหลัง preview → Apply ไม่ได้; payload ที่ผู้ใช้ตรวจตรงกับ payload ที่บันทึก; rebase ไม่เรียก AI หรือทิ้ง edits โดยไม่จำเป็น

### WP12 — บันทึก policy และ receipt ให้กู้ต่อได้

**ไฟล์:** `tools/portfolio/domain/models.py`, `ports/repository_port.py`, `transaction_store.py`, `transactional_repository.py`, `portfolio_planning_adapter.py`, investor bucket service/intents, projection/mirror adapters, related schemas

**งานย่อย:**

1. เพิ่ม `allocation_policy` และ `allocation_plan_origin` ใน PortfolioState พร้อม full applied BucketPolicy, mapping และ source refs
2. รักษา allocation_targets เดิมสำหรับหน้า/consumer เก่า แต่เชื่อม bucket_id ให้ตรง policy; manual edit targets ต้องอัปเดต policy หรือ mark modified_since_apply
3. เพิ่ม durable command receipt ใน portfolio transaction store keyed by portfolio_id/use_case/idempotency_key พร้อม request_hash และ committed checkpoint
4. commit state/targets/assignments/policy/receipt ใน SQLite transaction เดียวของ portfolio source ไม่แยก append receipt หลัง commit
5. ตรวจ receipt ก่อน checkpoint conflict: key+payload เดิมคืน receipt เดิมแม้ checkpoint ก้าวแล้ว; key เดิม payload ใหม่คืน 409
6. investor runtime persist apply intent พร้อม frozen payload และ stable command_id ก่อนเรียก portfolio adapter; release runtime transaction ก่อนข้าม storage
7. portfolio committed แล้ว runtime save ล้มเหลว: retry/reconciler ดึง durable receipt มา finalize draft/intents ได้โดยไม่ commit state ใหม่
8. projection ล้มเหลวหลัง canonical commit: receipt บอก canonical committed, projection pending; rebuild จาก transaction source แล้ว update projection status ตามจริง
9. audit/repair ต้องตรวจ command ID, fingerprint และ checkpoint ที่เกี่ยวข้อง ไม่ตอบ receipt 'repaired' แบบไม่ตรวจหลักฐาน
10. applied plan immutable; แก้ต่อสร้าง revision/plan ใหม่ แทนแก้ applied draft แล้วเปลี่ยน receipt เดิม
11. Apply เปลี่ยนเฉพาะ planning state ที่ระบุ; verify units/costs/NAV/FX/trade ledger ไม่เปลี่ยนจากการ remap

**ตรวจรับ:** fault ก่อน/หลัง canonical commit, ระหว่าง runtime finalize และ projection write กู้ได้; retry ไม่เพิ่ม sequence ซ้ำ; full policy ยังอยู่หลัง replay/reload; failure ทั้งหมดแสดงสถานะที่ตรง committed state

### WP13 — ทำหน้าที่ผู้ใช้ใช้ดูแลพอร์ตจริง

**ไฟล์:** components ใน `portfolio/essence/`, `PortfolioOverviewTab.tsx`, `PortfolioHoldingsTab.tsx`, `PortfolioGoalsTab.tsx`, portfolio API/query service, hook

**งานย่อย:**

1. Essence แสดงหลักฐาน/coverage/ประเด็นที่ยังต้องตอบและสถานะฉบับยืนยันอย่างถูกต้อง
2. Context form ใช้ labels/หน่วย/status ที่เข้าใจง่ายและ field errors; ค่าเก่ายังอยู่เมื่อ save fail
3. Axis แสดง confirmed/draft/stale ตามแหล่งจริง พร้อมปุ่มทบทวนต้นทาง
4. Bucket builder แสดง target THB/%/floor, goal/deadline, current/gap, monthly inflow, allowed assets, withdrawal/refill/rebalance rules
5. แยก actuals ที่เปลี่ยนตามตลาดจาก policy targets; zero NAV ไม่ทำให้แผนเงินก้อนหาย
6. เปลี่ยน presets เป็น templates ของบทบาทเงิน ใช้ goals/context ของผู้ใช้สร้างค่า; ไม่ใส่ 10 ล้าน/300,000/2% ตายตัว
7. ใช้ stable IDs เมื่อปรับ template และตรวจ matrix/assignments หลังเปลี่ยน; ไม่มี goal ที่ตรงให้เลือก purpose explicit
8. Overview อ่าน applied policy และแสดงสิ่งที่ต้องทำจากข้อมูลจริง เช่นจำนวนเงินเข้าแต่ละ bucket, funding gap, unmet floor และ rebalance review
9. Holdings แสดง assigned purpose, policy compatibility และทางแก้ Unassigned; Goals แสดง contribution จากพอร์ตนี้และความต่างจากภาพรวม
10. reload/resume ได้ applied receipt/active policy/status; regenerate ไม่โชว์ความสำเร็จของแผนเดิม
11. product copy ใช้คำว่าแผน/ข้อเสนอ/ต้องตรวจเมื่อยังเป็น draft และบอกงานใดระบบเพียงแนะนำกับงานที่มีผลบันทึกจริงให้ตรง implementation

**ตรวจรับ:** ผู้ใช้ตอบได้ว่าเงินแต่ละก้อนทำหน้าที่อะไรและเดือนนี้จัดอย่างไรจากหน้า Overview; ไม่ต้องกลับไปเปิด draft เก่าเพื่ออ่าน policy ที่ Apply แล้ว

### WP14 — Migration และ compatibility ที่ไม่สวมฉบับผิด

**ไฟล์:** `api/db/connection.py`, repository parsers, portfolio/goal models/adapters, script ใหม่สำหรับ backfill/audit ใน `scripts/`, migration tests

**งานย่อย:**

1. ออกแบบ additive migrations ตั้งแต่ WP02 และสร้าง dry-run รายงาน counts/refs/records ที่กู้ได้กับไม่ได้
2. legacy Essence/Axis: กู้จาก exact artifact + content hash หรือ durable confirmation receipt/evidence ที่พิสูจน์ final content ได้เท่านั้น
3. ถ้าหลักฐานไม่ครบให้ legacy_unverified พร้อมปุ่มทบทวนยืนยันใหม่; ไม่ใช้ latest draft เป็น source ของ confirmed
4. แยก policy semantic hash ใหม่จาก Vault artifact content hash เก่า; เก็บ legacy refs/hash เดิมไว้เพื่อ audit ไม่เขียนทับย้อนหลัง
5. goal_id สร้างครั้งเดียวและ persist พร้อม legacy identity mapping; rename หลังย้ายต้องไม่หลุด bucket linkage
6. context เก่าที่ parser อ่าน numeric/range ไม่ได้เก็บ raw value และ needs_review; ไม่ทิ้งหรือเปลี่ยนเป็น 0
7. targets เก่าไม่มี role/rules ให้ policy metadata_status=needs_review แต่ actual holdings/targets เดิมยังอ่านได้
8. remap orphan IDs เข้ารายการ review ใน derived UI โดยไม่เปลี่ยน assignments จริงอัตโนมัติ
9. backfill policy ที่ Apply แล้วต้องมี evidence จาก applied plan/receipt/checkpoint; หากกู้ไม่ได้อย่าอ้าง operational rules ที่ไม่มี
10. migration รันซ้ำได้, verify state/holdings/NAV/ledger ก่อนหลัง และ append migration transaction เมื่อจำเป็นแทน rewrite event history
11. เตรียม rollback ของ code/schema compatibility และข้อมูลสำรองใน rehearsal; หลังมี V2 writes reader เก่าต้องไม่ทำ fields ใหม่หาย

**ตรวจรับ:** migration แบบ dry-run ไม่เปลี่ยนข้อมูล; รันจริงซ้ำแล้วไม่เพิ่ม IDs/revisions/receipts ซ้ำ; unsupported records ถูกระบุ; ไม่มีหลักฐานผิดฉบับและไม่มี financial state เปลี่ยน

### WP15 — ตรวจรับทั้งระบบ

**ไฟล์:** unit/API/integration/React tests, temporary fixtures, browser test/manual acceptance script, synthetic AI evaluation fixtures

**งานย่อย:**

1. รัน suites ที่เกี่ยวข้องกับ investor, repository, portfolio recovery/goals/state และ frontend Portfolio/essence หลังแต่ละชุดงาน
2. API contract tests ตรวจ required fields, source refs/revision และ structured errors จริง
3. integration tests ใช้ SQLite runtime + production PortfolioPlanningAdapter/transaction source + production broker ผ่าน isolated Vault
4. fault injection ตาม WP12 รวม concurrent save/confirm/apply และผล request เก่าจาก generator
5. browser ตรวจ happy path, failure/retry, สลับพอร์ต, reload/resume, mobile layout และ long Thai text
6. ประเมิน AI ด้วยผู้ใช้สังเคราะห์ 6 profiles: เงินใช้ใกล้ deadline, สะสมระยะยาว, มี withdrawals, มี reserves นอกพอร์ต, มี goals ต่างพอร์ต, ข้อมูลไม่ครบ/unsure
7. ใช้ rubric ตรวจ evidence, specificity, source alignment, missing data handling, feasibility explanation และการไม่สร้าง goal/ตัวเลขขึ้นเอง
8. LLM output ไม่ต้องเหมือนกันทุกคำ แต่ทุก output ต้องผ่าน deterministic validators และย้อนกลับหลักฐานได้
9. export OpenAPI/gen types/check drift, typecheck/build/lint และ affected regression suites ก่อนปิดงาน
10. บันทึกผลตาม test IDs และ remaining issues; ไม่ถือว่า build หรือ happy path ผ่านแปลว่าผ่าน release gate

**ตรวจรับ:** ผ่านรายการในส่วน 8 ทั้งหมด และยังไม่พบ P1 ที่เปิดอยู่

## 5. ชุดส่งมอบที่แนะนำ

| ชุด | ขอบเขต | ผลที่ผู้ใช้/ผู้รีวิวตรวจได้ |
| --- | --- | --- |
| D1 | WP00 + WP01 | ไม่แก้ผิดพอร์ต, บันทึก fail แล้วหยุด, edits ไม่หาย |
| D2 | WP02 + WP03 | DTO ตรงกัน, confirmed snapshot/history/ref/hash ถูกต้อง |
| D3 | WP04 + WP05 | publication ตรงจริง, interview มีความหมาย, summary retry/clarification ได้ |
| D4 | WP06 + WP07 | context/goals ครบ, readiness/risk/Axis ตรวจได้ |
| D5 | WP08 + WP09 | buckets อิง holdings จริงและมี funding plan ที่คำนวณได้ |
| D6 | WP10 + WP11 | matrix/assignments valid, preview ตรง revision/checkpoint |
| D7 | WP12 | applied policy อยู่ในพอร์ตและ retry/recovery ถูกต้อง |
| D8 | WP13 | Overview/Holdings/Goals ใช้ buckets ปฏิบัติจริงได้ |
| D9 | WP14 + WP15 | legacy migration ผ่าน rehearsal และ end-to-end ผ่าน |

ออกแบบ migrations ควบคู่ทุกชุดที่เพิ่ม schema; D9 เป็นการ rehearsal/backfill/ตรวจรับรวม ไม่ใช่เพิ่งออกแบบ migration ตอนจบ จำนวน PR อาจมากกว่า 9 เพื่อให้ review ง่าย แต่ gate/dependencies ต้องคงเดิม

## 6. แผน migration และ rollout

1. Inventory แบบอ่านอย่างเดียว: จำนวน confirmed refs ที่ snapshot ว่าง, draft ที่ refs เป็น placeholder, applied plans/receipts, orphan assignments และ goals ชื่อซ้ำ
2. Deploy additive readers/schema และ versioned serializers ก่อนสร้างข้อมูล V2; tests ต้องพิสูจน์ว่า legacy state ยัง round-trip ได้
3. ทำ rehearsal บน copies ใน isolated runtime/Vault, ตรวจ diff ที่คาดไว้และความครบของ refs
4. รัน backfill ที่กู้ได้จากหลักฐานจริง แยกต้องทบทวนออกมาชัดเจน
5. เปิด V2 writes ที่ผ่าน WP01–WP12 แล้ว; UI ใช้ statuses/readiness gates ตาม V2 และบอก legacy records ที่ต้องทบทวน
6. ตรวจ receipt replay, policy retention และ allocation conservation หลังเริ่มใช้; rollback code ได้เฉพาะ reader ที่รักษา V2 fields ได้
7. ไม่ rewrite confirmation/event history เดิมและไม่ reclassify holdings โดยไม่มี reviewed plan

## 7. Mapping จากผลรีวิวไปยังงานแก้

| Finding ใน REVIEW.md | งานแก้หลัก |
| --- | --- |
| P1-01 Wrong-portfolio mutations | WP01 |
| P1-02 Confirmed claims สูญหาย | WP02, WP03 |
| P1-03 Choice text หายจาก adaptive input | WP05 |
| P1-04 Failure ไม่หยุด flow | WP01, WP11 |
| P1-05 Context parsing/input fields | WP02, WP06 |
| P1-06 Readiness/risk validation | WP06, WP07 |
| P1-07 Holdings value/weight=0 | WP08 |
| P1-08 Matrix/DCA/IDs validation | WP09, WP10 |
| P1-09 Unassign/orphan remapping | WP10, WP11 |
| P1-10 Applied policy metadata สูญหาย | WP09, WP12, WP13 |
| P1-11 Knowledge broker contract | WP04 |
| P1-12 Commit สำเร็จแต่ runtime/retry ล้มเหลว | WP12 |
| P2-01 Draft แสดงเป็น confirmed | WP01, WP07, WP13 |
| P2-02 Apply ไม่มี/มีปัญหา preview | WP11 |
| P2-03 Q10 summary retry ไม่ได้ | WP05 |
| P2-04 Placeholder provenance | WP03, WP08, WP14 |
| P2-05 Hash ไม่รวม allocation | WP02, WP03 |
| P2-06 Stale checkpoint/rebase | WP11 |
| P2-07 Readiness DTO drift | WP02, WP06 |
| P2-08 Presets ตายตัว | WP09, WP13 |
| P2-09 Tolerance ต่างกัน | WP02, WP10 |
| P2-10 Receipt/applied UI state ค้าง | WP01, WP12, WP13 |
| P2-11 Fingerprint ไม่รวม payload | WP02, WP03, WP12 |
| P2-12 Bucket input ขาด lineage/context | WP03, WP06, WP08 |
| เพิ่มจากการออกแบบ: latest-draft recovery ผิดฉบับ | WP03, WP14 |
| ความลึกและ practicality | WP05, WP06, WP07, WP09, WP13, WP15 |

## 8. Acceptance scenarios ที่ต้องผ่าน

| ID | กรณีทดสอบ | ผลที่ต้องได้ |
| --- | --- | --- |
| AC01 | เปิด A, สลับ B, request A ตอบทีหลัง | B ไม่รับ state A; mutation ทั้งหมดของ B ใช้ B |
| AC02 | save context/axis/buckets ล้มเหลว | ขั้นถัดไปไม่ถูกเรียก; edits ยังอยู่ |
| AC03 | batch confirm claims ที่ไม่กด exact ทีละข้อ | snapshot/Axis มี accepted claims ตามรายการที่ยืนยันครบ |
| AC04 | เริ่ม session/draft ใหม่หลังมี confirmed | downstream ยังใช้ฉบับเดิมจนยืนยันใหม่ |
| AC05 | choice + free text → ถามต่อ | generator เห็น selected text กับคำอธิบายครบ |
| AC06 | unsure/skipped ใน pillar สำคัญ | coverage ไม่ครบ; มี clarification/unresolved โดยไม่เติมข้อเท็จจริง |
| AC07 | Q10 summary fail/reload/retry | คำตอบครบ 10; retry summary ได้; ไม่เริ่มใหม่ |
| AC08 | ค่า comma/range/none/unknown/0 | parse หรือ 422 ตาม contract; ไม่เกิด 500/เปลี่ยนความหมาย |
| AC09 | ไม่ทราบ deadline/budget/reserves | readiness/calc บอกส่วนที่ยังประเมินไม่ได้ |
| AC10 | risk=-10/NaN/unit ว่าง | confirm ไม่ผ่าน; ไม่มี fallback เลขตายตัว |
| AC11 | 900,000 จาก NAV 1,000,000 และ >10 holdings | 90%; coverage ของ holdings/assignments ครบ |
| AC12 | buckets 90/10 แต่ matrix 70/30 | blocking validation ที่ row/column ที่ผิด |
| AC13 | DCA 180%, budget 20,000 | reject; routing ที่ valid รวมไม่เกิน 20,000 |
| AC14 | preset/add/remove bucket | IDs/links/matrix/assignments ไม่มี dangling refs |
| AC15 | orphan/Unassigned holdings | แสดงจำนวน/THB ชัด; allocation values รวมครบ NAV |
| AC16 | preview แล้ว draft/portfolio/source เปลี่ยน | Apply คืน 409; rebase/review รักษา edits |
| AC17 | Apply ไม่มี preview หรือคนละ revision | ปุ่มปิดและ server ปฏิเสธ |
| AC18 | Apply แล้ว reload/replay | policy/rules/goals/receipt ครบและแสดงจาก operational state |
| AC19 | retry key เดิม payload เดิม | receipt/command/checkpoint เดิม; ไม่มี commit ซ้ำ |
| AC20 | retry key เดิม payload ใหม่ | 409 idempotency conflict |
| AC21 | canonical commit แล้ว runtime finalize fail | retry/reconcile กู้ receipt ได้และไม่เพิ่ม sequence ซ้ำ |
| AC22 | projection fail หลัง commit | canonical committed + projection pending; repair ตรวจหลักฐานได้ |
| AC23 | broker reject/timeout แล้ว retry | runtime confirmed; publication failed/pending; retry ได้ ref จริง |
| AC24 | เปลี่ยน allocation อย่างเดียว | semantic hash เปลี่ยน; old snapshot ยังอ่านได้ |
| AC25 | rename goal และ goals ชื่อซ้ำต่างพอร์ต | stable links และ funding attribution ไม่เปลี่ยนผิดพอร์ต |
| AC26 | aggregate NAV + reserve + dated goal | ไม่ double-count targets; budgets/gaps อยู่ใน scope เดียวกัน |
| AC27 | legacy pointer มีแต่ latest draft | legacy_unverified; ไม่ประกาศ draft ว่าเคย confirmed |
| AC28 | migration dry-run/run/run ซ้ำ | dry-run ไม่เขียน; run ซ้ำไม่ duplicate; holdings/NAV/ledger เดิมครบ |
| AC29 | manual edit targets หลัง Apply | policy consistency ถูกปรับหรือมี modified_since_apply ชัด |
| AC30 | ผู้ใช้สังเคราะห์ 6 profiles ผ่าน UI | ได้ buckets ที่สอดคล้องข้อมูลจริงและมีรายการปฏิบัติที่คำนวณได้ |

## 9. เกณฑ์ปิดงาน

ทุก WP มีผลตรวจรับ, Findings ทั้ง 24 ข้อมี regression test หรือหลักฐานตรวจรับที่ตรงปัญหา, AC01–AC30 ผ่าน, generated types/build/lint/affected suites ผ่าน, migration/recovery มี rehearsal และหน้า Overview ตอบได้ว่าแต่ละ bucket มีเงินเท่าไร ขาดเท่าไร เงินเข้าเดือนนี้ไปไหน และควรทบทวนเมื่อไร

เริ่มลงมือจาก D1 แล้ว D2 ตามลำดับ เพื่อให้การแก้ข้อมูลและ UX ในชุดถัดไปตั้งอยู่บน state และฉบับยืนยันที่ถูกต้อง
