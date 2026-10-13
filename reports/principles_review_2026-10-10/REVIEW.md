# รีวิว Portfolio Principles — 10 ตุลาคม 2026

เป้าหมาย: ตรวจหน้า `http://localhost:5173/portfolio?tab=principles` ตั้งแต่ Essence → Axis → Purpose Buckets → Apply ว่าโค้ดทำงานถูกต้อง ส่งข้อมูลครบ และสร้างแผนที่ปฏิบัติได้จริง

**ผลประเมิน: ยังไม่ผ่านสำหรับนำไปใช้จัดพอร์ตจริงแบบครบ flow** โครงสร้างมีครบและ build/test เดิมผ่าน แต่การส่งข้อมูลสำคัญขาดตอน การตรวจความสอดคล้องของแผนยังไม่ครบ และ Apply เก็บรายละเอียดนโยบายไม่ครบ

รีวิวจาก working tree ปัจจุบันซึ่งมีการแก้ไขก่อนเริ่มงานหลายไฟล์ ไม่ได้แก้โค้ดผลิตภัณฑ์หรือข้อมูลพอร์ตจริง สร้างเฉพาะรายงานและเคสจำลองด้วยข้อมูลสังเคราะห์ Browser ของเซสชันนี้ใช้งานไม่ได้ จึงยังไม่ได้ตรวจภาพหน้าจอและการกดผ่าน browser จริง ไม่ได้เรียก LLM จริงเพื่อวัดคุณภาพข้อความที่สร้าง

## ผลตรวจแต่ละช่วง

| ช่วง | สิ่งที่มีแล้ว | สิ่งที่ยังไม่ผ่าน |
| --- | --- | --- |
| ค้นหาแก่นแท้ | คำถาม 10 ข้อ, 4 ตัวเลือก, free text/unsure, ประวัติคำตอบ, ทบทวน/แก้ claims | คำถามถัดไปไม่ได้รับข้อความตัวเลือกที่เลือก; confirmed essence ไม่มี snapshot เนื้อหาสำหรับ downstream; coverage ยังอิงคำสั่งให้ AI ทำ |
| สร้างแกนหลัก | นโยบาย 8 หัวข้อ, risk limits, allocation, non-actions, goals | อ่าน summary ของ session ล่าสุดแทนฉบับยืนยัน; financial context ตกหล่น; readiness/ค่าตัวเลขยังเชื่อถือไม่ได้; draft อาจถูกแสดงว่าใช้งานจริง |
| ออกแบบ buckets | role, linked goal name, eligible assets, DCA %, action rules | มูลค่า holdings ใน prompt เป็น 0; matrix/DCA ไม่ถูกตรวจ; presets ตายตัว; remapping ใช้ไม่ได้อย่างมั่นใจ |
| Preview/Apply | บันทึก targets และ remap holdings ใน portfolio unit of work, ตรวจ sequence ตอน apply | ไม่มี preview ก็ Apply ได้; การบันทึกล้มเหลวไม่หยุด flow; ไม่เก็บ operational policy ในพอร์ต; recovery ข้าม storage ยังไม่ครบ |

## ข้อบกพร่องที่ควรแก้ก่อนใช้งานจริง (P1)

### 1. เปลี่ยนพอร์ตแล้ว draft ของพอร์ตเดิมค้าง และ mutation ถูกส่งไปผิดพอร์ต

- หลักฐาน: `web/src/hooks/useInvestorPrinciples.ts:121–140` ตั้งค่า draft/plan เฉพาะเมื่อ API คืน object แต่ไม่ clear เมื่อคืน null/error; `web/src/pages/Portfolio.tsx:698` ใช้ component เดิมโดยเปลี่ยน portfolioId ไม่มี key ที่ remount
- พิสูจน์ใน React: เปิด A ซึ่งมี `plan-A`, สลับ B ที่ไม่มีแผน → hook รายงาน portfolioId B แต่ bucketPlan.portfolio_id A → update ส่งไป `plan-A`
- มีความเสี่ยงจาก request เก่าที่ตอบกลับช้าด้วย เพราะ mounted ref ไม่ตรวจ portfolioId/request generation
- แก้: reset state เฉพาะพอร์ตทุกครั้งที่เปลี่ยน ID, ยกเลิก/ไม่รับผลจาก request เก่า, ตรวจ draft.portfolio_id เทียบ selected portfolio ก่อนทุก mutation

### 2. ยืนยันแก่นแท้ทั้งชุดแล้ว Axis อาจได้รับ accepted_claims ว่าง

- หลักฐาน: `application/investor_essence/confirmation_service.py:137` ตั้ง pointer แต่ไม่บันทึก confirmed snapshot และไม่อัปเดต is_accepted ใน summary; `application/investor_essence/investment_axis_service.py:143` อ่าน current session summary แล้วกรอง `c.is_accepted`
- พิสูจน์: สรุป 2 claims ที่ยังไม่กด exact รายข้อ → confirm โดย accepted IDs ทั้งชุดสำเร็จ count 2 → generate Axis ได้ accepted_claims `[]`
- ถ้าเริ่ม session ใหม่ก่อนยืนยันใหม่ Axis ก็จะอ่าน session ใหม่ ทั้งที่ pointer ยังเป็น essence เก่า
- `/essence/current` ส่งเพียง scope/kind/pointer ขัดกับ ConfirmedEssenceDTO ซึ่งต้องการ statement/accepted_claims/artifact_id ทำให้การแสดงฉบับยืนยันไม่สมบูรณ์ด้วย
- แก้: บันทึก immutable confirmed snapshot พร้อม claims, statement, evidence, unresolved topics; downstream ต้องอ่าน snapshot ที่ตรง pointer/ref เท่านั้น

### 3. การสัมภาษณ์แบบ adaptive สูญเสียความหมายของคำตอบแบบ choice

- หลักฐาน: `application/investor_essence/interview_service.py:247` history สำหรับถามข้อถัดไปมี text ของคำถามและ option_id เช่น `opt_1_a` แต่ไม่มีตัวเลือก/ข้อความคำตอบที่เลือก
- พิสูจน์: input สำหรับสร้าง Q2 มี answer.kind=choice, option_id=opt_1_a, free_text=null; ไม่มี quote หรือ options
- AI จึงไม่รู้ว่า A หมายถึงอะไรในคำถามก่อนหน้า จะถามต่อให้ลึกตามคำตอบจริงได้ไม่ครบ
- แก้: ส่ง selected option text, answer_id, evidence_type และเหตุผลเพิ่มเติม; ส่งทั้ง choice และ free text เมื่อมีทั้งสองอย่าง

### 4. บันทึกไม่สำเร็จแต่ flow ยังดำเนินต่อ

- หลักฐาน: hook จับ error ใน saveFinancialContext/updateAxis/updateBucketPlan แต่ resolve ตามปกติ; `FinancialContextForm.tsx:321`, `InvestmentAxisEditor.tsx:282`, `EssenceBucketBuilder.tsx:256` จึง await แล้วทำขั้นต่อไป
- พิสูจน์ React: context save reject แต่ createAxisDraft ยังถูกเรียก; bucket save reject แต่หน้า Review ยังปรากฏ
- ผล: ผู้ใช้อาจยืนยันข้อมูลเก่าทั้งที่คิดว่าบันทึกการแก้ไขล่าสุดแล้ว
- แก้: mutation คืนผลลัพธ์ที่ชัดเจนหรือ throw; confirmation/review ใช้ ID/revision จากผล save ที่สำเร็จ; หยุด dependent step เมื่อ save/preview ล้มเหลว

### 5. Financial context รับค่าที่ UI แนะนำไม่ได้ และส่งข้อมูลต่อไม่ครบ

- หลักฐาน: `FinancialContextForm.tsx:163/177/224` ตัวอย่าง `5-10 ปี`, `500,000 บาท`, `6 เดือน`; backend `financial_context_service.py:62` ใช้ Decimal ตรง ๆ
- พิสูจน์: ค่าตัวอย่างทั้ง 3 ทำให้ decimal.InvalidOperation โดยไม่ได้แปลงเป็น domain validation error
- `investment_axis_service.py:133` ไม่ส่ง target_use_timeline, emergency_reserves_months, obligations_description, withdrawal_frequency, withdrawal_amount, experience_description ไปให้ Axis generator แม้ backend เก็บได้
- Form ไม่มี input จำนวนถอนต่อครั้ง และยังไม่มีงบลงทุนต่อเดือน/สถานะเงินสำรองอยู่นอกพอร์ตหรือในพอร์ต
- แก้: แยก numeric value/unit/range ให้ชัด รับเงินที่มี comma อย่างตั้งใจ คืน 422 พร้อมข้อความเฉพาะช่อง และส่ง context snapshot ครบโดยรักษาความแตกต่างระหว่างไม่ทราบ/ไม่มี/0

### 6. Readiness และ risk validation ไม่พอสำหรับนโยบายที่มีตัวเลข

- หลักฐาน: `core/investor_essence/financial_context.py:26/41/52` เงื่อนไข unknown/missing ไม่สอดคล้องกัน; Axis create/confirm ไม่บังคับ readiness
- พิสูจน์: horizon ไม่ทราบ เงินสำรองและภาระยังไม่มี แต่ readiness เป็น true ถ้า unknown_fields มีแค่ horizon_years
- พิสูจน์: risk_limits=-10, unit ว่าง, calculation_basis ว่าง ยังผ่าน validate_axis_completeness เมื่อ is_confirmed=true
- UI ยังเติม MDD=15 และ loss=2 หากค่าไม่มี และ save ให้เป็น user input ที่ยืนยันแล้ว
- แก้: แยก readiness สำหรับ qualitative draft กับ numeric confirmation; ตรวจ finite/range/unit/basis; ค่าจาก AI ต้องรักษา origin/assumptions และไม่เติมค่าจาก UI โดยไม่มีสถานะว่าเป็นข้อเสนอ

### 7. AI ออกแบบ buckets ได้ holdings ที่มูลค่าและน้ำหนักเป็น 0

- หลักฐาน: snapshot adapter ส่ง market_value_thb/asset_type/bucket_id (`portfolio_planning_adapter.py:63`); service กลับอ่าน value_thb/weight_percent (`bucket_planning_service.py:194`)
- พิสูจน์: holding สังเคราะห์มีมูลค่า 900,000 จาก NAV 1,000,000 แต่ prompt ได้ value_thb='0', weight_percent='0'
- ยังตัดเหลือ 10 holdings โดยไม่มีการบอกว่าตัด ไม่มี asset_type/bucket_id ใน prompt
- แก้: ใช้ DTO เดียวกัน คำนวณ weight จาก NAV ส่ง holdings ที่จำเป็นครบหรือสรุปแบบมี coverage และรักษาประเภทสินทรัพย์/หมวดเดิม

### 8. แผนผ่านเพียงเพราะสัดส่วนรวม 100% ทั้งที่ขัดกับ Axis และ DCA

- หลักฐาน: `bucket_planning_service.py:52/464` ตรวจผลรวม targets เท่านั้น; import validate_allocation_matrix แต่ไม่ได้เรียก; adapter preview ไม่ตรวจ matrix/remapping/rules
- พิสูจน์: buckets 90/10, matrix เก่ายัง 70/30, DCA 90/90 รวม 180 → is_valid=true และ service ส่ง apply ต่อ
- ไม่มี validation จำนวน buckets, unique IDs, references, funding goal linkage, eligible-assets ที่สอดคล้องกับข้อห้ามใน Axis
- แก้: validator เดียวสำหรับ generate/update/preview/apply ตรวจ row/column totals เทียบ confirmed axis, DCA รวม, IDs/reference integrity และกฎที่มีผลต่อการใช้งาน; edits/presets ต้องปรับ matrix/remapping ตาม

### 9. Remapping เดิมอาจล้างการจัดหมวดหรืออ้าง bucket ที่ไม่มีอยู่

- หลักฐาน: `bucket_planning_service.py:237` เทียบชื่อ bucket ใหม่กับ ID เก่า;ไม่ตรงให้ None; builder ไม่มี remapping editor
- พิสูจน์: core_equities → Purpose Buckets ภาษาไทย → target_bucket_id=None; adapter apply ทำ holding.bucket_id=None
- พิสูจน์เพิ่ม: target_bucket_id='missing-bucket' ยัง commit ได้ ทั้งที่ไม่มี target ID นี้
- preset เปลี่ยน IDs ทั้งชุดแต่ส่ง update เฉพาะ purpose_buckets ทำให้ remapping/matrix เก่าอ้าง IDs ที่หายได้
- `tools/portfolio/domain/calculations.py:349` holdings ที่มี bucket_id ซึ่งไม่อยู่ใน targets จะไม่ปรากฏเป็น known bucket หรือ Unassigned
- แก้: preview/แก้ mapping ราย holding หรือหมวดเดิมให้ตรวจได้ บอก Unassigned ชัดเจน ตรวจ orphan IDs ก่อน apply และให้ allocation view รวมเงินได้ครบ NAV

### 10. Apply เก็บแค่ชื่อ/สี/% ไม่เก็บ policy ที่ทำให้ buckets practical

- หลักฐาน: `portfolio_planning_adapter.py:143` สร้าง AllocationTarget ด้วย bucket_id/name/target_percent/color เท่านั้น; accepted_policy_snapshot/source_refs/mapping_weights ไม่ถูกบันทึกใน portfolio state
- พิสูจน์: หลัง apply target ไม่มี role, goal linkage, eligible_assets, cash_inflow_routing_pct, action_rules
- รายละเอียดเหล่านี้ยังอยู่ใน planning draft แต่ไม่ได้อยู่ใน operational portfolio ซึ่งเป็นจุดที่ผู้ใช้ต้องใช้ดูแลพอร์ต และไม่ได้ถูกเชื่อมให้ flow อื่นใช้
- แก้: persist versioned bucket policy/provenance ใน portfolio หรืออ้าง immutable accepted policy แล้ว resolve ให้ UI/การบันทึก/การตรวจพอร์ตทุกจุดอ่านได้ การติดตาม DCA และข้อห้ามเป็นข้อมูลช่วยตัดสินใจ ไม่ควรอ้างว่าเกิดอัตโนมัติถ้ายังไม่มี implementation

### 11. Knowledge publication ใช้ contract ไม่ตรงกับ production broker

- หลักฐาน: dependencies inject KnowledgeWriteBroker; confirmation_service ส่ง dict; `tools/archivist/write_broker.py:145` รับเฉพาะ KnowledgeWriteCommand และปฏิเสธ dict ก่อน I/O
- พิสูจน์โดยใช้ production type guard กับข้อมูลสังเคราะห์: broker ปฏิเสธ แต่ service จับ exception เป็น warning แล้วยังส่ง committed และข้อความ published successfully
- Axis ส่ง dict ในรูปแบบเดียวกัน จึงมีปัญหา contract เดียวกัน
- แก้: adapter ให้ตรง typed command และ payload schema, ตรวจ receipt/status, แยก runtime-confirmed กับ knowledge-published ถ้ายัง publish ไม่สำเร็จ

### 12. Apply commit พอร์ตแล้ว runtime ล้มเหลว ไม่มี recovery ที่เชื่อถือได้

- หลักฐาน: `bucket_planning_service.py:510` เรียก portfolio apply ก่อนบันทึก applied draft/receipt ใน investor runtime; adapter get_apply_receipt คืน None เสมอ
- พิสูจน์ด้วย fault injection: portfolio sequence เพิ่มจาก 1 เป็น 2 แล้ว save applied draft ล้มเหลว → API operation ล้มเหลว; retry ด้วย idempotency key เดิมเจอ checkpoint conflict เพราะไม่มี receipt ให้ replay
- Atomicity ของ targets+holdings ภายในพอร์ตมีอยู่ แต่ไม่ได้ครอบคลุม runtime receipt และ knowledge state
- แก้: durable command intent ก่อน apply, receipt ที่ผูก command/payload อยู่ใน portfolio transaction source และ recovery/reconciliation ที่ดึง receipt เดิมได้เมื่อ runtime save ล้มเหลว

## ข้อบกพร่องเพิ่มเติม (P2)

1. **Draft แสดงเหมือนฉบับใช้งานจริง**: `InvestmentAxisEditor.tsx:68/72` ใช้ axisDraft ก่อน confirmedAxis แต่ badge/read-only อิงการมี confirmedAxis; React พิสูจน์ว่า draft ใหม่ที่ยังไม่ยืนยันแสดงพร้อม badge ใช้งานจริง ทั้งที่ Bucket generator อ่าน confirmed snapshot เก่า
2. **Apply ได้แม้ไม่มี preview/มี validation issues**: `BucketApplyReview.tsx:226` disable เฉพาะ checkbox/loading; React พิสูจน์ทั้งสองกรณี ต้องผูกกับ preview ของ draft revision/payload เดียวกันและผ่าน validation
3. **Q10 summary ล้มเหลวแล้วไม่มีทาง retry ใน UI**: record_answer ให้ is_complete=true ทันที; hook มี summary=null หาก summarize error; Tab เปลี่ยนไป Result แล้วไม่มี action สรุปซ้ำ ควรมี summary_pending/retry/resume โดยไม่ให้ตอบใหม่ทั้งชุด
4. **Provenance เป็น placeholder**: bucket essence_ref/axis_ref ใช้ latest/revision 1/hash='hash', context_ref เป็น ctx_portfolio แทน snapshot ID จริง; ไม่สามารถตรวจแผนเก่าว่ายังสอดคล้องกับ essence/axis/context ปัจจุบัน
5. **Axis content_hash ไม่ครอบคลุม allocation**: `core/investor_essence/investment_axis.py:103` hash จาก complete_sections ซึ่งไม่มี allocation_basis/allocation_rows; พิสูจน์ว่าเปลี่ยน 70/30 เป็น 50/50 แล้ว hash เท่าเดิม
6. **Preview หลังพอร์ตเปลี่ยนไม่บอก conflict ล่วงหน้า**: preview ใช้ checkpoint ปัจจุบันแต่ไม่เทียบ expected checkpoint ของ draft; apply ยังยึด checkpoint เก่า ควร rebase/review ได้โดยไม่ต้อง regenerate AI ทิ้ง edits
7. **DTO readiness ไม่ตรง API**: backend code/severity/message/field_path; frontend ReadinessIssueDTO และ Form ใช้ issue/level/field → ข้อความว่างและ blocking แสดงเป็นแนะนำ
8. **Presets ไม่เป็นส่วนบุคคล**: target 10 ล้านบาท, reserve 300,000, horizon 7–10 ปี, trade loss 2% และ ratios ตายตัว ทั้งที่อาจไม่มี goal นี้; preset ล้าง lineage IDs ต้องสร้างจาก goal/context ของผู้ใช้หรือแสดงชัดว่าเป็นตัวอย่างที่ยังต้องปรับ
9. **Client/server tolerance ไม่เท่ากัน**: UI ยอมประมาณ 0.05 จุดเปอร์เซ็นต์ แต่ backend 0.01 ทำให้ UI บอกครบและ backend reject ได้
10. **Receipt/state ค้างหลังสร้างแผนใหม่หรือ reload**: generate/update ไม่ clear applyReceipt; refresh ไม่ restore applied receipt/status ให้ UI; อาจแสดงสำเร็จของแผนเก่าบนแผนใหม่ หรือ Applied plan กลายเป็น draft ในหน้า
11. **Idempotency fingerprint ไม่รวม revision/payload**: confirm_axis/apply_bucket_plan hash แค่ draft_id; reuse key หลังแก้ draft ยังได้ receipt เก่าแทน conflict ควรผูกกับ payload hash/revision/ref ที่ยืนยัน
12. **Input สำหรับ buckets ยังขาด lineage/context**: generator รองรับ accepted_claims แต่ service ไม่ส่งให้; confirmed_axis_dict ไม่ส่ง allocation_basis ของ Axis และไม่ส่ง financial context เต็มฉบับ ทำให้ source_value_ids และที่มาของตัวเลข/เงื่อนไขยังตรวจย้อนกลับไม่ได้

## ความลึกของข้อมูล: มีโครงที่ดี แต่ยังไม่พอสำหรับคำว่า practical

ด้านที่มีแล้วและควรรักษา: 6 pillars ของ Essence, แยก user-stated/AI-inferred, evidence refs, ให้ผู้ใช้แก้ข้อสรุป, Axis 8 หัวข้อ, และ bucket roles/eligible assets/action rules ข้อมูลเหล่านี้เป็นฐานที่ดี

ช่องว่างที่ต้องเติม:

- **Essence**: ทดสอบ priority เมื่อเป้าหมายชนกัน, เหตุการณ์ drawdown จริงเทียบคำตอบสมมติ, loss ที่รับได้เป็นเงินบาทเทียบ %NAV, เวลาที่ดูแลพอร์ตได้, พฤติกรรมที่ทำต่อเนื่องได้ และข้อจำกัดที่ยอมเปลี่ยนไม่ได้ ถ้าตอบไม่แน่ใจต้องถามเจาะต่อเฉพาะเรื่องก่อนกำหนดตัวเลข
- **Coverage**: backend ยังไม่ได้บังคับว่าครบ 6 pillars จริง หรือ evidence เพียงพอ; summary รับ supporting_question_ids/quotes จาก AI โดยไม่มีการตรวจว่าข้อความ quote ตรงคำตอบและไม่ใช่ unsure/skipped; endpoint clarification ยังไม่มีใน flow ปัจจุบัน อย่าตีความว่า 10 คำถามแปลว่ารู้ครบแล้ว
- **Financial capacity**: ต้องรู้แหล่งเงินและขอบเขตพอร์ต, เงินสำรองนอกพอร์ต, net contribution ต่อเดือน, withdrawal amount/schedule, เงินที่ต้องใช้ตาม deadline, goal currency และกรอบความยืดหยุ่นของ deadline/amount เพื่อบอก funding gap ได้
- **Axis**: แยก desired risk จาก loss ที่สถานะการเงินรองรับ, ระบุฐานและหน่วยให้ตรวจได้, กำหนด concentration/rebalance trigger/action และแยกการลงทุนระยะยาวกับวิธีที่มี per-trade risk ถ้าใช้ต่างบทบาทกัน
- **Goals scope**: adapter/frontend ดึง goals ทุกพอร์ตก่อน fallback พอร์ตปัจจุบัน ควรแสดงขอบเขตว่าใช้เพื่อคำนึงถึงภาพรวม หรือ goal ใดได้รับเงินจากพอร์ตนี้ พร้อม source portfolio; ไม่ควรถือว่าเป้าทั้งหมดต้องถูก fund ซ้ำในแต่ละพอร์ต

## เกณฑ์ของ buckets ที่ practical

Bucket แต่ละใบควรตอบคำถามได้ครบ: เงินนี้ไว้ทำอะไร, ต้องมีเท่าไร, ต้องพร้อมใช้เมื่อไร, ตอนนี้มีเท่าไร, เดือนนี้ต้องเติมเท่าไร, ซื้ออะไรได้, ถอนเมื่อไร/จากไหน, และเมื่อสถานการณ์เปลี่ยนต้องทำอะไร

| ข้อมูล | รูปแบบที่ควรมี | ใช้ทำอะไร |
| --- | --- | --- |
| เป้าหมาย | stable goal ID + funding scope + amount/currency/deadline + priority | ป้องกัน goal ที่เป็นข้อความลอย ๆ หรือ fund ซ้ำระหว่างพอร์ต |
| เป้าเงินใน bucket | amount floor/target และ %NAV ที่คำนวณจากขอบเขตเดียวกัน | reserve/เงินใช้ตามกำหนดไม่ควรมีแต่ % ที่เปลี่ยนตาม NAV |
| เวลา/สภาพคล่อง | need-by date, withdrawal amount/schedule, liquidity constraint | รู้ว่าเงินต้องพร้อมใช้เมื่อไร |
| สถานะจริง | current THB/% + assigned holdings + funding gap | เห็นสิ่งที่ต้องทำ ไม่เห็นแต่ target |
| เงินเข้า | monthly investable cash, routing % และ THB, เมื่อเต็มเป้าให้ส่งต่อที่ไหน | ใช้ DCA ได้จริงและไม่จัดเงินเกินที่มี |
| เงินออก | withdrawal order, amount cap, condition และ refill rule | มีวิธีใช้งานเมื่อเกิดความต้องการเงินจริง |
| สินทรัพย์ | allowed/excluded categories และ mapping กับ Axis/holdings | ตรวจได้ว่าของที่ถือสอดคล้องกับบทบาทของเงิน |
| Rebalance | schedule, threshold พร้อมหน่วย, action, priority | รู้ว่าเมื่อไรต้อง review และจัดเงินเข้า/ออกทางไหน |
| ความเสี่ยง | numeric constraints พร้อม unit/basis/provenance | ตรวจข้อจำกัดได้ตรงความหมาย |
| หลักฐาน | essence/axis/context/goal refs + revisions/hashes + as-of | ทราบว่าแผน stale และควรทบทวนเมื่อไร |

ตัวอย่างผลลัพธ์ที่ product ควรให้ (เป็นสูตรการแสดงข้อมูล ไม่ใช่สัดส่วนแนะนำการลงทุน):

- เป้าหมายใช้เงิน = G, เงินที่จัดไว้แล้ว = C → แสดง gap = max(0, G − C)
- เงินลงทุนใหม่ต่อเดือน = M, route_i = r_i → แสดงเงินเข้า bucket i = M × r_i/100 และรวมไม่เกิน M
- reserve floor มาจากค่าใช้จ่ายจำเป็นและจำนวนเดือนที่ผู้ใช้ยืนยัน พร้อมหัก reserve ที่มีนอกพอร์ตตาม funding scope
- หากไม่มีข้อมูล M/deadline ให้แสดงว่ายังประเมินแผนเติมเงินไม่ได้ แทนสร้างตัวเลขขึ้นเอง
- เป้าหมาย NAV รวมไม่ควรถูกนับเป็นอีกก้อนแยกเพิ่มจาก buckets ที่เป็นส่วนประกอบของ NAV นั้น

โครง bucket สามารถใช้บทบาทเงินที่จำเป็นจริง เช่น เงินต้องพร้อมใช้, เป้าหมายตามกำหนด, เงินสะสมระยะยาว และโอกาสเสริมเมื่อมีความต้องการนั้น จำนวนใบควรตามเป้าหมายและความสามารถดูแล ไม่บังคับ 3–4 ใบหรือชื่อแบบเดียวกันทุกคน

## ลำดับแก้ที่แนะนำ

1. แก้ portfolio state isolation และหยุด dependent steps เมื่อบันทึกล้มเหลว
2. ทำ canonical confirmed snapshots/typed publication ให้ครบ แล้วส่ง adaptive answers/context/holdings ให้ถูกและครบ
3. ทำ shared validator สำหรับ Axis/Buckets/Preview/Apply พร้อมแก้ matrix/remapping ให้สัมพันธ์กับ edits
4. บันทึก policy/provenance/receipts และ recovery ข้าม storage ให้ครบ
5. เปลี่ยน presets ให้ใช้ข้อมูลผู้ใช้ เพิ่ม gap/เงินเข้า/เงินออก/เงื่อนไขใช้งาน และแสดง draft/confirmed/stale ตรงสถานะ
6. ตรวจด้วย integration tests ผ่าน production adapters และทดสอบผ่าน browser จริงก่อนถือว่าพร้อมใช้งาน

## หลักฐานการทดสอบ

- `npm run build`: ผ่าน TypeScript และ production build
- `npm run test -- src/components/portfolio/essence`: 3 files, 13 tests ผ่าน
- `.\.venv\Scripts\python.exe -m pytest tests/unit/investor_essence tests/api/test_investor_essence.py -q`: 45 tests ผ่าน
- เคสจำลอง React แยกจาก normal suite: 7/7 พิสูจน์พฤติกรรมบกพร่องที่ระบุไว้ ไม่ใช่ 7 tests ยืนยันว่าฟีเจอร์ถูกต้อง
- Backend diagnostic: 16 observations โดยใช้ in-memory state/synthetic repo และ production broker type guard; รายละเอียดอยู่ใน `backend_evidence.json`
- Existing API tests ใช้ fake generators/portfolio adapter และ frontend tests ส่วนใหญ่ตรวจ stepper/interview/result จึงไม่ได้พิสูจน์การส่งข้อมูลครบหรือการ apply ผ่าน production adapter

รันเคสจำลอง:

```powershell
# จาก repository root
.\.venv\Scripts\python.exe reports/principles_review_2026-10-10/backend_audit.py

# จาก web/
npm run test -- --config scratch/principles-review/vitest.config.ts
```

ไฟล์ reproduction ฝั่ง React อยู่ที่ `web/scratch/principles-review/` ซึ่งอยู่ภายใต้ gitignored scratch และไม่ได้เพิ่มเข้า normal tests/build ของแอป
