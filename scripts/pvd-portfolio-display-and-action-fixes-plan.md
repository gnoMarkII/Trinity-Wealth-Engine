# แผนแก้การแสดงผลและปุ่มจัดการ PVD — ข้อ 1–3

- วันที่จัดทำและตรวจโค้ด: 2026-10-15
- สถานะ: implement ตามแผนแล้ว; tests/build ผ่าน; Manual QA ยังรอทำ
- ขอบเขต: P/L รายกอง, วันที่และสถานะราคา, ปุ่มแก้ไข/ลบและ batch actions ของ PVD
- อ้างอิง: [แผนนำเข้า PVD](pvd-transaction-import-hexagonal-plan.md), [แผนแยกแท็บ Transactions](portfolio-transactions-assets-pvd-tabs-plan.md)

เอกสารนี้ยึดโค้ดใน working tree ที่ตรวจ ณ วันที่ข้างต้นเป็น baseline รวมโค้ด PVD และ Transactions ที่เพิ่มแล้ว ไม่ใช้ข้อความสถานะเริ่มงานในแผนเก่าเป็นสถานะ implementation ปัจจุบัน

## 1. ปัญหาและผลลัพธ์ที่ต้องการ

| ข้อ | พฤติกรรมปัจจุบัน | หลังแก้ |
| --- | --- | --- |
| 1. P/L รายกอง | เงินต้นรวมถูกจัดสรรตามสัดส่วน NAV ทำให้ P/L และ Avg Cost รายกองเป็นค่าจัดสรร แต่ Holdings และ Analytics แสดงเหมือนต้นทุนและผลตอบแทนรายกองจริง | แสดงมูลค่า หน่วย และ NAV ตาม Statement; พักการแสดง Avg Cost/P/L รายกอง PVD; ใช้ผลประโยชน์รวมตาม Statement อธิบายทั้งพอร์ต |
| 2. วันที่/สถานะราคา | Summary ใช้ `last_updated` เป็นวันที่ราคาพร้อมข้อความ yfinance; status `Statement NAV ... (วันที่)` อาจถูกนับเป็น refresh สำเร็จเพราะมีตัวเลข | แยกเวลาบันทึกพอร์ตจากวันที่ valuation PVD; แยก Statement ออกจากผล refresh ราคาตลาด และมีทางไปอัปเดต Statement |
| 3. ปุ่มจัดการ | แถว PVD ยังมี Edit/Delete และเลือกผ่าน Select all ได้ แม้ API จะปฏิเสธการแก้ไข ลบ หรือย้าย bucket | แถว PVD อ่านอย่างเดียว มีทางดู Statement; selection และ batch actions รับเฉพาะสินทรัพย์ทั่วไปที่จัดการได้ตามพฤติกรรมเดิม |

ตัวอย่างสังเคราะห์สำหรับทำความเข้าใจข้อ 1: เงินต้นรวม 100,000 บาทและ NAV 120,000 บาท เมื่อกระจายเงินต้นตามมูลค่าปัจจุบัน แต่ละกองจะดูเหมือนได้ผลตอบแทนประมาณ 20% เท่ากัน ตัวเลขนี้ใช้จัดอันดับผลงานรายกองไม่ได้ แต่ผลประโยชน์รวม 20,000 บาทยังถูกต้อง

## 2. สิ่งที่ยืนยันจากโค้ดและข้อมูลปัจจุบัน

| จุดตรวจ | หลักฐานและผลต่อแผน |
| --- | --- |
| `tools/portfolio/domain/pvd_calculations.py::allocate_cost_basis()` | จัดสรร total contribution ตาม reconciled market value และปรับเศษสตางค์ เป็นฐานบัญชีเพื่อรักษายอดรวม |
| `tools/portfolio/domain/calculations.py` | คำนวณ `avg_cost_thb` และ `unrealized_pnl_*` รายกองจากต้นทุนจัดสรร; summary รวมใช้ต้นทุนเหล่านี้ด้วย จึงห้ามล้างค่าที่บันทึกเพื่อแก้เฉพาะหน้าจอ |
| `web/src/components/portfolio/PortfolioHoldingsTab.tsx` | ใช้ Avg/PnL เดียวกับหุ้น; sort PnL จากค่าดิบ; checkbox, Unassigned shortcut, Edit/Delete และ Batch Assign ยังไม่แยก PVD |
| `web/src/components/portfolio/PortfolioAnalyticsTab.tsx` | ส่ง `h.unrealized_pnl_percent` ไปที่ treemap เป็น `return_pct` |
| `web/src/components/charts/TreemapChart.tsx` | รองรับ `return_pct` ที่ไม่มีค่าอยู่แล้ว: ไม่แสดง Return tooltip และใช้สีประเภท/สีที่ส่งมา จึงใช้ความสามารถเดิมได้ |
| `web/src/components/portfolio/Modals/PvdStatementModal.tsx` | ระบุว่าต้นทุนเป็นการจัดสรรแล้ว แต่ยังแสดงคอลัมน์กำไร/ขาดทุนรายกอง และใช้คำว่ามูลค่าพอร์ตปัจจุบันกับยอดเดิมในระบบ |
| `web/src/components/portfolio/PortfolioSummaryCards.tsx` | ใช้ `lastUpdated` ใต้ Market Prices Status; `isPriceRefreshSuccess()` มี fallback ตรวจตัวเลข จึงนับ Statement NAV เป็น market success ได้ |
| `tools/portfolio/adapters/composite_price_adapter.py`, `price_yfinance_adapter.py` | ข้าม PVD ใน market provider และคืน status `Statement NAV ...` หรือ `Statement NAV unavailable` ใน map เดียวกับผล refresh |
| `api/schemas/portfolio.py` | มี `pvd_valuation.as_of_date`, `statement_nav`, `cost_basis` และ `pvd_statements` ใน state อยู่แล้ว ใช้ DTO เดิมสำหรับงานนี้ได้ |
| `tools/portfolio/services/trading_service.py`, `portfolio_state_service.py` | มี guard correction/remove และ reserved bucket สำหรับ PVD รวม mixed batch อยู่แล้ว ต้องรักษาไว้ |
| `web/src/pages/Portfolio.tsx` | มี URL handler เปิด Transactions/PVD และ `selectedPortfolioIdRef`; Transactions ใช้ scoped success callback แล้ว แต่ Holdings ยังรับ callback ตรงและยังไม่มี portfolio key |
| ข้อมูลที่ใช้รีวิว | ประวัติจริง 22 รายการแยกเป็น 3 Statement ผ่าน helper หน้า PVD ทั้งหมด ไม่มี asset row ปะปนและไม่มียอดกระทบยอดค้าง ผลนี้เป็นการตรวจข้อมูลที่บันทึก ไม่ใช่ visual QA หรือตรวจต้นฉบับ PDF ใหม่ |

## 3. ขอบเขตและสัญญาที่ต้องรักษา

1. เปลี่ยนการนำเสนอและความสามารถของปุ่ม โดยรักษา NAV, เงินต้นรวม, ผลประโยชน์รวม, units, snapshot ledger และ reconciliation adjustment เดิม
2. `asset_type="PVD"`, prefix `PVD_`, source, archive reference และ import/replay semantics ต้องคงเดิม ห้ามเปลี่ยนเป็น Fund/Stock เพื่อใช้ UI เดิม เพราะจะกระทบ pricing guards
3. API routes, DTOs และ generated contracts ใช้ของเดิม ไม่ต้อง migration, rebuild projection หรือเขียนทับ statement เก่า
4. การเรียก API จากภายนอกยังต้องถูก backend guards ป้องกัน แม้ frontend จะซ่อนปุ่มแล้ว
5. หน้า/คำสั่งของสินทรัพย์ทั่วไปต้องทำงานต่อ รวม trade, correction, delete, bucket assignment, journal และ price refresh
6. การแยกข้อมูลสำหรับ bucket/rebalance รวมถึงการปลด `retirement_pvd` เป็นงานถัดไป งานนี้ยังรักษา reserved bucket และไม่แก้ target percentages
7. ใช้ข้อมูลสังเคราะห์ใน tests ไม่เพิ่มข้อมูลสมาชิก PDF จริง หรือ snapshot การเงินจริงลง repository
8. ไม่คำนวณ TWR/IRR หรือ annualized return จาก balance snapshots และไม่สร้างประวัติราคาย้อนหลังจากวันที่นำเข้า

## 4. Helper กลางสำหรับการแสดงผล

เพิ่ม `web/src/lib/pvdPresentation.ts` เป็น pure helper และ `pvdPresentation.test.ts` เพื่อใช้ policy เดียวกันใน Holdings, Summary, Analytics และ modal guards

### 4.1 การระบุ PVD และความสามารถของแถว

- `isPvdHolding(holding)` เป็นจริงเมื่อ asset type หลัง trim/normalize เป็น PVD หรือ symbol ขึ้นต้น `PVD_` หรือมี `pvd_valuation`
- ถ้าสัญญาณไม่ตรงกัน เช่น asset type เป็น Fund แต่ symbol เป็น `PVD_*` ให้ใช้การแสดงผล PVD แบบอ่านอย่างเดียว ไม่เปิด mutation จากข้อมูลที่ไม่ครบ
- อย่าเปลี่ยนค่าของ holding/DTO ต้นฉบับเพื่อทำให้ช่อง P/L หายไป การเลือกว่าจะ render/sort อะไรเป็นหน้าที่ presentation helper/component
- ถือว่า PVD ทุกกองใน contract ปัจจุบันยังไม่มีต้นทุนซื้อจริงรายกอง แม้ DTO จะมีตัวเลข `cost_basis` และ `unrealized_pnl_*`

### 4.2 Context วันที่ valuation

ให้ `getPvdValuationContext(holdings, statements)` คืนข้อมูลอย่างน้อย:

| ค่า | ความหมาย |
| --- | --- |
| `hasPvd` | มี holding PVD หรือมี metadata Statement ใน state |
| `hasMarketHoldings` | มี holding สินทรัพย์ทั่วไปที่เข้าราคา refresh ตาม policy เดิม: ไม่ใช่ PVD/Cash, ไม่ archived และ units มากกว่า 0 |
| `hasNonPvdValue` | มีมูลค่าของ holding ที่ไม่ใช่ PVD รวม Cash ใช้เลือกคำอธิบายเปอร์เซ็นต์รวม ไม่ใช้ตัดสินว่ามี market holding ที่ refresh ได้ |
| `dates` | วันที่ `pvd_valuation.as_of_date` ที่ตรวจเป็นวันจริงแล้ว เรียงและไม่ซ้ำ ของ PVD ที่ยังมี units หรือมูลค่า ไม่เอากอง archived มูลค่าศูนย์มาทำให้วันที่รวมดูเก่า |
| `hasUnknownDate` | PVD ที่มี units/มูลค่าบางรายการไม่มี valuation date ที่ตรวจสอบได้ |
| `latestRecordedStatementDate` | วันล่าสุดที่ตรวจสอบได้จาก state metadata ใช้บอกประวัติ Statement ที่บันทึกไว้ ไม่ใช้เดาวัน valuation ของ holding ที่ข้อมูลขาด |

- วันเดียวและไม่มี unknown → แสดง `PVD ตาม Statement ณ …`
- หลายวัน → แสดงว่ามีหลายวันที่พร้อมช่วงวันเก่าสุด–ใหม่สุด ห้ามใช้แค่วันที่ใหม่สุดทำให้ทั้งยอดดูสดเท่ากัน
- บางรายการไม่ทราบวัน → แสดง `ข้อมูลวันที่ PVD บางรายการไม่ครบ` พร้อมวันที่ที่ทราบได้
- PVD เป็นศูนย์ทั้งหมด → แสดง `Statement ล่าสุดที่บันทึก …` เมื่อ metadata มีวัน valid; ถ้าไม่มีให้แสดงไม่ทราบวันที่
- ตรวจ ISO date เป็นวันที่จริง ไม่ยอมรับวันที่เช่น 2026-02-30; แสดง date-only โดยไม่ให้ timezone เปลี่ยนวัน
- งานนี้แสดงวันที่ชัดเจนก่อน ไม่กำหนดเกณฑ์เตือน “เก่าเกิน N วัน” แบบตายตัว เพราะความถี่ Statement ยังไม่ได้กำหนดเป็น contract

### 4.3 การแยกผล refresh

ให้ `partitionPriceRefreshInfo(info, holdings)` แยก `marketEntries` และ `pvdEntries` ก่อนคำนวณ success/failure:

- รู้จัก PVD จาก symbol ที่ตรงกับ holding PVD, prefix `PVD_` หรือ status เริ่มด้วย `Statement NAV`
- `Statement NAV … (วันที่)` และ `Statement NAV unavailable` เป็นข้อมูล Statement ทั้งคู่ ไม่ใช่ market success หรือ market failure
- `USDTHB` และผลของสินทรัพย์ทั่วไปยังอยู่ฝั่ง market และใช้เกณฑ์ success/failure เดิม
- ห้ามอ่านเลขวันที่จาก status มาแทน typed valuation date หรือใช้ข้อความที่มีตัวเลขเป็นหลักฐานว่า PVD ถูก refresh จากตลาดแล้ว

## 5. ข้อ 1 — แสดงผลประโยชน์ PVD อย่างถูกความหมาย

### 5.1 Holdings

- แถว PVD ยังแสดง symbol, units, NAV ตาม Statement, market value และ bucket ได้
- เปลี่ยนช่อง Avg Cost เป็น `—` พร้อมคำอธิบายที่เข้าถึงได้ว่าไม่มีต้นทุนซื้อจริงรายกองจากข้อมูลที่นำเข้า
- เปลี่ยนทั้ง PnL เปอร์เซ็นต์และจำนวนเงินของ PVD เป็น `—` พร้อมข้อความ `ดูผลประโยชน์รวมตาม Statement` ไม่แสดง 0% แทนข้อมูลที่ไม่มี
- เมื่อ sort ตาม PnL ให้ PVD อยู่ในกลุ่มข้อมูลไม่พร้อมตามรูปแบบ missing-value ของตาราง ไม่ใช้เปอร์เซ็นต์จัดสรรจัดอันดับผลงาน
- มีคำอธิบายสั้นหนึ่งครั้งในตารางเมื่อมี PVD: `Statement นี้มีเงินต้นรวม จึงยังแสดงต้นทุนและผลตอบแทนจริงรายกองไม่ได้`
- Journal และประวัติ Statement ยังใช้งานได้ตามข้อ 7

### 5.2 Analytics / Treemap

- สำหรับ PVD ส่ง `return_pct: undefined`; สีต้องเป็นสีหมวด/สีกลางคงที่ ไม่สื่อกำไรหรือขาดทุน
- ยังคงขนาดพื้นที่ตาม market value และ share ของ NAV ได้
- tooltip ของ PVD แสดงมูลค่าและสัดส่วน โดยไม่มีตัวเลข Return/PnL สังเคราะห์หรือ +0.00%
- สินทรัพย์ทั่วไปยังส่งข้อมูลและแสดงผลตาม contract เดิม ไม่แก้การคำนวณ daily performance หรือหน่วย return ของ chart ทั่วไปในงานนี้
- ใช้การรองรับ optional return ของ `TreemapChart` ที่มีอยู่ก่อน ไม่ขยาย shared chart API หากไม่จำเป็น

### 5.3 Modal นำเข้า Statement

- คงคอลัมน์ `เงินต้นจัดสรร` เพื่อกระทบยอด พร้อมข้อความว่าค่านี้จัดสรรจากเงินต้นรวมตามมูลค่า ไม่ใช่ต้นทุนซื้อจริงรายกอง
- เอาคอลัมน์ `กำไร/ขาดทุน` รายกองออกจาก preview
- เพิ่มเงินต้นสะสมและผลประโยชน์รวมในส่วน summary ของ Statement โดยคำนวณจาก decimal strings (`total_nav - total_contribution`) ผ่าน money helper เดิม ไม่ทำ arithmetic ด้วย floating point
- ถ้าค่า summary ที่จำเป็นไม่มี/ไม่ถูกต้อง ให้แสดงไม่พร้อมคำนวณ ไม่ fallback เป็นศูนย์
- แก้ข้อความ `มูลค่าพอร์ตปัจจุบัน` เป็น `ยอด PVD ล่าสุดที่บันทึกในระบบ`; ส่วนต่าง NAV ใช้ชื่อ `ยอดเปลี่ยนจากที่บันทึก` ไม่เรียกผลตอบแทน เพราะรวมผลของเงินสมทบและการสับเปลี่ยนได้
- analyze/commit, vesting, idempotency, PDF reference และการส่ง state กลับคง workflow เดิม

### 5.4 ยอดรวมบน Summary

- คงสูตรและตัวเลข aggregate จาก state เดิม ไม่รวม metadata ซ้ำหลายรอบตามจำนวนแถวกองทุน
- พอร์ตมี PVD และไม่มีมูลค่าจาก holding อื่นรวม Cash → ใช้ชื่อ `ผลประโยชน์สะสมตาม Statement` และคำอธิบายเปอร์เซ็นต์ว่าเทียบเงินต้นสะสม
- พอร์ตผสม รวมกรณี PVD กับ Cash ที่มียอด → ใช้ชื่อกำไร/ผลประโยชน์สะสมรวม และอธิบายเปอร์เซ็นต์ว่าเทียบต้นทุนรวมพอร์ตตามสูตรเดิม โดยต้นทุนรวมของ backend นับ Cash อยู่ด้วย ห้ามเรียกเปอร์เซ็นต์นี้ว่าอัตราผลประโยชน์ PVD เทียบเงินต้น PVD อย่างเดียว
- ถ้า PVD บาง holding ไม่มี valuation metadata ให้แสดงข้อมูลไม่ครบและไม่อ้างว่า aggregate P/L ทั้งยอดเป็นผลประโยชน์ Statement ที่ยืนยันแล้ว
- อัตราส่วนทั้งสองกรณีไม่ใช่ผลตอบแทนรายปี/TWR/IRR
- พอร์ตไม่มี PVD → รักษาข้อความและพฤติกรรม P/L เดิม

## 6. ข้อ 2 — แยกวันบันทึก วัน Statement และสถานะราคาตลาด

### 6.1 Holdings และ Summary

- แถว PVD ระบุ `NAV ตาม Statement` และวันที่จาก holding valuation ใกล้ราคา/มูลค่า; ถ้าไม่มีให้แสดง `ไม่ทราบวันที่ Statement`
- Summary รับ valuation context เพิ่มจาก `Portfolio.tsx` ผ่าน helper กลาง ไม่อ่าน/parse raw transaction ledger ซ้ำ
- เปลี่ยนความหมายของ `lastUpdated` ที่แสดงเป็น `บันทึกพอร์ตล่าสุด` เพราะ backend ไม่ได้ให้ timestamp ของการ refresh ราคาตลาดโดยเฉพาะ
- แสดง PVD valuation date/ช่วงวันแยกจาก lastUpdated และแก้คำบรรยาย NAV ให้บอกว่าเป็นยอดล่าสุดที่บันทึก พร้อมวันที่ PVD ที่เกี่ยวข้อง
- เลิกเหมารวมทุกแหล่งราคาว่าเป็น yfinance: ผลราคาสินทรัพย์ทั่วไปให้ใช้รายละเอียดจาก refresh status ที่มี ส่วน PVD อ้าง Statement
- การ import วันนี้แต่ Statement เป็นงวดก่อน ต้องแสดงทั้งสองวันโดยไม่สื่อว่ายอด PVD เป็นราคาวันนี้

### 6.2 ปุ่มและผล refresh ตามชนิดข้อมูล

| ข้อมูลในพอร์ต | พฤติกรรม |
| --- | --- |
| สินทรัพย์ทั่วไป ไม่มี PVD | ปุ่ม market refresh และ success/failure คงการทำงานเดิม; เวลาบันทึกไม่อ้างว่าเป็นเวลา provider ส่งราคา |
| มีเฉพาะ PVD และ Cash/ไม่มี market holding ที่ refresh ได้ | แสดงปุ่ม `ดู/อัปเดต Statement PVD` เป็นทางหลัก; ไม่มี market refresh ที่สื่อว่าจะทำให้ราคา PVD ใหม่ขึ้น |
| มีทั้ง PVD และ market holdings | มี market refresh สำหรับสินทรัพย์ทั่วไป และทางไป Statement แยกกัน; อธิบายสั้นว่า refresh ราคาตลาดไม่เปลี่ยนยอด PVD |

- ตัวนับ market refresh ใช้เฉพาะ `marketEntries`; รายการ Statement ไม่เพิ่มยอดสำเร็จและไม่ลดอัตราสำเร็จ
- ถ้าไม่มี market entry เลย ให้แสดงสถานะ Statement ตามข้อมูลที่มี ไม่แสดง `อัปเดตราคาสำเร็จ (0 รายการ)` หรือผลสำเร็จจาก PVD ล้วน
- PVD status ที่ unavailable แสดงข้อมูลไม่ครบในส่วน Statement ไม่ใช้เป็นเหตุแจ้ง market provider ล้มเหลว
- กรณี mixed refresh มีหุ้นสำเร็จและหุ้นล้มเหลว ยังคงรายงาน partial failure จริง หลังตัด PVD entries ออก
- ไม่เปลี่ยน price adapters หรือเปิด public market lookup ของ PVD เพื่อตอบสนองปุ่มใหม่

### 6.3 ทางไปอัปเดต Statement

- เพิ่ม callback `onViewPvdStatements` ให้ Summary และใช้ `handleTransactionViewChange('pvd')` ที่มีอยู่ใน `Portfolio.tsx`
- URL ต้องเป็น `tab=transactions&tx_view=pvd` ในพอร์ตเดิม; รักษา portfolio query และจัดการ symbol ที่ขัดกับแท็บตาม handler เดิม
- ใช้ปุ่มนำเข้าใน `PvdStatementsTable` ต่อ ไม่สร้าง modal นำเข้าชุดที่สองที่มี state แยกจาก workflow เดิม
- ทดสอบ browser Back/Forward และการเปลี่ยนพอร์ตให้ context วันที่/ปุ่มเปลี่ยนตามพอร์ต ไม่ค้างจากพอร์ตเดิม

## 7. ข้อ 3 — ป้องกันการจัดการแถว PVD ผ่านเครื่องมือทั่วไป

### 7.1 แถวและ selection

- ซ่อน Edit/Delete ของแถว PVD และป้องกัน handler ก่อนเรียก API แม้เกิด stale selection
- checkbox รายกอง PVD disabled พร้อม accessible explanation: `ยอด PVD อัปเดตจาก Statement จึงแก้ไขหรือลบรายกองไม่ได้`
- ปุ่มประวัติของ PVD ใช้คำว่า `Statement` และเรียก `onViewTransactions(symbol)` เดิม; สินทรัพย์ทั่วไปยังใช้ข้อความ/route เดิม
- Journal ยังคงใช้ได้ เพราะไม่ใช่การแก้ units/price ของ snapshot
- Select all ใช้เฉพาะ eligible holdings ที่มองเห็นหลังกรอง โดยตัด PVD ออก; checked/indeterminate และจำนวน Selected ต้องตรงกับแถวที่เลือกได้
- หากไม่มี eligible holding ให้ disable Select all; การมี Cash ในพอร์ตยังใช้ความสามารถเดิมของ Cash ไม่ขยายการแก้ cash semantics ในงานนี้
- Unassigned shortcut ต้องไม่เลือก PVD ที่ bucket หายไปเข้าสู่ Batch Assign; แสดงสถานะอ่านอย่างเดียวของ PVD ที่ข้อมูลไม่ครบแทน

### 7.2 Batch actions และ modal guard

- ก่อน Delete/Batch Assign ตรวจทุก selected symbol เทียบกับ holdings ปัจจุบัน: ต้องมีอยู่ในพอร์ตและไม่ใช่ PVD
- ถ้าพบ selection ที่ไม่ผ่าน ให้หยุดทั้งคำสั่งก่อน confirm/API พร้อมสถานะที่ผู้ใช้เห็นได้ ห้ามตัด PVD ทิ้งเงียบ ๆ แล้วลบรายการอื่นต่อโดยที่ผู้ใช้ยังเห็น selection เดิม
- เมื่อ holdings เปลี่ยน ให้ปรับ selection และปิด correction/assignment modal ที่เป้าหมายหายไปหรือกลายเป็น PVD
- `HoldingCorrectionModal` ต้องตรวจ PVD เองด้วย หากถูก render ด้วย PVD จาก caller เก่าหรือ stale props ให้แสดงข้อมูลอ่านอย่างเดียว และไม่ส่ง `editHolding`
- `BatchAssignBucketModal` ต้องป้องกัน `PVD_*` ก่อน submit เป็นชั้นสำรอง ส่วน parent ต้องตรวจ asset type/valuation ด้วยเพื่อครอบคลุม symbol ที่ไม่ใช่ prefix มาตรฐาน
- ยังคง backend guards สำหรับ single/batch correction/remove/assignment และ mixed batch; ไม่เพิ่มปุ่ม reset เพื่อทดแทน Delete ในแผนนี้

### 7.3 เปลี่ยนพอร์ตระหว่างจัดการรายการ

- เพิ่ม `key={selectedPortfolioId}` ให้ `PortfolioHoldingsTab` ในหน้า Portfolio เพื่อรีเซ็ต selection และ modal state เมื่อเปลี่ยนพอร์ต
- ส่ง scoped `onSuccess` ให้ Holdings ด้วย `selectedPortfolioIdRef` ตามรูปแบบที่ Transactions ใช้อยู่ ป้องกัน mutation ของพอร์ตเก่าที่ตอบช้ามาเขียนทับหน้าพอร์ตใหม่
- รักษาการส่ง portfolio ID เดิมของ request และไม่ให้ callback เก่าเปลี่ยน context ของพอร์ตที่เปิดอยู่

## 8. รายการไฟล์และ ownership

| ไฟล์ | งาน |
| --- | --- |
| เพิ่ม `web/src/lib/pvdPresentation.ts` และ `.test.ts` | PVD classification, valuation-date context และ refresh-status partition; pure functions |
| แก้ `web/src/components/portfolio/PortfolioHoldingsTab.tsx` และ `.test.tsx` | ซ่อน Avg/PnL สังเคราะห์, วันที่, read-only actions, selection/batch guards |
| แก้ `web/src/components/portfolio/PortfolioAnalyticsTab.tsx` และ `.test.tsx` | ไม่ส่ง PVD return และตรวจ tooltip/สี/มูลค่า |
| แก้ `web/src/components/portfolio/PortfolioSummaryCards.tsx` และ `.test.tsx` | Aggregate wording, state timestamp, valuation context, ปุ่ม Statement และ refresh counts |
| แก้ `web/src/components/portfolio/Modals/PvdStatementModal.tsx` และ `.test.tsx` | ชื่อเงินต้นจัดสรร, ตัด P/L รายกอง, summary รวมและวันที่/ส่วนต่าง |
| แก้ `web/src/components/portfolio/Modals/HoldingCorrectionModal.tsx`; เพิ่ม/ขยาย test ของ modal | Guard เมื่อได้รับ PVD โดยตรง |
| แก้ `web/src/components/portfolio/Modals/BatchAssignBucketModal.tsx`; เพิ่ม/ขยาย test ของ modal | Guard symbol PVD ก่อนเรียก API |
| แก้ `web/src/pages/Portfolio.tsx` และ `.test.tsx` | ส่ง context, Statement navigation, Holdings key และ scoped callback |
| ใช้ `web/src/components/charts/TreemapChart.tsx` เดิม | ไม่เปลี่ยน shared chart contract; ทดสอบการ render PVD ผ่าน Analytics |
| ใช้ backend domain/service/adapters เดิม | รักษาสูตรรวมและ snapshot guards; รัน regression ที่เกี่ยวข้อง |

## 9. ลำดับ implementation และเกณฑ์ผ่านแต่ละระยะ

| ระยะ | งาน | เกณฑ์ผ่าน |
| --- | --- | --- |
| 1 | เพิ่ม helper และ synthetic fixtures | ระบุ PVD แบบ type/prefix/valuation ได้; แยกวันจริง/หลายวัน/unknown และ market/Statement status ได้ |
| 2 | แก้ข้อ 1 ใน Holdings, Analytics, modal และ aggregate copy | ไม่มี PVD Avg/PnL รายกองที่อ้างเป็นผลจริง; NAV และยอดรวมเดิม; ค่า unknown ไม่กลายเป็นศูนย์ |
| 3 | แก้ข้อ 2 ใน Summary/Holdings และเชื่อมทางไป Statement | วัน import ไม่กลายเป็นวัน valuation; PVD ไม่ปน market success/failure; route และ mixed refresh ถูกต้อง |
| 4 | แก้ข้อ 3 ในแถว, selection, modal และ portfolio scope | ไม่มี PVD mutation จาก UI; batch สินทรัพย์ทั่วไปยังใช้ได้; stale request ไม่ทับอีกพอร์ต |
| 5 | รัน regression/build และ manual QA | ผ่านเงื่อนไขหัวข้อ 10–11; บันทึกผลจริงแยกจากรายการที่ยังไม่ได้ทดสอบ |

## 10. แผนทดสอบ

### 10.1 Automated cases ที่ต้องมี

| กลุ่ม | กรณีสำคัญ |
| --- | --- |
| Classification | PVD ตาม asset type, prefix แบบตัวเล็ก/มี whitespace, valuation ที่มีอยู่; สินทรัพย์ทั่วไปไม่ถูกปิดความสามารถผิดตัว |
| P/L และ Avg | PVD มีตัวเลขจาก backend ก็ยังไม่แสดงเป็นผลจริง; หุ้นข้างเคียงยังแสดงค่าเดิม; sort ไม่จัดอันดับ PVD จาก synthetic return; PVD กับ Cash ที่มียอดยังอธิบาย denominator รวม Cash ถูกต้อง |
| Treemap | NAV/ขนาด/สัดส่วนเดิม; PVD ไม่มี Return tooltip และสีไม่อิงกำไร; ไม่มี fallback +0%; regular holdings ไม่ถูกลบออก |
| Preview | allocated principal มีคำอธิบาย; ไม่มี P/L รายกอง; ผลประโยชน์รวมคำนวณจาก decimal strings; metadata ขาดไม่กลายเป็นยอดศูนย์ |
| วันที่ | วันบันทึกใหม่กว่า Statement, หลาย valuation dates, missing metadata, invalid ISO day, archived zero rows และ zero-only PVD |
| Refresh | PVD-only ไม่มี market-success claim; mixed หุ้นสำเร็จ/ล้มเหลวนับถูก; `Statement NAV unavailable` ไม่เป็น market failure; FX ยังนับตามเดิม |
| Selection | เลือก PVD ไม่ได้; Select all ใน mixed table เลือกเฉพาะ eligible; filter แล้ว count/checked/indeterminate ถูก; Unassigned PVD ไม่เข้า batch |
| Mutation | Edit/Delete PVD ไม่มีปุ่ม/API call; direct correction/batch modal ไม่ submit PVD; invalid/stale mixed selection ไม่ลบสินทรัพย์อื่นบางส่วน |
| Portfolio scope | เปลี่ยนพอร์ตแล้ว selection/modal/context รีเซ็ต; mutation สินทรัพย์ทั่วไปที่ตอบช้าหลังเปลี่ยนพอร์ตไม่ทับ state ใหม่ |
| Navigation | ปุ่ม Statement จาก Summary และแถว PVD เปิดพอร์ตเดิม/แท็บ PVD ถูก; Back/Forward ทำงาน |
| Backend regression | สูตรจัดสรร/ผลรวมเดิม; PVD ไม่ถึง market providers; mixed mutation guards ไม่เปลี่ยน state/ledger/checkpoint หลัง reject |

ใช้ fixture PVD ที่ backend ยังมี synthetic P/L เป็นบวก/ลบ เพื่อพิสูจน์ว่า UI ไม่เผลอ render ค่าเหล่านี้ ไม่ใช้ fixture ที่ล้าง P/L ให้เป็น null อยู่แล้วเพียงอย่างเดียว

### 10.2 คำสั่งตรวจหลัง implementation

จาก `web`:

```powershell
npm run test -- src/components/portfolio src/lib/pvdPresentation.test.ts src/lib/pvdStatementHistory.test.ts src/pages/Portfolio.test.tsx
npm run build
```

จาก root:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/portfolio/test_pvd_calculations.py tests/unit/portfolio/test_pvd_price_bypass.py tests/unit/portfolio/test_pvd_mutation_guards.py tests/integration/test_pvd_atomic_recovery.py -q -o addopts=''
git diff --check
```

ถ้า implementation จำเป็นต้องเปลี่ยน API contract จริง ให้ทบทวนขอบเขตก่อน regenerate types และรัน `npm run check:types`/API contract tests ตาม workflow เดิม ไม่แก้ generated types ด้วยมือ งานตามแบบในเอกสารนี้ไม่จำเป็นต้องเปลี่ยน DTO

### 10.3 Manual QA

- ใช้พอร์ตทดสอบแยก: สินทรัพย์ทั่วไป, PVD พร้อม Cash เริ่มต้น, และ mixed portfolio
- ตรวจ Desktop และหน้าจอ 320/360px ใน light/dark: วันที่/คำอธิบายไม่ถูกตัดจนความหมายหาย, disabled controls เข้าใจได้, ตารางยัง scroll และ modal ใช้งานได้
- เปิด Holdings → Statement → import preview → commit งวดใหม่ในพอร์ตทดสอบ; วัน valuation เปลี่ยนตามเอกสาร และยอดรวมตรง Statement
- กด market refresh ใน mixed portfolio; PVD ยังใช้ Statement เดิมและไม่ถูกนับเป็น market-refresh success
- ลอง Select all และ batch action ของสินทรัพย์ทั่วไป; PVD ไม่ถูกเลือกหรือส่งไป API
- เปลี่ยนพอร์ตขณะ modal เปิดและขณะ request สินทรัพย์ทั่วไปยังไม่ตอบ; ตรวจว่า state ใหม่ไม่ถูกทับ
- ข้อมูลจริงที่ใช้ตรวจย้อนหลังเป็น read-only spot check; การทดสอบ mutation/import ใหม่ใช้พอร์ตทดสอบ และไม่แก้หรือลบข้อมูลพอร์ตผู้ใช้เพื่อพิสูจน์ guard

## 11. Checklist ก่อนปิดงาน

- [x] ไม่มี Avg Cost/PnL สังเคราะห์ของ PVD ที่แสดงเป็นผลรายกองจริงใน Holdings, Analytics และ import preview
- [x] เงินต้น/NAV/ผลประโยชน์รวมเดิมและการกระทบยอดคงเดิม; aggregate copy ไม่อ้างผลตอบแทนรายปี
- [x] เห็นวัน Statement ของยอด PVD ชัดเจน แยกจากวันบันทึกพอร์ต รวมกรณีข้อมูลวันที่ไม่ครบ
- [x] market refresh counts และ failure details ไม่รวม PVD Statement entries
- [x] ปุ่ม Statement ใช้ workflow/portfolio scope เดิม และ market refresh ของสินทรัพย์ทั่วไปยังทำงาน
- [x] PVD ไม่เข้า correction/delete/batch assignment จาก UI; guard ฝั่ง backend ยังคงอยู่
- [x] Select all, Unassigned shortcut, stale selection และ direct modal props ไม่เปิดช่องส่ง mutation PVD
- [x] เปลี่ยนพอร์ตแล้ว selection/modal/context ไม่ค้าง และ callback ของพอร์ตเก่าไม่ทับพอร์ตใหม่
- [x] Automated tests และ build ผ่าน พร้อมบันทึกคำสั่ง/ผลจริง
- [ ] Manual QA ที่ทำแล้วมีบันทึกผล; รายการที่ยังทำไม่ได้ระบุข้อจำกัดตามจริง
- [x] ไม่มีการเปลี่ยน bucket targets, snapshot ledger, API contract หรือข้อมูลพอร์ตจริงจากงานนี้

## 12. สถานะการดำเนินงาน

- 2026-10-15: implement helper แสดงผล PVD, Holdings/Summary/Analytics/Statement UI, refresh classification, PVD read-only guards และ callback scope ตามแผน; ไม่เปลี่ยน API contract, backend formulas, bucket targets, ledger หรือข้อมูลพอร์ตจริง
- Automated: `npm run test -- src/components/portfolio src/lib/pvdPresentation.test.ts src/lib/pvdStatementHistory.test.ts src/pages/Portfolio.test.tsx` — 26 test files, 128 tests passed
- Build: `npm run build` — passed
- Backend regression: `.\.venv\Scripts\python.exe -m pytest tests/unit/portfolio/test_pvd_calculations.py tests/unit/portfolio/test_pvd_price_bypass.py tests/unit/portfolio/test_pvd_mutation_guards.py tests/integration/test_pvd_atomic_recovery.py -q -o addopts=''` — 11 passed
- `git diff --check` — passed; มีเพียงคำเตือน line endings ของไฟล์เดิมใน working tree
- Manual QA ใน browser ยังไม่ได้ทำ; automated UI tests ใช้ข้อมูลสังเคราะห์ และยังต้องตรวจขนาดจอ/light-dark กับ workflow PDF ในพอร์ตทดสอบ
