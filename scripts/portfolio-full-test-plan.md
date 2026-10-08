# แผนงานทดสอบหน้า Portfolio ทั้งหมด

วันที่จัดทำ: 5 ตุลาคม 2026  
สถานะ: ดำเนินการและผ่านการตรวจสอบครบถ้วน (Completed & Verified) — Backend 388+ tests ผ่าน 100%, Frontend 383 Vitest tests ผ่าน 100%, TypeScript build ผ่าน  
ขอบเขตอ้างอิง: โค้ดใน working tree ปัจจุบัน ครอบคลุมการแก้ไขความเสี่ยงวิกฤต (P0), ADR-011 Vault V2, และ Hexagonal Architecture Parity

แผนนี้มี 140 scenarios และ 8 end-to-end workflows; scenario ที่ระบุหลาย provider/ค่า input ต้องแตกเป็น test variants เพิ่มเมื่อดำเนินการ

## 1. เป้าหมายและขอบเขต

ตรวจว่าผู้ใช้ใช้งาน `/portfolio` ได้ครบทุกฟังก์ชัน ข้อมูลถูกบันทึกและกลับมาเหมือนเดิมหลัง reload ตัวเลขบนแต่ละส่วนสอดคล้องกัน และการทำรายการไม่กระทบพอร์ตอื่น โดยตรวจตั้งแต่ UI → API → service → ledger/runtime → projection

| ส่วนของหน้า | สิ่งที่ต้องตรวจ | โค้ดอ้างอิง |
|---|---|---|
| Page shell / URL / loading | เปิดหน้า, session, URL parameters, เปลี่ยนแท็บ, retry, response ที่มาช้า | [Portfolio.tsx](../web/src/pages/Portfolio.tsx) |
| Multi-portfolio | รายการพอร์ต, สร้าง, เปลี่ยนชื่อ, สลับ, ลบ, default portfolio | [router_state.py](../api/routers/portfolio/router_state.py) |
| Summary cards | NAV, cost basis, unrealized/realized P&L, income, dividends, sparkline, refresh ราคา/FX | [PortfolioSummaryCards.tsx](../web/src/components/portfolio/PortfolioSummaryCards.tsx) |
| Executive Dashboard | Goals รวมทุกพอร์ต, Performance ของพอร์ตที่เลือก, ย่อ/ขยาย | [PortfolioGoalsTab.tsx](../web/src/components/portfolio/PortfolioGoalsTab.tsx), [PortfolioAnalyticsTab.tsx](../web/src/components/portfolio/PortfolioAnalyticsTab.tsx) |
| Strategy Buckets & Allocation | Target, actual, variance, chart, เพิ่ม/แก้/ลบ bucket, สี, drill-down | [PortfolioOverviewTab.tsx](../web/src/components/portfolio/PortfolioOverviewTab.tsx) |
| Holdings & Trading Journal | ตาราง, sort, bucket filter, selection, จัดกลุ่ม/ลบหลายรายการ, correction, journal | [PortfolioHoldingsTab.tsx](../web/src/components/portfolio/PortfolioHoldingsTab.tsx) |
| Transactions | ค้นหา, BUY/SELL, sort, pagination, Native/THB, note, edit, void | [PortfolioTransactionsTab.tsx](../web/src/components/portfolio/PortfolioTransactionsTab.tsx) |
| Dividends & Income | Received/Upcoming, source filter, sync, dividend rounds, manual income | [PortfolioIncomesTab.tsx](../web/src/components/portfolio/PortfolioIncomesTab.tsx) |
| Watchlist | เพิ่ม/แก้/ลบ, ราคาเป้าหมาย, notes | [PortfolioWatchlistTab.tsx](../web/src/components/portfolio/PortfolioWatchlistTab.tsx) |
| Corporate Calendar | Earnings/Ex-dividend, เดือน, Upcoming, holding/watchlist, partial failure | [PortfolioCalendarTab.tsx](../web/src/components/portfolio/PortfolioCalendarTab.tsx) |
| Trade / Cash / Income | THB/USD, FX, วันที่ย้อนหลัง, validation, cash balance, quick top-up | [Modals](../web/src/components/portfolio/Modals) |
| Broker import | Dime, WealthX, SCBAM; scan, staging, selection, commit, stream, deduplication, PDF inspection | [DimeSyncModal.tsx](../web/src/components/portfolio/Modals/DimeSyncModal.tsx) |
| Clean Slate Reset | ข้อความยืนยัน, backup, ขอบเขตข้อมูลที่ล้าง/คงไว้, recovery | [ResetConfirmModal.tsx](../web/src/components/portfolio/Modals/ResetConfirmModal.tsx) |
| Data integrity | replay, atomic mutation, concurrency, isolation, projection rebuild | [ADR-011](../docs/adr/ADR-011-portfolio-projection-boundary.md), [runbook](../docs/runbooks/portfolio-projection-rebuild.md) |

ไม่รวมการทดสอบฟังก์ชันของหน้า Macro/Equity ทั้งหมด ให้ตรวจเฉพาะจุดเชื่อมโยงที่ปรากฏบน Portfolio และ shared components ที่เกี่ยวข้อง

ข้อกำหนดที่ต้องรักษา:

- มี 6 แท็บหลัก: `overview`, `holdings`, `transactions`, `incomes`, `watchlist`, `calendar`
- Goals และ Performance อยู่ใน Executive Dashboard; URL เดิม `tab=goals` และ `tab=analytics` ถูกเปลี่ยนเป็น `overview`
- Goals แสดงทุกพอร์ตรวมกันและมี badge ระบุพอร์ต แต่ค่าความคืบหน้าต้องคำนวณจากพอร์ตที่ goal ผูกอยู่
- Holdings, transactions, allocations, watchlist, calendar และ performance ต้องใช้ `portfolio_id` ที่เลือก
- ใช้ runtime/ledger ตรวจตัวเลขตาม ADR-011; Markdown ที่สร้างจากระบบเป็น projection ไม่ใช่คำสั่งแก้รายการซื้อขาย
- ระบบปัจจุบันใช้ single-user session ไม่ได้มี role-based access ให้ทดสอบ login/session และการเข้าถึง staging ต่าง session

## 2. ความสำคัญและวิธีทดสอบ

| ระดับ | ความหมาย | ตัวอย่าง |
|---|---|---|
| P0 | ความถูกต้องของเงิน/หน่วย, ข้อมูลสูญหาย, เขียนผิดพอร์ต, เข้าถึงโดยไม่มี session | trade, cashflow, replay, void, import ซ้ำ, reset, mutation race |
| P1 | ฟังก์ชันหลักใช้งานไม่ได้หรือข้อมูลหลักคลาดเคลื่อน | CRUD, filters, Goals, sync ราคา, calendar, error recovery |
| P2 | ประสบการณ์ใช้งานและรายละเอียดการแสดงผล | layout, สี, tooltip, keyboard, responsive |

สัญลักษณ์ในรายการกรณีทดสอบ: `U` = unit/domain, `C` = component/Vitest, `A` = API, `I` = integration ใช้ runtime และที่เก็บข้อมูลทดสอบจริง, `E` = browser end-to-end, `M` = manual/visual

รายการที่มีหลายวิธีให้ใช้แต่ละวิธีตรวจคนละชั้น เช่น C ตรวจการส่ง payload และสถานะปุ่ม ส่วน I ตรวจผลจริงใน ledger ห้ามใช้ mock UI เป็นหลักฐานเพียงอย่างเดียวสำหรับ P0

## 3. สถานะชุดทดสอบเดิมและช่องว่าง

จากการอ่านไฟล์พบ frontend Portfolio โดยตรง **13 ไฟล์ รวม 57 `it(...)`** ตัวเลขนี้เป็นจำนวนที่นับจาก source ไม่ใช่จำนวนที่รันผ่าน และยังไม่รวม shared component tests

| ไฟล์/กลุ่ม | จำนวนกรณีใน source | สิ่งที่มีอยู่แล้ว | งานที่ต้องเติม |
|---|---:|---|---|
| Portfolio page | 3 | initial load, Holdings tab, Executive tabs | URL, multi-portfolio CRUD, isolation, late responses, partial failures |
| Summary cards | 9 | ค่า THB, loading, refresh state/status | numeric oracle ครบ, zero/negative/null, persistence หลัง mutation |
| Overview / BucketTargetModal | 3 / 3 | warning, chart, select bucket, สุ่มสี | target validation, bucket CRUD, orphan holding, persistence |
| Holdings | 4 | table, bucket filter, selection, journal expand | sort, correction, single/batch delete, selection หลังสลับพอร์ต |
| Transactions / EditTransactionModal | 6 / 5 | table, SELL sum, filter, note, edit, void, date/time | pagination, fees/net reconciliation, imported immutable fields, replay, races |
| Incomes | 5 | summary, Received/Upcoming, sync, filters, rounds | eligibility, FX/tax, duplicate sync, manual conflict, YTD boundary |
| Analytics / Calendar | 4 / 3 | charts, breakdown warning, loading/error/events | ranges, cash accounting, date boundaries, navigation, race, missing history |
| DimeSyncModal / DimePdfViewerModal | 6 / 4 | provider switching, email, batch, commit, PDF/text | partial selection, retries, quarantine, duplicates, stale staging, stream interruption |
| TradeModal | 2 | cash badge, quick top-up mock flow | real backend insufficient-cash flow, THB/USD buy/sell, invalid input, double submit |
| ยังไม่มี test file โดยตรง | — | Watchlist, Goals และ modal บางตัวอาศัย test ชั้นอื่น | WatchlistTab, GoalsTab, WatchlistModal, GoalModal, CashFlowModal, IncomeModal, JournalModal, HoldingCorrectionModal, BatchAssignBucketModal, ResetConfirmModal |

Backend มีฐานทดสอบใน `tests/api/test_portfolio*.py`, `tests/tools/portfolio`, `tests/unit/portfolio` และ integration ที่เกี่ยวข้อง ให้อ่าน assertions และใช้ของเดิมก่อนเพิ่มกรณีใหม่ การมีไฟล์ทดสอบไม่ได้ยืนยันว่า flow ผ่านหน้าเว็บครบแล้ว

ยังไม่พบ config ของ Playwright/Cypress หรือ npm script สำหรับ browser E2E จากการสำรวจครั้งนี้ งาน E2E automation จึงเป็นงานที่ต้องตั้งค่าเพิ่ม หากยังไม่ตั้งค่าให้ใช้ manual E2E พร้อมหลักฐานตามรายการเดียวกัน

## 4. จุดเสี่ยงที่พบจากการอ่านโค้ด

ข้อสังเกตต่อไปนี้ยังไม่ได้ยืนยันด้วยการรัน ให้สร้าง regression case แล้วแยกผล Actual/Expected ก่อนตัดสินว่าเป็น defect

| ความเสี่ยง | หลักฐานจาก source | กรณีตรวจ |
|---|---|---|
| Partial selection ใน broker import อาจนำเข้ามากกว่าที่เลือก | DimeSyncModal มี checkbox และ `selectedItemIds`; `handleCommit` ส่ง IDs ให้ SCB เท่านั้น ส่วน Dime/WealthX ส่ง scan ID และ portfolio ID | IMP-06, IMP-07 |
| Quick top-up อาจไม่ปรากฏเมื่อ backend แจ้งเงินไม่พอ | TradeModal ตรวจข้อความ `Insufficient cash balance`; TradingService มีข้อความ `Insufficient cash in CASH_...` | TRD-06, E2E-03 |
| Transactions ของพอร์ตเก่าอาจทับพอร์ตใหม่ | `loadTransactions()` ไม่มี request-version guard ต่างจาก fetch หลักของหน้า | NAV-08, TXN-10 |
| Error ของ performance/journal อาจแสดงเหมือนข้อมูลว่าง | Portfolio.tsx ใช้ `.catch(() => set...([]))`; goal/allocation refresh บางจุด catch แล้วไม่แสดง error | NAV-06, JRN-06, ANA-07 |
| ข้อความ Reset กับผลหลัง reset อาจไม่ตรงกัน | Modal บอกเหลือ CASH_THB/CASH_USD ยอด 0; Markdown repository reset เป็น `holdings=[]` และ existing API test คาด array ว่าง | RST-03, RST-04 |
| ยอด SELL summary อาจใช้ cost basis แทนเงินขาย | router_ledger รวม `cost_thb` เป็น `total_sell_thb`; table ใช้ `cost_thb + realized_pnl_thb` ในบางคอลัมน์ | TXN-04, TXN-05 |
| Heatmap อาจตีความเงินฝากเป็นผลตอบแทน | Analytics สร้าง daily value จากผลต่าง NAV ระหว่าง snapshot; ต้องแยก NAV change กับ investment return | ANA-08 |
| การลบ holdings มี implementation สองจุด | PortfolioStateService และ TradingService มี batch remove; facade/dependency ต้องพา API ไปใช้เส้นทางที่รักษา ledger/reimport behavior | HLD-08, DAT-07 |

หาก UI และ service ให้คำตอบต่างกัน ให้บันทึกเป็น contract mismatch ไม่ปรับ expected ให้ตรงกับ implementation ที่ผิดเพียงเพื่อให้ test ผ่าน

## 5. สภาพแวดล้อมและข้อมูลทดสอบ

### 5.1 การเตรียมระบบ

1. บันทึก commit hash และ diff ของ working tree ก่อนรัน baseline เพราะปัจจุบันมีการแก้ไข Portfolio/import และงานอื่นที่ยังไม่ commit
2. ใช้ vault, state DB, checkpoint DB และไฟล์ provider cursor แยกสำหรับการทดสอบทั้งหมด อย่าชี้ manual/E2E backend ไป `memories/` จริง
3. ใช้ `tests/conftest.py` และ fixture isolation เดิมสำหรับ pytest; สำหรับ E2E ให้สร้าง bootstrap/seed/reset fixture แยกโดยไม่สมมติว่า pytest fixtures จะครอบคลุม server ภายนอก
4. ตรึงวันที่เป็น 2026-10-05 และ timezone Asia/Bangkok; ใช้ UTC timestamp และ US market date ในชุด boundary เพิ่มเติม
5. Mock/freeze ราคา, FX, fundamentals, dividends, corporate events และ Gmail ในชุด deterministic; ใช้ test mailbox และข้อมูลปิดบังส่วนบุคคลใน live smoke
6. ปิด background workers/scheduler ที่ไม่เกี่ยวข้อง ให้ cursor/TTL/cache reset ได้ต่อกรณี
7. เปิด backend และ frontend ตาม README หลังตั้งค่าทางเดินข้อมูลทดสอบแล้ว แยก terminal สำหรับแต่ละ server
8. Browser หลัก Chromium/Chrome และ Edge; ตรวจ mobile layout ที่ 390×844, tablet 768×1024 และ desktop 1440×900 เพิ่ม 200% zoom

### 5.2 ชุดข้อมูล

| Fixture | รายละเอียด | ใช้ตรวจ |
|---|---|---|
| D0 Empty | default portfolio, ไม่มี non-cash holding/transaction/watchlist/goal/history | first use, empty state, reset |
| D1 Mixed | CASH_THB 100,000; CASH_USD 1,000; FX ปัจจุบัน 35; PTT 100 หน่วย cost 30/price 35 THB; AAPL 10 หน่วย cost 100/price 110 USD; FUND_TEST 20 หน่วย cost 10/price 12 THB | summary, currency, cash, allocation, charts |
| D2 Multi | default + QA-A + QA-B; มี ticker เดียวกันแต่ units/buckets/notes ต่างกัน; goal ผูกแต่ละพอร์ต | isolation, global Goals, races |
| D3 Trades | chronological THB/USD BUY/SELL, fractional units, same timestamp, historical FX, notes ไทย/อังกฤษ | replay, transaction sums, date/time |
| D4 Income | received/upcoming, THB/USD, pay date/ex-date ต่างกัน, manual/synced/none, holdings เปลี่ยนก่อน/หลัง ex-date | eligibility, tax, FX, source conflict |
| D5 History | 0/1/2/90/400 snapshots, flat NAV, zero NAV, missing dates, legacy rows ไม่มี breakdown, breakdown sum ผิด | analytics/ranges/warnings |
| D6 Import | PDF/HTML ของ Dime, WealthX, SCBAM; BUY/SELL, order ID, fees/net, exact duplicate, conflict, fractional rounding, malformed/encrypted files | scan, commit, quarantine, dedup |
| D7 Events | holding/watchlist ที่ซ้ำกัน, Earnings/Ex-dividend, วันนี้/อดีต/อนาคต, month/year boundary, ticker fail บางตัว | calendar/date handling |
| D8 Volume | 500 holdings, 5,000 transactions, 1,000 journal entries, 400 snapshots | pagination, filtering, rendering และขอบเขต API |

### 5.3 ตัวเลขอ้างอิงที่คำนวณแยกจากระบบ

**D1 Mixed (ตามสูตรปัจจุบันที่ใช้ FX ปัจจุบันแปลง USD cost และ market value):**

| ค่า | Expected |
|---|---:|
| มูลค่า CASH_THB | 100,000.00 THB |
| มูลค่า CASH_USD | 35,000.00 THB |
| มูลค่า PTT / Unrealized P&L | 3,500.00 / 500.00 THB |
| มูลค่า AAPL / Unrealized P&L | 38,500.00 / 3,500.00 THB |
| มูลค่า FUND_TEST / Unrealized P&L | 240.00 / 40.00 THB |
| Total NAV รวม cash | **177,240.00 THB** |
| Total cost basis รวม cash | **173,200.00 THB** |
| Total unrealized profit | **4,040.00 THB** |
| NAV goal target 200,000 | raw progress **88.62%**; แสดง **88.6%** |

เทียบเงินที่แสดงด้วย tolerance ไม่เกิน 0.01 THB; ค่า internal units/price ใช้ precision ใน domain models; percentage ที่แสดงใช้จำนวนทศนิยมของ component ห้ามเทียบผลด้วยการเรียก helper ตัวเดียวกับ implementation

**THB lifecycle สำหรับ E2E-01:** เริ่ม cash 10,000; buy 100@30 → cash 7,000, units 100, avg cost 30; buy เพิ่ม 50@40 → cash 5,000, units 150, avg cost 33.333333; sell 60@45 → cash 7,700, units 90, realized profit 700.00 ก่อน fees ใช้ราคาตลาดตรึง 35 หลัง refresh → NAV 10,850.00, remaining asset cost 3,000.00 และ unrealized 150.00

**Dividend oracle:** eligible units 10 × DPS 1 USD → gross 10.00 USD; tax rate fixture 15% → net 8.50 USD; FX 35 → net 297.50 THB ใช้อัตราภาษีนี้เป็นข้อมูลทดสอบ ไม่ใช่อัตราที่ใช้กับสินทรัพย์ทุกประเภท

**Imported fees oracle:** BUY gross 1,000.00 + commission 2.00 + VAT 0.14 + other fees 0.50 → net 1,002.64; SELL gross 1,000.00 − fees 2.64 → net 997.36 ค่า cash movement ต้องอ้าง net และ cash-adjustment policy ของรายการนั้น

## 6. รายการกรณีทดสอบครบฟังก์ชัน

แต่ละ ID เป็น scenario; กรณีที่ระบุหลายค่า/หลาย provider ต้องแตกเป็น parameterized cases ตอนดำเนินการ ทุก mutation ตรวจ response, UI, reload และข้อมูลใน runtime/ledger พร้อมพอร์ตควบคุมที่ไม่ควรเปลี่ยน

### 6.1 Page shell, navigation และ asynchronous state

| ID | ระดับ/วิธี | ขั้นตอนหรือเงื่อนไข | ผลที่ต้องได้ |
|---|---|---|---|
| NAV-01 | P1 C/E | เปิด `/portfolio` ด้วย valid session | default portfolio, overview, header และ Executive Dashboard แสดงครบ |
| NAV-02 | P0 A/E | ไม่มี session/หมดอายุ; เรียก read และ mutation รวม import/PDF | กลับ login หรือได้รับ auth error ตาม contract; ไม่มีข้อมูลถูกเขียน |
| NAV-03 | P1 C/E | เปิดทั้ง 6 tabs; Back/Forward และ reload | URL/active tab ตรงกันและไม่ทำให้ portfolio_id หาย |
| NAV-04 | P1 C/E | deep link `portfolio_id`, `bucket`, `symbol`; legacy goals/analytics | context ถูกต้อง; legacy URL replace เป็น overview; filter ใช้งานได้ |
| NAV-05 | P1 C/A | tab ไม่รู้จัก, portfolio_id ไม่อยู่จริง/รูปแบบผิด, bucket/symbol ไม่พบ | มี fallback/error/empty state ชัดเจน ไม่เป็นหน้าว่างหรือสร้างพอร์ตใหม่โดยไม่ตั้งใจ |
| NAV-06 | P1 C/E | API หลักตัวใดตัวหนึ่ง fail, performance/journal fail, retry สำเร็จ | แยกโหลดไม่สำเร็จจากข้อมูลว่าง ไม่แสดงยอด 0 เสมือนข้อมูลจริง |
| NAV-07 | P1 C/E | loading/refreshing; กดซ้ำ; navigate/unmount ระหว่าง request | spinner จบถูกเวลา ไม่ติดค้าง ไม่ตั้ง state จาก context ที่เลิกใช้ |
| NAV-08 | P0 C/E | delay response พอร์ต A แล้วสลับ B; ให้ A กลับมาทีหลัง | ทุก section/modal ยังคงข้อมูล B; Goals คง semantics รวมทุกพอร์ต |

### 6.2 Multi-portfolio management

| ID | ระดับ/วิธี | ขั้นตอนหรือเงื่อนไข | ผลที่ต้องได้ |
|---|---|---|---|
| PRT-01 | P1 A/E | list portfolios ครั้งแรก/หลายพอร์ต | default มีหนึ่งรายการ; selector name/id ถูกต้อง |
| PRT-02 | P1 C/A/E | สร้างชื่อไทย/อังกฤษ, whitespace, ชื่อยาวและชื่อซ้ำ | trim ถูกต้อง; policy ชื่อซ้ำชัดเจน; ID ไม่ชน; selector/URL ไปพอร์ตใหม่ |
| PRT-03 | P1 C/A | ชื่อว่าง, double submit, create fail | validation และ loading; ไม่สร้างหลายพอร์ตจากการกดซ้ำ |
| PRT-04 | P1 C/A/E | rename default/พอร์ตอื่น แล้ว reload | ชื่อใหม่คงอยู่; id และข้อมูลเดิมไม่เปลี่ยน; goal badge ใช้ชื่อใหม่ |
| PRT-05 | P0 A/I/E | CRUD/trade/import/watchlist/journal ใน QA-A เมื่อ QA-B มี ticker เดียวกัน | QA-B ไม่เปลี่ยนทั้ง state, ledger, cursor และ projections |
| PRT-06 | P0 A/E | ลบพอร์ต: cancel/confirm/fail; ลบ default โดยเรียก API ตรง | cancel ไม่เปลี่ยน; success กลับ default; default ถูกป้องกันทั้ง UI/API |
| PRT-07 | P0 I/E | ลบพอร์ตที่มี goal/history/import แล้ว restart/reload | ไม่มี stale selector/mirror กลับมา; การจัดการ goal ที่อ้างพอร์ตที่ลบมี contract ชัดเจน |

### 6.3 Summary, prices และ FX

| ID | ระดับ/วิธี | ขั้นตอนหรือเงื่อนไข | ผลที่ต้องได้ |
|---|---|---|---|
| SUM-01 | P0 U/I/E | ใช้ D1 เทียบทุก summary card | ได้ NAV/cost/P&L ตาม numeric oracle รวม cash เพียงครั้งเดียว |
| SUM-02 | P1 C/U | 0, negative P&L, null, missing timestamp, ตัวเลขใหญ่ | ไม่มี NaN/Infinity; สี/เครื่องหมาย/THB และ unavailable state ถูกต้อง |
| SUM-03 | P1 C/A/I | เปิดหน้าปกติเทียบ explicit refresh | initial load ไม่เรียก provider refresh โดยไม่จำเป็น; refresh ส่ง flags ถูกต้อง |
| SUM-04 | P1 C/I/E | refresh สำเร็จทั้งหมด/บาง ticker fail/FX fail | badge และรายละเอียดตรง provider statuses; ไม่รายงาน success ทั้งหมดเมื่อ fail บางตัว |
| SUM-05 | P0 I/E | refresh mixed Stock/Fund/USD/Cash | provider routing, units/cost คงเดิม; market value/FX/goals/allocation สอดคล้องกัน |
| SUM-06 | P1 C/I | sparkline 0/1/หลาย snapshot, วันที่ไม่เรียง, flat NAV | เรียงวันที่และวาดได้; ไม่หารศูนย์หรือใช้พอร์ตเก่า |
| SUM-07 | P0 U/I | ใช้ FX ย้อนหลังใน trade แล้ว refresh FX ปัจจุบัน | historical transaction FX ไม่ทับ global current FX; เงิน THB/USD ไม่ปะปน |

### 6.4 Strategy Buckets & Allocation

| ID | ระดับ/วิธี | ขั้นตอนหรือเงื่อนไข | ผลที่ต้องได้ |
|---|---|---|---|
| BKT-01 | P1 U/C/I | targets/actual/variance จาก D1; warning | chart/table/name/color ตรงกัน; actual% ใช้ denominator ที่ตกลง; แสดง warning |
| BKT-02 | P1 C/E | click bucket จาก chart/table | ไป Holdings พร้อม bucket filter; clear filter คืนรายการครบ |
| BKT-03 | P1 C/A/I | เพิ่ม/แก้ชื่อ/แก้สัดส่วน/ลบ bucket แล้ว reload | persisted targets ถูกต้อง; holdings ที่อ้าง bucket ที่ลบจัดการตาม contract |
| BKT-04 | P1 C/A | ผลรวม 100, 99.98, 100.02 และ boundary tolerance ±0.01 | UI/backend ใช้ tolerance ตรงกันและปฏิเสธค่าที่เกินขอบเขต |
| BKT-05 | P1 C/A | target <0/>100, ชื่อ/id ว่างหรือซ้ำ, ไม่มี targets | reject/normalize ตาม contract; ไม่มี duplicate chart keys หรือบันทึกเสีย |
| BKT-06 | P2 C/M | auto color, picker, random row/all, เพิ่มเร็วติดกัน | สีใช้ได้และ id ไม่ชน; reload รักษาสี |
| BKT-07 | P1 U/C | NAV 0, unassigned holdings, cash, archived/zero-unit assets | chart ไม่หารศูนย์; ผลรวมและรายการที่รวมสอดคล้องกับ summary |

### 6.5 Holdings, correction และ batch actions

| ID | ระดับ/วิธี | ขั้นตอนหรือเงื่อนไข | ผลที่ต้องได้ |
|---|---|---|---|
| HLD-01 | P1 C/E | D1/D0; THB/USD/fund/cash/null fundamentals | columns/สองบรรทัด/units/price/cost/P&L ถูกต้อง; empty CTA ใช้ได้ |
| HLD-02 | P1 C | sort ทุกหัวตารางที่กดได้ ascending/descending; null/ties | sort ตามชนิดข้อมูล ไม่ sort ตัวเลขแบบ string |
| HLD-03 | P1 C/E | select row/all, filter bucket, เปลี่ยนพอร์ต, รายการถูกลบ | selection/action count ถูกต้อง; ไม่เหลือ selected ticker จาก context เก่า |
| HLD-04 | P1 C/A/I | assign single/batch ไป bucket หรือ unassigned | payload ตรงจำนวน; persisted และ allocation recalculated |
| HLD-05 | P1 A/I | batch มี ticker ไม่พบ/ซ้ำ/invalid bucket, fail กลางทาง | atomicity/partial policy ชัดเจน; ไม่แสดงสำเร็จเกินรายการที่เปลี่ยนจริง |
| HLD-06 | P0 C/A/I/E | correction units/cost/dividend/type/bucket/reason | field และ currency ถูกต้อง; totals/journal/projection อัปเดตตามผลจริง |
| HLD-07 | P0 A/I | negative units/cost, no-op, nonexistent, edit cash sentinel | reject โดยไม่เปลี่ยน state/ledger; ไม่สร้าง holding ใหม่โดยผิดพลาด |
| HLD-08 | P0 A/I/E | remove single/batch, cancel/confirm; มี imported trades; resync | affected transactions/reversals และ cash reconcile; reimport ได้ตาม policy; พอร์ตอื่นคงเดิม |
| HLD-09 | P0 A/I | ลบ cash sentinel โดย API ตรง/ปนใน batch | ไม่ทำลาย cash state; ไม่มีสรุป success ที่ซ่อนการข้ามรายการ |
| HLD-10 | P1 C/E | click View Transactions ของ ticker | ไป Transactions พร้อม symbol และ portfolio_id ถูกต้อง |

### 6.6 Trading Journal

| ID | ระดับ/วิธี | ขั้นตอนหรือเงื่อนไข | ผลที่ต้องได้ |
|---|---|---|---|
| JRN-01 | P1 C/A/E | เขียนจาก ticker row และ All entries | prefix/content/timestamp ถูกต้อง; เพิ่มหนึ่งรายการและ reload พบ |
| JRN-02 | P1 C/A | whitespace-only, ภาษาไทย, Markdown, multi-line, Unicode | ไม่รับข้อความว่าง; content ไม่สูญหาย |
| JRN-03 | P1 C/E | ค้น keyword ต่อเนื่อง, debounce 300 ms, clear | query ล่าสุดเท่านั้นมีผล; ล้างแล้วคืนข้อมูล |
| JRN-04 | P1 C/I | ticker `A`/`AA`/`AAPL`; notes มี ticker ในคำอื่น | journal association ไม่จับผิด symbol จาก substring; ถ้า contract ยังใช้ prose ให้บันทึกข้อจำกัด |
| JRN-05 | P1 C/E | expand row, All entries, modal ซ้อน, close/Escape | เปิดรายการถูก ticker; focus/scroll และ modal layer ใช้ได้ |
| JRN-06 | P1 C/A | limit 100, ช่วง 365 วัน, load/save failure | ขอบเขตข้อมูลและความผิดพลาดชัดเจน; ไม่อ้างว่า All entries ครบถ้าถูกจำกัด |

### 6.7 Trade entry

| ID | ระดับ/วิธี | ขั้นตอนหรือเงื่อนไข | ผลที่ต้องได้ |
|---|---|---|---|
| TRD-01 | P0 U/A/I/E | THB buy ใหม่/เพิ่ม/partial sell/full sell | units, weighted average, cash, realized/unrealized, ledger และ journal reconcile |
| TRD-02 | P0 U/A/I | USD buy/sell, historical FX/current FX, fractional shares | cash สกุลถูกต้อง; decimal precision และ THB conversion ถูกต้อง |
| TRD-03 | P0 A/I | sell มากกว่าหน่วย/ไม่มี holding, ผสม currency ใน ticker เดิม | reject ไม่มี partial mutation |
| TRD-04 | P0 C/A | units/price ≤0, NaN/Infinity ผ่าน API, symbol ว่าง/cash sentinel, date ผิด | validation ฝั่ง UI/API/domain ครบ ไม่เขียน ledger เสีย |
| TRD-05 | P1 C/A/E | asset types ที่ UI รองรับ, bucket/unassigned, lowercase/space symbol, notes/date | normalized payload ตรงชนิดสินทรัพย์และพอร์ต; persisted metadata ครบ |
| TRD-06 | P0 C/A/E | insufficient cash โดยใช้ backend response จริง แล้ว quick top-up | error/ยอดขาดถูกต้อง; action ปรากฏ; deposit+buy สำเร็จครั้งเดียว |
| TRD-07 | P0 I/E | top-up สำเร็จแต่ buy fail/timeout แล้ว retry | แสดงเงินจริงหลัง top-up; retry ไม่ฝากซ้ำจากยอด cash เก่า และไม่ซื้อซ้ำ |
| TRD-08 | P0 C/I/E | double submit/concurrent buys/backend timeout หลัง commit | ไม่ lost update/overspend; reload/reconciliation บอกผลจริงก่อน retry |
| TRD-09 | P1 C/A | date change รวดเร็ว, FX historical/live/fallback/failure | source/rate ตรงวันที่สุดท้าย; ค่า fallback ไม่แสดงเหมือน historical |

### 6.8 Cash flow และ manual income

| ID | ระดับ/วิธี | ขั้นตอนหรือเงื่อนไข | ผลที่ต้องได้ |
|---|---|---|---|
| CSH-01 | P0 U/A/I/E | ฝาก/ถอน THB และ USD แล้ว reload | cash native เปลี่ยนตาม amount; NAV แปลง FX ถูกต้อง |
| CSH-02 | P0 A/I | ถอนเท่าที่มี/เกิน/ไม่มี cash holding | ไม่ติดลบจาก invalid withdrawal; cash creation ตาม contract |
| CSH-03 | P0 C/A | amount ≤0/non-finite, currency/action/date ผิด | reject ไม่เขียน ledger/state |
| CSH-04 | P0 I | deposit/withdraw แล้วเทียบ unrealized กับก่อนทำรายการ | cash movement ไม่สร้าง unrealized profit; cost/summary ถูกสูตร |
| CSH-05 | P1 C/A/E | currency switch, optional FX, backdate, notes, cancel, server fail | payload/feedback ถูกต้อง; cancel ไม่มี mutation |
| INC-01 | P0 U/A/I/E | Dividend/Interest/Rental/Other, มี/ไม่มี source ticker | cash และ passive income เปลี่ยนถูกจำนวน; source association ถูกต้อง |
| INC-02 | P0 C/A/I | income ≤0, invalid/unknown source, duplicate submit | validation ไม่มี partial state; ไม่เพิ่ม cash ซ้ำจากการกดซ้ำ |
| INC-03 | P0 I | วันที่ข้ามปี, future/backdated income, manual dividend แล้ว sync | YTD ตาม contract; ไม่คิด manual+synced ซ้ำและไม่ overwrite manual |

### 6.9 Transactions, edit และ void

| ID | ระดับ/วิธี | ขั้นตอนหรือเงื่อนไข | ผลที่ต้องได้ |
|---|---|---|---|
| TXN-01 | P1 C/A/E | load, empty, error/retry, search symbol/notes, BUY/SELL/ALL, clear symbol | rows/count/filter/URL ถูกต้อง |
| TXN-02 | P1 C/E | sort timestamp/symbol/units/price/sum/P&L, null และ tie | sort ถูกชนิดและทิศทาง; ไม่แก้ข้อมูลต้นทาง |
| TXN-03 | P1 C/E | page size 10/25/50/100, next/previous, filter ขณะอยู่หน้าท้าย | page/row index/count ถูกต้อง; reset/clamp หน้าเมื่อผลลดลง |
| TXN-04 | P0 U/A/C/I | BUY/SELL ทั้ง Native/THB, realized loss, FX, fees, net | แยก cost basis, gross proceeds และ net cash; ไม่รวมคนละสกุลเป็นยอดเดียว |
| TXN-05 | P0 A/C/I | full summary กับ filtered totals; รวมหลาย page | SELL totals ใช้นิยามเดียวกัน; filtered totals รวมทุก filtered row ไม่ใช่เฉพาะหน้า |
| TXN-06 | P1 C/A/I | note inline: save/cancel/empty/multiline/error | แก้เฉพาะ note; persisted; economic fields/cash ไม่เปลี่ยน |
| TXN-07 | P0 C/A/I/E | edit manual units/price/date/time/FX, adjust_cash true/false | preview delta ถูก; replay ทั้งลำดับ; cash/P&L/state/projection ตรง policy |
| TXN-08 | P0 C/A/I | Dime imported immutable fields, notes-only, direct API edit | UI disable และ backend enforce; bypass UI ไม่แก้ economic fields |
| TXN-09 | P0 A/I/E | void BUY/SELL, void ซ้ำ, void reversal, dependency ทำให้ oversell | reversal สัมพันธ์ original; cash ตาม original cash_adjusted; ไม่คืนเงินซ้ำ; invalid replay ถูกป้องกัน |
| TXN-10 | P0 C/E | load/edit/void เกิดพร้อมสลับพอร์ตหรือ reload | ไม่ส่ง tx ของ A ไป B; late response ไม่ทับ B |
| TXN-11 | P1 C/U | timestamp ISO/space/date-only, วันนี้/เมื่อวาน/ตอนนี้, timezone/year boundary | ค่าที่แสดง/ส่งตรงเวลา fixture และ source restrictions |

### 6.10 Broker import: Dime, WealthX และ SCBAM

ใช้ IMP-01 ถึง IMP-12 กับทุก provider ที่รองรับช่องทางนั้น; SCBAM ใช้ email/HTML ไม่คาดหวัง PDF upload ที่ UI ไม่มี

| ID | ระดับ/วิธี | ขั้นตอนหรือเงื่อนไข | ผลที่ต้องได้ |
|---|---|---|---|
| IMP-01 | P1 C/A/E | สลับ provider/upload/email ระหว่างกรอก | ช่องทาง/labels/API ตรง provider; state/scan/warnings ไม่ปะปน |
| IMP-02 | P1 C/A/I | upload valid PDF, encrypted ถูก/ผิด/ไม่มี password, ไม่เลือกไฟล์ | valid scan ได้; failure มีเหตุผล; password show/hide ทำงาน |
| IMP-03 | P0 A/I | non-PDF magic header, corrupt/oversize/empty/no trades/unsupported layout | ไม่สร้าง trade จากข้อมูลไม่ครบ; structured error/quarantine |
| IMP-04 | P1 C/A/I | search query, no emails, IMAP unavailable, attachment missing | query/limit ถูกต้อง; feedback และ retry ใช้ได้ |
| IMP-05 | P0 U/I/E | scan แล้ว inspect date/action/units/price/gross/all fees/net/IDs | staging ตรงเอกสาร; ยังไม่เปลี่ยน portfolio ก่อน commit |
| IMP-06 | P0 C/A/I/E | มี 3 staged items เลือก 1 แล้ว commit | นำเข้าเฉพาะ 1 ที่เลือก หรือ UI ต้องไม่มี selection ถ้า contract เป็นทั้ง batch; ห้ามเลือก 1 แล้วเขียน 3 |
| IMP-07 | P0 C/A/I | deselect all, select all, จำนวนบนปุ่ม/summary | ไม่มีรายการที่เลือกต้องไม่ commit ทั้งชุดโดยไม่ตั้งใจ; count ตรง payload |
| IMP-08 | P0 U/A/I | exact duplicate ภายใน batch/ใน ledger, order ID ต่างแต่ economics เหมือนกัน, identity conflict | duplicate ข้าม; distinct orders คงครบ; conflict ไม่ silently merge |
| IMP-09 | P0 A/I | commit ซ้ำ/concurrent, expired/not-found scan, scan ต่าง session | idempotency/TTL/isolation enforce; ไม่เขียนพอร์ตซ้ำหรือใช้ staging คนอื่น |
| IMP-10 | P0 U/I | fee allocation multi-line, reconciliation tolerance/fractional bound, mismatched statement | fees รวมตรงเอกสาร; accepted rounding มี bound; mismatch quarantine |
| IMP-11 | P0 I/E | import USD/THB/Fund; cash_adjusted true/false; เงินไม่พอ/SELL ก่อน BUY | cash/net/units/cost/P&L ตาม provider policy; invalid sequence ไม่เขียนครึ่งชุด |
| IMP-12 | P0 I/E | void/remove holding แล้ว rescan/reimport เทียบ ledger active กับ provider cursor | รายการถูก void กลับมานำเข้าได้; active เดิมไม่ถูกนำเข้าซ้ำ |
| IMP-13 | P1 C/A/E | batch stream: progress/warning/complete/error, 0 results, all synced, force_rescan | จำนวนและข้อความถูกต้อง; force rescan ไม่ปิด dedup; loading จบทุกเส้นทาง |
| IMP-14 | P0 C/A/I/E | stream disconnect/truncated frame, close/source switch ระหว่าง scan, restart backend | ไม่ auto-commit; ไม่มีผล provider เก่าทับใหม่; recovery ตรวจผลจริงก่อน retry |
| IMP-15 | P1 C/A/E | quarantine reason, PDF iframe/download/open, raw text หลายหน้า, SCB HTML | เอกสารตรง warning/provider; error ใช้ได้; ไม่มีการรัน script จาก HTML อีเมล |
| IMP-16 | P0 I/E | commit ไป QA-A/QA-B, delay callback 1.2 s, reload, DB/projection/journal failure | state/ledger/cursor ตรงพอร์ต; success หลัง commit จริง; recover ไม่ duplicate |

### 6.11 Dividends & Income tab

| ID | ระดับ/วิธี | ขั้นตอนหรือเงื่อนไข | ผลที่ต้องได้ |
|---|---|---|---|
| DIV-01 | P1 C/E | Received/Upcoming, native USD/THB totals, zero/null | summary และ rows partition ถูกต้อง; Cash ไม่เป็น dividend holding |
| DIV-02 | P1 C/E | search ticker/company และ source ALL/synced/manual/none | filtered counts/results ถูก; clear แล้วคืนข้อมูล |
| DIV-03 | P1 C/E | open/close rounds, received/upcoming status, ex/pay date, today/past/future | ทุก field ตรง fixture; ไม่มี round จาก ticker เก่า |
| DIV-04 | P0 U/I | units ณ ex-date เทียบ buy/sell ก่อน/หลัง, timezone-aware provider date | eligibility ไม่ใช้ current units แทนประวัติเมื่อมีการเปลี่ยนถือครอง |
| DIV-05 | P0 U/I | DPS/tax/native FX conversion, no pay date, missing historical FX | gross/net/YTD/accumulated ตาม oracle และ policy fallback |
| DIV-06 | P0 A/I/E | sync ครั้งแรก/ซ้ำ, provider fail บางตัว, manual edit พร้อม sync | ไม่เพิ่ม cash/dividend ซ้ำ; preserve manual; partial result รายงานครบ |
| DIV-07 | P0 I/E | edit/void trade หลัง sync แล้ว resync | eligibility cache/source ถูก invalidated และยอดคำนวณใหม่ถูกต้อง |
| DIV-08 | P1 C/E | sync จบแต่ state reload fail; สลับพอร์ตระหว่าง sync | ไม่สรุปว่าข้อมูลหน้าอัปเดตครบ; ไม่ใช้ syncResult เก่ากับพอร์ตใหม่ |

### 6.12 Goals

| ID | ระดับ/วิธี | ขั้นตอนหรือเงื่อนไข | ผลที่ต้องได้ |
|---|---|---|---|
| GOL-01 | P1 C/A/E | ดู Goals จาก QA-A และ QA-B | เห็น goals รวมครบทุกพอร์ต; badge/name/type/updated time ถูกต้อง |
| GOL-02 | P0 U/A/I | nav_target/cash_target/passive_income_ytd/bucket_target | current amount/progress ใช้ข้อมูลพอร์ตและ bucket ที่ผูกจริง |
| GOL-03 | P1 C/A/I/E | create/edit/delete, name ไทย/duplicate, cancel/fail | ไม่ overwrite goal อื่น; response ยังคง goals รวมครบ; reload คงข้อมูล |
| GOL-04 | P1 C/A | target ≤0, date/years_from_now, expired/today/future deadline, bucket invalid | validation/deadline days left ตรง fixture; ไม่มี NaN progress |
| GOL-05 | P1 C/U | progress <0/0/>100, completed badge/bar | bar/display clamp ตาม UI; current/target amount ยังตรงข้อมูลจริง |
| GOL-06 | P0 I/E | trade/cash/income/correction/reset เปลี่ยน current value | goal refresh ถูกพอร์ต; ไม่ค้างค่าก่อน mutation |
| GOL-07 | P1 C/E | goal ผูกพอร์ตอื่น, พอร์ตถูก rename/delete, bucket ถูกลบ | ไม่มี label/metric ที่อ้างผิด; orphan policy แสดงชัดเจน |

### 6.13 Performance Analytics

| ID | ระดับ/วิธี | ขั้นตอนหรือเงื่อนไข | ผลที่ต้องได้ |
|---|---|---|---|
| ANA-01 | P1 C/E | Goals ↔ Performance, ย่อ/ขยายจากทุก main tab | section เปิดถูก mode; ข้อมูลและ portfolio context ถูกต้อง |
| ANA-02 | P1 C/A/E | 30/90/365/all, พอร์ตต่างกัน, rapid range switch | query/range/date/count ถูก; response เก่าไม่ทับช่วงใหม่ |
| ANA-03 | P1 C/U | 0/1/2 snapshots, flat/zero/negative values, unsorted dates | fallback เมื่อ <2 จุด; line/table/tooltip ถูกและไม่หารศูนย์ |
| ANA-04 | P0 U/I/C | Treemap current asset types/cash THB+USD/archived assets | ผลรวมตรง NAV; cash ไม่หายหรือซ้ำ; ไม่อ้าง sector ที่ไม่มี taxonomy |
| ANA-05 | P1 C/I | historical breakdown มี/ไม่มี/ขาดช่วง, absolute/100%, categories เปลี่ยน | ไม่สร้าง historical allocation จาก holdings ปัจจุบัน; gap ไม่ถูกอ้างเป็นข้อมูลจริง |
| ANA-06 | P1 A/C | breakdown sum ต่าง NAV >1 THB, valid sum | coverage warning ถูก; valid rows ไม่เตือนผิด |
| ANA-07 | P1 C/A/E | performance API fail/slow, refresh ราคาแล้ว snapshot วันเดียวกัน | แยก error/empty; snapshot upsert ไม่ duplicate; ข้อมูลหลัง refresh ทันสมัย |
| ANA-08 | P0 U/I/C | ฝาก 10,000 โดยราคาไม่เปลี่ยน เทียบวันก่อน | NAV change เพิ่มตามฝาก; heatmap/labels ไม่อ้างว่าเป็น trading profit หรือ TWR |

### 6.14 Watchlist และ Corporate Calendar

| ID | ระดับ/วิธี | ขั้นตอนหรือเงื่อนไข | ผลที่ต้องได้ |
|---|---|---|---|
| WAT-01 | P1 C/A/E | add/edit/delete, optional target/notes, empty state | CRUD/persistence และ updated timestamp ถูกต้อง |
| WAT-02 | P1 C/A | ticker lowercase/space/special chars, duplicate, target invalid | normalize/URL encode/validation; ไม่เพิ่มซ้ำผิด policy |
| WAT-03 | P0 A/I/E | watchlist ticker เดียวกันในสองพอร์ต | แก้พอร์ตหนึ่งไม่กระทบอีกพอร์ต |
| WAT-04 | P1 C/E | cancel/delete confirm, save/delete fail, reload | feedback ถูก; ไม่ลบ/เปลี่ยน local state ก่อนสำเร็จ |
| CAL-01 | P1 C/A/E | D7 มีทั้ง earnings/ex_dividend จาก holdings/watchlist | date/ticker/company/bucket/estimates/chips ตรง response |
| CAL-02 | P1 C/E | previous/next month, วันนี้, year rollover, leap February | grid/เดือน/วันและ event placement ถูกต้อง |
| CAL-03 | P1 C/E | Upcoming sidebar open/close, past/today/future | เรียง days_until; ตัด past จาก Upcoming; tooltip/detail ใช้ได้ |
| CAL-04 | P1 A/I | ticker ซ้ำ holding+watchlist, failed ticker บางตัว/ทั้งหมด, ไม่มี events | dedup/bucket precedence ตาม contract; failure ไม่ทำให้ events ที่สำเร็จหาย |
| CAL-05 | P1 C/I | UTC/Bangkok/US event date ใกล้เที่ยงคืน | วัน event และ days_until ตรง market/date semantics |
| CAL-06 | P0 C/E | สลับพอร์ตขณะ fetch, watchlist/holding เปลี่ยนแล้วเปิด Calendar ใหม่ | ไม่แสดง ticker ของพอร์ตเก่า; reload/cache invalidation ได้ข้อมูลล่าสุด |

### 6.15 Clean Slate Reset

| ID | ระดับ/วิธี | ขั้นตอนหรือเงื่อนไข | ผลที่ต้องได้ |
|---|---|---|---|
| RST-01 | P0 C/E | เปิด/cancel, คำผิด, `RESET`/`reset`/space | submit ได้เฉพาะ confirmation ที่ normalize ตาม contract; cancel ไม่เขียน |
| RST-02 | P0 A/I/E | reset QA-A เมื่อ QA-B มีข้อมูลครบ | ล้างเฉพาะ QA-A; API/state/selector/goals refresh ถูก context |
| RST-03 | P0 A/I | ตรวจ cash state, NAV, allocation defaults, ledger หลัง reset | NAV/cash=0, targets default; ยืนยัน contract holdings ว่างหรือสอง cash sentinels ให้ UI/API ตรงกัน |
| RST-04 | P0 I/E | เทียบ holdings sidecars, journal, watchlist, goals, history, trades, cursor ก่อน/หลัง | ล้างและคงข้อมูลตามข้อความที่ผู้ใช้เห็น; ไม่ทิ้ง trades ที่ replay กลับมาสร้าง holding เอง |
| RST-05 | P0 I | backup fail/commit fail/projection fail; restore/restart | ไม่แจ้งสำเร็จเมื่อ reset ไม่ครบ; backup ใช้กู้ได้และไม่มีข้อมูลครึ่งชุด |
| RST-06 | P0 I/E | reset ซ้ำ/พร้อม trade; rescan หลัง reset; reload backend | ไม่มี lost update/duplicate import; policy cursor/active ledger หลัง reset ชัดเจน |

### 6.16 Data integrity, recovery และการใช้งาน

| ID | ระดับ/วิธี | ขั้นตอนหรือเงื่อนไข | ผลที่ต้องได้ |
|---|---|---|---|
| DAT-01 | P0 U/I | replay chronological, same timestamp, edit backdate, reversal | units/cost/cash/P&L สอดคล้อง ledger อย่าง deterministic |
| DAT-02 | P0 I | failure ระหว่าง state/ledger/system journal/projection writes | atomic commit/recovery ไม่มี duplicate/lost events; reconciliation ตรวจพบ mismatch |
| DAT-03 | P0 A/I | lock timeout/concurrent mutation, domain invalid, DTO invalid | HTTP mapping ตาม contract: domain 400/404, request schema 422, lock 503, internal 500; ไม่รายงาน 200 เท็จ |
| DAT-04 | P0 I | projection rebuild กับ annotation ของผู้ใช้ | numeric projection มาจาก runtime checkpoint; prose/annotation คงอยู่ |
| DAT-05 | P0 A/I | portfolio_id/symbol/goal name/scan_id มี path traversal หรือ URL chars | ไม่เข้าถึงไฟล์นอกขอบเขต; encoded identifiers ใช้งานได้ |
| DAT-06 | P1 A/C/I | API types/OpenAPI parity, legacy action casing/ledger columns | frontend parse ได้; normalized fields ไม่ทำให้เศรษฐศาสตร์รายการเปลี่ยน |
| DAT-07 | P0 A/I | ทดสอบผ่าน real DI/facade แทน patch legacy modules เท่านั้น | route ใช้ service ที่ถูกต้อง; behavior remove/void/reimport เหมือน contract |
| UX-01 | P2 E/M | 3 viewport sizes, 200% zoom, long Thai/name/numbers, all tables/modals | controls ไม่ทับ/หลุดจอ; ตาราง scroll ได้; ตัวเลขสำคัญอ่านได้ |
| UX-02 | P2 C/E/M | keyboard-only Tab/Shift+Tab/Enter/Escape, focus trap/return, nested dialogs | ปุ่ม/icon มี accessible name; close ใช้ได้; focus ไม่หลุด modal |
| UX-03 | P1 A/C/E | malicious notes/Markdown/HTML, formula-prefix notes, PDF/email links | ไม่ execute script; CSV/ledger output ไม่สร้างสูตรโดยไม่ตั้งใจ; secret ไม่อยู่ในหลักฐาน/log |
| UX-04 | P1 E/M | D8 load/filter/sort/pagination, rapid tab switch | ไม่มี freeze/unbounded requests; ผลข้อมูลถูก; บันทึกเวลาและ bottleneck |
| UX-05 | P2 E/M | Chromium/Edge, refresh/relogin, offline → online | core workflows ใช้ได้เหมือนกัน; ข้อมูล committed คงอยู่หลัง reconnect |

## 7. End-to-end scenarios ที่ต้องผ่านก่อนปิดงาน

### E2E-01: THB portfolio lifecycle

1. Login → สร้าง QA-A → ฝาก THB 10,000
2. ตั้ง buckets ให้รวม 100% → buy PTT 100@30 → buy เพิ่ม 50@40 → sell 60@45
3. Refresh ด้วย price fixture 35 แล้วเทียบ cash 7,700, units 90, cost เฉลี่ย 33.333333, realized 700, unrealized 150 และ NAV 10,850
4. ตรวจ Overview, Summary, Holdings, Transactions, journal และ NAV goal
5. Reload frontend/restart test backend → ตรวจ persisted state/ledger และ QA-B ไม่เปลี่ยน

### E2E-02: USD trade และ historical FX

ฝาก USD 1,000 → buy 2@100 ใช้ historical FX 34 → refresh current price 110/FX 35 → ตรวจ CASH_USD 800, stock value 7,700 THB, cash value 28,000 THB, NAV 35,700 THB และ unrealized 700 THB ตามสูตรปัจจุบัน ตรวจ transaction FX ยังเป็น 34 และ goal/asset allocation ใช้ current valuation

### E2E-03: Insufficient cash และ top-up recovery

ลอง buy ที่เงินไม่พอ → ตรวจข้อความ backend จริงและปุ่ม top-up → ทำให้ฝากสำเร็จแต่ buy fail → reload ยืนยันยอดฝากจริง → retry ซื้อโดยไม่ฝากหรือซื้อซ้ำ ตรวจ double click และ timeout หลัง commit เพิ่มเติม

### E2E-04: Edit/void และ replay

สร้าง BUY สองครั้งและ SELL → แก้ BUY ย้อนหลังพร้อม adjust_cash true/false ใน fixture แยก → ตรวจ replay/realized/cash ทั้งหมด → void SELL → void ซ้ำ → ตรวจ reversal และ idempotency; ใช้ fixture Dime ตรวจว่าแก้ economic fields ไม่ได้แต่ notes ได้

### E2E-05: Broker import, duplicates และ reimport

ทำแยก Dime/WealthX/SCBAM: scan fixture 3 รายการ → ตรวจ staged ไม่มี portfolio mutation → เลือก 1 → commit → ตรวจ cash/units/ledger → scan+commit ซ้ำ → ไม่มี duplicate → void/remove holding → scan ใหม่ → recovery item กลับมา → commit ได้หนึ่งครั้ง ตรวจ warnings/PDF/text/HTML ใน fixture ที่กักกัน

### E2E-06: Income, dividends และ Goals

บันทึก manual income → sync dividend received/upcoming โดยใช้ eligible units fixture → ตรวจ net/native/THB/tax/YTD และ source → sync ซ้ำ → จำนวนไม่เพิ่มซ้ำ → สร้างครบ 4 goal types ผูก QA-A/QA-B → เปลี่ยนพอร์ตแล้วยังเห็น goals รวมและ progress ของแต่ละพอร์ตถูกต้อง

### E2E-07: Navigation, calendar และ stale responses

เปิด deep link bucket/symbol → เปลี่ยนแท็บและ Back/Forward → เปิด calendar จาก holdings/watchlist → delay requests ของ QA-A และสลับ QA-B → ปล่อย response A ทีหลัง → ตรวจทุก section และ modal ไม่มีข้อมูล A ทับ B พร้อม range switch ใน Analytics

### E2E-08: Reset และ recovery

ใช้สำเนา fixture ที่มีทุกชนิดข้อมูล → cancel/wrong confirmation → reset QA-A จริง → เทียบข้อความ reset กับข้อมูลที่ล้าง/คงอยู่และตรวจ backup → reload/restart → QA-B คงเดิม → rescan/reimport ตาม policy → inject backup/commit failure อีก fixture แล้วตรวจ recovery

ทุก E2E เก็บ screenshot หน้าหลัก/ผล mutation, request-response ที่ปิดบังข้อมูลส่วนตัว, state/ledger ก่อนและหลัง, console errors และผลพอร์ตควบคุม

## 8. แผนงานและลำดับดำเนินการ

ประมาณการ 7–10 วันทำงานสำหรับผู้พัฒนาและผู้ทดสอบที่ร่วมกันทำงานได้; เป็นกรอบวางงาน ไม่ใช่กำหนดส่งตายตัว จำนวน defect และความพร้อม test mailbox มีผลต่อระยะเวลา

| Phase | งาน | ผู้รับผิดชอบตามบทบาท | ผลส่งมอบและเกณฑ์จบ | ประมาณการ |
|---|---|---|---|---|
| 0 | inventory/traceability, snapshot working tree, fixture isolation, baseline เดิม | Developer + QA | baseline report แยก pre-existing failure; environments ไม่ชี้ข้อมูลจริง | 0.5–1 วัน |
| 1 | numeric oracles และ P0 API/domain: trade/cash/edit/void/import/reset/isolation | Backend developer + QA | targeted tests ตรวจ state/ledger จริง; mismatch มี defect record | 1.5–2 วัน |
| 2 | เติม frontend component tests ที่ขาด พร้อม failure/race/payload cases | Frontend developer | ทุก action และ modal มี happy/error/cancel coverage ที่จำเป็น | 1.5–2 วัน |
| 3 | real DI/integration, atomic recovery, projection/cursor/dedup, year/FX boundaries | Backend developer | deterministic fixtures ผ่าน; recovery/reimport ไม่ duplicate | 1–1.5 วัน |
| 4 | setup browser E2E หรือ manual harness; รัน E2E-01 ถึง 08 | QA + Developer | screenshot/trace และ numeric assertions ของทั้ง 8 workflows | 1–1.5 วัน |
| 5 | responsive/accessibility/volume และ provider live smoke ใน test mailbox | QA | UI ใช้ได้ทุกขนาด; measured performance; provider results แยกจาก offline tests | 0.5–1 วัน |
| 6 | triage/fix/retest และ regression รอบสุดท้าย | Developer + QA | P0/P1 ผ่าน, final report, defects ที่เหลือระบุผลกระทบ | 1 วันขึ้นไปตาม defect |

งานเร่งด่วนใน Phase 1–2: IMP-06/07, TRD-06/07, TXN-04/05/09/10, HLD-08, RST-03/04 และ NAV-08 เพราะมีความเสี่ยงจาก source ที่ต้องยืนยันก่อน

### งานที่ต้องสร้างหรือขยาย

- ขยาย existing tests แทนทำซ้ำ: Portfolio page, TradeModal, Transactions, EditTransactionModal, DimeSyncModal, Incomes, Analytics และ backend portfolio suites
- เพิ่ม direct component tests สำหรับ Goals, Watchlist และ modal ที่ขาดตามตารางส่วน 3
- เพิ่ม fixture builders ที่แชร์ deterministic input แต่คำนวณ expected แยกจาก implementation
- เพิ่ม integration ผ่าน real dependency injection สำหรับ UI/API paths สำคัญและ cross-portfolio control assertions
- ถ้าเลือก Playwright ให้เพิ่ม dependency/config/isolated server bootstrap/fixtures และ E2E script ก่อนอ้างว่า run command ใช้งานได้; ยังไม่ถือว่ามีเครื่องมือนี้ใน repo ปัจจุบัน
- ใช้ `test-results/portfolio/` เก็บรายงานใหม่ เพื่อไม่เขียนทับรายงานเก่าที่ tracked อยู่ใน `tests/`

## 9. คำสั่งเริ่มทดสอบที่ใช้กับ repo ปัจจุบัน

คำสั่งด้านล่างเป็นคำสั่งสำหรับขั้นดำเนินการ ยังไม่ได้รันในงานจัดทำแผนนี้ ตรวจ fixture isolation ก่อนเริ่ม และใช้ terminal ของแต่ละ working directory

### Frontend — รันจาก `web/`

```powershell
npm run test -- src/pages/Portfolio.test.tsx src/components/portfolio src/api/client.test.ts src/components/charts/charts.test.tsx src/components/ui/Modal.test.tsx src/components/ui/SegmentedControl.test.tsx src/components/RequireAuth.test.tsx
npm run build
```

สำหรับ coverage ตามไฟล์ที่เกี่ยวข้อง:

```powershell
npm run test:coverage -- src/pages/Portfolio.test.tsx src/components/portfolio --coverage.include=src/pages/Portfolio.tsx --coverage.include=src/components/portfolio/** --coverage.reportsDirectory=../test-results/portfolio/frontend-coverage
```

Coverage เป็นหลักฐานประกอบ traceability ห้ามใช้เปอร์เซ็นต์แทน assertions ของยอดเงินและ recovery

### Backend — รันจาก root

```powershell
New-Item -ItemType Directory -Force -Path test-results/portfolio
.\.venv\Scripts\python.exe -m pytest -o "addopts=" tests/api/test_portfolio.py tests/api/test_portfolio_actual_routes.py tests/api/test_portfolio_actual_mutations.py tests/api/test_portfolio_calendar.py tests/unit/api/test_portfolio_router_di.py tests/unit/api/test_dime_sync_router.py tests/unit/portfolio tests/tools/portfolio tests/unit/test_portfolio_service_contract.py -q --junitxml=test-results/portfolio/backend.xml
```

Integration/recovery ที่เกี่ยวข้อง:

```powershell
.\.venv\Scripts\python.exe -m pytest -o "addopts=" tests/integration/test_portfolio_service_di.py tests/integration/test_portfolio_projection_rebuild.py tests/integration/test_portfolio_atomic_recovery.py tests/architecture/test_vault_write_boundaries.py tests/test_vault_isolation.py -q --junitxml=test-results/portfolio/integration.xml
```

`-o "addopts="` ปิดค่า default ที่เขียน report/coverage ไปไฟล์เดิม; เลือก coverage/output path ใหม่เมื่อต้องการรายงานเพิ่มเติม ก่อนรัน integration ให้อ่าน markers/fixtures และแยก provider live tests ออกจาก deterministic run

ตรวจ API schema/contract หลังแก้ endpoint หรือ DTO ด้วย `tests/api/test_openapi_contract.py` และ workflow `check:types` ของ web ตาม repo; อย่า regenerate/accept snapshot เพียงเพื่อซ่อน contract drift

### Browser / manual

เปิด backend ด้วย `uv run uvicorn api.main:app --reload` จาก root และ `npm run dev` จาก web **หลังตั้งค่า DB/vault ทดสอบแล้ว** เข้า `/portfolio` ด้วย test session รัน E2E-01 ถึง 08 ตาม fixtures จนกว่าจะมี E2E runner ที่ติดตั้งและตรวจคำสั่งจริงแล้ว

## 10. เกณฑ์รับงานและรายงานผล

### Entry criteria

- [ ] source snapshot และ scope ของ working tree ถูกบันทึก
- [ ] test DB/vault/mailbox/cursors แยกจากข้อมูลจริงและ reset fixture ได้
- [ ] baseline เดิมถูกรันและ failures เดิมถูกบันทึก
- [ ] numeric fixtures และ expected values ถูกตรวจแยกจาก implementation
- [ ] contract ที่ยังไม่ตรงกัน โดยเฉพาะ reset/partial import/SELL totals ถูกระบุใน defect register

### Exit criteria

- [ ] ทุก scenario มีผล Passed/Failed/Blocked/Not run และ mapping ไปหลักฐาน/test file
- [ ] P0/P1 ผ่านทั้งหมด ไม่มี defect ค้างที่ทำให้เงิน/หน่วยผิด ข้อมูลสูญหาย เขียนผิดพอร์ต หรือ core workflow ใช้งานไม่ได้
- [ ] E2E-01 ถึง E2E-08 ผ่านบน isolated environment โดยเทียบ numeric state จริง
- [ ] ไม่มี duplicate/lost mutation ใน retry/concurrency/void/reimport/recovery cases
- [ ] reload/restart และ projection rebuild รักษา committed state และ human annotations
- [ ] UI/API types และ build ผ่าน; targeted backend/frontend regression ผ่านหลังแก้ไขล่าสุด
- [ ] responsive/keyboard/browser checks ครบ; P2 ที่เหลือมี owner และผลกระทบชัดเจน
- [ ] provider live smoke รายงานแยก หาก credentials/provider unavailable ให้ระบุ Blocked ไม่รวมเป็น Passed
- [ ] final report มี source version, fixtures, commands, counts, failures/defects และหลักฐาน; ไม่อ้างว่าทดสอบครบถ้ายังมี mandatory case Not run/Blocked

Performance budget เสนอให้ตั้งใน Phase 0 บนเครื่อง/เครือข่ายที่ระบุ เช่น local deterministic page usable p95 ≤3 วินาที และ filter/sort D8 p95 ≤500 ms แยก provider scan/refresh latency ออก บันทึกผลหลายรอบ ไม่ตัดสินจากหนึ่งครั้งหรือกำหนด threshold โดยไม่ระบุ environment

### แบบบันทึกผลต่อกรณี

| Field | รายละเอียด |
|---|---|
| Test ID / variant | เช่น IMP-06 / Dime / partial selection / QA-A |
| Source | commit hash + diff snapshot ที่ใช้รัน |
| Environment / fixture | browser, timezone, provider mock/live, DB/vault paths ทดสอบ |
| Preconditions / steps | ข้อมูลก่อนทำรายการและขั้นตอนที่ทำจริง |
| Expected / Actual | รวม numeric state/ledger/count/payload เมื่อเกี่ยวข้อง |
| Status | Passed / Failed / Blocked / Not run |
| Evidence | screenshot/trace/log/report พร้อมปิดบัง credentials/ข้อมูลส่วนตัว |
| Defect / owner / retest | issue ID, severity, ผู้รับผิดชอบ, รุ่นและผล retest |

ลำดับ triage: เงิน/ข้อมูล/isolation → trade/import/replay/reset → CRUD/navigation/data refresh → layout รายงาน failure ของ provider/credential แยกจาก bug ของแอป และรัน regression ซ้ำเฉพาะส่วนที่เปลี่ยนหรือมีความเสี่ยงร่วมก่อนทำ final pass
