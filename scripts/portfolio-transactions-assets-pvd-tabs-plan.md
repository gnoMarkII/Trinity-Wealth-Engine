# แผนแยกแท็บสินทรัพย์และ PVD ภายใน Transactions

วันที่จัดทำ: 2026-10-13  
วันที่ตรวจและปรับแผน: 2026-10-14  
สถานะ: พร้อมเริ่มเขียนโค้ดตามสัญญาและลำดับงานในเอกสารนี้; การแยกแท็บตาม checklist ระยะ 1–4 ยังไม่ได้เริ่ม  
ขอบเขต: `Portfolio → Transactions → [สินทรัพย์ | PVD]`  
เอกสารอ้างอิง: [แผนนำเข้า PVD](pvd-transaction-import-hexagonal-plan.md), [ตัวอย่างหน้าตา PVD](../work/visualizations/pvd-statement-preview.html)

แผนนี้ต่อยอด PVD import ที่มีโค้ดอยู่ใน working tree แล้ว ไม่เริ่มสร้าง parser/service/ledger ใหม่ สถานะ “ยังไม่ได้เพิ่มฟีเจอร์ PVD” ในแผนนำเข้าเดิมไม่ใช่สถานะโค้ดปัจจุบัน ผลตรวจความเข้ากันได้และข้อจำกัดของหลักฐานอยู่ในหัวข้อ 13

## 1. ผลลัพธ์ที่ผู้ใช้จะได้รับ

หน้า Transactions มีแท็บย่อยสองแท็บอยู่ภายในหน้าเดิม:

| แท็บย่อย | ข้อมูลหลัก | หน่วยของแถว | เครื่องมือ |
| --- | --- | --- | --- |
| สินทรัพย์ | ประวัติรายการของสินทรัพย์ทั่วไป เช่น BUY, SELL และ reversal เดิม | หนึ่งรายการธุรกรรม | เพิ่มรายการ, Sync โบรกเกอร์, ค้นหา, กรอง action, สกุลเงิน, แก้ไข, Note, Void |
| PVD | ประวัติยอดตาม Statement และการเปลี่ยนแปลงรายกองทุน | หนึ่ง Statement | อัปเดต PVD, ค้นหากองทุน, ขยายรายละเอียดงวด, เปิด PDF |

ค่าเริ่มต้นเป็นแท็บ **สินทรัพย์**; เมื่อเปิดประวัติของ symbol `PVD_*` จาก Holdings ให้เปิดแท็บ **PVD** และค้นหากองทุนนั้นอัตโนมัติ

งานนี้แยกตารางและเครื่องมือบนหน้าจอ ใช้ transaction stream และ API เดิมเป็นแหล่งข้อมูล โดยเพิ่ม field อ่านอย่างเดียวที่จำเป็นสำหรับความแม่นยำ ไม่มีแท็บ PVD ใหม่ระดับเดียวกับ Holdings และไม่มีการสร้าง SELL/BUY ย้อนจากส่วนต่างระหว่าง Statement

## 2. สิ่งที่พบจากโค้ดปัจจุบัน

| ไฟล์ | สิ่งที่พบ | งานที่ต้องทำ |
| --- | --- | --- |
| `web/src/components/portfolio/PortfolioTransactionsTab.tsx` | ตารางเดียวรวมรายการสินทรัพย์กับ `PVD_SNAPSHOT`; ใช้คอลัมน์ Shares, Price, Fee/Tax, Summ และ Realized PnL ร่วมกัน | แยก coordinator และตารางสองชนิด; แยก summary/filter/action ของแต่ละแท็บ |
| `web/src/pages/Portfolio.tsx` | ส่ง `initialSymbol` จาก query `symbol`; เปิดประวัติด้วย `handleViewTransactionsForSymbol()` | เพิ่มสถานะแท็บย่อยและ URL navigation ภายใน Transactions |
| `tools/portfolio/services/pvd_sync_service.py` | หนึ่ง Statement สร้างหลายแถว; `Confirmation_No` คือ semantic fingerprint ของทั้ง Statement; `Fingerprint` ของแต่ละแถวแตกต่างกัน | ใช้ identity ระดับ Statement เพื่อจัดกลุ่ม ไม่ใช้ row fingerprint เป็น group key |
| `PVD_Metadata_JSON` | มีวันที่, scope, NAV รวม, เงินต้นสะสม, fingerprint และ archive reference ซ้ำในทุกแถว | ตรวจความสอดคล้องและอ่าน summary เพียงครั้งเดียวต่อ Statement |
| `api/schemas/portfolio.py` | `TransactionItemDTO` มี `pvd_metadata_json` และ `gross_amount` แต่ยังไม่มี `reconciliation_adjustment` รายกองทุน | เพิ่ม optional read field และอัปเดต TypeScript/OpenAPI contracts |
| `web/src/api/client.ts` | มี `getTransactions()` และ `getPvdDocumentUrl(archiveId, portfolioId)` | ใช้วิธีอ่านและเปิด PDF ที่มีอยู่ |
| `api/routers/portfolio/router_ledger.py` | response ส่งรายการทั้งหมดของพอร์ต; count/ยอดซื้อขายรวม BUY/SELL ส่วน realized summary รวมทุกแถวที่มีค่า | ใช้ response เดิม; แบ่งหน้า PVD หลังจัดกลุ่ม Statement ใน frontend |
| `web/src/pages/Portfolio.tsx` | มี `fetchIdRef` กัน stale fetch ของหน้าหลัก แต่ `handlePortfolioStateSuccess()` ยังรับ state โดยไม่ตรวจพอร์ตต้นทาง | coordinator ต้องตรวจอายุ request และพอร์ต ก่อนส่ง callback ของ mutation ให้หน้าหลัก |
| OpenAPI baseline | routes เก่าทั้ง 135 รายการคงสัญญาเดิม แต่ manifest ขาด 32 routes ที่มีใน HEAD และ 3 routes PVD | ปรับเฉพาะรายการ routes ที่เพิ่มหลังตรวจเทียบ; ผลตรวจและคำสั่งอยู่ในหัวข้อ 13 |

## 3. ขอบเขตงานและเงื่อนไขที่ต้องรักษา

- ยอดถือครอง PVD ล่าสุดยังมาจาก snapshot/replay เดิม
- ประวัติ PVD อ่านอย่างเดียว; ตารางใหม่ไม่มีปุ่มแก้ไขราคา/จำนวนหน่วยหรือ Void รายกองทุน
- Modal นำเข้า PVD และการตรวจเอกสารซ้ำ/ย้อนหลัง/สมาชิก/กองทุนที่ขาด ยังคงใช้ service เดิม
- กรณีแถวกองทุนเดิมหายไป ไม่ตีความเองว่าเป็นศูนย์หรือเกิด SELL; แสดงข้อมูลขาดและใช้ guard เดิมตอนนำเข้า
- ตารางสินทรัพย์ยังรองรับ action เดิมทั้งหมดที่ไม่ใช่ PVD รวม reversal; ไม่กรองทิ้งให้เหลือเพียง BUY/SELL
- รักษา `portfolio_id` ตลอดการโหลด รายละเอียด PDF การนำเข้า และการส่ง `onSuccess`
- ใช้ข้อมูลตัวอย่างสังเคราะห์ใน automated tests; PDF จริงและข้อมูลสมาชิกไม่เพิ่มเข้า repository

## 4. โครงสร้างหน้าจอ

### 4.1 ส่วนกลางของ Transactions

1. แถบเลือกแท็บย่อย **สินทรัพย์ / PVD** ใช้ native buttons ตามรูปแบบ UI ของโครงการ
2. โหลดรายการ transactions ครั้งเดียวต่อพอร์ต; การสลับแท็บไม่เรียก API ซ้ำโดยไม่มีเหตุจำเป็น
3. loading/error แสดงสถานะการโหลดร่วมกัน ส่วน empty state แสดงตามแท็บที่เลือก
4. แยก state ของคำค้น sort และ pagination ระหว่างแท็บ; สลับกลับแล้วรักษาตัวกรองของแท็บเดิมในพอร์ตเดียวกัน
5. เมื่อเปลี่ยนพอร์ต ให้ล้างข้อมูลเก่าและป้องกัน response ของพอร์ตเดิมเขียนทับพอร์ตใหม่; ปิด dialog ที่ผูกกับพอร์ตเดิม

คง component ของทั้งสอง `tabpanel` ไว้และใช้ `hidden` สำหรับแผงที่ไม่ active เพื่อรักษาตัวกรองภายในพอร์ตเดียวกัน ปิด dialog ของแผงเดิมเมื่อสลับแท็บ และ reset component state เมื่อ portfolio เปลี่ยน; child components ไม่โหลด transaction list ซ้ำเอง

### 4.2 ตารางสินทรัพย์

- แสดงเฉพาะรายการที่ไม่ถูกจัดประเภทเป็น PVD
- คงคอลัมน์และ workflow ซื้อขายเดิม รวมถึง Note, Edit, Void และ Sync โบรกเกอร์
- ตัวกรอง action ใช้ All/Buy/Sell เดิมสำหรับสินทรัพย์; ย้ายปุ่ม PVD เดิมออกจาก action filter มาเป็นแท็บย่อย
- Summary Buy/Sell/Realized PnL และ footer คำนวณจากธุรกรรมสินทรัพย์ตามเงื่อนไขเดิม
- Summary ด้านบนสร้างจาก asset partition เพื่อไม่พึ่ง API summary ในกรณีข้อมูลเก่ามี PVD ที่ action ผิดประเภท: count/ยอดซื้อขายใช้ BUY/SELL; realized summary รวมค่าใน asset rows ตามสูตรเดิมของ router; footer รักษาสูตรเดิมหลังกรอง
- ปุ่มอัปเดต PVD และ summary ของ PVD แสดงในแท็บ PVD

### 4.3 ตาราง PVD ระดับ Statement

แสดงงวดใหม่ก่อน และค่าเริ่มต้นขยายรายละเอียดงวดล่าสุดที่มีข้อมูลสมบูรณ์

| คอลัมน์ | ความหมาย |
| --- | --- |
| วันที่ Statement | `as_of_date` แสดงเป็นวันที่ไทย; ไม่มีเวลาธุรกรรมสมมติ |
| กองทุนใน Statement | จำนวนรหัสกองทุนที่ไม่ซ้ำ แยกจำนวนที่มียอดและยอดศูนย์ในข้อความประกอบ |
| ยอดรวม PVD | `total_nav` ของ Statement หนึ่งครั้งต่อกลุ่ม |
| เปลี่ยนจากงวดก่อน | NAV รวมงวดนี้ลบงวดก่อนใน scope เดียวกัน |
| รายละเอียด | ขยาย/ย่อรายกองทุน และปุ่มเปิด PDF ของงวดนั้น |

การ์ดด้านบนแสดง **ยอด PVD ล่าสุด** พร้อมวันที่ และ **ยอดเปลี่ยนจากงวดก่อน** พร้อมวันที่ฐานเปรียบเทียบ โดยไม่บวกยอดของหลาย Statement เข้าด้วยกัน

ข้อความประกอบส่วนต่าง: “ส่วนต่างแสดงการเปลี่ยนแปลงของยอดคงเหลือระหว่างงวด รวมเงินสะสมและเงินสมทบ” หลีกเลี่ยงชื่อ “กำไร”, “Realized PnL”, “เงินขายกองทุน” หรือ “มูลค่าการสับเปลี่ยน” สำหรับส่วนต่างนี้

### 4.4 รายละเอียดรายกองทุน

| คอลัมน์ | การแสดงผล |
| --- | --- |
| กองทุน | รหัสกองทุนย่อย เช่น `SINSA_GTO`; ตัด prefix `PVD_` ในข้อความแสดงผล |
| ยอดงวดก่อน | มูลค่าของกองทุนในงวดฐาน; กองทุนที่ไม่มีแถวในงวดก่อนแสดง `—` |
| ยอดงวดนี้ | มูลค่าที่ปรับกระทบยอดแล้วในงวดปัจจุบัน |
| ส่วนต่าง | คำนวณจากข้อมูลครบสองงวดตามข้อ 5 |
| สถานะยอด | ข้อความที่อธิบายการเปลี่ยนแปลงโดยไม่อ้างว่าเกิดซื้อหรือขายจริง |

สถานะที่ต้องมี:

- **กองทุนเพิ่มในงวดนี้**: รหัสไม่ปรากฏในงวดก่อนที่สมบูรณ์ และเริ่มปรากฏพร้อมยอดในงวดนี้
- **ยอดเป็นศูนย์**: งวดก่อนมียอด/หน่วย และแถวในงวดนี้มีทั้งยอดและหน่วยเป็นศูนย์
- **ยอดศูนย์ทั้งสองงวด**: แถวอยู่ครบและยอด/หน่วยเป็นศูนย์ทั้งสองงวด
- **ยอดเพิ่ม / ยอดลด / คงเดิม**: เปรียบเทียบมูลค่าระหว่างงวด
- **ไม่พบแถวใน Statement ใหม่**: รหัสเดิมหายไป; แสดงยอดปัจจุบันและส่วนต่างเป็น `—` พร้อมข้อความตรวจสอบข้อมูล

งวดแรกแสดงจำนวนหน่วย, NAV ต่อหน่วย และมูลค่ารายกองทุน พร้อมชื่อ “งวดเริ่มต้นที่ใช้เปรียบเทียบ” ไม่สร้างยอดงวดก่อนเป็นศูนย์ทุกกองทุน

ข้อมูลจำนวนหน่วย/NAV และยอดก่อนปรับกระทบยอดเปิดดูได้ในรายละเอียดเพิ่มเติมของกองทุน โดยระบุวันที่ของค่าเหล่านั้นให้ชัดเจน

## 5. วิธีจัดกลุ่มและคำนวณข้อมูล

สร้าง pure helper `web/src/lib/pvdStatementHistory.ts` แยกจาก React component และ API client

### 5.1 การจำแนกรายการ

- ให้ helper เดียวเป็นผู้จำแนก PVD จาก action `PVD_SNAPSHOT`, source `BBLAM_PVD`, asset type `PVD` หรือ symbol `PVD_*` แบบ case-insensitive
- สัญญาณ PVD ที่ข้อมูลอื่นไม่สมบูรณ์ต้องแสดงเป็นรายการ PVD ที่ตรวจข้อมูลไม่ได้ ไม่เล็ดลอดไปในตารางสินทรัพย์
- รับเฉพาะ snapshot ที่ผ่านการตรวจข้อมูลในการสร้าง Statement group; action PVD อื่นที่ไม่รองรับแสดงข้อความอธิบายและไม่รวมยอด

### 5.2 การจัดกลุ่ม Statement

1. อ่าน `pvd_metadata_json` ด้วย safe parsing และตรวจชนิด/date/decimal fields ที่จำเป็น
2. Group key ใช้ portfolio scope ร่วมกับ `Confirmation_No` ของ Statement; ถ้าขาดให้ใช้ `metadata.semantic_fingerprint` ที่ตรวจแล้ว
3. ห้ามใช้ `tx.fingerprint` เพราะเป็น fingerprint รายแถว และห้ามใช้วันที่เพียงอย่างเดียว
4. ถ้าทั้งสอง fingerprint fields มีค่า ต้องตรงกัน; สมาชิก/fund scope/วันที่/NAV/archive reference ในทุกแถวของกลุ่มต้องสอดคล้องกัน
5. ตรวจรหัสกองทุนซ้ำภายในกลุ่ม; แถวซ้ำ transaction ID ที่เหมือนกันทุก field สามารถ deduplicate ได้ ส่วน duplicate ที่ขัดแย้งต้องแจ้งข้อมูลไม่สมบูรณ์
6. ตรวจผลรวมมูลค่าที่ปรับกระทบยอดแล้วกับ NAV รวมเป็นจำนวนสตางค์
7. ถ้า metadata ผิดรูปแบบหรือยอดไม่ตรง ให้แสดงงวด/รายการที่มีปัญหาและปิดการเปรียบเทียบของกลุ่มนั้น โดยไม่ทำให้ทั้งหน้า crash หรือซ่อนข้อมูลเงียบ ๆ

ข้อจำกัดของข้อมูลเดิม: metadata ไม่ได้บันทึกจำนวนแถวกองทุนที่คาดหวัง จึงไม่อ้างว่าตรวจพบแถวศูนย์ที่หายไปได้ทั้งหมดจากยอดรวมเพียงอย่างเดียว การอ่านรายการครบชุดและ import guard เดิมยังเป็นฐานของความสมบูรณ์

### 5.3 ตัวเลขและ reconciliation

- ยอดรายกองทุนใช้ `gross_amount` จาก Statement และ `reconciliation_adjustment` รายแถว; fallback ไป `net_amount` ได้เฉพาะรูปแบบ PVD เดิมที่ตรวจความสอดคล้องแล้ว
- ไม่คำนวณมูลค่าทดแทนด้วย `units × price` เพราะอาจต่างจากยอดที่ Statement ปัดเศษ
- เพิ่ม `reconciliation_adjustment: Optional[str] = None` ใน API `TransactionItemDTO` และ `reconciliation_adjustment?: string | null` ใน TypeScript DTO
- Field นี้อ่านคอลัมน์ `Reconciliation_Adjustment` ที่มีอยู่แล้วผ่านการ normalize keys; ไม่มี migration ของ ledger
- ข้อมูลเก่าที่ไม่มี field รายแถวและ metadata adjustment รวมเป็นศูนย์ใช้ adjustment ศูนย์ได้; ถ้า adjustment รวมไม่เป็นศูนย์แต่ค่ารายแถวขาด ให้แจ้งว่าเปรียบเทียบยอดที่ปรับแล้วไม่ได้ ห้ามเดาว่ากองทุนไหนรับเศษ
- คำนวณยอดและส่วนต่างด้วย signed integer cents (`bigint`) ใน helper; ตรวจ string/ค่าติดลบ/ค่าว่างก่อนแปลง และไม่ใช้ floating-point ในการบวกยอด
- กำหนด signed conversion ให้รองรับ adjustment เช่น `-0.01` โดยไม่เปลี่ยนเครื่องหมาย
- หน่วยและ NAV ใช้เพื่อแสดงรายละเอียดเท่านั้น ไม่รวมจำนวนหน่วยข้ามกองทุน

### 5.4 การเทียบงวด

1. เรียง Statement ทั้งหมดตามวันที่ภายใน member/fund scope ก่อนค้นหา sort สำหรับจอ หรือแบ่งหน้า
2. งวดฐานคือ Statement ก่อนหน้าทันทีใน scope เดียวกัน ไม่ใช่แถวก่อนหน้าบนหน้าจอ
3. ถ้างวดก่อนหน้ามีข้อมูลไม่สมบูรณ์ หรือมี Statement ขัดแย้งในวันที่เดียวกัน ให้ระงับส่วนต่างพร้อมเหตุผล; ไม่ข้ามไปเทียบงวดเก่าแล้วเรียกว่า “งวดก่อน”
4. NAV delta ใช้ `current.total_nav − previous.total_nav` และคงเครื่องหมายบวก/ลบ
5. กองทุนใหม่ใน snapshot ที่ครบ ใช้ฐานไม่มีการถือครองเพื่อคำนวณส่วนต่าง พร้อมแสดงยอดงวดก่อนเป็น `—` และสถานะ “กองทุนเพิ่มในงวดนี้”; แถวใหม่ที่มีหน่วย/มูลค่าศูนย์ใช้สถานะ “ยอดเป็นศูนย์”
6. รหัสเดิมที่หายไปจากงวดใหม่ใช้สถานะข้อมูลขาด ไม่ตั้งยอดศูนย์
7. แบ่งหน้าตามจำนวน Statement หลังสร้าง view model แล้ว ไม่แบ่งหน้าตาม snapshot rows

## 6. Navigation, state และการ refresh

- ใช้ query parameter `tx_view=assets|pvd` ร่วมกับ `tab=transactions` เพื่อคงแท็บย่อยเมื่อ refresh/back/forward
- ถ้า `tx_view` ว่าง ให้เลือกจาก `symbol`: `PVD_*` เปิด PVD; symbol อื่นหรือไม่มี symbol เปิดสินทรัพย์
- `handleViewTransactionsForSymbol()` ตั้ง `tab`, `tx_view` และ `symbol` ให้สอดคล้องกัน
- เมื่อผู้ใช้สลับแท็บเอง ให้ล้าง symbol filter ใน URL ที่เป็นของอีกประเภท เพื่อไม่ให้กลับไปแท็บเดิมอัตโนมัติหรือแสดงตารางว่างผิดประเภท
- Query ที่ไม่รองรับใช้ค่าเริ่มต้นอย่างปลอดภัย; state ใหม่ไม่เขียน raw member ID/hash ลง URL
- คำค้น PVD เลือก Statement ที่มีกองทุนตรงคำค้น และเน้นกองทุนที่พบในรายละเอียด โดยคงแถวกองทุนอื่น ยอดรวม และงวดฐานเปรียบเทียบจากข้อมูลเต็ม
- หลัง PVD commit เรียก `onSuccess` และ reload transactions แล้วเลือก/ขยายงวดล่าสุดจากข้อมูลที่ API ส่งกลับ
- หลังสินทรัพย์ Note/Edit/Void/Sync สำเร็จ refresh ตาม workflow เดิม และรักษาแท็บย่อยที่ผู้ใช้เลือก
- ป้องกัน stale response เมื่อพอร์ตเปลี่ยน โดยใช้ request generation หรือ abort/drop result ที่ไม่ตรง portfolio ปัจจุบัน

## 7. ไฟล์ที่วางแผนแก้

| ไฟล์ | งาน |
| --- | --- |
| `web/src/components/portfolio/PortfolioTransactionsTab.tsx` | เป็น coordinator: โหลดข้อมูล แบ่งประเภท เลือกแท็บย่อย และจัดการ refresh |
| `web/src/components/portfolio/AssetTransactionsTable.tsx` (ใหม่) | ย้าย UI และ workflow ของสินทรัพย์จากตารางเดิม รักษาพฤติกรรมเดิม |
| `web/src/components/portfolio/PvdStatementsTable.tsx` (ใหม่) | Summary PVD, รายการ Statement, รายละเอียดรายกองทุน, PDF และปุ่มเปิด import modal |
| `web/src/lib/pvdStatementHistory.ts` (ใหม่) | จำแนกรายการ safe parsing จัดกลุ่ม exact cents และเปรียบเทียบงวด |
| `web/src/pages/Portfolio.tsx` | Navigation ของแท็บย่อยและลิงก์ประวัติจาก Holdings |
| `api/schemas/portfolio.py` | เพิ่ม optional field `reconciliation_adjustment` ของ transaction response |
| `web/src/api/types.ts` | ประกาศ read field ให้ตรง API |
| `web/openapi.json`, `web/src/api/types.generated.ts` | Regenerate จาก schema ตาม workflow โครงการ |
| `web/src/components/portfolio/PortfolioTransactionsTab.test.tsx` | ทดสอบแบ่งแท็บและ workflow เดิม พร้อม initial symbol/portfolio changes |
| `web/src/components/portfolio/PvdStatementsTable.test.tsx` (ใหม่) | ทดสอบ grouped rows, expand, search, pagination, PDF และ import refresh |
| `web/src/lib/pvdStatementHistory.test.ts` (ใหม่) | ทดสอบการคำนวณ identity chronology invalid data และ reconciliation |
| `web/src/pages/Portfolio.test.tsx` | เพิ่มจาก 3 tests เดิม: query `tx_view`, ประวัติจาก Holdings, back/forward และ scope ของ callback |
| `tests/api/test_portfolio_actual_routes.py` | เพิ่ม read-contract test สำหรับ adjustment และค่า default ของธุรกรรมเดิม |

ไฟล์ backend service/parser/replay ใช้พฤติกรรมที่มีอยู่; ถ้าพบข้อผิดพลาดใหม่ระหว่าง implementation ให้บันทึกเหตุผลและเพิ่มขอบเขตอย่างเจาะจงก่อนแก้

## 8. ลำดับงาน

### ระยะที่ 1: Contract และ helper

เริ่มจาก DTO/read-contract แล้วจึง helper ตามสัญญาหัวข้อ 12.1–12.3 เกณฑ์จบระยะ: helper tests ครอบคลุม TX-PVD-01–07, 09–12, 15 และ generated contract ไม่มี drift โดย UI เดิมยังผ่าน

- [x] เพิ่ม transaction DTO adjustment แบบ optional พร้อม read-contract test
- [x] Regenerate OpenAPI และ generated types
- [x] สร้าง helper ที่จำแนก PVD และแปลง signed money เป็น integer cents
- [x] สร้าง grouping/view models พร้อมสถานะข้อมูลไม่สมบูรณ์
- [x] สร้าง previous-statement comparison ก่อนค้นหา/แบ่งหน้า

### ระยะที่ 2: แยกตารางและแท็บย่อย

ทำหลัง helper ผ่าน แยกสินทรัพย์ด้วยการย้ายพฤติกรรมเดิมก่อนเชื่อม PVD ตาม props ในหัวข้อ 12.4 เกณฑ์จบระยะ: ตารางหลัก PVD มีหนึ่งแถวต่องวดและ workflow สินทรัพย์เดิมผ่าน; import callback ส่ง state ให้หน้าหลักเพียงครั้งเดียว

- [x] ย้ายสินทรัพย์ไป `AssetTransactionsTable` พร้อม workflow/filter/state เดิม
- [x] เพิ่มแท็บย่อยสินทรัพย์/PVD ใน coordinator และโหลดข้อมูลร่วมกัน
- [x] สร้าง PVD summary และตารางหนึ่ง Statement ต่อแถว
- [x] เพิ่มรายละเอียดรายกองทุนและสถานะยอดที่อธิบายได้
- [x] วางปุ่ม PVD import/PDF ในแท็บ PVD และเชื่อม refresh หลัง commit

### ระยะที่ 3: Navigation และความพร้อมใช้งาน

เชื่อม route หลัง coordinator ใช้งานได้ ตามหัวข้อ 12.5 เกณฑ์จบระยะ: URL/keyboard tests ผ่าน รวม deferred responses ทั้ง success/error/finally และ mutation ที่เสร็จหลังเปลี่ยนพอร์ต

- [x] เชื่อม query `tx_view` กับ route และ initial symbol จาก Holdings
- [x] รักษาตัวกรองแยกแท็บ และล้างข้อมูลที่ข้ามพอร์ต
- [x] รองรับ request ที่เสร็จผิดลำดับและ loading/error/empty state
- [x] ทำแท็บแบบ `tablist/tab/tabpanel` พร้อม keyboard navigation และ focus ที่มองเห็นได้
- [x] ปุ่มขยายใช้ `aria-expanded`/`aria-controls`; กดได้ด้วย keyboard
- [ ] ตรวจจอ 320–360px และ desktop รวม light/dark mode; บนมือถือปุ่มรายละเอียดต้องอยู่ในจอ และตารางรายละเอียดเลื่อนได้ภายในบริเวณของตาราง

### ระยะที่ 4: Regression และ acceptance

รันคำสั่งหัวข้อ 10 และตรวจ browser/PDF ตามเกณฑ์หัวข้อ 9 บันทึกผลแยกจาก baseline ก่อนพัฒนาในหัวข้อ 13 ไม่ใช้ผล baseline แทน acceptance ของ UI ที่ยังไม่ได้สร้าง

- [x] ทดสอบสินทรัพย์เดิม: Summary, BUY/SELL filter, Native/THB, Note, Edit, Void, Sync
- [x] ทดสอบ PVD grouping และส่วนต่างด้วย fixtures สังเคราะห์
- [x] ทดสอบค้นหา/แบ่งหน้าแล้วงวดฐานและยอดรวมไม่เปลี่ยน
- [x] ทดสอบยอดศูนย์ กองทุนใหม่ ข้อมูลขาด JSON เสีย และ adjustment ติดลบ
- [x] ทดสอบเปิด PDF ส่ง archive ID และ portfolio ID ของงวดที่เลือก
- [x] ทดสอบหลัง import/portfolio switch/refresh/back/forward
- [x] ตรวจ contract drift และ typecheck/build
- [ ] ตรวจ visual QA ใน browser ทั้ง mobile/desktop และ light/dark mode
- [ ] ตรวจ acceptance กับ PDF สองงวดจริงในพื้นที่ทดสอบชั่วคราว

## 9. เกณฑ์ทดสอบสำคัญ

| รหัส | กรณี | ผลที่ต้องได้ |
| --- | --- | --- |
| TX-PVD-01 | Response มี BUY/SELL/reversal และ PVD snapshots ปนกัน รวม PVD legacy ที่ action ผิดประเภท | สินทรัพย์ไม่เห็น PVD; PVD ไม่เห็นธุรกรรมสินทรัพย์; summary ไม่ปนกัน; PVD ที่ action ไม่รองรับมีข้อความตรวจข้อมูล |
| TX-PVD-02 | Statement สองงวดมี 7 และ 8 subfund rows | ตารางหลักมี 2 แถวและแบ่งหน้าตามงวด ไม่ใช่ 15 แถว |
| TX-PVD-03 | งวดใหม่เพิ่ม `SINSA_GTO` | รายละเอียดแสดง “กองทุนเพิ่มในงวดนี้”; ไม่มี action BUY สังเคราะห์ |
| TX-PVD-04 | `SINSA1` และ `SINSA_BTP` จากยอดบวกเป็นหน่วย/มูลค่าศูนย์ | แสดง “ยอดเป็นศูนย์” พร้อมส่วนต่างติดลบ; ไม่มี SELL สังเคราะห์ |
| TX-PVD-05 | กองทุนเดิมหายจากงวดใหม่โดยไม่มีแถวศูนย์ | ข้อความข้อมูลขาดและส่วนต่างรายกองทุนนั้นเป็น `—` |
| TX-PVD-06 | ค้นหากองทุนที่เริ่มปรากฏในงวดใหม่ ทำให้งวดเก่าไม่ปรากฏในผลค้นหา | งวดฐานยังเป็นงวดเก่าจริง; NAV delta และ summary ใช้ข้อมูลเต็ม |
| TX-PVD-07 | เปลี่ยน sort/หน้า/ขนาดหน้า | previous statement อ้างอิง chronology เดิมเสมอ |
| TX-PVD-08 | Initial `symbol=PVD_SINSA_GTO` และ stock symbol | เปิดแท็บที่ถูกต้องและค้นหาที่ถูกประเภท |
| TX-PVD-09 | metadata JSON เสีย/ขาด/ซ้ำขัดแย้ง/identity ไม่ตรง | UI ยังทำงาน แจ้งข้อมูลไม่สมบูรณ์ และไม่รวมยอดผิด |
| TX-PVD-10 | adjustment รายกองทุน +0.01/-0.01 | ผลรวม detail ตรง total NAV ทุกสตางค์ และเครื่องหมายไม่เปลี่ยน |
| TX-PVD-11 | raw market value เป็นศูนย์หรือ NAV × units ต่างจาก Statement | ใช้ explicit value ศูนย์และยอดตาม Statement ไม่ใช้ fallback ที่ทำให้ยอดเปลี่ยน |
| TX-PVD-12 | การ์ดล่าสุดและหลาย Statement | ยอดล่าสุดไม่ใช่ผลบวกของยอดทุกงวด; ส่วนต่างไม่ถูกเรียกว่ากำไร |
| TX-PVD-13 | เปลี่ยนพอร์ตก่อน response เก่าเสร็จ/PDF/commit callback | แสดงและอัปเดตเฉพาะพอร์ตที่ตรง request; ไม่ใช้ข้อมูลข้ามพอร์ต |
| TX-PVD-14 | refresh/back/forward และสลับแท็บด้วย keyboard | view และ focus ถูกต้อง; ไม่มี URL member ID/hash |
| TX-PVD-15 | DTO ใหม่กับรายการสินทรัพย์เดิมที่ไม่มี adjustment | default compatibility ถูกต้อง และ BUY/SELL summary ไม่เปลี่ยน |

### ตรวจด้วย PDF ที่ผู้ใช้ให้ไว้

ใช้ไฟล์ `ใบแจ้งยอดเงินของสมาชิก_25681231.pdf` และ `ใบแจ้งยอดเงินของสมาชิก_25690630.pdf` ใน Downloads ตามเส้นทางที่ผู้ใช้ระบุ นำเข้าแบบต่อเนื่องใน temporary portfolio เท่านั้น:

- งวด 31 ธ.ค. 2568 มี 7 กองทุน และงวด 30 มิ.ย. 2569 มี 8 กองทุน
- งวดใหม่มี `SINSA_GTO` เพิ่ม และ `SINSA1`/`SINSA_BTP` แสดงแถวศูนย์
- UI แสดง Statement สองแถว รายละเอียดตรงกับ PDF และ NAV delta ตรงกับการลบยอดรวมสองงวด
- จำนวนล่าสุดคือ 6 กองทุนมียอดคงเหลือและ 2 กองทุนยอดศูนย์
- เปิด PDF ได้ตรงงวด และตรวจ source bytes/hash ในพื้นที่ทดสอบ
- ไม่คัดลอก PDF จริง ชื่อสมาชิก เลขสมาชิก หรือรหัสผ่านลง fixtures/screenshots ที่เตรียม commit

## 10. คำสั่งตรวจสอบเมื่อพัฒนาแล้ว

รันจากโฟลเดอร์ `web`:

```powershell
npm run gen:types
npm run test -- src/components/portfolio src/lib/pvdStatementHistory.test.ts src/pages/Portfolio.test.tsx src/api/client.test.ts
npm run check:types
npm run build
```

รันจาก root ของโครงการสำหรับ contract ที่เปลี่ยน:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/api/test_portfolio_actual_routes.py tests/api/test_openapi_contract.py tests/integration/test_pvd_atomic_recovery.py -q -o addopts=''
git diff --check
```

ตรวจ browser จริงเพิ่มเติมสำหรับ expand/collapse, การลิงก์จาก Holdings, URL navigation, portfolio switch และ responsive layout ไม่มีเหตุจำเป็นต้องรันชุด Python ทั้งโครงการสำหรับการเขียนแผนนี้

## 11. เงื่อนไขเสร็จงาน

- Checklist ระยะ 1–4 ผ่านและมีผลตรวจที่บันทึกไว้
- Transactions มีสองแท็บย่อยตามที่ผู้ใช้ขอ และตาราง/summary/action ไม่ปนประเภท
- หนึ่ง Statement เป็นหนึ่งแถว พร้อมยอดรายกองทุนและส่วนต่างที่อ้างอิงงวดก่อนถูกต้อง
- Workflow สินทรัพย์เดิมผ่าน regression และ PVD snapshot/history/guard เดิมยังทำงาน
- รูปแบบตัวเลข การเปิด PDF การเปลี่ยนพอร์ต และตัวอย่างสับเปลี่ยนสองงวดผ่าน acceptance

เมื่อพัฒนาตามแผน ให้เปลี่ยนสถานะและติ๊ก checklist ตามหลักฐานที่ผ่านจริง ไม่ถือว่าการมี mockup หรือเอกสารแผนเท่ากับ implementation เสร็จ

## 12. สัญญาสำหรับลงมือเขียนโค้ด

### 12.1 API และ backward compatibility

- GET `/api/portfolio/actual/transactions?portfolio_id=...` ยังคืน `{ portfolio_id, transactions, summary }` เหมือนเดิม; โหลดทั้งพอร์ตโดยไม่ส่ง `symbol` เพื่อไม่ตัดงวดฐานออกก่อนเปรียบเทียบ
- เพิ่มเพียง `TransactionItemDTO.reconciliation_adjustment: Optional[str] = None` และ field optional ตรงกันใน `web/src/api/types.ts`; ไม่ทำให้ client เก่าต้องส่ง field ใหม่
- `_normalize_keys()` เดิมแปลง `Reconciliation_Adjustment` เป็น `reconciliation_adjustment` ได้อยู่แล้ว อย่าแก้ serializer/header หรือค่าของ BUY/SELL เพื่อเพิ่ม read field นี้
- แถวเก่าที่ไม่มี field ต้องคืน `null`; field ที่เป็น string ต้องคงค่า `0.00`, `0.01`, `-0.01` ไว้ครบ ไม่มีการแปลงเป็น float ใน DTO
- GET, PATCH note และ response ที่ใช้ `TransactionItemDTO` ต้องผ่าน read-contract test ทั้งแถว PVD และแถวสินทรัพย์เดิม
- `web/openapi.json` และ `types.generated.ts` ต้องมาจาก `npm run gen:types` ส่วน `types.ts` ยังเป็น interface ที่แก้ด้วยมือ; `check:types` ตรวจ generated file ไม่ได้ตรวจ interface นี้แทน
- ไม่มี endpoint ใหม่ในงานแยกแท็บนี้ ใช้ analyze/commit/PDF เดิม และไม่เปลี่ยนสูตร summary ของ backend

### 12.2 Pure helper และ view model

export จาก `web/src/lib/pvdStatementHistory.ts` ให้มีหน้าที่ชัดเจนดังนี้:

```ts
import type { TransactionItemDTO } from '../api/types'

export type TransactionView = 'assets' | 'pvd'
export type PvdIssue = {
  code: string
  message: string
  transactionIds: string[]
}
export type PvdFundRow = {
  symbol: string
  code: string
  units: number
  nav: number
  rawValueCents: bigint
  adjustmentCents: bigint
  valueCents: bigint
}
export type PvdFundComparison = {
  code: string
  previous: PvdFundRow | null
  current: PvdFundRow | null
  deltaCents: bigint | null
  status: 'baseline' | 'new' | 'zero' | 'zero_both'
    | 'increase' | 'decrease' | 'unchanged' | 'missing' | 'unavailable'
}
export type PvdStatementGroup = {
  id: string
  scopeKey: string | null
  asOfDate: string | null
  archiveId: string | null
  totalNavCents: bigint | null
  isValid: boolean
  funds: PvdFundRow[]
  issues: PvdIssue[]
  previousId: string | null
  deltaNavCents: bigint | null
  comparisons: PvdFundComparison[]
}
export type PvdHistory = {
  statements: PvdStatementGroup[]
  unresolvedIssues: PvdIssue[]
  latestStatementId: string | null
  summaryUnavailableReason: string | null
}

// Classification partitions every input row exactly once.
export declare function isPvdTransaction(tx: TransactionItemDTO): boolean
export declare function partitionTransactions(rows: readonly TransactionItemDTO[]): {
  assets: TransactionItemDTO[]
  pvd: TransactionItemDTO[]
}
export declare function parseMoneyCents(value: string): bigint
export declare function formatMoneyCents(value: bigint, showPositiveSign?: boolean): string
export declare function buildPvdStatementHistory(
  portfolioId: string,
  rows: readonly TransactionItemDTO[],
): PvdHistory
```

โค้ดข้างบนเป็นสัญญา types/function signatures ไม่ใช่ implementation ให้ helper รับ DTO จาก API โดยไม่ import React, API client หรือเรียก request และไม่แก้ input rows

รายละเอียดที่ต้องใช้ร่วมกัน:

- Normalize action/source/asset type/symbol โดย trim และเทียบ case-insensitive; row ใดมีสัญญาณ PVD ต้องอยู่ partition PVD แม้ข้อมูลไม่ครบ
- Snapshot ปกติใช้ THB, หน่วยและ NAV เป็น finite number ไม่ติดลบ; metadata ใช้ `payload_version=1`, ISO date ที่เป็นวันจริง, fund/member scope และ semantic fingerprint ที่ถูกชนิด
- identity ภายใน group คือ portfolio + fingerprint ของ Statement; chronology scope คือ portfolio + `fund_code` + `member_hash` เก็บ identifiers เหล่านี้ภายใน helper ไม่เขียนลง URL/ข้อความผู้ใช้
- คง source transaction IDs ของ error เพื่อใช้ key/ข้อความรายการที่ตรวจไม่ได้; error ต้องไม่ทำให้แถวสูญหายหรือทำให้หน้า crash
- กองทุน sort ตาม code เพื่อให้ผล deterministic; Statement chronology ใช้ ISO date ภายใน scope ก่อน display sort/search/pagination
- แถว transaction ID เดียวที่ซ้ำเหมือนกันหลัง normalize deduplicate ได้; ID เดียวที่ค่าต่างกันหรือหลาย ID ใช้ fund code เดียวใน Statement เดียวเป็น conflict
- metadata ที่ใช้ตรวจความตรงกันมี scope/date/NAV/contribution/fingerprint/archive/source hash และ reconciliation รวม; ความต่างของ Notes ไม่ใช่ความต่างของ metadata
- semantic fingerprint, member hash, source hash และ archive ID ของข้อมูล version 1 ใช้ hexadecimal SHA-256 64 ตัว Normalize fingerprint เป็น lowercase ก่อนตรวจ identity; archive reference ที่ขาด/ผิดต้องมี issue และเปิด PDF ไม่ได้
- ถ้าข้อมูลแถวใดไม่ทราบ scope/date จนวาง chronology ไม่ได้ ให้เก็บใน `unresolvedIssues` และระงับการ์ดล่าสุด/ส่วนต่างพร้อมเหตุผล แทนการอ้างงวดเก่าที่เหลือว่าเป็นงวดล่าสุด
- ถ้างวดที่ใหม่สุดระบุได้แต่ invalid ต้องคงแถวนั้นและระงับ summary; สามารถขยายงวดสมบูรณ์ล่าสุดเพื่อดูรายละเอียดได้ แต่ห้ามเปลี่ยนความหมายของ “ยอดล่าสุด”
- ในข้อมูลปกติหนึ่งพอร์ตรองรับ scope เดียวตาม import guard; ถ้าพบหลาย scope ในข้อมูลอ่าน ต้องแสดงข้อขัดแย้งและระงับ summary รวม ห้ามรวมยอด/เปรียบเทียบข้ามสมาชิก

### 12.3 Money, validation และ chronology

`parseMoneyCents()` รับ decimal string ตามรูปแบบ `^-?\d+(?:\.\d{1,2})?$` เท่านั้นหลัง trim; ค่าว่าง, comma, exponent, NaN/Infinity และเกินสองตำแหน่งทศนิยมเป็น invalid ไม่ใช่ศูนย์ ไม่มีการปัดยอดใหม่ระหว่างอ่าน Statement

แยกเครื่องหมายก่อนแปลง whole/fraction แล้วคูณทั้งจำนวนด้วยเครื่องหมาย เช่น `-0.01 → -1n`, `-1.25 → -125n`; `formatMoneyCents()` จัดกลุ่มหลักและทศนิยมจาก bigint โดยตรง ไม่ผ่าน `Number(cents)` การใช้ number สำหรับ units/NAV เป็นรายละเอียดแสดงผลเท่านั้น

สูตรและ null behavior ที่ต้องทดสอบ:

| รายการ | สูตร/เงื่อนไข |
| --- | --- |
| มูลค่ากองทุน | `rawValueCents + adjustmentCents` ต้องไม่ติดลบ |
| กระทบยอด Statement | ผลรวมมูลค่ากองทุนตรง `metadata.total_nav` และผลรวม adjustment รายแถวตรง adjustment รวม |
| มูลค่าศูนย์ | `gross_amount='0.00'` เป็นค่าจริง; fallback เฉพาะ null/undefined/empty ของรูปแบบเก่าที่ผ่าน validation ไม่ใช้ truthiness ของค่าที่แปลงเป็น number |
| Fallback แบบเก่า | snapshot THB ที่ metadata version 1/identity/chronology ผ่านแต่ gross ขาด ใช้ net ที่เป็น decimal string ไม่ติดลบแล้วตรวจ sum เทียบ NAV; ถ้า gross มีค่าแต่ invalid ไม่ fallback ถ้าทั้ง gross/net มีค่าและไม่เท่ากัน ให้ issue |
| Adjustment แบบเก่า | null/undefined/empty ถือว่า field ขาด ไม่ใช่ค่าปรับที่ตรวจแล้ว; field ทุกแถวขาด + adjustment รวมเป็นศูนย์จึงใช้ศูนย์ได้; ถ้ารวมไม่เป็นศูนย์หรือมีค่าบางแถวแต่บางแถวขาด กลุ่ม invalid ไม่เดาค่าที่เหลือ; metadata version 1 ที่ไม่มี aggregate adjustment ใช้ default ศูนย์ตาม domain model เดิม |
| NAV delta | `current.totalNavCents - previous.totalNavCents` ใช้ Statement ก่อนหน้าทันทีที่ valid ใน scope เดียวกัน; ถ้างวดก่อน invalid ให้ delta null |
| กองทุนใหม่ | previous ไม่พบ code ในงวด valid ที่ครบ และ current มีแถวพร้อมยอด: ยอดก่อน `—`, delta เท่ายอดปัจจุบัน, status `new` |
| แถวศูนย์เพิ่มใหม่ | previous ไม่พบ code และ current มี units/value ศูนย์: status `zero`, ไม่อ้างว่ามีเงินเพิ่มหรือซื้อกองทุน |
| กองทุนเดิมหาย | previous มี code แต่ current ไม่มีแถว: current/delta null, status `missing`; ระงับ total comparison ระหว่างคู่นี้เพราะตรวจความครบรายกองทุนไม่ได้ |
| ไม่มีงวดก่อน | previous/delta null, status `baseline`; ไม่สร้างฐานศูนย์สมมติ |
| วันเดียวกันหลาย fingerprint | invalid/conflict ใน chronology และระงับ comparison ของงวดถัดไป; sort order ไม่ใช้ตัดสินเลือกงวดใดเป็นจริง |

parse error ต้องถูกจับใน `buildPvdStatementHistory()` และแปลงเป็น `PvdIssue` การค้นหาที่ซ่อนงวดฐานหรือการแบ่งหน้าจะไม่เปลี่ยน `previousId`, delta, comparisons หรือ summary เพราะคำนวณเสร็จก่อนแล้ว

### 12.4 Component boundaries และ callbacks

| Component | สัญญาและความรับผิดชอบ |
| --- | --- |
| `PortfolioTransactionsTab` | รับ props เดิมทั้งหมด เพิ่ม `activeView?: TransactionView`, `onViewChange?: (view: TransactionView) => void`; controlled view เมื่อ prop มีค่าและ fallback ภายในเป็น assets/PVD ตาม initial symbol สำหรับผู้เรียกเดิม |
| `AssetTransactionsTable` | รับ `portfolioId`, asset rows, `holdings`, `active`, `initialSymbol`, callback clear/open trade/success/refresh; รับผิดชอบ filter/sort/page/currency/Note/Edit/Void/Dime dialogs เดิม ไม่ GET transaction list เอง |
| `PvdStatementsTable` | รับ `portfolioId`, `history`, `active`, `initialSymbol`, callback clear/commit success; รับผิดชอบ PVD search/display sort/page/expanded IDs/import dialog และ PDF ไม่ GET transaction list เอง |

กำหนด callback ของการ refresh แยกจาก callback state: `onRefresh: () => Promise<void>` สำหรับ Note/refresh ที่ไม่มี state และ `onMutationSuccess: (state: ActualPortfolioStateDTO) => void` สำหรับ Edit/Void/Sync/PVD commit; coordinator ส่ง state ให้ `onSuccess` และ reload list ครั้งเดียว child ไม่เรียกสอง callback ซ้ำสำหรับ operation เดียว

- Coordinator เป็นผู้โหลดรายการเดียวและ memoize partitions/history; shared loading/error อยู่ด้านนอก tabpanels
- ตั้ง `key={selectedPortfolioId}` ที่ coordinator ใน `Portfolio.tsx` เพื่อ reset child filters/dialogs เมื่อเปลี่ยนพอร์ต ภายในพอร์ตคงสอง child components mounted และใช้ `hidden`/`active`
- `active=false` ปิด dialog ที่อยู่ใน child นั้นและยกเลิก inline edit; filter/sort/page ยังอยู่ Dialog ที่มี request ค้างต้องทิ้ง modal UI result หลังปิด แต่ commit สำเร็จยังส่ง state/refresh ผ่าน coordinator ที่ยังมีชีวิตในพอร์ตเดิมได้ โดยไม่บังคับผู้ใช้กลับมาแท็บ PVD
- ใช้ monotonic generation สำหรับ list load และ lifetime/request token สำหรับ mutations ตรวจทั้ง response portfolio ID (list), captured portfolio ID, coordinator lifetime และ request ล่าสุดก่อน set data/error/loading หรือเรียก `onSuccess`; modal state ตรวจ modal lifetime เพิ่มอีกชั้น
- ครอบ callback จาก PVD/Dime/Edit modal ด้วย scope check ด้วย ไม่ใช้เพียง key/remount เพราะ Promise ของ component เก่ายังเรียก callback ได้
- Reset page เป็น 1 เมื่อคำค้น/filter/page size เปลี่ยน; clamp page เมื่อ refresh ทำให้จำนวนหน้าลดลง; expanded IDs ต้องยังอยู่ในผลข้อมูลปัจจุบัน
- สินทรัพย์รักษา default page size 50 เดิม; PVD ใช้ 10 Statements ต่อหน้า (ตัวเลือก 10/20/50) และ sort วันที่ใหม่ก่อน; ตัวเลขกองทุนมียอดนับแถวที่ units หรือ adjusted value มากกว่าศูนย์ ส่วนแถวศูนย์ต้องเป็นศูนย์ทั้งคู่
- PVD commit เลือก/ขยายงวดล่าสุดหลัง reload สำเร็จจาก API ไม่ใช้ข้อมูล scan เป็น history ที่ commit แล้ว; reload ล้มเหลวให้แสดง retry โหลดรายการ โดยไม่สั่ง commit ซ้ำ
- Summary สินทรัพย์ใช้ asset partition เต็ม; footer ใช้ asset rows หลังกรองด้วยสูตรเดิม ห้ามรวม PVD snapshot หรือแปลง reversal เป็น BUY/SELL
- PDF เรียก `api.getPvdDocumentUrl(group.archiveId, portfolioId)` เฉพาะ archive reference ที่ผ่าน validation ปุ่ม disabled พร้อมเหตุผลเมื่อ reference ผิด/ขาด

### 12.5 Route และ accessibility

- `Portfolio.tsx` เป็นเจ้าของ query; `tx_view` ที่ถูกต้องมีลำดับเหนือ symbol inference ค่า query ที่ไม่รองรับใช้ inference เหมือน query ว่าง และไม่เขียน raw scope IDs
- ตัวอย่าง `?tab=transactions&symbol=PVD_SINSA_GTO` เปิด PVD; `?tab=transactions&tx_view=assets` เปิดสินทรัพย์; Holdings handler ตั้งทั้ง `tab`, `tx_view`, `symbol`
- ถ้า view ที่ระบุชัดขัดกับชนิด symbol ให้ไม่ส่ง symbol นั้นเป็น filter ให้ child ผิดประเภท; เมื่อสลับเองล้าง symbol ต่างประเภทใน URL และคง local filters ของแต่ละ child
- initial symbol ที่เปลี่ยนเป็นค่าใหม่จาก route ต้องปรับคำค้นและ page ของ child ประเภทนั้น การสลับแท็บที่ล้าง URL symbol ไม่ใช้เป็นคำสั่งล้าง local filters ทั้งสองแท็บ; ปุ่ม clear filter ต้องล้างทั้ง local search และ URL
- คำค้น PVD เทียบ case-insensitive ได้ทั้ง code ที่แสดงและ symbol เต็ม (`SINSA_GTO`/`PVD_SINSA_GTO`) โดย normalize prefix ก่อนค้นหา; อย่าตัดแถวกองทุนอื่นออกจากรายละเอียดของ Statement ที่ตรงคำค้น
- ใช้ `role=tablist`, `role=tab`, `aria-selected`, `aria-controls`, roving `tabIndex`; ArrowLeft/Right, Home/End เปลี่ยน focus และเลือกแท็บ; tabpanel มี `aria-labelledby`
- ตาราง PVD ใช้ header `<th scope=col>` และปุ่ม expand มี stable `aria-controls`; ID ของ DOM ใช้ local ID ที่ไม่ใส่ member hash
- Light/dark ใช้ conventions ของหน้า Portfolio เดิม; ค่า negative delta มีเครื่องหมายและข้อความ ไม่อาศัยสีอย่างเดียว; invalid/empty/loading แยกกันชัดเจน

### 12.6 Fixtures และ regression gates

ใช้ fixture สังเคราะห์สองงวดใน helper/UI tests: 7 และ 8 subfunds รวมกองทุนเพิ่ม แถวศูนย์ และมูลค่าที่ `units × NAV` ไม่ตรง raw value; fingerprints/source hashes ใช้ synthetic SHA-256 และ metadata ไม่ใช้สมาชิกจริง

เพิ่ม variants สำหรับ signed adjustment ±0.01, duplicate transaction ID, duplicate fund code, same-day conflict, JSON เสีย, missing scope, invalid latest และ invalid previous อย่างละกรณี; ใส่ BUY/SELL/VOID_SELL และ PVD legacy ที่ action/source/symbol ต่างรูปแบบลงใน response เดียวเพื่อทดสอบ partition

เพิ่ม deferred Promise tests สำหรับ list A → switch B → B เสร็จ → A success/error และ PVD/Edit/Void/Dime callback ของ A เสร็จหลังเปลี่ยนพอร์ต; callback เก่าต้องไม่เขียน state ของ B และไม่เปลี่ยน loading/error ของ B

ก่อนปิดแต่ละระยะให้รัน focused tests ของส่วนที่เปลี่ยน และปิดระยะ 4 ด้วย suite ในหัวข้อ 10 การเพิ่ม read field นี้ไม่ต้องรัน live providers หรือแก้ฐานข้อมูล production ผล PDF/browser acceptance ต้องระบุว่าใช้ temporary portfolio และข้อมูลจริงไม่อยู่ใน fixture/commit

## 13. ผลตรวจโค้ด PVD ปัจจุบันกับ Portfolio เดิม (2026-10-14)

ส่วนนี้เป็น baseline ก่อนแยกแท็บ ไม่ใช่ผลผ่าน checklist UI ใหม่

### 13.1 ขอบเขตที่ตรวจ

- Domain, PVD parser/archive/calculations/replay/guards/serialization และ unit tests ของ Portfolio เดิม
- Trading/ledger/cash flow/watchlist/goals/performance/multi-portfolio รวม Dime/WealthX/SCBAM/ARR Gold regression ที่อยู่ใน suites ด้านล่าง
- Transaction source, projection recovery และ Portfolio API เดิม/DTOs
- UI Portfolio เดิม, PVD modal, Dime และ API client รวม generated types และ production build
- เพิ่ม integration tests ใน `tests/integration/test_pvd_atomic_recovery.py`: นำเข้า PVD สองงวดในพอร์ตที่มี PTT/AAPL/เงินสด THB/USD/realized P&L, restart/recalc/DTO reload; Note/Edit/Void/ฝากเงินหลังมี PVD; reject PVD mutation และ mixed batch โดย state/ledger/checkpoint ไม่เปลี่ยน
- เพิ่ม mixed response test ใน `tests/api/test_portfolio_actual_routes.py`: API เดิมส่ง PVD metadata/gross value พร้อม BUY/SELL โดย count/ยอดซื้อขาย/realized summary ไม่รวม snapshot
- แก้ fixture PVD ให้ใช้ `monkeypatch.setattr()` กับ vault paths เพื่อคืนค่าหลัง test และไม่ทิ้ง reference ข้าม test
- Compose imports ของ fixture/service ตอนรัน test เพื่อใช้ domain/mutation classes รุ่นเดียวกัน หลัง fixtures Portfolio เดิม reload modules; ไม่เปลี่ยน production type checks เพื่อให้ test ผ่าน

### 13.2 OpenAPI baseline ที่ตรวจพบและแก้

การรันครั้งแรกพบ 441 tests ผ่าน และ 1 test ล้ม (`test_openapi_paths_and_operations`) เพราะ manifest ยังมี 135 paths แต่ API ปัจจุบันมี 170 paths ตรวจเทียบก่อนปรับแล้ว:

- ไม่พบ route เก่าถูกลบหรือ manifest ของ operation เดิมเปลี่ยน ทั้ง 135 paths ตรงเดิม
- 32 paths เพิ่มอยู่ใน `HEAD:web/openapi.json` แล้ว (Investor Essence/buckets/ARR Gold)
- 3 paths ที่เพิ่มใน working tree คือ PVD analyze/commit/document

ปรับ `tests/fixtures/manifest_openapi_schema.json` เพิ่มเฉพาะรายการที่ขาด รักษา operation เดิมครบ ไม่มีการ relax test ที่บังคับ exact path set หลักฐานก่อนปรับอยู่ใน [OpenAPI audit](../work/pvd-plan-readiness/openapi-audit.json)

### 13.3 คำสั่งและผลตรวจ

จาก root:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/portfolio tests/tools/portfolio tests/architecture/test_pvd_architecture.py tests/integration/test_pvd_atomic_recovery.py tests/integration/test_portfolio_atomic_recovery.py tests/integration/test_portfolio_projection_rebuild.py tests/integration/test_portfolio_service_di.py tests/api/test_portfolio_actual_routes.py tests/api/test_portfolio_actual_mutations.py tests/api/test_arr_gold_sync.py tests/api/test_openapi_contract.py -q -o addopts=''
```

ผลรันสุดท้ายหลังแก้ baseline และ fixture/เพิ่ม mixed portfolio tests: **446 passed** ใน 149.22 วินาที ไม่มี test fail (มี dependency deprecation warning 1 รายการ)

Architecture เพิ่มเติม:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/architecture/test_dependency_rules.py -q -o addopts=''
```

ผล: **19 passed**; รวม backend/architecture ที่ตรวจ 465 tests ผ่าน และ `git diff --check` ผ่าน

จาก `web`:

```powershell
npm run test -- src/components/portfolio src/api/client.test.ts
npm run test -- src/pages/Portfolio.test.tsx
npm run check:types
npm run build
```

- Frontend: 22 test files, 107 tests ผ่าน (component/client 104 และหน้า Portfolio เดิม 3)
- Architecture เดิม: `tests/architecture/test_dependency_rules.py` ผ่าน 19 tests เพิ่มเติม
- Generated types: ตรงกับ OpenAPI ไม่มี drift
- Build: TypeScript และ production build ผ่าน
- Vitest ต้องรันนอก sandbox หลัง startup ถูกบล็อกด้วย `spawn EPERM`; เป็นข้อจำกัดการสร้าง process ไม่ใช่ผล test fail

### 13.4 ข้อจำกัดและจุดเริ่มงาน

การ implementation ตามหัวข้อ 12 เสร็จแล้ว และ regression ปัจจุบันยืนยันการทำงานร่วมกับ Portfolio เดิมในขอบเขต automated tests ที่ระบุไว้ด้านบน ส่วน production/browser/PDF acceptance ยังไม่ยืนยัน; ดูผลการลงมือทำและข้อจำกัดที่เหลือในหัวข้อ 14

## 14. ผลการลงมือทำ (2026-10-14)

- Backend Portfolio/API/OpenAPI และ PVD atomic recovery: `18 passed`.
- Frontend regression: `24 test files, 124 tests passed`; ครอบคลุมตารางสินทรัพย์เดิม, route/History, portfolio-switch race, PVD grouping/search/pagination/import refresh และ document reference.
- `npm run check:types`, `npm run build` และ `git diff --check` ผ่าน.
- ยังไม่ได้ตรวจ visual QA ใน browser เพราะ Computer Use ไม่พบ browser ที่เปิดใช้งาน และยังไม่ได้ทดสอบ PDF จริงสองงวด เพราะ workspace ไม่มีไฟล์ PDF; checklist ข้อ 213, visual QA ในข้อ 225 และข้อ 226 จึงยังเปิดไว้.
