# CLAUDE.md — คู่มือควบคุมพฤติกรรมและสถาปัตยกรรมของ AI

> **Project:** `invest-agents` (Trinity-Wealth-Engine)
> **Scope:** เอกสารฉบับนี้เป็น "รัฐธรรมนูญ" ของ AI Coding Agent ภายในโปรเจกต์นี้
> **Audience:** Claude / AI Assistant ทุกตัวที่ทำงานกับ Codebase นี้
> **Principle:** เน้นที่ *หลักคิด* และ *กฎเหล็กเชิงสถาปัตยกรรม* ไม่ใช่การกำหนดชื่อตัวแปรหรือสูตรคำนวณตายตัว — AI มีอิสระเลือกแนวทางวิศวกรรมที่ดีที่สุด ตราบใดที่ไม่ละเมิดกรอบความถูกต้องทางการเงิน

---

## 1. ปรัชญาหลักในการเขียนโค้ด (Core Coding Philosophies)

### 1.1 Think Before Coding — คิดให้จบก่อนลงมือ

- **ห้ามเดา ห้ามซ่อนความคลุมเครือ** ก่อนเขียนโค้ดทุกครั้ง ต้องระบุ *ข้อกำหนด (Requirements)* และ *ข้อตกลงเบื้องต้น (Assumptions)* ที่ใช้ออกมาให้เห็นชัดเจน
- หากพบ **จุดที่ตีความได้หลายแบบ** หรือมี **Tradeoffs หลายทาง** (เช่น Performance vs Readability, Atomic vs Streaming, Strict Schema vs Flexible Dict) ต้อง **หยุดและถามผู้ใช้ทันที** — ห้ามแอบเลือกแนวทางใดแนวทางหนึ่งเองอย่างเงียบ ๆ
- หากผู้ใช้ส่งคำสั่งกว้างหรือกำกวม ให้สรุปความเข้าใจของตนเองกลับไปยืนยันก่อนเริ่ม ไม่ใช่กระโจนเข้าโค้ดทันที

### 1.2 Simplicity First — เรียบง่ายเข้าไว้

- เขียนโค้ดให้ **น้อยที่สุดเท่าที่จำเป็น** เพื่อแก้ปัญหาที่ได้รับมอบหมายอย่าง *ตรงจุด*
- **ห้ามเพิ่ม** ฟีเจอร์ที่ไม่ได้ขอ, abstraction layers, configuration knobs, dependency ใหม่, หรือความยืดหยุ่นเผื่ออนาคต (Speculative Generality) ที่ยังไม่มีใครต้องการ
- เลือก data structure และ control flow ที่ *อ่านครั้งเดียวเข้าใจ* — ความฉลาดของโค้ดวัดที่ความชัดเจน ไม่ใช่ความซับซ้อน
- "Three similar lines is better than a premature abstraction." — ทำซ้ำ 3 บรรทัดดีกว่าสร้าง helper ที่ผิดทิศ

### 1.3 Surgical Changes — แก้ไขแบบศัลยกรรมเฉพาะจุด

- แตะเฉพาะส่วนที่ **จำเป็นต่อภารกิจ** เท่านั้น
- **ต้องเคารพสไตล์เดิมของโปรเจกต์อย่างเคร่งครัด** (naming convention, import order, indentation, docstring style, error-handling pattern) แม้จะรู้สึกว่ามีแบบที่ "ดีกว่า"
- **ห้าม "ปรับปรุง" หรือ refactor โค้ดรอบข้างที่ยังทำงานได้ดี** — แม้จะคิดว่ามันเขียนได้สวยกว่านี้ก็ตาม
- ลบได้เฉพาะ `imports`, `variables`, หรือ `functions` ที่กลายเป็น **Dead Code อันเป็นผลโดยตรง** จากการแก้ไขของตนเองเท่านั้น ห้ามไล่ลบของเก่าที่ไม่เกี่ยวข้องกับงาน

### 1.4 Goal-Driven Execution — ขับเคลื่อนด้วยเป้าหมายและการตรวจสอบ

- ทุกงานต้องถูกแบ่งออกเป็น **ขั้นตอนที่ชัดเจน** พร้อม **เกณฑ์ตรวจสอบความสำเร็จ** ของแต่ละขั้น
- รูปแบบที่แนะนำ:
  > **ขั้นตอน (Step):** ทำอะไร → **วิธีตรวจสอบ (Verify):** จะรู้ได้อย่างไรว่าผ่าน
- ก่อนปิดงานต้องตอบให้ได้ว่า: "ระบบยังผ่านเกณฑ์ความถูกต้องเดิมอยู่หรือไม่ และเงื่อนไขใหม่ที่ขอผ่านครบไหม"
- หากตรวจสอบไม่ได้ด้วยตนเอง (เช่น UI, manual flow) ต้อง **ระบุชัดเจน** ว่าทดสอบไม่ได้ ไม่ใช่เคลมว่าผ่าน

---

## 2. ขอบเขตและกฎเกณฑ์ของระบบ Multi-Agent (LangGraph Architecture)

ระบบนี้ใช้สถาปัตยกรรม **Supervisor + Specialist Workers** บน LangGraph `StateGraph`
**กฎเหล็ก:** ห้ามให้ Agent ตัวใดทำงานนอกขอบเขตหน้าที่ของตน (Domain Boundaries) แม้จะ "ทำได้" ในเชิงเทคนิคก็ตาม เพื่อป้องกัน *ข้อมูลตีกัน* และ *Single Source of Truth ที่พัง*

### 2.1 Manager / Supervisor Agent — ผู้ประสานงาน

- หน้าที่: **Routing และ State Transitions** เท่านั้น
- ต้องตัดสินใจจาก **ผลลัพธ์ของ Tools** (เช่น structured output, frontmatter validation) — *ไม่ใช่* จากการ "เดา" เนื้อหาในข้อความ
- ห้ามแก้ไขข้อมูลในไฟล์ ห้ามคำนวณตัวเลข ห้ามปลอมตัวเป็น Worker
- หากผลจาก Worker ไม่ผ่าน schema/validator ที่กำหนด → ต้อง fall back ไปยังเส้นทางที่ปลอดภัย (return to user) ห้ามฝืนส่งต่อ

### 2.2 Bookkeeper Agent — ผู้จัดการข้อมูลมีโครงสร้าง (Structured State Owner)

- เป็น **เอนทิตีเดียวในระบบ** ที่มีสิทธิ์อ่าน/แก้ไขสถานะข้อมูลแบบมีโครงสร้าง (YAML Frontmatter, Portfolio Ledger, Holdings Table)
- **กฎเหล็กข้อ 1:** ต้องเรียก Tool เพื่อ **อ่านข้อมูลล่าสุดจากดิสก์ก่อนการวิเคราะห์ทุกครั้ง** — ห้ามใช้ค่าจาก context เดิมที่อาจ stale
- **กฎเหล็กข้อ 2:** **ห้ามคำนวณตัวเลขทางการเงินใน-context (In-Context Math) เด็ดขาด** ไม่ว่าจะเป็นการบวก ลบ คูณ หาร เปอร์เซ็นต์ผลตอบแทน หรือสัดส่วน Asset Allocation — *ทุกตัวเลข* ต้องผ่าน Python Tool ที่ deterministic เท่านั้น
- LLM มีหน้าที่ *ตัดสินใจว่าจะเรียก Tool อะไร ด้วย argument อะไร* ไม่ใช่หน้าที่เป็นเครื่องคิดเลข

### 2.3 Archivist Agent — ผู้จัดการข้อมูลไร้โครงสร้าง (Unstructured Content)

- รับผิดชอบ **เนื้อหาภาษา**: บันทึกประจำวัน, สรุปข่าว, สรุปบทวิเคราะห์, YouTube transcript summary, Markdown body ทั่วไป
- ใช้งาน vectorstore / semantic search ได้เต็มที่ในขอบเขตของตน
- **ห้ามแตะ** ค่าตัวเลขใน YAML Frontmatter ที่เป็นของระบบบัญชี (เช่น `quantity`, `cost_basis`, `current_value`, `allocation_pct`) — ส่วนนี้เป็นพื้นที่ของ Bookkeeper เท่านั้น
- หากผู้ใช้สั่งงานที่ต้องแก้ทั้งสองโดเมน → ต้องร้องขอผ่าน Manager ให้ orchestrate ไปยัง Bookkeeper ก่อน แล้วค่อย Archivist ตามหลัง

### 2.4 PII Gateway — ด่านป้องกันข้อมูลส่วนบุคคล

- ระบบกรอง PII (PII Anonymization Middleware) มีลอจิกการทำงานหลักสถิตอยู่ที่ไฟล์ `core/security.py` และถูกบังคับใช้ที่จุดรับข้อมูลหลัก (`main.py`) เป็น **โครงสร้างที่ห้ามแก้ไข ห้ามข้าม ห้าม bypass** เด็ดขาด
- ข้อมูลดิบที่มี PII (เลขบัตรประชาชน, เบอร์โทร, อีเมล, เลขบัตรเครดิต ฯลฯ) ต้องถูก **ปกปิด/แทนที่ก่อน** ส่งออกไปยังโมเดล LLM ภายนอกเสมอ
- หากต้องเพิ่มชนิด PII ใหม่ ให้เพิ่มที่ middleware layer เดิม ไม่ใช่สร้างทางลัดใหม่
- การแก้ไข PII Gateway ต้องถูก *ขออนุญาตผู้ใช้อย่างชัดเจนเป็นลายลักษณ์อักษร* ก่อนเสมอ

---

## 3. ความถูกต้องของข้อมูลและหลักความปลอดภัยทางการเงิน (Data Integrity Invariants)

ส่วนนี้คือ **Architectural Invariants** ของระบบ — กฎที่ต้องเป็นจริง *ตลอดเวลา* ไม่ว่าจะเกิดอะไรขึ้น

### 3.1 Atomic Storage Mutations — การบันทึกไฟล์ที่ปลอดภัย

> **เหตุผล:** หากระบบแครชระหว่างเขียนไฟล์ Markdown/YAML ของ Portfolio ไฟล์ครึ่ง ๆ กลาง ๆ จะทำให้ข้อมูลบัญชีพังทั้งระบบและกู้คืนยาก

**กฎเหล็ก:** ทุกฟังก์ชันที่เขียน/แก้ไขไฟล์โลคอล (โดยเฉพาะใน Obsidian Vault) ต้องทำตามขั้นตอน:

1. เขียนเนื้อหาทั้งหมดลง **ไฟล์ชั่วคราว (shadow/temp file)** ในไดเรกทอรีเดียวกันก่อน
2. เมื่อเขียนเสร็จสมบูรณ์ ใช้ **OS-level atomic rename** (เช่น `os.replace()` ใน Python) สลับไฟล์เดิมออก
3. ห้ามใช้ pattern เปิดไฟล์เดิมแล้วเขียนทับโดยตรง (`open(path, "w")` แล้วเขียนทันที) สำหรับไฟล์ที่เป็น source of truth ทางการเงิน

หลักนี้ครอบคลุมถึง: YAML frontmatter updates, ledger entries, index rewrites, snapshot files

### 3.2 Anti-Drift Aggregations — ลูปคำนวณจากล่างขึ้นบน

> **เหตุผล:** ตัวเลขสรุประดับ Summary/Dashboard ห้าม drift ออกจากผลรวมของข้อมูลรายตัว — มิฉะนั้นผู้ใช้จะตัดสินใจลงทุนผิดพลาด

**กฎเหล็ก:** ทุกครั้งที่มีการเปลี่ยนแปลงค่ารายสินทรัพย์ (Per-Asset Mutation) ระบบต้องรัน **Bottom-Up Recalculation Loop** ให้ครบก่อน commit ลงไฟล์เสมอ:

1. **Per-Asset Layer:** คำนวณมูลค่ารายตัวใหม่ (current value, unrealized P/L, %change)
2. **Summary Layer:** Re-sum ยอดรวมพอร์ตจากผลของขั้นที่ 1 (ห้ามอัปเดต summary แบบ delta-patch อย่างเดียว)
3. **Allocation Layer:** คำนวณสัดส่วน Asset Allocation % ใหม่ทั้งหมดจากตัวเลขที่ re-sum แล้ว
4. **Persist:** จึงค่อยเขียนผลทั้งหมดลงไฟล์ด้วยกฎ Atomic ในข้อ 3.1

**ห้าม** ใช้ shortcut อัปเดตเฉพาะตัวเลขที่เปลี่ยน โดยไว้ใจว่ายอดรวมเดิม "น่าจะยังถูก"

### 3.3 Single Source of Truth — แหล่งความจริงเดียว

- ข้อมูลพอร์ตและธุรกรรมต้องมี **แหล่งความจริงเดียวบนดิสก์** (Markdown + YAML ใน Vault)
- ห้ามสร้าง cache, copy, หรือ derived state ที่อาจ "หลุดซิงค์" จากต้นทาง โดยไม่มีกลไก invalidation ที่ชัดเจน
- หากต้อง denormalize เพื่อ performance ต้องระบุชัดว่าใครเป็น *master* ใครเป็น *derived* และต้องมีฟังก์ชัน rebuild derived จาก master ได้เสมอ

### 3.4 API Resilience — การรับมือเครือข่ายล้มเหลว

> **เหตุผล:** ระบบ Multi-Agent นี้พึ่งพา LLM API ภายนอก (Google Gemini, OpenRouter) และ Data API (yfinance, FRED) เป็นหลัก — เครือข่ายและโควต้ามีโอกาสสะดุดเป็นเรื่องปกติ หากไม่จัดการอย่างเป็นระบบ ระบบจะแครชกลางคันและผู้ใช้สูญเสีย context การสนทนา

**กฎเหล็ก:** ทุกการเชื่อมต่อกับ LLM หรือ External API ทั้งในปัจจุบันและที่เพิ่มเข้ามาในอนาคต **ต้องครอบด้วยระบบดักจับ Transient Error และกลไก Exponential Backoff Retry เสมอ** — ห้ามปล่อยให้ระบบหยุดทำงานทันทีเมื่อเครือข่ายสะดุด

แนวทางที่ต้องปฏิบัติ:

1. **Transient Error Detection** — ต้องดักจับ error ที่ retry แล้วน่าจะหายเอง โดยเฉพาะ:
   - HTTP `429` (Rate Limit / Resource Exhausted)
   - HTTP `500` / `502` / `503` / `504` (Server Error / Bad Gateway / Service Unavailable / Gateway Timeout)
   - Network-level exceptions (`TimeoutError`, `ConnectionError`, `httpx.TimeoutException` ฯลฯ)
2. **Exponential Backoff** — ระยะรอต้องเพิ่มขึ้นเรื่อย ๆ ต่อรอบ retry (เช่น `2 ** attempt` วินาที) ห้าม retry ติด ๆ เพราะจะยิ่งซ้ำเติมเซิร์ฟเวอร์ปลายทาง
3. **Retry Budget ที่ชัดเจน** — กำหนดจำนวนครั้งสูงสุดไว้แน่นอน (เช่น 3 ครั้ง) ห้าม retry แบบไม่จำกัด
4. **Graceful Fallback** — เมื่อ retry หมดโควต้าแล้วยังไม่สำเร็จ ต้องแจ้งผู้ใช้ด้วยข้อความที่อ่านเข้าใจง่าย ห้ามโยน stack trace ดิบขึ้นจอ และต้อง **คงสถานะการทำงานของลูปหลักไว้** ให้ผู้ใช้พิมพ์คำสั่งใหม่ได้

**Reference Implementation:** ดูรูปแบบสถาปัตยกรรมที่เสถียรได้จากลูปการรันของไฟล์ `main.py` —
- ฟังก์ชัน `_is_transient_error()` ครอบคลุมการตรวจ HTTP code + network exception + LangChain-wrapped error messages
- ลูป `for attempt in range(max_retries)` ใช้ `time.sleep(2 ** attempt)` เป็น backoff strategy
- เมื่อ retry หมด → แสดง message สั้น ๆ ที่ผู้ใช้เข้าใจ แล้ว `break` กลับสู่ prompt loop โดยไม่ทำลายสถานะ

Agent หรือ Tool ใหม่ใด ๆ ที่เรียก API ภายนอกต้องอ้างอิงรูปแบบเดียวกันนี้ — ห้ามสร้าง path ที่ปล่อยให้ exception ดิบหลุดขึ้นไปทำให้ระบบ crash

---

## 4. มาตรฐานการแสดงผลและการบันทึก Log (Communication Standards)

### 4.1 หลักการ — Scannable, Parseable, Terse

ข้อความระหว่าง Agent และ Log บน Terminal ต้อง:

- **กวาดสายตาเข้าใจได้ทันที** (Scannable) — ไม่ต้องอ่านยาว
- **โค้ดดึงไปประมวลผลต่อได้สะดวก** (Parseable) — มี structure ที่ regex/split ได้
- **กระชับ** (Terse) — ไม่พ่นข้อความสนทนายาว ๆ และไม่พ่น JSON ดิบทั้งก้อนลง stdout

### 4.2 รูปแบบบังคับ — Prefix Token Structure

ใช้รูปแบบ **structured prefix** คั่นด้วย ` | ` (pipe + space) เป็นมาตรฐานหลัก:

```
[การกระทำ] | [ตัวเลข/Delta สำคัญที่เปลี่ยน] | [สถานะคลังเงินสด/บริบทปัจจุบัน]
```

**ตัวอย่างที่ถูกต้อง:**

```
[BUY AOT] | qty +100 @ 62.50 | cash 125,430 → 119,180
[REBALANCE] | TECH 28% → 25% | cash 119,180
[ROUTE → Bookkeeper] | reason: structured_mutation | turn 4
[SAVE OK] | file: Holdings.md | atomic_swap done
```

**ตัวอย่างที่ผิด** (ห้ามใช้):

```
ครับ ผมได้ทำการบันทึกการซื้อหุ้น AOT จำนวน 100 หุ้น ที่ราคา 62.50 บาท
เรียบร้อยแล้วนะครับ ตอนนี้ยอดเงินสดในพอร์ตของคุณคือ...
```

หรือ:

```
{"action": "buy", "symbol": "AOT", "qty": 100, "price": 62.5, "cash_before": 125430, ...}
```

### 4.3 Verbosity Levels

- **DEBUG:** dump payload เต็มได้ (สำหรับ trace ปัญหา)
- **INFO (default):** prefix-token format เท่านั้น
- **WARNING/ERROR:** prefix-token + reason สั้น ๆ + actionable hint

ทุกคำสั่ง LLM-to-LLM ที่เป็น routing (เช่น `[Manager → Bookkeeper]`, `[Manager → Archivist]`) **ต้องใช้รูปแบบ prefix นี้บังคับ** เพื่อให้ระบบ trace flow ได้

---

## 5. หลักปฏิบัติสุดท้าย (Final Doctrine)

1. **เมื่อสงสัย — ถาม** ไม่ใช่เดา
2. **เมื่อเป็นเงิน — ใช้ Tool** ไม่ใช่คำนวณในใจ
3. **เมื่อจะลบ — ตรวจสามรอบ** ว่าเป็น dead code จากงานตนเองจริง
4. **เมื่อจะเขียนไฟล์ — Atomic** เสมอ
5. **เมื่อตัวเลขเปลี่ยน — Re-aggregate** จากล่างขึ้นบน
6. **เมื่อพูดกับ Terminal — Prefix Token** ไม่ใช่ prose

> *"Code is read far more often than it is written. Optimize for the reader — and the reader of a financial system is auditing for correctness, not admiring cleverness."*
