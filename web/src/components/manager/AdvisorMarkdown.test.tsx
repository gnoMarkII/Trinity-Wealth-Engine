import { render, screen, fireEvent } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import AdvisorMarkdown, {
  allocateDeterministicIntegerWeights,
  extractTableMetrics,
  splitMarkdownAndTables,
} from './AdvisorMarkdown'

describe('AdvisorMarkdown Component', () => {
  it('correctly splits markdown text and markdown tables', () => {
    const content = `นี่คือภาพรวมพอร์ตของคุณ:

| สัญลักษณ์ | มูลค่า (THB) | กำไร/ขาดทุน (%) |
| :--- | :--- | :--- |
| **FTNT** | 13,065.73 | +71.86% |
| **NFLX** | 31,867.14 | -20.70% |

ข้อสังเกตจาก AI:
* พอร์ตโดยรวมแข็งแกร่ง`

    const segments = splitMarkdownAndTables(content)
    expect(segments.length).toBe(3)
    expect(segments[0]?.type).toBe('markdown')
    expect(segments[1]?.type).toBe('table')
    expect(segments[2]?.type).toBe('markdown')

    if (segments[1]?.type === 'table') {
      expect(segments[1].data.headers).toEqual(['สัญลักษณ์', 'มูลค่า (THB)', 'กำไร/ขาดทุน (%)'])
      expect(segments[1].data.rows.length).toBe(2)
      expect(segments[1].data.rows[0]).toEqual(['**FTNT**', '13,065.73', '+71.86%'])
    }
  })

  it('renders table elements with green and red P&L badges and ticker chip', () => {
    const content = `| สัญลักษณ์ | มูลค่า | กำไร/ขาดทุน |
| :--- | :--- | :--- |
| **FTNT** | 13,065.73 | +71.86% |
| **NFLX** | 31,867.14 | -20.70% |`

    render(<AdvisorMarkdown content={content} />)

    // Check table headers
    expect(screen.getByText('สัญลักษณ์')).toBeInTheDocument()
    expect(screen.getByText('มูลค่า')).toBeInTheDocument()
    expect(screen.getByText('กำไร/ขาดทุน')).toBeInTheDocument()

    // Check styled ticker chip (rendered as button with label $FTNT)
    expect(screen.getByText('$FTNT')).toBeInTheDocument()
    expect(screen.getByText('$NFLX')).toBeInTheDocument()

    // Check P&L badges
    const positiveBadge = screen.getByText('+71.86%')
    expect(positiveBadge).toBeInTheDocument()
    expect(positiveBadge.className).toContain('text-emerald-700')

    const negativeBadge = screen.getByText('-20.70%')
    expect(negativeBadge).toBeInTheDocument()
    expect(negativeBadge.className).toContain('text-rose-700')
  })

  it('renders streaming caret when isStreaming is true', () => {
    const { container } = render(<AdvisorMarkdown content="กำลังวิเคราะห์..." isStreaming />)
    expect(screen.getByText('กำลังวิเคราะห์...')).toBeInTheDocument()
    expect(container.querySelector('.animate-stream-caret')).not.toBeNull()
  })

  it('renders institutional evidence badges for DOCUMENTED, TESTED, INFERRED, UNVERIFIED', () => {
    const content = 'รายได้เติบโต 45% [DOCUMENTED: SEC 10-Q p.12] และ DCF Fair Value คือ $185 [TESTED: Python DCF] คาดว่าจะเร่งตัว [INFERRED] แต่มีข่าวลือชิปรุ่นใหม่ [UNVERIFIED: Twitter leak]'

    render(<AdvisorMarkdown content={content} />)

    expect(screen.getByText('DOC: SEC 10-Q p.12')).toBeInTheDocument()
    expect(screen.getByText('MATH: Python DCF')).toBeInTheDocument()
    expect(screen.getByText('INFERRED')).toBeInTheDocument()
    expect(screen.getByText('UNVERIFIED: Twitter leak')).toBeInTheDocument()
  })

  it('renders interactive view switcher on numeric tables and toggles to chart view', () => {
    const content = `| สัญลักษณ์ | กำไร/ขาดทุน |
| :--- | :--- |
| **NVDA** | +45.5% |
| **TSLA** | -12.3% |`

    render(<AdvisorMarkdown content={content} />)

    // Switcher controls must be present
    expect(screen.getByText('📋 ตาราง')).toBeInTheDocument()
    const chartBtn = screen.getByText('📊 กราฟเปรียบเทียบ')
    expect(chartBtn).toBeInTheDocument()

    // Toggle to chart view
    fireEvent.click(chartBtn)
    expect(screen.getByText(/เปรียบเทียบสัดส่วนและผลลัพธ์ของแต่ละรายการ/)).toBeInTheDocument()
    expect(screen.getByText('+45.5%')).toBeInTheDocument()
    expect(screen.getByText('-12.3%')).toBeInTheDocument()
  })

  it('toggles to interactive simulator mode and dispatches custom weights via onActionClick', () => {
    const onAction = vi.fn()
    const content = `| สัญลักษณ์ | สัดส่วน |
| :--- | :--- |
| **NVDA** | 60% |
| **MSFT** | 40% |`

    render(<AdvisorMarkdown content={content} onActionClick={onAction} />)

    const simBtn = screen.getByText('🎛️ จำลองสัดส่วน')
    fireEvent.click(simBtn)

    expect(screen.getByText(/จำลองปรับสัดส่วนเป้าหมาย/)).toBeInTheDocument()
    expect(screen.getByText('🚀 ส่งแผนสัดส่วนให้ Hermes คำนวณ')).toBeInTheDocument()

    // Click submit button
    fireEvent.click(screen.getByText('🚀 ส่งแผนสัดส่วนให้ Hermes คำนวณ'))
    expect(onAction).toHaveBeenCalledTimes(1)
    expect(onAction.mock.calls[0]?.[0]).toContain('ช่วยคำนวณแผนการปรับสัดส่วนพอร์ต')
    expect(onAction.mock.calls[0]?.[1]).toEqual(['@Portfolio'])
  })

  it('opens inline ticker inspector drawer when clicking a $TICKER badge and triggers deep dive action', () => {
    const onAction = vi.fn()
    const content = `| สัญลักษณ์ | กำไร/ขาดทุน |
| :--- | :--- |
| **FTNT** | +71.86% |`

    render(<AdvisorMarkdown content={content} onActionClick={onAction} />)

    const tickerPill = screen.getByText('$FTNT')
    fireEvent.click(tickerPill)

    // Drawer should open
    expect(screen.getByText('Interactive Chart Drawer')).toBeInTheDocument()
    const dcfBtn = screen.getByText('🔬 วิเคราะห์ Valuation & DCF ($FTNT)')
    expect(dcfBtn).toBeInTheDocument()

    fireEvent.click(dcfBtn)
    expect(onAction).toHaveBeenCalledTimes(1)
    expect(onAction.mock.calls[0]?.[0]).toContain('Valuation, DCF และความเสี่ยงของหุ้น FTNT')
    expect(onAction.mock.calls[0]?.[1]).toEqual(['@Portfolio', '$FTNT'])
  })

  it('renders streaming skeleton for unclosed widget block and RadialGauge when closed', () => {
    // 1. Unclosed streaming widget block
    const streamingContent = `กำลังประเมินความเสี่ยง:\n\`\`\`widget\n{"widgetType": "gauge", "payload": {"value": 1.2}}`
    const { rerender } = render(<AdvisorMarkdown content={streamingContent} isStreaming />)
    expect(screen.getByText('กำลังสตรีม Interactive Widget...')).toBeInTheDocument()

    // 2. Closed widget block
    const closedContent = `กำลังประเมินความเสี่ยง:\n\`\`\`widget\n{"widgetType": "gauge", "title": "ความเสี่ยงเชิงมหภาค", "payload": {"value": 1.5, "min": -3, "max": 3}}\n\`\`\``
    rerender(<AdvisorMarkdown content={closedContent} />)
    expect(screen.getByText('ความเสี่ยงเชิงมหภาค')).toBeInTheDocument()
  })

  it('R06/W06: reconciles simulator weights incrementally during streaming, deduplicates tickers, preserves user edits, and guards submission', () => {
    const onAction = vi.fn()
    const streamStep1 = `| สัญลักษณ์ | มูลค่า (THB) |
| :--- | :--- |
| **PTT.BK** | 1000 |`

    const { rerender } = render(
      <AdvisorMarkdown content={streamStep1} isStreaming onActionClick={onAction} />
    )

    // Switch to simulator while streaming 1 row -> PTT.BK = 100%, submit MUST be disabled during streaming
    fireEvent.click(screen.getByText('🎛️ จำลองสัดส่วน'))
    expect(screen.getByLabelText('สัดส่วนเป้าหมาย PTT.BK')).toHaveValue('100')
    const submitBtnWhileStreaming = screen.getByText('🚀 ส่งแผนสัดส่วนให้ Hermes คำนวณ')
    expect(submitBtnWhileStreaming).toBeDisabled()
    fireEvent.click(submitBtnWhileStreaming)
    expect(onAction).not.toHaveBeenCalled()

    // Stream appends duplicate PTT.BK lot and SCB.BK row, then stream finishes
    const streamStep2 = `| สัญลักษณ์ | มูลค่า (THB) |
| :--- | :--- |
| **PTT.BK** | 1000 |
| **PTT.BK** | 500 |
| **SCB.BK** | 2000 |`

    rerender(<AdvisorMarkdown content={streamStep2} isStreaming={false} onActionClick={onAction} />)

    const pttSlider = screen.getByLabelText('สัดส่วนเป้าหมาย PTT.BK') as HTMLInputElement
    const scbSlider = screen.getByLabelText('สัดส่วนเป้าหมาย SCB.BK') as HTMLInputElement
    expect(pttSlider.value).toBe('50')
    expect(scbSlider.value).toBe('50')

    // User adjusts PTT.BK to 70% while total != 100 -> submit disabled
    fireEvent.change(pttSlider, { target: { value: '70' } })
    expect(screen.getByText('รวม: 120%')).toBeInTheDocument()
    expect(screen.getByText('🚀 ส่งแผนสัดส่วนให้ Hermes คำนวณ')).toBeDisabled()

    // Third ticker ADVANC.BK arrives -> retained user edit 70% on PTT.BK, unedited share 30% split to 15/15
    const streamStep3 = `| สัญลักษณ์ | มูลค่า (THB) |
| :--- | :--- |
| **PTT.BK** | 1000 |
| **SCB.BK** | 2000 |
| **ADVANC.BK** | 3000 |`
    rerender(<AdvisorMarkdown content={streamStep3} isStreaming={false} onActionClick={onAction} />)
    expect((screen.getByLabelText('สัดส่วนเป้าหมาย PTT.BK') as HTMLInputElement).value).toBe('70')
    expect((screen.getByLabelText('สัดส่วนเป้าหมาย SCB.BK') as HTMLInputElement).value).toBe('15')
    expect((screen.getByLabelText('สัดส่วนเป้าหมาย ADVANC.BK') as HTMLInputElement).value).toBe('15')
    expect(screen.getByText('รวม: 100%')).toBeInTheDocument()

    const submitBtn = screen.getByText('🚀 ส่งแผนสัดส่วนให้ Hermes คำนวณ')
    expect(submitBtn).toBeEnabled()
    fireEvent.click(submitBtn)
    expect(onAction).toHaveBeenCalledTimes(1)
    expect(onAction.mock.calls[0]?.[0]).toContain('PTT.BK: 70%, SCB.BK: 15%, ADVANC.BK: 15%')

    // Also verify read-only mode (no onActionClick) disables submit and equal 3-ticker fallback sums to 100
    const threeWeights = allocateDeterministicIntegerWeights(['A', 'B', 'C'], {}, false)
    expect(threeWeights.A! + threeWeights.B! + threeWeights.C!).toBe(100)

    rerender(<AdvisorMarkdown content={streamStep3} />)
    expect(screen.getByText('🚀 ส่งแผนสัดส่วนให้ Hermes คำนวณ')).toBeDisabled()
  })

  it('R12/W06: selects single shared metric column, parses plain numbers, ignores bold actions as tickers, and handles zero/missing/mixed currency', () => {
    // 1. Ticker | Action | Value THB | Return (plain numbers + missing Return + bold BUY action)
    const content = `| Ticker | Action | Value THB | Return |
| :--- | :--- | :--- | :--- |
| **PTT** | **BUY** | 100 | 10% |
| **SCB** | **HOLD** | 200 | - |
| **KBANK** | **TRIM** | -50 | 0% |`

    const metrics = extractTableMetrics(splitMarkdownAndTables(content)[0]?.type === 'table'
      ? (splitMarkdownAndTables(content)[0] as { type: 'table'; data: any }).data
      : { headers: [], alignments: [], rows: [] })

    expect(metrics.map((m) => m.ticker)).toEqual(['PTT', 'SCB', 'KBANK'])
    // Return column selected cleanly: SCB is missing (undefined, NOT 200 THB from Value THB!), KBANK is true 0
    expect(metrics.map((m) => m.numericValue)).toEqual([10, undefined, 0])

    render(<AdvisorMarkdown content={content} />)
    // Action words MUST NOT render as clickable $BUY/$HOLD/$TRIM ticker pills
    expect(screen.queryByText('$BUY')).not.toBeInTheDocument()
    expect(screen.queryByText('$HOLD')).not.toBeInTheDocument()
    expect(screen.getByText('$PTT')).toBeInTheDocument()

    // Switch to Chart view -> Return column shows missing notice for SCB and 0-width bar for KBANK
    fireEvent.click(screen.getByText('📊 กราฟเปรียบเทียบ'))
    expect(screen.getAllByText('ไม่มีข้อมูล').length).toBeGreaterThanOrEqual(1)
    expect(screen.getByText('0%')).toBeInTheDocument()

    // Switch chart metric selector to plain-number Value THB column
    const colSelect = screen.getByLabelText('เลือกคอลัมน์ตัวชี้วัด')
    fireEvent.change(colSelect, { target: { value: '2' } })
    expect(screen.getByText('100 THB')).toBeInTheDocument()
    expect(screen.getByText('200 THB')).toBeInTheDocument()
    expect(screen.getByText('-50 THB')).toBeInTheDocument()

    // 2. Mixed currency column rejects single-scale bar rendering
    const mixedCurrency = `| Ticker | Value |
| :--- | :--- |
| **PTT** | 1000 THB |
| **NVDA** | 500 USD |`
    render(<AdvisorMarkdown content={mixedCurrency} />)
    const chartBtns = screen.getAllByText('📊 กราฟเปรียบเทียบ')
    fireEvent.click(chartBtns[chartBtns.length - 1]!)
    expect(screen.getByText(/มีหน่วย\/สกุลเงินผสมกัน/)).toBeInTheDocument()
  })

  it('R13/W06: validates widget schemas, rejects nested/non-widget code fences, handles aborted streams & unsupported types safely', () => {
    // 1. Nested widget inside 4-backtick code example and ```widget_helper must remain markdown, not active widgets
    const nestedContent = `ตัวอย่างโค้ด:\n\`\`\`\`markdown\n\`\`\`widget\n{"widgetType": "gauge", "payload": {"value": 1}}\n\`\`\`\n\`\`\`\`\n\n\`\`\`widget_helper\n{"helper": true}\n\`\`\``
    const nestedSegs = splitMarkdownAndTables(nestedContent)
    expect(nestedSegs.some((s) => s.type === 'widget')).toBe(false)

    // 2. Valid JSON with object title must NOT crash React ("Objects are not valid as a React child")
    const badTitleWidget = `\`\`\`widget\n{"widgetType": "gauge", "title": {"bad": "obj"}, "payload": {"value": 1.2}}\n\`\`\``
    render(<AdvisorMarkdown content={badTitleWidget} />)
    expect(screen.getByText(/ฟิลด์ title ของ Widget ต้องเป็นข้อความ/)).toBeInTheDocument()

    // 3. Reversed/zero min-max or non-finite gauge value shows readable validation alert
    const badRangeWidget = `\`\`\`widget\n{"widgetType": "gauge", "title": "Bad Range", "payload": {"value": 1, "min": 5, "max": 1}}\n\`\`\``
    render(<AdvisorMarkdown content={badRangeWidget} />)
    expect(screen.getByText(/ช่วงค่า min\/max ของมาตรวัดไม่ถูกต้อง/)).toBeInTheDocument()

    // 4. Closed malformed JSON or scalar root does NOT show infinite loading skeleton when isStreaming=false
    const malformedJsonWidget = `\`\`\`widget\n{not-valid-json}\n\`\`\`\n\n\`\`\`widget\n12345\n\`\`\``
    render(<AdvisorMarkdown content={malformedJsonWidget} isStreaming={false} />)
    expect(screen.queryByText('กำลังสตรีม Interactive Widget...')).not.toBeInTheDocument()
    expect(screen.getByText(/รูปแบบ JSON ของ Widget ไม่ถูกต้อง/)).toBeInTheDocument()
    expect(screen.getByText(/ข้อมูล Widget ต้องเป็น JSON Object/)).toBeInTheDocument()

    // 5. Unclosed widget block after stream terminates (isStreaming=false) does NOT spin indefinitely
    const abortedStreamWidget = `\`\`\`widget\n{"widgetType": "gauge", "payload": {"value": 1.2`
    render(<AdvisorMarkdown content={abortedStreamWidget} isStreaming={false} />)
    expect(screen.queryByText('กำลังสตรีม Interactive Widget...')).not.toBeInTheDocument()
    expect(screen.getByText(/ข้อมูลสตรีมไม่สมบูรณ์/)).toBeInTheDocument()

    // 6. Unknown widget type shows explicit unsupported fallback rather than raw JSON dump
    const unknownWidget = `\`\`\`widget\n{"widgetType": "custom_heatmap_3d", "title": "Heatmap", "payload": {"x": 1}}\n\`\`\``
    render(<AdvisorMarkdown content={unknownWidget} />)
    expect(screen.getByText(/ไม่รองรับ Interactive Widget ประเภท "custom_heatmap_3d"/)).toBeInTheDocument()

    // 7. Supported rebalance_simulator widget renders interactively and dispatches exact normalized targets
    const onAction = vi.fn()
    const rebalWidget = `\`\`\`widget\n{"widgetType": "rebalance_simulator", "title": "พอร์ตจำลอง", "payload": {"targets": [{"symbol": "PTT.BK", "targetWeightPct": 33.4}, {"symbol": "SCB.BK", "targetWeightPct": 33.3}, {"symbol": "NVDA", "targetWeightPct": 33.3}]}}\n\`\`\``
    render(<AdvisorMarkdown content={rebalWidget} onActionClick={onAction} />)
    expect(screen.getByText('พอร์ตจำลอง')).toBeInTheDocument()
    fireEvent.click(screen.getByText('🚀 ส่งแผนสัดส่วนให้ Hermes คำนวณ'))
    expect(onAction).toHaveBeenCalledTimes(1)
    expect(onAction.mock.calls[0]?.[0]).toContain('PTT.BK: 34%, SCB.BK: 33%, NVDA: 33%')
  })
})
