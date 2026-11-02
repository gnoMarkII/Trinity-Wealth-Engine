import { render, screen, fireEvent } from '@testing-library/react'
import { describe, it, expect, vi, beforeEach } from 'vitest'
import ManagerFavoritePromptsPopover from './ManagerFavoritePromptsPopover'
import { DEFAULT_FAVORITE_PROMPTS } from '../../lib/favoritePromptsStorage'

describe('ManagerFavoritePromptsPopover Component', () => {
  const onClose = vi.fn()
  const onActiveTickerChange = vi.fn()
  const onTogglePin = vi.fn()
  const onCreateCustom = vi.fn()
  const onUpdateCustom = vi.fn()
  const onDeleteCustom = vi.fn()
  const onSelectToEdit = vi.fn()
  const onSelectAndSend = vi.fn()

  beforeEach(() => {
    vi.clearAllMocks()
  })

  function renderPopover(isOpen = true, ticker = 'NVDA') {
    return render(
      <ManagerFavoritePromptsPopover
        isOpen={isOpen}
        onClose={onClose}
        prompts={DEFAULT_FAVORITE_PROMPTS}
        activeTicker={ticker}
        onActiveTickerChange={onActiveTickerChange}
        onTogglePin={onTogglePin}
        onCreateCustom={onCreateCustom}
        onUpdateCustom={onUpdateCustom}
        onDeleteCustom={onDeleteCustom}
        onSelectToEdit={onSelectToEdit}
        onSelectAndSend={onSelectAndSend}
      />
    )
  }

  it('renders nothing when isOpen is false', () => {
    const { container } = renderPopover(false)
    expect(container.firstChild).toBeNull()
  })

  it('renders pinned tab by default with favorite prompts', () => {
    renderPopover(true)
    expect(screen.getByText('คลังคำสั่งอัจฉริยะ (Favorite Prompts Library)')).toBeInTheDocument()

    // Default pinned items include Dalio Cycle and 5-Tier
    expect(screen.getByText(/วิเคราะห์ Macro Regime & Dalio Cycle/i)).toBeInTheDocument()
    expect(screen.getByText(/เจาะลึกหุ้น NVDA ครบ 5 มิติ/i)).toBeInTheDocument()
  })

  it('switches between category tabs', async () => {
    renderPopover(true)

    // Switch to Macro tab
    const macroTabBtn = screen.getByRole('button', { name: /เศรษฐกิจมหภาค/i })
    fireEvent.click(macroTabBtn)

    expect(screen.getByText(/เจาะลึก US Treasury Yield Curve/i)).toBeInTheDocument()

    // Switch to Workflows tab
    const workflowTabBtn = screen.getByRole('button', { name: /สั่งรันงาน & ความรู้/i })
    fireEvent.click(workflowTabBtn)

    expect(screen.getByText(/คัดกรองเหตุการณ์สำคัญเข้าสู่ News Funnel/i)).toBeInTheDocument()
  })

  it('filters prompts via search bar', async () => {
    renderPopover(true)

    const searchInput = screen.getByPlaceholderText(/ค้นหาคำสั่ง เช่น DCF/i)
    fireEvent.change(searchInput, { target: { value: 'Yield Curve' } })

    // Yield curve should match
    expect(screen.getByText(/เจาะลึก US Treasury Yield Curve/i)).toBeInTheDocument()
    // Unrelated should be filtered out
    expect(screen.queryByText(/5-Tier Institutional Thesis/i)).not.toBeInTheDocument()
  })

  it('updates ticker when quick ticker button or input is changed', async () => {
    renderPopover(true, 'AAPL')

    // Switch to equity tab
    fireEvent.click(screen.getByRole('button', { name: /หุ้นรายตัว/i }))

    // Title should be resolved with AAPL
    expect(screen.getByText(/เจาะลึกหุ้น AAPL ครบ 5 มิติ/i)).toBeInTheDocument()

    // Click TSM quick button
    const tsmBtn = screen.getByRole('button', { name: 'TSM' })
    fireEvent.click(tsmBtn)
    expect(onActiveTickerChange).toHaveBeenCalledWith('TSM')
  })

  it('calls onSelectToEdit and closes modal when clicking "แก้ไขก่อน"', () => {
    renderPopover(true, 'NVDA')

    const editBtns = screen.getAllByRole('button', { name: /แก้ไขก่อน/i })
    fireEvent.click(editBtns[0]!)

    expect(onSelectToEdit).toHaveBeenCalledTimes(1)
    expect(onClose).toHaveBeenCalledTimes(1)
  })

  it('calls onSelectAndSend and closes modal when clicking "ส่งทันที"', () => {
    renderPopover(true, 'NVDA')

    const sendBtns = screen.getAllByRole('button', { name: /ส่งทันที/i })
    fireEvent.click(sendBtns[0]!)

    expect(onSelectAndSend).toHaveBeenCalledTimes(1)
    expect(onClose).toHaveBeenCalledTimes(1)
  })

  it('triggers pin toggle when clicking star button', () => {
    renderPopover(true)

    const starBtns = screen.getAllByTitle(/ปักหมุด|ถอนหมุด/i)
    fireEvent.click(starBtns[0]!)

    expect(onTogglePin).toHaveBeenCalled()
  })

  it('opens custom form and submits new custom prompt', async () => {
    renderPopover(true)

    const createBtn = screen.getByRole('button', { name: /สร้างคำสั่งใหม่/i })
    fireEvent.click(createBtn)

    expect(screen.getByText(/สร้างคำสั่งส่วนตัวใหม่/i)).toBeInTheDocument()

    const titleInput = screen.getByPlaceholderText(/ชื่อคำสั่ง/i)
    const promptInput = screen.getByPlaceholderText(/ข้อความคำสั่งที่ต้องการให้ AI ตอบ/i)

    fireEvent.change(titleInput, { target: { value: 'My Strategy Prompt' } })
    fireEvent.change(promptInput, { target: { value: 'ตรวจสอบ [TICKER] ด้วยสูตรลับ' } })

    const submitBtn = screen.getByRole('button', { name: /บันทึกคำสั่ง/i })
    fireEvent.click(submitBtn)

    expect(onCreateCustom).toHaveBeenCalledWith(
      expect.objectContaining({
        title: 'My Strategy Prompt',
        prompt: 'ตรวจสอบ [TICKER] ด้วยสูตรลับ',
      })
    )
  })
})

