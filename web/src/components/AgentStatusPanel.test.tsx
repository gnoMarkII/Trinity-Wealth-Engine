import { act, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { api } from '../api/client'
import type { ActiveAgentStatusDTO } from '../api/types'
import AgentStatusPanel from './AgentStatusPanel'

function setDocumentHidden(hidden: boolean) {
  Object.defineProperty(document, 'hidden', { configurable: true, get: () => hidden })
  document.dispatchEvent(new Event('visibilitychange'))
}

const idleStatus: ActiveAgentStatusDTO = { running: false, flow: null, node: null, job_id: null }

beforeEach(() => {
  vi.useFakeTimers()
})

afterEach(() => {
  vi.useRealTimers()
  vi.restoreAllMocks()
  Object.defineProperty(document, 'hidden', { configurable: true, get: () => false })
})

describe('AgentStatusPanel', () => {
  it('poll ทันทีตอน mount แล้ววนซ้ำทุก 4 วินาที', async () => {
    const spy = vi.spyOn(api, 'getActiveAgentStatus').mockResolvedValue(idleStatus)
    render(<AgentStatusPanel />)
    expect(spy).toHaveBeenCalledTimes(1)

    await act(() => vi.advanceTimersByTimeAsync(4000))
    expect(spy).toHaveBeenCalledTimes(2)
    await act(() => vi.advanceTimersByTimeAsync(4000))
    expect(spy).toHaveBeenCalledTimes(3)
  })

  it('หยุด poll ตอน tab ถูกซ่อน และ poll ทันทีเมื่อกลับมา visible', async () => {
    const spy = vi.spyOn(api, 'getActiveAgentStatus').mockResolvedValue(idleStatus)
    render(<AgentStatusPanel />)
    expect(spy).toHaveBeenCalledTimes(1)

    act(() => setDocumentHidden(true))
    await act(() => vi.advanceTimersByTimeAsync(12000))
    expect(spy).toHaveBeenCalledTimes(1) // ไม่มี poll เพิ่มระหว่างซ่อน

    act(() => setDocumentHidden(false))
    expect(spy).toHaveBeenCalledTimes(2) // กลับมา visible → poll ทันทีไม่รอ interval
  })

  it('แสดงจุดสถานะ active ตาม node ที่กำลังรัน (flow manager)', async () => {
    vi.spyOn(api, 'getActiveAgentStatus').mockResolvedValue({
      running: true,
      flow: 'manager',
      node: 'researcher',
      job_id: 'j1',
    })
    render(<AgentStatusPanel />)
    // ให้ promise ของ poll แรก resolve
    await act(() => vi.advanceTimersByTimeAsync(0))

    const researcherRow = screen.getByText('Researcher').closest('li')
    expect(researcherRow?.querySelector('.bg-emerald-500')).not.toBeNull()
    const managerRow = screen.getByText('Manager').closest('li')
    expect(managerRow?.querySelector('.bg-emerald-500')).toBeNull()
  })

  it('flow อื่น (news_youtube) แสดง banner Running แทนการ mark agent รายตัว', async () => {
    vi.spyOn(api, 'getActiveAgentStatus').mockResolvedValue({
      running: true,
      flow: 'news_youtube',
      node: 'fetch_news',
      job_id: 'j2',
    })
    render(<AgentStatusPanel />)
    await act(() => vi.advanceTimersByTimeAsync(0))

    expect(screen.getByText(/Running: News\/YouTube/)).toBeInTheDocument()
  })
})
