import { act, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import LiveTerminal from './LiveTerminal'

/** จำลอง EventSource ให้ควบคุมการยิง event ได้จาก test — เก็บทุก instance ที่ถูกสร้าง
 * เพื่อเช็คว่า connection ถูกเปิด/ปิดถูกจังหวะ */
class MockEventSource {
  static instances: MockEventSource[] = []
  url: string
  closed = false
  onmessage: ((e: MessageEvent) => void) | null = null
  private listeners = new Map<string, ((e: Event) => void)[]>()

  constructor(url: string) {
    this.url = url
    MockEventSource.instances.push(this)
  }

  addEventListener(type: string, cb: (e: Event) => void) {
    const list = this.listeners.get(type) ?? []
    list.push(cb)
    this.listeners.set(type, list)
  }

  close() {
    this.closed = true
  }

  emitMessage(data: unknown) {
    this.onmessage?.(new MessageEvent('message', { data: JSON.stringify(data) }))
  }

  emitNamed(type: string, data?: unknown) {
    const event =
      data === undefined
        ? new Event(type) // native error (network blip) — ไม่มี .data
        : new MessageEvent(type, { data: JSON.stringify(data) })
    this.listeners.get(type)?.forEach((cb) => cb(event))
  }

  static latest(): MockEventSource {
    const instance = MockEventSource.instances[MockEventSource.instances.length - 1]
    if (!instance) throw new Error('ยังไม่มี EventSource ถูกสร้าง')
    return instance
  }
}

beforeEach(() => {
  MockEventSource.instances = []
  vi.stubGlobal('EventSource', MockEventSource)
})

afterEach(() => {
  vi.unstubAllGlobals()
})

const line = (node: string, content: string) => ({ node, content, role: 'reply', label: null })

describe('LiveTerminal', () => {
  it('ไม่มี jobId → สถานะ idle ไม่เปิด connection', () => {
    render(<LiveTerminal jobId={null} />)
    expect(screen.getByText('รอสั่งงาน...')).toBeInTheDocument()
    expect(MockEventSource.instances).toHaveLength(0)
  })

  it('เปิด SSE ตาม jobId และ render log ที่ stream เข้ามาแบบ group ตาม node', () => {
    render(<LiveTerminal jobId="job-1" />)
    const source = MockEventSource.latest()
    expect(source.url).toBe('/api/agents/stream/job-1')

    act(() => {
      source.emitMessage(line('supervisor', 'มอบหมายงาน'))
      source.emitMessage(line('researcher', 'ค้นข้อมูล 1'))
      source.emitMessage(line('researcher', 'ค้นข้อมูล 2'))
    })

    expect(screen.getByText('Manager')).toBeInTheDocument()
    expect(screen.getByText('Researcher')).toBeInTheDocument()
    expect(screen.getByText('ค้นข้อมูล 2')).toBeInTheDocument()
  })

  it('event done → สถานะ done, ปิด connection, แจ้ง onStatusChange', () => {
    const onStatusChange = vi.fn()
    render(<LiveTerminal jobId="job-1" onStatusChange={onStatusChange} />)
    const source = MockEventSource.latest()

    act(() => source.emitNamed('done', {}))

    expect(onStatusChange).toHaveBeenLastCalledWith('done')
    expect(source.closed).toBe(true)
  })

  it('awaiting_approval ส่ง interrupt_payload ให้ onAwaitingApproval', () => {
    const onAwaitingApproval = vi.fn()
    render(<LiveTerminal jobId="job-1" onAwaitingApproval={onAwaitingApproval} />)
    const source = MockEventSource.latest()
    const payload = { type: 'news_youtube_approval', news_candidates: [], youtube_candidates: [] }

    act(() => source.emitNamed('awaiting_approval', { interrupt_payload: payload }))

    expect(onAwaitingApproval).toHaveBeenCalledWith(payload)
    expect(source.closed).toBe(true)
  })

  it('error จาก backend (มี .data) → สถานะ error / network blip (ไม่มี .data) → ไม่ใช่ error', () => {
    const onStatusChange = vi.fn()
    const { unmount } = render(<LiveTerminal jobId="job-1" onStatusChange={onStatusChange} />)
    let source = MockEventSource.latest()

    // network blip: native Event ไม่มี data — ต้องไม่ mark เป็น error (แค่ปิด connection)
    act(() => source.emitNamed('error'))
    expect(onStatusChange).not.toHaveBeenCalledWith('error')
    expect(source.closed).toBe(true)
    unmount()

    render(<LiveTerminal jobId="job-2" onStatusChange={onStatusChange} />)
    source = MockEventSource.latest()
    act(() => source.emitNamed('error', { detail: 'job not found' }))
    expect(onStatusChange).toHaveBeenLastCalledWith('error')
  })

  it('payload ผิดรูปแบบ → ข้ามบรรทัดนั้น stream ไม่พัง', () => {
    render(<LiveTerminal jobId="job-1" />)
    const source = MockEventSource.latest()

    act(() => {
      source.onmessage?.(new MessageEvent('message', { data: '{broken json' }))
      source.emitMessage(line('researcher', 'บรรทัดดี'))
    })

    expect(screen.getByText('บรรทัดดี')).toBeInTheDocument()
  })

  it('hideUi ไม่ render UI แต่ side-effects ยังทำงาน (background driver)', () => {
    const onNodeUpdate = vi.fn()
    const onLogEntry = vi.fn()
    const { container } = render(
      <LiveTerminal jobId="job-1" hideUi onNodeUpdate={onNodeUpdate} onLogEntry={onLogEntry} />
    )
    const source = MockEventSource.latest()

    act(() => source.emitMessage(line('researcher', 'ทำงานเบื้องหลัง')))

    expect(container).toBeEmptyDOMElement()
    expect(onNodeUpdate).toHaveBeenCalledWith('researcher')
    expect(onLogEntry).toHaveBeenCalledOnce()
  })

  it('unmount ระหว่าง streaming → ปิด connection ทิ้ง', () => {
    const { unmount } = render(<LiveTerminal jobId="job-1" />)
    const source = MockEventSource.latest()
    expect(source.closed).toBe(false)
    unmount()
    expect(source.closed).toBe(true)
  })
})
