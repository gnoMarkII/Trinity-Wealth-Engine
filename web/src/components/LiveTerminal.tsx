import { useEffect, useRef, useState } from 'react'
import type { JobStatusDTO, NewsYoutubeApprovalPayload } from '../api/types'
import type { TerminalStatus } from '../lib/agentStatus'
import { nodeDisplayName } from '../lib/nodeDisplayNames'

interface LogLine {
  node: string | null
  content: string
  role: 'instruction' | 'reply'
  label: string | null
}

interface StepBlock {
  key: string
  node: string | null
  messages: LogLine[]
}

// รวมบรรทัด log ที่ node เดียวกันติดกันเป็น step เดียว — เปลี่ยน node เมื่อไหร่ขึ้น step ใหม่
// ต่อท้ายด้านล่างทันที (รองรับ supervisor วนกลับมาเรียก node เดิมซ้ำ เพราะกราฟจริงเป็นแบบ
// dynamic dispatch ไม่มีลำดับตายตัว — ดู agents/manager_agent.py supervisor_node)
function groupIntoSteps(lines: LogLine[]): StepBlock[] {
  const blocks: StepBlock[] = []
  for (const line of lines) {
    const last = blocks[blocks.length - 1]
    if (last && last.node === line.node) {
      last.messages.push(line)
    } else {
      blocks.push({ key: `${blocks.length}-${line.node ?? 'system'}`, node: line.node, messages: [line] })
    }
  }
  return blocks
}

const LAST_STEP_DOT_CLASS: Record<TerminalStatus, string> = {
  idle: 'bg-zinc-300',
  streaming: 'animate-pulse bg-emerald-500',
  done: 'bg-emerald-500',
  error: 'bg-red-500',
  awaiting_approval: 'bg-amber-500',
}

interface Props {
  jobId: string | null
  onStatusChange?: (status: TerminalStatus) => void
  onNodeUpdate?: (node: string | null) => void
  onAwaitingApproval?: (payload: NewsYoutubeApprovalPayload) => void
  onLineCountChange?: (count: number) => void
  onLogEntry?: () => void
  /** true = รัน SSE/side-effects ตามปกติแต่ไม่ render กล่อง UI — ใช้เป็น "background driver"
   * ของ job ที่ active อยู่ เพื่อขับ auto column-transition + workspace preview บนการ์ด
   * โดยไม่ต้องมีกล่อง terminal ใหญ่ค้างอยู่ท้ายหน้า (ย้ายไปโชว์ใน KanbanDetailDrawer แทน) */
  hideUi?: boolean
}

export default function LiveTerminal({ jobId, onStatusChange, onNodeUpdate, onAwaitingApproval, onLineCountChange, onLogEntry, hideUi }: Props) {
  const [lines, setLines] = useState<LogLine[]>([])
  const [status, setStatus] = useState<TerminalStatus>('idle')
  const containerRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!jobId) {
      setLines([])
      setStatus('idle')
      return
    }
    setLines([])
    setStatus('streaming')

    const source = new EventSource(`/api/agents/stream/${jobId}`)

    source.onmessage = (e) => {
      let payload: LogLine
      try {
        payload = JSON.parse(e.data) as LogLine
      } catch {
        return // payload ผิดรูปแบบ — ข้ามบรรทัดนี้ไป ไม่ให้ทั้ง stream พัง
      }
      setLines((prev) => [...prev, payload])
      onNodeUpdate?.(payload.node)
      onLogEntry?.()
    }
    source.addEventListener('done', () => {
      setStatus('done')
      onNodeUpdate?.(null)
      source.close()
    })
    source.addEventListener('awaiting_approval', (e: MessageEvent) => {
      let dto: JobStatusDTO
      try {
        dto = JSON.parse(e.data) as JobStatusDTO
      } catch {
        return
      }
      setStatus('awaiting_approval')
      onNodeUpdate?.(null)
      if (dto.interrupt_payload) onAwaitingApproval?.(dto.interrupt_payload)
      source.close()
    })
    source.addEventListener('error', (e: Event) => {
      // แยก "backend ส่ง event: error จริง" (มี .data เช่น job not found) ออกจาก
      // "connection หลุดชั่วคราว" (native EventSource error, ไม่มี .data) — เจอจริงจาก
      // live test: network blip ธรรมดาทำให้ job ถูกเข้าใจผิดว่า error แล้วโดนย้ายกลับ backlog
      // ทั้งที่ backend ยังรันอยู่ปกติ ไม่ auto-reconnect เพราะ backend replay จาก seq 0 ทุกครั้ง
      // (จะทำให้ log ซ้ำ) — ปิด connection เงียบๆ แล้วให้ผู้ใช้เปิด drawer ใหม่เพื่อดูสถานะจริง
      if (e instanceof MessageEvent && typeof e.data === 'string') {
        setStatus('error')
        onNodeUpdate?.(null)
      }
      source.close()
    })

    return () => source.close()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [jobId])

  useEffect(() => {
    containerRef.current?.scrollTo({ top: containerRef.current.scrollHeight })
    onLineCountChange?.(lines.length)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [lines])

  useEffect(() => {
    onStatusChange?.(status)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [status])

  const statusColor =
    status === 'streaming'
      ? 'text-emerald-700'
      : status === 'done'
        ? 'text-zinc-500'
        : status === 'awaiting_approval'
          ? 'text-amber-700'
          : status === 'error'
            ? 'text-red-700'
            : 'text-zinc-400'

  if (hideUi) return null

  const steps = groupIntoSteps(lines)

  return (
    <div className="rounded-xl border border-zinc-200 bg-surface shadow-sm shadow-black/5">
      <div className="flex items-center justify-between border-b border-zinc-200 px-3 py-2">
        <span className="font-mono text-xs text-zinc-500">Live Execution Terminal</span>
        <span className={`flex items-center gap-1.5 font-mono text-xs ${statusColor}`}>
          {status === 'streaming' && (
            <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-emerald-500" />
          )}
          {status}
        </span>
      </div>
      <div ref={containerRef} className="max-h-96 overflow-y-auto p-3">
        {steps.length === 0 && <p className="text-xs text-zinc-400">รอสั่งงาน...</p>}
        {steps.map((step, i) => {
          const isLast = i === steps.length - 1
          const dotClass = isLast ? LAST_STEP_DOT_CLASS[status] : 'bg-zinc-400'
          return (
            <div key={step.key} className="flex gap-3">
              <div className="flex flex-col items-center">
                <span className={`mt-1 h-2 w-2 shrink-0 rounded-full ${dotClass}`} />
                {!isLast && <div className="w-px flex-1 bg-zinc-200" />}
              </div>
              <div className={`min-w-0 flex-1 ${isLast ? 'pb-1' : 'pb-3'}`}>
                <p className="text-xs font-semibold text-zinc-700">{nodeDisplayName(step.node)}</p>
                {step.messages.map((m, j) =>
                  m.role === 'instruction' ? (
                    <p key={j} className="mt-0.5 font-mono text-xs text-zinc-500">
                      <span className="text-zinc-400">↳ [{m.label ?? m.node}]</span> {m.content}
                    </p>
                  ) : (
                    <p key={j} className="mt-0.5 font-mono text-xs text-emerald-700">
                      {m.content}
                    </p>
                  ),
                )}
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}
