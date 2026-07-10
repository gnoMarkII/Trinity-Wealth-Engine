import { useEffect, useRef, useState } from 'react'
import { api, ApiError } from '../../api/client'
import type { KanbanCardDTO, NewsYoutubeApprovalPayload } from '../../api/types'
import LiveTerminal from '../LiveTerminal'
import ApprovalPanel from '../ApprovalPanel'
import { FLOW_TAG } from '../../lib/flowTags'
import { columnForStatus, type TerminalStatus } from '../../lib/agentStatus'

interface Props {
  card: KanbanCardDTO | null
  onClose: () => void
  onCardTransition: () => void
}

const WIDTH_STORAGE_KEY = 'kanban-drawer-width'
const DEFAULT_WIDTH = 384
const MIN_WIDTH = 300
const MAX_WIDTH = 720

function loadStoredWidth(): number {
  const raw = window.localStorage.getItem(WIDTH_STORAGE_KEY)
  const parsed = raw ? parseInt(raw, 10) : NaN
  return Number.isFinite(parsed) ? Math.min(Math.max(parsed, MIN_WIDTH), MAX_WIDTH) : DEFAULT_WIDTH
}

/**
 * แผงด้านขวาแบบ split-pane ถาวร (ไม่หุบหายเมื่อไม่มีการ์ดถูกเลือก) ปรับความกว้างซ้าย-ขวาได้
 * ผูกกับ card.job_id โดยตรง — ไม่ใช้ activeJobId ระดับหน้าเพจ (ดู Rev.2 §0: จุดเสี่ยง
 * สถาปัตยกรรมที่สุด) ทำให้คลิกการ์ดไหนก็เห็น terminal/approval ของงานนั้นจริง แม้จะไม่ใช่
 * งานล่าสุดที่เพิ่ง dispatch ก็ตาม — SSE endpoint replay log ทั้งหมดจาก seq 0 เสมอตอนเปิด
 * connection ใหม่ จึงใช้งานกับ job เก่าที่ทำเสร็จไปแล้วได้ปกติ
 */
export default function KanbanDetailDrawer({ card, onClose, onCardTransition }: Props) {
  const [approvalPayload, setApprovalPayload] = useState<NewsYoutubeApprovalPayload | null>(null)
  const [approving, setApproving] = useState(false)
  const [terminalKey, setTerminalKey] = useState(0)
  const [error, setError] = useState<string | null>(null)
  const [width, setWidth] = useState(loadStoredWidth)
  const [resizing, setResizing] = useState(false)
  const widthRef = useRef(width)

  useEffect(() => {
    setApprovalPayload(null)
    setTerminalKey((k) => k + 1)
    setError(null)
  }, [card?.card_id])

  useEffect(() => {
    if (!resizing) return

    function onMouseMove(e: MouseEvent) {
      const next = Math.min(Math.max(window.innerWidth - e.clientX, MIN_WIDTH), MAX_WIDTH)
      widthRef.current = next
      setWidth(next)
    }
    function onMouseUp() {
      setResizing(false)
      window.localStorage.setItem(WIDTH_STORAGE_KEY, String(widthRef.current))
    }

    window.addEventListener('mousemove', onMouseMove)
    window.addEventListener('mouseup', onMouseUp)
    document.body.style.cursor = 'col-resize'
    document.body.style.userSelect = 'none'
    return () => {
      window.removeEventListener('mousemove', onMouseMove)
      window.removeEventListener('mouseup', onMouseUp)
      document.body.style.cursor = ''
      document.body.style.userSelect = ''
    }
  }, [resizing])

  function handleStatusChange(status: TerminalStatus) {
    if (!card) return
    const targetColumn = columnForStatus(status)
    if (!targetColumn || card.column_name === targetColumn) return
    api
      .moveKanbanCard(card.card_id, targetColumn)
      .then(onCardTransition)
      .catch((e) => setError(e instanceof ApiError ? e.message : 'อัปเดตสถานะการ์ดไม่สำเร็จ'))
  }

  async function handleApprove(approvedNewsLinks: string[], approvedYoutubeLinks: string[]) {
    if (!card?.job_id) return
    setApproving(true)
    setError(null)
    try {
      await api.resumeJob(card.job_id, approvedNewsLinks, approvedYoutubeLinks)
      setApprovalPayload(null)
      await api.moveKanbanCard(card.card_id, 'executing')
      onCardTransition()
      setTerminalKey((k) => k + 1)
    } catch (e) {
      setError(e instanceof ApiError ? e.message : 'อนุมัติไม่สำเร็จ')
    } finally {
      setApproving(false)
    }
  }

  return (
    <aside
      style={{ width }}
      className="sticky top-0 -mr-8 -my-8 flex h-screen shrink-0 border border-r-0 border-zinc-200 bg-white shadow-sm shadow-black/5"
    >
      {/* resize handle — ลากซ้าย/ขวาเพื่อปรับความกว้าง Drawer */}
      <div
        onMouseDown={(e) => {
          e.preventDefault()
          setResizing(true)
        }}
        className={`group -ml-1.5 w-3 shrink-0 cursor-col-resize ${resizing ? 'bg-terra-light/20' : ''}`}
      >
        <div className="mx-auto h-full w-px bg-zinc-200 transition-colors group-hover:bg-terra-light/60" />
      </div>

      <div className="flex min-w-0 flex-1 flex-col">
        <div className="flex shrink-0 items-center justify-between border-b border-zinc-200 px-4 py-3">
          {card ? (
            <div className="flex items-center gap-2">
              {card.display_seq != null && (
                <span className="font-mono text-xs text-zinc-400">#AG-{card.display_seq}</span>
              )}
              <span className="rounded border border-purple-200/60 bg-purple-50 px-1.5 py-0.5 text-[10px] text-purple-700">
                {FLOW_TAG[card.flow] ?? `#${card.flow}`}
              </span>
            </div>
          ) : (
            <span className="text-xs font-semibold text-zinc-500">Task Details</span>
          )}
          {card && (
            <button
              onClick={onClose}
              className="rounded p-1 text-zinc-400 transition-colors hover:bg-zinc-100 hover:text-zinc-700 focus-visible:outline focus-visible:outline-2 focus-visible:outline-terra"
              title="ยกเลิกการเลือก"
              aria-label="ยกเลิกการเลือก"
            >
              ✕
            </button>
          )}
        </div>

        <div className="min-h-0 flex-1 space-y-4 overflow-y-auto p-4">
          {!card ? (
            <div className="flex h-full flex-col items-center justify-center gap-2 text-center text-zinc-400">
              <span className="text-2xl">🗂️</span>
              <p className="text-xs">คลิกการ์ดในบอร์ดเพื่อดูรายละเอียด</p>
            </div>
          ) : (
            <>
              {error && (
                <p className="rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-xs text-red-700">{error}</p>
              )}
              <div>
                <h2 className="text-sm font-medium leading-snug text-zinc-900">{card.title}</h2>
                <p className="mt-1 text-xs text-zinc-500">
                  สร้างเมื่อ {new Date(card.created_at * 1000).toLocaleString('th-TH')}
                </p>
                {card.prompt && (
                  <p className="mt-2 whitespace-pre-wrap rounded-lg border border-zinc-200 bg-surface p-2 text-xs text-zinc-600">
                    {card.prompt}
                  </p>
                )}
              </div>

              {card.job_id ? (
                <div className="space-y-3">
                  <LiveTerminal
                    key={terminalKey}
                    jobId={card.job_id}
                    onStatusChange={handleStatusChange}
                    onAwaitingApproval={setApprovalPayload}
                  />
                  {approvalPayload && (
                    <ApprovalPanel payload={approvalPayload} onApprove={handleApprove} submitting={approving} />
                  )}
                </div>
              ) : (
                <p className="rounded-lg border border-zinc-200 bg-surface p-3 text-xs text-zinc-500">
                  การ์ดนี้ยังไม่ได้ถูก dispatch — กดปุ่ม ▶ บนการ์ดใน Backlog เพื่อเริ่มทำงาน
                </p>
              )}
            </>
          )}
        </div>
      </div>
    </aside>
  )
}
