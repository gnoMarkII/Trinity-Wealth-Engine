import { useEffect, useRef, useState } from 'react'
import { api, ApiError } from '../../api/client'
import type { KanbanCardDTO, NotebookLMAvailableSourceDTO, NotebookLMStatusDTO } from '../../api/types'

const POLL_INTERVAL_MS = 10_000
const TERMINAL_JOB_STATUSES = new Set(['done', 'done_with_warnings', 'done_with_errors', 'error'])


const MANIFEST_STATUS_LABEL: Record<string, string> = {
  initialized: 'กำลังเตรียม...',
  notebook_created: 'กำลังเตรียม...',
  source_added: 'กำลังเตรียม...',
  research_done: 'กำลังค้นคว้าเพิ่มเติม...',
  audio_generating: 'กำลังสร้าง Audio...',
  completed: 'Audio พร้อมแล้ว',
}

interface Props {
  card: KanbanCardDTO
  onCardTransition: () => void
}

export default function NotebookLMCardDetail({ card, onCardTransition }: Props) {
  const [status, setStatus] = useState<NotebookLMStatusDTO | null>(null)
  const [dispatching, setDispatching] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [availableSources, setAvailableSources] = useState<NotebookLMAvailableSourceDTO[] | null>(null)
  const [loadingSources, setLoadingSources] = useState(false)
  const [selectedPath, setSelectedPath] = useState('')
  const intervalRef = useRef<number | null>(null)

  useEffect(() => {
    setStatus(null)
    setError(null)
    setSelectedPath('')
  }, [card.card_id])

  // ยังไม่เลือกไฟล์ให้การ์ดนี้ (card.prompt ว่าง) — โหลดรายชื่อ Briefing Book ให้เลือก
  useEffect(() => {
    if (card.prompt) return
    setLoadingSources(true)
    api
      .getNotebookLMAvailableSources()
      .then(setAvailableSources)
      .catch(() => setError('โหลดรายชื่อ Briefing Book ไม่สำเร็จ'))
      .finally(() => setLoadingSources(false))
  }, [card.card_id, card.prompt])

  useEffect(() => {
    const jobId = card.job_id
    if (!jobId) return undefined
    let cancelled = false

    async function pollOnce() {
      try {
        const s = await api.getNotebookLMStatus(jobId as string)
        if (cancelled) return
        setStatus(s)
        if (TERMINAL_JOB_STATUSES.has(s.status)) {
          if (intervalRef.current) window.clearInterval(intervalRef.current)
          onCardTransition() // ให้บอร์ดรีเฟรช column_name ที่ backend ย้ายให้แล้ว (executing -> done/backlog)
        }
      } catch {
        // เงียบไว้ — poll รอบถัดไปลองใหม่เอง ไม่ต้องขึ้น error ทุกครั้งที่เน็ตสะดุด
      }
    }

    pollOnce() // เช็คทันทีตอนเปิด Drawer ไม่ต้องรอ interval แรกผ่านไป 10s ก่อน
    intervalRef.current = window.setInterval(pollOnce, POLL_INTERVAL_MS)
    return () => {
      cancelled = true
      if (intervalRef.current) window.clearInterval(intervalRef.current)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [card.job_id])

  async function handleGenerate() {
    if (!card.prompt && !selectedPath) return
    setDispatching(true)
    setError(null)
    try {
      const res = await api.generateNotebookLMAudio(card.card_id, card.prompt ? undefined : selectedPath)
      setStatus({ job_id: res.job_id, status: res.status, audio_path: null, notebook_id: null, error: null })
      onCardTransition()
    } catch (e) {
      setError(e instanceof ApiError ? e.message : 'สั่งสร้าง Audio ไม่สำเร็จ')
    } finally {
      setDispatching(false)
    }
  }

  if (!card.prompt) {
    return (
      <div className="space-y-3">
        {error && <p className="rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-xs text-red-700">{error}</p>}
        <div>
          <label htmlFor="notebooklm-source-select" className="mb-1 block text-xs font-medium text-zinc-600">
            เลือก Briefing Book
          </label>
          {loadingSources && <p className="text-xs text-zinc-500">กำลังโหลด...</p>}
          {availableSources && availableSources.length === 0 && (
            <p className="text-xs text-zinc-500">ไม่พบไฟล์ใน NotebookLM_Sources/</p>
          )}
          {availableSources && availableSources.length > 0 && (
            <select
              id="notebooklm-source-select"
              value={selectedPath}
              onChange={(e) => setSelectedPath(e.target.value)}
              className="w-full rounded-lg border border-edge bg-panel px-2 py-1.5 text-xs text-zinc-900 outline-none focus:border-sky-500 focus:ring-1 focus:ring-sky-500/30"
            >
              <option value="">— เลือกไฟล์ —</option>
              {availableSources.map((s) => (
                <option key={s.file_path} value={s.file_path}>
                  {s.is_verified ? '🟢' : '🟡'} {s.title} {s.date_part ? `(${s.date_part})` : ''}
                </option>
              ))}
            </select>
          )}
        </div>
        <button
          onClick={handleGenerate}
          disabled={!selectedPath || dispatching}
          className="rounded-md border border-emerald-300 bg-emerald-100/90 px-2.5 py-1 text-xs font-semibold text-emerald-700 hover:border-emerald-600 hover:bg-emerald-600 hover:text-white disabled:cursor-not-allowed disabled:opacity-40"
        >
          {dispatching ? 'กำลังส่ง...' : 'ยืนยันและสร้าง Audio'}
        </button>
      </div>
    )
  }

  const audioPath = status?.audio_path ?? null
  const isCompleted = card.column_name === 'done' || !!audioPath
  const isError = status?.status === 'error'
  const canRetry = isError

  let progressLabel = 'ยังไม่สร้าง Audio'
  if (isCompleted) {
    progressLabel = 'Audio พร้อมแล้ว'
  } else if (status) {
    progressLabel = MANIFEST_STATUS_LABEL[status.status] ?? (status.status === 'queued' ? 'รอคิว...' : status.status)
  } else if (card.job_id) {
    progressLabel = 'กำลังเตรียม...'
  }

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-2">
        <span
          className={`rounded px-1.5 py-0.5 text-[10px] font-medium ${
            card.is_verified
              ? 'border border-emerald-200 bg-emerald-50 text-emerald-700'
              : 'border border-amber-200 bg-amber-50 text-amber-700'
          }`}
        >
          {card.is_verified ? '🟢 verified' : '🟡 unverified draft'}
        </span>
      </div>



      {error && <p className="rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-xs text-red-700">{error}</p>}
      {isError && status?.error && !error && (
        <p className="rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-xs text-red-700">{status.error}</p>
      )}
      {status?.recovery_status && (
        <p className="rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-xs text-amber-800 font-mono">
          ⚠️ Recovery Blocker: {status.recovery_status}
        </p>
      )}

      {card.prompt && (
        <div className="rounded-md border border-zinc-200 bg-surface px-2.5 py-2">
          <p className="text-[10px] font-medium text-zinc-500 uppercase tracking-wider mb-1">Selected Briefing Book</p>
          <p className="truncate font-mono text-xs text-zinc-700" title={card.prompt}>
            📄 {card.prompt.split(/[\\/]/).pop()}
          </p>
        </div>
      )}

      <div
        className={`flex items-center gap-1.5 rounded-md border px-2 py-1 text-xs font-medium ${
          isCompleted
            ? 'border-emerald-200 bg-emerald-50 text-emerald-700'
            : isError
              ? 'border-red-200 bg-red-50 text-red-700'
              : card.job_id
                ? 'border-sky-200 bg-sky-50 text-sky-700'
                : 'border-zinc-200 bg-surface text-zinc-500'
        }`}
      >
        {!isCompleted && !isError && card.job_id && <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-sky-500" />}
        <span>{progressLabel}</span>
      </div>

      {audioPath && (
        <p className="truncate text-xs text-zinc-500" title={audioPath}>
          📄 {audioPath}
        </p>
      )}

      {canRetry && (
        <button
          onClick={handleGenerate}
          disabled={dispatching}
          className="rounded-md border border-emerald-300 bg-emerald-100/90 px-2.5 py-1 text-xs font-semibold text-emerald-700 hover:border-emerald-600 hover:bg-emerald-600 hover:text-white disabled:cursor-not-allowed disabled:opacity-40"
        >
          {dispatching ? 'กำลังส่ง...' : 'ลองใหม่'}
        </button>
      )}
    </div>
  )
}
