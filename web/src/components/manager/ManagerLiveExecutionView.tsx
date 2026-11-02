import { useEffect, useRef, useState } from 'react'
import { api } from '../../api/client'
import type { JobOutputsDTO, JobStatusDTO } from '../../api/types'
import type { TerminalStatus } from '../../lib/agentStatus'
import { nodeDisplayName } from '../../lib/nodeDisplayNames'
import { groupIntoSteps, type LogLine } from '../../lib/terminalSteps'
import {
  MACRO_PIPELINE,
  EQUITY_PIPELINE,
} from '../../lib/agentPipelines'

export interface ManagerLiveExecutionViewProps {
  jobId: string | null
  allJobIds?: string[]
  onSelectJob?: (jobId: string) => void
  onTriggerQuickJob?: (prompt: string) => void
  onSwitchToSwarm?: () => void
}

export default function ManagerLiveExecutionView({
  jobId,
  allJobIds = [],
  onSelectJob,
  onTriggerQuickJob,
  onSwitchToSwarm,
}: ManagerLiveExecutionViewProps) {
  const [lines, setLines] = useState<LogLine[]>([])
  const [status, setStatus] = useState<TerminalStatus>('idle')
  const [currentNode, setCurrentNode] = useState<string | null>(null)
  const [jobDetail, setJobDetail] = useState<JobStatusDTO | null>(null)
  const [outputs, setOutputs] = useState<JobOutputsDTO | null>(null)
  const [errorMessage, setErrorMessage] = useState<string | null>(null)
  const [copied, setCopied] = useState(false)
  const [elapsedSeconds, setElapsedSeconds] = useState(0)

  const logContainerRef = useRef<HTMLDivElement>(null)

  // 1. Reset & Setup Live Stream when jobId changes
  useEffect(() => {
    if (!jobId) {
      setLines([])
      setStatus('idle')
      setCurrentNode(null)
      setJobDetail(null)
      setOutputs(null)
      setErrorMessage(null)
      setElapsedSeconds(0)
      return
    }

    setLines([])
    setStatus('streaming')
    setErrorMessage(null)
    setElapsedSeconds(0)

    // Initial fetch of job metadata
    api
      .getJobStatus(jobId)
      .then((dto) => {
        setJobDetail(dto)
        if (dto.current_node) setCurrentNode(dto.current_node)
        if (dto.status) setStatus(dto.status as TerminalStatus)
        if (dto.error_message) setErrorMessage(dto.error_message)
      })
      .catch(() => {})

    if (typeof EventSource === 'undefined') return

    const source = new EventSource(`/api/agents/stream/${jobId}`)

    source.onmessage = (e) => {
      try {
        const payload = JSON.parse(e.data) as LogLine
        setLines((prev) => [...prev, payload])
        if (payload.node) setCurrentNode(payload.node)
      } catch {
        // Skip malformed log
      }
    }

    source.addEventListener('done', () => {
      setStatus('done')
      setCurrentNode(null)
      source.close()
      // Fetch completed outputs
      api.getJobOutputs(jobId).then(setOutputs).catch(() => {})
    })

    source.addEventListener('done_with_warnings', () => {
      setStatus('done_with_warnings')
      setCurrentNode(null)
      source.close()
      api.getJobOutputs(jobId).then(setOutputs).catch(() => {})
    })

    source.addEventListener('done_with_errors', () => {
      setStatus('done_with_errors')
      setCurrentNode(null)
      source.close()
      api.getJobOutputs(jobId).then(setOutputs).catch(() => {})
    })

    source.addEventListener('awaiting_approval', () => {
      setStatus('awaiting_approval')
      setCurrentNode(null)
      source.close()
    })

    source.addEventListener('error', (e: Event) => {
      if (e instanceof MessageEvent && typeof e.data === 'string') {
        let detail = 'เกิดข้อผิดพลาดในการประมวลผลงานวิจัย'
        try {
          const dto = JSON.parse(e.data) as { error_message?: string; detail?: string }
          detail = dto.error_message || dto.detail || detail
        } catch {
          // ignore
        }
        setErrorMessage(detail)
        setStatus('error')
        setCurrentNode(null)
      }
      source.close()
    })

    return () => source.close()
  }, [jobId])

  // 2. Elapsed Timer while running
  useEffect(() => {
    if (status !== 'streaming' && status !== 'idle') return
    const timer = setInterval(() => {
      setElapsedSeconds((prev) => prev + 1)
    }, 1000)
    return () => clearInterval(timer)
  }, [status])

  // 3. Auto-scroll logs
  useEffect(() => {
    if (logContainerRef.current) {
      logContainerRef.current.scrollTop = logContainerRef.current.scrollHeight
    }
  }, [lines])

  // 4. Copy Job ID helper
  function handleCopyJobId() {
    if (!jobId) return
    navigator.clipboard.writeText(jobId).then(() => {
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    })
  }

  // Determine Pipeline type: Equity or Macro
  const isEquity =
    lines.some((l) => l.node?.includes('equity')) ||
    currentNode?.includes('equity') ||
    Boolean(jobDetail?.card_id?.toLowerCase().includes('equity'))

  const pipeline = isEquity ? EQUITY_PIPELINE : MACRO_PIPELINE

  // Calculate Stage States
  const isDone =
    status === 'done' || status === 'done_with_warnings' || status === 'done_with_errors'
  const isError = status === 'error'

  // Map each stage to completed / active / pending
  const stagesWithState = pipeline.map((stage, idx) => {
    const isStageActive =
      !isDone && !isError && Boolean(currentNode && stage.nodeKeys.includes(currentNode))
    
    // Check if stage has produced any logs
    const hasLogged = lines.some((l) => l.node && stage.nodeKeys.includes(l.node))

    // Index of the highest reached stage
    let highestIdx = -1
    for (let i = pipeline.length - 1; i >= 0; i--) {
      const pStage = pipeline[i]
      if (
        (currentNode && pStage?.nodeKeys.includes(currentNode)) ||
        lines.some((l) => l.node && pStage?.nodeKeys.includes(l.node))
      ) {
        highestIdx = i
        break
      }
    }

    let state: 'completed' | 'active' | 'pending' = 'pending'
    if (isDone) {
      state = 'completed'
    } else if (isStageActive) {
      state = 'active'
    } else if (idx < highestIdx || (hasLogged && idx < highestIdx)) {
      state = 'completed'
    }

    return {
      ...stage,
      state,
    }
  })

  // Calculate Progress %
  const completedStagesCount = stagesWithState.filter((s) => s.state === 'completed').length
  const progressPercent = isDone
    ? 100
    : isError
      ? 100
      : Math.min(95, Math.max(12, Math.round((completedStagesCount / pipeline.length) * 100)))

  // Formatted Elapsed Time
  const minutes = Math.floor(elapsedSeconds / 60)
  const seconds = elapsedSeconds % 60
  const formattedTimer = `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}s`

  // -------------------------------------------------------------
  // EMPTY STATE: No active or selected job
  // -------------------------------------------------------------
  if (!jobId) {
    return (
      <div className="flex h-full flex-col min-h-0 overflow-y-auto custom-scrollbar p-4 space-y-4 animate-page-in">
        <div className="rounded-2xl border border-sky-100 bg-gradient-to-br from-white/95 via-sky-50/40 to-cyan-50/60 p-5 text-center shadow-sm backdrop-blur-md">
          <div className="mx-auto mb-3 flex h-14 w-14 items-center justify-center rounded-2xl bg-gradient-to-tr from-sky-400 via-cyan-400 to-sky-600 text-white shadow-[0_6px_20px_rgba(14,165,233,0.3)] animate-pulse">
            <span className="text-2xl">⚡</span>
          </div>
          <h3 className="text-sm font-bold text-sky-950 mb-1">
            พร้อมติดตามความคืบหน้างานวิจัย
          </h3>
          <p className="text-xs text-zinc-500 leading-relaxed max-w-xs mx-auto mb-4">
            ยังไม่มีงานวิจัยที่กำลังทำงานอยู่ เมื่อคุณสั่งให้ AI วิเคราะห์ภาพรวมเศรษฐกิจหรือหุ้นรายตัว รายละเอียดขั้นตอนจะแสดงสดที่นี่ทันที
          </p>

          <div className="flex flex-col gap-2">
            {onTriggerQuickJob && (
              <>
                <button
                  type="button"
                  onClick={() => onTriggerQuickJob('อัปเดต macro report')}
                  className="flex items-center justify-center gap-2 rounded-xl border border-sky-200 bg-white px-3 py-2 text-xs font-semibold text-sky-800 shadow-sm hover:bg-sky-50 transition-colors"
                >
                  <span>🌐</span>
                  <span>สั่งอัปเดต Macro Report</span>
                </button>
                <button
                  type="button"
                  onClick={() => onTriggerQuickJob('วิเคราะห์หุ้น NVDA เชิงลึก')}
                  className="flex items-center justify-center gap-2 rounded-xl border border-sky-200 bg-white px-3 py-2 text-xs font-semibold text-sky-800 shadow-sm hover:bg-sky-50 transition-colors"
                >
                  <span>📈</span>
                  <span>สั่งวิเคราะห์หุ้น NVDA</span>
                </button>
              </>
            )}

            {onSwitchToSwarm && (
              <button
                type="button"
                onClick={onSwitchToSwarm}
                className="mt-1 flex items-center justify-center gap-1.5 rounded-xl border border-dashed border-sky-300 bg-sky-50/60 px-3 py-2 text-xs font-medium text-sky-700 hover:bg-sky-100/70 transition-colors"
              >
                <span>🐝 ดูผังการทำงานของ Trinity Swarm (5 Specialists)</span>
              </button>
            )}
          </div>
        </div>
      </div>
    )
  }

  // -------------------------------------------------------------
  // ACTIVE JOB EXECUTION VIEW
  // -------------------------------------------------------------
  const groupedSteps = groupIntoSteps(lines)

  return (
    <div className="flex h-full flex-col min-h-0 overflow-y-auto custom-scrollbar p-4 space-y-4 animate-page-in">
      {/* Job Selector Switcher (if multiple jobs available) */}
      {allJobIds.length > 1 && (
        <div className="flex items-center gap-2 overflow-x-auto pb-1 custom-scrollbar">
          <span className="text-[10px] font-bold text-zinc-400 shrink-0">เลือกงาน:</span>
          {allJobIds.map((id) => (
            <button
              key={id}
              type="button"
              onClick={() => onSelectJob?.(id)}
              className={`shrink-0 rounded-lg px-2 py-0.5 text-[10px] font-mono transition-all ${
                id === jobId
                  ? 'border border-sky-300 bg-sky-100 text-sky-900 font-bold shadow-sm'
                  : 'border border-zinc-200 bg-white/70 text-zinc-600 hover:bg-sky-50'
              }`}
            >
              {id.slice(0, 8)}...
            </button>
          ))}
        </div>
      )}

      {/* Main Execution Card Header */}
      <div className="rounded-2xl border border-sky-100 bg-gradient-to-br from-white/90 via-sky-50/40 to-cyan-50/60 p-3.5 shadow-sm backdrop-blur-md">
        <div className="flex items-start justify-between gap-2">
          <div className="flex items-center gap-2.5">
            <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-gradient-to-tr from-sky-500 to-cyan-500 text-white shadow-sm">
              <span className="text-base">{isEquity ? '🔬' : '🌐'}</span>
            </div>
            <div>
              <h3 className="text-xs font-bold text-sky-950 flex items-center gap-1.5">
                <span>{isEquity ? 'Equity Deep Dive' : 'Macro Strategy Update'}</span>
              </h3>
              <div className="flex items-center gap-1.5 mt-0.5">
                <button
                  type="button"
                  onClick={handleCopyJobId}
                  title="คลิกเพื่อคัดลอก Job ID"
                  className="font-mono text-[10px] text-zinc-500 hover:text-sky-700 flex items-center gap-1 transition-colors"
                >
                  <span>ID: {jobId.slice(0, 14)}...</span>
                  <span className="text-[11px]">{copied ? '✓' : '📋'}</span>
                </button>
              </div>
            </div>
          </div>

          {/* Status Badge & Elapsed Timer */}
          <div className="flex flex-col items-end gap-1 shrink-0">
            <span
              className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-0.5 text-[10px] font-semibold transition-all ${
                isDone
                  ? 'border border-emerald-200 bg-emerald-50 text-emerald-700'
                  : isError
                    ? 'border border-rose-200 bg-rose-50 text-rose-700'
                    : 'border border-sky-200 bg-sky-50 text-sky-700 shadow-[0_0_8px_rgba(14,165,233,0.25)]'
              }`}
            >
              <span
                className={`h-2 w-2 rounded-full ${
                  isDone
                    ? 'bg-emerald-500'
                    : isError
                      ? 'bg-rose-500'
                      : 'animate-pulse bg-sky-500'
                }`}
              />
              {isDone
                ? 'เสร็จสมบูรณ์'
                : isError
                  ? 'เกิดข้อผิดพลาด'
                  : 'กำลังประมวลผล'}
            </span>

            <span className="font-mono text-[10px] text-zinc-400 flex items-center gap-1">
              <span>⏱️</span>
              <span>{formattedTimer}</span>
            </span>
          </div>
        </div>

        {/* Shimmering Progress Bar */}
        <div className="mt-3">
          <div className="flex justify-between text-[10px] font-semibold text-zinc-500 mb-1">
            <span>ความคืบหน้ารวม</span>
            <span className="font-mono text-sky-800">{progressPercent}%</span>
          </div>
          <div className="h-1.5 w-full rounded-full bg-sky-100/80 overflow-hidden">
            <div
              className={`h-full rounded-full transition-all duration-500 ${
                isDone
                  ? 'bg-emerald-500'
                  : isError
                    ? 'bg-rose-500'
                    : 'bg-gradient-to-r from-sky-400 via-cyan-400 to-sky-600 animate-shimmer'
              }`}
              style={{ width: `${progressPercent}%` }}
            />
          </div>
        </div>
      </div>

      {/* Visual Multi-Stage Workflow Pipeline */}
      <div className="rounded-2xl border border-sky-100 bg-white/80 p-3.5 shadow-sm">
        <div className="flex items-center justify-between mb-3 px-0.5">
          <h4 className="text-[10px] font-bold uppercase tracking-wider text-zinc-500 flex items-center gap-1">
            <span>🗺️ ลำดับขั้นตอนการทำงาน (Workflow Pipeline)</span>
          </h4>
          <span className="text-[10px] font-mono text-zinc-400">
            {completedStagesCount}/{pipeline.length} เสร็จสิ้น
          </span>
        </div>

        <div className="space-y-2 relative">
          {stagesWithState.map((stage, idx) => {
            const isLast = idx === stagesWithState.length - 1
            const isStageActive = stage.state === 'active'
            const isStageCompleted = stage.state === 'completed'

            return (
              <div key={stage.id} className="relative">
                {/* Connecting Line */}
                {!isLast && (
                  <div
                    className={`absolute left-4 top-8 -bottom-2 w-0.5 transition-colors z-0 ${
                      isStageCompleted ? 'bg-emerald-300' : 'bg-sky-100'
                    }`}
                  />
                )}

                <div
                  className={`relative z-10 flex items-center justify-between gap-2.5 rounded-xl border p-2 transition-all ${
                    isStageActive
                      ? 'border-sky-300 bg-gradient-to-r from-sky-50 to-cyan-50/80 shadow-[0_2px_12px_rgba(14,165,233,0.15)] ring-2 ring-sky-300/60 animate-card-in'
                      : isStageCompleted
                        ? 'border-emerald-100 bg-emerald-50/50 text-emerald-950'
                        : 'border-zinc-100 bg-white/50 opacity-60'
                  }`}
                >
                  <div className="flex items-center gap-2.5 min-w-0">
                    {/* Stage Icon with State Halo */}
                    <div
                      className={`relative flex h-8 w-8 shrink-0 items-center justify-center rounded-xl text-sm transition-transform ${
                        isStageActive
                          ? 'bg-gradient-to-tr from-sky-400 to-cyan-500 text-white shadow-[0_2px_8px_rgba(14,165,233,0.35)] scale-105'
                          : isStageCompleted
                            ? 'bg-emerald-100 text-emerald-700'
                            : 'bg-zinc-100 text-zinc-400'
                      }`}
                    >
                      {isStageCompleted ? (
                        <span className="font-bold text-emerald-700">✓</span>
                      ) : (
                        <span>{stage.icon}</span>
                      )}

                      {/* Active Beacon Pulse */}
                      {isStageActive && (
                        <span className="absolute -top-1 -right-1 flex h-2.5 w-2.5">
                          <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-sky-400 opacity-75" />
                          <span className="relative inline-flex h-2.5 w-2.5 rounded-full bg-sky-500 border border-white" />
                        </span>
                      )}
                    </div>

                    <div className="min-w-0">
                      <div className="flex items-center gap-1.5">
                        <span
                          className={`text-xs font-bold truncate ${
                            isStageActive
                              ? 'text-sky-950 font-bold'
                              : isStageCompleted
                                ? 'text-emerald-950 font-semibold'
                                : 'text-zinc-600'
                          }`}
                        >
                          {stage.label}
                        </span>
                      </div>
                      <p className="text-[10px] text-zinc-400 truncate">
                        {stage.sublabel}
                      </p>
                    </div>
                  </div>

                  {/* Stage Pill */}
                  <div className="shrink-0">
                    {isStageActive ? (
                      <span className="inline-flex items-center gap-1 rounded-full border border-sky-300 bg-sky-100 px-2 py-0.5 text-[9px] font-bold text-sky-800 shadow-sm">
                        <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-sky-600" />
                        กำลังรัน
                      </span>
                    ) : isStageCompleted ? (
                      <span className="inline-flex items-center gap-1 rounded-full border border-emerald-200 bg-emerald-100/70 px-1.5 py-0.5 text-[9px] font-semibold text-emerald-800">
                        เสร็จแล้ว
                      </span>
                    ) : (
                      <span className="inline-flex items-center rounded-full px-1.5 py-0.5 text-[9px] font-medium text-zinc-400">
                        รอเริ่ม
                      </span>
                    )}
                  </div>
                </div>
              </div>
            )
          })}
        </div>
      </div>

      {/* Completed Outputs Summary (if done) */}
      {isDone && outputs?.executive_summary && (
        <div className="rounded-2xl border border-emerald-200 bg-gradient-to-br from-emerald-50/80 via-white to-cyan-50/50 p-3.5 shadow-sm animate-notice-in">
          <div className="flex items-center gap-2 mb-2">
            <span className="flex h-6 w-6 items-center justify-center rounded-lg bg-emerald-500 text-white text-xs">
              🎉
            </span>
            <h4 className="text-xs font-bold text-emerald-950">
              ผลลัพธ์บทวิเคราะห์โดยสรุป
            </h4>
          </div>
          <p className="text-[11px] leading-relaxed text-zinc-700 line-clamp-4 bg-white/70 p-2.5 rounded-xl border border-emerald-100">
            {outputs.executive_summary}
          </p>

          <div className="mt-2.5 flex items-center justify-end gap-2">
            <a
              href={isEquity ? '/equity' : '/macro'}
              className="rounded-lg border border-emerald-300 bg-emerald-600 px-2.5 py-1 text-[10px] font-bold text-white shadow-sm hover:bg-emerald-700 transition-colors"
            >
              เปิดดูรายงานเต็ม →
            </a>
          </div>
        </div>
      )}

      {/* Error Banner */}
      {errorMessage && (
        <div className="rounded-2xl border border-rose-200 bg-rose-50/90 p-3 text-xs text-rose-800 shadow-sm animate-notice-in">
          <div className="flex items-center gap-2 font-bold mb-1">
            <span>⚠️</span>
            <span>ข้อผิดพลาดระหว่างประมวลผล</span>
          </div>
          <p className="text-[11px] leading-relaxed text-rose-700">{errorMessage}</p>
        </div>
      )}

      {/* Live Agent Thought Stream & Logs */}
      <div className="rounded-2xl border border-sky-100 bg-white/90 p-3.5 shadow-sm">
        <div className="flex items-center justify-between mb-2.5 px-0.5">
          <div className="flex items-center gap-1.5">
            <span className="h-2 w-2 rounded-full bg-emerald-500 animate-pulse" />
            <h4 className="text-[10px] font-bold uppercase tracking-wider text-zinc-500">
              ความเคลื่อนไหวสด (Live Thought Stream)
            </h4>
          </div>
          <span className="text-[10px] font-mono text-zinc-400">
            {lines.length} กิจกรรม
          </span>
        </div>

        {/* Scrollable Log Terminal Container */}
        <div
          ref={logContainerRef}
          className="max-h-64 overflow-y-auto space-y-2 custom-scrollbar p-1"
        >
          {groupedSteps.length === 0 && !isDone && (
            <div className="py-6 text-center text-xs text-zinc-400 flex flex-col items-center gap-2">
              <span className="h-4 w-4 animate-spin rounded-full border-2 border-sky-400 border-t-transparent" />
              <span>กำลังเตรียมเชื่อมต่อ Agent และเริ่มต้นการประมวลผล...</span>
            </div>
          )}

          {groupedSteps.map((step) => (
            <div
              key={step.key}
              className="rounded-xl border border-sky-100/90 bg-sky-50/40 p-2 text-xs animate-card-in"
            >
              <div className="flex items-center justify-between mb-1 pb-1 border-b border-sky-100/60">
                <span className="font-bold text-sky-950 flex items-center gap-1">
                  <span>🤖</span>
                  <span>{nodeDisplayName(step.node)}</span>
                </span>
                <span className="font-mono text-[9px] text-zinc-400">
                  {step.node ?? 'system'}
                </span>
              </div>

              <div className="space-y-1">
                {step.messages.map((m, j) => (
                  <div
                    key={j}
                    className={`text-[11px] leading-relaxed ${
                      m.role === 'instruction'
                        ? 'text-zinc-600 font-mono italic bg-white/60 p-1.5 rounded-lg border border-sky-100/40'
                        : 'text-sky-950 font-normal'
                    }`}
                  >
                    {m.role === 'instruction' && (
                      <span className="text-sky-600 font-semibold not-italic mr-1">
                        ↳ [คำสั่ง]:
                      </span>
                    )}
                    <span>{m.content}</span>
                  </div>
                ))}
              </div>
            </div>
          ))}

          {/* Typing / Heartbeat Indicator when streaming */}
          {!isDone && !isError && (
            <div className="flex items-center gap-2 py-2 px-1 text-[11px] text-sky-700 font-medium">
              <span className="flex gap-1">
                <span className="h-1.5 w-1.5 rounded-full bg-sky-500 animate-bounce [animation-delay:-0.3s]" />
                <span className="h-1.5 w-1.5 rounded-full bg-sky-500 animate-bounce [animation-delay:-0.15s]" />
                <span className="h-1.5 w-1.5 rounded-full bg-sky-500 animate-bounce" />
              </span>
              <span>AI กำลังวิเคราะห์และคำนวณข้อมูล...</span>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
