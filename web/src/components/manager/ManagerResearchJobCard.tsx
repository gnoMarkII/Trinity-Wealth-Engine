import { useEffect, useRef, useState } from 'react'
import { api } from '../../api/client'
import type { JobOutputsDTO, SpecialistOutputDTO } from '../../api/types'
import {
  determineStageIndex,
  getPipelineFor,
} from '../../lib/agentPipelines'
import { nodeDisplayName } from '../../lib/nodeDisplayNames'
import AdvisorMarkdown from './AdvisorMarkdown'

export interface ManagerResearchJobCardProps {
  jobId: string
  ticker: string
  jobType?: string
  status?: 'queued' | 'running' | 'completed' | 'failed'
  onOpenKanban?: () => void
  onSelectJob?: (jobId: string) => void
}

export default function ManagerResearchJobCard({
  jobId,
  ticker,
  jobType = 'equity_intel',
  status = 'queued',
  onSelectJob,
  onOpenKanban,
}: ManagerResearchJobCardProps) {
  const [liveStatus, setLiveStatus] = useState<'queued' | 'running' | 'completed' | 'failed'>(status)
  const [currentNode, setCurrentNode] = useState<string | null>(null)
  const [lastLogMessage, setLastLogMessage] = useState<string>('')
  const [elapsedSeconds, setElapsedSeconds] = useState<number>(0)
  const [outputs, setOutputs] = useState<JobOutputsDTO | null>(null)
  const [selectedSpecialist, setSelectedSpecialist] = useState<SpecialistOutputDTO | null>(null)
  const [copied, setCopied] = useState<boolean>(false)
  const [isExpanded, setIsExpanded] = useState<boolean>(true)

  const isMacro = jobType?.includes('macro') || (!ticker && !jobType?.includes('equity'))
  const isEquity = !isMacro
  const pipeline = getPipelineFor(jobType, ticker)
  const activeStageIndex = determineStageIndex(pipeline, currentNode)
  const activeStage = pipeline[activeStageIndex] ?? pipeline[0]!

  const timerRef = useRef<number | null>(null)
  const eventSourceRef = useRef<EventSource | null>(null)

  // 1. Initial status check and output fetch if already completed
  useEffect(() => {
    let cancelled = false
    if (!jobId || jobId.startsWith('sim_')) return

    api
      .getJobStatus(jobId)
      .then((dto) => {
        if (cancelled) return
        if (dto.current_node) setCurrentNode(dto.current_node)
        if (dto.status === 'done') {
          setLiveStatus('completed')
          api.getJobOutputs(jobId).then((out) => {
            if (!cancelled) setOutputs(out)
          }).catch(() => {})
        } else if (dto.status === 'error') {
          setLiveStatus('failed')
        } else if (dto.status === 'running') {
          setLiveStatus('running')
        } else {
          setLiveStatus('queued')
        }
      })
      .catch(() => {})

    return () => {
      cancelled = true
    }
  }, [jobId])

  // 2. Elapsed Timer while running or queued
  useEffect(() => {
    const isWorking = liveStatus === 'running' || liveStatus === 'queued'
    if (isWorking) {
      timerRef.current = window.setInterval(() => {
        setElapsedSeconds((prev) => prev + 1)
      }, 1000)
    } else {
      if (timerRef.current) {
        window.clearInterval(timerRef.current)
        timerRef.current = null
      }
    }
    return () => {
      if (timerRef.current) window.clearInterval(timerRef.current)
    }
  }, [liveStatus])

  // 3. Connect to SSE Stream while running/queued for live Claude-style stage updates
  useEffect(() => {
    if (!jobId || jobId.startsWith('sim_')) return
    if (liveStatus === 'completed' || liveStatus === 'failed') return
    if (typeof EventSource === 'undefined') return

    const source = new EventSource(`/api/agents/stream/${jobId}`)
    eventSourceRef.current = source

    source.onmessage = (e) => {
      try {
        const payload = JSON.parse(e.data)
        if (payload.node) setCurrentNode(payload.node)
        if (payload.message) {
          // Keep clean agent thought/action description
          const msg = String(payload.message).trim()
          if (msg) setLastLogMessage(msg)
        }
        setLiveStatus('running')
      } catch {
        // Skip malformed log chunk
      }
    }

    source.addEventListener('done', () => {
      setLiveStatus('completed')
      setCurrentNode(null)
      source.close()
      api.getJobOutputs(jobId).then(setOutputs).catch(() => {})
    })

    source.addEventListener('done_with_warnings', () => {
      setLiveStatus('completed')
      setCurrentNode(null)
      source.close()
      api.getJobOutputs(jobId).then(setOutputs).catch(() => {})
    })

    source.addEventListener('done_with_errors', () => {
      setLiveStatus('failed')
      setCurrentNode(null)
      source.close()
    })

    source.onerror = () => {
      // In case of network disconnect, fallback to polling
      api.getJobStatus(jobId).then((dto) => {
        if (dto.status === 'done') {
          setLiveStatus('completed')
          source.close()
          api.getJobOutputs(jobId).then(setOutputs).catch(() => {})
        } else if (dto.status === 'error') {
          setLiveStatus('failed')
          source.close()
        }
      }).catch(() => {})
    }

    return () => {
      source.close()
      eventSourceRef.current = null
    }
  }, [jobId, liveStatus])

  // Progress percentage calculation
  const progressPercent =
    liveStatus === 'completed'
      ? 100
      : liveStatus === 'failed'
        ? 100
        : Math.min(95, Math.max(15, Math.round(((activeStageIndex + 1) / pipeline.length) * 100)))

  function handleCopyReport() {
    const textToCopy =
      outputs?.executive_summary ||
      `Job ID: ${jobId}\nสถานะ: ${liveStatus}\nผลการวิเคราะห์ ${ticker || 'Macro Strategy'}`
    navigator.clipboard.writeText(textToCopy).then(() => {
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    })
  }

  // =========================================================================
  // RENDER: Running / Queued State (Compact & Minimal Live Working Indicator)
  // =========================================================================
  if (liveStatus === 'running' || liveStatus === 'queued') {
    return (
      <div className="my-2.5 overflow-hidden rounded-xl border border-sky-200/90 bg-gradient-to-r from-sky-50/80 via-white to-sky-50/50 p-3 shadow-xs backdrop-blur-md animate-card-in">
        <div className="flex items-center justify-between gap-3">
          {/* Left: Glowing Pulse Beacon & Compact Working Status */}
          <div className="flex items-center gap-2.5 min-w-0">
            {/* Glowing animated pulse beacon */}
            <div className="relative flex h-7 w-7 shrink-0 items-center justify-center rounded-lg bg-gradient-to-tr from-sky-400 to-cyan-500 text-white shadow-xs">
              <span className="text-xs animate-pulse">⚡</span>
              <span className="absolute -top-0.5 -right-0.5 flex h-2 w-2">
                <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-sky-400 opacity-75" />
                <span className="relative inline-flex h-2 w-2 rounded-full bg-sky-500 border border-white" />
              </span>
            </div>

            <div className="min-w-0 flex-1">
              <div className="flex items-center gap-2">
                <span className="truncate text-xs font-bold text-sky-950">
                  {isMacro ? 'Trinity Macro Swarm กำลังดำเนินการวิจัย...' : `AI กำลังวิเคราะห์หุ้น ${ticker}...`}
                </span>
                <span className="shrink-0 rounded-md border border-sky-200 bg-sky-100/70 px-1.5 py-0.2 text-[9px] font-semibold text-sky-800 animate-pulse">
                  {liveStatus === 'running' ? 'กำลังประมวลผล' : 'อยู่ในคิว'}
                </span>
              </div>
              <div className="flex items-center gap-1.5 text-[11px] text-zinc-500 font-mono truncate">
                <span className="text-sky-700 font-medium truncate">
                  {lastLogMessage ? `${nodeDisplayName(currentNode)}: ${lastLogMessage}` : `ขั้นตอน: ${activeStage.label}`}
                </span>
              </div>
            </div>
          </div>

          {/* Right: Elapsed Timer */}
          <div className="flex items-center shrink-0">
            <span className="font-mono text-[11px] font-medium text-sky-800 bg-sky-100/80 rounded-lg px-2 py-0.5 flex items-center gap-1 shadow-2xs">
              <span>⏱️</span>
              <span>{elapsedSeconds}s</span>
            </span>
          </div>
        </div>

        {/* Micro Shimmering Progress Line */}
        <div className="mt-2 h-1 w-full rounded-full bg-sky-100/70 overflow-hidden">
          <div
            className="h-full rounded-full bg-gradient-to-r from-sky-400 via-cyan-400 to-sky-600 animate-shimmer transition-all duration-500"
            style={{ width: `${progressPercent}%` }}
          />
        </div>
      </div>
    )
  }

  // =========================================================================
  // RENDER: Completed State with Full Research Report (พอทำงานเสร็จถึงค่อยรายงาน)
  // =========================================================================
  return (
    <div className="my-3 rounded-2xl border border-emerald-200 bg-gradient-to-b from-emerald-50/40 via-white to-sky-50/30 p-4 shadow-sm backdrop-blur-md transition-all hover:border-emerald-300 animate-card-in">
      {/* Header Bar */}
      <div className="flex flex-wrap items-center justify-between gap-3 pb-3 border-b border-emerald-100">
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-gradient-to-tr from-emerald-500 to-teal-400 text-white shadow-sm">
            <span className="text-base">{isEquity ? '🔬' : '🌐'}</span>
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="text-xs font-bold text-sky-950">
                {isMacro
                  ? 'รายงานผลการวิเคราะห์สภาวะเศรษฐกิจมหภาค (Macro Intel)'
                  : `รายงานผลการวิเคราะห์หุ้น ${ticker} (ฉบับสมบูรณ์)`}
              </span>
              <span className="rounded-md border border-emerald-200 bg-emerald-100/80 px-2 py-0.5 text-[10px] font-bold text-emerald-800">
                ✅ วิจัยเสร็จสมบูรณ์
              </span>
            </div>
            <div className="mt-0.5 flex items-center gap-2 text-[11px] text-zinc-500 font-mono">
              <span>ID: {jobId}</span>
              {elapsedSeconds > 0 && <span>• เสร็จใน {elapsedSeconds}s</span>}
            </div>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={handleCopyReport}
            className="flex items-center gap-1 rounded-lg border border-emerald-200 bg-white px-2.5 py-1 text-xs font-medium text-emerald-800 shadow-sm hover:bg-emerald-50 transition-colors"
          >
            <span>{copied ? '✓ คัดลอกแล้ว' : '📋 คัดลอก'}</span>
          </button>
          <button
            type="button"
            onClick={() => setIsExpanded((prev) => !prev)}
            className="rounded-lg border border-sky-100 bg-white px-2 py-1 text-xs font-medium text-zinc-600 hover:text-sky-900 transition-colors"
          >
            {isExpanded ? 'ย่อเนื้อหา ▴' : 'ขยายอ่านเต็ม ▾'}
          </button>
          {onSelectJob && (
            <button
              type="button"
              onClick={() => onSelectJob(jobId)}
              className="flex items-center gap-1 rounded-lg border border-sky-300 bg-sky-50 px-2.5 py-1 text-xs font-semibold text-sky-800 shadow-sm hover:bg-sky-100 transition-colors"
            >
              <span>ดู Terminal</span>
              <span>⚡</span>
            </button>
          )}
          {onOpenKanban && (
            <button
              type="button"
              onClick={onOpenKanban}
              className="rounded-lg border border-sky-200 bg-white px-2.5 py-1 text-xs font-medium text-sky-700 shadow-sm hover:bg-sky-50 transition-colors"
            >
              ดูในการ์ดบอร์ดงาน →
            </button>
          )}
        </div>
      </div>

      {/* Report Body (Expanded View) */}
      {isExpanded && (
        <div className="mt-3 space-y-3 animate-card-in">
          {outputs?.executive_summary ? (
            <div className="rounded-xl border border-sky-100 bg-white/95 p-4 shadow-xs">
              <div className="flex items-center justify-between mb-2 pb-1.5 border-b border-sky-50">
                <span className="text-[11px] font-bold uppercase tracking-wider text-sky-900 flex items-center gap-1.5">
                  <span>📑</span>
                  <span>บทสรุปสำหรับผู้บริหาร (Executive Summary)</span>
                </span>
                <span className="text-[10px] text-zinc-400 font-mono">
                  อัปเดตสดจากระบบ
                </span>
              </div>
              <AdvisorMarkdown content={outputs.executive_summary} />
            </div>
          ) : (
            <div className="rounded-xl border border-sky-100 bg-white/90 p-4 text-center">
              <span className="text-xs text-zinc-500">
                ประมวลผลข้อมูลเสร็จเรียบร้อยแล้ว รายงานถูกบันทึกลง Knowledge Vault และบอร์ดงานแล้วครับ
              </span>
            </div>
          )}

          {/* Specialist Output Breakdown Pills (if available) */}
          {outputs?.specialists && outputs.specialists.length > 0 && (
            <div className="rounded-xl border border-sky-100 bg-sky-50/50 p-3">
              <div className="text-[11px] font-bold text-sky-950 mb-2 flex items-center gap-1">
                <span>🔍</span>
                <span>มุมมองเจาะลึกราย Specialist (คลิกเพื่อดูรายละเอียด):</span>
              </div>
              <div className="flex flex-wrap gap-1.5">
                {outputs.specialists.map((sp) => {
                  const isSelected = selectedSpecialist?.node_name === sp.node_name
                  return (
                    <button
                      key={sp.node_name}
                      type="button"
                      onClick={() => setSelectedSpecialist(isSelected ? null : sp)}
                      className={`flex items-center gap-1 rounded-lg border px-2.5 py-1 text-xs transition-all ${
                        isSelected
                          ? 'border-sky-300 bg-sky-500 text-white font-semibold shadow-sm'
                          : 'border-sky-100 bg-white text-sky-900 hover:bg-sky-50 font-medium'
                      }`}
                    >
                      <span>{sp.label || nodeDisplayName(sp.node_name)}</span>
                    </button>
                  )
                })}
              </div>

              {/* Selected Specialist Drawer Content */}
              {selectedSpecialist && (
                <div className="mt-2.5 rounded-xl border border-sky-200 bg-white p-3 text-xs shadow-xs animate-card-in">
                  <div className="flex items-center justify-between mb-1.5 font-bold text-sky-950">
                    <span>{selectedSpecialist.label || nodeDisplayName(selectedSpecialist.node_name)}</span>
                  </div>
                  <AdvisorMarkdown content={selectedSpecialist.content} />
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  )
}
