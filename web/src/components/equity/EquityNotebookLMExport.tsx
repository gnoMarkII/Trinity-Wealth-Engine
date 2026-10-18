import React, { useEffect, useState, useCallback, useRef } from 'react'
import { api, ApiError } from '../../api/client'
import type { EquityNotebookLMExportStatusDTO } from '../../api/types'

export interface EquityNotebookLMExportProps {
  ticker: string
  market?: string
  onNotify?: (message: string, type: 'info' | 'success' | 'error') => void
}

const IN_PROGRESS_STATES = ['queued', 'preparing', 'uploading', 'verifying', 'running']
const COMPLETED_STATES = ['ready', 'ready_with_warnings', 'completed']
const FAILED_STATES = ['failed', 'partial', 'blocked']

const DOCUMENT_ITEMS = [
  { prefix: '00', label: 'คู่มือแนวทางการค้นคว้า & Prompt ตัวอย่าง (Research Guide)' },
  { prefix: '01', label: 'บทวิเคราะห์ปัจจัยพื้นฐาน, 4-Pillar Scorecard & Reverse DCF' },
  { prefix: '02', label: 'ประวัติบทวิเคราะห์ย้อนหลัง & Thesis Evolution' },
  { prefix: '03', label: 'งบการเงิน, อัตราส่วนทางการเงิน & SEC Filings (10-K/10-Q)' },
  { prefix: '04', label: 'ธุรกรรมผู้บริหาร SEC Form 4 & โครงสร้างผู้ถือหุ้น (Insider & Ownership)' },
  { prefix: '05', label: 'สรุปการประชุมผลประกอบการ (Earnings Calls)' },
  { prefix: '06', label: 'ข่าวสารบริษัทที่คัดกรองแล้ว (Curated News)' },
  { prefix: '07', label: 'ตลาดอนุพันธ์ Cboe Options, Max Pain & Analyst Consensus' },
  { prefix: '08', label: 'ข้อมูลราคาตลาด OHLCV & บันทึกนักลงทุน (Price & User Notes)' },
  { prefix: '09', label: 'ภาคผนวกข้อมูลโครงสร้างเชิงลึก (Structured JSON Appendix)' },
]


export const EquityNotebookLMExport: React.FC<EquityNotebookLMExportProps> = ({
  ticker,
  market = 'US',
  onNotify,
}) => {
  const [exportData, setExportData] = useState<EquityNotebookLMExportStatusDTO | null>(null)
  const [loading, setLoading] = useState(false)
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [showDetails, setShowDetails] = useState(false)
  const popoverRef = useRef<HTMLDivElement>(null)

  // Polling ref to prevent concurrent poll loops
  const pollTimerRef = useRef<number | null>(null)

  const stopPolling = () => {
    if (pollTimerRef.current !== null) {
      window.clearTimeout(pollTimerRef.current)
      pollTimerRef.current = null
    }
  }

  const pollExport = useCallback(
    (exportId: string) => {
      stopPolling()
      pollTimerRef.current = window.setTimeout(async () => {
        try {
          const updated = await api.getEquityNotebookLMExport(exportId)
          setExportData(updated)
          if (IN_PROGRESS_STATES.includes(updated.state)) {
            pollExport(exportId)
          } else if (COMPLETED_STATES.includes(updated.state)) {
            onNotify?.(
              `ส่งข้อมูล ${ticker} ไปยัง NotebookLM สำเร็จ พร้อมค้นคว้าแล้ว`,
              'success'
            )
          } else if (FAILED_STATES.includes(updated.state)) {
            onNotify?.(
              `การส่งข้อมูล ${ticker} ไปยัง NotebookLM: ${updated.error || updated.state}`,
              'error'
            )
          }
        } catch (err) {
          console.error(`Failed to poll equity export for ${ticker}:`, err)
        }
      }, 2500)
    },
    [onNotify, ticker]
  )

  // Fetch latest export on mount or when ticker/market changes
  useEffect(() => {
    let isMounted = true
    stopPolling()

    if (!ticker) {
      setExportData(null)
      return
    }

    const checkLatest = async () => {
      setLoading(true)
      try {
        if (!api.getLatestEquityNotebookLMExport) {
          if (isMounted) setLoading(false)
          return
        }
        const latest = await api.getLatestEquityNotebookLMExport(ticker, market)
        if (!isMounted) return
        setExportData(latest)
        if (latest && IN_PROGRESS_STATES.includes(latest.state)) {
          pollExport(latest.export_id)
        }
      } catch (err: any) {
        if (isMounted) {
          const status =
            err?.status ??
            (typeof ApiError !== 'undefined' && err instanceof ApiError ? err.status : undefined)
          if (status === 404) {
            setExportData(null)
          } else {
            console.warn(`Failed to fetch latest equity export for ${ticker}:`, err)
          }
        }
      } finally {
        if (isMounted) setLoading(false)
      }
    }

    checkLatest()
    return () => {
      isMounted = false
      stopPolling()
    }
  }, [ticker, market, pollExport])

  // Close popover when clicking outside
  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (popoverRef.current && !popoverRef.current.contains(e.target as Node)) {
        setShowDetails(false)
      }
    }
    if (showDetails) {
      document.addEventListener('mousedown', handleClickOutside)
    }
    return () => {
      document.removeEventListener('mousedown', handleClickOutside)
    }
  }, [showDetails])

  const handleExport = async () => {
    if (isSubmitting || !ticker) return
    setIsSubmitting(true)
    try {
      const resp = await api.exportEquityToNotebookLM(ticker, {
        market,
        mode: 'all_retained',
      })
      onNotify?.(
        `เริ่มกระบวนการจัดเตรียมและอัปโหลดข้อมูล ${ticker} ไปยัง NotebookLM`,
        'info'
      )
      pollExport(resp.export_id)
      const current = await api.getEquityNotebookLMExport(resp.export_id)
      setExportData(current)
    } catch (err: any) {
      console.error(`Failed to trigger equity export for ${ticker}:`, err)
      const status =
        err?.status ??
        (typeof ApiError !== 'undefined' && err instanceof ApiError ? err.status : undefined)
      const errorMsg =
        status === 409
          ? `ไม่สามารถส่งออกได้: จำเป็นต้องมีบทวิเคราะห์พื้นฐานของ ${ticker} ก่อน`
          : err?.message || 'ไม่สามารถส่งข้อมูลไปยัง NotebookLM ได้'
      onNotify?.(errorMsg, 'error')
    } finally {
      setIsSubmitting(false)
    }
  }

  const handleRetry = async () => {
    if (!exportData || isSubmitting) return
    setIsSubmitting(true)
    try {
      const resp = await api.retryEquityNotebookLMExport(exportData.export_id)
      onNotify?.(`สั่งลองส่งข้อมูล ${ticker} ใหม่อีกครั้งแล้ว`, 'info')
      pollExport(resp.export_id)
      const current = await api.getEquityNotebookLMExport(resp.export_id)
      setExportData(current)
    } catch (err: any) {
      console.error(`Failed to retry equity export for ${ticker}:`, err)
      const msg = err?.message || 'ไม่สามารถลองส่งข้อมูลใหม่ได้'
      onNotify?.(msg, 'error')
    } finally {
      setIsSubmitting(false)
    }
  }

  const isCompleted = Boolean(exportData && COMPLETED_STATES.includes(exportData.state))
  const isFailed = Boolean(exportData && FAILED_STATES.includes(exportData.state))
  const isBlocked = exportData?.state === 'blocked'
  const isInProgress =
    (exportData && IN_PROGRESS_STATES.includes(exportData.state)) || isSubmitting
  const primaryNotebook =
    exportData?.notebooks && exportData.notebooks.length > 0 ? exportData.notebooks[0] : null
  const okSourcesCount =
    exportData?.source_results?.filter((s) => ['success', 'ready', 'ok'].includes(s.status))
      .length ?? 0
  const totalSourcesCount = exportData?.source_results?.length || 10

  return (
    <div className="relative inline-flex items-center gap-2" ref={popoverRef}>
      {/* 1. If completed, show "เปิด NotebookLM" link + status badge */}
      {isCompleted && primaryNotebook && (
        <div className="flex items-center gap-1.5">
          <a
            href={primaryNotebook.url}
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-1.5 rounded-xl bg-gradient-to-r from-purple-600 via-indigo-600 to-sky-600 px-3.5 py-1.5 text-xs font-semibold text-white shadow-xs transition-all hover:opacity-95 hover:shadow focus:outline-hidden focus:ring-2 focus:ring-indigo-400"
            title="เปิดสมุดค้นคว้าบน Google NotebookLM ในแท็บใหม่"
          >
            <span>📓</span>
            <span>เปิด NotebookLM เพื่อค้นคว้า</span>
            <span className="text-[10px] opacity-80">↗</span>
          </a>

          {/* Details Dropdown trigger */}
          <button
            type="button"
            onClick={() => setShowDetails(!showDetails)}
            className="flex items-center gap-1 rounded-xl border border-indigo-200 bg-indigo-50/60 px-2.5 py-1.5 text-xs font-medium text-indigo-800 transition-all hover:bg-indigo-100/70"
            title="ดูรายละเอียดชุดข้อมูลที่ส่งออกและตัวเลือกอัปเดต"
            aria-label="ดูรายละเอียดชุดข้อมูลที่ส่งออกและตัวเลือกอัปเดต"
            aria-expanded={showDetails}
          >
            <span className="inline-block w-1.5 h-1.5 rounded-full bg-emerald-500" />
            <span className="hidden sm:inline">
              {okSourcesCount}/{totalSourcesCount} แหล่ง
            </span>
            <span className="text-[10px] text-indigo-500">▼</span>
          </button>
        </div>
      )}

      {/* 2. If in progress, show spinner and progress */}
      {isInProgress && (
        <div className="flex items-center gap-2 rounded-xl border border-indigo-200 bg-indigo-50/70 px-3.5 py-1.5 text-xs font-medium text-indigo-800 animate-pulse shadow-xs">
          <span className="animate-spin text-sm">⏳</span>
          <span>
            กำลังส่งออก {ticker} ({okSourcesCount}/{totalSourcesCount})...
          </span>
        </div>
      )}

      {/* 3. If failed or blocked, show error badge with retry button if allowed */}
      {isFailed && !isInProgress && (
        <div className="flex items-center gap-1.5">
          <button
            type="button"
            onClick={() => setShowDetails(!showDetails)}
            className="flex items-center gap-1 rounded-xl border border-rose-200 bg-rose-50 px-2.5 py-1.5 text-xs font-medium text-rose-700 cursor-pointer hover:bg-rose-100/80 transition-colors"
            title={exportData?.error || (isBlocked ? 'ต้องมีบทวิเคราะห์ก่อน' : 'การส่งออกขัดข้อง')}
          >
            <span>⚠️</span>
            <span>{isBlocked ? 'ข้อมูลไม่เพียงพอ' : 'ส่งออกไม่สำเร็จ'}</span>
            <span className="text-[10px] text-rose-500">▼</span>
          </button>
          {exportData?.can_retry && (
            <button
              type="button"
              onClick={handleRetry}
              disabled={isSubmitting}
              className="flex items-center gap-1 rounded-xl border border-rose-300 bg-white px-2.5 py-1.5 text-xs font-semibold text-rose-700 shadow-xs hover:bg-rose-50 disabled:opacity-50"
              title="ลองส่งข้อมูลใหม่อีกครั้ง"
            >
              <span>🔄</span>
              <span>ลองใหม่</span>
            </button>
          )}
        </div>
      )}

      {/* 4. If no export data exists yet, or initial state, show initial export button */}
      {(!exportData || (!isCompleted && !isFailed && !isInProgress)) &&
        !loading &&
        !isInProgress && (
          <button
            type="button"
            onClick={handleExport}
            disabled={isSubmitting}
            className="flex items-center gap-1.5 rounded-xl border border-indigo-200 bg-gradient-to-r from-white via-indigo-50/40 to-purple-50/50 px-3.5 py-1.5 text-xs font-semibold text-indigo-800 shadow-xs transition-all hover:border-indigo-300 hover:bg-indigo-50/80 hover:shadow disabled:opacity-50"
            title="รวบรวมข้อมูล Equity ทุกมิติส่งไปยัง Google NotebookLM เพื่อค้นคว้า"
          >
            <span>📓</span>
            <span>ส่งข้อมูล Equity ไป NotebookLM</span>
          </button>
        )}

      {/* Popover Disclosure: Source Coverage & Re-export */}
      {showDetails && exportData && (
        <div className="absolute right-0 top-full mt-2 w-80 sm:w-96 rounded-2xl border border-indigo-100 bg-white/95 p-4 shadow-xl backdrop-blur-md z-50 animate-in fade-in zoom-in-95">
          <div className="flex items-center justify-between pb-3 border-b border-indigo-50">
            <div className="flex items-center gap-1.5">
              <span className="text-base">🔬</span>
              <span className="font-bold text-xs text-zinc-900">
                NotebookLM Research Bundle: {ticker} ({market})
              </span>
            </div>
            <span
              className={`rounded-full px-2 py-0.5 text-[10px] font-semibold ${
                isCompleted
                  ? 'bg-emerald-50 text-emerald-700 border border-emerald-200'
                  : isFailed
                  ? 'bg-rose-50 text-rose-700 border border-rose-200'
                  : 'bg-indigo-50 text-indigo-700 border border-indigo-200'
              }`}
            >
              {isCompleted ? 'พร้อมใช้งาน' : isBlocked ? 'ถูกระงับ' : isFailed ? 'ล้มเหลว' : 'กำลังประมวลผล'}
            </span>
          </div>

          <div className="py-2.5 space-y-2 text-[11px] text-zinc-600">
            <div className="flex justify-between items-center text-[10px] text-zinc-500">
              <span>รายงานประเมินเมื่อ:</span>
              <span className="font-mono text-zinc-700">
                {exportData.snapshot_at
                  ? new Date(exportData.snapshot_at).toLocaleString('th-TH')
                  : 'N/A'}
              </span>
            </div>
            <div className="flex justify-between items-center text-[10px] text-zinc-500">
              <span>Snapshot Hash:</span>
              <span className="font-mono text-zinc-700" title={exportData.bundle_hash}>
                {exportData.bundle_hash ? `${exportData.bundle_hash.slice(0, 14)}...` : 'N/A'}
              </span>
            </div>

            {/* Warnings list if any */}
            {exportData.warnings && exportData.warnings.length > 0 && (
              <div className="rounded-xl bg-amber-50/80 p-2 border border-amber-200/70 text-[10px] text-amber-900 space-y-1">
                <div className="font-semibold flex items-center gap-1">
                  <span>⚠️</span>
                  <span>ข้อสังเกตความสมบูรณ์ของข้อมูล:</span>
                </div>
                <ul className="list-disc list-inside space-y-0.5 pl-1 text-[9.5px] text-amber-800">
                  {exportData.warnings.map((w, idx) => (
                    <li key={idx}>{w}</li>
                  ))}
                </ul>
              </div>
            )}

            {/* Error explanation if failed */}
            {exportData.error && (
              <div className="rounded-xl bg-rose-50/80 p-2 border border-rose-200 text-[10px] text-rose-800">
                <span className="font-semibold block mb-0.5">ข้อผิดพลาด:</span>
                <p className="text-[9.5px]">{exportData.error}</p>
              </div>
            )}

            {/* Sources Checklist */}
            <div className="mt-2 space-y-1 rounded-xl bg-slate-50/80 p-2.5 border border-slate-100">
              <span className="font-semibold text-zinc-700 block mb-1">
                เอกสารในชุดข้อมูล ({okSourcesCount}/{totalSourcesCount}):
              </span>
              <div className="grid grid-cols-1 gap-1 text-[10px] max-h-48 overflow-y-auto pr-1">
                {DOCUMENT_ITEMS.map((doc) => {
                  const matchingResult = exportData.source_results?.find((s) =>
                    s.file_name.startsWith(doc.prefix)
                  )
                  const isOk = matchingResult
                    ? ['success', 'ready', 'ok'].includes(matchingResult.status)
                    : isCompleted

                  return (
                    <div key={doc.prefix} className="flex items-center gap-1.5 justify-between">
                      <div className="flex items-center gap-1.5 truncate">
                        <span className={isOk ? 'text-emerald-500 font-bold' : 'text-zinc-400'}>
                          {isOk ? '✓' : '•'}
                        </span>
                        <span className="truncate" title={doc.label}>
                          {doc.prefix}: {doc.label}
                        </span>
                      </div>
                      {matchingResult && !isOk && (
                        <span className="text-[9px] text-zinc-400 shrink-0">
                          {matchingResult.status}
                        </span>
                      )}
                    </div>
                  )
                })}
              </div>
            </div>

            <div className="text-[10px] text-zinc-400 italic pt-1">
              * Research Companion สำหรับค้นคว้าเท่านั้น — ไม่มีการสร้าง Podcast, Discord หรือเขียนทับ conviction
            </div>
          </div>

          {/* Action Row inside popover */}
          <div className="pt-2 border-t border-indigo-50 flex items-center justify-between gap-2">
            <button
              type="button"
              onClick={() => {
                setShowDetails(false)
                handleExport()
              }}
              disabled={isSubmitting}
              className="rounded-lg border border-indigo-200 bg-white px-2.5 py-1 text-[11px] font-semibold text-indigo-700 hover:bg-indigo-50 transition-colors disabled:opacity-50"
              title="สร้าง Snapshot ใหม่และส่งออกอีกครั้ง"
            >
              🔄 ส่งออกชุดข้อมูลใหม่
            </button>

            {primaryNotebook?.url && (
              <a
                href={primaryNotebook.url}
                target="_blank"
                rel="noopener noreferrer"
                className="rounded-lg bg-indigo-600 px-3 py-1 text-[11px] font-semibold text-white hover:bg-indigo-700 transition-colors"
              >
                เปิดในเบราว์เซอร์ ↗
              </a>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
