import React, { useEffect, useState, useCallback, useRef } from 'react'
import { api, ApiError } from '../../../api/client'
import type { MacroNotebookLMExportStatusDTO } from '../../../api/types'
import {
  AlertTriangleIcon,
  BookOpenIcon,
  CheckSmallIcon,
  ChevronDownIcon,
  ExternalLinkIcon,
  RefreshIcon,
  SparklesIcon,
} from '../../ui/AppIcons'

interface MacroNotebookLMExportProps {
  onNotify?: (message: string, type: 'info' | 'success' | 'error') => void
}

const IN_PROGRESS_STATES = ['queued', 'preparing', 'uploading', 'verifying', 'running']
const COMPLETED_STATES = ['ready', 'ready_with_warnings', 'completed']
const FAILED_STATES = ['failed', 'partial', 'blocked']

export const MacroNotebookLMExport: React.FC<MacroNotebookLMExportProps> = ({ onNotify }) => {
  const [exportData, setExportData] = useState<MacroNotebookLMExportStatusDTO | null>(null)
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

  const pollExport = useCallback((exportId: string) => {
    stopPolling()
    pollTimerRef.current = window.setTimeout(async () => {
      try {
        const updated = await api.getMacroNotebookLMExport(exportId)
        setExportData(updated)
        if (IN_PROGRESS_STATES.includes(updated.state)) {
          pollExport(exportId)
        } else if (COMPLETED_STATES.includes(updated.state)) {
          onNotify?.('ส่งข้อมูล Macro ไปยัง NotebookLM สำเร็จ พร้อมค้นคว้าแล้ว', 'success')
        } else if (FAILED_STATES.includes(updated.state)) {
          onNotify?.(`การส่งข้อมูล Macro ไปยัง NotebookLM: ${updated.error || updated.state}`, 'error')
        }
      } catch (err) {
        console.error('Failed to poll macro export:', err)
      }
    }, 2500)
  }, [onNotify])

  // Fetch latest export on mount
  useEffect(() => {
    let isMounted = true
    const checkLatest = async () => {
      setLoading(true)
      try {
        const latest = await api.getLatestMacroNotebookLMExport()
        if (!isMounted) return
        setExportData(latest)
        if (latest && IN_PROGRESS_STATES.includes(latest.state)) {
          pollExport(latest.export_id)
        }
      } catch (err) {
        // 404 or null is normal if no export has been made yet
        if (err instanceof ApiError && err.status === 404) {
          setExportData(null)
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
  }, [pollExport])

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
    if (isSubmitting) return
    setIsSubmitting(true)
    try {
      const resp = await api.exportMacroToNotebookLM({ mode: 'all_retained' })
      onNotify?.('เริ่มกระบวนการจัดเตรียมและอัปโหลดข้อมูล Macro ไปยัง NotebookLM', 'info')
      pollExport(resp.export_id)
      // Instant optimistic fetch
      const current = await api.getMacroNotebookLMExport(resp.export_id)
      setExportData(current)
    } catch (err) {
      console.error('Failed to trigger macro export:', err)
      onNotify?.(
        err instanceof ApiError ? err.message : 'ไม่สามารถส่งข้อมูลไปยัง NotebookLM ได้',
        'error'
      )
    } finally {
      setIsSubmitting(false)
    }
  }

  const handleRetry = async () => {
    if (!exportData || isSubmitting) return
    setIsSubmitting(true)
    try {
      const resp = await api.retryMacroNotebookLMExport(exportData.export_id)
      onNotify?.('สั่งลองส่งข้อมูล Macro ใหม่อีกครั้งแล้ว', 'info')
      pollExport(resp.export_id)
      const current = await api.getMacroNotebookLMExport(resp.export_id)
      setExportData(current)
    } catch (err) {
      console.error('Failed to retry macro export:', err)
      onNotify?.(
        err instanceof ApiError ? err.message : 'ไม่สามารถลองส่งข้อมูลใหม่ได้',
        'error'
      )
    } finally {
      setIsSubmitting(false)
    }
  }

  const isCompleted = Boolean(exportData && COMPLETED_STATES.includes(exportData.state))
  const isFailed = Boolean(exportData && FAILED_STATES.includes(exportData.state))
  const isInProgress = (exportData && IN_PROGRESS_STATES.includes(exportData.state)) || isSubmitting
  const primaryNotebook = exportData?.notebooks && exportData.notebooks.length > 0 ? exportData.notebooks[0] : null
  const okSourcesCount = exportData?.source_results?.filter((s) => ['success', 'ready', 'ok'].includes(s.status)).length ?? 0
  const totalSourcesCount = exportData?.source_results?.length || 9

  return (
    <div className="relative inline-flex items-center gap-2" ref={popoverRef}>
      {/* 1. If completed, show "เปิด NotebookLM" link + status badge */}
      {isCompleted && primaryNotebook && (
        <div className="flex items-center gap-1.5">
          <a
            href={primaryNotebook.url}
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-1.5 rounded-lg bg-gradient-to-r from-blue-700 via-blue-600 to-sky-600 px-3 py-1.5 text-xs font-semibold text-white shadow-sm transition-all hover:opacity-95 hover:shadow focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-section-accent/45 whitespace-nowrap"
            title="เปิดสมุดค้นคว้าบน Google NotebookLM ในแท็บใหม่"
          >
            <BookOpenIcon className="h-3.5 w-3.5 shrink-0" />
            <span>เปิด NotebookLM เพื่อค้นคว้า</span>
            <ExternalLinkIcon className="h-3 w-3 opacity-85 shrink-0" />
          </a>

          {/* Details Dropdown trigger */}
          <button
            type="button"
            onClick={() => setShowDetails(!showDetails)}
            className="flex items-center gap-1.5 rounded-lg border border-sky-200 bg-sky-50/75 px-2.5 py-1.5 text-xs font-medium text-sky-900 shadow-sm transition-all hover:bg-sky-100/80 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-section-accent/45 whitespace-nowrap"
            title="ดูรายละเอียดชุดข้อมูลที่ส่งออกและตัวเลือกอัปเดต"
            aria-expanded={showDetails}
          >
            <span className="inline-block w-1.5 h-1.5 rounded-full bg-emerald-500 shrink-0" />
            <span className="hidden sm:inline">
              {okSourcesCount}/{totalSourcesCount} แหล่ง
            </span>
            <ChevronDownIcon className={`h-3 w-3 text-sky-700 transition-transform ${showDetails ? 'rotate-180' : ''}`} />
          </button>
        </div>
      )}

      {/* 2. If in progress, show spinner and progress */}
      {isInProgress && (
        <div className="flex items-center gap-2 rounded-lg border border-sky-200 bg-sky-50/80 px-3 py-1.5 text-xs font-medium text-sky-900 animate-pulse shadow-sm">
          <RefreshIcon className="h-3.5 w-3.5 animate-spin text-section-accent shrink-0" />
          <span>
            กำลังส่งออก Macro ({okSourcesCount}/{totalSourcesCount})...
          </span>
        </div>
      )}

      {/* 3. If failed, show error badge with retry button */}
      {isFailed && !isInProgress && (
        <div className="flex items-center gap-1.5">
          <button
            type="button"
            onClick={() => setShowDetails(!showDetails)}
            className="flex items-center gap-1.5 rounded-lg border border-rose-200 bg-rose-50 px-2.5 py-1.5 text-xs font-medium text-rose-700 hover:bg-rose-100/80 transition-colors"
            title={exportData?.error || 'การส่งออกขัดข้อง'}
          >
            <AlertTriangleIcon className="h-3.5 w-3.5 text-rose-600 shrink-0" />
            <span>ส่งออกไม่สำเร็จ</span>
          </button>
          <button
            type="button"
            onClick={handleRetry}
            disabled={isSubmitting}
            className="flex items-center gap-1 rounded-lg border border-rose-300 bg-white px-2.5 py-1.5 text-xs font-semibold text-rose-700 shadow-sm hover:bg-rose-50 disabled:opacity-50"
            title="ลองส่งข้อมูลใหม่อีกครั้ง"
          >
            <RefreshIcon className={`h-3.5 w-3.5 shrink-0 ${isSubmitting ? 'animate-spin' : ''}`} />
            <span>ลองใหม่</span>
          </button>
        </div>
      )}

      {/* 4. If no export data exists yet, or initial state, show initial export button */}
      {(!exportData || (!isCompleted && !isFailed && !isInProgress)) && !loading && !isInProgress && (
        <button
          type="button"
          onClick={handleExport}
          disabled={isSubmitting}
          className="flex items-center gap-1.5 rounded-lg border border-sky-200 bg-gradient-to-r from-white via-sky-50/60 to-blue-50/70 px-3 py-1.5 text-xs font-semibold text-section-accent shadow-sm transition-all hover:border-sky-300 hover:bg-sky-50 hover:shadow disabled:opacity-50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-section-accent/45"
          title="รวบรวมข้อมูล Macro ทั้งหมดส่งไปยัง Google NotebookLM เพื่อค้นคว้า (ไม่สร้างพอดแคสต์ ไม่เรียก Discord)"
        >
          <BookOpenIcon className="h-3.5 w-3.5 shrink-0" />
          <span>ส่งข้อมูล Macro ไป NotebookLM</span>
        </button>
      )}

      {/* Popover Disclosure: Source Coverage & Re-export */}
      {showDetails && exportData && (
        <div className="absolute right-0 top-full mt-2 w-80 sm:w-96 rounded-2xl border border-sky-200/80 bg-white/95 p-4 shadow-xl backdrop-blur-md z-50 animate-in fade-in zoom-in-95">
          <div className="flex items-center justify-between pb-3 border-b border-slate-100">
            <div className="flex items-center gap-1.5">
              <SparklesIcon className="h-4 w-4 text-section-accent shrink-0" />
              <span className="font-bold text-xs text-slate-900">
                NotebookLM Research Bundle
              </span>
            </div>
            <span
              className={`rounded-full px-2 py-0.5 text-[11px] font-semibold ${
                isCompleted
                  ? 'bg-emerald-50 text-emerald-700 border border-emerald-200'
                  : 'bg-rose-50 text-rose-700 border border-rose-200'
              }`}
            >
              {isCompleted ? 'พร้อมใช้งาน' : isFailed ? 'ล้มเหลว' : 'กำลังประมวลผล'}
            </span>
          </div>

          <div className="py-2.5 space-y-2 text-[11px] text-slate-600">
            <div className="flex justify-between items-center text-[11px] text-slate-500">
              <span>รายงานประเมินเมื่อ:</span>
              <span className="font-mono text-slate-700">
                {exportData.snapshot_at
                  ? new Date(exportData.snapshot_at).toLocaleString('th-TH')
                  : 'N/A'}
              </span>
            </div>
            <div className="flex justify-between items-center text-[11px] text-slate-500">
              <span>Snapshot Hash:</span>
              <span className="font-mono text-slate-700" title={exportData.bundle_hash}>
                {exportData.bundle_hash ? `${exportData.bundle_hash.slice(0, 14)}...` : 'N/A'}
              </span>
            </div>

            {/* Sources Checklist */}
            <div className="mt-2 space-y-1 rounded-xl bg-slate-50/80 p-2.5 border border-slate-100">
              <span className="font-semibold text-slate-700 block mb-1">
                เอกสารที่ส่งออก ({okSourcesCount}/{totalSourcesCount}):
              </span>
              <div className="grid grid-cols-1 gap-1 text-[11px]">
                <div className="flex items-center gap-1.5">
                  <CheckSmallIcon className="h-3 w-3 text-emerald-600 shrink-0" />
                  <span>00: คู่มือแนวทางการค้นคว้า & Prompt ตัวอย่าง</span>
                </div>
                <div className="flex items-center gap-1.5">
                  <CheckSmallIcon className="h-3 w-3 text-emerald-600 shrink-0" />
                  <span>01: รายงานภาพรวมเศรษฐกิจ Macro ปัจจุบัน</span>
                </div>
                <div className="flex items-center gap-1.5">
                  <CheckSmallIcon className="h-3 w-3 text-emerald-600 shrink-0" />
                  <span>02: ประวัติรายงานเศรษฐกิจย้อนหลัง</span>
                </div>
                <div className="flex items-center gap-1.5">
                  <CheckSmallIcon className="h-3 w-3 text-emerald-600 shrink-0" />
                  <span>03: ตลาดสหรัฐฯ & สภาพคล่องทางการเงิน</span>
                </div>
                <div className="flex items-center gap-1.5">
                  <CheckSmallIcon className="h-3 w-3 text-emerald-600 shrink-0" />
                  <span>04: ข้อมูลเศรษฐกิจ & หนี้สาธารณะไทย</span>
                </div>
                <div className="flex items-center gap-1.5">
                  <CheckSmallIcon className="h-3 w-3 text-emerald-600 shrink-0" />
                  <span>05: ตัวชี้วัดตลาดโลก & สินค้าโภคภัณฑ์</span>
                </div>
                <div className="flex items-center gap-1.5">
                  <CheckSmallIcon className="h-3 w-3 text-emerald-600 shrink-0" />
                  <span>06: การหมุนเวียนกลุ่มอุตสาหกรรม (Sector Rotation)</span>
                </div>
                <div className="flex items-center gap-1.5">
                  <CheckSmallIcon className="h-3 w-3 text-emerald-600 shrink-0" />
                  <span>07: ข่าวกรองและแหล่งอ้างอิง (Curated News)</span>
                </div>
                <div className="flex items-center gap-1.5">
                  <CheckSmallIcon className="h-3 w-3 text-emerald-600 shrink-0" />
                  <span>08: ภาคผนวกข้อมูลเชิงโครงสร้าง (Structured JSON)</span>
                </div>
              </div>
            </div>

            <div className="text-[11px] text-slate-500 italic pt-1">
              * Research Companion สำหรับค้นคว้าเท่านั้น — ไม่มีการสร้าง Podcast, Discord หรือเขียนทับ conviction
            </div>
          </div>

          {/* Action Row inside popover */}
          <div className="pt-2 border-t border-slate-100 flex items-center justify-between gap-2">
            <button
              type="button"
              onClick={() => {
                setShowDetails(false)
                handleExport()
              }}
              disabled={isSubmitting}
              className="inline-flex items-center gap-1.5 rounded-lg border border-sky-200 bg-white px-2.5 py-1 text-[11px] font-semibold text-section-accent hover:bg-sky-50 transition-colors disabled:opacity-50"
              title="สร้าง Snapshot ใหม่และส่งออกอีกครั้ง"
            >
              <RefreshIcon className="h-3 w-3 shrink-0" />
              <span>ส่งออกชุดข้อมูลใหม่</span>
            </button>

            {primaryNotebook?.url && (
              <a
                href={primaryNotebook.url}
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-1 rounded-lg bg-section-accent px-3 py-1 text-[11px] font-semibold text-white hover:opacity-90 transition-opacity"
              >
                <span>เปิดในเบราว์เซอร์</span>
                <ExternalLinkIcon className="h-3 w-3 shrink-0" />
              </a>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
