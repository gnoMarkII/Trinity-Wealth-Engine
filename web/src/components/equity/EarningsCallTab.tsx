import React, { useState, useEffect, useRef, useCallback } from 'react'
import { Link } from 'react-router-dom'
import ReactMarkdown from 'react-markdown'
import { api } from '../../api/client'
import type {
  EarningsCallSummarizeResponse,
  EarningsCallRunResponse,
  EarningsCallNoteItem,
} from '../../api/types'

interface EarningsCallTabProps {
  ticker: string
  market?: 'US' | 'TH'
}

type TabUiStatus = 'idle' | 'loading' | 'pending' | 'success' | 'failed' | 'error'

export const EarningsCallTab: React.FC<EarningsCallTabProps> = ({ ticker }) => {
  // Existing notes state
  const [existingCalls, setExistingCalls] = useState<EarningsCallNoteItem[]>([])
  const [selectedCallIndex, setSelectedCallIndex] = useState<number>(0)
  const [isLoadingExisting, setIsLoadingExisting] = useState<boolean>(true)
  const [showNewForm, setShowNewForm] = useState<boolean>(false)

  // Form submission & workflow state
  const [period, setPeriod] = useState('Q2 2026')
  const [transcript, setTranscript] = useState('')
  const [status, setStatus] = useState<TabUiStatus>('idle')
  const [errorMessage, setErrorMessage] = useState<string | null>(null)
  const [result, setResult] = useState<EarningsCallSummarizeResponse | EarningsCallRunResponse | null>(null)
  const [isRetrying, setIsRetrying] = useState(false)

  const pollIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const clearPolling = () => {
    if (pollIntervalRef.current) {
      clearInterval(pollIntervalRef.current)
      pollIntervalRef.current = null
    }
  }

  const fetchExistingCalls = useCallback(async () => {
    try {
      setIsLoadingExisting(true)
      const res = await api.getEarningsCalls(ticker)
      const items = res.items || []
      setExistingCalls(items)
      if (items.length > 0) {
        setSelectedCallIndex(0)
        setShowNewForm(false)
      } else {
        setShowNewForm(true)
      }
    } catch (err) {
      console.error('Failed to fetch existing earnings calls:', err)
      setShowNewForm(true)
    } finally {
      setIsLoadingExisting(false)
    }
  }, [ticker])

  useEffect(() => {
    fetchExistingCalls()
    return () => clearPolling()
  }, [fetchExistingCalls])

  const startPolling = (runId: string) => {
    clearPolling()
    let attempts = 0
    const maxAttempts = 20

    pollIntervalRef.current = setInterval(async () => {
      attempts++
      try {
        const run = await api.getEarningsCallRun(ticker, runId)
        setResult(run)

        if (run.status === 'completed') {
          clearPolling()
          setStatus('success')
          fetchExistingCalls()
        } else if (run.status === 'failed' || run.kanban_status === 'failed') {
          clearPolling()
          setStatus('failed')
        } else if (attempts >= maxAttempts) {
          clearPolling()
          setStatus('pending')
        }
      } catch (err) {
        console.error('Polling error:', err)
      }
    }, 3000)
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    clearPolling()
    const cleanPeriod = period.trim()
    const cleanTranscript = transcript.trim()

    if (!cleanPeriod) {
      setErrorMessage('กรุณาระบุไตรมาส เช่น Q2 2026 หรือ 2Q26')
      return
    }

    if (cleanTranscript.length < 20) {
      setErrorMessage('ข้อความ Transcript สั้นเกินไป (ต้องมีอย่างน้อย 20 ตัวอักษร)')
      return
    }

    if (cleanTranscript.length > 120000) {
      setErrorMessage('ข้อความ Transcript ยาวเกินกำหนด (จำกัดไม่เกิน 120,000 ตัวอักษร)')
      return
    }

    try {
      setStatus('loading')
      setErrorMessage(null)
      const res = await api.summarizeEarningsCall(ticker, cleanPeriod, cleanTranscript)
      setResult(res)

      if (res.status === 'completed') {
        setStatus('success')
        fetchExistingCalls()
      } else if (res.status === 'failed') {
        setStatus('failed')
      } else {
        setStatus('pending')
        startPolling(res.run_id)
      }
    } catch (err: any) {
      console.error('Failed to summarize earnings call:', err)
      setStatus('error')
      setErrorMessage(err.message || 'เกิดข้อผิดพลาดในการประมวลผล Earnings Call')
    }
  }

  const handleRetry = async () => {
    if (!result || !result.run_id) return
    try {
      setIsRetrying(true)
      setErrorMessage(null)
      const res = await api.retryEarningsCallRun(ticker, result.run_id)
      setResult(res)

      if (res.status === 'completed') {
        setStatus('success')
        fetchExistingCalls()
      } else {
        setStatus('pending')
        startPolling(res.run_id)
      }
    } catch (err: any) {
      console.error('Failed to retry earnings call:', err)
      setErrorMessage(err.message || 'เกิดข้อผิดพลาดในการ Retry')
    } finally {
      setIsRetrying(false)
    }
  }

  const handleReset = () => {
    clearPolling()
    setTranscript('')
    setStatus('idle')
    setResult(null)
    setErrorMessage(null)
  }

  const currentActiveCall = existingCalls[selectedCallIndex] || null

  return (
    <div className="space-y-6">
      {/* Header Info */}
      <div className="rounded-2xl border border-edge bg-gradient-to-r from-sky-500/10 via-surface to-surface p-5 shadow-sm">
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
          <div>
            <div className="flex items-center gap-2">
              <span className="text-xl">🎙️</span>
              <h3 className="text-lg font-bold text-zinc-900">
                Earnings Call Insights & Transcript Analyzer
              </h3>
              <span className="rounded-md bg-sky-100 px-2 py-0.5 text-xs font-semibold text-sky-800">
                {ticker}
              </span>
            </div>
            <p className="text-xs text-zinc-600 mt-1 max-w-2xl leading-relaxed">
              วิเคราะห์และสกัดสรุปบทวิเคราะห์ผลประกอบการ 5 ด้านสำคัญ (Financial Highlights, Guidance, Q&A Takeaways, Risks, และ Executive Tone) บันทึกลง Obsidian Vault พร้อมระบบติดตามงานผ่าน Kanban
            </p>
          </div>

          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={() => setShowNewForm(!showNewForm)}
              className={`px-3.5 py-2 rounded-xl text-xs font-semibold flex items-center gap-1.5 transition-all shadow-sm ${
                showNewForm
                  ? 'bg-zinc-200 text-zinc-800 hover:bg-zinc-300'
                  : 'bg-sky-600 text-white hover:bg-sky-700'
              }`}
            >
              <span>{showNewForm ? '✕ ซ่อนแบบฟอร์ม' : '✨ สกัด Transcript ใหม่'}</span>
            </button>
          </div>
        </div>
      </div>

      {/* Loading Skeleton */}
      {isLoadingExisting && (
        <div className="rounded-2xl border border-edge bg-panel p-8 text-center text-zinc-500 animate-pulse space-y-3">
          <div className="text-2xl">⏳</div>
          <div className="text-sm font-medium">กำลังโหลดข้อมูล Earnings Call จาก Obsidian Vault...</div>
        </div>
      )}

      {/* Input Form for New Transcript (Collapsible) */}
      {showNewForm && (
        <div className="rounded-2xl border border-sky-200 bg-sky-50/40 p-5 shadow-sm space-y-4 animate-page-in">
          <div className="flex items-center justify-between border-b border-sky-200/60 pb-3">
            <h4 className="text-sm font-bold text-sky-900 flex items-center gap-2">
              <span>✨</span>
              <span>แบบฟอร์มสกัดและบันทึก Transcript ใหม่ ({ticker})</span>
            </h4>
            <span className="text-xs text-sky-700">AI จะสกัด 5 หัวข้อสำคัญและบันทึกลง Obsidian อัตโนมัติ</span>
          </div>

          <form onSubmit={handleSubmit} className="space-y-4">
            <div className="flex flex-col sm:flex-row gap-4">
              <div className="sm:w-1/3">
                <label htmlFor="period-input" className="block text-xs font-semibold text-zinc-700 uppercase tracking-wider mb-1.5">
                  ไตรมาส / งวดผลประกอบการ <span className="text-rose-500">*</span>
                </label>
                <input
                  id="period-input"
                  type="text"
                  value={period}
                  onChange={(e) => setPeriod(e.target.value)}
                  placeholder="เช่น Q2 2026 หรือ 2Q26"
                  className="w-full rounded-lg border border-edge bg-surface px-3 py-2 text-sm text-zinc-900 placeholder:text-zinc-400 focus:border-sky-500 focus:outline-none focus:ring-1 focus:ring-sky-500 transition-colors"
                  disabled={status === 'loading'}
                  required
                />
              </div>
            </div>

            <div>
              <div className="flex justify-between items-center mb-1.5">
                <label htmlFor="transcript-input" className="block text-xs font-semibold text-zinc-700 uppercase tracking-wider">
                  เนื้อหา Transcript (Copy & Paste มาวางที่นี่) <span className="text-rose-500">*</span>
                </label>
                <span className={`text-xs ${transcript.length > 120000 ? 'text-rose-500 font-semibold' : 'text-zinc-400'}`}>
                  {transcript.length.toLocaleString()} / 120,000 ตัวอักษร
                </span>
              </div>
              <textarea
                id="transcript-input"
                rows={8}
                value={transcript}
                onChange={(e) => setTranscript(e.target.value)}
                placeholder="วางเนื้อหา Transcript ภาษาอังกฤษหรือไทยที่นี่ เช่น บทสนทนาของผู้บริหารและนักวิเคราะห์..."
                className="w-full rounded-lg border border-edge bg-surface p-3 text-sm text-zinc-900 placeholder:text-zinc-400 font-mono text-xs leading-relaxed focus:border-sky-500 focus:outline-none focus:ring-1 focus:ring-sky-500 transition-colors"
                disabled={status === 'loading'}
                required
              />
            </div>

            <div className="flex items-center gap-3">
              <button
                type="submit"
                disabled={status === 'loading' || !transcript.trim()}
                className="px-4 py-2.5 rounded-lg bg-sky-600 hover:bg-sky-700 disabled:bg-zinc-300 disabled:cursor-not-allowed text-white text-sm font-semibold flex items-center gap-2 shadow-sm transition-colors"
              >
                {status === 'loading' ? (
                  <>
                    <svg className="animate-spin h-4 w-4 text-white" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z" />
                    </svg>
                    <span>กำลังสกัด Highlights & บันทึก...</span>
                  </>
                ) : (
                  <>
                    <span>✨</span>
                    <span>แปลงเป็น Highlights & ส่งเข้า Kanban</span>
                  </>
                )}
              </button>

              {transcript && status !== 'loading' && (
                <button
                  type="button"
                  onClick={handleReset}
                  className="px-3 py-2 rounded-lg border border-edge bg-surface hover:bg-surface-strong text-zinc-600 text-sm font-medium transition-colors"
                >
                  ล้างข้อความ
                </button>
              )}
            </div>
          </form>
        </div>
      )}

      {/* Error Message */}
      {(status === 'error' || errorMessage) && (
        <div className="rounded-xl border border-red-200 bg-red-50 p-4 text-sm text-red-700" role="alert">
          <div className="font-semibold flex items-center gap-2">
            <span>⚠️</span>
            <span>เกิดข้อผิดพลาด</span>
          </div>
          <p className="mt-1">{errorMessage}</p>
        </div>
      )}

      {/* Pending / In-progress Status Banner */}
      {status === 'pending' && result && (
        <div className="rounded-xl border border-amber-200 bg-amber-50 p-4 space-y-3 animate-pulse">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <span className="flex h-2.5 w-2.5 rounded-full bg-amber-500" />
              <span className="font-semibold text-amber-900 text-sm">
                บันทึก Note ใน Obsidian สำเร็จแล้ว กำลังจัดส่งเข้า Kanban Backlog...
              </span>
            </div>
            <button
              type="button"
              onClick={handleRetry}
              disabled={isRetrying}
              className="px-3 py-1 bg-amber-600 hover:bg-amber-700 disabled:opacity-50 text-white rounded text-xs font-semibold flex items-center gap-1 shadow-sm transition-colors"
            >
              {isRetrying ? 'กำลัง Retry...' : '🔄 ลองส่งใหม่ (Retry)'}
            </button>
          </div>
          {result.vault_path && (
            <div className="text-xs text-amber-800 font-mono">
              ไฟล์ Note: {result.vault_path}
            </div>
          )}
        </div>
      )}

      {/* Failed Delivery Banner */}
      {status === 'failed' && result && (
        <div className="rounded-xl border border-rose-200 bg-rose-50 p-4 space-y-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <span className="flex h-2.5 w-2.5 rounded-full bg-rose-500" />
              <span className="font-semibold text-rose-900 text-sm">
                บันทึก Obsidian สำเร็จ แต่การส่งเข้า Kanban ขัดข้อง
              </span>
            </div>
            <button
              type="button"
              onClick={handleRetry}
              disabled={isRetrying}
              className="px-3 py-1 bg-rose-600 hover:bg-rose-700 disabled:opacity-50 text-white rounded text-xs font-semibold flex items-center gap-1 shadow-sm transition-colors"
            >
              {isRetrying ? 'กำลัง Retry...' : '🔄 ลองส่งใหม่ (Retry)'}
            </button>
          </div>
          {result.vault_path && (
            <div className="text-xs text-rose-800 font-mono">
              ไฟล์ Note: {result.vault_path}
            </div>
          )}
        </div>
      )}

      {/* Existing Quarter Tabs / Switcher if notes exist */}
      {existingCalls.length > 0 && (
        <div className="flex flex-wrap items-center justify-between gap-3 border-b border-edge/60 pb-3">
          <div className="flex items-center gap-2 overflow-x-auto py-1">
            <span className="text-xs font-semibold uppercase tracking-wider text-zinc-400 mr-1">
              ไตรมาสที่บันทึกแล้ว:
            </span>
            {existingCalls.map((call, idx) => (
              <button
                key={call.vault_path || idx}
                onClick={() => setSelectedCallIndex(idx)}
                className={`px-3.5 py-1.5 rounded-lg text-xs font-semibold transition-all flex items-center gap-1.5 ${
                  selectedCallIndex === idx
                    ? 'bg-sky-600 text-white shadow-sm ring-2 ring-sky-600/30'
                    : 'bg-surface-strong text-zinc-700 hover:bg-zinc-200 hover:text-zinc-900 border border-edge/60'
                }`}
              >
                <span>📅</span>
                <span>{call.period}</span>
              </button>
            ))}
          </div>

          {currentActiveCall && (
            <div className="flex items-center gap-2 text-xs text-zinc-500">
              <span>อัปเดต: {currentActiveCall.date}</span>
            </div>
          )}
        </div>
      )}

      {/* Display Active / Selected Earnings Call Highlights */}
      {!isLoadingExisting && currentActiveCall && (
        <div className="space-y-5 animate-page-in">
          {/* Metadata Cards */}
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 text-xs">
            <div className="rounded-xl border border-edge bg-panel p-3.5 shadow-sm">
              <div className="flex items-center gap-2 text-zinc-500 font-medium mb-1">
                <span>📁</span>
                <span>Obsidian Vault Note</span>
              </div>
              <div className="font-mono text-zinc-800 font-semibold truncate" title={currentActiveCall.vault_path}>
                {currentActiveCall.vault_path}
              </div>
              <div className="text-[11px] text-zinc-400 mt-1">
                มีข้อมูล AI Highlights และ Full Transcript
              </div>
            </div>

            <div className="rounded-xl border border-edge bg-panel p-3.5 shadow-sm">
              <div className="flex items-center gap-2 text-zinc-500 font-medium mb-1">
                <span>📅</span>
                <span>รอบผลประกอบการ</span>
              </div>
              <div className="text-sm font-bold text-sky-600">
                {currentActiveCall.period}
              </div>
              <div className="text-[11px] text-zinc-400 mt-1">
                วันที่วิเคราะห์: {currentActiveCall.date}
              </div>
            </div>

            <div className="rounded-xl border border-edge bg-panel p-3.5 shadow-sm">
              <div className="flex items-center gap-2 text-zinc-500 font-medium mb-1">
                <span>📋</span>
                <span>Kanban Workflow</span>
              </div>
              <div>
                <Link
                  to="/kanban"
                  className="inline-flex items-center gap-1 text-sky-600 hover:text-sky-800 font-semibold underline underline-offset-2"
                >
                  <span>เปิดดูในการ์ด Kanban Board</span>
                  <span>→</span>
                </Link>
              </div>
              <div className="text-[11px] text-emerald-600 font-medium mt-1">
                ✓ เชื่อมโยงในระบบเรียบร้อย
              </div>
            </div>
          </div>

          {/* Rendered Highlights in Clean Typography & Beautiful Layout */}
          <div className="rounded-2xl border border-edge bg-panel p-6 sm:p-8 shadow-sm">
            <div className="border-b border-edge pb-4 mb-6 flex flex-wrap items-center justify-between gap-3">
              <div>
                <h4 className="text-lg font-bold text-zinc-900 flex items-center gap-2">
                  <span>🤖</span>
                  <span>{currentActiveCall.title || `${ticker} Earnings Call — ${currentActiveCall.period}`}</span>
                </h4>
                <p className="text-xs text-zinc-500 mt-0.5">
                  สกัดและจัดระเบียบสาระสำคัญโดย AI Multi-Agent Engine
                </p>
              </div>
              <span className="px-3 py-1 rounded-full bg-emerald-50 text-emerald-700 border border-emerald-200 text-xs font-semibold flex items-center gap-1.5">
                <span className="h-2 w-2 rounded-full bg-emerald-500" />
                <span>Executive Ready</span>
              </span>
            </div>

            <div className="prose prose-sm sm:prose max-w-none text-zinc-800 leading-relaxed">
              <ReactMarkdown
                components={{
                  a: ({ children, href }) => (
                    <a href={href} target="_blank" rel="noreferrer" className="font-semibold text-sky-600 underline underline-offset-2 hover:text-sky-800">
                      {children}
                    </a>
                  ),
                  code: ({ children, className }) => (
                    <code className={className || "rounded bg-surface-strong px-1.5 py-0.5 font-mono text-[0.84em] text-purple-700 font-semibold"}>{children}</code>
                  ),
                  h1: ({ children }) => <h1 className="text-xl font-bold text-zinc-900 border-b border-edge pb-2 mt-6 mb-3 first:mt-0">{children}</h1>,
                  h2: ({ children }) => <h2 className="text-lg font-bold text-zinc-900 border-b border-edge/60 pb-1.5 mt-6 mb-3 first:mt-0">{children}</h2>,
                  h3: ({ children }) => (
                    <div className="rounded-xl bg-surface-strong/70 border border-edge/80 px-4 py-2.5 mt-6 mb-3">
                      <h3 className="text-sm font-bold text-zinc-900 m-0">{children}</h3>
                    </div>
                  ),
                  h4: ({ children }) => <h4 className="text-sm font-semibold text-zinc-800 mt-3 mb-1.5 first:mt-0">{children}</h4>,
                  p: ({ children }) => <p className="my-3 leading-relaxed text-zinc-700 text-sm">{children}</p>,
                  ul: ({ children }) => <ul className="my-3 space-y-2 list-disc pl-5 text-zinc-700 text-sm">{children}</ul>,
                  ol: ({ children }) => <ol className="my-3 space-y-2 list-decimal pl-5 text-zinc-700 text-sm">{children}</ol>,
                  li: ({ children }) => <li className="pl-0.5 leading-relaxed">{children}</li>,
                  strong: ({ children }) => <strong className="font-bold text-zinc-900">{children}</strong>,
                  blockquote: ({ children }) => (
                    <blockquote className="border-l-4 border-sky-500 bg-sky-50/60 italic px-4 py-3 my-4 rounded-r-xl text-zinc-800 text-sm shadow-sm">
                      {children}
                    </blockquote>
                  ),
                  hr: () => <hr className="my-6 border-edge" />,
                }}
              >
                {currentActiveCall.highlights}
              </ReactMarkdown>
            </div>
          </div>
        </div>
      )}

      {/* Empty State when no calls exist and form not open */}
      {!isLoadingExisting && existingCalls.length === 0 && !showNewForm && (
        <div className="rounded-2xl border border-dashed border-edge bg-panel p-10 text-center space-y-3">
          <div className="text-3xl">🎙️</div>
          <div className="text-sm font-semibold text-zinc-700">
            ยังไม่มีข้อมูล Earnings Call สำหรับ {ticker}
          </div>
          <p className="text-xs text-zinc-500 max-w-md mx-auto">
            คุณสามารถวางเนื้อหาบทสนทนาแถลงผลประกอบการ (Transcript) เพื่อให้ AI สกัดประเด็นสำคัญและบันทึกลง Obsidian ได้ทันที
          </p>
          <button
            type="button"
            onClick={() => setShowNewForm(true)}
            className="mt-2 px-4 py-2 bg-sky-600 hover:bg-sky-700 text-white rounded-lg text-xs font-semibold inline-flex items-center gap-1.5 shadow-sm transition-colors"
          >
            <span>✨ วาง Transcript ใหม่</span>
          </button>
        </div>
      )}
    </div>
  )
}
