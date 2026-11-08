import React, { useCallback, useEffect, useRef, useState } from 'react'
import { api } from '../../api/client'
import type { MorningReviewPacketDTO } from '../../api/types'

export interface MorningDeltaCardProps {
  portfolioId?: string
  initialPacket?: MorningReviewPacketDTO | null
  className?: string
  defaultExpanded?: boolean
  onSignOffSuccess?: () => void
}

export const MorningDeltaCard: React.FC<MorningDeltaCardProps> = ({
  portfolioId = 'default',
  initialPacket = null,
  className = '',
  defaultExpanded = true,
  onSignOffSuccess,
}) => {
  const [loadedPacket, setLoadedPacket] = useState<MorningReviewPacketDTO | null>(initialPacket)
  const [packetPortfolioId, setPacketPortfolioId] = useState<string | null>(initialPacket ? portfolioId : null)
  const [loading, setLoading] = useState<boolean>(!initialPacket)
  const [error, setError] = useState<string | null>(null)
  const [isExpanded, setIsExpanded] = useState<boolean>(defaultExpanded)
  const [isSignOffModalOpen, setIsSignOffModalOpen] = useState(false)
  const [signOffNotes, setSignOffNotes] = useState('')
  const [verifiedSampleIds, setVerifiedSampleIds] = useState<string[]>([])
  const [acknowledgedExceptionIds, setAcknowledgedExceptionIds] = useState<string[]>([])
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [reviewingExceptionId, setReviewingExceptionId] = useState<string | null>(null)
  const [assessmentText, setAssessmentText] = useState('')
  const [isSavingAssessment, setIsSavingAssessment] = useState(false)
  const currentPortfolioRef = useRef(portfolioId)
  currentPortfolioRef.current = portfolioId
  const requestGenerationRef = useRef(0)
  const signoffActionGenerationRef = useRef(0)
  const assessmentActionGenerationRef = useRef(0)
  const packet = packetPortfolioId === portfolioId ? loadedPacket : null
  const requiredExceptionIds = (packet?.thesis_exceptions ?? [])
    .filter((exception) => !['ACKNOWLEDGED', 'REVISED'].includes(String(exception.status ?? 'PENDING').toUpperCase()))
    .map((exception) => exception.exception_id)

  useEffect(() => {
    setVerifiedSampleIds([])
    setAcknowledgedExceptionIds([])
  }, [packet?.packet_sha256])

  const loadPacket = useCallback(async (targetPortfolio: string = portfolioId) => {
    if (currentPortfolioRef.current !== targetPortfolio) return
    const generation = ++requestGenerationRef.current
    setLoading(true)
    setError(null)
    try {
      const data = await api.getMorningReviewToday(targetPortfolio)
      if (requestGenerationRef.current === generation && currentPortfolioRef.current === targetPortfolio) {
        setLoadedPacket(data)
        setPacketPortfolioId(targetPortfolio)
      }
    } catch (err: any) {
      if (requestGenerationRef.current === generation && currentPortfolioRef.current === targetPortfolio) {
        console.error('Failed to load morning delta briefing:', err)
        setError(err?.message || 'Failed to load morning delta briefing')
      }
    } finally {
      if (requestGenerationRef.current === generation && currentPortfolioRef.current === targetPortfolio) {
        setLoading(false)
      }
    }
  }, [portfolioId])

  useEffect(() => {
    requestGenerationRef.current += 1
    signoffActionGenerationRef.current += 1
    assessmentActionGenerationRef.current += 1
    setVerifiedSampleIds([])
    setAcknowledgedExceptionIds([])
    setSignOffNotes('')
    setIsSignOffModalOpen(false)
    setReviewingExceptionId(null)
    setAssessmentText('')
    setIsSubmitting(false)
    setIsSavingAssessment(false)
    setError(null)
    if (initialPacket) {
      setLoadedPacket(initialPacket)
      setPacketPortfolioId(portfolioId)
      setLoading(false)
    } else {
      setLoadedPacket(null)
      setPacketPortfolioId(null)
      setLoading(true)
      void loadPacket(portfolioId)
    }
    return () => {
      requestGenerationRef.current += 1
    }
  }, [portfolioId, initialPacket, loadPacket])

  const toggleSampleCheck = (id: string) => {
    setVerifiedSampleIds((prev) =>
      prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id]
    )
  }

  const handleVerifyAll = () => {
    if (!packet?.review_sample) return
    const allIds = packet.review_sample.map((s) => s.sample_id)
    setVerifiedSampleIds(allIds)
  }

  const toggleExceptionAcknowledgement = (id: string) => {
    setAcknowledgedExceptionIds((prev) =>
      prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id]
    )
  }

  const handleSignOff = async () => {
    const targetPortfolio = portfolioId
    const targetPacket = packet
    const actionGeneration = ++signoffActionGenerationRef.current
    const requestGeneration = requestGenerationRef.current
    if (!targetPacket || currentPortfolioRef.current !== targetPortfolio) return

    setIsSubmitting(true)
    try {
      await api.signOffMorningReview({
        portfolio_id: targetPortfolio,
        review_date: targetPacket.review_date,
        signed_off_by: 'HUMAN_PM',
        notes: signOffNotes.trim() || undefined,
        packet_sha256: targetPacket.packet_sha256,
        verified_sample_ids: verifiedSampleIds,
        acknowledged_exception_ids: acknowledgedExceptionIds,
      })
      if (currentPortfolioRef.current !== targetPortfolio || signoffActionGenerationRef.current !== actionGeneration) return
      if (requestGenerationRef.current !== requestGeneration) {
        setIsSubmitting(false)
        void loadPacket(targetPortfolio)
        return
      }

      setIsSignOffModalOpen(false)
      setSignOffNotes('')
      setVerifiedSampleIds([])
      setAcknowledgedExceptionIds([])
      setIsSubmitting(false)
      const refresh = loadPacket(targetPortfolio)
      const refreshGeneration = requestGenerationRef.current
      await refresh
      if (currentPortfolioRef.current === targetPortfolio && requestGenerationRef.current === refreshGeneration) {
        onSignOffSuccess?.()
      }
    } catch (err: any) {
      if (currentPortfolioRef.current === targetPortfolio && signoffActionGenerationRef.current === actionGeneration) {
        alert('Sign-off failed: ' + (err?.message || 'Unknown error'))
      }
    } finally {
      if (currentPortfolioRef.current === targetPortfolio && signoffActionGenerationRef.current === actionGeneration) {
        setIsSubmitting(false)
      }
    }
  }

  const handleSaveAssessment = async (exceptionId: string) => {
    const targetPortfolio = portfolioId
    const actionGeneration = ++assessmentActionGenerationRef.current
    const requestGeneration = requestGenerationRef.current
    if (currentPortfolioRef.current !== targetPortfolio) return

    setIsSavingAssessment(true)
    try {
      await api.reviewCatalystException(exceptionId, assessmentText.trim() || 'Acknowledged by PM')
      if (currentPortfolioRef.current !== targetPortfolio || assessmentActionGenerationRef.current !== actionGeneration) return
      if (requestGenerationRef.current !== requestGeneration) {
        setIsSavingAssessment(false)
        void loadPacket(targetPortfolio)
        return
      }

      setReviewingExceptionId(null)
      setAssessmentText('')
      setIsSavingAssessment(false)
      const refresh = loadPacket(targetPortfolio)
      await refresh
    } catch (err: any) {
      if (currentPortfolioRef.current === targetPortfolio && assessmentActionGenerationRef.current === actionGeneration) {
        alert('Failed to save assessment: ' + (err?.message || 'Unknown error'))
      }
    } finally {
      if (currentPortfolioRef.current === targetPortfolio && assessmentActionGenerationRef.current === actionGeneration) {
        setIsSavingAssessment(false)
      }
    }
  }

  // Slim single-line skeleton while loading
  if (loading && !packet) {
    return (
      <div
        data-testid="morning-delta-card-loading"
        className={`flex h-12 w-full items-center justify-between rounded-2xl border border-sky-100 bg-white/70 px-4 text-xs text-sky-800 animate-pulse ${className}`}
      >
        <div className="flex items-center gap-2">
          <div className="h-5 w-5 rounded-md bg-sky-200" />
          <div className="h-4 w-40 rounded bg-sky-200" />
        </div>
        <div className="h-4 w-20 rounded bg-sky-200" />
      </div>
    )
  }

  // Informative compact error banner
  if (error && !packet) {
    return (
      <div
        data-testid="morning-delta-card-error"
        className={`flex items-center justify-between rounded-2xl border border-rose-200 bg-rose-50/80 p-4 text-xs text-rose-800 shadow-2xs ${className}`}
      >
        <div className="flex items-center gap-2">
          <span>⚠️</span>
          <span>ไม่สามารถโหลด Morning PM Review: {error}</span>
        </div>
        <button
          type="button"
          onClick={() => loadPacket()}
          className="rounded-lg bg-rose-600 px-3 py-1 text-white text-[11px] font-semibold hover:bg-rose-700 transition-colors shadow-2xs"
        >
          ลองใหม่
        </button>
      </div>
    )
  }

  if (!packet) {
    return null
  }

  const isSignedOff = packet.status === 'SIGNED_OFF'

  return (
    <section
      data-testid="morning-delta-card"
      className={`flex flex-col gap-3 rounded-2xl border border-sky-200/90 bg-white/90 p-4 shadow-xs backdrop-blur-md transition-all ${className}`}
    >
      {/* 1. Header Card (Clear & Uncluttered) */}
      <div className="flex flex-col gap-2.5 border-b border-sky-100 pb-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2.5">
            <div className="flex h-7 w-7 items-center justify-center rounded-xl bg-gradient-to-tr from-sky-500 to-cyan-500 text-white shadow-xs">
              <span className="text-xs">🏛️</span>
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h3 className="text-xs font-bold text-sky-950">Morning PM Review (Delta-Only)</h3>
                <span className="rounded bg-sky-100 text-sky-800 text-[9px] px-1.5 py-0.2 font-medium" title="สรุปการเปลี่ยนแปลงในรอบ 24 ชม. เท่านั้น ช่วยประหยัดเวลาอ่าน">
                  สรุปด่วน 2 นาที
                </span>
              </div>
              <p className="text-[10px] text-zinc-500 font-mono">
                {packet.review_date} • ตรวจสุขภาพพอร์ตและเกณฑ์ความเสี่ยงก่อนตลาดเปิด
              </p>
            </div>
          </div>

          <button
            type="button"
            data-testid="toggle-quadrants-btn"
            onClick={() => setIsExpanded((prev) => !prev)}
            className="flex items-center gap-1 rounded-lg border border-sky-200 bg-white hover:bg-sky-50 px-2 py-1 text-[11px] font-semibold text-sky-800 transition-colors shadow-2xs"
          >
            <span>{isExpanded ? 'ย่อ ▴' : '4 Quadrants ▾'}</span>
          </button>
        </div>

        {/* Status Badge & Primary Action Button */}
        <div className="flex items-center justify-between gap-2 pt-0.5">
          {isSignedOff ? (
            <span
              data-testid="status-signed-off"
              className="inline-flex items-center gap-1.5 rounded-full border border-emerald-300 bg-emerald-50 px-3 py-1 font-mono text-[11px] font-bold text-emerald-800 shadow-2xs"
            >
              <span className="h-2 w-2 rounded-full bg-emerald-500" />
              SIGNED OFF ({packet.signoff_info?.signed_off_by || 'CHINO_PM'})
            </span>
          ) : (
            <span
              data-testid="status-review-required"
              className="inline-flex items-center gap-1.5 rounded-full border border-amber-300 bg-amber-50 px-3 py-1 font-mono text-[11px] font-bold text-amber-800 shadow-2xs animate-pulse"
            >
              <span className="h-2 w-2 rounded-full bg-amber-500" />
              REVIEW REQUIRED
            </span>
          )}

          {!isSignedOff && (
            <button
              type="button"
              onClick={() => setIsSignOffModalOpen(true)}
              className="rounded-xl bg-sky-600 hover:bg-sky-700 px-3.5 py-1.5 text-xs font-bold text-white shadow-xs transition-colors"
            >
              Sign-off &amp; Acknowledge
            </button>
          )}
        </div>

        {/* Institutional Routine Run Quality Banner (Ch. 10) */}
        {packet.run_quality === 'MISSED_RUN' ? (
          <div
            data-testid="missed-run-warning"
            className="flex items-center gap-2 rounded-lg border border-rose-300 bg-rose-50/90 px-3 py-2 text-[11px] font-medium text-rose-900 shadow-2xs"
          >
            <span>🚨</span>
            <span>
              <strong>MISSED_RUN: Scheduled routine was missed (+{packet.drift_minutes || 0}m delay). Check runner logs before trading.</strong>
            </span>
          </div>
        ) : (packet.run_quality === 'LATE_RUN' || packet.run_quality === 'LATE_RUN_STALE' || (packet.drift_minutes ?? 0) > 30) ? (
          <div
            data-testid="late-run-warning"
            className="flex items-center gap-2 rounded-lg border border-amber-300 bg-amber-50/90 px-3 py-2 text-[11px] font-medium text-amber-900 shadow-2xs"
          >
            <span>⏰</span>
            <span>
              <strong>Late Run (+{packet.drift_minutes || 0}m จาก Cutoff 08:30 น.):</strong> ตลาดอาจเปิดทำการแล้ว กรุณาตรวจสอบราคา As-Of ให้แน่ชัดก่อนดำเนินการ
            </span>
          </div>
        ) : packet.run_quality === 'ON_TIME' ? (
          <div
            data-testid="routine-fresh-badge"
            className="flex items-center gap-2 rounded-lg border border-emerald-200 bg-emerald-50/80 px-3 py-1.5 text-[10px] font-medium text-emerald-800 shadow-2xs"
          >
            <span>🟢</span>
            <span>
              <strong>Routine Fresh:</strong> Morning packet generated on schedule ({packet.drift_minutes || 0}m drift).
            </span>
          </div>
        ) : null}

        {/* Institutional Quarantined Holdings Alert (Ch. 8, p. 57) */}
        {packet.quarantined_holdings && packet.quarantined_holdings.length > 0 && (
          <div
            data-testid="quarantined-holdings-alert"
            className="flex flex-col gap-1.5 rounded-lg border border-rose-300 bg-rose-50/90 px-3 py-2 text-[11px] text-rose-950 shadow-2xs"
          >
            <div className="flex items-center gap-1.5 font-bold text-rose-900">
              <span>🚨</span>
              <span>
                กักกันความปลอดภัย {packet.quarantined_holdings.length} สินทรัพย์ (ระงับ Rebalance ชั่วคราว):
              </span>
            </div>
            <div className="flex flex-wrap gap-1.5">
              {packet.quarantined_holdings.map((q, idx) => (
                <span
                  key={idx}
                  className="inline-flex items-center gap-1 rounded bg-rose-200/80 px-2 py-0.5 font-mono text-[10px] font-bold text-rose-900"
                  title={q.reason}
                >
                  ${q.symbol} ({q.guardrail_code || 'MISSING_PRICE'})
                </span>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* 2. Four Vertical Sections (Stacked Vertically — 1 -> 2 -> 3 -> 4) */}
      <div
        data-testid="quadrants-tray"
        className={`flex flex-col gap-3 ${isExpanded ? 'block animate-fade-in' : 'hidden'}`}
      >
        {/* Section 1: Portfolio Delta */}
        <div className="rounded-xl border border-sky-100 bg-sky-50/30 p-3.5 space-y-2.5 shadow-2xs">
          <div className="flex items-center justify-between">
            <span className="text-xs font-bold text-sky-950 flex items-center gap-1.5">
              <span>📈 Portfolio Delta</span>
            </span>
            <span
              data-testid="daily-pnl-percent"
              className={"font-mono text-xs font-bold px-2 py-0.5 rounded-md border " + (
                packet.portfolio_delta.daily_pnl_pct == null
                  ? 'bg-zinc-50 text-zinc-500 border-zinc-200'
                  : packet.portfolio_delta.daily_pnl_pct >= 0
                    ? 'bg-emerald-50 text-emerald-700 border-emerald-200'
                    : 'bg-rose-50 text-rose-700 border-rose-200'
              )}
            >
              {packet.portfolio_delta.daily_pnl_pct == null
                ? 'N/A'
                : (packet.portfolio_delta.daily_pnl_pct >= 0 ? '+' : '') + packet.portfolio_delta.daily_pnl_pct.toFixed(2) + '%'}
            </span>
          </div>

          <div className="grid grid-cols-2 gap-2 text-xs">
            <div className="rounded-lg bg-white/80 border border-sky-100/70 p-2">
              <div className="flex items-center justify-between">
                <span className="text-[10px] text-zinc-500 block">NAV รวม</span>
                <span className="text-[9px] text-zinc-400">มูลค่าพอร์ต</span>
              </div>
              <span className="font-mono font-bold text-zinc-900 text-xs">
                ฿{packet.portfolio_delta.total_nav.toLocaleString()}
              </span>
            </div>
            <div className="rounded-lg bg-white/80 border border-sky-100/70 p-2">
              <div className="flex items-center justify-between">
                <span className="text-[10px] text-zinc-500 block">กำไร/ขาดทุน 24h</span>
                <span className="text-[9px] text-zinc-400">เทียบเมื่อวาน</span>
              </div>
              <span
                data-testid="daily-pnl-value"
                className={"font-mono font-bold text-xs " + (
                  packet.portfolio_delta.daily_pnl_value == null
                    ? 'text-zinc-500'
                    : packet.portfolio_delta.daily_pnl_value >= 0 ? 'text-emerald-700' : 'text-rose-700'
                )}
              >
                {packet.portfolio_delta.daily_pnl_value == null
                  ? 'N/A'
                  : (packet.portfolio_delta.daily_pnl_value >= 0 ? '+' : '-') + '\u0e3f' + Math.abs(packet.portfolio_delta.daily_pnl_value).toLocaleString()}
              </span>
            </div>
          </div>

          <div className="flex items-center justify-between text-[11px] pt-0.5 text-zinc-600 px-1">
            <span>
              เงินสดสำรอง: <strong className="font-mono text-zinc-800">{packet.portfolio_delta.cash_weight_pct.toFixed(2)}%</strong>
            </span>
            {packet.portfolio_delta.top_mover && (
              <span className="font-mono text-[10px] text-sky-800 inline-flex items-center gap-1">
                <span>Top Mover:</span>
                <strong>${packet.portfolio_delta.top_mover.ticker}</strong>
                <span>({packet.portfolio_delta.top_mover.change_pct >= 0 ? '+' : ''}{packet.portfolio_delta.top_mover.change_pct}%)</span>
                {packet.portfolio_delta.top_mover.is_lifetime_fallback && (
                  <span className="rounded bg-amber-100 text-amber-900 border border-amber-200 text-[8px] px-1 font-sans" title="Lifetime unrealized return fallback">
                    Lifetime
                  </span>
                )}
              </span>
            )}
          </div>

          {/* Institutional Factor Exposure & Beta Badge (Ch. 8) */}
          {(packet.portfolio_delta.portfolio_beta !== undefined || packet.portfolio_delta.net_exposure_pct !== undefined) && (
            <div
              data-testid="factor-exposure-beta-badge"
              className="flex flex-wrap items-center justify-between gap-1.5 pt-1 border-t border-sky-100/70 text-[10px] font-mono text-zinc-600 px-1"
            >
              <div className="flex items-center gap-1.5" title="สัดส่วนการลงทุนจริงเทียบกับเงินสด (Net Exposure: 100% คือถือหุ้นเต็มพอร์ต, ต่ำกว่า 100% คือมีเงินสดสำรอง)">
                <span className="text-zinc-500">สัดส่วนลงทุน (Exposure):</span>
                <span className="font-bold text-zinc-800">
                  {packet.portfolio_delta.net_exposure_pct !== undefined ? `${packet.portfolio_delta.net_exposure_pct}% Net` : '100% Net'}
                </span>
                {packet.portfolio_delta.short_market_value !== undefined && Number(packet.portfolio_delta.short_market_value) < 0 && (
                  <span className="text-zinc-500">
                    ({packet.portfolio_delta.long_exposure_pct}% L / {packet.portfolio_delta.short_exposure_pct}% S)
                  </span>
                )}
              </div>
              {packet.portfolio_delta.portfolio_beta !== undefined && packet.portfolio_delta.portfolio_beta !== null && Number.isFinite(Number(packet.portfolio_delta.portfolio_beta)) ? (
                <span
                  data-testid="portfolio-beta-badge"
                  className="inline-flex items-center gap-1 rounded bg-sky-100/90 px-1.5 py-0.5 font-bold text-sky-900 border border-sky-200/70 shadow-2xs cursor-help"
                  title="Beta (β): ความผันผวนของพอร์ตเทียบกับตลาด (β < 1.0 แกว่งตัวน้อยกว่าตลาด ปลอดภัยกว่า, β > 1.0 แกว่งตัวแรงกว่า)"
                >
                  <span>β</span>
                  <span>{Number(packet.portfolio_delta.portfolio_beta).toFixed(2)}</span>
                  <span className="text-[9px] text-sky-700">({packet.portfolio_delta.beta_benchmark || 'SPY'})</span>
                </span>
              ) : (
                <span
                  data-testid="portfolio-beta-badge"
                  className="inline-flex items-center gap-1 rounded bg-zinc-100 px-1.5 py-0.5 font-medium text-zinc-600 border border-zinc-200 shadow-2xs"
                  title="Beta unavailable (insufficient benchmark observations)"
                >
                  <span>β</span>
                  <span>N/A</span>
                  <span className="text-[9px] text-zinc-500">({packet.portfolio_delta.beta_benchmark || 'SPY'})</span>
                </span>
              )}
            </div>
          )}

          {/* Institutional State Hash Integrity Status (Ch. 3) */}
          {packet.portfolio_delta.state_integrity && (
            <div
              data-testid="state-integrity-badge"
              className="flex items-center justify-between gap-1 pt-1 border-t border-sky-100/70 text-[10px] font-mono px-1"
            >
              <span className="text-zinc-500">State Baseline:</span>
              <span
                className={`font-bold inline-flex items-center gap-1 ${
                  packet.portfolio_delta.state_integrity.is_valid
                    ? 'text-emerald-700'
                    : 'text-rose-700'
                }`}
                title={packet.portfolio_delta.state_integrity.message}
              >
                <span>{packet.portfolio_delta.state_integrity.is_valid ? '🔒' : '🚨'}</span>
                <span>{packet.portfolio_delta.state_integrity.summary_status}</span>
                <span className="text-zinc-400">({packet.portfolio_delta.state_integrity.checked_files_count} inputs)</span>
              </span>
            </div>
          )}
        </div>

        {/* Section 2: Macro Regime & PIT Vintage Walk */}
        <div className="rounded-xl border border-sky-100 bg-sky-50/30 p-3.5 space-y-2.5 shadow-2xs">
          <div className="flex items-center justify-between">
            <span className="text-xs font-bold text-sky-950 flex items-center gap-1.5">
              <span>🌐 Macro Regime</span>
              <span className="text-[10px] text-zinc-400 font-normal hidden sm:inline">(สภาวะเศรษฐกิจมหภาค)</span>
            </span>
            <span className="rounded-md bg-sky-100/90 border border-sky-200 px-2 py-0.5 text-[10px] font-bold text-sky-900" title="วัฏจักรเศรษฐกิจโลก ณ ปัจจุบัน">
              {packet.macro_delta.macro_regime_label}
            </span>
          </div>

          <div className="grid grid-cols-2 gap-2 text-xs">
            <div className="rounded-lg bg-white/80 border border-sky-100/70 p-2" title="ส่วนต่างผลตอบแทนพันธบัตร 10 ปี - 2 ปี (ถ้าติดลบ คือ Yield Curve Inverted ส่งสัญญาณเตือนเศรษฐกิจชะลอตัว)">
              <div className="flex items-center justify-between">
                <span className="text-[10px] text-zinc-500 block">US 10Y-2Y Spread</span>
                <span className="text-[9px] text-zinc-400">ส่วนต่างดอกเบี้ย</span>
              </div>
              <span className="font-mono font-bold text-zinc-900 text-xs">
                {packet.macro_delta.yield_spread_10y_2y_bps !== null && packet.macro_delta.yield_spread_10y_2y_bps !== undefined && Number.isFinite(Number(packet.macro_delta.yield_spread_10y_2y_bps))
                  ? `${Number(packet.macro_delta.yield_spread_10y_2y_bps).toFixed(1)} bps`
                  : '—'}
              </span>
            </div>
            <div className="rounded-lg bg-white/80 border border-sky-100/70 p-2" title="ดัชนีวัดความตึงตัวของระบบการเงิน (ค่ายิ่งติดลบ = สภาพคล่องไหลเวียนปกติ, ค่าสูงเกิน 1.0 = ตลาดเริ่มตึงตัว)">
              <div className="flex items-center justify-between">
                <span className="text-[10px] text-zinc-500 block">OFR Stress Index</span>
                <span className="text-[9px] text-zinc-400">ความตึงเครียดตลาด</span>
              </div>
              <span className="font-mono font-bold text-zinc-900 text-xs">
                {packet.macro_delta.ofr_stress_index !== null && packet.macro_delta.ofr_stress_index !== undefined && Number.isFinite(Number(packet.macro_delta.ofr_stress_index))
                  ? Number(packet.macro_delta.ofr_stress_index).toFixed(2)
                  : '—'}
              </span>
            </div>
          </div>

          {/* Chapter 6: Point-in-Time Macro Vintage Walk Table */}
          {packet.macro_delta.vintage_walk && packet.macro_delta.vintage_walk.length > 0 && (
            <div data-testid="macro-vintage-walk-section" className="pt-1.5 border-t border-sky-100/70 space-y-1.5">
              <div className="flex items-center justify-between text-[11px]">
                <span className="font-medium text-zinc-700 flex items-center gap-1">
                  <span>📈 Historical Vintage Walk</span>
                </span>
                <span className="inline-flex items-center gap-1 rounded bg-emerald-50 px-1.5 py-0.5 text-[9px] font-mono font-bold text-emerald-700 border border-emerald-200">
                  <span>🛡️ No Look-Ahead</span>
                </span>
              </div>
              <div className="overflow-x-auto rounded-lg border border-sky-100/80 bg-white/90">
                <table className="w-full text-left text-[10px]">
                  <thead className="bg-sky-50/50 text-[9px] font-semibold text-zinc-500 border-b border-sky-100/60">
                    <tr>
                      <th className="px-2 py-1">Series</th>
                      <th className="px-2 py-1">Period</th>
                      <th className="px-2 py-1 text-right">As-Of Value</th>
                      <th className="px-2 py-1 text-right">Latest Value</th>
                      <th className="px-2 py-1 text-right">Revision</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-zinc-100 font-mono">
                    {packet.macro_delta.vintage_walk.map((vw, idx) => (
                      <tr key={idx} className="hover:bg-sky-50/30">
                        <td className="px-2 py-1 font-bold text-sky-950 font-sans">{vw.series_id}</td>
                        <td className="px-2 py-1 text-zinc-500">{vw.observation_date}</td>
                        <td className="px-2 py-1 text-right font-bold text-zinc-800">
                          {vw.asof_value !== null && vw.asof_value !== undefined ? vw.asof_value : 'N/A'}
                        </td>
                        <td className="px-2 py-1 text-right text-zinc-500">
                          {vw.latest_value !== null && vw.latest_value !== undefined ? vw.latest_value : 'N/A'}
                        </td>
                        <td className="px-2 py-1 text-right">
                          {vw.revision_after_cutoff !== null && vw.revision_after_cutoff !== undefined ? (
                            <span className={`inline-block px-1 rounded text-[9px] font-bold ${
                              vw.revision_after_cutoff > 0
                                ? 'bg-emerald-50 text-emerald-700'
                                : vw.revision_after_cutoff < 0
                                ? 'bg-rose-50 text-rose-700'
                                : 'bg-zinc-50 text-zinc-600'
                            }`}>
                              {vw.revision_after_cutoff > 0 ? '+' : ''}{vw.revision_after_cutoff}
                            </span>
                          ) : (
                            <span className="text-zinc-400">-</span>
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>

        {/* Section 3: Thesis Watchdog */}
        <div className="rounded-xl border border-sky-100 bg-sky-50/30 p-3.5 space-y-2.5 shadow-2xs">
          <div className="flex items-center justify-between">
            <span className="text-xs font-bold text-sky-950 flex items-center gap-1.5">
              <span>⚠️ Thesis Watchdog</span>
              <span className="text-[10px] text-zinc-400 font-normal hidden sm:inline">(เฝ้าระวังความเสี่ยงหุ้น)</span>
            </span>
            <span
              className={`rounded-md px-2 py-0.5 text-[10px] font-bold font-mono border ${
                packet.thesis_exceptions.length > 0
                  ? 'bg-rose-100 text-rose-800 border-rose-200'
                  : 'bg-emerald-100 text-emerald-800 border-emerald-200'
              }`}
            >
              {packet.thesis_exceptions.length} ALERTS
            </span>
          </div>

          {/* Thesis Coverage Indicator */}
          {packet.thesis_coverage && packet.thesis_coverage.total_holdings_count > 0 && (
            <div className="rounded-lg bg-white/80 border border-sky-100/70 p-2.5 space-y-1.5">
              <div className="flex items-center justify-between text-[11px]">
                <span className="text-zinc-600 font-medium">ความคุ้มกันพอร์ต (Coverage):</span>
                <span className="font-mono font-bold text-sky-950">
                  {packet.thesis_coverage.covered_count}/{packet.thesis_coverage.total_holdings_count} สินทรัพย์ ({packet.thesis_coverage.coverage_pct.toFixed(0)}%)
                </span>
              </div>
              <div className="h-1.5 w-full bg-zinc-100 rounded-full overflow-hidden">
                <div
                  className="h-full bg-gradient-to-r from-sky-400 to-emerald-400 rounded-full transition-all duration-500"
                  style={{ width: `${Math.min(100, Math.max(5, packet.thesis_coverage.coverage_pct))}%` }}
                />
              </div>

              {packet.thesis_coverage.uncovered_count > 0 && (
                <div className="flex items-center justify-between pt-1 text-[10px] text-zinc-500">
                  <span className="truncate">
                    ยังไม่มี Thesis: <strong className="text-zinc-700">{packet.thesis_coverage.uncovered_tickers.slice(0, 3).join(', ')}{packet.thesis_coverage.uncovered_count > 3 ? '...' : ''}</strong>
                  </span>
                  <span className="text-sky-700 font-semibold shrink-0 ml-1">
                    ({packet.thesis_coverage.uncovered_count} ตัว)
                  </span>
                </div>
              )}
            </div>
          )}

          {packet.thesis_exceptions.length === 0 ? (
            <div className="rounded-lg bg-white/80 border border-sky-100/70 p-2.5 flex items-start gap-2">
              <span className="text-emerald-500 text-xs">🛡️</span>
              <p className="text-[11px] text-zinc-600 leading-relaxed">
                {packet.thesis_coverage && packet.thesis_coverage.covered_count > 0
                  ? `สมมติฐานของ ${packet.thesis_coverage.covered_count} สินทรัพย์ที่มีในระบบอยู่ในเกณฑ์ปกติ (Thesis Intact) ไม่พบเงื่อนไขผิดสมมติฐานในรอบ 24 ชม.`
                  : 'สมมติฐานหุ้นทุกตัวอยู่ในเกณฑ์ปกติ (Thesis Intact) ไม่พบเงื่อนไขผิดสมมติฐานหรือ Falsifier Breach ในรอบ 24 ชม.'}
              </p>
            </div>
          ) : (
            <div className="space-y-1.5">
              {packet.thesis_exceptions.map((exc) => {
                const isCatalyst = Boolean(exc.catalyst_id || (exc as any).classification)
                if (isCatalyst) {
                  const classification = (exc as any).classification || exc.exception_type
                  const entityId = (exc as any).entity_id || exc.ticker
                  const observedFact = (exc as any).observed_fact || exc.metric_name || 'Catalyst change detected'
                  const priorDate = (exc as any).prior_date
                  const newDate = (exc as any).new_date
                  const humanAssessment = (exc as any).human_assessment
                  const status = (exc as any).status || 'REVIEW_REQUIRED'

                  return (
                    <div
                      key={exc.exception_id}
                      data-testid={`catalyst-exception-${exc.exception_id}`}
                      className="rounded-lg border border-amber-200 bg-amber-50/70 p-2.5 text-xs font-sans text-amber-950 flex flex-col gap-1.5 shadow-2xs"
                    >
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-1.5 font-bold font-mono">
                          <span className="text-amber-900">📅 ${entityId}</span>
                          <span
                            className={`rounded px-1.5 py-0.5 text-[9px] font-bold ${
                              classification === 'UNTRUSTED_INSTRUCTION'
                                ? 'bg-rose-100 text-rose-800 border border-rose-300'
                                : classification === 'DATE_CHANGED'
                                ? 'bg-amber-100 text-amber-800 border border-amber-300'
                                : 'bg-sky-100 text-sky-800 border border-sky-300'
                            }`}
                          >
                            {classification}
                          </span>
                        </div>
                        <span className="font-mono text-[9px] text-zinc-500">
                          {status}
                        </span>
                      </div>

                      <p className="text-[11px] text-zinc-700 leading-snug">
                        {observedFact}
                      </p>

                      {(priorDate || newDate) && (
                        <div className="font-mono text-[10px] text-zinc-600 bg-white/80 rounded px-2 py-0.5 border border-amber-100 flex items-center gap-2">
                          <span className="text-zinc-500">Date Transition:</span>
                          <span className="line-through text-zinc-400">{priorDate || 'N/A'}</span>
                          <span>➔</span>
                          <span className="font-bold text-amber-900">{newDate || 'N/A'}</span>
                        </div>
                      )}

                      {humanAssessment ? (
                        <div className="text-[10px] bg-emerald-50 text-emerald-900 rounded p-1.5 border border-emerald-200">
                          <strong>PM Assessment:</strong> {humanAssessment}
                        </div>
                      ) : (
                        <div className="flex items-center gap-1.5 pt-1">
                          {reviewingExceptionId === exc.exception_id ? (
                            <div className="flex flex-col gap-1.5 w-full bg-white p-2 rounded border border-amber-200">
                              <input
                                type="text"
                                placeholder="Record Materiality Assessment (e.g. Non-material shift)..."
                                value={assessmentText}
                                onChange={(e) => setAssessmentText(e.target.value)}
                                className="w-full text-xs p-1.5 border border-zinc-200 rounded"
                                data-testid={`assessment-input-${exc.exception_id}`}
                              />
                              <div className="flex items-center justify-end gap-1.5">
                                <button
                                  type="button"
                                  onClick={() => setReviewingExceptionId(null)}
                                  className="px-2 py-1 text-[10px] text-zinc-600 hover:text-zinc-800"
                                >
                                  Cancel
                                </button>
                                <button
                                  type="button"
                                  disabled={isSavingAssessment}
                                  onClick={() => handleSaveAssessment(exc.exception_id)}
                                  className="px-2.5 py-1 text-[10px] font-bold bg-amber-600 hover:bg-amber-700 text-white rounded"
                                  data-testid={`save-assessment-btn-${exc.exception_id}`}
                                >
                                  {isSavingAssessment ? 'Saving...' : 'Save Assessment'}
                                </button>
                              </div>
                            </div>
                          ) : (
                            <button
                              type="button"
                              onClick={() => {
                                setReviewingExceptionId(exc.exception_id)
                                setAssessmentText('')
                              }}
                              className="text-[10px] font-semibold text-amber-800 hover:text-amber-900 underline"
                              data-testid={`open-assessment-btn-${exc.exception_id}`}
                            >
                              Assess Materiality ✍️
                            </button>
                          )}
                        </div>
                      )}
                    </div>
                  )
                }

                return (
                  <div
                    key={exc.exception_id}
                    className="rounded-lg border border-rose-200 bg-rose-50/80 p-2 text-xs font-mono text-rose-800 flex items-center justify-between"
                  >
                    <span>
                      ${exc.ticker}: {exc.exception_type}
                    </span>
                    {exc.observed_value && (
                      <span className="text-[10px] text-rose-600">({exc.observed_value})</span>
                    )}
                  </div>
                )
              })}
            </div>
          )}
        </div>

        {/* Section 4: PM Actions */}
        <div className="rounded-xl border border-sky-100 bg-sky-50/30 p-3.5 space-y-2.5 shadow-2xs">
          <div className="flex items-center justify-between">
            <span className="text-xs font-bold text-sky-950 flex items-center gap-1.5">
              <span>📋 PM Actions</span>
              <span className="text-[10px] text-zinc-400 font-normal hidden sm:inline">(สิ่งที่ต้องตัดสินใจวันนี้)</span>
            </span>
            <span className="rounded-md bg-zinc-100 border border-zinc-200 px-2 py-0.5 text-[10px] font-bold font-mono text-zinc-700">
              {packet.pending_decisions.length} ITEMS
            </span>
          </div>

          {packet.pending_decisions.length === 0 ? (
            <div className="rounded-lg bg-white/80 border border-sky-100/70 p-2.5 flex items-start gap-2">
              <span className="text-zinc-400 text-xs">✅</span>
              <p className="text-[11px] text-zinc-600 leading-relaxed">
                สัดส่วนพอร์ตยังอยู่ในเกณฑ์สมดุล ไม่มีความจำเป็นต้อง Rebalance ในวันนี้
              </p>
            </div>
          ) : (
            <div className="space-y-2">
              {packet.pending_decisions.map((dec, idx) => {
                const isUrgent = dec.urgency === 'HIGH' || dec.action_code === 'URGENT_REDUCE'
                const isTrim = dec.action_code === 'TRIM_OVERWEIGHT'
                return (
                  <div
                    key={idx}
                    className={`rounded-lg border p-2.5 text-xs transition-colors ${
                      isUrgent
                        ? 'border-rose-200 bg-rose-50/60 text-rose-950'
                        : isTrim
                        ? 'border-amber-200 bg-amber-50/60 text-amber-950'
                        : 'border-sky-200 bg-white/90 text-zinc-800'
                    }`}
                  >
                    <div className="flex items-center justify-between gap-1.5 pb-1">
                      <span className="font-bold text-xs flex items-center gap-1">
                        {isUrgent ? '🚨' : isTrim ? '⚖️' : '🔄'}
                        <span>{dec.ticker ? `$${dec.ticker}` : 'Rebalance'}</span>
                      </span>
                      <div className="flex items-center gap-1">
                        {Boolean(dec.failure_class) && (
                          <span className="rounded bg-zinc-100 border border-zinc-200 px-1 py-0.5 font-mono text-[8px] font-semibold text-zinc-600">
                            {String(dec.failure_class)}
                          </span>
                        )}
                        {dec.action_code && (
                          <span
                            className={`rounded px-1.5 py-0.5 font-mono text-[9px] font-bold ${
                              isUrgent
                                ? 'bg-rose-100 text-rose-800'
                                : isTrim
                                ? 'bg-amber-100 text-amber-800'
                                : 'bg-sky-100 text-sky-800'
                            }`}
                          >
                            {dec.action_code}
                          </span>
                        )}
                      </div>
                    </div>
                    <p className="text-[11px] leading-relaxed text-zinc-600">
                      {dec.action}
                    </p>
                    {Boolean(dec.smallest_repair) && (
                      <div
                        data-testid="smallest-repair-box"
                        className="mt-1.5 rounded border border-edge/60 bg-zinc-50/80 px-2 py-1 text-[10px] text-zinc-700 flex items-start gap-1"
                      >
                        <span className="font-bold text-sky-800">🛠️ Minimal Repair:</span>
                        <span className="text-zinc-600">{String(dec.smallest_repair)}</span>
                      </div>
                    )}
                  </div>
                )
              })}
            </div>
          )}
        </div>

        {/* Audit Fingerprint Footer */}
        <div className="pt-1 text-center">
          <span className="font-mono text-[10px] text-zinc-400">
            Audit Checksum: SHA-256 {packet.packet_sha256.substring(0, 14)}...
            {Boolean(packet.signoff_info?.verified_sample_ids && packet.signoff_info.verified_sample_ids.length > 0) && (
              <span className="ml-2 text-emerald-600 font-bold">
                ✓ {packet.signoff_info?.verified_sample_ids?.length} Samples Audited
              </span>
            )}
          </span>
        </div>
      </div>

      {/* 3. Sign-off Modal */}
      {isSignOffModalOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 backdrop-blur-xs p-4">
          <div className="w-full max-w-md rounded-2xl border border-sky-200 bg-white p-6 shadow-xl space-y-4">
            <div className="flex items-center justify-between">
              <h4 className="text-base font-bold text-sky-950">Acknowledge &amp; Sign-off</h4>
              <button
                type="button"
                onClick={() => setIsSignOffModalOpen(false)}
                className="text-zinc-400 hover:text-zinc-600 text-sm font-bold"
              >
                ✕
              </button>
            </div>
            <p className="text-xs text-zinc-600 leading-relaxed">
              Confirm your review of the 4 quadrants for{' '}
              <strong className="text-sky-900">{packet.review_date}</strong>. This creates an
              immutable audit record with SHA-256 fingerprint{' '}
              <code className="font-mono text-[10px] bg-zinc-100 px-1 py-0.5 rounded">
                {packet.packet_sha256.substring(0, 12)}...
              </code>
            </p>

            {requiredExceptionIds.length > 0 && (
              <div className="space-y-2 rounded-xl border border-rose-100 bg-rose-50/50 p-3">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-bold text-rose-950">
                    Pending Exception Acknowledgements
                  </span>
                  <span className="rounded bg-rose-100 px-1.5 py-0.5 text-[9px] font-mono text-rose-800">
                    {acknowledgedExceptionIds.filter((id) => requiredExceptionIds.includes(id)).length}/{requiredExceptionIds.length}
                  </span>
                </div>
                <div className="space-y-1.5 max-h-40 overflow-y-auto pr-1">
                  {requiredExceptionIds.map((exceptionId) => {
                    const exception = packet.thesis_exceptions.find((row) => row.exception_id === exceptionId)
                    const isChecked = acknowledgedExceptionIds.includes(exceptionId)
                    return (
                      <label
                        key={exceptionId}
                        data-testid="exception-ack-checkbox"
                        className="flex items-start gap-2 rounded-lg border border-rose-200 bg-white p-2 text-xs text-rose-950 cursor-pointer"
                      >
                        <input
                          type="checkbox"
                          checked={isChecked}
                          onChange={() => toggleExceptionAcknowledgement(exceptionId)}
                          className="mt-0.5 rounded border-zinc-300 text-rose-600 focus:ring-rose-500"
                        />
                        <span>
                          <span className="font-mono font-bold">{exceptionId}</span>
                          {' - '}
                          {exception?.ticker ?? 'Unknown'}: {exception?.exception_type ?? 'REVIEW_REQUIRED'}
                        </span>
                      </label>
                    )
                  })}
                </div>
              </div>
            )}

            {/* Risk-Based Review Sampling Checklist (Ch. 9 & App. F, p. 108) */}
            {packet.review_sample && packet.review_sample.length > 0 && (
              <div className="space-y-2 rounded-xl border border-sky-100 bg-sky-50/50 p-3">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-bold text-sky-950 flex items-center gap-1.5">
                    <span>🔍 Institutional Sample Checks</span>
                    <span className="rounded bg-sky-100 px-1.5 py-0.5 text-[9px] font-mono text-sky-800">
                      {verifiedSampleIds.length}/{packet.review_sample.length}
                    </span>
                  </span>
                  <button
                    type="button"
                    onClick={handleVerifyAll}
                    className="text-[10px] font-semibold text-sky-700 hover:text-sky-900 underline"
                  >
                    Verify All
                  </button>
                </div>
                <div className="space-y-1.5 max-h-48 overflow-y-auto pr-1">
                  {packet.review_sample.map((sample) => {
                    const isChecked = verifiedSampleIds.includes(sample.sample_id)
                    const isCritical = sample.urgency === 'CRITICAL'
                    const isMover = sample.sample_type === 'TOP_MOVER'
                    return (
                      <label
                        key={sample.sample_id}
                        data-testid={`sample-item-${sample.sample_id}`}
                        className={`flex items-start gap-2 rounded-lg border p-2 text-xs cursor-pointer transition-colors ${
                          isChecked
                            ? 'border-emerald-200 bg-emerald-50/50 text-emerald-950'
                            : isCritical
                            ? 'border-rose-200 bg-rose-50/40 text-rose-950'
                            : isMover
                            ? 'border-amber-200 bg-amber-50/40 text-amber-950'
                            : 'border-zinc-200 bg-white text-zinc-800'
                        }`}
                      >
                        <input
                          type="checkbox"
                          checked={isChecked}
                          onChange={() => toggleSampleCheck(sample.sample_id)}
                          className="mt-0.5 rounded border-zinc-300 text-sky-600 focus:ring-sky-500"
                        />
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center justify-between gap-1">
                            <span className="font-bold font-mono text-xs">
                              ${sample.symbol}
                            </span>
                            <span
                              className={`rounded px-1.5 py-0.5 font-mono text-[9px] font-bold ${
                                isCritical
                                  ? 'bg-rose-100 text-rose-800'
                                  : isMover
                                  ? 'bg-amber-100 text-amber-800'
                                  : 'bg-zinc-100 text-zinc-700'
                              }`}
                            >
                              {sample.sample_type}
                            </span>
                          </div>
                          <p className="text-[11px] font-medium text-zinc-700 truncate">
                            {sample.headline}
                          </p>
                          <p className="text-[10px] text-zinc-500 font-mono truncate">
                            {sample.target_value}
                          </p>
                        </div>
                      </label>
                    )
                  })}
                </div>
              </div>
            )}

            <div className="space-y-1.5">
              <label className="text-xs font-bold text-zinc-700">Optional Notes / Rationale</label>
              <textarea
                value={signOffNotes}
                onChange={(e) => setSignOffNotes(e.target.value)}
                placeholder="e.g. Acknowledged NVDA margin drift; maintain overweight stance."
                rows={3}
                className="w-full rounded-xl border border-sky-200 p-2.5 text-xs text-zinc-900 focus:outline-none focus:ring-2 focus:ring-sky-400"
              />
            </div>
            <div className="flex justify-end gap-2.5 pt-2">
              <button
                type="button"
                onClick={() => setIsSignOffModalOpen(false)}
                className="rounded-xl px-4 py-2 text-xs font-semibold text-zinc-600 hover:bg-zinc-100"
              >
                Cancel
              </button>
              <button
                type="button"
                disabled={
                  isSubmitting ||
                  ((packet.review_sample?.length ?? 0) > 0 && verifiedSampleIds.length !== (packet.review_sample?.length ?? 0)) ||
                  requiredExceptionIds.some((id) => !acknowledgedExceptionIds.includes(id))
                }
                onClick={handleSignOff}
                className="rounded-xl bg-sky-600 px-4 py-2 text-xs font-bold text-white hover:bg-sky-700 disabled:opacity-50 shadow-sm"
              >
                {isSubmitting
                  ? 'Signing off...'
                  : packet.review_sample && packet.review_sample.length > 0
                  ? `Confirm Sign-off (${verifiedSampleIds.length}/${packet.review_sample.length})`
                  : 'Confirm Sign-off'}
              </button>
            </div>
          </div>
        </div>
      )}
    </section>
  )
}

export default MorningDeltaCard
