import { useState } from 'react'
import type { ActualHoldingDTO, ActualSummaryDTO, PerformanceSnapshotDTO } from '../../api/types'
import { partitionPriceRefreshInfo, type PvdValuationContext } from '../../lib/pvdPresentation'
import { PortfolioSummaryCardsSkeleton } from '../ui/FlowSkeleton'

interface Props {
  summary: ActualSummaryDTO | null
  lastUpdated: string | null
  loading?: boolean
  refreshingPrices?: boolean
  priceRefreshInfo?: Record<string, string> | null
  onRefreshPrices?: () => void
  onViewPvdStatements?: () => void
  pvdContext?: PvdValuationContext
  holdings?: ActualHoldingDTO[]
  performanceRows?: PerformanceSnapshotDTO[]
}
import { formatTHB } from '../../utils/formatters'

/**
 * ตรวจสอบว่าผลการอัปเดตราคาของสินทรัพย์/FX สำเร็จหรือไม่
 * สำเร็จ: มีราคาตัวเลข (เช่น "146.92 USD", "32.9100 (live)"), "ok", "updated", "cached"
 * ล้มเหลว: มีคำว่า "fail", "error", "timeout", "no_data"
 */
export function isPriceRefreshSuccess(status: string | undefined | null): boolean {
  if (!status) return false
  const lower = status.toLowerCase()
  if (lower.trimStart().startsWith('statement nav')) return false
  if (
    lower.includes('fail') ||
    lower.includes('error') ||
    lower.includes('timeout') ||
    lower.includes('no_data')
  ) {
    return false
  }
  return (
    lower.includes('ok') ||
    lower.includes('updated') ||
    lower.includes('cached') ||
    lower.includes('live') ||
    lower.includes('fallback') ||
    /\d+(\.\d+)?/.test(status)
  )
}

function getPvdDateLabel(context: PvdValuationContext): string {
  const { dates, hasUnknownDate, latestRecordedStatementDate } = context
  if (dates.length > 1) {
    const range = `${dates[0]}–${dates[dates.length - 1]}`
    return hasUnknownDate ? `ข้อมูลวันที่ PVD บางรายการไม่ครบ · วันที่ที่ทราบ ${range}` : `PVD มีหลายวัน Statement · ${range}`
  }
  if (dates.length === 1) {
    return hasUnknownDate ? `ข้อมูลวันที่ PVD บางรายการไม่ครบ · วันที่ที่ทราบ ${dates[0]}` : `PVD ตาม Statement ณ ${dates[0]}`
  }
  if (hasUnknownDate) {
    return `ข้อมูลวันที่ PVD บางรายการไม่ครบ${latestRecordedStatementDate ? ` · Statement ล่าสุดที่บันทึก ${latestRecordedStatementDate}` : ''}`
  }
  if (latestRecordedStatementDate) return `Statement ล่าสุดที่บันทึก ${latestRecordedStatementDate}`
  return 'ไม่ทราบวันที่ Statement'
}


export default function PortfolioSummaryCards({
  summary,
  lastUpdated,
  loading = false,
  refreshingPrices = false,
  priceRefreshInfo = null,
  onRefreshPrices,
  onViewPvdStatements,
  pvdContext,
  holdings = [],
  performanceRows = [],
}: Props) {
  const [showDetails, setShowDetails] = useState(false)
  if (loading) {
    return <PortfolioSummaryCardsSkeleton />
  }
  if (!summary) {
    return null
  }

  const nav = summary.total_value_thb
  const unrealizedPnL = summary.total_unrealized_profit
  const costBasis = summary.total_cost_basis_thb ?? 0
  const pnlPct = costBasis > 0 ? (unrealizedPnL / costBasis) * 100 : null
  const isPositive = unrealizedPnL >= 0
  const hasPvd = pvdContext?.hasPvd ?? false
  const hasMarketHoldings = pvdContext?.hasMarketHoldings ?? true
  const hasNonPvdValue = pvdContext?.hasNonPvdValue ?? !hasPvd
  const hasUnknownPvdValuation = pvdContext?.hasUnknownValuation ?? false
  const pvdOnly = hasPvd && !hasNonPvdValue
  const pvdDateLabel = hasPvd && pvdContext ? getPvdDateLabel(pvdContext) : null
  const refreshPartition = partitionPriceRefreshInfo(priceRefreshInfo, holdings)
  const marketRefreshEntries = refreshPartition.marketEntries
  const pvdRefreshEntries = refreshPartition.pvdEntries
  const hasUnavailablePvdStatus = pvdRefreshEntries.some(([, status]) => /unavailable|no_data|fail|error/i.test(status))

  // เตรียม Sparkline สำหรับ Total NAV
  const sortedSpark = [...performanceRows]
    .sort((a, b) => a.Date.localeCompare(b.Date))
    .slice(-15) // เอา 15 จุดล่าสุดมาทำ Sparkline

  let sparkPath = ''
  let sparkArea = ''
  let isSparkPositive = true

  if (sortedSpark.length >= 2) {
    const navs = sortedSpark.map((r) => r.Total_NAV)
    const minNav = Math.min(...navs)
    const maxNav = Math.max(...navs)
    const range = maxNav - minNav || 1
    const width = 110
    const height = 26
    const padTop = 3
    const padBottom = 3

    const points = navs.map((val, idx) => {
      const x = (idx / (navs.length - 1)) * width
      const y = height - padBottom - ((val - minNav) / range) * (height - padTop - padBottom)
      return { x, y }
    })

    sparkPath = points.map((p, i) => `${i === 0 ? 'M' : 'L'} ${p.x.toFixed(1)} ${p.y.toFixed(1)}`).join(' ')
    sparkArea = `${sparkPath} L ${points[points.length - 1]!.x.toFixed(1)} ${height} L 0 ${height} Z`
    isSparkPositive = navs[navs.length - 1]! >= navs[0]!
  }

  return (
    <div className="grid grid-cols-1 gap-3.5 sm:grid-cols-2 xl:grid-cols-4">
      {/* Card 1: Total NAV */}
      <div className="flex flex-col justify-between rounded-2xl border border-sky-100 bg-gradient-to-br from-panel via-panel to-sky-50/40 p-4 sm:p-5 shadow-sm transition-all hover:shadow-md relative overflow-hidden">
        <div className="flex items-center justify-between text-xs sm:text-sm font-bold uppercase tracking-wider text-zinc-500">
          <span className="truncate">Total Portfolio NAV</span>
          <span className="rounded-full bg-flow-cyan/10 px-2 py-0.5 text-xs font-bold text-flow-blue shrink-0">THB</span>
        </div>
        <div className="mt-2 flex items-baseline justify-between gap-2 overflow-hidden">
          <div
            className="text-xl sm:text-2xl 2xl:text-3xl font-extrabold font-sans tabular-nums tracking-tight text-zinc-900 truncate"
            title={formatTHB(nav)}
          >
            {formatTHB(nav)}
          </div>
          {/* Sparkline SVG */}
          {sortedSpark.length >= 2 && (
            <div className="h-7 w-20 sm:w-24 shrink-0">
              <svg viewBox="0 0 110 26" className="h-full w-full overflow-visible">
                <defs>
                  <linearGradient id="navSparkGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor={isSparkPositive ? '#059669' : '#e11d48'} stopOpacity={0.25} />
                    <stop offset="100%" stopColor={isSparkPositive ? '#059669' : '#e11d48'} stopOpacity={0.0} />
                  </linearGradient>
                </defs>
                <path d={sparkArea} fill="url(#navSparkGrad)" />
                <path
                  d={sparkPath}
                  stroke={isSparkPositive ? '#059669' : '#e11d48'}
                  strokeWidth={2}
                  fill="none"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              </svg>
            </div>
          )}
        </div>
        <div className="mt-2 flex items-center justify-between text-xs text-zinc-500 font-medium">
          <span className="truncate">{hasPvd ? 'มูลค่าล่าสุดที่บันทึก (รวมเงินสด)' : 'มูลค่าสุทธิปัจจุบัน (รวมเงินสด)'}</span>
          {sortedSpark.length >= 2 && (
            <span className={`font-sans tabular-nums text-xs font-bold shrink-0 ${isSparkPositive ? 'text-emerald-600' : 'text-rose-600'}`}>
              {isSparkPositive ? '▲' : '▼'} 15D
            </span>
          )}
        </div>
        {pvdDateLabel && <div className="mt-1 text-[11px] leading-snug text-teal-800">{pvdDateLabel}</div>}
      </div>

      {/* Card 2: Total Unrealized Profit */}
      <div className="flex flex-col justify-between rounded-2xl border border-sky-100 bg-gradient-to-br from-panel via-panel to-sky-50/40 p-4 sm:p-5 shadow-sm transition-all hover:shadow-md overflow-hidden">
        <div className="flex items-center justify-between text-xs sm:text-sm font-bold uppercase tracking-wider text-zinc-500">
          <span className="truncate">{hasUnknownPvdValuation ? 'ผลต่างมูลค่ารวม' : pvdOnly ? 'ผลประโยชน์สะสมตาม Statement' : hasPvd ? 'ผลประโยชน์สะสมรวม' : 'Unrealized Profit/Loss'}</span>
          <span
            className={`rounded-full px-2 py-0.5 text-xs font-extrabold font-sans tabular-nums shrink-0 ${
              pnlPct === null || hasUnknownPvdValuation
                ? 'bg-zinc-100 text-zinc-500'
                : isPositive
                  ? 'bg-emerald-50 text-emerald-700 border border-emerald-200'
                  : 'bg-rose-50 text-rose-700 border border-rose-200'
            }`}
          >
            {pnlPct === null ? '—' : `${isPositive ? '+' : ''}${pnlPct.toFixed(2)}%`}
          </span>
        </div>
        <div
            className={`mt-2 text-xl sm:text-2xl 2xl:text-3xl font-extrabold font-sans tabular-nums tracking-tight truncate ${
            hasUnknownPvdValuation ? 'text-zinc-700' : isPositive ? 'text-emerald-600' : 'text-rose-600'
          }`}
          title={`${isPositive ? '+' : ''}${formatTHB(unrealizedPnL)}`}
        >
          {isPositive ? '+' : ''}{formatTHB(unrealizedPnL)}
        </div>
        <div className="mt-2 flex items-center text-xs text-zinc-500 font-medium truncate">
          <span>
            {hasUnknownPvdValuation
              ? 'ข้อมูล PVD ไม่ครบ จึงยังยืนยันผลประโยชน์ตาม Statement ไม่ได้'
              : pvdOnly
                ? `เทียบเงินต้นสะสม · ไม่ใช่ผลตอบแทนรายปี${pvdContext?.hasUnknownDate ? ' · วันที่ Statement บางรายการไม่ครบ' : ''}`
                : hasPvd
                  ? 'เทียบต้นทุนรวมพอร์ตตามสูตรเดิม · ไม่ใช่ผลตอบแทนรายปี'
                  : 'กำไร/ขาดทุนที่ยังไม่รับรู้'}
          </span>
        </div>
      </div>

      {/* Card 3: Passive Income YTD */}
      <div className="flex flex-col justify-between rounded-2xl border border-sky-100 bg-gradient-to-br from-panel via-panel to-sky-50/40 p-4 sm:p-5 shadow-sm transition-all hover:shadow-md overflow-hidden">
        <div className="flex items-center justify-between text-xs sm:text-sm font-bold uppercase tracking-wider text-zinc-500">
          <span className="truncate">Passive Income YTD</span>
          <span className="rounded-full bg-amber-50 border border-amber-200 px-2 py-0.5 text-xs font-bold text-amber-700 shrink-0">
            เงินปันผลสะสม
          </span>
        </div>
        <div
          className="mt-2 text-xl sm:text-2xl 2xl:text-3xl font-extrabold font-sans tabular-nums tracking-tight text-zinc-900 truncate"
          title={formatTHB(summary.total_accumulated_dividend || summary.passive_income_ytd || 0)}
        >
          {formatTHB(summary.total_accumulated_dividend || summary.passive_income_ytd || 0)}
        </div>
        <div className="mt-2 flex items-center justify-between text-xs text-zinc-500 font-medium">
          <span className="truncate">เงินปันผลสะสมทั้งหมด</span>
          {summary.passive_income_ytd > 0 && summary.total_accumulated_dividend !== summary.passive_income_ytd && (
            <span className="text-[11px] text-zinc-400 font-mono shrink-0">
              (YTD: {formatTHB(summary.passive_income_ytd)})
            </span>
          )}
        </div>
      </div>

      {/* Card 4: Last Updated & Refresh */}
      <div className="flex flex-col justify-between rounded-2xl border border-sky-100 bg-gradient-to-br from-panel via-panel to-sky-50/40 p-4 sm:p-5 shadow-sm transition-all hover:shadow-md overflow-hidden">
        <div className="flex items-center justify-between text-xs sm:text-sm font-bold uppercase tracking-wider text-zinc-500">
          <span className="truncate">{hasPvd ? 'Portfolio Data Status' : 'Market Prices Status'}</span>
          {refreshingPrices && (!hasPvd || hasMarketHoldings) && (
            <span className="inline-flex items-center gap-1 rounded-full bg-sky-100 px-2 py-0.5 text-[11px] font-bold text-sky-800 animate-pulse shrink-0">
              Refreshing...
            </span>
          )}
        </div>
        <div className="mt-2 flex flex-col justify-center overflow-hidden">
          <span
            className="text-xs sm:text-sm font-bold font-sans tabular-nums text-zinc-800 truncate"
            title={lastUpdated ? new Date(lastUpdated).toLocaleString('th-TH') : 'ไม่ทราบ'}
          >
            {lastUpdated ? new Date(lastUpdated).toLocaleString('th-TH') : 'ไม่ทราบ'}
          </span>
          <span className="mt-0.5 text-[11px] sm:text-xs text-zinc-500 font-medium">บันทึกพอร์ตล่าสุด · ไม่ใช่เวลาที่ผู้ให้บริการส่งราคา</span>
          {hasPvd && pvdDateLabel && <span className="mt-1 text-[11px] leading-snug text-teal-800">{pvdDateLabel}</span>}
        </div>
        <div className="mt-2 pt-1">
          {hasPvd && !hasMarketHoldings ? (
            <button
              type="button"
              onClick={onViewPvdStatements}
              disabled={!onViewPvdStatements}
              className="flex w-full items-center justify-center gap-1.5 rounded-xl border border-teal-200 bg-teal-50 px-3 py-1.5 text-xs font-bold text-teal-800 transition-all hover:bg-teal-100 disabled:cursor-not-allowed disabled:opacity-50"
            >
              ดู/อัปเดต Statement PVD
            </button>
          ) : (
            <>
              <button
                type="button"
                onClick={onRefreshPrices}
                disabled={refreshingPrices}
                title="ดึงราคาตลาดล่าสุดของหุ้น/กองทุน พร้อมบันทึก Performance Snapshot ประจำวันนี้"
                className="flex w-full items-center justify-center gap-1.5 rounded-xl border border-sky-200 bg-flow-cyan/10 px-3 py-1.5 text-xs font-bold text-flow-blue transition-all hover:bg-flow-cyan/20 disabled:cursor-not-allowed disabled:opacity-50 shadow-xs"
              >
                <svg
                  className={`h-3.5 w-3.5 shrink-0 ${refreshingPrices ? 'animate-spin' : ''}`}
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                  strokeWidth={2}
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    d="M16.023 9.348h4.992v-.001M2.985 19.644v-4.992m0 0h4.992m-4.993 0l3.181 3.183a8.25 8.25 0 0013.803-3.7M4.031 9.865a8.25 8.25 0 0113.803-3.7l3.181 3.182m0-4.991v4.99"
                  />
                </svg>
                {refreshingPrices ? 'กำลังอัปเดตราคา & Snapshot...' : 'อัปเดตราคาตลาด & Snapshot วันนี้'}
              </button>
              {hasPvd && (
                <div className="mt-1.5 text-center">
                  <button type="button" onClick={onViewPvdStatements} className="text-[11px] font-semibold text-teal-800 underline disabled:opacity-50" disabled={!onViewPvdStatements}>
                    ดู/อัปเดต Statement PVD
                  </button>
                  <p className="mt-0.5 text-[10px] text-zinc-500">การ refresh ราคาตลาดไม่เปลี่ยนยอด PVD</p>
                </div>
              )}
            </>
          )}
        </div>
        {hasUnavailablePvdStatus && (
          <p role="status" className="mt-2 rounded-lg border border-amber-200 bg-amber-50 px-2 py-1.5 text-[10px] font-medium text-amber-800">
            ไม่พบมูลค่า Statement PVD สำหรับบางรายการ
          </p>
        )}
        {(() => {
          const refreshEntries = marketRefreshEntries
          const totalRefreshCount = refreshEntries.length
          if (totalRefreshCount === 0) return null

          const failedRefreshEntries = refreshEntries.filter(
            ([_, status]) => !isPriceRefreshSuccess(status)
          )
          const isAllRefreshOk = failedRefreshEntries.length === 0

          return (
            <div className="mt-2.5 rounded-xl border border-zinc-200/80 bg-zinc-50/80 p-2 text-xs transition-all">
              <div className="flex items-center justify-between">
                {isAllRefreshOk ? (
                  <span className="inline-flex items-center gap-1.5 font-medium text-emerald-700">
                    <span className="h-2 w-2 rounded-full bg-emerald-500 animate-pulse" />
                    อัปเดตราคาสำเร็จ ({totalRefreshCount} รายการ) & บันทึก Snapshot แล้ว
                  </span>
                ) : (
                  <span className="inline-flex items-center gap-1.5 font-medium text-amber-700">
                    <span className="h-2 w-2 rounded-full bg-amber-500 animate-pulse" />
                    สำเร็จ {totalRefreshCount - failedRefreshEntries.length}/{totalRefreshCount} รายการ
                  </span>
                )}
                <button
                  type="button"
                  onClick={() => setShowDetails(!showDetails)}
                  className="text-[11px] font-semibold text-zinc-500 hover:text-zinc-800 transition-colors"
                >
                  {showDetails ? 'ซ่อน' : 'รายละเอียด'}
                </button>
              </div>

              {!isAllRefreshOk && !showDetails && (
                <div className="mt-1.5 space-y-0.5 border-t border-zinc-200/60 pt-1 text-[10px] font-mono text-amber-700">
                  {failedRefreshEntries.map(([sym, status]) => (
                    <div key={sym} className="flex justify-between items-center">
                      <span className="font-bold">{sym}:</span>
                      <span>{status}</span>
                    </div>
                  ))}
                </div>
              )}

              {showDetails && (
                <div className="mt-1.5 max-h-28 overflow-y-auto space-y-0.5 border-t border-zinc-200/60 pt-1.5 text-[10px] font-mono tabular-nums">
                  {refreshEntries.map(([sym, status]) => {
                    const isOk = isPriceRefreshSuccess(status)
                    return (
                      <div key={sym} className="flex justify-between items-center">
                        <span className="font-bold text-zinc-700">{sym}:</span>
                        <span className={isOk ? 'text-emerald-600 font-medium' : 'text-amber-600 font-semibold'}>
                          {status}
                        </span>
                      </div>
                    )
                  })}
                </div>
              )}
            </div>
          )
        })()}
      </div>
    </div>
  )
}
