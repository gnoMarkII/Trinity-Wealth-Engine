import { useState } from 'react'
import type { ActualSummaryDTO, PerformanceSnapshotDTO } from '../../api/types'

interface Props {
  summary: ActualSummaryDTO | null
  lastUpdated: string | null
  loading?: boolean
  refreshingPrices?: boolean
  priceRefreshInfo?: Record<string, string> | null
  onRefreshPrices?: () => void
  performanceRows?: PerformanceSnapshotDTO[]
}
import { formatTHB } from '../../utils/formatters'


export default function PortfolioSummaryCards({
  summary,
  lastUpdated,
  loading = false,
  refreshingPrices = false,
  priceRefreshInfo = null,
  onRefreshPrices,
  performanceRows = [],
}: Props) {
  const [showDetails, setShowDetails] = useState(false)
  if (loading || !summary) {
    return (
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {[1, 2, 3, 4].map((i) => (
          <div key={i} className="h-28 animate-pulse rounded-2xl border border-sky-100 bg-panel/60 p-5 shadow-sm" />
        ))}
      </div>
    )
  }

  const nav = summary.total_value_thb
  const unrealizedPnL = summary.total_unrealized_profit
  const costBasis = summary.total_cost_basis_thb ?? 0
  const pnlPct = costBasis > 0 ? (unrealizedPnL / costBasis) * 100 : null
  const isPositive = unrealizedPnL >= 0

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
    <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
      {/* Card 1: Total NAV */}
      <div className="flex flex-col justify-between rounded-2xl border border-sky-100 bg-gradient-to-br from-panel via-panel to-sky-50/40 p-5 sm:p-6 shadow-sm transition-all hover:shadow-md relative overflow-hidden">
        <div className="flex items-center justify-between text-xs sm:text-sm font-bold uppercase tracking-wider text-zinc-500">
          <span>Total Portfolio NAV</span>
          <span className="rounded-full bg-flow-cyan/10 px-2.5 py-0.5 text-xs font-bold text-flow-blue">THB</span>
        </div>
        <div className="mt-2.5 flex items-baseline justify-between gap-2">
          <div className="text-2xl sm:text-3xl lg:text-4xl font-extrabold font-sans tabular-nums tracking-tight text-zinc-900">
            {formatTHB(nav)}
          </div>
          {/* Sparkline SVG */}
          {sortedSpark.length >= 2 && (
            <div className="h-8 w-28 shrink-0">
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
          <span>มูลค่าสุทธิปัจจุบัน (รวมเงินสด)</span>
          {sortedSpark.length >= 2 && (
            <span className={`font-sans tabular-nums text-xs font-bold ${isSparkPositive ? 'text-emerald-600' : 'text-rose-600'}`}>
              {isSparkPositive ? '▲' : '▼'} 15D Trend
            </span>
          )}
        </div>
      </div>

      {/* Card 2: Total Unrealized Profit */}
      <div className="flex flex-col justify-between rounded-2xl border border-sky-100 bg-gradient-to-br from-panel via-panel to-sky-50/40 p-5 sm:p-6 shadow-sm transition-all hover:shadow-md">
        <div className="flex items-center justify-between text-xs sm:text-sm font-bold uppercase tracking-wider text-zinc-500">
          <span>Unrealized Profit/Loss</span>
          <span
            className={`rounded-full px-2.5 py-0.5 text-xs font-extrabold font-sans tabular-nums ${
              pnlPct === null
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
          className={`mt-2.5 text-2xl sm:text-3xl lg:text-4xl font-extrabold font-sans tabular-nums tracking-tight ${
            isPositive ? 'text-emerald-600' : 'text-rose-600'
          }`}
        >
          {isPositive ? '+' : ''}
          {formatTHB(unrealizedPnL)}
        </div>
        <div className="mt-2 flex items-center text-xs text-zinc-500 font-medium">
          <span>กำไร/ขาดทุนที่ยังไม่รับรู้</span>
        </div>
      </div>

      {/* Card 3: Passive Income YTD */}
      <div className="flex flex-col justify-between rounded-2xl border border-sky-100 bg-gradient-to-br from-panel via-panel to-sky-50/40 p-5 sm:p-6 shadow-sm transition-all hover:shadow-md">
        <div className="flex items-center justify-between text-xs sm:text-sm font-bold uppercase tracking-wider text-zinc-500">
          <span>Passive Income YTD</span>
          <span className="rounded-full bg-amber-50 border border-amber-200 px-2.5 py-0.5 text-xs font-bold text-amber-700">
            เงินปันผลสะสม
          </span>
        </div>
        <div className="mt-2.5 text-2xl sm:text-3xl lg:text-4xl font-extrabold font-sans tabular-nums tracking-tight text-zinc-900">
          {formatTHB(summary.total_accumulated_dividend || summary.passive_income_ytd || 0)}
        </div>
        <div className="mt-2 flex items-center justify-between text-xs text-zinc-500 font-medium">
          <span>เงินปันผลสะสมทั้งหมด</span>
          {summary.passive_income_ytd > 0 && summary.total_accumulated_dividend !== summary.passive_income_ytd && (
            <span className="text-[11px] text-zinc-400 font-mono">
              (YTD: {formatTHB(summary.passive_income_ytd)})
            </span>
          )}
        </div>
      </div>

      {/* Card 4: Last Updated & Refresh */}
      <div className="flex flex-col justify-between rounded-2xl border border-sky-100 bg-gradient-to-br from-panel via-panel to-sky-50/40 p-5 sm:p-6 shadow-sm transition-all hover:shadow-md">
        <div className="flex items-center justify-between text-xs sm:text-sm font-bold uppercase tracking-wider text-zinc-500">
          <span>Market Prices Status</span>
          {refreshingPrices && (
            <span className="inline-flex items-center gap-1 rounded-full bg-sky-100 px-2.5 py-0.5 text-xs font-bold text-sky-800 animate-pulse">
              Refreshing...
            </span>
          )}
        </div>
        <div className="mt-2 flex flex-col justify-center">
          <span className="text-sm sm:text-base font-bold font-sans tabular-nums text-zinc-800">
            {lastUpdated ? new Date(lastUpdated).toLocaleString('th-TH') : 'N/A'}
          </span>
          <span className="mt-0.5 text-xs text-zinc-500 font-medium">อัปเดตราคาล่าสุดจาก yfinance</span>
        </div>
        <div className="mt-2 pt-1">
          <button
            type="button"
            onClick={onRefreshPrices}
            disabled={refreshingPrices}
            className="flex w-full items-center justify-center gap-2 rounded-xl border border-sky-200 bg-flow-cyan/10 px-3.5 py-2 text-xs sm:text-sm font-bold text-flow-blue transition-all hover:bg-flow-cyan/20 disabled:cursor-not-allowed disabled:opacity-50 shadow-sm"
          >
            <svg
              className={`h-4 w-4 ${refreshingPrices ? 'animate-spin' : ''}`}
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
            {refreshingPrices ? 'กำลังอัปเดตราคา...' : 'อัปเดตราคาตลาด (Refresh)'}
          </button>
        </div>
        {(() => {
          if (!priceRefreshInfo) return null
          const refreshEntries = Object.entries(priceRefreshInfo)
          const totalRefreshCount = refreshEntries.length
          if (totalRefreshCount === 0) return null

          const failedRefreshEntries = refreshEntries.filter(
            ([_, status]) => !status.toLowerCase().includes('ok') && !status.toLowerCase().includes('updated')
          )
          const isAllRefreshOk = failedRefreshEntries.length === 0

          return (
            <div className="mt-2.5 rounded-xl border border-zinc-200/80 bg-zinc-50/80 p-2 text-xs transition-all">
              <div className="flex items-center justify-between">
                {isAllRefreshOk ? (
                  <span className="inline-flex items-center gap-1.5 font-medium text-emerald-700">
                    <span className="h-2 w-2 rounded-full bg-emerald-500 animate-pulse" />
                    อัปเดตราคาสำเร็จ ({totalRefreshCount} รายการ)
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
                    const isOk = status.toLowerCase().includes('ok') || status.toLowerCase().includes('updated')
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
