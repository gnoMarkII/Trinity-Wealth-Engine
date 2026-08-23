import { useState, useMemo } from 'react'
import type { FinancialSummaryChartPointDTO } from '../../api/types'

interface FinancialSummaryChartProps {
  data: FinancialSummaryChartPointDTO[]
  currency?: string
}

type ChartMetricFilter = 'all' | 'revenue_profit' | 'cash_flow' | 'margins'

export function FinancialSummaryChart({ data, currency = '$' }: FinancialSummaryChartProps) {
  const [hoveredIdx, setHoveredIdx] = useState<number | null>(null)
  const [metricFilter, setMetricFilter] = useState<ChartMetricFilter>('all')

  const currencySymbol = currency === 'THB' ? '฿' : '$'

  // Reverse data so chronological order is left (older) to right (latest)
  const chronological = useMemo(() => {
    if (!data || data.length === 0) return []
    return [...data].reverse()
  }, [data])

  // Latest summary stats for top KPI mini-cards
  const latestPoint = chronological[chronological.length - 1]

  // Scaling factor calculation
  const maxVal = useMemo(() => {
    if (chronological.length === 0) return 1
    return Math.max(
      ...chronological.map((d) =>
        Math.max(
          Math.abs(d.revenue || 0),
          Math.abs(d.gross_profit || 0),
          Math.abs(d.net_income || 0),
          Math.abs(d.free_cash_flow || 0)
        )
      ),
      1
    )
  }, [chronological])

  let scaleDivisor = 1
  let scaleSuffix = ''
  if (maxVal >= 1_000_000_000) {
    scaleDivisor = 1_000_000_000
    scaleSuffix = 'B'
  } else if (maxVal >= 1_000_000) {
    scaleDivisor = 1_000_000
    scaleSuffix = 'M'
  } else if (maxVal >= 1_000) {
    scaleDivisor = 1_000
    scaleSuffix = 'K'
  }

  const formatScaledVal = (val: number | null | undefined, forceFull = false) => {
    if (val === null || val === undefined) return '-'
    if (forceFull) {
      return `${currencySymbol}${val.toLocaleString(undefined, { maximumFractionDigits: 0 })}`
    }
    const scaled = val / scaleDivisor
    return `${currencySymbol}${scaled.toLocaleString(undefined, {
      minimumFractionDigits: Math.abs(scaled) >= 10 ? 1 : 2,
      maximumFractionDigits: 2,
    })}${scaleSuffix ? ` ${scaleSuffix}` : ''}`
  }

  if (chronological.length === 0) {
    return (
      <div className="flex items-center justify-center h-48 bg-panel border border-edge rounded-2xl text-muted text-sm shadow-sm">
        No financial trend data available
      </div>
    )
  }

  // SVG dimensions
  const svgWidth = 840
  const svgHeight = 240
  const paddingLeft = 65
  const paddingRight = 55
  const paddingTop = 28
  const paddingBottom = 40

  const chartWidth = svgWidth - paddingLeft - paddingRight
  const chartHeight = svgHeight - paddingTop - paddingBottom

  // Value bounds for bars
  const minValY = Math.min(
    0,
    ...chronological.map((d) =>
      Math.min(
        metricFilter === 'cash_flow' ? (d.free_cash_flow || 0) : (d.net_income || 0),
        d.free_cash_flow || 0
      )
    )
  )
  const maxValY = Math.max(
    1,
    ...chronological.map((d) =>
      Math.max(
        metricFilter === 'cash_flow' ? (d.free_cash_flow || 0) : (d.revenue || 0),
        d.gross_profit || 0
      )
    )
  )

  const valRange = maxValY - minValY || 1
  const zeroY = paddingTop + chartHeight - ((0 - minValY) / valRange) * chartHeight

  const getY = (val: number) => {
    return paddingTop + chartHeight - ((val - minValY) / valRange) * chartHeight
  }

  // Margins scale (0% - 100%)
  const margins = chronological
    .map((d) => d.operating_margin_pct)
    .filter((m): m is number => m !== null && m !== undefined)
  const maxMargin = margins.length > 0 ? Math.max(40, Math.ceil(Math.max(...margins) / 10) * 10) : 40
  const minMargin = margins.length > 0 ? Math.min(0, Math.floor(Math.min(...margins) / 10) * 10) : 0
  const marginRange = maxMargin - minMargin || 1

  const getMarginY = (pct: number) => {
    return paddingTop + chartHeight - ((pct - minMargin) / marginRange) * chartHeight
  }

  // Slot calculations
  const numSlots = chronological.length
  const slotWidth = chartWidth / numSlots

  // Bar dimensions based on active filter
  const showRevenue = metricFilter === 'all' || metricFilter === 'revenue_profit'
  const showNetIncome = metricFilter === 'all' || metricFilter === 'revenue_profit'
  const showFCF = metricFilter === 'all' || metricFilter === 'cash_flow'
  const showMarginLine = metricFilter === 'all' || metricFilter === 'margins'

  const activeBarCount = (showRevenue ? 1 : 0) + (showNetIncome ? 1 : 0) + (showFCF ? 1 : 0)
  const barGroupWidth = Math.min(slotWidth * 0.72, activeBarCount * 18 + 12)
  const barWidth = activeBarCount > 0 ? barGroupWidth / activeBarCount : 0

  // Margin Line Points
  const marginPoints = chronological
    .map((d, i) => {
      if (d.operating_margin_pct === null || d.operating_margin_pct === undefined) return null
      const cx = paddingLeft + i * slotWidth + slotWidth / 2
      const cy = getMarginY(d.operating_margin_pct)
      return { x: cx, y: cy, pct: d.operating_margin_pct, period: d.period_key }
    })
    .filter((p): p is { x: number; y: number; pct: number; period: string } => p !== null)

  const marginPath = marginPoints.reduce((acc, p, idx) => {
    return idx === 0 ? `M ${p.x} ${p.y}` : `${acc} L ${p.x} ${p.y}`
  }, '')

  return (
    <div className="bg-panel border border-edge/80 rounded-2xl p-5 shadow-sm space-y-4 transition-all">
      {/* Top Header: KPI Highlights & Controls */}
      <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4 border-b border-edge/60 pb-4">
        {/* KPI Mini-Cards */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 flex-1">
          <div className="bg-surface/60 border border-edge/60 rounded-xl p-3">
            <span className="text-[11px] font-medium text-muted block mb-0.5">Latest Revenue</span>
            <div className="text-base font-bold text-primary font-mono tabular-nums tracking-tight">
              {formatScaledVal(latestPoint?.revenue)}
            </div>
          </div>

          <div className="bg-surface/60 border border-edge/60 rounded-xl p-3">
            <span className="text-[11px] font-medium text-muted block mb-0.5">Net Income</span>
            <div className={`text-base font-bold font-mono tabular-nums tracking-tight ${(latestPoint?.net_income || 0) >= 0 ? 'text-emerald-500' : 'text-rose-500'}`}>
              {formatScaledVal(latestPoint?.net_income)}
            </div>
          </div>

          <div className="bg-surface/60 border border-edge/60 rounded-xl p-3">
            <span className="text-[11px] font-medium text-muted block mb-0.5">Free Cash Flow</span>
            <div className={`text-base font-bold font-mono tabular-nums tracking-tight ${(latestPoint?.free_cash_flow || 0) >= 0 ? 'text-indigo-400' : 'text-rose-400'}`}>
              {formatScaledVal(latestPoint?.free_cash_flow)}
            </div>
          </div>

          <div className="bg-surface/60 border border-edge/60 rounded-xl p-3">
            <span className="text-[11px] font-medium text-muted block mb-0.5">Operating Margin</span>
            <div className="text-base font-bold text-amber-500 font-mono tabular-nums tracking-tight">
              {latestPoint?.operating_margin_pct !== null && latestPoint?.operating_margin_pct !== undefined
                ? `${latestPoint.operating_margin_pct}%`
                : '-'}
            </div>
          </div>
        </div>

        {/* View Metric Filter Pills */}
        <div className="flex items-center gap-1 bg-surface p-1 rounded-xl border border-edge text-xs self-start lg:self-center">
          <button
            onClick={() => setMetricFilter('all')}
            className={`px-2.5 py-1.5 rounded-lg font-medium transition-all ${
              metricFilter === 'all'
                ? 'bg-panel text-primary shadow-sm border border-edge'
                : 'text-muted hover:text-primary'
            }`}
          >
            All Trends
          </button>
          <button
            onClick={() => setMetricFilter('revenue_profit')}
            className={`px-2.5 py-1.5 rounded-lg font-medium transition-all ${
              metricFilter === 'revenue_profit'
                ? 'bg-panel text-primary shadow-sm border border-edge'
                : 'text-muted hover:text-primary'
            }`}
          >
            Revenue & Profit
          </button>
          <button
            onClick={() => setMetricFilter('cash_flow')}
            className={`px-2.5 py-1.5 rounded-lg font-medium transition-all ${
              metricFilter === 'cash_flow'
                ? 'bg-panel text-primary shadow-sm border border-edge'
                : 'text-muted hover:text-primary'
            }`}
          >
            Free Cash Flow
          </button>
          <button
            onClick={() => setMetricFilter('margins')}
            className={`px-2.5 py-1.5 rounded-lg font-medium transition-all ${
              metricFilter === 'margins'
                ? 'bg-panel text-primary shadow-sm border border-edge'
                : 'text-muted hover:text-primary'
            }`}
          >
            Margins
          </button>
        </div>
      </div>

      {/* Legend */}
      <div className="flex flex-wrap items-center justify-between gap-2 text-xs">
        <div className="flex items-center gap-2">
          <span className="font-semibold text-primary text-xs">Quarterly Multi-Period Trend</span>
          <span className="text-[11px] text-muted font-mono">
            ({scaleSuffix ? `Scaled in ${currencySymbol}${scaleSuffix}` : `in ${currencySymbol}`})
          </span>
        </div>

        <div className="flex flex-wrap items-center gap-4 text-xs font-medium">
          {showRevenue && (
            <div className="flex items-center gap-1.5">
              <span className="w-3 h-3 rounded-md bg-gradient-to-t from-blue-600 to-cyan-400 shadow-sm" />
              <span className="text-muted">Revenue</span>
            </div>
          )}
          {showNetIncome && (
            <div className="flex items-center gap-1.5">
              <span className="w-3 h-3 rounded-md bg-gradient-to-t from-emerald-600 to-emerald-400 shadow-sm" />
              <span className="text-muted">Net Income</span>
            </div>
          )}
          {showFCF && (
            <div className="flex items-center gap-1.5" title="Standard bar displays Calculated GAAP FCF (OCF - CapEx). Hover card displays 8-K Company-Reported Non-GAAP comparison if adjustments exist.">
              <span className="w-3 h-3 rounded-md bg-gradient-to-t from-indigo-600 to-purple-400 shadow-sm" />
              <span className="text-muted">Calculated FCF (GAAP)</span>
            </div>
          )}
          {showMarginLine && (
            <div className="flex items-center gap-1.5">
              <span className="w-3 h-1 bg-amber-400 rounded-full" />
              <span className="w-2 h-2 rounded-full bg-amber-400 -ml-2.5 border-2 border-white dark:border-zinc-900" />
              <span className="text-muted">Op. Margin %</span>
            </div>
          )}
        </div>
      </div>

      {/* Main SVG Chart */}
      <div className="w-full overflow-x-auto">
        <svg
          viewBox={`0 0 ${svgWidth} ${svgHeight}`}
          className="w-full h-auto min-w-[600px] select-none"
          onMouseLeave={() => setHoveredIdx(null)}
        >
          {/* Gradients */}
          <defs>
            <linearGradient id="revGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#38bdf8" stopOpacity="0.95" />
              <stop offset="100%" stopColor="#2563eb" stopOpacity="0.85" />
            </linearGradient>
            <linearGradient id="netGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#34d399" stopOpacity="0.95" />
              <stop offset="100%" stopColor="#059669" stopOpacity="0.85" />
            </linearGradient>
            <linearGradient id="fcfGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#a78bfa" stopOpacity="0.95" />
              <stop offset="100%" stopColor="#6366f1" stopOpacity="0.85" />
            </linearGradient>
            <linearGradient id="lossGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#f87171" stopOpacity="0.95" />
              <stop offset="100%" stopColor="#dc2626" stopOpacity="0.85" />
            </linearGradient>
            <filter id="glow" x="-20%" y="-20%" width="140%" height="140%">
              <feDropShadow dx="0" dy="2" stdDeviation="3" floodColor="#f59e0b" floodOpacity="0.3" />
            </filter>
          </defs>

          {/* Grid lines */}
          {[0, 0.25, 0.5, 0.75, 1.0].map((frac) => {
            const yVal = minValY + frac * valRange
            const yPos = getY(yVal)
            return (
              <g key={frac}>
                <line
                  x1={paddingLeft}
                  y1={yPos}
                  x2={svgWidth - paddingRight}
                  y2={yPos}
                  stroke="currentColor"
                  className="text-edge/40"
                  strokeWidth="0.75"
                  strokeDasharray={frac === 0 || yVal === 0 ? undefined : '3 3'}
                />
                <text
                  x={paddingLeft - 10}
                  y={yPos + 3.5}
                  textAnchor="end"
                  fontSize="10"
                  className="fill-muted font-mono"
                >
                  {currencySymbol}
                  {(yVal / scaleDivisor).toFixed(Math.abs(yVal / scaleDivisor) >= 10 ? 0 : 1)}
                  {scaleSuffix}
                </text>
              </g>
            )
          })}

          {/* Right Y Axis: Margin % */}
          {showMarginLine && (
            <>
              <text
                x={svgWidth - paddingRight + 10}
                y={getMarginY(maxMargin) + 3.5}
                textAnchor="start"
                fontSize="10"
                className="fill-amber-500 font-mono font-medium"
              >
                {maxMargin}%
              </text>
              <text
                x={svgWidth - paddingRight + 10}
                y={getMarginY(minMargin) + 3.5}
                textAnchor="start"
                fontSize="10"
                className="fill-amber-500 font-mono font-medium"
              >
                {minMargin}%
              </text>
            </>
          )}

          {/* Zero baseline */}
          <line
            x1={paddingLeft}
            y1={zeroY}
            x2={svgWidth - paddingRight}
            y2={zeroY}
            stroke="currentColor"
            className="text-edge"
            strokeWidth="1.2"
          />

          {/* Bars */}
          {chronological.map((point, idx) => {
            const slotCenterX = paddingLeft + idx * slotWidth + slotWidth / 2
            const groupStartX = slotCenterX - barGroupWidth / 2

            let currentOffset = 0

            // Revenue Bar
            const rev = point.revenue || 0
            const revY = getY(rev)
            const revH = Math.max(Math.abs(zeroY - revY), 2)
            const revTop = rev >= 0 ? revY : zeroY
            const revX = groupStartX + currentOffset * barWidth
            if (showRevenue) currentOffset++

            // Net Income Bar
            const netInc = point.net_income || 0
            const netY = getY(netInc)
            const netH = Math.max(Math.abs(zeroY - netY), 2)
            const netTop = netInc >= 0 ? netY : zeroY
            const netX = groupStartX + currentOffset * barWidth
            if (showNetIncome) currentOffset++

            // FCF Bar
            const fcf = point.free_cash_flow || 0
            const fcfY = getY(fcf)
            const fcfH = Math.max(Math.abs(zeroY - fcfY), 2)
            const fcfTop = fcf >= 0 ? fcfY : zeroY
            const fcfX = groupStartX + currentOffset * barWidth

            const isHovered = hoveredIdx === idx

            return (
              <g
                key={point.period_key}
                onMouseEnter={() => setHoveredIdx(idx)}
                className="cursor-pointer transition-opacity"
              >
                {/* Background Column Highlight */}
                {isHovered && (
                  <rect
                    x={paddingLeft + idx * slotWidth}
                    y={paddingTop}
                    width={slotWidth}
                    height={chartHeight}
                    fill="currentColor"
                    className="text-primary/5"
                    rx="8"
                  />
                )}

                {/* Revenue */}
                {showRevenue && (
                  <rect
                    x={revX}
                    y={revTop}
                    width={Math.max(barWidth - 2, 2)}
                    height={revH}
                    fill="url(#revGrad)"
                    rx="3"
                    className="transition-all hover:opacity-100"
                    opacity={isHovered ? 1 : 0.9}
                  />
                )}

                {/* Net Income */}
                {showNetIncome && (
                  <rect
                    x={netX}
                    y={netTop}
                    width={Math.max(barWidth - 2, 2)}
                    height={netH}
                    fill={netInc >= 0 ? 'url(#netGrad)' : 'url(#lossGrad)'}
                    rx="3"
                    className="transition-all hover:opacity-100"
                    opacity={isHovered ? 1 : 0.9}
                  />
                )}

                {/* Free Cash Flow */}
                {showFCF && (
                  <rect
                    x={fcfX}
                    y={fcfTop}
                    width={Math.max(barWidth - 2, 2)}
                    height={fcfH}
                    fill={fcf >= 0 ? 'url(#fcfGrad)' : 'url(#lossGrad)'}
                    rx="3"
                    className="transition-all hover:opacity-100"
                    opacity={isHovered ? 1 : 0.9}
                  />
                )}

                {/* X Axis Period Label */}
                <text
                  x={slotCenterX}
                  y={svgHeight - 14}
                  textAnchor="middle"
                  fontSize="11"
                  className={
                    isHovered
                      ? 'fill-primary font-bold transition-all'
                      : 'fill-muted font-mono font-medium'
                  }
                >
                  {point.period_key}
                </text>
              </g>
            )
          })}

          {/* Margin Line */}
          {showMarginLine && marginPath && (
            <path
              d={marginPath}
              fill="none"
              stroke="#f59e0b"
              strokeWidth="2.5"
              strokeLinecap="round"
              strokeLinejoin="round"
              filter="url(#glow)"
              className="pointer-events-none"
            />
          )}

          {/* Margin Line Nodes */}
          {showMarginLine &&
            marginPoints.map((p, idx) => {
              const isHovered = hoveredIdx === idx
              return (
                <g key={idx} className="pointer-events-none">
                  <circle
                    cx={p.x}
                    cy={p.y}
                    r={isHovered ? 5.5 : 3.5}
                    fill="#f59e0b"
                    stroke="#ffffff"
                    strokeWidth="2"
                    className="transition-all"
                  />
                </g>
              )
            })}
        </svg>
      </div>

      {/* Floating Hover Card */}
      {hoveredIdx !== null && chronological[hoveredIdx] && (
        <div className="p-3.5 bg-panel/95 backdrop-blur-md border border-edge rounded-xl shadow-xl flex flex-wrap items-center justify-between gap-4 text-xs animate-fade-in">
          <div className="flex items-center gap-2">
            <span className="font-bold text-primary text-sm font-mono">
              {chronological[hoveredIdx].period_key}
            </span>
            <span className="text-muted text-[11px] font-mono">
              ({chronological[hoveredIdx].date})
            </span>
          </div>

          <div className="flex flex-wrap items-center gap-5">
            <span className="text-blue-500 font-mono flex items-center gap-1.5">
              <span className="w-2 h-2 rounded-full bg-blue-500" />
              <span>Revenue:</span>
              <strong className="text-primary tabular-nums">
                {formatScaledVal(chronological[hoveredIdx].revenue)}
              </strong>
            </span>

            <span className="text-emerald-500 font-mono flex items-center gap-1.5">
              <span className="w-2 h-2 rounded-full bg-emerald-500" />
              <span>Net Income:</span>
              <strong className="text-primary tabular-nums">
                {formatScaledVal(chronological[hoveredIdx].net_income)}
              </strong>
            </span>

            <span className="text-indigo-400 font-mono flex items-center gap-1.5">
              <span className="w-2 h-2 rounded-full bg-indigo-400" />
              <span>FCF:</span>
              <strong className="text-primary tabular-nums">
                {formatScaledVal(chronological[hoveredIdx].free_cash_flow)}
              </strong>
              {chronological[hoveredIdx].reported_free_cash_flow !== null &&
                chronological[hoveredIdx].reported_free_cash_flow !== undefined &&
                chronological[hoveredIdx].reported_free_cash_flow !== chronological[hoveredIdx].free_cash_flow && (
                  <span className="text-purple-400 text-[10px]" title="Company-Reported Non-GAAP FCF from 8-K">
                    (Rep: {formatScaledVal(chronological[hoveredIdx].reported_free_cash_flow)})
                  </span>
                )}
            </span>

            <span className="text-amber-500 font-mono flex items-center gap-1.5">
              <span className="w-2 h-2 rounded-full bg-amber-500" />
              <span>Op. Margin:</span>
              <strong className="text-primary tabular-nums">
                {chronological[hoveredIdx].operating_margin_pct !== null &&
                chronological[hoveredIdx].operating_margin_pct !== undefined
                  ? `${chronological[hoveredIdx].operating_margin_pct}%`
                  : '-'}
              </strong>
            </span>
          </div>
        </div>
      )}
    </div>
  )
}
