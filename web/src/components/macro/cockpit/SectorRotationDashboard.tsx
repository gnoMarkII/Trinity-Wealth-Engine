import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { api } from '../../../api/client'
import type { SectorAnalysisDTO, SectorRotationHistoryDTO, SectorRotationResponseDTO, SectorRotationRowDTO } from '../../../api/types'
import { SourceProvenanceBadge } from './SourceProvenanceBadge'

// Quadrant configurations with rich HSL/curated palettes
const QUADRANT_CONFIG = {
  Leading: {
    label: 'Leading',
    thLabel: 'นำตลาด',
    description: 'แนวโน้มแกร่ง + โมเมนตัมพุ่ง (Outperforming)',
    color: '#059669', // emerald-600
    bgLight: 'bg-emerald-50/90',
    border: 'border-emerald-200',
    text: 'text-emerald-700',
    dot: 'bg-emerald-500',
    quadrantBg: 'rgba(16, 185, 129, 0.08)',
  },
  Weakening: {
    label: 'Weakening',
    thLabel: 'ชะลอตัว',
    description: 'แนวโน้มดี แต่โมเมนตัมเริ่มแผ่ว (At Risk)',
    color: '#d97706', // amber-600
    bgLight: 'bg-amber-50/90',
    border: 'border-amber-200',
    text: 'text-amber-700',
    dot: 'bg-amber-500',
    quadrantBg: 'rgba(245, 158, 11, 0.07)',
  },
  Lagging: {
    label: 'Lagging',
    thLabel: 'ตามหลัง',
    description: 'แนวโน้มอ่อน + โมเมนตัมต่ำ (Underperforming)',
    color: '#e11d48', // rose-600
    bgLight: 'bg-rose-50/90',
    border: 'border-rose-200',
    text: 'text-rose-700',
    dot: 'bg-rose-500',
    quadrantBg: 'rgba(244, 63, 94, 0.07)',
  },
  Improving: {
    label: 'Improving',
    thLabel: 'ฟื้นตัว',
    description: 'แนวโน้มต่ำ แต่โมเมนตัมเริ่มฟื้น (Potential Leaders)',
    color: '#0284c7', // sky-600
    bgLight: 'bg-sky-50/90',
    border: 'border-sky-200',
    text: 'text-sky-700',
    dot: 'bg-sky-500',
    quadrantBg: 'rgba(14, 165, 233, 0.07)',
  },
} as const

type QuadrantKey = keyof typeof QUADRANT_CONFIG

const SECTOR_INFO: Record<string, { th: string; icon: string; category: string }> = {
  XLK: { th: 'เทคโนโลยี', icon: '💻', category: 'Technology' },
  XLC: { th: 'สื่อสาร & มีเดีย', icon: '📡', category: 'Communication Services' },
  XLV: { th: 'สาธารณสุข & ยา', icon: '🏥', category: 'Healthcare' },
  XLE: { th: 'พลังงาน & น้ำมัน', icon: '⚡', category: 'Energy' },
  XLI: { th: 'อุตสาหกรรม', icon: '🏭', category: 'Industrials' },
  XLP: { th: 'สินค้าจำเป็น', icon: '🛒', category: 'Consumer Staples' },
  XLU: { th: 'สาธารณูปโภค', icon: '💡', category: 'Utilities' },
  XLY: { th: 'สินค้าฟุ่มเฟือย', icon: '🛍️', category: 'Consumer Discretionary' },
  XLRE: { th: 'อสังหาริมทรัพย์', icon: '🏢', category: 'Real Estate' },
  XLF: { th: 'การเงิน & ธนาคาร', icon: '🏦', category: 'Financials' },
  XLB: { th: 'วัสดุอุตสาหกรรม', icon: '🧱', category: 'Materials' },
}

function formatMetric(value: number | null | undefined, unit: '%' | ' pp' = '%') {
  if (value == null || !Number.isFinite(value)) return '—'
  return `${value > 0 ? '+' : ''}${value.toFixed(2)}${unit}`
}

function metricTitle(row: SectorRotationRowDTO, horizon: string) {
  const metric = row.return_metrics[horizon]
  if (!metric) return 'Unavailable'
  return `${metric.start_date || '—'} to ${metric.end_date || '—'} · ${metric.status} · ${metric.freshness}`
}

function metricQualityLabel(row: SectorRotationRowDTO, horizon: string) {
  const metric = row.return_metrics[horizon]
  if (!metric || metric.status === 'unavailable') return 'no data'
  if (metric.freshness !== 'fresh') return metric.freshness
  return metric.status === 'partial' ? 'partial' : ''
}

interface SectorMapProps {
  rows: SectorRotationRowDTO[]
  selectedTicker: string
  hoveredTicker: string | null
  onSelect: (ticker: string) => void
  onHover: (ticker: string | null) => void
  zoomMode: 'auto' | '3' | '6' | '10' | '20'
  showTrails: boolean
  activeQuadrantFilter: string | null
}

function SectorMap({
  rows,
  selectedTicker,
  hoveredTicker,
  onSelect,
  onHover,
  zoomMode,
  showTrails,
  activeQuadrantFilter,
}: SectorMapProps) {
  const [tooltipData, setTooltipData] = useState<{
    row: SectorRotationRowDTO
    x: number
    y: number
  } | null>(null)

  const plot = { left: 68, right: 600, top: 32, bottom: 420 }
  const centerX = (plot.left + plot.right) / 2
  const centerY = (plot.top + plot.bottom) / 2

  // 1. Calculate dynamic symmetrical domain centered strictly at 100
  const maxDeviation = useMemo(() => {
    let maxDev = 2.0
    rows.forEach((r) => {
      if (r.relative_trend != null) maxDev = Math.max(maxDev, Math.abs(r.relative_trend - 100))
      if (r.relative_momentum != null) maxDev = Math.max(maxDev, Math.abs(r.relative_momentum - 100))
      if (showTrails) {
        r.history?.forEach((h) => {
          if (h.relative_trend != null) maxDev = Math.max(maxDev, Math.abs(h.relative_trend - 100))
          if (h.relative_momentum != null) maxDev = Math.max(maxDev, Math.abs(h.relative_momentum - 100))
        })
      }
    })
    return maxDev
  }, [rows, showTrails])

  const effectiveRadius = useMemo(() => {
    if (zoomMode === '3') return 3
    if (zoomMode === '6') return 6
    if (zoomMode === '10') return 10
    if (zoomMode === '20') return 20
    // 'auto': add ~35% headroom, rounded nicely to 1 decimal place, minimum 2.5
    return Math.max(Math.ceil(maxDeviation * 1.35 * 10) / 10, 2.5)
  }, [zoomMode, maxDeviation])

  const minVal = 100 - effectiveRadius
  const maxVal = 100 + effectiveRadius

  const scaleX = useCallback(
    (val: number) => {
      const clamped = Math.min(maxVal, Math.max(minVal, val))
      return plot.left + ((clamped - minVal) / (maxVal - minVal)) * (plot.right - plot.left)
    },
    [minVal, maxVal, plot.left, plot.right]
  )

  const scaleY = useCallback(
    (val: number) => {
      const clamped = Math.min(maxVal, Math.max(minVal, val))
      return plot.bottom - ((clamped - minVal) / (maxVal - minVal)) * (plot.bottom - plot.top)
    },
    [minVal, maxVal, plot.bottom, plot.top]
  )

  return (
    <div className="relative overflow-x-auto rounded-xl bg-slate-900/[0.02]">
      <svg
        viewBox="0 0 648 468"
        role="img"
        aria-label="US sector relative rotation map"
        className="min-w-[540px] w-full select-none"
        onMouseLeave={() => {
          setTooltipData(null)
          onHover(null)
        }}
      >
        <defs>
          {/* Subtle quadrant gradients */}
          <linearGradient id="grad-improving" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="#0284c7" stopOpacity="0.08" />
            <stop offset="100%" stopColor="#f0f9ff" stopOpacity="0.3" />
          </linearGradient>
          <linearGradient id="grad-leading" x1="100%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%" stopColor="#059669" stopOpacity="0.12" />
            <stop offset="100%" stopColor="#ecfdf5" stopOpacity="0.3" />
          </linearGradient>
          <linearGradient id="grad-lagging" x1="0%" y1="100%" x2="100%" y2="0%">
            <stop offset="0%" stopColor="#e11d48" stopOpacity="0.09" />
            <stop offset="100%" stopColor="#fff1f2" stopOpacity="0.3" />
          </linearGradient>
          <linearGradient id="grad-weakening" x1="100%" y1="100%" x2="0%" y2="0%">
            <stop offset="0%" stopColor="#d97706" stopOpacity="0.09" />
            <stop offset="100%" stopColor="#fffbeb" stopOpacity="0.3" />
          </linearGradient>
          {/* Glow filter for selected/hovered dots */}
          <filter id="glow-halo" x="-50%" y="-50%" width="200%" height="200%">
            <feGaussianBlur stdDeviation="3.5" result="coloredBlur" />
            <feMerge>
              <feMergeNode in="coloredBlur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
        </defs>

        {/* 1. Quadrant Background Rectangles */}
        <rect
          x={plot.left}
          y={plot.top}
          width={centerX - plot.left}
          height={centerY - plot.top}
          fill="url(#grad-improving)"
          className="transition-opacity duration-300"
          opacity={!activeQuadrantFilter || activeQuadrantFilter === 'Improving' ? 1 : 0.3}
        />
        <rect
          x={centerX}
          y={plot.top}
          width={plot.right - centerX}
          height={centerY - plot.top}
          fill="url(#grad-leading)"
          className="transition-opacity duration-300"
          opacity={!activeQuadrantFilter || activeQuadrantFilter === 'Leading' ? 1 : 0.3}
        />
        <rect
          x={plot.left}
          y={centerY}
          width={centerX - plot.left}
          height={plot.bottom - centerY}
          fill="url(#grad-lagging)"
          className="transition-opacity duration-300"
          opacity={!activeQuadrantFilter || activeQuadrantFilter === 'Lagging' ? 1 : 0.3}
        />
        <rect
          x={centerX}
          y={centerY}
          width={plot.right - centerX}
          height={plot.bottom - centerY}
          fill="url(#grad-weakening)"
          className="transition-opacity duration-300"
          opacity={!activeQuadrantFilter || activeQuadrantFilter === 'Weakening' ? 1 : 0.3}
        />

        {/* 2. Subdued Grid Lines */}
        <line
          x1={centerX}
          y1={plot.top}
          x2={centerX}
          y2={plot.bottom}
          stroke="#94a3b8"
          strokeDasharray="4 4"
          strokeWidth="1.2"
        />
        <line
          x1={plot.left}
          y1={centerY}
          x2={plot.right}
          y2={centerY}
          stroke="#94a3b8"
          strokeDasharray="4 4"
          strokeWidth="1.2"
        />

        {/* 3. Quadrant Corner Badges */}
        <g className="pointer-events-none select-none">
          {/* Improving */}
          <rect x={plot.left + 8} y={plot.top + 8} width="84" height="20" rx="4" fill="#0284c7" fillOpacity="0.12" />
          <text x={plot.left + 50} y={plot.top + 22} textAnchor="middle" fill="#0369a1" fontSize="10" fontWeight="700" letterSpacing="0.05em">
            IMPROVING
          </text>
          {/* Leading */}
          <rect x={plot.right - 92} y={plot.top + 8} width="84" height="20" rx="4" fill="#059669" fillOpacity="0.12" />
          <text x={plot.right - 50} y={plot.top + 22} textAnchor="middle" fill="#047857" fontSize="10" fontWeight="700" letterSpacing="0.05em">
            LEADING
          </text>
          {/* Lagging */}
          <rect x={plot.left + 8} y={plot.bottom - 28} width="84" height="20" rx="4" fill="#e11d48" fillOpacity="0.12" />
          <text x={plot.left + 50} y={plot.bottom - 14} textAnchor="middle" fill="#be123c" fontSize="10" fontWeight="700" letterSpacing="0.05em">
            LAGGING
          </text>
          {/* Weakening */}
          <rect x={plot.right - 96} y={plot.bottom - 28} width="88" height="20" rx="4" fill="#d97706" fillOpacity="0.12" />
          <text x={plot.right - 52} y={plot.bottom - 14} textAnchor="middle" fill="#b45309" fontSize="10" fontWeight="700" letterSpacing="0.05em">
            WEAKENING
          </text>
        </g>

        {/* Center Origin Crosshair Badge */}
        <circle cx={centerX} cy={centerY} r="3" fill="#64748b" />
        <rect x={centerX - 16} y={centerY - 10} width="32" height="13" rx="3" fill="#ffffff" stroke="#cbd5e1" strokeWidth="1" />
        <text x={centerX} y={centerY} textAnchor="middle" dominantBaseline="middle" fontSize="8.5" fontWeight="600" fill="#64748b">
          100
        </text>

        {/* 4. Sector Rotation Trails & Current Points */}
        {rows.map((row) => {
          if (row.relative_trend == null || row.relative_momentum == null) return null
          const isSelected = selectedTicker === row.ticker
          const isHovered = hoveredTicker === row.ticker
          const isDimmed =
            (hoveredTicker != null && !isHovered && !isSelected) ||
            (activeQuadrantFilter != null && row.quadrant !== activeQuadrantFilter)

          const qConfig = QUADRANT_CONFIG[row.quadrant as QuadrantKey] || {
            color: '#64748b',
            dot: 'bg-slate-500',
          }
          const color = qConfig.color
          const x = scaleX(row.relative_trend)
          const y = scaleY(row.relative_momentum)

          // Outward angle for label placement to avoid center clumping
          const isRight = x >= centerX
          const isTop = y <= centerY
          const labelX = isRight ? x + 9 : x - 9
          const labelY = isTop ? y - 6 : y + 14
          const textAnchor = isRight ? 'start' : 'end'

          return (
            <g
              key={row.ticker}
              role="button"
              tabIndex={0}
              aria-label={`Select ${row.ticker} (${row.name})`}
              onClick={() => onSelect(row.ticker)}
              onMouseEnter={() => {
                onHover(row.ticker)
                setTooltipData({ row, x, y })
              }}
              onMouseLeave={() => {
                onHover(null)
                setTooltipData(null)
              }}
              onKeyDown={(event) => {
                if (event.key === 'Enter' || event.key === ' ') onSelect(row.ticker)
              }}
              className="cursor-pointer transition-opacity duration-200"
              opacity={isDimmed ? 0.25 : 1}
            >
              {/* Historical Trail */}
              {showTrails &&
                row.history &&
                row.history.length > 1 &&
                row.history.slice(0, -1).map((point, index) => {
                  const next = row.history[index + 1]
                  if (
                    !next ||
                    point.relative_trend == null ||
                    point.relative_momentum == null ||
                    next.relative_trend == null ||
                    next.relative_momentum == null
                  )
                    return null

                  const x1 = scaleX(point.relative_trend)
                  const y1 = scaleY(point.relative_momentum)
                  const x2 = scaleX(next.relative_trend)
                  const y2 = scaleY(next.relative_momentum)

                  // Comet trail: older segments are fainter and thinner
                  const progress = (index + 1) / row.history.length
                  const segOpacity = isSelected || isHovered ? 0.35 + progress * 0.65 : 0.15 + progress * 0.55
                  const segWidth = isSelected || isHovered ? 1.5 + progress * 1.5 : 1 + progress * 1

                  return (
                    <g key={`${row.ticker}-seg-${point.as_of}`}>
                      <line
                        x1={x1}
                        y1={y1}
                        x2={x2}
                        y2={y2}
                        stroke={color}
                        strokeOpacity={segOpacity}
                        strokeWidth={segWidth}
                        strokeLinecap="round"
                      />
                      {/* Waypoint micro-dot */}
                      <circle cx={x1} cy={y1} r={1.5} fill={color} fillOpacity={segOpacity * 0.8} />
                    </g>
                  )
                })}

              {/* Halo pulse when selected or hovered */}
              {(isSelected || isHovered) && (
                <circle
                  cx={x}
                  cy={y}
                  r={14}
                  fill={color}
                  fillOpacity="0.2"
                  className="animate-pulse"
                />
              )}

              {/* Latest Position Dot */}
              <circle
                cx={x}
                cy={y}
                r={isSelected || isHovered ? 8 : 5.5}
                fill={color}
                stroke="#ffffff"
                strokeWidth={isSelected || isHovered ? 2.5 : 1.8}
                filter={isSelected || isHovered ? 'url(#glow-halo)' : undefined}
              />

              {/* Ticker Label Badge with subtle stroke backdrop */}
              <g transform={`translate(${labelX}, ${labelY})`}>
                <text
                  x="0"
                  y="0"
                  textAnchor={textAnchor}
                  fontSize={isSelected || isHovered ? '11.5' : '10.5'}
                  fontWeight={isSelected || isHovered ? '800' : '700'}
                  fill={isSelected || isHovered ? '#0f172a' : '#334155'}
                  stroke="#ffffff"
                  strokeWidth="3"
                  strokeLinejoin="round"
                  paintOrder="stroke fill"
                  className="font-mono tracking-tight"
                >
                  {row.ticker}
                </text>
              </g>
            </g>
          )
        })}

        {/* 5. Axis Labels & Scale Indicators */}
        <text x={centerX} y={plot.bottom + 34} textAnchor="middle" fontSize="11" fontWeight="600" fill="#475569">
          Relative Trend (RS-Ratio) →
        </text>
        <text
          x="16"
          y={centerY}
          textAnchor="middle"
          fontSize="11"
          fontWeight="600"
          fill="#475569"
          transform={`rotate(-90 16 ${centerY})`}
        >
          Relative Momentum (RS-Momentum) →
        </text>

        {/* X-axis Ticks */}
        <g className="text-slate-400 select-none" fontSize="9.5" fill="#64748b" textAnchor="middle">
          <text x={plot.left} y={plot.bottom + 16}>
            {minVal.toFixed(1)}
          </text>
          <text x={(plot.left + centerX) / 2} y={plot.bottom + 16}>
            {((minVal + 100) / 2).toFixed(1)}
          </text>
          <text x={centerX} y={plot.bottom + 16} fontWeight="700" fill="#0f172a">
            100.0
          </text>
          <text x={(centerX + plot.right) / 2} y={plot.bottom + 16}>
            {((100 + maxVal) / 2).toFixed(1)}
          </text>
          <text x={plot.right} y={plot.bottom + 16}>
            {maxVal.toFixed(1)}
          </text>
        </g>

        {/* Y-axis Ticks */}
        <g className="text-slate-400 select-none" fontSize="9.5" fill="#64748b" textAnchor="end">
          <text x={plot.left - 8} y={plot.top + 4}>
            {maxVal.toFixed(1)}
          </text>
          <text x={plot.left - 8} y={(plot.top + centerY) / 2 + 3}>
            {((100 + maxVal) / 2).toFixed(1)}
          </text>
          <text x={plot.left - 8} y={centerY + 3} fontWeight="700" fill="#0f172a">
            100.0
          </text>
          <text x={plot.left - 8} y={(centerY + plot.bottom) / 2 + 3}>
            {((minVal + 100) / 2).toFixed(1)}
          </text>
          <text x={plot.left - 8} y={plot.bottom + 3}>
            {minVal.toFixed(1)}
          </text>
        </g>
      </svg>

      {/* Floating Rich Tooltip */}
      {tooltipData && (
        <div
          className="pointer-events-none absolute z-20 rounded-xl border border-slate-200/90 bg-white/95 p-3 shadow-xl backdrop-blur-md transition-all duration-150"
          style={{
            left: `${Math.min(Math.max(tooltipData.x - 70, 10), 440)}px`,
            top: `${Math.max(tooltipData.y - 120, 10)}px`,
            minWidth: '200px',
          }}
        >
          <div className="flex items-center justify-between gap-2 border-b border-slate-100 pb-1.5">
            <div className="flex items-center gap-1.5">
              <span className="text-base">{SECTOR_INFO[tooltipData.row.ticker]?.icon || '📊'}</span>
              <span className="font-bold text-slate-900">{tooltipData.row.ticker}</span>
              <span className="text-xs text-slate-500">
                {SECTOR_INFO[tooltipData.row.ticker]?.th || tooltipData.row.name}
              </span>
            </div>
            {tooltipData.row.quadrant && (
              <span
                className={`rounded-full px-2 py-0.5 text-[10px] font-bold ${
                  QUADRANT_CONFIG[tooltipData.row.quadrant as QuadrantKey]?.bgLight
                } ${QUADRANT_CONFIG[tooltipData.row.quadrant as QuadrantKey]?.text}`}
              >
                {tooltipData.row.quadrant}
              </span>
            )}
          </div>
          <div className="mt-2 grid grid-cols-2 gap-2 text-xs">
            <div>
              <span className="text-[10px] text-slate-400">Relative Trend</span>
              <div className="font-mono font-bold text-slate-800">
                {tooltipData.row.relative_trend?.toFixed(2) ?? '—'}
              </div>
            </div>
            <div>
              <span className="text-[10px] text-slate-400">Relative Momentum</span>
              <div className="font-mono font-bold text-slate-800">
                {tooltipData.row.relative_momentum?.toFixed(2) ?? '—'}
              </div>
            </div>
            <div>
              <span className="text-[10px] text-slate-400">1M Excess vs SPY</span>
              <div
                className={`font-mono font-bold ${
                  (tooltipData.row.returns_pct['1M_excess_pp'] ?? 0) >= 0 ? 'text-emerald-600' : 'text-rose-600'
                }`}
              >
                {formatMetric(tooltipData.row.returns_pct['1M_excess_pp'], ' pp')}
              </div>
            </div>
            <div>
              <span className="text-[10px] text-slate-400">1M Absolute</span>
              <div
                className={`font-mono font-semibold ${
                  (tooltipData.row.returns_pct['1M_absolute_pct'] ?? 0) >= 0 ? 'text-emerald-600' : 'text-rose-600'
                }`}
              >
                {formatMetric(tooltipData.row.returns_pct['1M_absolute_pct'])}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

function RelativePriceChart({
  points,
  baseDate,
  ticker,
}: {
  points: Array<{ as_of: string; sector_spy_rebased_100: number | null; status?: 'available' | 'unavailable'; reason?: string | null }>
  baseDate: string | null
  ticker: string
}) {
  const [hoverIndex, setHoverIndex] = useState<number | null>(null)

  const availablePoints = points.map((point, index) => ({ point, index })).filter(({ point }) => point.sector_spy_rebased_100 != null)
  if (!points || availablePoints.length < 2) {
    return (
      <div className="flex h-32 items-center justify-center rounded-xl bg-slate-50 text-xs text-slate-400">
        ข้อมูลประวัติราคาเปรียบเทียบยังไม่เพียงพอสำหรับกราฟ
      </div>
    )
  }

  const firstPoint = points[0]
  const lastPoint = availablePoints[availablePoints.length - 1]?.point
  if (!firstPoint || !lastPoint) {
    return null
  }

  const values = availablePoints.map(({ point }) => point.sector_spy_rebased_100 as number)
  const low = Math.min(100, ...values)
  const high = Math.max(100, ...values)
  const spread = Math.max(high - low, 1.2)
  const floor = low - spread * 0.1
  const ceiling = high + spread * 0.1

  const width = 640
  const height = 160
  const padding = { left: 40, right: 30, top: 20, bottom: 30 }

  const scaleX = (index: number) =>
    padding.left + (index / (points.length - 1)) * (width - padding.left - padding.right)
  const scaleY = (val: number) =>
    padding.top + (1 - (val - floor) / (ceiling - floor)) * (height - padding.top - padding.bottom)

  const lineSegments: string[] = []
  const areaSegments: string[] = []
  let currentSegment: string[] = []
  let segmentStart: number | null = null
  points.forEach((point, index) => {
    if (point.sector_spy_rebased_100 == null) {
      if (currentSegment.length > 1 && segmentStart != null) {
        lineSegments.push(currentSegment.join(' '))
        areaSegments.push(`${currentSegment.join(' ')} L ${scaleX(index - 1)} ${scaleY(100)} L ${scaleX(segmentStart)} ${scaleY(100)} Z`)
      }
      currentSegment = []
      segmentStart = null
      return
    }
    segmentStart ??= index
    currentSegment.push(`${currentSegment.length === 0 ? 'M' : 'L'} ${scaleX(index)} ${scaleY(point.sector_spy_rebased_100)}`)
  })
  if (currentSegment.length > 1 && segmentStart != null) {
    lineSegments.push(currentSegment.join(' '))
    areaSegments.push(`${currentSegment.join(' ')} L ${scaleX(points.length - 1)} ${scaleY(100)} L ${scaleX(segmentStart)} ${scaleY(100)} Z`)
  }
  const linePath = lineSegments.join(' ')
  const areaPath = areaSegments.join(' ')
  const y100 = scaleY(100)

  const latestVal = lastPoint.sector_spy_rebased_100 as number
  const netRelativeChange = latestVal - 100

  const hoveredPoint = hoverIndex !== null ? points[hoverIndex] : null
  const activePoint = hoveredPoint?.sector_spy_rebased_100 != null ? hoveredPoint : lastPoint

  return (
    <div className="relative overflow-hidden rounded-xl border border-slate-100 bg-gradient-to-b from-slate-50/50 to-white p-3">
      <div className="mb-2 flex flex-wrap items-center justify-between gap-2 text-xs">
        <div className="flex items-center gap-2">
          <span className="font-semibold text-slate-900">{ticker} vs SPY Relative Performance</span>
          <span className="rounded bg-slate-100 px-1.5 py-0.5 text-[10px] text-slate-600">Rebased = 100</span>
        </div>
        <div className="flex items-center gap-3 font-mono text-xs">
          <span>
            ปัจจุบัน: <strong className="text-slate-900">{(activePoint.sector_spy_rebased_100 ?? latestVal).toFixed(2)}</strong>
          </span>
          <span className={`font-bold ${netRelativeChange >= 0 ? 'text-emerald-600' : 'text-rose-600'}`}>
            ({netRelativeChange > 0 ? '+' : ''}
            {netRelativeChange.toFixed(2)}% vs SPY)
          </span>
          <span className="text-[10px] text-slate-400">({activePoint.as_of})</span>
        </div>
      </div>

      <div className="overflow-x-auto">
        <svg
          viewBox={`0 0 ${width} ${height}`}
          role="img"
          aria-label="Sector relative price performance chart"
          className="min-w-[460px] w-full"
          onMouseLeave={() => setHoverIndex(null)}
        >
          <defs>
            <linearGradient id="rel-price-area-grad" x1="0%" y1="0%" x2="0%" y2="100%">
              <stop offset="0%" stopColor="#3b82f6" stopOpacity="0.25" />
              <stop offset="100%" stopColor="#3b82f6" stopOpacity="0.0" />
            </linearGradient>
          </defs>

          {/* SPY Benchmark Line (100) */}
          <line
            x1={padding.left}
            y1={y100}
            x2={width - padding.right}
            y2={y100}
            stroke="#94a3b8"
            strokeDasharray="4 4"
            strokeWidth="1.2"
          />
          <text x={width - padding.right + 4} y={y100 + 3} fontSize="9" fill="#94a3b8" fontWeight="600">
            SPY (100)
          </text>

          {/* Area fill */}
          <path d={areaPath} fill="url(#rel-price-area-grad)" />

          {/* Main Price Path */}
          <path
            d={linePath}
            fill="none"
            stroke={latestVal >= 100 ? '#2563eb' : '#e11d48'}
            strokeWidth="2.5"
            strokeLinecap="round"
            strokeLinejoin="round"
          />

          {/* Interactive Crosshair & Points */}
          {points.map((p, idx) => {
            if (p.sector_spy_rebased_100 == null) return null
            const cx = scaleX(idx)
            const cy = scaleY(p.sector_spy_rebased_100)
            const isHovered = hoverIndex === idx
            return (
              <g
                key={p.as_of}
                onMouseEnter={() => setHoverIndex(idx)}
                className="cursor-pointer"
              >
                <circle
                  cx={cx}
                  cy={cy}
                  r={isHovered ? 5.5 : 2.5}
                  fill={isHovered ? '#1d4ed8' : '#3b82f6'}
                  stroke="#ffffff"
                  strokeWidth={isHovered ? 2 : 1}
                />
              </g>
            )
          })}

          {/* Date Axis Ticks */}
          <text x={padding.left} y={height - 8} fontSize="9.5" fill="#64748b">
            {firstPoint.as_of}
          </text>
          <text x={width - padding.right} y={height - 8} textAnchor="end" fontSize="9.5" fill="#64748b">
            {lastPoint.as_of}
          </text>
        </svg>
      </div>
      <p className="mt-1 text-[10px] text-slate-400">
        เส้นกราฟแสดงมูลค่าเปรียบเทียบ Sector/SPY Rebased เป็น 100 ณ {baseDate || 'จุดเริ่มต้นของช่วงเวลา'} · ค่าสูงกว่า 100 หมายถึง Outperform ตลาด
      </p>
    </div>
  )
}

export function SectorRotationDashboard({ aiAnalysis }: { aiAnalysis?: SectorAnalysisDTO | null }) {
  const [timeframe, setTimeframe] = useState<'weekly' | 'daily'>('weekly')
  const [data, setData] = useState<SectorRotationResponseDTO | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const requestSequence = useRef(0)
  const [archiveId, setArchiveId] = useState<string | null>(null)
  const [archiveData, setArchiveData] = useState<SectorRotationResponseDTO | null>(null)
  const [historyRange, setHistoryRange] = useState<'3m' | '6m' | '1y' | '2y'>('1y')
  const [historyData, setHistoryData] = useState<SectorRotationHistoryDTO | null>(null)
  const [historyLoading, setHistoryLoading] = useState(false)
  const [historyError, setHistoryError] = useState<string | null>(null)
  const pollAttempts = useRef(0)

  // Interactive controls
  const [selectedTicker, setSelectedTicker] = useState('XLK')
  const [hoveredTicker, setHoveredTicker] = useState<string | null>(null)
  const [activeQuadrantFilter, setActiveQuadrantFilter] = useState<string | null>(null)
  const [zoomMode, setZoomMode] = useState<'auto' | '3' | '6' | '10' | '20'>('auto')
  const [showTrails, setShowTrails] = useState(true)
  const [searchQuery, setSearchQuery] = useState('')
  const [sortBy, setSortBy] = useState<'excess' | '1m' | '3m' | 'ticker' | 'trend'>('excess')
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('desc')

  const load = useCallback(
    async (refresh = false) => {
      const sequence = ++requestSequence.current
      if (refresh) pollAttempts.current = 0
      setLoading(true)
      setError(null)
      try {
        const next = refresh
          ? await api.refreshSectorRotation(timeframe, timeframe === 'weekly' ? 12 : 20)
          : await api.getSectorRotation(timeframe, timeframe === 'weekly' ? 12 : 20)
        if (sequence !== requestSequence.current) return
        setData(next)
        if (next.refresh_state === 'running' && pollAttempts.current < 30) {
          pollAttempts.current += 1
          window.setTimeout(() => {
            if (sequence === requestSequence.current) void load(false)
          }, Math.max(1000, (next.retry_after_seconds || 2) * 1000))
        } else if (next.refresh_state !== 'running') {
          pollAttempts.current = 0
        }
      } catch (err) {
        if (sequence === requestSequence.current)
          setError(err instanceof Error ? err.message : 'Sector rotation is temporarily unavailable')
      } finally {
        if (sequence === requestSequence.current) setLoading(false)
      }
    },
    [timeframe]
  )

  useEffect(() => {
    pollAttempts.current = 0
    void load(false)
    return () => {
      requestSequence.current += 1
    }
  }, [load])

  useEffect(() => {
    let active = true
    if (!archiveId) {
      setArchiveData(null)
      return () => {
        active = false
      }
    }
    void api
      .getSectorRotationSnapshot(archiveId, timeframe, timeframe === 'weekly' ? 12 : 20)
      .then((response) => {
        if (active) setArchiveData(response)
      })
      .catch((err) => {
        if (active) setError(err instanceof Error ? err.message : 'Archived sector evidence is unavailable')
      })
    return () => {
      active = false
    }
  }, [archiveId, timeframe])

  const activeResponse = archiveId ? archiveData : data
  const snapshot = activeResponse?.timeframe === timeframe ? activeResponse.snapshot : null
  const summary = activeResponse?.summary
  const currentSnapshotId = snapshot?.snapshot_id

  useEffect(() => {
    let active = true
    const snapshotId = snapshot?.snapshot_id
    if (!snapshotId) {
      setHistoryData(null)
      setHistoryLoading(false)
      return () => {
        active = false
      }
    }
    setHistoryLoading(true)
    setHistoryError(null)
    void api.getSectorRotationHistory(snapshotId, timeframe, historyRange)
      .then((response) => {
        if (active) setHistoryData(response)
      })
      .catch((err) => {
        if (active) {
          setHistoryData(null)
          setHistoryError(err instanceof Error ? err.message : 'Historical sector data is unavailable')
        }
      })
      .finally(() => {
        if (active) setHistoryLoading(false)
      })
    return () => {
      active = false
    }
  }, [snapshot?.snapshot_id, timeframe, historyRange])

  // Quadrant Counts
  const quadrantStats = useMemo(() => {
    const counts = { Leading: 0, Weakening: 0, Lagging: 0, Improving: 0 }
    const tickers = {
      Leading: [] as string[],
      Weakening: [] as string[],
      Lagging: [] as string[],
      Improving: [] as string[],
    }
    snapshot?.rows.forEach((row) => {
      if (row.quadrant && row.quadrant in counts) {
        counts[row.quadrant as QuadrantKey]++
        tickers[row.quadrant as QuadrantKey].push(row.ticker)
      }
    })
    return { counts, tickers }
  }, [snapshot?.rows])

  // Filtered and Sorted Rows
  const processedRows = useMemo(() => {
    if (!snapshot?.rows) return []
    let list = [...snapshot.rows]

    // 1. Filter by quadrant
    if (activeQuadrantFilter) {
      list = list.filter((r) => r.quadrant === activeQuadrantFilter)
    }

    // 2. Filter by search query
    if (searchQuery.trim()) {
      const q = searchQuery.toLowerCase().trim()
      list = list.filter(
        (r) =>
          r.ticker.toLowerCase().includes(q) ||
          r.name.toLowerCase().includes(q) ||
          (SECTOR_INFO[r.ticker]?.th || '').toLowerCase().includes(q)
      )
    }

    // 3. Sort
    list.sort((a, b) => {
      let av = 0
      let bv = 0
      if (sortBy === 'excess') {
        av = a.returns_pct['3M_excess_pp'] ?? -Infinity
        bv = b.returns_pct['3M_excess_pp'] ?? -Infinity
      } else if (sortBy === '1m') {
        av = a.returns_pct['1M_absolute_pct'] ?? -Infinity
        bv = b.returns_pct['1M_absolute_pct'] ?? -Infinity
      } else if (sortBy === '3m') {
        av = a.returns_pct['3M_absolute_pct'] ?? -Infinity
        bv = b.returns_pct['3M_absolute_pct'] ?? -Infinity
      } else if (sortBy === 'trend') {
        av = a.relative_trend ?? -Infinity
        bv = b.relative_trend ?? -Infinity
      } else if (sortBy === 'ticker') {
        return sortOrder === 'asc' ? a.ticker.localeCompare(b.ticker) : b.ticker.localeCompare(a.ticker)
      }
      return sortOrder === 'desc' ? bv - av : av - bv
    })

    return list
  }, [snapshot?.rows, activeQuadrantFilter, searchQuery, sortBy, sortOrder])

  const selectedRow = snapshot?.rows.find((row) => row.ticker === selectedTicker) || snapshot?.rows[0]
  const firstTicker = snapshot?.rows[0]?.ticker
  const breadth3m = activeResponse?.summary?.sector_breadth_3m
  const validBreadthRows = snapshot?.rows.filter((row) => row.returns_pct['1M_excess_pp'] != null) || []
  const outperformingCount = validBreadthRows.filter((row) => (row.returns_pct['1M_excess_pp'] ?? 0) > 0).length

  useEffect(() => {
    if (firstTicker && !snapshot?.rows.some((row) => row.ticker === selectedTicker)) {
      setSelectedTicker(firstTicker)
    }
  }, [firstTicker, snapshot?.snapshot_id, selectedTicker])

  const selectedRowHistory = historyData && snapshot && historyData.snapshot_id === snapshot.snapshot_id && historyData.timeframe === timeframe
    ? historyData.rows.find((row) => row.ticker === selectedRow?.ticker)
    : undefined

  return (
    <section
      className="space-y-5 rounded-2xl border border-slate-200/90 bg-white p-4 shadow-sm transition-all duration-200 sm:p-6"
      aria-labelledby="sector-rotation-title"
    >
      {/* 1. Header Bar */}
      <div className="flex flex-wrap items-start justify-between gap-4 border-b border-slate-100 pb-4">
        <div>
          <div className="flex items-center gap-2.5">
            <span className="text-xl">🌐</span>
            <h2 id="sector-rotation-title" className="text-lg font-bold tracking-tight text-slate-900">
              US Sector Rotation (RRG)
            </h2>
            <span className="rounded-full border border-indigo-200 bg-indigo-50/80 px-2.5 py-0.5 text-[10px] font-semibold text-indigo-700 shadow-sm">
              Python calculation
            </span>
          </div>
          <p className="mt-1 text-xs text-slate-500">
            กราฟ Relative Rotation Graph วิเคราะห์ทิศทางและวัฏจักร 11 หมวดอุตสาหกรรมสหรัฐฯ เทียบดัชนี SPY (Benchmark)
          </p>
        </div>

        {/* Global Controls */}
        <div className="flex flex-wrap items-center gap-2">
          {/* Timeframe selector */}
          <div className="flex items-center rounded-lg border border-slate-200 bg-slate-50 p-0.5 text-xs">
            <button
              type="button"
              onClick={() => setTimeframe('weekly')}
              className={`rounded-md px-3 py-1.5 font-semibold transition-all ${
                timeframe === 'weekly' ? 'bg-white text-slate-900 shadow-sm' : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              รายสัปดาห์ (Weekly)
            </button>
            <button
              type="button"
              onClick={() => setTimeframe('daily')}
              className={`rounded-md px-3 py-1.5 font-semibold transition-all ${
                timeframe === 'daily' ? 'bg-white text-slate-900 shadow-sm' : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              รายวัน (Daily)
            </button>
          </div>

          {/* Refresh Button */}
          <button
            type="button"
            onClick={() => void load(true)}
            disabled={loading}
            className="flex items-center gap-1.5 rounded-lg border border-slate-200 bg-white px-3 py-1.5 text-xs font-semibold text-slate-700 shadow-sm transition hover:bg-slate-50 disabled:opacity-50"
          >
            <span className={loading ? 'animate-spin' : ''}>🔄</span>
            {loading ? 'กำลังคำนวณ…' : 'Refresh'}
          </button>
        </div>
      </div>

      {/* Error & Status Alerts */}
      {data?.capability_status === 'disabled' && (
        <p className="rounded-xl bg-slate-50 p-4 text-sm text-slate-600">Sector rotation data is disabled.</p>
      )}
      {error && (
        <p role="alert" className="rounded-xl border border-rose-200 bg-rose-50 p-4 text-sm font-medium text-rose-700">
          ⚠️ {error}
        </p>
      )}
      {snapshot && activeResponse?.refresh_state === 'failed' && (
        <p role="status" className="rounded-xl border border-amber-200 bg-amber-50 p-3 text-xs text-amber-800">
          ℹ️ แสดง last-good snapshot; การ refresh ล่าสุดไม่สำเร็จ ({activeResponse.error_code || 'unknown error'})
        </p>
      )}
      {!archiveId && snapshot && activeResponse?.freshness === 'stale' && (
        <p role="status" className="rounded-xl border border-amber-200 bg-amber-50 p-3 text-xs text-amber-800">
          Latest completed session expected {activeResponse.expected_session || '—'}; this snapshot is {activeResponse.missing_sessions} trading session(s) behind. Historical values remain visible with their dates.
        </p>
      )}
      {!snapshot && activeResponse?.refresh_state === 'running' && (
        <div className="flex items-center justify-center gap-3 rounded-2xl border border-sky-100 bg-sky-50/60 p-12 text-sm text-sky-800">
          <span className="animate-spin text-lg">⏳</span>
          <span>กำลังดึงข้อมูลย้อนหลังและคำนวณค่า Relative Trend & Momentum…</span>
        </div>
      )}

      {/* AI Macro Interpretation Banner */}
      {aiAnalysis && (
        <div className="space-y-3 rounded-2xl border border-violet-200/70 bg-gradient-to-br from-violet-50/70 via-white to-purple-50/40 p-4 shadow-sm">
          <div className="flex flex-wrap items-center justify-between gap-2 border-b border-violet-100/70 pb-2.5">
            <div className="flex flex-wrap items-center gap-2">
              <span className="text-base">🧠</span>
              <h3 className="text-xs font-bold uppercase tracking-wider text-violet-950">AI Macro Interpretation</h3>
              <span className="rounded-full border border-violet-200 bg-white px-2 py-0.5 text-[10px] font-semibold text-violet-800">
                สถานะ: {aiAnalysis.analysis_status || 'unavailable'}
              </span>
              {aiAnalysis.unavailable_reason && (
                <span className="text-[10px] text-amber-800">({aiAnalysis.unavailable_reason})</span>
              )}
            </div>
            {aiAnalysis.snapshot_id && (
              <div className="flex items-center gap-2 text-[10px] text-violet-700">
                <span>
                  Snapshot {aiAnalysis.snapshot_id.slice(0, 14)}… · {aiAnalysis.as_of_date || 'date unavailable'}
                </span>
                {archiveId ? (
                  <button type="button" onClick={() => setArchiveId(null)} className="font-semibold text-violet-900 underline hover:text-violet-700">
                    ดูฉบับล่าสุด
                  </button>
                ) : (
                  <button
                    type="button"
                    onClick={() => setArchiveId(aiAnalysis.snapshot_id)}
                    className="font-semibold text-violet-900 underline hover:text-violet-700"
                  >
                    เปิดหลักฐานใน Snapshot นี้
                  </button>
                )}
              </div>
            )}
          </div>
          {aiAnalysis.summary_th && (
            <p className="text-xs font-medium leading-relaxed text-violet-950 sm:text-sm">
              {aiAnalysis.summary_th}
            </p>
          )}
          {aiAnalysis.fact_claims.length > 0 && (
            <ul className="grid grid-cols-1 gap-2 sm:grid-cols-2 text-xs">
              {aiAnalysis.fact_claims.map((claim, index) => {
                const metric = aiAnalysis.resolved_metrics.find(
                  (item) => item.metric_ref === claim.metric_ref && item.ticker === claim.ticker
                )
                const fact =
                  metric?.numeric_value != null
                    ? `${metric.numeric_value > 0 ? '+' : ''}${metric.numeric_value.toFixed(2)} ${metric.unit} (${metric.horizon})`
                    : metric?.categorical_value || 'Fact not resolved'
                return (
                  <li
                    key={`${claim.ticker}-${claim.metric_ref}-${index}`}
                    className="flex flex-col rounded-lg border border-violet-100 bg-white/80 p-2 text-violet-950 shadow-2xs"
                  >
                    <span className="font-bold text-violet-900">
                      {claim.ticker} · {fact}
                    </span>
                    <span className="text-[11px] text-slate-600">{claim.interpretation_th}</span>
                  </li>
                )
              })}
            </ul>
          )}
        </div>
      )}

      {snapshot && (
        <>
          {/* Metadata & Breadth Summary */}
          <div className="flex flex-wrap items-center justify-between gap-3 rounded-xl bg-slate-50/80 px-3.5 py-2.5 text-xs text-slate-600 border border-slate-200/60">
            <div className="flex flex-wrap items-center gap-2">
              <span>ข้อมูล ณ วันที่ <strong className="text-slate-900">{snapshot.as_of_date || '—'}</strong></span>
              <span>·</span>
              <span>ครบถ้วน <strong>{snapshot.available_sectors}/{snapshot.expected_sectors}</strong> หมวดอุตสาหกรรม</span>
              <span>·</span>
              <SourceProvenanceBadge
                origin="provider"
                sourceName="Yahoo Finance (Total-Return Adj. Close)"
                observedAt={snapshot.as_of_date || undefined}
                compact
              />
            </div>
            <div className="flex items-center gap-2">
              <span className="font-medium text-slate-700">
                {breadth3m
                  ? `${breadth3m.outperforming} จาก ${breadth3m.expected_sectors} หมวด Outperform ใน 3M · ข้อมูล valid ${breadth3m.valid_sectors}/${breadth3m.expected_sectors}`
                  : `${outperformingCount} จาก ${validBreadthRows.length} หมวด Outperform ตลาด (1M Excess > 0)`}
              </span>
            </div>
          </div>

          {/* 2. Quadrant KPI Summary Cards */}
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
            {(['Leading', 'Improving', 'Weakening', 'Lagging'] as QuadrantKey[]).map((qKey) => {
              const q = QUADRANT_CONFIG[qKey]
              const count = quadrantStats.counts[qKey]
              const tickers = quadrantStats.tickers[qKey]
              const isActive = activeQuadrantFilter === qKey

              return (
                <button
                  type="button"
                  key={qKey}
                  onClick={() => setActiveQuadrantFilter(isActive ? null : qKey)}
                  className={`group flex flex-col justify-between rounded-xl border p-3 text-left transition-all duration-200 ${
                    isActive
                      ? `${q.bgLight} ${q.border} shadow-md ring-2 ring-offset-1 ring-slate-400`
                      : 'border-slate-200 bg-white hover:border-slate-300 hover:shadow-xs'
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <span className="flex items-center gap-1.5 text-xs font-bold text-slate-800">
                      <span className={`h-2.5 w-2.5 rounded-full ${q.dot}`} />
                      {q.label}
                    </span>
                    <span className={`rounded-full px-2 py-0.5 text-xs font-bold ${q.bgLight} ${q.text}`}>
                      {count}
                    </span>
                  </div>
                  <p className="mt-1 text-[11px] text-slate-500 line-clamp-1">{q.description}</p>
                  <div className="mt-2 flex flex-wrap gap-1">
                    {tickers.length > 0 ? (
                      tickers.map((t) => (
                        <span
                          key={t}
                          className="rounded bg-slate-100 px-1.5 py-0.5 font-mono text-[10px] font-semibold text-slate-700"
                        >
                          {t}
                        </span>
                      ))
                    ) : (
                      <span className="text-[10px] text-slate-400">ไม่มีหุ้นในหมวดนี้</span>
                    )}
                  </div>
                </button>
              )
            })}
          </div>

          {/* 3. Main Display Grid: Map on Left, Performance Table on Right */}
          <div className="grid grid-cols-1 gap-5 xl:grid-cols-[1.25fr_0.95fr]">
            {/* Left: Enhanced Relative Rotation Map */}
            <div className="flex flex-col rounded-2xl border border-slate-200/90 bg-white p-3.5 shadow-2xs sm:p-4">
              {/* Map Controls Toolbar */}
              <div className="mb-3 flex flex-wrap items-center justify-between gap-2 border-b border-slate-100 pb-2.5 text-xs">
                <div className="flex items-center gap-1.5 text-slate-700">
                  <span className="font-bold">Relative Rotation Map</span>
                  <span
                    className="cursor-help text-slate-400"
                    title="กราฟวิเคราะห์โมเมนตัมและทิศทางเชิงสัมพัทธ์ (Relative Rotation Graph) ศูนย์กลางแกนอยู่ที่ 100 คำนวณด้วย Rolling Z-scores ของ Sector/SPY"
                  >
                    ⓘ
                  </span>
                </div>

                <div className="flex items-center gap-2 text-xs">
                  {/* Trails Toggle */}
                  <button
                    type="button"
                    onClick={() => setShowTrails(!showTrails)}
                    className={`rounded-md border px-2 py-1 text-[11px] font-medium transition ${
                      showTrails
                        ? 'border-indigo-200 bg-indigo-50 text-indigo-700'
                        : 'border-slate-200 text-slate-500 hover:bg-slate-50'
                    }`}
                  >
                    {showTrails ? '✓ หางลาก (Tails)' : 'เฉพาะจุดล่าสุด'}
                  </button>

                  {/* Zoom Mode Selector */}
                  <div className="flex items-center gap-1 rounded-md border border-slate-200 bg-slate-50 p-0.5 text-[11px]">
                    <span className="px-1 text-[10px] text-slate-400">Zoom:</span>
                    {(['auto', '3', '6', '10', '20'] as const).map((z) => (
                      <button
                        key={z}
                        type="button"
                        onClick={() => setZoomMode(z)}
                        className={`rounded px-1.5 py-0.5 font-medium transition ${
                          zoomMode === z
                            ? 'bg-white font-bold text-slate-900 shadow-2xs'
                            : 'text-slate-500 hover:text-slate-800'
                        }`}
                      >
                        {z === 'auto' ? 'Auto' : `±${z}`}
                      </button>
                    ))}
                  </div>
                </div>
              </div>

              {/* The SVG Map */}
              <SectorMap
                rows={snapshot.rows}
                selectedTicker={selectedTicker}
                hoveredTicker={hoveredTicker}
                onSelect={setSelectedTicker}
                onHover={setHoveredTicker}
                zoomMode={zoomMode}
                showTrails={showTrails}
                activeQuadrantFilter={activeQuadrantFilter}
              />
            </div>

            {/* Right: Sector Performance Table */}
            <div className="flex flex-col rounded-2xl border border-slate-200/90 bg-white p-3.5 shadow-2xs sm:p-4">
              {/* Table Toolbar */}
              <div className="mb-3 flex flex-wrap items-center justify-between gap-2 border-b border-slate-100 pb-2.5">
                <div className="flex items-center gap-2">
                  <span className="text-xs font-bold text-slate-900">ผลตอบแทนรายหมวด</span>
                  {activeQuadrantFilter && (
                    <span className="flex items-center gap-1 rounded-full bg-slate-100 px-2 py-0.5 text-[10px] text-slate-700">
                      หมวด: <strong>{activeQuadrantFilter}</strong>
                      <button
                        type="button"
                        onClick={() => setActiveQuadrantFilter(null)}
                        className="text-slate-400 hover:text-slate-700"
                      >
                        ✕
                      </button>
                    </span>
                  )}
                </div>

                {/* Search Bar */}
                <input
                  type="text"
                  placeholder="ค้นหา Sector..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="rounded-lg border border-slate-200 px-2.5 py-1 text-xs text-slate-800 placeholder-slate-400 focus:border-indigo-500 focus:outline-none"
                />
              </div>

              {/* Table Body */}
              <div className="overflow-x-auto">
                <table className="w-full min-w-[760px] text-left text-xs">
                  <thead className="bg-slate-50/80 text-[10px] font-bold uppercase tracking-wider text-slate-500">
                    <tr>
                      <th
                        className="cursor-pointer px-2 py-2 hover:text-slate-800"
                        onClick={() => {
                          if (sortBy === 'ticker') setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc')
                          else {
                            setSortBy('ticker')
                            setSortOrder('asc')
                          }
                        }}
                      >
                        Sector {sortBy === 'ticker' ? (sortOrder === 'asc' ? '▲' : '▼') : ''}
                      </th>
                      <th className="px-2 py-2">Quadrant</th>
                      <th
                        className="cursor-pointer px-2 py-2 text-right hover:text-slate-800"
                        onClick={() => {
                          if (sortBy === '1m') setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc')
                          else {
                            setSortBy('1m')
                            setSortOrder('desc')
                          }
                        }}
                      >
                        1M {sortBy === '1m' ? (sortOrder === 'asc' ? '▲' : '▼') : ''}
                      </th>
                      <th
                        className="cursor-pointer px-2 py-2 text-right hover:text-slate-800"
                        onClick={() => {
                          if (sortBy === 'excess') setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc')
                          else {
                            setSortBy('excess')
                            setSortOrder('desc')
                          }
                        }}
                      >
                        3M Excess vs SPY {sortBy === 'excess' ? (sortOrder === 'asc' ? '▲' : '▼') : ''}
                      </th>
                      <th
                        className="cursor-pointer px-2 py-2 text-right hover:text-slate-800"
                        onClick={() => {
                          if (sortBy === '3m') setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc')
                          else {
                            setSortBy('3m')
                            setSortOrder('desc')
                          }
                        }}
                      >
                        3M {sortBy === '3m' ? (sortOrder === 'asc' ? '▲' : '▼') : ''}
                      </th>
                      <th className="px-2 py-2 text-right">1W</th>
                      <th className="px-2 py-2 text-right">6M</th>
                      <th className="px-2 py-2 text-right">YTD</th>
                      <th className="px-2 py-2 text-right">1Y</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100">
                    {processedRows.map((row) => {
                      const isSelected = selectedTicker === row.ticker
                      const isHovered = hoveredTicker === row.ticker
                      const qConfig = QUADRANT_CONFIG[row.quadrant as QuadrantKey]
                      const excessVal = row.returns_pct['3M_excess_pp']

                      // Calculate diverging bar percentage (cap at +- 10pp for max visual width)
                      const maxBarExcess = 10
                      const barPercent = excessVal == null ? 0 : Math.min(Math.abs(excessVal) / maxBarExcess, 1) * 50

                      return (
                        <tr
                          key={row.ticker}
                          onClick={() => setSelectedTicker(row.ticker)}
                          onMouseEnter={() => setHoveredTicker(row.ticker)}
                          onMouseLeave={() => setHoveredTicker(null)}
                          onKeyDown={(event) => {
                            if (event.key === 'Enter' || event.key === ' ') setSelectedTicker(row.ticker)
                          }}
                          tabIndex={0}
                          aria-selected={isSelected}
                          className={`cursor-pointer transition-colors ${
                            isSelected
                              ? 'bg-indigo-50/80'
                              : isHovered
                              ? 'bg-slate-50'
                              : 'hover:bg-slate-50/70'
                          }`}
                        >
                          <td className="px-2 py-2.5">
                            <div className="flex items-center gap-1.5">
                              <span className="text-sm">{SECTOR_INFO[row.ticker]?.icon || '📊'}</span>
                              <div>
                                <div className="font-bold text-slate-900">{row.ticker}</div>
                                <div className="text-[10px] text-slate-500">
                                  {SECTOR_INFO[row.ticker]?.th || row.name}
                                </div>
                              </div>
                            </div>
                          </td>
                          <td className="px-2 py-2.5">
                            {qConfig ? (
                              <span
                                className={`inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-[10px] font-semibold ${qConfig.bgLight} ${qConfig.border} ${qConfig.text}`}
                              >
                                <span className={`h-1.5 w-1.5 rounded-full ${qConfig.dot}`} />
                                {row.quadrant}
                              </span>
                            ) : (
                              <span className="text-slate-400">—</span>
                            )}
                          </td>
                          <td
                            title={metricTitle(row, '1M')}
                            className={`px-2 py-2.5 text-right font-mono font-semibold tabular-nums ${
                              row.returns_pct['1M_absolute_pct'] == null ? 'text-slate-400' : row.returns_pct['1M_absolute_pct'] >= 0 ? 'text-emerald-600' : 'text-rose-600'
                            }`}
                          >
                            {formatMetric(row.returns_pct['1M_absolute_pct'])}
                            {metricQualityLabel(row, '1M') && <span className="ml-1 text-[9px] text-amber-700">{metricQualityLabel(row, '1M')}</span>}
                          </td>
                          <td className="px-2 py-2.5 text-right font-mono tabular-nums">
                            <div title={metricTitle(row, '3M')} className="flex flex-col items-end">
                              <span
                                className={`font-bold ${
                                  excessVal == null ? 'text-slate-400' : excessVal >= 0 ? 'text-emerald-600' : 'text-rose-600'
                                }`}
                              >
                                {formatMetric(excessVal, ' pp')}
                                {metricQualityLabel(row, '3M') && <span className="ml-1 text-[9px] text-amber-700">{metricQualityLabel(row, '3M')}</span>}
                              </span>
                              {/* Horizontal Diverging Bar (Center = 0) */}
                              <div className="mt-1 flex h-1.5 w-16 overflow-hidden rounded-full bg-slate-100">
                                <div className="flex h-full w-1/2 justify-end">
                                  {excessVal != null && excessVal < 0 && (
                                    <div
                                      className="h-full rounded-l-full bg-rose-500"
                                      style={{ width: `${barPercent * 2}%` }}
                                    />
                                  )}
                                </div>
                                <div className="h-full w-px bg-slate-300" />
                                <div className="flex h-full w-1/2 justify-start">
                                  {excessVal != null && excessVal > 0 && (
                                    <div
                                      className="h-full rounded-r-full bg-emerald-500"
                                      style={{ width: `${barPercent * 2}%` }}
                                    />
                                  )}
                                </div>
                              </div>
                            </div>
                          </td>
                          <td
                            title={metricTitle(row, '3M')}
                            className={`px-2 py-2.5 text-right font-mono tabular-nums ${
                              row.returns_pct['3M_absolute_pct'] == null ? 'text-slate-400' : row.returns_pct['3M_absolute_pct'] >= 0 ? 'text-emerald-600' : 'text-rose-600'
                            }`}
                          >
                            {formatMetric(row.returns_pct['3M_absolute_pct'])}
                            {metricQualityLabel(row, '3M') && <span className="ml-1 text-[9px] text-amber-700">{metricQualityLabel(row, '3M')}</span>}
                          </td>
                          {(['1W', '6M', 'YTD', '1Y'] as const).map((horizon) => {
                            const value = row.returns_pct[`${horizon}_absolute_pct`]
                            return (
                              <td key={horizon} title={metricTitle(row, horizon)}
                                className={`px-2 py-2.5 text-right font-mono tabular-nums ${
                                  value == null ? 'text-slate-400' : value >= 0 ? 'text-emerald-600' : 'text-rose-600'
                                }`}>
                                {formatMetric(value)}
                                {metricQualityLabel(row, horizon) && <span className="ml-1 text-[9px] text-amber-700">{metricQualityLabel(row, horizon)}</span>}
                              </td>
                            )
                          })}
                        </tr>
                      )
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          {/* 4. Selected Sector Deep-Dive Card */}
          {selectedRow && (
            <div className="space-y-4 rounded-2xl border border-slate-200/80 bg-white p-4 shadow-sm sm:p-5">
              <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-100 pb-3">
                <div className="flex items-center gap-3">
                  <span className="flex h-10 w-10 items-center justify-center rounded-xl bg-slate-100 text-xl shadow-2xs">
                    {SECTOR_INFO[selectedRow.ticker]?.icon || '📊'}
                  </span>
                  <div>
                    <div className="flex items-center gap-2">
                      <h3 className="text-base font-bold text-slate-900">{selectedRow.ticker}</h3>
                      <span className="text-xs text-slate-500">·</span>
                      <span className="text-xs font-semibold text-slate-700">
                        {SECTOR_INFO[selectedRow.ticker]?.th || selectedRow.name}
                      </span>
                      {selectedRow.quadrant && (
                        <span
                          className={`rounded-full border px-2.5 py-0.5 text-[10px] font-bold ${
                            QUADRANT_CONFIG[selectedRow.quadrant as QuadrantKey]?.bgLight
                          } ${QUADRANT_CONFIG[selectedRow.quadrant as QuadrantKey]?.border} ${
                            QUADRANT_CONFIG[selectedRow.quadrant as QuadrantKey]?.text
                          }`}
                        >
                          {selectedRow.quadrant}
                        </span>
                      )}
                    </div>
                    <p className="text-[11px] text-slate-400">
                      {SECTOR_INFO[selectedRow.ticker]?.category || selectedRow.name}
                    </p>
                  </div>
                </div>

                {/* Key Metrics Chips */}
                <div className="flex flex-wrap items-center gap-3 font-mono text-xs">
                  <div className="rounded-lg border border-slate-200 bg-slate-50 px-2.5 py-1">
                    <span className="text-[10px] text-slate-400">Relative Trend: </span>
                    <strong className="text-slate-800">{selectedRow.relative_trend?.toFixed(2) ?? '—'}</strong>
                  </div>
                  <div className="rounded-lg border border-slate-200 bg-slate-50 px-2.5 py-1">
                    <span className="text-[10px] text-slate-400">Relative Momentum: </span>
                    <strong className="text-slate-800">{selectedRow.relative_momentum?.toFixed(2) ?? '—'}</strong>
                  </div>
                  <div className="rounded-lg border border-slate-200 bg-slate-50 px-2.5 py-1">
                    <span className="text-[10px] text-slate-400">1M Excess: </span>
                    <strong
                      className={
                        (selectedRow.returns_pct['1M_excess_pp'] ?? 0) >= 0 ? 'text-emerald-600' : 'text-rose-600'
                      }
                    >
                      {formatMetric(selectedRow.returns_pct['1M_excess_pp'], ' pp')}
                    </strong>
                  </div>
                  <div className="rounded-lg border border-slate-200 bg-slate-50 px-2.5 py-1">
                    <span className="text-[10px] text-slate-400">Time in quadrant: </span>
                    <strong className="text-slate-800">
                      {summary?.periods_in_quadrant?.[selectedRow.ticker] ?? '—'} bars · {summary?.elapsed_days_in_quadrant?.[selectedRow.ticker] ?? '—'} days
                    </strong>
                  </div>
                  <div className="rounded-lg border border-slate-200 bg-slate-50 px-2.5 py-1">
                    <span className="text-[10px] text-slate-400">Momentum delta: </span>
                    <strong className="text-slate-800">{summary?.momentum_delta?.[selectedRow.ticker]?.toFixed(3) ?? '—'}</strong>
                  </div>
                  <div className="rounded-lg border border-slate-200 bg-slate-50 px-2.5 py-1">
                    <span className="text-[10px] text-slate-400">3 interval heading: </span>
                    <strong className="text-slate-800">
                      {summary?.heading_deg?.[selectedRow.ticker] != null
                        ? `${summary.heading_deg[selectedRow.ticker]}°`
                        : '—'}
                    </strong>
                  </div>
                </div>
              </div>

              {/* Relative Performance vs SPY Chart */}
              <div className="flex flex-wrap items-center justify-between gap-2">
                <label className="flex items-center gap-2 text-xs font-medium text-slate-600">
                  History range
                  <select
                    value={historyRange}
                    onChange={(event) => setHistoryRange(event.target.value as '3m' | '6m' | '1y' | '2y')}
                    className="rounded-md border border-slate-200 bg-white px-2 py-1 text-xs text-slate-800"
                    aria-label="Sector history range"
                  >
                    <option value="3m">3 months</option>
                    <option value="6m">6 months</option>
                    <option value="1y">1 year</option>
                    <option value="2y">2 years</option>
                  </select>
                </label>
                <span className="text-[10px] text-slate-400">
                  {historyLoading ? 'Loading snapshot history…' : historyData && historyData.snapshot_id === currentSnapshotId
                    ? `${historyData.from_date} to ${historyData.to_date || '—'} · ${historyData.snapshot_id}`
                    : historyError || ''}
                </span>
              </div>
              <RelativePriceChart
                points={selectedRowHistory?.relative_price_history ?? selectedRow.relative_price_history}
                baseDate={selectedRow.relative_price_base_date}
                ticker={selectedRow.ticker}
              />

              {/* Quadrant Transition Timeline */}
              {(selectedRowHistory?.quadrant_transitions ?? selectedRow.quadrant_transitions).length > 0 && (
                <div className="space-y-1.5">
                  <span className="text-[11px] font-semibold text-slate-600">ประวัติการเปลี่ยนสภาวะ (Quadrant Transitions):</span>
                  <div className="flex flex-wrap gap-2">
                    {(selectedRowHistory?.quadrant_transitions ?? selectedRow.quadrant_transitions).slice(-6).map((event) => (
                      <span
                        key={event.event_id}
                        className="inline-flex items-center gap-1.5 rounded-lg border border-slate-200 bg-slate-50 px-2.5 py-1 text-[10px] text-slate-700 shadow-2xs"
                        title={event.event_id}
                      >
                        <span className="font-semibold">{event.from_quadrant}</span>
                        <span className="text-slate-400">→</span>
                        <span className="font-bold text-slate-900">{event.to_quadrant}</span>
                        <span className={event.event_type === 'confirmed_transition' ? 'text-emerald-700' : 'text-amber-700'}>
                          · {event.event_type === 'confirmed_transition' ? 'confirmed' : 'started'} · {event.confirmed_at || event.changed_at || event.previous_valid_at}
                        </span>
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}

          <p className="text-[10px] text-slate-400">
            * แกนพิกัด RRG อิงค่า Relative Trend (RS-Ratio) และ Relative Momentum (RS-Momentum) โดยมีค่ามาตรฐานกลางที่ 100
            ตัวเลขผลตอบแทนเปรียบเทียบแต่ละ ETF กับดัชนี SPY เพื่อประเมินภาวะตลาดเชิงสัมพัทธ์ (Relative Strength)
          </p>
        </>
      )}
    </section>
  )
}
