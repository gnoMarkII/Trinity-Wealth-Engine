import React, { useMemo } from 'react'
import type { TreasuryYieldCurveDTO } from '../../../api/types'
import { SourceProvenanceBadge } from './SourceProvenanceBadge'

interface YieldCurveChartProps {
  data: TreasuryYieldCurveDTO | null
  loading?: boolean
  error?: string | null
  className?: string
}

// Tenor ordering for proper cross-section layout
const TENOR_ORDER = [
  '1 Mo',
  '2 Mo',
  '3 Mo',
  '4 Mo',
  '6 Mo',
  '1 Yr',
  '2 Yr',
  '3 Yr',
  '5 Yr',
  '7 Yr',
  '10 Yr',
  '20 Yr',
  '30 Yr',
]

export const YieldCurveChart: React.FC<YieldCurveChartProps> = ({
  data,
  loading = false,
  error = null,
  className = '',
}) => {
  const yieldsMap = useMemo(() => {
    if (!data?.yields) return new Map<string, number>()
    const m = new Map<string, number>()
    for (const pt of data.yields) {
      if (typeof pt.yield_percent === 'number' && !isNaN(pt.yield_percent)) {
        m.set(pt.maturity, pt.yield_percent)
      }
    }
    return m
  }, [data])

  const orderedPoints = useMemo(() => {
    return TENOR_ORDER.map((tenor) => ({
      tenor,
      yield: yieldsMap.get(tenor) ?? null,
    }))
  }, [yieldsMap])

  const validYields = useMemo(
    () => orderedPoints.filter((p) => p.yield !== null).map((p) => p.yield as number),
    [orderedPoints]
  )

  const minYield = validYields.length > 0 ? Math.floor(Math.min(...validYields) * 2) / 2 - 0.25 : 3.0
  const maxYield = validYields.length > 0 ? Math.ceil(Math.max(...validYields) * 2) / 2 + 0.25 : 6.0
  const yieldRange = maxYield - minYield || 1

  // SVG dimensions
  const width = 640
  const height = 240
  const padLeft = 45
  const padRight = 30
  const padTop = 20
  const padBottom = 35
  const plotWidth = width - padLeft - padRight
  const plotHeight = height - padTop - padBottom

  const coords = useMemo(() => {
    const pts = orderedPoints.filter((p) => p.yield !== null)
    if (pts.length < 2) return []
    const step = plotWidth / (orderedPoints.length - 1)

    return orderedPoints.map((p, idx) => {
      const x = padLeft + idx * step
      if (p.yield === null) return { ...p, x, y: null }
      const y = padTop + plotHeight - ((p.yield - minYield) / yieldRange) * plotHeight
      return { ...p, x, y }
    })
  }, [orderedPoints, plotWidth, plotHeight, padLeft, padTop, minYield, yieldRange])

  const polylinePoints = useMemo(() => {
    return coords
      .filter((c) => c.y !== null)
      .map((c) => `${c.x},${c.y}`)
      .join(' ')
  }, [coords])

  const spread10y2y = data?.spread_10y_2y_bps
  const spread10y3m = data?.spread_10y_3m_bps
  const isInverted = (spread10y2y !== undefined && spread10y2y !== null && spread10y2y < 0) ||
                     (spread10y3m !== undefined && spread10y3m !== null && spread10y3m < 0)

  if (loading) {
    return (
      <div className={`rounded-2xl border border-sky-100 bg-white/90 p-5 shadow-sm animate-pulse ${className}`}>
        <div className="h-6 w-48 bg-slate-200 rounded mb-4" />
        <div className="h-56 bg-slate-100 rounded-xl" />
      </div>
    )
  }

  if (error || !data) {
    return (
      <div className={`rounded-2xl border border-amber-200 bg-amber-50/60 p-5 text-xs text-amber-900 ${className}`}>
        <div className="font-bold flex items-center gap-1.5">
          <span>⚠️</span>
          <span>US Treasury Yield Curve: ไม่สามารถโหลดข้อมูลได้</span>
        </div>
        <p className="mt-1 text-zinc-600">{error || 'ไม่มีข้อมูล Yield Curve จากกระทรวงการคลังสหรัฐฯ (Fiscal Data)'}</p>
      </div>
    )
  }

  return (
    <div className={`rounded-2xl border border-sky-100 bg-white/90 p-5 shadow-sm ${className}`}>
      {/* Header & Provenance */}
      <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between border-b border-sky-100/70 pb-3">
        <div>
          <div className="flex items-center gap-2">
            <h3 className="font-bold text-zinc-900 text-sm tracking-tight">
              US Treasury Yield Curve (Cross-Section)
            </h3>
            {isInverted ? (
              <span className="rounded-md border border-rose-200 bg-rose-50 px-2 py-0.5 text-[10px] font-bold text-rose-700">
                INVERTED (กลับหัว)
              </span>
            ) : (
              <span className="rounded-md border border-emerald-200 bg-emerald-50 px-2 py-0.5 text-[10px] font-bold text-emerald-700">
                NORMAL (ชันปกติ)
              </span>
            )}
          </div>
          <p className="text-xs text-zinc-500 mt-0.5">
            อัตราผลตอบแทนพันธบัตรรัฐบาลสหรัฐฯ ณ สิ้นวันทำการข้าม Tenor (1M ถึง 30Y)
          </p>
        </div>

        <SourceProvenanceBadge
          origin="provider"
          sourceName="US Treasury"
          observedAt={data.observation_date}
          fetchedAt={data.fetched_at ? new Date(data.fetched_at * 1000).toLocaleTimeString('th-TH', { hour: '2-digit', minute: '2-digit' }) : undefined}
          compact
        />
      </div>

      {/* Yield Spreads Key Metrics */}
      <div className="mt-3 grid grid-cols-2 sm:grid-cols-4 gap-2 text-xs">
        <div className="rounded-xl bg-slate-50 p-2.5 border border-slate-100">
          <span className="text-[10px] text-zinc-500 block">Spread 10Y – 2Y</span>
          <span className={`font-mono text-base font-extrabold ${
            (spread10y2y ?? 0) < 0 ? 'text-rose-600' : 'text-emerald-700'
          }`}>
            {spread10y2y !== null && spread10y2y !== undefined ? `${spread10y2y > 0 ? '+' : ''}${spread10y2y} bps` : '—'}
          </span>
          <span className="text-[9px] text-zinc-400 block mt-0.5">สัญญาณเศรษฐกิจถดถอย</span>
        </div>

        <div className="rounded-xl bg-slate-50 p-2.5 border border-slate-100">
          <span className="text-[10px] text-zinc-500 block">Spread 10Y – 3M</span>
          <span className={`font-mono text-base font-extrabold ${
            (spread10y3m ?? 0) < 0 ? 'text-rose-600' : 'text-emerald-700'
          }`}>
            {spread10y3m !== null && spread10y3m !== undefined ? `${spread10y3m > 0 ? '+' : ''}${spread10y3m} bps` : '—'}
          </span>
          <span className="text-[9px] text-zinc-400 block mt-0.5">โมเดล Fed NY ติดตาม</span>
        </div>

        <div className="rounded-xl bg-slate-50 p-2.5 border border-slate-100">
          <span className="text-[10px] text-zinc-500 block">2-Year Note Yield</span>
          <span className="font-mono text-base font-extrabold text-zinc-900">
            {yieldsMap.get('2 Yr') !== undefined ? `${yieldsMap.get('2 Yr')?.toFixed(2)}%` : '—'}
          </span>
          <span className="text-[9px] text-zinc-400 block mt-0.5">นโยบาย Fed ระยะสั้น</span>
        </div>

        <div className="rounded-xl bg-slate-50 p-2.5 border border-slate-100">
          <span className="text-[10px] text-zinc-500 block">10-Year Benchmark</span>
          <span className="font-mono text-base font-extrabold text-sky-800">
            {yieldsMap.get('10 Yr') !== undefined ? `${yieldsMap.get('10 Yr')?.toFixed(2)}%` : '—'}
          </span>
          <span className="text-[9px] text-zinc-400 block mt-0.5">อัตราปลอดความเสี่ยงโลก</span>
        </div>
      </div>

      {/* SVG Cross-Section Chart */}
      <div className="mt-4 overflow-hidden rounded-xl border border-sky-100 bg-gradient-to-b from-slate-50/80 via-white to-sky-50/20 p-3.5 shadow-2xs">
        <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-auto select-none" role="img" aria-label="Yield Curve Chart">
          {/* Y Axis Grid lines */}
          {[0, 0.25, 0.5, 0.75, 1].map((frac, i) => {
            const y = padTop + plotHeight * (1 - frac)
            const val = minYield + yieldRange * frac
            return (
              <g key={i}>
                <line x1={padLeft} y1={y} x2={width - padRight} y2={y} stroke="#e2e8f0" strokeDasharray="3 3" />
                <text x={padLeft - 6} y={y + 3} textAnchor="end" fontSize="10" fill="#64748b" className="font-mono">
                  {val.toFixed(2)}%
                </text>
              </g>
            )
          })}

          {/* Polyline curve */}
          {polylinePoints && (
            <polyline
              fill="none"
              stroke="#0284c7"
              strokeWidth="2.5"
              strokeLinecap="round"
              strokeLinejoin="round"
              points={polylinePoints}
            />
          )}

          {/* Data Points */}
          {coords.map((c, i) => {
            if (c.y === null || c.yield === null) return null
            const isHighlight = c.tenor === '3 Mo' || c.tenor === '2 Yr' || c.tenor === '10 Yr'
            return (
              <g key={i}>
                <circle
                  cx={c.x}
                  cy={c.y}
                  r={isHighlight ? 5.5 : 3.5}
                  fill={isHighlight ? '#0284c7' : '#ffffff'}
                  stroke={isHighlight ? '#ffffff' : '#0284c7'}
                  strokeWidth={isHighlight ? 2 : 1.5}
                />
                <text
                  x={c.x}
                  y={height - 10}
                  textAnchor="middle"
                  fontSize="9"
                  fill="#64748b"
                  className="font-mono font-medium"
                >
                  {c.tenor}
                </text>
              </g>
            )
          })}
        </svg>
      </div>

      <p className="mt-2 text-[11px] text-zinc-400">
        ℹ️ กราฟนี้เป็นภาพตัดขวาง (Cross-section) ณ วันเดียว ไม่ใช่แนวโน้มตามเวลา จุดเด่น 3M, 2Y และ 10Y ใช้คำนวณ Spread ชี้วัดวงจรเศรษฐกิจ
      </p>
    </div>
  )
}
