import { useState, useMemo } from 'react'
import type { PerformanceSnapshotDTO } from '../../api/types'
import { formatTHB } from './PortfolioSummaryCards'

interface Props {
  performanceRows: PerformanceSnapshotDTO[]
  daysRange: number | undefined
  onChangeDaysRange: (days: number | undefined) => void
}

export default function PortfolioAnalyticsTab({
  performanceRows,
  daysRange,
  onChangeDaysRange,
}: Props) {
  const [hoverIndex, setHoverIndex] = useState<number | null>(null)

  const sortedRows = useMemo(() => {
    return [...performanceRows].sort((a, b) => a.Date.localeCompare(b.Date))
  }, [performanceRows])

  // Chart dimensions & calculations
  const width = 760
  const height = 260
  const padTop = 25
  const padBottom = 35
  const padLeft = 70
  const padRight = 20
  const chartWidth = width - padLeft - padRight
  const chartHeight = height - padTop - padBottom

  let navLinePath = ''
  let navAreaPath = ''
  let costLinePath = ''
  let points: { x: number; yNav: number; yCost: number; date: string; nav: number; cost: number; pnl: number }[] = []
  let gridLines: { y: number; val: number }[] = []

  if (sortedRows.length >= 2) {
    const allVals = sortedRows.flatMap((r) => [r.Total_NAV, r.Total_Cost])
    const minVal = Math.min(...allVals)
    const maxVal = Math.max(...allVals)
    const padding = (maxVal - minVal) * 0.08 || minVal * 0.05 || 1000
    const minY = Math.max(0, minVal - padding)
    const maxY = maxVal + padding
    const range = maxY - minY || 1

    points = sortedRows.map((r, idx) => {
      const x = padLeft + (idx / (sortedRows.length - 1)) * chartWidth
      const yNav = height - padBottom - ((r.Total_NAV - minY) / range) * chartHeight
      const yCost = height - padBottom - ((r.Total_Cost - minY) / range) * chartHeight
      return {
        x,
        yNav,
        yCost,
        date: r.Date,
        nav: r.Total_NAV,
        cost: r.Total_Cost,
        pnl: r.Unrealized_PnL,
      }
    })

    navLinePath = points.map((p, i) => `${i === 0 ? 'M' : 'L'} ${p.x.toFixed(1)} ${p.yNav.toFixed(1)}`).join(' ')
    navAreaPath = `${navLinePath} L ${points[points.length - 1]!.x.toFixed(1)} ${height - padBottom} L ${padLeft} ${height - padBottom} Z`
    costLinePath = points.map((p, i) => `${i === 0 ? 'M' : 'L'} ${p.x.toFixed(1)} ${p.yCost.toFixed(1)}`).join(' ')

    // Grid lines (4 levels)
    gridLines = [0, 0.33, 0.66, 1].map((pct) => ({
      y: height - padBottom - pct * chartHeight,
      val: minY + pct * range,
    }))
  }

  const activePoint =
    hoverIndex !== null && points[hoverIndex]
      ? points[hoverIndex]
      : points.length > 0
        ? points[points.length - 1]
        : null

  const handleMouseMove = (e: React.MouseEvent<SVGRectElement, MouseEvent>) => {
    if (points.length < 2) return
    const rect = e.currentTarget.getBoundingClientRect()
    const mouseX = e.clientX - rect.left
    // find closest point x
    let closestIdx = 0
    let minDist = Infinity
    points.forEach((p, idx) => {
      const dist = Math.abs(p.x - padLeft - (mouseX / rect.width) * chartWidth)
      if (dist < minDist) {
        minDist = dist
        closestIdx = idx
      }
    })
    setHoverIndex(closestIdx)
  }

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Header + Range Selector */}
      <div className="flex flex-col justify-between gap-3 sm:flex-row sm:items-center">
        <div>
          <h3 className="text-base font-bold text-zinc-900">
            Performance History ({performanceRows.length} จุดข้อมูล)
          </h3>
          <p className="text-xs text-zinc-500">
            ติดตามประวัติ NAV Snapshot การเปลี่ยนแปลงต้นทุน และกำไร/ขาดทุนที่ยังไม่เกิดขึ้นจริง (Unrealized PnL)
          </p>
        </div>

        {/* Days selector bar */}
        <div className="flex items-center gap-2 rounded-xl border border-sky-100 bg-panel p-2 shadow-2xs">
          <span className="text-xs font-semibold text-zinc-600 ml-1">ช่วงเวลา:</span>
          <div className="flex gap-1">
            {[
              { label: '30 วัน', val: 30 },
              { label: '90 วัน', val: 90 },
              { label: '1 ปี', val: 365 },
              { label: 'ทั้งหมด', val: undefined },
            ].map((opt) => (
              <button
                key={opt.label}
                type="button"
                onClick={() => onChangeDaysRange(opt.val)}
                className={`rounded-lg px-3 py-1 text-xs font-semibold transition-colors ${
                  daysRange === opt.val
                    ? 'bg-flow-blue text-white shadow-2xs font-bold'
                    : 'bg-sky-50 text-zinc-500 hover:bg-sky-100'
                }`}
              >
                {opt.label}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* SVG Area Chart Panel */}
      <div className="rounded-2xl border border-sky-100 bg-panel p-6 shadow-sm">
        {/* Active Point Tracker Header */}
        <div className="flex flex-wrap items-center justify-between gap-4 border-b border-sky-100 pb-4 mb-4">
          <div className="flex items-center gap-6">
            <div>
              <span className="text-[11px] font-semibold uppercase text-zinc-400">Snapshot Date</span>
              <div className="text-sm font-bold font-mono text-zinc-800">
                {activePoint ? activePoint.date : 'N/A'}
              </div>
            </div>
            <div>
              <span className="text-[11px] font-semibold uppercase text-zinc-400">Total NAV</span>
              <div className="text-lg font-extrabold font-mono tabular-nums text-flow-blue">
                {activePoint ? formatTHB(activePoint.nav) : '—'}
              </div>
            </div>
            <div>
              <span className="text-[11px] font-semibold uppercase text-zinc-400">Cost Basis</span>
              <div className="text-sm font-bold font-mono tabular-nums text-zinc-600">
                {activePoint ? formatTHB(activePoint.cost) : '—'}
              </div>
            </div>
            <div>
              <span className="text-[11px] font-semibold uppercase text-zinc-400">Unrealized PnL</span>
              <div
                className={`text-sm font-bold font-mono tabular-nums ${
                  activePoint && activePoint.pnl >= 0 ? 'text-emerald-600' : 'text-rose-600'
                }`}
              >
                {activePoint
                  ? `${activePoint.pnl >= 0 ? '+' : ''}${formatTHB(activePoint.pnl)}`
                  : '—'}
              </div>
            </div>
          </div>

          <div className="flex items-center gap-4 text-xs font-medium">
            <span className="flex items-center gap-1.5 text-zinc-700">
              <span className="inline-block h-3 w-3 rounded-sm bg-flow-blue" />
              Total NAV Area
            </span>
            <span className="flex items-center gap-1.5 text-zinc-500">
              <span className="inline-block h-0.5 w-3 bg-slate-500 border-t border-dashed border-slate-500" />
              Cost Basis
            </span>
          </div>
        </div>

        {/* Chart Surface */}
        {sortedRows.length >= 2 ? (
          <div className="relative w-full overflow-hidden">
            <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-auto overflow-visible">
              <defs>
                <linearGradient id="analyticsNavGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#0284c7" stopOpacity={0.25} />
                  <stop offset="100%" stopColor="#0284c7" stopOpacity={0.01} />
                </linearGradient>
              </defs>

              {/* Grid lines & Left Axis Labels */}
              {gridLines.map((gl, i) => (
                <g key={i}>
                  <line
                    x1={padLeft}
                    y1={gl.y}
                    x2={width - padRight}
                    y2={gl.y}
                    stroke="#f1f5f9"
                    strokeWidth="1"
                    strokeDasharray="4,4"
                  />
                  <text
                    x={padLeft - 10}
                    y={gl.y + 4}
                    textAnchor="end"
                    className="text-[10px] font-mono fill-zinc-400"
                  >
                    {(gl.val / 1000).toFixed(0)}k
                  </text>
                </g>
              ))}

              {/* Date Ticks on Bottom */}
              {[
                { idx: 0, anchor: 'start' as const },
                { idx: Math.floor((points.length - 1) / 2), anchor: 'middle' as const },
                { idx: points.length - 1, anchor: 'end' as const },
              ].map(({ idx, anchor }) => {
                const pt = points[idx]
                if (!pt) return null
                return (
                  <text
                    key={idx}
                    x={pt.x}
                    y={height - 10}
                    textAnchor={anchor}
                    className="text-[10px] font-mono fill-zinc-400"
                  >
                    {pt.date}
                  </text>
                )
              })}

              {/* Area & Lines */}
              <path d={navAreaPath} fill="url(#analyticsNavGrad)" />
              <path d={costLinePath} stroke="#64748b" strokeWidth="1.5" strokeDasharray="5,5" fill="none" />
              <path d={navLinePath} stroke="#0284c7" strokeWidth="2.5" fill="none" strokeLinecap="round" />

              {/* Active Hover Marker */}
              {activePoint && (
                <g>
                  <line
                    x1={activePoint.x}
                    y1={padTop}
                    x2={activePoint.x}
                    y2={height - padBottom}
                    stroke="#0284c7"
                    strokeWidth="1.5"
                    strokeDasharray="3,3"
                  />
                  <circle
                    cx={activePoint.x}
                    cy={activePoint.yNav}
                    r={5}
                    fill="#0284c7"
                    stroke="#ffffff"
                    strokeWidth={2}
                  />
                  <circle
                    cx={activePoint.x}
                    cy={activePoint.yCost}
                    r={4}
                    fill="#64748b"
                    stroke="#ffffff"
                    strokeWidth={1.5}
                  />
                </g>
              )}

              {/* Transparent Overlay for Hover Tracking */}
              <rect
                x={padLeft}
                y={padTop}
                width={chartWidth}
                height={chartHeight}
                fill="transparent"
                className="cursor-crosshair"
                onMouseMove={handleMouseMove}
                onMouseLeave={() => setHoverIndex(null)}
              />
            </svg>
          </div>
        ) : (
          <div className="py-20 text-center font-sans text-zinc-400">
            ต้องการข้อมูล Snapshot อย่างน้อย 2 วันเพื่อแสดงกราฟแนวโน้ม (ปัจจุบัน: {sortedRows.length} วัน)
          </div>
        )}
      </div>

      {/* Performance Table (ย้ายมาอยู่ใต้อาณาบริเวณกราฟ) */}
      <div className="rounded-2xl border border-sky-100 bg-panel shadow-sm overflow-hidden">
        <div className="border-b border-sky-100 bg-sky-50/40 px-6 py-4">
          <h4 className="text-sm font-bold text-zinc-900">ตารางบันทึกประวัติ Snapshot (Table Data)</h4>
          <p className="text-xs text-zinc-500">
            แสดงรายละเอียดตัวเลขประจำวันของ NAV Snapshot ในช่วงเวลาที่เลือก
          </p>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-left text-sm">
            <thead>
              <tr className="border-b border-sky-100 bg-zinc-50/80 text-xs font-semibold uppercase tracking-wider text-zinc-500">
                <th className="px-6 py-3.5">Date</th>
                <th className="px-6 py-3.5 text-right">Total NAV</th>
                <th className="px-6 py-3.5 text-right">Total Cost</th>
                <th className="px-6 py-3.5 text-right">Unrealized PnL</th>
                <th className="px-6 py-3.5 text-right">Cash Balance</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-sky-50 font-mono tabular-nums text-xs">
              {sortedRows.map((row) => {
                const isPos = row.Unrealized_PnL >= 0
                return (
                  <tr key={row.Date} className="hover:bg-sky-50/60 transition-colors">
                    <td className="px-6 py-4 font-bold text-zinc-900">{row.Date}</td>
                    <td className="px-6 py-4 text-right font-bold text-zinc-900">{formatTHB(row.Total_NAV)}</td>
                    <td className="px-6 py-4 text-right text-zinc-600">{formatTHB(row.Total_Cost)}</td>
                    <td className={`px-6 py-4 text-right font-bold ${isPos ? 'text-emerald-600' : 'text-rose-600'}`}>
                      {isPos ? '+' : ''}
                      {formatTHB(row.Unrealized_PnL)}
                    </td>
                    <td className="px-6 py-4 text-right text-zinc-700">{formatTHB(row.Cash_Balance)}</td>
                  </tr>
                )
              })}
              {sortedRows.length === 0 && (
                <tr>
                  <td colSpan={5} className="py-16 text-center font-sans text-zinc-400">
                    ยังไม่มีประวัติการบันทึก NAV Snapshot ในช่วงเวลาที่เลือก
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}
