import React, { useState, useMemo } from 'react'
import type { PerformanceSnapshotDTO } from '../../api/types'
import { formatTHB } from '../../utils/formatters'

interface Props {
  performanceRows: PerformanceSnapshotDTO[]
  daysRange: number | undefined
  onChangeDaysRange: (days: number | undefined) => void
  portfolioId?: string
  onSnapshotTriggered?: () => void
}

export default function PortfolioPerformanceCard({
  performanceRows,
  daysRange,
  onChangeDaysRange,
}: Props) {
  const [hoverIndex, setHoverIndex] = useState<number | null>(null)
  const [tableOpen, setTableOpen] = useState(false)

  // Sort rows ascending by date
  const sortedRows = useMemo(() => {
    return [...performanceRows].sort((a, b) => a.Date.localeCompare(b.Date))
  }, [performanceRows])

  // Smart fallback: Check if the current filter yielded fewer than 2 rows
  const isFallbackApplied = daysRange !== undefined && sortedRows.length < 2

  // Check if today already has a snapshot
  const todayStr = useMemo(() => {
    const d = new Date()
    return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`
  }, [])
  const hasSnapshotToday = useMemo(() => {
    return sortedRows.some((r) => r.Date === todayStr)
  }, [sortedRows, todayStr])

  // Chart dimensions & calculations
  const width = 760
  const height = 240
  const padTop = 20
  const padBottom = 30
  const padLeft = 65
  const padRight = 15
  const chartWidth = width - padLeft - padRight
  const chartHeight = height - padTop - padBottom

  let navLinePath = ''
  let navAreaPath = ''
  let costLinePath = ''
  let points: {
    x: number
    yNav: number
    yCost: number
    date: string
    nav: number
    cost: number
    pnl: number
  }[] = []
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

    // 4 horizontal grid lines
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
        : sortedRows.length > 0
          ? {
              date: sortedRows[sortedRows.length - 1]!.Date,
              nav: sortedRows[sortedRows.length - 1]!.Total_NAV,
              cost: sortedRows[sortedRows.length - 1]!.Total_Cost,
              pnl: sortedRows[sortedRows.length - 1]!.Unrealized_PnL,
            }
          : null

  const handleMouseMove = (e: React.MouseEvent<SVGRectElement, MouseEvent>) => {
    if (points.length < 2) return
    const rect = e.currentTarget.getBoundingClientRect()
    const mouseX = e.clientX - rect.left
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

  const latestReturnPct = useMemo(() => {
    if (!activePoint || activePoint.cost <= 0) return null
    return (activePoint.pnl / activePoint.cost) * 100
  }, [activePoint])

  return (
    <div className="rounded-2xl border border-sky-100 bg-panel p-5 shadow-sm transition-all space-y-4">
      {/* Header & Controls */}
      <div className="flex flex-col justify-between gap-4 sm:flex-row sm:items-center border-b border-sky-100 pb-4">
        <div>
          <div className="flex items-center gap-2">
            <h3 className="text-sm sm:text-base font-extrabold text-zinc-900 tracking-tight">
              ผลการดำเนินงานและการเติบโต (Performance & NAV Growth)
            </h3>
            <span className="rounded-full bg-sky-100 px-2 py-0.5 text-[10px] font-bold text-sky-800">
              {sortedRows.length} จุดข้อมูล
            </span>
          </div>
          <p className="text-xs text-zinc-500 mt-0.5">
            ติดตามประวัติการเติบโตของมูลค่าพอร์ต (NAV) เปรียบเทียบกับต้นทุนสะสม
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          {/* Days selector */}
          <div className="flex items-center rounded-xl border border-sky-100 bg-surface p-1 shadow-2xs">
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
                className={`rounded-lg px-2.5 py-1 text-xs font-semibold transition-colors ${
                  daysRange === opt.val
                    ? 'bg-flow-blue text-white shadow-2xs font-bold'
                    : 'text-zinc-500 hover:bg-sky-50'
                }`}
              >
                {opt.label}
              </button>
            ))}
          </div>

          {/* Snapshot Status Indicator */}
          <div
            className={`inline-flex items-center gap-1.5 rounded-xl border px-3 py-1.5 text-xs font-semibold ${
              hasSnapshotToday
                ? 'border-emerald-200 bg-emerald-50/70 text-emerald-700'
                : 'border-zinc-200 bg-zinc-50 text-zinc-500'
            }`}
            title={
              hasSnapshotToday
                ? `บันทึก Snapshot ประจำวันนี้ (${todayStr}) แล้ว`
                : 'ยังไม่มี Snapshot วันนี้ — กดปุ่ม "อัปเดตราคาตลาด & Snapshot วันนี้" ด้านบนเพื่อบันทึก'
            }
          >
            <span className={`h-2 w-2 rounded-full ${hasSnapshotToday ? 'bg-emerald-500' : 'bg-amber-400'}`} />
            <span>{hasSnapshotToday ? '✓ มี Snapshot วันนี้แล้ว' : 'ยังไม่มี Snapshot วันนี้'}</span>
          </div>
        </div>
      </div>

      {/* Fallback Notice */}
      {isFallbackApplied && (
        <div className="flex items-center justify-between rounded-xl border border-amber-200 bg-amber-50/70 px-3.5 py-2 text-xs text-amber-800">
          <span>ช่วงเวลา {daysRange} วันล่าสุดยังไม่มี Snapshot</span>
          <button
            type="button"
            onClick={() => onChangeDaysRange(undefined)}
            className="font-bold underline text-amber-900 hover:text-amber-950"
          >
            สลับไปดูประวัติ "ทั้งหมด"
          </button>
        </div>
      )}

      {/* Metrics Row */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 bg-surface rounded-xl p-3 border border-sky-50">
        <div>
          <span className="text-[10px] font-bold uppercase tracking-wider text-zinc-400">Snapshot Date</span>
          <div className="text-xs font-mono font-bold text-zinc-800 mt-0.5">
            {activePoint ? activePoint.date : '—'}
          </div>
        </div>

        <div>
          <span className="text-[10px] font-bold uppercase tracking-wider text-zinc-400">Total NAV</span>
          <div className="text-sm font-mono font-extrabold tabular-nums text-flow-blue mt-0.5">
            {activePoint ? formatTHB(activePoint.nav) : '—'}
          </div>
        </div>

        <div>
          <span className="text-[10px] font-bold uppercase tracking-wider text-zinc-400">Cost Basis</span>
          <div className="text-xs font-mono font-semibold tabular-nums text-zinc-600 mt-0.5">
            {activePoint ? formatTHB(activePoint.cost) : '—'}
          </div>
        </div>

        <div>
          <span className="text-[10px] font-bold uppercase tracking-wider text-zinc-400">Unrealized PnL</span>
          <div
            className={`text-xs font-mono font-bold tabular-nums mt-0.5 ${
              activePoint && activePoint.pnl >= 0 ? 'text-emerald-600' : 'text-rose-600'
            }`}
          >
            {activePoint
              ? `${activePoint.pnl >= 0 ? '+' : ''}${formatTHB(activePoint.pnl)}${
                  latestReturnPct !== null ? ` (${latestReturnPct >= 0 ? '+' : ''}${latestReturnPct.toFixed(1)}%)` : ''
                }`
              : '—'}
          </div>
        </div>
      </div>

      {/* SVG Chart Surface */}
      {sortedRows.length >= 2 ? (
        <div className="relative w-full overflow-hidden">
          <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-auto overflow-visible">
            <defs>
              <linearGradient id="perfCardNavGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#0284c7" stopOpacity={0.22} />
                <stop offset="100%" stopColor="#0284c7" stopOpacity={0.01} />
              </linearGradient>
            </defs>

            {/* Grid lines */}
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
                  x={padLeft - 8}
                  y={gl.y + 4}
                  textAnchor="end"
                  className="text-[10px] font-mono fill-zinc-400"
                >
                  {gl.val >= 1000 ? `${(gl.val / 1000).toFixed(0)}k` : gl.val.toFixed(0)}
                </text>
              </g>
            ))}

            {/* Date Ticks */}
            {Array.from(new Set([0, Math.floor((points.length - 1) / 2), points.length - 1])).map((idx) => {
              const pt = points[idx]
              if (!pt) return null
              const anchor = idx === 0 ? 'start' : idx === points.length - 1 ? 'end' : 'middle'
              return (
                <text
                  key={idx}
                  x={pt.x}
                  y={height - 8}
                  textAnchor={anchor}
                  className="text-[10px] font-mono fill-zinc-400"
                >
                  {pt.date}
                </text>
              )
            })}

            {/* Paths */}
            <path d={navAreaPath} fill="url(#perfCardNavGrad)" />
            <path d={costLinePath} stroke="#94a3b8" strokeWidth="1.5" strokeDasharray="4,4" fill="none" />
            <path d={navLinePath} stroke="#0284c7" strokeWidth="2.5" fill="none" strokeLinecap="round" />

            {/* Active Marker */}
            {hoverIndex !== null && points[hoverIndex] && (
              <g>
                <line
                  x1={points[hoverIndex]!.x}
                  y1={padTop}
                  x2={points[hoverIndex]!.x}
                  y2={height - padBottom}
                  stroke="#0284c7"
                  strokeWidth="1"
                  strokeDasharray="3,3"
                />
                <circle
                  cx={points[hoverIndex]!.x}
                  cy={points[hoverIndex]!.yNav}
                  r={5}
                  fill="#0284c7"
                  stroke="#ffffff"
                  strokeWidth={2}
                />
                <circle
                  cx={points[hoverIndex]!.x}
                  cy={points[hoverIndex]!.yCost}
                  r={3.5}
                  fill="#94a3b8"
                  stroke="#ffffff"
                  strokeWidth={1.5}
                />
              </g>
            )}

            {/* Transparent Tracking Overlay */}
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

          {/* Legend */}
          <div className="flex items-center justify-end gap-4 text-[11px] font-medium pt-2 text-zinc-500">
            <span className="flex items-center gap-1.5">
              <span className="inline-block h-2.5 w-2.5 rounded-sm bg-flow-blue" />
              <span>Total NAV</span>
            </span>
            <span className="flex items-center gap-1.5">
              <span className="inline-block h-0.5 w-3 bg-slate-400 border-t border-dashed border-slate-400" />
              <span>Cost Basis</span>
            </span>
          </div>
        </div>
      ) : (
        <div className="py-12 text-center text-xs text-zinc-400 bg-surface rounded-xl border border-dashed border-sky-100">
          ต้องการข้อมูล Snapshot อย่างน้อย 2 วันเพื่อวาดกราฟแนวโน้ม (ปัจจุบัน: {sortedRows.length} วัน)
          <p className="mt-2 text-[11px] text-zinc-500">
            กดปุ่ม <span className="font-bold text-flow-blue">"อัปเดตราคาตลาด & Snapshot วันนี้"</span> ด้านบนเพื่อเริ่มสร้างประวัติพอร์ต
          </p>
        </div>
      )}

      {/* Collapsible History Table Drawer */}
      <div className="border-t border-sky-100 pt-3">
        <button
          type="button"
          onClick={() => setTableOpen(!tableOpen)}
          className="flex items-center justify-between w-full text-xs font-semibold text-zinc-600 hover:text-zinc-900 transition-colors py-1"
        >
          <span>ดูตารางประวัติ Snapshot ({sortedRows.length} วัน)</span>
          <span className="text-zinc-400 text-sm">{tableOpen ? '▴ ซ่อนตาราง' : '▾ กางตาราง'}</span>
        </button>

        {tableOpen && (
          <div className="mt-3 overflow-x-auto rounded-xl border border-sky-100 animate-fade-in">
            <table className="w-full text-left text-xs">
              <thead>
                <tr className="border-b border-sky-100 bg-sky-50/50 font-bold uppercase text-[10px] text-zinc-500">
                  <th className="px-4 py-2.5">Date</th>
                  <th className="px-4 py-2.5 text-right">Total NAV</th>
                  <th className="px-4 py-2.5 text-right">Total Cost</th>
                  <th className="px-4 py-2.5 text-right">Unrealized PnL</th>
                  <th className="px-4 py-2.5 text-right">Cash Balance</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-sky-50 font-mono tabular-nums">
                {sortedRows.map((r) => {
                  const isPos = r.Unrealized_PnL >= 0
                  return (
                    <tr key={r.Date} className="hover:bg-sky-50/40 transition-colors">
                      <td className="px-4 py-2 font-bold text-zinc-800">{r.Date}</td>
                      <td className="px-4 py-2 text-right font-bold text-zinc-900">{formatTHB(r.Total_NAV)}</td>
                      <td className="px-4 py-2 text-right text-zinc-600">{formatTHB(r.Total_Cost)}</td>
                      <td className={`px-4 py-2 text-right font-bold ${isPos ? 'text-emerald-600' : 'text-rose-600'}`}>
                        {isPos ? '+' : ''}
                        {formatTHB(r.Unrealized_PnL)}
                      </td>
                      <td className="px-4 py-2 text-right text-zinc-600">{formatTHB(r.Cash_Balance)}</td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  )
}
