import React from 'react'
import type { MarketBreadthDTO } from '../../../api/types'
import { SourceProvenanceBadge } from './SourceProvenanceBadge'

interface BreadthStackedBarProps {
  breadth: MarketBreadthDTO | null
  loading?: boolean
  error?: string | null
  className?: string
}

export const BreadthStackedBar: React.FC<BreadthStackedBarProps> = ({
  breadth,
  loading = false,
  error = null,
  className = '',
}) => {
  if (loading) {
    return (
      <div className={`rounded-2xl border border-sky-100 bg-white/90 p-5 shadow-sm animate-pulse ${className}`}>
        <div className="h-6 w-48 bg-slate-200 rounded mb-4" />
        <div className="h-32 bg-slate-100 rounded-xl" />
      </div>
    )
  }

  if (error || !breadth) {
    return (
      <div className={`rounded-2xl border border-amber-200 bg-amber-50/60 p-5 text-xs text-amber-900 ${className}`}>
        <div className="font-bold flex items-center gap-1.5">
          <span>⚠️</span>
          <span>SET Market Breadth: ไม่สามารถโหลดข้อมูลได้</span>
        </div>
        <p className="mt-1 text-zinc-600">{error || 'ไม่มีข้อมูลการกระจายตัวของราคาหุ้นจาก Settrade'}</p>
      </div>
    )
  }

  const gainers = typeof breadth.gainers === 'number' ? breadth.gainers : 0
  const losers = typeof breadth.losers === 'number' ? breadth.losers : 0
  const unchanged = typeof breadth.unchanged === 'number' ? breadth.unchanged : 0
  const total = gainers + losers + unchanged

  if (total === 0) {
    return (
      <div className={`rounded-2xl border border-slate-200 bg-slate-50 p-4 text-xs text-zinc-500 text-center ${className}`}>
        ยังไม่มีข้อมูลสัดส่วนหุ้นขึ้น/ลง (Denominator เป็น 0)
      </div>
    )
  }

  const gainPct = (gainers / total) * 100
  const unchPct = (unchanged / total) * 100
  const lossPct = (losers / total) * 100
  const adRatio = losers > 0 ? (gainers / losers).toFixed(2) : '—'

  return (
    <div className={`rounded-2xl border border-sky-100 bg-white/90 p-5 shadow-sm ${className}`}>
      {/* Header & Provenance */}
      <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between border-b border-sky-100/70 pb-3">
        <div>
          <h3 className="font-bold text-zinc-900 text-sm tracking-tight flex items-center gap-2">
            <span>SET Market Breadth (100% Stacked Distribution)</span>
            <span className="font-mono text-[10px] bg-slate-100 text-zinc-600 font-semibold px-2 py-0.5 rounded">
              {breadth.market || 'SET'}
            </span>
          </h3>
          <p className="text-xs text-zinc-500 mt-0.5">
            สัดส่วนการกระจายตัวของหุ้นที่ราคาปรับขึ้น เสมอตัว และปรับลงทั้งตลาด
          </p>
        </div>

        <SourceProvenanceBadge
          origin="provider"
          sourceName="Settrade"
          observedAt={breadth.as_of}
          compact
        />
      </div>

      {/* A/D Ratio metric */}
      <div className="mt-3 flex items-baseline justify-between">
        <div className="text-xs text-zinc-600">
          Advance/Decline Ratio (A/D):
        </div>
        <div className="flex items-baseline gap-1.5 font-mono">
          <span className="text-xl font-extrabold text-zinc-900">{adRatio}x</span>
          <span className={`text-[10px] font-bold uppercase rounded px-1.5 py-0.2 ${
            losers > 0 && gainers / losers >= 1.2
              ? 'bg-emerald-100 text-emerald-800'
              : losers > 0 && gainers / losers <= 0.8
              ? 'bg-rose-100 text-rose-800'
              : 'bg-zinc-100 text-zinc-700'
          }`}>
            {losers > 0 && gainers / losers >= 1.2 ? 'Bullish Breadth' : losers > 0 && gainers / losers <= 0.8 ? 'Bearish Breadth' : 'Neutral'}
          </span>
        </div>
      </div>

      {/* 100% Stacked Bar */}
      <div className="mt-3">
        <div className="h-3 w-full overflow-hidden rounded-full bg-slate-100 flex shadow-inner">
          <div
            style={{ width: `${gainPct}%` }}
            className="bg-emerald-500 transition-all duration-500"
            title={`บวก: ${gainers} ตัว (${gainPct.toFixed(1)}%)`}
          />
          <div
            style={{ width: `${unchPct}%` }}
            className="bg-slate-300 transition-all duration-500"
            title={`เสมอ: ${unchanged} ตัว (${unchPct.toFixed(1)}%)`}
          />
          <div
            style={{ width: `${lossPct}%` }}
            className="bg-rose-500 transition-all duration-500"
            title={`ลบ: ${losers} ตัว (${lossPct.toFixed(1)}%)`}
          />
        </div>
      </div>

      {/* Breakdown counters */}
      <div className="mt-3 grid grid-cols-3 gap-2 text-center text-xs">
        <div className="rounded-lg bg-emerald-50/70 p-2 border border-emerald-100">
          <span className="text-[10px] text-emerald-800 block">ขึ้น (Gainers)</span>
          <span className="font-mono text-base font-extrabold text-emerald-700">{gainers}</span>
          <span className="text-[10px] text-zinc-400 block font-mono">({gainPct.toFixed(1)}%)</span>
        </div>
        <div className="rounded-lg bg-slate-50 p-2 border border-slate-100">
          <span className="text-[10px] text-zinc-600 block">เสมอ (Unchanged)</span>
          <span className="font-mono text-base font-extrabold text-zinc-700">{unchanged}</span>
          <span className="text-[10px] text-zinc-400 block font-mono">({unchPct.toFixed(1)}%)</span>
        </div>
        <div className="rounded-lg bg-rose-50/70 p-2 border border-rose-100">
          <span className="text-[10px] text-rose-800 block">ลง (Losers)</span>
          <span className="font-mono text-base font-extrabold text-rose-700">{losers}</span>
          <span className="text-[10px] text-zinc-400 block font-mono">({lossPct.toFixed(1)}%)</span>
        </div>
      </div>
    </div>
  )
}
