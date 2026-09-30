import React from 'react'
import type { ThaiFundFlowDTO } from '../../../api/types'
import { SourceProvenanceBadge } from './SourceProvenanceBadge'

interface DivergingFlowBarProps {
  flow: ThaiFundFlowDTO | null
  loading?: boolean
  error?: string | null
  className?: string
}

function formatThbMil(val: number | null | undefined): string {
  if (val === null || val === undefined || isNaN(val)) return '—'
  const inMil = val / 1_000_000
  const sign = inMil > 0 ? '+' : ''
  return `${sign}${inMil.toLocaleString('th-TH', { minimumFractionDigits: 1, maximumFractionDigits: 1 })} ล้านบาท`
}

export const DivergingFlowBar: React.FC<DivergingFlowBarProps> = ({
  flow,
  loading = false,
  error = null,
  className = '',
}) => {
  if (loading) {
    return (
      <div className={`rounded-2xl border border-sky-100 bg-white/90 p-5 shadow-sm animate-pulse ${className}`}>
        <div className="h-6 w-48 bg-slate-200 rounded mb-4" />
        <div className="h-44 bg-slate-100 rounded-xl" />
      </div>
    )
  }

  if (error || !flow || !flow.investors || flow.investors.length === 0) {
    return (
      <div className={`rounded-2xl border border-amber-200 bg-amber-50/60 p-5 text-xs text-amber-900 ${className}`}>
        <div className="font-bold flex items-center gap-1.5">
          <span>⚠️</span>
          <span>SET Investor Fund Flow: ไม่สามารถโหลดข้อมูลได้</span>
        </div>
        <p className="mt-1 text-zinc-600">{error || 'ไม่มีข้อมูลกระแสเงินทุน 4 กลุ่มนักลงทุนจาก Settrade'}</p>
      </div>
    )
  }

  // Calculate max absolute net value for scale
  const maxAbsNet = Math.max(
    1_000_000,
    ...flow.investors.map((i) => Math.abs(i.net_value ?? 0))
  )

  // Sort descending by net_value
  const sorted = [...flow.investors].sort((a, b) => (b.net_value ?? 0) - (a.net_value ?? 0))

  return (
    <div className={`rounded-2xl border border-sky-100 bg-white/90 p-5 shadow-sm ${className}`}>
      {/* Header & Provenance */}
      <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between border-b border-sky-100/70 pb-3">
        <div>
          <h3 className="font-bold text-zinc-900 text-sm tracking-tight flex items-center gap-2">
            <span>SET 4-Investor Flow (Diverging Net Flow)</span>
            <span className="font-mono text-[10px] bg-slate-100 text-zinc-600 font-semibold px-2 py-0.5 rounded">
              {flow.market || 'SET'}
            </span>
          </h3>
          <p className="text-xs text-zinc-500 mt-0.5">
            ยอดซื้อ/ขายสุทธิรายวันจำแนกตามประเภทนักลงทุน (รอบแกนศูนย์ ไม่ใช้ 100% share)
          </p>
        </div>

        <SourceProvenanceBadge
          origin="provider"
          sourceName="Settrade"
          observedAt={flow.as_of}
          compact
        />
      </div>

      {/* Axis Scale Indicator */}
      <div className="mt-4 flex items-center justify-between text-[10px] text-zinc-400 font-semibold uppercase px-1">
        <span>ประเภทนักลงทุน</span>
        <div className="flex items-center gap-16 mr-2">
          <span>← ขายสุทธิ (Net Sell)</span>
          <span>ซื้อสุทธิ (Net Buy) →</span>
        </div>
      </div>

      {/* Bars list */}
      <div className="mt-2 space-y-2.5">
        {sorted.map((item) => {
          const net = item.net_value ?? 0
          const isPos = net >= 0
          const pct = Math.min(50, (Math.abs(net) / maxAbsNet) * 50)
          const isForeign = item.investor_type.toLowerCase().includes('foreign') || item.investor_type.includes('ต่างชาติ')

          return (
            <div
              key={item.investor_type}
              className={`rounded-xl p-3 border transition-all ${
                isForeign
                  ? 'bg-sky-50/70 border-sky-200/80 shadow-2xs'
                  : 'bg-slate-50/80 border-slate-100'
              }`}
            >
              <div className="flex items-center justify-between text-xs mb-1.5">
                <div className="flex items-center gap-2 font-medium text-zinc-800">
                  <span>{item.investor_type}</span>
                  {isForeign && (
                    <span className="rounded bg-sky-600 px-1.5 py-0.2 font-mono text-[9px] font-bold text-white">
                      FOREIGN
                    </span>
                  )}
                </div>
                <div
                  className={`font-mono text-xs font-bold ${
                    isPos ? 'text-emerald-700' : 'text-rose-600'
                  }`}
                >
                  {formatThbMil(net)}
                </div>
              </div>

              {/* Dual-sided Horizontal Diverging Track */}
              <div className="relative h-2.5 w-full overflow-hidden rounded-full bg-slate-200/90 flex">
                {/* Left side (Sell) */}
                <div className="relative w-1/2 h-full flex justify-end">
                  {!isPos && (
                    <div
                      className="h-full bg-rose-500 rounded-l-full transition-all duration-500"
                      style={{ width: `${pct * 2}%` }}
                    />
                  )}
                </div>
                {/* Zero Divider line */}
                <div className="w-[1.5px] h-full bg-zinc-500 z-10" />
                {/* Right side (Buy) */}
                <div className="relative w-1/2 h-full flex justify-start">
                  {isPos && (
                    <div
                      className="h-full bg-emerald-500 rounded-r-full transition-all duration-500"
                      style={{ width: `${pct * 2}%` }}
                    />
                  )}
                </div>
              </div>

              {/* Buy & Sell details */}
              <div className="mt-1 flex items-center justify-between text-[10px] text-zinc-400 font-mono">
                <span>ซื้อ: {(item.buy_value ? item.buy_value / 1e6 : 0).toFixed(1)} M</span>
                <span>ขาย: {(item.sell_value ? item.sell_value / 1e6 : 0).toFixed(1)} M</span>
              </div>
            </div>
          )
        })}
      </div>

      <div className="mt-3 flex items-center justify-between text-[11px] text-zinc-400 border-t border-slate-100 pt-2 font-mono">
        <span>มูลค่าการซื้อขายรวม:</span>
        <span className="font-semibold text-zinc-700">
          {flow.total_value ? `${(flow.total_value / 1e6).toLocaleString()} ล้านบาท` : '—'}
        </span>
      </div>
    </div>
  )
}
