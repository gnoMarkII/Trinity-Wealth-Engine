import React from 'react'
import type { CommodityVolSnapshotDTO } from '../../api/types'

interface CommodityVolCardProps {
  indices: CommodityVolSnapshotDTO[]
  className?: string
}

const REGIME_BADGE_STYLE: Record<string, { bg: string; text: string; border: string; label: string }> = {
  complacent: { bg: 'bg-emerald-950/60', text: 'text-emerald-400', border: 'border-emerald-800/60', label: 'Complacent' },
  normal: { bg: 'bg-sky-950/60', text: 'text-sky-400', border: 'border-sky-800/60', label: 'Normal' },
  elevated: { bg: 'bg-amber-950/60', text: 'text-amber-400', border: 'border-amber-800/60', label: 'Elevated' },
  extreme_panic: { bg: 'bg-rose-950/60', text: 'text-rose-400', border: 'border-rose-800/60', label: 'Extreme Panic' },
}

export const CommodityVolCard: React.FC<CommodityVolCardProps> = ({ indices, className = '' }) => {
  if (!indices || indices.length === 0) return null

  return (
    <div className={`rounded-xl border border-slate-800 bg-slate-900/90 p-5 shadow-lg backdrop-blur-sm ${className}`}>
      <div className="mb-4 flex flex-wrap items-baseline justify-between gap-2 border-b border-slate-800/80 pb-3">
        <div>
          <h3 className="text-base font-semibold text-slate-100 flex items-center gap-2">
            <span>Commodity Volatility Radar</span>
            <span className="rounded-full bg-slate-800 text-slate-300 font-mono text-[10px] font-bold px-2 py-0.5">
              CBOE 30D IV
            </span>
          </h3>
          <p className="text-xs text-slate-400">
            Gold (GVZ), Silver (VXSLV), and Crude Oil (OVX) implied volatility & 52-week percentiles
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {indices.map((idx) => {
          const regime = (idx.regime_label ? REGIME_BADGE_STYLE[idx.regime_label] : undefined) || {
            bg: 'bg-slate-800',
            text: 'text-slate-300',
            border: 'border-slate-700',
            label: idx.regime_label || 'N/A',
          }

          const changePts = idx.change_1d_points !== null && idx.change_1d_points !== undefined
            ? `${idx.change_1d_points >= 0 ? '+' : ''}${idx.change_1d_points.toFixed(2)} pts`
            : 'N/A'

          return (
            <div
              key={idx.index_symbol}
              className="rounded-lg border border-slate-800 bg-slate-950/70 p-4 transition-all duration-150 hover:border-slate-700"
            >
              <div className="flex items-center justify-between">
                <span className="font-mono text-sm font-bold text-slate-100">{idx.index_symbol}</span>
                <span className={`rounded-md border px-2 py-0.5 text-[10px] font-bold uppercase tracking-wider ${regime.bg} ${regime.text} ${regime.border}`}>
                  {regime.label}
                </span>
              </div>

              <div className="mt-1 text-[11px] text-slate-400 truncate">
                {idx.underlying_instrument}
              </div>

              <div className="mt-3 flex items-baseline justify-between">
                <div>
                  <span className="font-mono text-2xl font-bold text-slate-100">
                    {typeof idx.implied_volatility === 'number' && !isNaN(idx.implied_volatility)
                      ? `${idx.implied_volatility.toFixed(2)}%`
                      : 'N/A'}
                  </span>
                  <span className="ml-1 text-[11px] text-slate-500">IV</span>
                </div>
                <div className={`font-mono text-xs font-semibold ${
                  (idx.change_1d_points ?? 0) >= 0 ? 'text-rose-400' : 'text-emerald-400'
                }`}>
                  {changePts}
                </div>
              </div>

              <div className="mt-3 flex items-center justify-between border-t border-slate-800/80 pt-2 text-[11px]">
                <span className="text-slate-400">52W Percentile:</span>
                <span className="font-mono font-semibold text-sky-400">
                  {idx.percentile_52w !== null && idx.percentile_52w !== undefined
                    ? `${idx.percentile_52w.toFixed(1)}%`
                    : 'N/A (< 100d)'}
                </span>
              </div>
              <div className="flex items-center justify-between text-[11px] text-slate-500">
                <span>Close Date:</span>
                <span>{idx.close_date}</span>
              </div>
            </div>
          )
        })}
      </div>

      <div className="mt-3 text-[11px] text-slate-500 flex items-center gap-1.5">
        <span>ℹ️</span>
        <span>
          Measures 30-day annualized implied volatility of ETF options (GLD, SLV, USO). Regime label is a statistical heuristic from 52-week close percentiles.
        </span>
      </div>
    </div>
  )
}
