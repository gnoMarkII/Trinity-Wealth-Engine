import React from 'react'
import type { PriceMoveDecompositionDTO } from '../../api/types'

export interface PriceReturnDecompositionBarProps {
  decomposition: PriceMoveDecompositionDTO
  className?: string
}

export const PriceReturnDecompositionBar: React.FC<PriceReturnDecompositionBarProps> = ({
  decomposition,
  className = '',
}) => {
  const {
    horizon_label,
    total_return_pct,
    eps_growth_pct,
    pe_expansion_pct,
    diagnosis_tier,
    diagnosis_summary,
  } = decomposition

  // Calculate relative proportion for stacked visual representation
  const epsAbs = Math.max(0.1, Math.abs(eps_growth_pct))
  const peAbs = Math.max(0.1, Math.abs(pe_expansion_pct))
  const totalWeight = epsAbs + peAbs
  const epsRatio = Math.round((epsAbs / totalWeight) * 100)
  const peRatio = 100 - epsRatio

  const isFundamentalDriven = diagnosis_tier.includes('fundamental') || eps_growth_pct > pe_expansion_pct
  const isSpeculative = diagnosis_tier.includes('speculative') || pe_expansion_pct > eps_growth_pct

  return (
    <div
      data-testid="price-return-decomposition-bar"
      className={`rounded-xl border border-edge/80 bg-surface/90 p-4 transition-all space-y-3.5 ${className}`}
    >
      {/* Header: Horizon & Diagnosis Tier Badge */}
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <span className="text-base">📈</span>
          <span className="text-xs font-bold text-zinc-900 tracking-tight">
            Price Move Decomposition ({horizon_label})
          </span>
        </div>
        <span
          className={`px-2.5 py-0.5 text-[10px] font-bold rounded-full uppercase tracking-wider border ${
            isFundamentalDriven
              ? 'bg-emerald-50 text-emerald-800 border-emerald-300'
              : isSpeculative
              ? 'bg-amber-50 text-amber-800 border-amber-300'
              : 'bg-sky-50 text-sky-800 border-sky-300'
          }`}
        >
          {diagnosis_tier.replace(/_/g, ' ').toUpperCase()}
        </span>
      </div>

      {/* 3-Column Metric Highlights with Tier-1 Giant Numbers */}
      <div className="grid grid-cols-3 gap-2.5 text-center">
        {/* Total Return */}
        <div className="p-2.5 rounded-lg bg-surface border border-edge/60">
          <span className="text-[10px] font-semibold text-zinc-500 uppercase tracking-wider block">
            Total Return
          </span>
          <span
            className={`text-lg sm:text-xl font-black tracking-tight ${
              total_return_pct >= 0 ? 'text-emerald-700' : 'text-rose-700'
            }`}
          >
            {total_return_pct >= 0 ? '+' : ''}
            {total_return_pct.toFixed(1)}%
          </span>
        </div>

        {/* EPS Growth Driver */}
        <div className="p-2.5 rounded-lg bg-surface border border-edge/60">
          <span className="text-[10px] font-semibold text-zinc-500 uppercase tracking-wider block">
            Earnings Growth
          </span>
          <span
            className={`text-lg sm:text-xl font-black tracking-tight ${
              eps_growth_pct >= 0 ? 'text-emerald-700' : 'text-rose-700'
            }`}
          >
            {eps_growth_pct >= 0 ? '+' : ''}
            {eps_growth_pct.toFixed(1)}%
          </span>
        </div>

        {/* P/E Expansion Driver */}
        <div className="p-2.5 rounded-lg bg-surface border border-edge/60">
          <span className="text-[10px] font-semibold text-zinc-500 uppercase tracking-wider block">
            P/E Expansion
          </span>
          <span
            className={`text-lg sm:text-xl font-black tracking-tight ${
              pe_expansion_pct >= 0 ? 'text-amber-700' : 'text-zinc-600'
            }`}
          >
            {pe_expansion_pct >= 0 ? '+' : ''}
            {pe_expansion_pct.toFixed(1)}%
          </span>
        </div>
      </div>

      {/* Visual Stacked Waterfall Breakdown Bar */}
      <div className="space-y-1.5 pt-1">
        <div className="flex items-center justify-between text-[11px] font-medium text-zinc-500">
          <span className="flex items-center gap-1.5 text-emerald-800">
            <span className="w-2 h-2 rounded-full bg-emerald-500 inline-block" />
            Earnings Driver ({epsRatio}%)
          </span>
          <span className="flex items-center gap-1.5 text-amber-800">
            <span className="w-2 h-2 rounded-full bg-amber-500 inline-block" />
            Valuation Multiple ({peRatio}%)
          </span>
        </div>

        <div className="w-full h-3 rounded-full bg-zinc-200/80 overflow-hidden flex shadow-inner">
          <div
            className="h-full bg-gradient-to-r from-emerald-500 to-emerald-600 transition-all duration-500"
            style={{ width: `${epsRatio}%` }}
            title={`Earnings Growth: ${eps_growth_pct.toFixed(1)}% (${epsRatio}% of move)`}
          />
          <div
            className="h-full bg-gradient-to-r from-amber-400 to-amber-500 transition-all duration-500"
            style={{ width: `${peRatio}%` }}
            title={`P/E Multiple: ${pe_expansion_pct.toFixed(1)}% (${peRatio}% of move)`}
          />
        </div>
      </div>

      {/* Summary Narrative */}
      {diagnosis_summary && (
        <div className="rounded-lg bg-zinc-50/90 border border-zinc-200/80 p-2.5 text-xs text-zinc-700 leading-relaxed">
          <span className="font-semibold text-zinc-900 block mb-0.5">💡 Driver Assessment:</span>
          {diagnosis_summary}
        </div>
      )}
    </div>
  )
}

export default PriceReturnDecompositionBar
