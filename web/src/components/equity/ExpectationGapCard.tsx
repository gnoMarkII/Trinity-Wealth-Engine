import React from 'react'

export interface ExpectationGapCardProps {
  marketImpliedRevenueGrowth?: number | null
  actualRevenueGrowth?: number | null
  marketImpliedOperatingMargin?: number | null
  actualOperatingMargin?: number | null
  solverStatus?: string | null
  pvOf5yFcf?: number | null
  pvOfTerminalValue?: number | null
  enterpriseValue?: number | null
  currency?: string
}

function formatLargeNum(val?: number | null, prefix = '$'): string {
  if (val == null || isNaN(val)) return 'N/A'
  const abs = Math.abs(val)
  if (abs >= 1e12) return `${prefix}${(val / 1e12).toFixed(2)}T`
  if (abs >= 1e9) return `${prefix}${(val / 1e9).toFixed(2)}B`
  if (abs >= 1e6) return `${prefix}${(val / 1e6).toFixed(2)}M`
  return `${prefix}${val.toLocaleString()}`
}

export const ExpectationGapCard: React.FC<ExpectationGapCardProps> = ({
  marketImpliedRevenueGrowth,
  actualRevenueGrowth,
  marketImpliedOperatingMargin,
  actualOperatingMargin,
  solverStatus,
  pvOf5yFcf,
  pvOfTerminalValue,
  enterpriseValue,
  currency = '$',
}) => {
  const hasGrowth = marketImpliedRevenueGrowth != null && actualRevenueGrowth != null
  const hasMargin = marketImpliedOperatingMargin != null && actualOperatingMargin != null

  if (!hasGrowth && !hasMargin) return null

  const growthGap = hasGrowth ? (actualRevenueGrowth! - marketImpliedRevenueGrowth!) : null
  const marginGap = hasMargin ? (actualOperatingMargin! - marketImpliedOperatingMargin!) : null

  // Value distribution percentages
  const totalPv = (pvOf5yFcf || 0) + (pvOfTerminalValue || 0)
  const fcfPct = totalPv > 0 && pvOf5yFcf ? (pvOf5yFcf / totalPv) * 100 : 25
  const tvPct = totalPv > 0 && pvOfTerminalValue ? (pvOfTerminalValue / totalPv) * 100 : 75

  return (
    <div className="rounded-2xl border border-edge/90 bg-surface/80 p-5 shadow-xs space-y-5 animate-card-in">
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-edge/60 pb-3">
        <div className="flex items-center gap-2">
          <span className="text-lg">⚖️</span>
          <div>
            <h4 className="text-sm font-bold text-zinc-900 tracking-tight">
              Reverse DCF: Market Expectation vs Reality Gap
            </h4>
            <p className="text-xs text-zinc-500">
              ราคาปัจจุบันตลาดแฝงความคาดหวังไว้ต่ำหรือสูงกว่าความสามารถจริงของกิจการ
            </p>
          </div>
        </div>

        {solverStatus && (
          <span className="px-2.5 py-0.5 rounded-full text-[10px] font-bold uppercase tracking-wider bg-zinc-100 text-zinc-700 border border-zinc-200">
            Solver: {solverStatus}
          </span>
        )}
      </div>

      {/* 2-Column Comparison Bar Matrix */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
        {/* Metric 1: Revenue Growth YoY */}
        {hasGrowth && (
          <div className="p-4 rounded-xl border border-edge bg-panel/60 space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold text-zinc-500 uppercase tracking-wider">
                1. Revenue Growth YoY
              </span>
              {growthGap != null && (
                <span
                  className={`text-xs px-2 py-0.5 rounded-full font-bold border ${
                    growthGap >= 0
                      ? 'bg-emerald-50 text-emerald-700 border-emerald-300'
                      : 'bg-rose-50 text-rose-700 border-rose-300'
                  }`}
                >
                  {growthGap >= 0 ? `+${growthGap.toFixed(2)}% Beat` : `${growthGap.toFixed(2)}% Miss`}
                </span>
              )}
            </div>

            {/* Big Number Tier 2 Display */}
            <div className="grid grid-cols-2 gap-2 text-center py-1">
              <div className="p-2 rounded-lg bg-surface border border-edge/80">
                <span className="text-[10px] font-bold uppercase tracking-wider text-zinc-400 block mb-0.5">
                  ตลาดคาดหวัง (Implied)
                </span>
                <span className="font-mono text-xl sm:text-2xl font-bold text-zinc-600">
                  {marketImpliedRevenueGrowth?.toFixed(2)}%
                </span>
              </div>
              <div className="p-2 rounded-lg bg-emerald-500/10 border border-emerald-500/30">
                <span className="text-[10px] font-bold uppercase tracking-wider text-emerald-700 block mb-0.5">
                  ผลงานจริง (Actual)
                </span>
                <span className="font-mono text-xl sm:text-2xl font-bold text-emerald-600">
                  {actualRevenueGrowth?.toFixed(2)}%
                </span>
              </div>
            </div>

            {/* Dual Comparative Bar */}
            {(() => {
              const maxG = Math.max(Math.abs(actualRevenueGrowth || 0), Math.abs(marketImpliedRevenueGrowth || 0), 1)
              const impW = Math.min(100, Math.max(10, (Math.abs(marketImpliedRevenueGrowth || 0) / maxG) * 100))
              const actW = Math.min(100, Math.max(10, (Math.abs(actualRevenueGrowth || 0) / maxG) * 100))
              return (
                <div className="space-y-1.5 pt-1">
                  <div className="flex justify-between text-[11px] text-zinc-500">
                    <span>ความคาดหวังในราคา</span>
                    <span>{marketImpliedRevenueGrowth?.toFixed(1)}%</span>
                  </div>
                  <div className="h-2 w-full rounded-full bg-zinc-200 overflow-hidden">
                    <div
                      className="h-full bg-zinc-400 rounded-full"
                      style={{ width: `${impW}%` }}
                    />
                  </div>

                  <div className="flex justify-between text-[11px] text-emerald-700 font-semibold pt-1">
                    <span>อัตราเติบโตจริงของบริษัท</span>
                    <span>{actualRevenueGrowth?.toFixed(1)}%</span>
                  </div>
                  <div className="h-2.5 w-full rounded-full bg-emerald-100 overflow-hidden">
                    <div
                      className="h-full bg-emerald-500 rounded-full transition-all"
                      style={{ width: `${actW}%` }}
                    />
                  </div>
                </div>
              )
            })()}
          </div>
        )}

        {/* Metric 2: Operating (EBIT) Margin */}
        {hasMargin && (
          <div className="p-4 rounded-xl border border-edge bg-panel/60 space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold text-zinc-500 uppercase tracking-wider">
                2. Operating (EBIT) Margin
              </span>
              {marginGap != null && (
                <span
                  className={`text-xs px-2 py-0.5 rounded-full font-bold border ${
                    marginGap >= 0
                      ? 'bg-emerald-50 text-emerald-700 border-emerald-300'
                      : 'bg-rose-50 text-rose-700 border-rose-300'
                  }`}
                >
                  {marginGap >= 0 ? `+${marginGap.toFixed(2)}% Beat` : `${marginGap.toFixed(2)}% Miss`}
                </span>
              )}
            </div>

            {/* Big Number Tier 2 Display */}
            <div className="grid grid-cols-2 gap-2 text-center py-1">
              <div className="p-2 rounded-lg bg-surface border border-edge/80">
                <span className="text-[10px] font-bold uppercase tracking-wider text-zinc-400 block mb-0.5">
                  ตลาดคาดหวัง (Implied)
                </span>
                <span className="font-mono text-xl sm:text-2xl font-bold text-zinc-600">
                  {marketImpliedOperatingMargin?.toFixed(2)}%
                </span>
              </div>
              <div className="p-2 rounded-lg bg-emerald-500/10 border border-emerald-500/30">
                <span className="text-[10px] font-bold uppercase tracking-wider text-emerald-700 block mb-0.5">
                  GAAP จริง (Actual)
                </span>
                <span className="font-mono text-xl sm:text-2xl font-bold text-emerald-600">
                  {actualOperatingMargin?.toFixed(2)}%
                </span>
              </div>
            </div>

            {/* Dual Comparative Bar */}
            {(() => {
              const maxM = Math.max(Math.abs(actualOperatingMargin || 0), Math.abs(marketImpliedOperatingMargin || 0), 1)
              const impMW = Math.min(100, Math.max(10, (Math.abs(marketImpliedOperatingMargin || 0) / maxM) * 100))
              const actMW = Math.min(100, Math.max(10, (Math.abs(actualOperatingMargin || 0) / maxM) * 100))
              return (
                <div className="space-y-1.5 pt-1">
                  <div className="flex justify-between text-[11px] text-zinc-500">
                    <span>อัตรากำไรที่ราคาตลาดแฝงไว้</span>
                    <span>{marketImpliedOperatingMargin?.toFixed(1)}%</span>
                  </div>
                  <div className="h-2 w-full rounded-full bg-zinc-200 overflow-hidden">
                    <div
                      className="h-full bg-zinc-400 rounded-full"
                      style={{ width: `${impMW}%` }}
                    />
                  </div>

                  <div className="flex justify-between text-[11px] text-emerald-700 font-semibold pt-1">
                    <span>อัตรากำไร GAAP จริงล่าสุด</span>
                    <span>{actualOperatingMargin?.toFixed(1)}%</span>
                  </div>
                  <div className="h-2.5 w-full rounded-full bg-emerald-100 overflow-hidden">
                    <div
                      className="h-full bg-emerald-500 rounded-full transition-all"
                      style={{ width: `${actMW}%` }}
                    />
                  </div>
                </div>
              )
            })()}
          </div>
        )}
      </div>

      {/* Enterprise Value Capital Distribution (Stacked Bar) */}
      {(pvOf5yFcf != null || pvOfTerminalValue != null) && (
        <div className="p-4 rounded-xl border border-edge bg-panel/40 space-y-2">
          <div className="flex items-center justify-between text-xs font-semibold text-zinc-600">
            <span>Enterprise Value Breakdown: {formatLargeNum(enterpriseValue, currency)}</span>
            <span className="text-zinc-400 text-[11px]">
              PV of 5Y FCF ({fcfPct.toFixed(1)}%) + Terminal Value ({tvPct.toFixed(1)}%)
            </span>
          </div>

          <div className="h-3 w-full rounded-full overflow-hidden flex bg-zinc-200">
            <div
              className="h-full bg-sky-500"
              style={{ width: `${fcfPct}%` }}
              title={`PV of 5-Year Free Cash Flow: ${formatLargeNum(pvOf5yFcf, currency)} (${fcfPct.toFixed(1)}%)`}
            />
            <div
              className="h-full bg-indigo-500"
              style={{ width: `${tvPct}%` }}
              title={`PV of Terminal Value: ${formatLargeNum(pvOfTerminalValue, currency)} (${tvPct.toFixed(1)}%)`}
            />
          </div>

          <div className="flex items-center justify-between text-[11px] text-zinc-500 pt-0.5">
            <span className="flex items-center gap-1.5">
              <span className="w-2 h-2 rounded-full bg-sky-500" />
              <span>5Y FCF: {formatLargeNum(pvOf5yFcf, currency)}</span>
            </span>
            <span className="flex items-center gap-1.5">
              <span className="w-2 h-2 rounded-full bg-indigo-500" />
              <span>Terminal Value: {formatLargeNum(pvOfTerminalValue, currency)}</span>
            </span>
          </div>
        </div>
      )}
    </div>
  )
}

export default ExpectationGapCard
