import React, { useState } from 'react'

export interface StrikeOiItem {
  strike: number
  callOi: number
  putOi: number
  callVol?: number
  putVol?: number
}

export interface OptionsOiStrikeLadderProps {
  symbol: string
  strikes: StrikeOiItem[]
  currentPrice?: number
  maxPainStrike?: number
  expirationDate?: string
  putCallRatio?: number
  className?: string
}

export const OptionsOiStrikeLadder: React.FC<OptionsOiStrikeLadderProps> = ({
  symbol,
  strikes,
  currentPrice,
  maxPainStrike,
  expirationDate,
  putCallRatio,
  className = '',
}) => {
  const [hoveredStrike, setHoveredStrike] = useState<StrikeOiItem | null>(null)

  // Sort strikes numerically
  const sortedStrikes = React.useMemo(() => {
    return [...strikes].sort((a, b) => a.strike - b.strike)
  }, [strikes])

  // Max OI for scale normalization
  const maxOi = React.useMemo(() => {
    let m = 1
    strikes.forEach((s) => {
      if (s.callOi > m) m = s.callOi
      if (s.putOi > m) m = s.putOi
    })
    return m
  }, [strikes])

  const totals = React.useMemo(() => {
    const totalCallOi = strikes.reduce((sum, s) => sum + s.callOi, 0)
    const totalPutOi = strikes.reduce((sum, s) => sum + s.putOi, 0)
    const pcr = totalCallOi > 0 ? totalPutOi / totalCallOi : 0
    return { totalCallOi, totalPutOi, pcr: putCallRatio ?? pcr }
  }, [strikes, putCallRatio])

  return (
    <div
      className={`rounded-2xl border border-sky-100 bg-white/80 p-5 shadow-[0_8px_25px_rgba(14,165,233,0.06)] backdrop-blur-md ${className}`}
    >
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-sky-100/60 pb-3">
        <div>
          <div className="flex items-center gap-2">
            <h3 className="font-semibold text-zinc-900 tracking-tight">
              {symbol} Options Open Interest by Strike
            </h3>
            {expirationDate && (
              <span className="rounded-full bg-sky-50 px-2 py-0.5 font-mono text-[11px] font-medium text-sky-700">
                Exp: {expirationDate}
              </span>
            )}
          </div>
          <p className="text-xs text-zinc-500 mt-0.5">
            Cboe Institutional Options Chain • Call OI (Left) vs Put OI (Right)
          </p>
        </div>

        {/* Stats Summary */}
        <div className="flex items-center gap-2 text-xs">
          <div className="rounded-lg bg-emerald-50 px-2.5 py-1 text-right">
            <span className="block text-[10px] uppercase font-bold text-emerald-700">Total Calls</span>
            <span className="font-mono font-bold text-emerald-800">
              {totals.totalCallOi.toLocaleString()}
            </span>
          </div>

          <div className="rounded-lg bg-rose-50 px-2.5 py-1 text-right">
            <span className="block text-[10px] uppercase font-bold text-rose-700">Total Puts</span>
            <span className="font-mono font-bold text-rose-800">
              {totals.totalPutOi.toLocaleString()}
            </span>
          </div>

          <div className="rounded-lg bg-sky-50 px-2.5 py-1 text-right">
            <span className="block text-[10px] uppercase font-bold text-sky-700">P/C Ratio</span>
            <span className="font-mono font-bold text-sky-800">{totals.pcr.toFixed(2)}</span>
          </div>
        </div>
      </div>

      {/* Semantic Caution Banner */}
      <div className="mt-3 flex items-start gap-2 rounded-lg bg-amber-50/70 p-2 text-[11px] text-amber-800 border border-amber-200/50">
        <span className="font-bold text-amber-600 shrink-0">ℹ Invariant Note:</span>
        <span>
          Max Pain is calculated from cumulative open interest at expiration — it represents the strike
          where aggregate option buyers expire with minimum value, not a guaranteed target price or
          definitive market maker pin.
        </span>
      </div>

      {/* Ladder Chart */}
      <div className="mt-4 flex flex-col gap-1 max-h-[380px] overflow-y-auto pr-1">
        {sortedStrikes.map((row) => {
          const callWidth = (row.callOi / maxOi) * 100
          const putWidth = (row.putOi / maxOi) * 100
          const isMaxPain = maxPainStrike !== undefined && Math.abs(row.strike - maxPainStrike) < 0.01
          const isAtTheMoney =
            currentPrice !== undefined &&
            Math.abs(row.strike - currentPrice) ===
              Math.min(...sortedStrikes.map((s) => Math.abs(s.strike - currentPrice)))
          const isHovered = hoveredStrike?.strike === row.strike

          return (
            <div
              key={row.strike}
              onMouseEnter={() => setHoveredStrike(row)}
              onMouseLeave={() => setHoveredStrike(null)}
              className={`flex items-center gap-2 rounded-md py-1 px-1.5 transition-colors ${
                isHovered
                  ? 'bg-sky-50/80 ring-1 ring-sky-300'
                  : isMaxPain
                  ? 'bg-amber-50/60'
                  : 'hover:bg-slate-50'
              }`}
            >
              {/* Call OI Bar (Right-aligned, extends to left) */}
              <div className="flex flex-1 items-center justify-end gap-1.5">
                <span className="font-mono text-[11px] text-zinc-500 w-12 text-right">
                  {row.callOi.toLocaleString()}
                </span>
                <div className="h-4 flex-1 max-w-[160px] flex justify-end bg-slate-100 rounded-sm overflow-hidden">
                  <div
                    className="h-full rounded-l-sm bg-emerald-500 transition-all duration-300"
                    style={{ width: `${callWidth}%` }}
                  />
                </div>
              </div>

              {/* Center Strike Marker */}
              <div
                className={`w-20 shrink-0 text-center font-mono text-xs font-bold py-0.5 rounded ${
                  isMaxPain
                    ? 'bg-amber-500 text-white shadow-sm'
                    : isAtTheMoney
                    ? 'bg-zinc-800 text-white'
                    : 'bg-slate-100 text-zinc-700'
                }`}
              >
                ${row.strike.toFixed(1)}
              </div>

              {/* Put OI Bar (Left-aligned, extends to right) */}
              <div className="flex flex-1 items-center justify-start gap-1.5">
                <div className="h-4 flex-1 max-w-[160px] bg-slate-100 rounded-sm overflow-hidden">
                  <div
                    className="h-full rounded-r-sm bg-rose-500 transition-all duration-300"
                    style={{ width: `${putWidth}%` }}
                  />
                </div>
                <span className="font-mono text-[11px] text-zinc-500 w-12 text-left">
                  {row.putOi.toLocaleString()}
                </span>
              </div>
            </div>
          )
        })}
      </div>

      {/* Legend & Current Price Marker */}
      <div className="mt-4 flex flex-wrap items-center justify-between gap-3 border-t border-sky-100/60 pt-3 text-xs text-zinc-500">
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-1.5">
            <span className="h-3 w-3 rounded-sm bg-emerald-500" />
            <span className="font-medium text-zinc-700">Call Open Interest</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="h-3 w-3 rounded-sm bg-rose-500" />
            <span className="font-medium text-zinc-700">Put Open Interest</span>
          </div>
        </div>

        <div className="flex items-center gap-3">
          {currentPrice && (
            <div className="flex items-center gap-1.5">
              <span className="inline-block px-1.5 py-0.5 rounded bg-zinc-800 text-white font-mono text-[10px] font-bold">
                ${currentPrice.toFixed(2)}
              </span>
              <span>Spot Price</span>
            </div>
          )}
          {maxPainStrike && (
            <div className="flex items-center gap-1.5">
              <span className="inline-block px-1.5 py-0.5 rounded bg-amber-500 text-white font-mono text-[10px] font-bold">
                ${maxPainStrike.toFixed(1)}
              </span>
              <span className="font-semibold text-amber-700">Max Pain Strike</span>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
