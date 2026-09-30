import React, { useState, useMemo } from 'react'

export interface VolSmileContract {
  strike: number
  option_type: 'call' | 'put'
  implied_volatility?: number | null
  open_interest?: number | null
  volume?: number | null
}

export interface OptionsVolSmileProps {
  symbol: string
  expiry: string
  currentPrice?: number | null
  contracts: VolSmileContract[]
  delayMinutes?: number
  asOfDate?: string
  title?: string
  height?: number
  className?: string
}

export const OptionsVolSmile: React.FC<OptionsVolSmileProps> = ({
  symbol,
  expiry,
  currentPrice,
  contracts,
  delayMinutes = 15,
  asOfDate,
  title,
  height = 340,
  className = '',
}) => {
  const [hoveredStrike, setHoveredStrike] = useState<{
    strike: number
    callIv?: number | null
    putIv?: number | null
    x: number
  } | null>(null)

  const width = 800
  const padLeft = 65
  const padRight = 35
  const padTop = 30
  const padBottom = 45
  const plotWidth = width - padLeft - padRight
  const plotHeight = height - padTop - padBottom

  // Normalizes IV to percentage (e.g. 0.25 -> 25.0, 25.0 -> 25.0)
  const normalizeIv = (rawIv?: number | null): number | null => {
    if (rawIv === null || rawIv === undefined || isNaN(rawIv) || rawIv <= 0) {
      return null
    }
    // If <= 2.5, assume decimal form (e.g. 0.25 -> 25%)
    // If > 2.5, assume already percent form (e.g. 25.0 -> 25%)
    return rawIv <= 2.5 ? rawIv * 100.0 : rawIv
  }

  // Filter and organize by strike
  const { strikeMap, sortedStrikes, minIv, maxIv, hasSufficientQuotes } = useMemo(() => {
    const map = new Map<number, { callIv?: number | null; putIv?: number | null }>()

    for (const c of contracts) {
      if (!c.strike || c.strike <= 0) continue
      const iv = normalizeIv(c.implied_volatility)
      if (iv === null) continue

      const entry = map.get(c.strike) || {}
      if (c.option_type === 'call') entry.callIv = iv
      if (c.option_type === 'put') entry.putIv = iv
      map.set(c.strike, entry)
    }

    const strikes = Array.from(map.keys()).sort((a, b) => a - b)
    const validIvs: number[] = []

    strikes.forEach((s) => {
      const e = map.get(s)!
      if (e.callIv !== undefined && e.callIv !== null) validIvs.push(e.callIv)
      if (e.putIv !== undefined && e.putIv !== null) validIvs.push(e.putIv)
    })

    const hasEnough = strikes.length >= 3 && validIvs.length >= 3
    const rawMin = validIvs.length > 0 ? Math.min(...validIvs) : 10
    const rawMax = validIvs.length > 0 ? Math.max(...validIvs) : 50

    return {
      strikeMap: map,
      sortedStrikes: strikes,
      minIv: Math.max(0, Math.floor(rawMin * 0.85)),
      maxIv: Math.ceil(rawMax * 1.15),
      hasSufficientQuotes: hasEnough,
    }
  }, [contracts])

  // Coordinate scales
  const minStrike: number = sortedStrikes[0] ?? 0
  const maxStrike: number = sortedStrikes[sortedStrikes.length - 1] ?? 100
  const strikeSpan: number = maxStrike - minStrike || 1
  const ivSpan: number = maxIv - minIv || 1

  const getX = (strike: number) => padLeft + ((strike - minStrike) / strikeSpan) * plotWidth
  const getY = (iv: number) => padTop + plotHeight - ((iv - minIv) / ivSpan) * plotHeight

  // Generate SVG path for Call and Put IV curves
  const { callPath, putPath, callPoints, putPoints } = useMemo(() => {
    let cp = ''
    let pp = ''
    const cPoints: Array<{ x: number; y: number; strike: number; iv: number }> = []
    const pPoints: Array<{ x: number; y: number; strike: number; iv: number }> = []

    sortedStrikes.forEach((s) => {
      const entry = strikeMap.get(s)
      if (!entry) return
      const x = getX(s)

      if (entry.callIv !== undefined && entry.callIv !== null) {
        const y = getY(entry.callIv)
        cPoints.push({ x, y, strike: s, iv: entry.callIv })
        cp += cp === '' ? `M ${x},${y}` : ` L ${x},${y}`
      }

      if (entry.putIv !== undefined && entry.putIv !== null) {
        const y = getY(entry.putIv)
        pPoints.push({ x, y, strike: s, iv: entry.putIv })
        pp += pp === '' ? `M ${x},${y}` : ` L ${x},${y}`
      }
    })

    return { callPath: cp, putPath: pp, callPoints: cPoints, putPoints: pPoints }
  }, [sortedStrikes, strikeMap, minStrike, strikeSpan, minIv, ivSpan, plotWidth, plotHeight, padLeft, padTop])

  if (!hasSufficientQuotes) {
    return (
      <div className={`rounded-xl border border-slate-800 bg-slate-900/80 p-6 text-center ${className}`}>
        {title && <h4 className="text-sm font-semibold text-slate-300">{title}</h4>}
        <div className="mt-3 text-sm font-medium text-amber-400">
          ⚠️ Insufficient implied volatility quotes to render smile (unavailable)
        </div>
        <p className="mt-2 text-xs text-slate-500">
          Options chain for {symbol} ({expiry}) does not contain at least 3 valid IV strikes. Values are not artificially interpolated.
        </p>
      </div>
    )
  }

  const currentPriceX = currentPrice && currentPrice >= minStrike && currentPrice <= maxStrike ? getX(currentPrice) : null

  return (
    <div className={`relative rounded-xl border border-slate-800 bg-slate-900/90 p-5 shadow-lg backdrop-blur-sm ${className}`}>
      <div className="mb-4 flex flex-wrap items-baseline justify-between gap-2 border-b border-slate-800/80 pb-3">
        <div>
          <h3 className="text-base font-semibold text-slate-100">
            {title || `${symbol} Implied Volatility Smile`}
          </h3>
          <p className="text-xs text-slate-400">
            Expiry: <span className="font-semibold text-slate-200">{expiry}</span> (Single Expiry View)
            {asOfDate && <span> • As of: {asOfDate}</span>}
          </p>
        </div>
        <div className="text-right">
          <span className="inline-flex items-center rounded-md border border-amber-800/60 bg-amber-950/40 px-2 py-0.5 text-[11px] text-amber-400">
            ⏱ Delayed by {delayMinutes}m (Cboe)
          </span>
        </div>
      </div>

      {/* Legend */}
      <div className="mb-3 flex flex-wrap items-center gap-5 text-xs">
        <div className="flex items-center gap-1.5">
          <span className="h-0.5 w-4 bg-emerald-400" />
          <span className="text-slate-300">Call Implied Volatility (IV)</span>
        </div>
        <div className="flex items-center gap-1.5">
          <span className="h-0.5 w-4 bg-amber-400" />
          <span className="text-slate-300">Put Implied Volatility (IV)</span>
        </div>
        {currentPrice && (
          <div className="flex items-center gap-1.5">
            <span className="h-0.5 w-3 border-t border-dashed border-sky-400" />
            <span className="text-slate-300">Spot Price (${currentPrice.toFixed(2)})</span>
          </div>
        )}
      </div>

      <div className="relative w-full overflow-hidden rounded-lg border border-slate-800 bg-slate-950">
        <svg
          viewBox={`0 0 ${width} ${height}`}
          className="h-full w-full select-none"
          preserveAspectRatio="none"
          role="figure"
          aria-label={`${symbol} Options Volatility Smile for ${expiry}`}
        >
          {/* Y Axis Grid Lines & Labels */}
          {[0, 0.25, 0.5, 0.75, 1].map((frac, idx) => {
            const y = padTop + plotHeight * (1 - frac)
            const ivVal = minIv + frac * ivSpan
            return (
              <g key={`y-${idx}`}>
                <line x1={padLeft} y1={y} x2={width - padRight} y2={y} stroke="#1e293b" strokeDasharray="3 3" />
                <text x={padLeft - 8} y={y + 4} textAnchor="end" fontSize="10" fill="#64748b">
                  {ivVal.toFixed(1)}%
                </text>
              </g>
            )
          })}

          {/* Current Spot Price Reference Line */}
          {currentPriceX && (
            <g>
              <line
                x1={currentPriceX}
                y1={padTop}
                x2={currentPriceX}
                y2={padTop + plotHeight}
                stroke="#38bdf8"
                strokeWidth={1.5}
                strokeDasharray="4 4"
              />
              <text x={currentPriceX} y={padTop - 8} textAnchor="middle" fontSize="10" fill="#38bdf8" fontWeight="600">
                Spot: ${currentPrice?.toFixed(2)}
              </text>
            </g>
          )}

          {/* Curves */}
          {callPath && (
            <path d={callPath} fill="none" stroke="#10b981" strokeWidth={2.5} strokeLinecap="round" />
          )}
          {putPath && (
            <path d={putPath} fill="none" stroke="#f59e0b" strokeWidth={2.5} strokeLinecap="round" />
          )}

          {/* Dots on strikes */}
          {callPoints.map((pt, i) => (
            <circle key={`cp-${i}`} cx={pt.x} cy={pt.y} r={3} fill="#10b981" stroke="#064e3b" strokeWidth={1} />
          ))}
          {putPoints.map((pt, i) => (
            <circle key={`pp-${i}`} cx={pt.x} cy={pt.y} r={3} fill="#f59e0b" stroke="#78350f" strokeWidth={1} />
          ))}

          {/* X Axis Strike Labels */}
          {sortedStrikes.filter((_, idx) => idx % Math.ceil(sortedStrikes.length / 8) === 0).map((s, idx) => (
            <text key={`x-${idx}`} x={getX(s)} y={height - 12} textAnchor="middle" fontSize="10" fill="#64748b">
              ${s.toFixed(0)}
            </text>
          ))}

          {/* Hover Crosshair & Trigger Regions */}
          {hoveredStrike && (
            <line
              x1={hoveredStrike.x}
              y1={padTop}
              x2={hoveredStrike.x}
              y2={padTop + plotHeight}
              stroke="#94a3b8"
              strokeDasharray="2 2"
            />
          )}

          {sortedStrikes.map((s, idx) => {
            const x = getX(s)
            const colW = plotWidth / sortedStrikes.length
            const entry = strikeMap.get(s)
            return (
              <rect
                key={`trigger-${idx}`}
                x={x - colW / 2}
                y={padTop}
                width={colW}
                height={plotHeight}
                fill="transparent"
                onMouseEnter={() =>
                  setHoveredStrike({
                    strike: s,
                    callIv: entry?.callIv,
                    putIv: entry?.putIv,
                    x,
                  })
                }
                onMouseLeave={() => setHoveredStrike(null)}
              />
            )
          })}
        </svg>

        {hoveredStrike && (
          <div
            className="pointer-events-none absolute z-20 -translate-x-1/2 -translate-y-full transform rounded-lg border border-slate-700 bg-slate-900/95 p-3 text-xs shadow-xl backdrop-blur-md"
            style={{
              left: Math.min(width - 120, Math.max(padLeft + 60, hoveredStrike.x)),
              top: padTop + 35,
            }}
          >
            <div className="border-b border-slate-800 pb-1 font-semibold text-slate-100">
              Strike: ${hoveredStrike.strike.toFixed(2)}
            </div>
            <div className="mt-1.5 space-y-1 text-[11px]">
              <div className="flex items-center justify-between gap-4 text-emerald-400">
                <span>Call IV:</span>
                <span className="font-mono font-semibold">
                  {hoveredStrike.callIv !== undefined && hoveredStrike.callIv !== null
                    ? `${hoveredStrike.callIv.toFixed(1)}%`
                    : 'N/A'}
                </span>
              </div>
              <div className="flex items-center justify-between gap-4 text-amber-400">
                <span>Put IV:</span>
                <span className="font-mono font-semibold">
                  {hoveredStrike.putIv !== undefined && hoveredStrike.putIv !== null
                    ? `${hoveredStrike.putIv.toFixed(1)}%`
                    : 'N/A'}
                </span>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
