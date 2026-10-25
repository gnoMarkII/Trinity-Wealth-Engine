import React from 'react'

export interface PriceActionDepthLadderProps {
  currentPrice: number
  keyResistance?: number | null
  keySupport?: number | null
  buyZoneMin?: number | null
  buyZoneMax?: number | null
  stopLoss?: number | null
  tacticalTarget?: number | null
  currency?: string
  className?: string
}

export const PriceActionDepthLadder: React.FC<PriceActionDepthLadderProps> = ({
  currentPrice,
  keyResistance,
  keySupport,
  buyZoneMin,
  buyZoneMax,
  stopLoss,
  tacticalTarget,
  currency = '$',
  className = '',
}) => {
  const resistance = keyResistance ?? tacticalTarget ?? currentPrice * 1.15
  const support = keySupport ?? stopLoss ?? currentPrice * 0.9

  // Calculate distance percentages from current price
  const distResistancePct = currentPrice > 0 ? ((resistance - currentPrice) / currentPrice) * 100 : 0
  const distSupportPct = currentPrice > 0 ? ((currentPrice - support) / currentPrice) * 100 : 0

  const isInBuyZone =
    buyZoneMin != null && buyZoneMax != null && currentPrice >= buyZoneMin && currentPrice <= buyZoneMax

  // Ladder position percentage for current price between support and resistance
  const range = Math.max(0.01, resistance - support)
  const currentPosPct = Math.min(95, Math.max(5, ((currentPrice - support) / range) * 100))
  const isBelowBuyZone = buyZoneMin != null && currentPrice < buyZoneMin
  const isAboveBuyZone = buyZoneMax != null && currentPrice > buyZoneMax

  let badgeText = '⚡ Evaluating Entry Zone'
  let badgeClass = 'bg-zinc-100 text-zinc-700 border-zinc-200'

  if (isInBuyZone) {
    badgeText = '🎯 Price in Optimal Entry Channel'
    badgeClass = 'bg-emerald-50 text-emerald-800 border-emerald-300'
  } else if (isBelowBuyZone) {
    badgeText = '⚠️ Below Buy Zone (Wait for Base)'
    badgeClass = 'bg-rose-50 text-rose-800 border-rose-300'
  } else if (isAboveBuyZone) {
    badgeText = '⏳ Above Buy Zone (Wait for Dip)'
    badgeClass = 'bg-zinc-100 text-zinc-700 border-zinc-200'
  }

  return (
    <div
      data-testid="price-action-depth-ladder"
      className={`rounded-xl border border-edge/80 bg-surface/90 p-4 space-y-4 ${className}`}
    >
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-edge/50 pb-2.5">
        <div className="flex items-center gap-2">
          <span className="text-base">🪜</span>
          <span className="text-xs font-bold text-zinc-800 tracking-tight">
            Support / Resistance Depth Ladder & Execution Channels
          </span>
        </div>
        <span className={`px-2 py-0.5 text-[10px] font-bold rounded-full border ${badgeClass}`}>
          {badgeText}
        </span>
      </div>

      {/* Vertical Depth Visualization */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-3 text-xs">
        {/* Level 1: Resistance Ceiling */}
        <div className="p-3 rounded-xl border border-rose-200 bg-rose-50/40 flex flex-col justify-between">
          <div className="flex items-center justify-between text-[10px] font-bold text-rose-800 uppercase">
            <span>🔴 Major Resistance (Ceiling)</span>
            <span>+{distResistancePct.toFixed(1)}%</span>
          </div>
          <div className="mt-2 font-mono font-black text-xl text-rose-950">
            {currency}{resistance.toFixed(2)}
          </div>
          <span className="text-[10px] text-zinc-500 mt-1">
            Weekly Pivot R1 / Target Resistance
          </span>
        </div>

        {/* Level 2: Current Analysis Price */}
        <div className="p-3 rounded-xl border-2 border-sky-400 bg-sky-50/60 shadow-sm flex flex-col justify-between">
          <div className="flex items-center justify-between text-[10px] font-bold text-sky-800 uppercase">
            <span>🔵 Current Analysis Close</span>
            <span className="animate-pulse">📍 PIN</span>
          </div>
          <div className="mt-2 font-mono font-black text-2xl text-sky-950">
            {currency}{currentPrice.toFixed(2)}
          </div>
          <span className="text-[10px] text-sky-700 font-semibold mt-1">
            Channel Position: {currentPosPct.toFixed(0)}% from floor
          </span>
        </div>

        {/* Level 3: Optimal Buy Zone Channel */}
        <div className="p-3 rounded-xl border border-emerald-200 bg-emerald-50/40 flex flex-col justify-between">
          <div className="flex items-center justify-between text-[10px] font-bold text-emerald-800 uppercase">
            <span>🟢 Accumulation Zone</span>
            {isInBuyZone && <span className="text-emerald-700 font-extrabold">IN ZONE</span>}
          </div>
          <div className="mt-2 font-mono font-bold text-lg text-emerald-950 truncate">
            {buyZoneMin != null && buyZoneMax != null
              ? `${currency}${buyZoneMin.toFixed(2)} - ${currency}${buyZoneMax.toFixed(2)}`
              : 'N/A'}
          </div>
          <span className="text-[10px] text-zinc-500 mt-1">
            Low-risk institutional pullback entry
          </span>
        </div>

        {/* Level 4: Stop Loss / Structural Support */}
        <div className="p-3 rounded-xl border border-zinc-200 bg-zinc-50/80 flex flex-col justify-between">
          <div className="flex items-center justify-between text-[10px] font-bold text-zinc-600 uppercase">
            <span>🛑 Key Support / Stop Loss</span>
            <span className="text-rose-600">-{distSupportPct.toFixed(1)}%</span>
          </div>
          <div className="mt-2 font-mono font-black text-xl text-zinc-900">
            {currency}{support.toFixed(2)}
          </div>
          <span className="text-[10px] text-zinc-500 mt-1">
            Invalidation Floor (Gap / Pivot S1)
          </span>
        </div>
      </div>

      {/* Channel Relative Depth Bar */}
      <div className="space-y-1.5 pt-1">
        <div className="flex items-center justify-between text-[10px] font-bold text-zinc-400">
          <span>FLOOR ({currency}{support.toFixed(2)})</span>
          <span className="text-sky-700 font-mono">CURRENT: {currency}{currentPrice.toFixed(2)}</span>
          <span>CEILING ({currency}{resistance.toFixed(2)})</span>
        </div>

        <div className="relative w-full h-3 rounded-full bg-gradient-to-r from-rose-200 via-emerald-100 to-rose-200 overflow-hidden shadow-inner border border-edge/60">
          {/* Optimal Buy Zone Range Highlight */}
          {buyZoneMin != null && buyZoneMax != null && (
            <div
              className="absolute top-0 bottom-0 bg-emerald-500/30 border-x border-emerald-500/50"
              style={{
                left: `${Math.max(0, ((buyZoneMin - support) / range) * 100)}%`,
                width: `${Math.min(100, Math.max(5, ((buyZoneMax - buyZoneMin) / range) * 100))}%`,
              }}
              title="Optimal Buy Zone Range"
            />
          )}

          {/* Current Price Marker Pin */}
          <div
            className="absolute top-0 bottom-0 w-3 bg-sky-600 border border-white rounded-full shadow-[0_0_8px_rgba(2,132,199,0.8)] -translate-x-1.5 transition-all duration-500"
            style={{ left: `${currentPosPct}%` }}
            title={`Current Price: ${currency}${currentPrice.toFixed(2)}`}
          />
        </div>
      </div>
    </div>
  )
}

export default PriceActionDepthLadder
