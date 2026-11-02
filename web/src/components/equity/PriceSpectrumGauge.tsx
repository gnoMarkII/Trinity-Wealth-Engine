import React from 'react'

export interface PriceSpectrumGaugeProps {
  currentPrice: number
  targetPrice12m?: number | null
  upside12mPct?: number | null
  intrinsicValue?: number | null
  buyZoneMin?: number | null
  buyZoneMax?: number | null
  isInBuyZone?: boolean | null
  stopLoss?: number | null
  currency?: string
  actionStance?: string
  stanceReason?: string | null
}

export const PriceSpectrumGauge: React.FC<PriceSpectrumGaugeProps> = ({
  currentPrice,
  targetPrice12m,
  upside12mPct,
  intrinsicValue,
  buyZoneMin,
  buyZoneMax,
  isInBuyZone,
  stopLoss,
  currency = '$',
  actionStance = 'HOLD_WAIT',
  stanceReason,
}) => {
  // Guard against missing essential numbers
  if (!currentPrice || isNaN(currentPrice)) return null

  // Collect all valid values to compute scale boundaries
  const values: number[] = [currentPrice]
  if (targetPrice12m != null && !isNaN(targetPrice12m)) values.push(targetPrice12m)
  if (intrinsicValue != null && !isNaN(intrinsicValue)) values.push(intrinsicValue)
  if (buyZoneMin != null && !isNaN(buyZoneMin)) values.push(buyZoneMin)
  if (buyZoneMax != null && !isNaN(buyZoneMax)) values.push(buyZoneMax)
  if (stopLoss != null && !isNaN(stopLoss)) values.push(stopLoss)

  const rawMin = Math.min(...values)
  const rawMax = Math.max(...values)
  const span = Math.max(rawMax - rawMin, 1)

  // Add 4% padding on each side for aesthetic breathing room
  const minBound = Math.max(0, rawMin - span * 0.04)
  const maxBound = rawMax + span * 0.04
  const totalRange = maxBound - minBound

  const getPercent = (val: number) => {
    return Math.min(100, Math.max(0, ((val - minBound) / totalRange) * 100))
  }

  const currentPct = getPercent(currentPrice)
  const stopLossPct = stopLoss != null ? getPercent(stopLoss) : null
  const buyMinPct = buyZoneMin != null ? getPercent(buyZoneMin) : null
  const buyMaxPct = buyZoneMax != null ? getPercent(buyZoneMax) : null
  const intrinsicPct = intrinsicValue != null ? getPercent(intrinsicValue) : null
  const targetPct = targetPrice12m != null ? getPercent(targetPrice12m) : null

  // Zone status text
  let zoneStatusText = ''
  let zoneStatusColor = 'text-zinc-600 bg-zinc-100 border-zinc-200'
  if (isInBuyZone) {
    zoneStatusText = '🎯 อยู่ในโซนเข้าซื้อที่ปลอดภัย (In Optimal Buy Zone)'
    zoneStatusColor = 'text-emerald-700 bg-emerald-50 border-emerald-300'
  } else if (buyZoneMax != null && currentPrice > buyZoneMax) {
    const diff = currentPrice - buyZoneMax
    const diffPct = (diff / buyZoneMax) * 100
    zoneStatusText = `⏳ ราคาสูงกว่าโซนซื้อ +${currency}${diff.toFixed(2)} (+${diffPct.toFixed(1)}%) — รอจังหวะย่อตัว (Wait for Dip)`
    zoneStatusColor = 'text-sky-700 bg-sky-50 border-sky-300'
  } else if (buyZoneMin != null && currentPrice < buyZoneMin) {
    zoneStatusText = '⚠️ ราคาหลุดต่ำกว่าโซนซื้อ — ตรวจสอบแนวรับและวอลุ่มขาย'
    zoneStatusColor = 'text-amber-700 bg-amber-50 border-amber-300'
  }

  return (
    <div className="rounded-2xl border border-edge/90 bg-surface/80 p-5 shadow-xs space-y-5 animate-card-in">
      {/* Big Number Tier 1 Header */}
      <div className="flex flex-wrap items-end justify-between gap-4 border-b border-edge/60 pb-4">
        {/* Left: Current Price & Target Price with Big Number Callout */}
        <div className="flex flex-wrap items-baseline gap-6 sm:gap-8">
          <div>
            <span className="text-[10px] font-bold uppercase tracking-widest text-zinc-400 block mb-0.5">
              <span>Current Close Price</span>
              <span className="font-normal text-zinc-400 ml-1.5 lowercase tracking-normal">(ราคาปิดล่าสุด)</span>
            </span>
            <div className="flex items-baseline gap-1.5">
              <span className="font-mono text-3xl sm:text-4xl font-extrabold tracking-tight text-zinc-900">
                {currency}{currentPrice.toFixed(2)}
              </span>
            </div>
          </div>

          {targetPrice12m != null && (
            <div className="pr-2 border-l border-edge/60 pl-6 sm:pl-8">
              <span className="text-[10px] font-bold uppercase tracking-widest text-zinc-400 block mb-0.5">
                <span>12M Target Price (Reverse DCF)</span>
                <span className="font-normal text-zinc-400 ml-1.5 lowercase tracking-normal">(เป้าหมาย 12 เดือน)</span>
              </span>
              <div className="flex items-baseline gap-2">
                <span className="font-mono text-3xl sm:text-4xl font-extrabold tracking-tight text-emerald-600">
                  {currency}{targetPrice12m.toFixed(2)}
                </span>
                {upside12mPct != null && (
                  <span
                    className={`text-xs px-2 py-0.5 rounded-full font-bold border ${
                      upside12mPct >= 0
                        ? 'bg-emerald-50 text-emerald-700 border-emerald-300'
                        : 'bg-rose-50 text-rose-700 border-rose-300'
                    }`}
                  >
                    {upside12mPct >= 0 ? '+' : ''}{upside12mPct.toFixed(1)}% Upside
                  </span>
                )}
              </div>
            </div>
          )}
        </div>

        {/* Right: Action Stance Badge & Dynamic Guidance */}
        <div className="flex flex-col items-start sm:items-end gap-1.5">
          <div className="flex items-center gap-2">
            <span className="text-[10px] font-bold uppercase tracking-widest text-zinc-400">
              Verdict Stance:
            </span>
            <span className="px-3 py-1 rounded-full text-xs font-extrabold tracking-wider uppercase border bg-sky-50 text-sky-800 border-sky-300 shadow-2xs flex items-center gap-1.5">
              <span className="w-2 h-2 rounded-full bg-sky-500 animate-pulse" />
              {actionStance.replace(/_/g, ' ')}
            </span>
          </div>
          {stanceReason && (
            <p className="text-xs text-zinc-500 max-w-sm text-left sm:text-right italic">
              {stanceReason}
            </p>
          )}
        </div>
      </div>

      {/* Infographic 1: Linear Price Spectrum Bar */}
      <div className="space-y-3 pt-1">
        <div className="flex items-center justify-between text-xs font-semibold text-zinc-500">
          <span className="flex items-center gap-1.5">
            <span>🎯</span>
            <span className="tracking-wide uppercase text-[11px] text-zinc-600 font-bold">
              Valuation & Execution Spectrum
            </span>
          </span>
          {zoneStatusText && (
            <span className={`px-2.5 py-0.5 rounded-full text-xs font-semibold border ${zoneStatusColor}`}>
              {zoneStatusText}
            </span>
          )}
        </div>

        {/* The Track Container */}
        <div className="relative pt-7 pb-8 px-2">
          {/* Background Track */}
          <div className="h-3 w-full rounded-full bg-zinc-200/80 relative overflow-hidden">
            {/* Optimal Buy Zone Highlight on Track */}
            {buyMinPct != null && buyMaxPct != null && (
              <div
                className="absolute top-0 bottom-0 bg-emerald-400/50 border-x border-emerald-500/60"
                style={{
                  left: `${buyMinPct}%`,
                  width: `${Math.max(1, buyMaxPct - buyMinPct)}%`,
                }}
                title={`Optimal Buy Zone: ${currency}${buyZoneMin?.toFixed(2)} - ${currency}${buyZoneMax?.toFixed(2)}`}
              />
            )}
            {/* Intrinsic to Target Upside Shading */}
            {intrinsicPct != null && targetPct != null && (
              <div
                className="absolute top-0 bottom-0 bg-sky-300/30"
                style={{
                  left: `${Math.min(intrinsicPct, targetPct)}%`,
                  width: `${Math.abs(targetPct - intrinsicPct)}%`,
                }}
              />
            )}
          </div>

          {/* Marker 1: Invalidation Stop Loss (Rose) */}
          {stopLoss != null && stopLossPct != null && (
            <div
              className="absolute top-2 -translate-x-1/2 flex flex-col items-center group cursor-pointer"
              style={{ left: `${stopLossPct}%` }}
              title={`Invalidation Stop Loss: ${currency}${stopLoss.toFixed(2)} (จุดตัดขาดทุนเมื่อสมมติฐานผิด)`}
            >
              <span className="text-[10px] font-bold font-mono text-rose-600 bg-rose-50 px-1.5 py-0.2 rounded border border-rose-200 whitespace-nowrap shadow-2xs">
                Stop {currency}{stopLoss.toFixed(2)} <span className="font-sans font-normal text-[9px] text-rose-500">(ตัดขาดทุน)</span>
              </span>
              <div className="w-0.5 h-4 bg-rose-500 mt-0.5" />
            </div>
          )}

          {/* Marker 2: Optimal Buy Zone Range Labels (Below track) */}
          {buyMinPct != null && buyMaxPct != null && (
            <div
              className="absolute top-10 flex justify-between text-[10px] font-bold font-mono text-emerald-800"
              style={{
                left: `${buyMinPct}%`,
                width: `${Math.max(buyMaxPct - buyMinPct, 1)}%`,
              }}
            >
              <div className="flex flex-col items-start -translate-x-1/2">
                <div className="w-0.5 h-2 bg-emerald-600" />
                <span className="bg-emerald-50 px-1 py-0.2 rounded border border-emerald-200 shadow-2xs whitespace-nowrap mt-0.5">
                  Buy {currency}{buyZoneMin?.toFixed(2)} <span className="font-sans font-normal text-[9px] text-emerald-600">(เริ่มซื้อ)</span>
                </span>
              </div>
              <div className="flex flex-col items-end translate-x-1/2">
                <div className="w-0.5 h-2 bg-emerald-600" />
                <span className="bg-emerald-50 px-1 py-0.2 rounded border border-emerald-200 shadow-2xs whitespace-nowrap mt-0.5">
                  Max {currency}{buyZoneMax?.toFixed(2)} <span className="font-sans font-normal text-[9px] text-emerald-600">(เพดานซื้อ)</span>
                </span>
              </div>
            </div>
          )}

          {/* Marker 3: Intrinsic Value (Sky) */}
          {intrinsicValue != null && intrinsicPct != null && (
            <div
              className="absolute top-2 -translate-x-1/2 flex flex-col items-center group cursor-pointer"
              style={{ left: `${intrinsicPct}%` }}
              title={`Intrinsic Value Today: ${currency}${intrinsicValue.toFixed(2)} (มูลค่าพื้นฐานวันนี้)`}
            >
              <span className="text-[10px] font-bold font-mono text-sky-700 bg-sky-50 px-1.5 py-0.2 rounded border border-sky-200 whitespace-nowrap shadow-2xs">
                Intrinsic {currency}{intrinsicValue.toFixed(2)} <span className="font-sans font-normal text-[9px] text-sky-600">(มูลค่าจริง)</span>
              </span>
              <div className="w-0.5 h-4 bg-sky-400 mt-0.5" />
            </div>
          )}

          {/* Marker 4: 12M Target Price (Emerald) */}
          {targetPrice12m != null && targetPct != null && (
            <div
              className="absolute top-2 -translate-x-1/2 flex flex-col items-center group cursor-pointer z-10"
              style={{ left: `${targetPct}%` }}
              title={`12M Target Price: ${currency}${targetPrice12m.toFixed(2)} (ราคาเป้าหมาย 12 เดือน)`}
            >
              <span className="text-[10px] font-bold font-mono text-emerald-800 bg-emerald-100 px-2 py-0.5 rounded border border-emerald-300 whitespace-nowrap shadow-xs">
                Target {currency}{targetPrice12m.toFixed(2)} <span className="font-sans font-normal text-[9px] text-emerald-700">(เป้าหมาย)</span>
              </span>
              <div className="w-0.5 h-4 bg-emerald-600 mt-0.5" />
            </div>
          )}

          {/* CURRENT PRICE PIN (Prominent Glowing Indicator) */}
          <div
            className="absolute top-0 -translate-x-1/2 flex flex-col items-center z-20"
            style={{ left: `${currentPct}%` }}
          >
            <div className="px-2 py-0.5 rounded-full bg-zinc-900 text-white font-mono text-[11px] font-bold shadow-md border border-white/20 whitespace-nowrap flex items-center gap-1">
              <span className="w-1.5 h-1.5 rounded-full bg-sky-400 animate-ping" />
              <span>Now: {currency}{currentPrice.toFixed(2)} <span className="font-sans font-normal text-[9px] text-zinc-300">(ราคาปัจจุบัน)</span></span>
            </div>
            <div className="w-3 h-3 bg-zinc-900 border-2 border-white rounded-full shadow-md -mt-0.5" />
          </div>
        </div>
      </div>
    </div>
  )
}

export default PriceSpectrumGauge
