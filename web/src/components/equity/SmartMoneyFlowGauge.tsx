import React from 'react'
import type { SmartMoneyFlagsDTO } from '../../api/types'
import type { InsiderConvictionProps } from './TacticalFlowMatrixCard'

export interface SmartMoneyFlowGaugeProps {
  smartMoney?: SmartMoneyFlagsDTO | null
  insider?: InsiderConvictionProps | null
  currency?: string
  className?: string
}

function formatLargeNum(val?: number | null, prefix = '$'): string {
  if (val == null || isNaN(val)) return 'N/A'
  const abs = Math.abs(val)
  if (abs >= 1e9) return `${prefix}${(val / 1e9).toFixed(2)}B`
  if (abs >= 1e6) return `${prefix}${(val / 1e6).toFixed(2)}M`
  if (abs >= 1e3) return `${prefix}${(val / 1e3).toFixed(1)}K`
  return `${prefix}${val.toLocaleString()}`
}

export const SmartMoneyFlowGauge: React.FC<SmartMoneyFlowGaugeProps> = ({
  smartMoney,
  insider,
  currency = '$',
  className = '',
}) => {
  if (!smartMoney && !insider) return null

  const overallSignal = smartMoney?.overall_smart_money_flag || 'neutral'
  const instOwnership = smartMoney?.institutional_ownership_pct ?? 0
  const shortInterest = smartMoney?.short_interest_pct ?? 0
  const squeezeRisk = smartMoney?.short_squeeze_risk ?? false

  const isBullish = overallSignal === 'bullish_signal' || (insider?.open_market_p_count_90d ?? 0) > 0
  const isBearish = overallSignal === 'bearish_signal'

  return (
    <div
      data-testid="smart-money-flow-gauge"
      className={`rounded-xl border border-edge/80 bg-surface/90 p-4 space-y-4 ${className}`}
    >
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-edge/50 pb-2.5">
        <div className="flex items-center gap-2">
          <span className="text-base">🕵️</span>
          <span className="text-xs font-bold text-zinc-800 tracking-tight">
            Institutional Ownership & Smart Money Capital Flow (90D)
          </span>
        </div>
        <span
          className={`px-2.5 py-0.5 text-xs font-bold rounded-full uppercase border ${
            isBullish
              ? 'bg-emerald-50 text-emerald-800 border-emerald-300'
              : isBearish
              ? 'bg-rose-50 text-rose-800 border-rose-300'
              : 'bg-zinc-100 text-zinc-700 border-zinc-200'
          }`}
        >
          {isBullish
            ? '🟢 Smart Money Accumulation'
            : isBearish
            ? '🔴 Institutional Distribution'
            : '⚪ Neutral Institutional Flow'}
        </span>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
        {/* Metric 1: Institutional Ownership Gauge Bar */}
        <div className="p-3 rounded-xl bg-surface border border-edge/60 space-y-2">
          <div className="flex items-center justify-between text-[11px] font-medium text-zinc-500">
            <span>Institutional Float</span>
            <span className="font-bold text-zinc-900 font-mono">
              {instOwnership > 0 ? `${instOwnership.toFixed(1)}%` : 'N/A'}
            </span>
          </div>
          <div className="w-full h-2.5 rounded-full bg-zinc-200/80 overflow-hidden shadow-inner">
            <div
              className="h-full bg-gradient-to-r from-indigo-500 to-sky-500 rounded-full transition-all duration-500"
              style={{ width: `${Math.min(100, Math.max(5, instOwnership))}%` }}
              title={`Institutional Ownership: ${instOwnership}%`}
            />
          </div>
          <div className="flex items-center justify-between text-[10px] text-zinc-400">
            <span>Retail / Insider Float</span>
            <span className="font-semibold text-zinc-600">
              {instOwnership > 0 ? `${(100 - instOwnership).toFixed(1)}%` : 'N/A'}
            </span>
          </div>
        </div>

        {/* Metric 2: SEC Form 4 Open-Market Purchases */}
        <div className="p-3 rounded-xl bg-surface border border-edge/60 space-y-1">
          <div className="text-[11px] font-medium text-zinc-500">Form 4 Insider Purchases (P)</div>
          <div className="font-mono text-xl font-black text-emerald-700">
            {insider
              ? `${insider.open_market_p_count_90d || 0} รายการ (${formatLargeNum(
                  insider.open_market_p_value_usd,
                  currency
                )})`
              : `0 (${currency}0)`}
          </div>
          <span className="text-[10px] text-zinc-400 block">
            C-Suite: {insider?.c_suite_p_count || 0} รายการ (CEO/CFO Conviction)
          </span>
        </div>

        {/* Metric 3: Short Interest & Squeeze Risk Gauge */}
        <div className="p-3 rounded-xl bg-surface border border-edge/60 space-y-1">
          <div className="flex items-center justify-between text-[11px] font-medium text-zinc-500">
            <span>Short Interest (% Float)</span>
            {squeezeRisk && (
              <span className="px-1.5 py-0.2 rounded bg-amber-100 text-amber-800 font-bold text-[9px] border border-amber-300 animate-pulse">
                ⚡ Squeeze Risk
              </span>
            )}
          </div>
          <div className="font-mono text-xl font-black text-zinc-900">
            {shortInterest > 0 ? `${shortInterest.toFixed(2)}%` : 'Low / Minimal'}
          </div>
          <span className="text-[10px] text-zinc-400 block">
            {squeezeRisk ? 'Bear crowding detected (>15% shorted)' : 'Healthy borrowing demand'}
          </span>
        </div>
      </div>
    </div>
  )
}

export default SmartMoneyFlowGauge
