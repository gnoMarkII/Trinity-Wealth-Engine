import React from 'react'
import type { DCFScenarioDTO } from '../../api/types'

export interface ScenarioItem {
  key: 'bear' | 'base' | 'bull'
  label: string
  data: DCFScenarioDTO & { target_price: number; upside_pct: number }
  color: string
  textCol: string
  bgCol: string
}

export interface ScenarioTargetSpreadBarProps {
  scenarios: ScenarioItem[]
  selectedScenarioKey: 'bear' | 'base' | 'bull'
  onSelectScenario: (key: 'bear' | 'base' | 'bull') => void
  currentPrice?: number | null
  currency?: string
  className?: string
}

export const ScenarioTargetSpreadBar: React.FC<ScenarioTargetSpreadBarProps> = ({
  scenarios,
  selectedScenarioKey,
  onSelectScenario,
  currentPrice,
  currency = '$',
  className = '',
}) => {
  if (!scenarios || scenarios.length === 0) return null

  const minPrice = Math.min(...scenarios.map((s) => s.data.target_price), currentPrice ?? 100) * 0.95
  const maxPrice = Math.max(...scenarios.map((s) => s.data.target_price), currentPrice ?? 100) * 1.05
  const priceRange = Math.max(1, maxPrice - minPrice)

  return (
    <div
      data-testid="scenario-target-spread-bar"
      className={`rounded-xl border border-edge/80 bg-surface/90 p-4 space-y-4 ${className}`}
    >
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-edge/50 pb-2.5">
        <div className="flex items-center gap-2">
          <span className="text-base">🎯</span>
          <span className="text-xs font-bold text-zinc-800 tracking-tight">
            DCF Valuation Scenario Spread (Bear vs Base vs Bull)
          </span>
        </div>
        <span className="text-[11px] text-zinc-400">
          คลิกที่การ์ดเพื่อสลับสมมติฐานการเติบโตและอัตราคิดลด
        </span>
      </div>

      {/* 3 Scenario Cards with Tier-2 Numbers */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
        {scenarios.map((sc) => {
          const isSelected = selectedScenarioKey === sc.key
          const upside = sc.data.upside_pct
          return (
            <button
              key={sc.key}
              type="button"
              onClick={() => onSelectScenario(sc.key)}
              className={`p-3.5 rounded-xl border text-left transition-all cursor-pointer ${
                isSelected
                  ? `${sc.bgCol} ring-2 ring-offset-2 ring-offset-surface ring-current shadow-sm scale-[1.02]`
                  : 'bg-surface border-edge/60 hover:border-zinc-300 opacity-80 hover:opacity-100'
              }`}
            >
              <div className="flex items-center justify-between text-xs font-bold mb-1.5">
                <span className={sc.textCol}>{sc.label}</span>
                {isSelected && (
                  <span className="text-[9px] font-black uppercase px-1.5 py-0.2 rounded bg-black/10">
                    ACTIVE
                  </span>
                )}
              </div>

              <div className="flex items-baseline justify-between">
                <span className="font-mono font-black text-xl sm:text-2xl text-zinc-900 tracking-tight">
                  {currency}{sc.data.target_price.toFixed(2)}
                </span>
                <span
                  className={`text-xs font-bold px-2 py-0.5 rounded-full border ${
                    upside >= 0
                      ? 'bg-emerald-50 text-emerald-800 border-emerald-300'
                      : 'bg-rose-50 text-rose-800 border-rose-300'
                  }`}
                >
                  {upside >= 0 ? '+' : ''}{upside.toFixed(1)}%
                </span>
              </div>

              <div className="mt-2 pt-2 border-t border-edge/40 flex items-center justify-between text-[10px] text-zinc-500 font-mono">
                <span>MoS: {sc.data.margin_of_safety_pct != null ? `${sc.data.margin_of_safety_pct.toFixed(1)}%` : 'N/A'}</span>
                <span>Exit: {sc.data.implied_exit_ebitda_multiple != null ? `${sc.data.implied_exit_ebitda_multiple.toFixed(1)}x` : 'N/A'}</span>
              </div>
            </button>
          )
        })}
      </div>

      {/* Visual Relative Spread Bar */}
      <div className="space-y-1.5 pt-1">
        <div className="flex items-center justify-between text-[10px] font-mono text-zinc-400">
          <span>MIN {currency}{minPrice.toFixed(0)}</span>
          {currentPrice != null && (
            <span className="text-sky-700 font-bold">ANALYSIS CLOSE: {currency}{currentPrice.toFixed(2)}</span>
          )}
          <span>MAX {currency}{maxPrice.toFixed(0)}</span>
        </div>

        <div className="relative w-full h-3 rounded-full bg-zinc-200/80 overflow-hidden shadow-inner border border-edge/50">
          {scenarios.map((sc) => {
            const pos = Math.min(95, Math.max(5, ((sc.data.target_price - minPrice) / priceRange) * 100))
            const isSelected = selectedScenarioKey === sc.key
            return (
              <div
                key={sc.key}
                className={`absolute top-0 bottom-0 w-3 rounded-full border border-white -translate-x-1.5 transition-all duration-500 ${sc.color} ${
                  isSelected ? 'shadow-[0_0_10px_rgba(0,0,0,0.5)] scale-125 z-10' : 'opacity-80'
                }`}
                style={{ left: `${pos}%` }}
                title={`${sc.label}: ${currency}${sc.data.target_price.toFixed(2)}`}
              />
            )
          })}
        </div>
      </div>
    </div>
  )
}

export default ScenarioTargetSpreadBar
