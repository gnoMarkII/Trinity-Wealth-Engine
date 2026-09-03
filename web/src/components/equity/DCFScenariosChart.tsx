import React from 'react'
import type { DCFResultDTO, DCFScenarioDTO } from '../../api/types'

interface DCFScenariosChartProps {
  dcf: DCFResultDTO
}

export const DCFScenariosChart: React.FC<DCFScenariosChartProps> = ({ dcf }) => {
  const scenarios = dcf.scenarios
    ? [
        { key: 'bear', label: 'Bear Case', barColor: 'bg-rose-500', data: dcf.scenarios.bear },
        { key: 'base', label: 'Base Case', barColor: 'bg-amber-500', data: dcf.scenarios.base },
        { key: 'bull', label: 'Bull Case', barColor: 'bg-emerald-500', data: dcf.scenarios.bull },
      ].filter((s): s is { key: string; label: string; barColor: string; data: DCFScenarioDTO & { target_price: number; upside_pct: number } } => s.data != null && s.data.target_price != null && s.data.upside_pct != null)
    : []

  if (scenarios.length === 0) {
    return (
      <div className="my-3 rounded-lg border border-edge/60 bg-surface/50 p-4 text-xs text-zinc-500">
        Targets unavailable or suppressed due to non-actionable valuation model.
      </div>
    )
  }

  const maxPrice = Math.max(...scenarios.map(s => s.data.target_price), 1)

  return (
    <div className="space-y-4 my-3">
      <div className="flex items-center justify-between text-xs text-zinc-500">
        <span>Scenarios Target Price</span>
        <span>
          WACC: <strong className="text-zinc-900 font-semibold">{dcf.wacc_pct ?? 'N/A'}%</strong> (Ke: {dcf.cost_of_equity_pct ?? 'N/A'}%, Kd: {dcf.cost_of_debt_pct ?? 'N/A'}%)
        </span>
      </div>

      <div className="space-y-3.5">
        {scenarios.map(sc => {
          const pct = Math.min(100, Math.max(10, (sc.data.target_price / maxPrice) * 100))
          return (
            <div key={sc.key} className="space-y-1.5">
              <div className="flex items-center justify-between text-sm">
                <span className="font-medium text-zinc-700">{sc.label}</span>
                <div className="flex items-center gap-2">
                  <span className="font-bold text-zinc-900">${sc.data.target_price}</span>
                  <span
                    className={`text-xs px-2 py-0.5 rounded-full font-semibold border ${
                      sc.data.upside_pct >= 0
                        ? 'bg-emerald-50 text-emerald-700 border-emerald-200'
                        : 'bg-rose-50 text-rose-700 border-rose-200'
                    }`}
                  >
                    {sc.data.upside_pct >= 0 ? '+' : ''}{sc.data.upside_pct}%
                  </span>
                </div>
              </div>
              <div className="w-full bg-zinc-100 rounded-full h-2.5 overflow-hidden border border-zinc-200/60">
                <div className={`h-2.5 rounded-full ${sc.barColor} transition-all duration-500`} style={{ width: `${pct}%` }} />
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}
