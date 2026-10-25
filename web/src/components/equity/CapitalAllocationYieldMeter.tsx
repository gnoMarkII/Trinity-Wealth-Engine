import React from 'react'

export interface CapitalAllocationYieldMeterProps {
  dividendYieldPct?: number | null
  buybackYieldPct?: number | null
  netShareholderYieldPct?: number | null
  fcfYieldPct?: number | null
  className?: string
}

export const CapitalAllocationYieldMeter: React.FC<CapitalAllocationYieldMeterProps> = ({
  dividendYieldPct = 0,
  buybackYieldPct,
  netShareholderYieldPct,
  fcfYieldPct,
  className = '',
}) => {
  const divYield = dividendYieldPct ?? 0
  const bbYield = buybackYieldPct ?? 0
  const netYield = netShareholderYieldPct ?? (divYield + bbYield)

  // Estimated dilution if netYield < divYield + bbYield
  const grossYield = divYield + bbYield
  const estimatedDilutionPct = grossYield > netYield ? grossYield - netYield : 0

  const isZeroDividend = divYield === 0 || divYield < 0.01

  return (
    <div
      data-testid="capital-allocation-yield-meter"
      className={`rounded-xl border border-edge/60 bg-surface/80 p-3.5 space-y-2.5 ${className}`}
    >
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-1.5">
          <span className="text-sm">💰</span>
          <span className="text-xs font-bold text-zinc-800 tracking-tight">
            Capital Allocation & Shareholder Yield
          </span>
        </div>
        <span
          className={`px-2 py-0.5 rounded-full text-[10px] font-bold border ${
            netYield > 0
              ? 'bg-emerald-50 text-emerald-800 border-emerald-300'
              : 'bg-zinc-100 text-zinc-700 border-zinc-200'
          }`}
        >
          {isZeroDividend ? 'Zero Dividend (Pure Reinvestment / Buybacks)' : 'Dividend Paying'}
        </span>
      </div>

      {/* Yield Components Grid with Tier-1 / Tier-2 Callouts */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-center text-xs">
        {/* Dividend Yield */}
        <div className="p-2 rounded-lg bg-surface border border-edge/40">
          <span className="text-[10px] text-zinc-400 uppercase tracking-wider block">
            Dividend Yield
          </span>
          <span className="font-bold text-zinc-800 text-sm">
            {divYield > 0 ? `${divYield.toFixed(2)}%` : '0.00%'}
          </span>
          {isZeroDividend && (
            <span className="text-[9px] text-zinc-400 block -mt-0.5">Reinvested in R&D</span>
          )}
        </div>

        {/* Share Buyback Yield */}
        <div className="p-2 rounded-lg bg-surface border border-edge/40">
          <span className="text-[10px] text-zinc-400 uppercase tracking-wider block">
            Buyback Yield
          </span>
          <span
            className={`font-bold text-sm ${
              bbYield > 0 ? 'text-emerald-700' : 'text-zinc-800'
            }`}
          >
            {bbYield != null ? `${bbYield >= 0 ? '+' : ''}${bbYield.toFixed(2)}%` : 'N/A'}
          </span>
          <span className="text-[9px] text-zinc-400 block -mt-0.5">Share Repurchases</span>
        </div>

        {/* SBC Dilution Offset (if applicable) */}
        <div className="p-2 rounded-lg bg-surface border border-edge/40">
          <span className="text-[10px] text-zinc-400 uppercase tracking-wider block">
            SBC Dilution Est.
          </span>
          <span
            className={`font-bold text-sm ${
              estimatedDilutionPct > 0 ? 'text-rose-600' : 'text-zinc-600'
            }`}
          >
            {estimatedDilutionPct > 0 ? `-${estimatedDilutionPct.toFixed(2)}%` : '0.00%'}
          </span>
          <span className="text-[9px] text-zinc-400 block -mt-0.5">Stock Compensation</span>
        </div>

        {/* Net Shareholder Yield */}
        <div className="p-2 rounded-lg bg-surface border border-edge/60 shadow-xs">
          <span className="text-[10px] font-bold text-zinc-500 uppercase tracking-wider block">
            Net Cash Yield
          </span>
          <span
            className={`text-base font-black tracking-tight ${
              netYield >= 0 ? 'text-emerald-700' : 'text-rose-600'
            }`}
          >
            {netYield >= 0 ? '+' : ''}
            {netYield.toFixed(2)}%
          </span>
          {fcfYieldPct != null && (
            <span className="text-[9px] text-zinc-400 block -mt-0.5" title="Free Cash Flow Yield">
              FCF Yield: {fcfYieldPct.toFixed(1)}%
            </span>
          )}
        </div>
      </div>

      {/* Visual Mini Allocation Bar */}
      <div className="space-y-1">
        <div className="w-full h-2 rounded-full bg-zinc-200/80 overflow-hidden flex shadow-inner">
          {divYield > 0 && (
            <div
              className="h-full bg-sky-500"
              style={{ width: `${Math.min(50, Math.max(10, divYield * 10))}%` }}
              title={`Dividend: ${divYield.toFixed(2)}%`}
            />
          )}
          {bbYield > 0 && (
            <div
              className="h-full bg-emerald-500"
              style={{ width: `${Math.min(70, Math.max(15, bbYield * 15))}%` }}
              title={`Buyback: ${bbYield.toFixed(2)}%`}
            />
          )}
          {estimatedDilutionPct > 0 && (
            <div
              className="h-full bg-rose-400"
              style={{ width: `${Math.min(40, Math.max(10, estimatedDilutionPct * 15))}%` }}
              title={`SBC Dilution: -${estimatedDilutionPct.toFixed(2)}%`}
            />
          )}
        </div>
        <div className="flex items-center justify-between text-[10px] text-zinc-500 pt-0.5">
          <span>Net Return to Equity Holders</span>
          <span className="font-semibold text-zinc-700">
            {netYield >= 0 ? 'Accretive Capital Return' : 'Net Dilutive Distribution'}
          </span>
        </div>
      </div>
    </div>
  )
}

export default CapitalAllocationYieldMeter
