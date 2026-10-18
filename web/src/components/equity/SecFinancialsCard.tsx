import React from 'react'
import type { SecCompanyFactsSnapshotDTO } from '../../api/types'

interface SecFinancialsCardProps {
  financials?: SecCompanyFactsSnapshotDTO | null
  symbol: string
  className?: string
}

export const SecFinancialsCard: React.FC<SecFinancialsCardProps> = ({
  financials,
  symbol,
  className = '',
}) => {
  if (!financials) {
    return (
      <div className={`rounded-2xl border border-edge/60 bg-surface/70 p-6 text-center ${className}`}>
        <h4 className="text-sm font-semibold text-zinc-700">SEC Financial Facts ({symbol})</h4>
        <p className="mt-2 text-xs text-zinc-500">No company-filed XBRL facts available for {symbol}.</p>
      </div>
    )
  }

  const fcf = financials.free_cash_flow_usd
  const fcfMargin = financials.free_cash_flow_margin
  const debtOcf = financials.debt_to_ocf_ratio

  return (
    <div className={`flow-panel rounded-2xl border border-edge/80 p-6 shadow-sm transition-all animate-card-in ${className}`}>
      <div className="mb-4 flex flex-wrap items-baseline justify-between gap-2 border-b border-edge/60 pb-3">
        <div>
          <h3 className="text-base font-semibold text-zinc-900 flex items-center gap-2">
            <span>Company-Filed XBRL Facts</span>
            <span className="rounded-full bg-sky-100 text-sky-800 border border-sky-200 font-mono text-[10px] font-bold px-2 py-0.5">
              SEC EDGAR
            </span>
          </h3>
          <p className="text-xs text-zinc-500 mt-0.5">
            {financials.entity_name} ({financials.symbol} • CIK: {financials.cik})
          </p>
        </div>
        <div className="text-right text-[11px] text-zinc-400">
          As of: {financials.as_of_date}
        </div>
      </div>

      {/* Derived Pure Ratios */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-4">
        <div className="rounded-xl border border-edge/60 bg-surface/80 p-3">
          <div className="text-[11px] font-medium text-zinc-500">Revenue</div>
          <div className="mt-1 font-mono text-base font-bold text-zinc-900">
            {financials.revenue_usd !== null && financials.revenue_usd !== undefined
              ? `$${(financials.revenue_usd / 1e9).toFixed(2)}B`
              : 'N/A'}
          </div>
          <div className="text-[10px] text-zinc-400 mt-0.5">Form 10-K / 10-Q</div>
        </div>

        <div className="rounded-xl border border-edge/60 bg-surface/80 p-3">
          <div className="text-[11px] font-medium text-zinc-500">Free Cash Flow (FCF)</div>
          <div className="mt-1 font-mono text-base font-bold text-emerald-700">
            {fcf !== null && fcf !== undefined ? `$${(fcf / 1e9).toFixed(2)}B` : 'N/A'}
          </div>
          <div className="text-[10px] text-zinc-400 mt-0.5">OCF - CapEx</div>
        </div>

        <div className="rounded-xl border border-edge/60 bg-surface/80 p-3">
          <div className="text-[11px] font-medium text-zinc-500">FCF Margin</div>
          <div className="mt-1 font-mono text-base font-bold text-emerald-700">
            {fcfMargin !== null && fcfMargin !== undefined ? `${(fcfMargin * 100).toFixed(1)}%` : 'N/A'}
          </div>
          <div className="text-[10px] text-zinc-400 mt-0.5">FCF / Revenue</div>
        </div>

        <div className="rounded-xl border border-edge/60 bg-surface/80 p-3">
          <div className="text-[11px] font-medium text-zinc-500">Debt-to-OCF</div>
          <div className="mt-1 font-mono text-base font-bold text-sky-700">
            {debtOcf !== null && debtOcf !== undefined ? `${debtOcf.toFixed(2)}x` : 'N/A'}
          </div>
          <div className="text-[10px] text-zinc-400 mt-0.5">Long-term Debt / OCF</div>
        </div>
      </div>

      {/* Facts Table */}
      {financials.facts && financials.facts.length > 0 && (
        <div className="overflow-x-auto rounded-xl border border-edge/60 bg-surface/50">
          <table className="w-full text-left text-xs">
            <thead className="border-b border-edge/60 bg-surface-strong/80 text-[11px] font-semibold text-zinc-600">
              <tr>
                <th className="px-3.5 py-2.5">Metric / Concept</th>
                <th className="px-3.5 py-2.5">Form</th>
                <th className="px-3.5 py-2.5">Period</th>
                <th className="px-3.5 py-2.5">Filed Date</th>
                <th className="px-3.5 py-2.5 text-right">Value (USD)</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-edge/40 font-mono text-[11px] text-zinc-700">
              {financials.facts.map((f, idx) => (
                <tr key={`${f.concept_tag}-${idx}`} className="hover:bg-surface/80 transition-colors">
                  <td className="px-3.5 py-2 font-sans font-medium text-zinc-900">
                    <div>{f.label}</div>
                    <div className="text-[10px] text-zinc-400 font-mono">{f.concept_tag}</div>
                  </td>
                  <td className="px-3.5 py-2 text-zinc-600">{f.form}</td>
                  <td className="px-3.5 py-2 text-zinc-600">
                    {f.fy ? `FY${f.fy}` : ''} {f.fp || ''}
                  </td>
                  <td className="px-3.5 py-2 text-zinc-500">{f.filed || '-'}</td>
                  <td className="px-3.5 py-2 text-right font-bold text-zinc-900">
                    {f.val !== null && f.val !== undefined ? `$${(f.val / 1e9).toFixed(3)}B` : '-'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Footnote on Un-audited 10-Q & Stock vs Flow */}
      <div className="mt-3 rounded-xl border border-edge/50 bg-surface-strong/40 p-3 text-[11px] text-zinc-500 space-y-1">
        <div>
          ℹ️ <strong>Company-Filed Disclosures:</strong> Sourced from official SEC EDGAR XBRL filings. Quarterly figures (Form 10-Q) are un-audited management disclosures, not universally audited GAAP.
        </div>
        <div>
          Debt is a balance sheet stock as of period end; cash flow metrics are flows accumulated over the period.
        </div>
      </div>
    </div>
  )
}
