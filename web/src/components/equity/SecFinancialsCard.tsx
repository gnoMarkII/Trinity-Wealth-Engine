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
      <div className={`rounded-xl border border-slate-800 bg-slate-900/80 p-5 text-center ${className}`}>
        <h4 className="text-sm font-semibold text-slate-300">SEC Financial Facts ({symbol})</h4>
        <p className="mt-3 text-xs text-slate-500">No company-filed XBRL facts available for {symbol}.</p>
      </div>
    )
  }

  const fcf = financials.free_cash_flow_usd
  const fcfMargin = financials.free_cash_flow_margin
  const debtOcf = financials.debt_to_ocf_ratio

  return (
    <div className={`rounded-xl border border-slate-800 bg-slate-900/90 p-5 shadow-lg backdrop-blur-sm ${className}`}>
      <div className="mb-4 flex flex-wrap items-baseline justify-between gap-2 border-b border-slate-800/80 pb-3">
        <div>
          <h3 className="text-base font-semibold text-slate-100 flex items-center gap-2">
            <span>Company-Filed XBRL Facts</span>
            <span className="rounded-full bg-slate-800 text-slate-300 font-mono text-[10px] font-bold px-2 py-0.5">
              SEC EDGAR
            </span>
          </h3>
          <p className="text-xs text-slate-400">
            {financials.entity_name} ({financials.symbol} • CIK: {financials.cik})
          </p>
        </div>
        <div className="text-right text-[11px] text-slate-500">
          As of: {financials.as_of_date}
        </div>
      </div>

      {/* Derived Pure Ratios */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-4">
        <div className="rounded-lg border border-slate-800 bg-slate-950/70 p-3">
          <div className="text-[11px] text-slate-400">Revenue</div>
          <div className="mt-1 font-mono text-base font-bold text-slate-100">
            {financials.revenue_usd !== null && financials.revenue_usd !== undefined
              ? `$${(financials.revenue_usd / 1e9).toFixed(2)}B`
              : 'N/A'}
          </div>
          <div className="text-[10px] text-slate-500">Form 10-K / 10-Q</div>
        </div>

        <div className="rounded-lg border border-slate-800 bg-slate-950/70 p-3">
          <div className="text-[11px] text-slate-400">Free Cash Flow (FCF)</div>
          <div className="mt-1 font-mono text-base font-bold text-emerald-400">
            {fcf !== null && fcf !== undefined ? `$${(fcf / 1e9).toFixed(2)}B` : 'N/A'}
          </div>
          <div className="text-[10px] text-slate-500">OCF - CapEx</div>
        </div>

        <div className="rounded-lg border border-slate-800 bg-slate-950/70 p-3">
          <div className="text-[11px] text-slate-400">FCF Margin</div>
          <div className="mt-1 font-mono text-base font-bold text-emerald-400">
            {fcfMargin !== null && fcfMargin !== undefined ? `${(fcfMargin * 100).toFixed(1)}%` : 'N/A'}
          </div>
          <div className="text-[10px] text-slate-500">FCF / Revenue</div>
        </div>

        <div className="rounded-lg border border-slate-800 bg-slate-950/70 p-3">
          <div className="text-[11px] text-slate-400">Debt-to-OCF</div>
          <div className="mt-1 font-mono text-base font-bold text-sky-400">
            {debtOcf !== null && debtOcf !== undefined ? `${debtOcf.toFixed(2)}x` : 'N/A'}
          </div>
          <div className="text-[10px] text-slate-500">Long-term Debt / OCF</div>
        </div>
      </div>

      {/* Facts Table */}
      {financials.facts && financials.facts.length > 0 && (
        <div className="overflow-x-auto rounded-lg border border-slate-800 bg-slate-950/50">
          <table className="w-full text-left text-xs">
            <thead className="border-b border-slate-800 bg-slate-900/80 text-[11px] font-semibold text-slate-400">
              <tr>
                <th className="px-3 py-2">Metric / Concept</th>
                <th className="px-3 py-2">Form</th>
                <th className="px-3 py-2">Period</th>
                <th className="px-3 py-2">Filed Date</th>
                <th className="px-3 py-2 text-right">Value (USD)</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60 font-mono text-[11px] text-slate-300">
              {financials.facts.map((f, idx) => (
                <tr key={`${f.concept_tag}-${idx}`} className="hover:bg-slate-900/40">
                  <td className="px-3 py-2 font-sans font-medium text-slate-200">
                    <div>{f.label}</div>
                    <div className="text-[10px] text-slate-500">{f.concept_tag}</div>
                  </td>
                  <td className="px-3 py-2 text-slate-400">{f.form}</td>
                  <td className="px-3 py-2 text-slate-400">
                    {f.fy ? `FY${f.fy}` : ''} {f.fp || ''}
                  </td>
                  <td className="px-3 py-2 text-slate-400">{f.filed || '-'}</td>
                  <td className="px-3 py-2 text-right font-bold text-slate-100">
                    {f.val !== null && f.val !== undefined ? `$${(f.val / 1e9).toFixed(3)}B` : '-'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Footnote on Un-audited 10-Q & Stock vs Flow */}
      <div className="mt-3 rounded-lg border border-slate-800/80 bg-slate-950/40 p-2.5 text-[11px] text-slate-400 space-y-1">
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
