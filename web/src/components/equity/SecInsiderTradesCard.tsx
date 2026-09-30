import React from 'react'
import type { SecInsiderTradeSnapshotDTO } from '../../api/types'

interface SecInsiderTradesCardProps {
  trades?: SecInsiderTradeSnapshotDTO | null
  symbol: string
  className?: string
}

export const SecInsiderTradesCard: React.FC<SecInsiderTradesCardProps> = ({
  trades,
  symbol,
  className = '',
}) => {
  if (!trades) {
    return (
      <div className={`rounded-xl border border-slate-800 bg-slate-900/80 p-5 text-center ${className}`}>
        <h4 className="text-sm font-semibold text-slate-300">SEC Form 4 Insider Trades ({symbol})</h4>
        <p className="mt-3 text-xs text-slate-500">No Form 4 insider transactions recorded for {symbol}.</p>
      </div>
    )
  }

  const ratio = trades.net_buy_ratio_90d
  const ratioColor =
    ratio === null || ratio === undefined
      ? 'text-slate-400'
      : ratio > 0.1
      ? 'text-emerald-400'
      : ratio < -0.1
      ? 'text-rose-400'
      : 'text-amber-400'

  return (
    <div className={`rounded-xl border border-slate-800 bg-slate-900/90 p-5 shadow-lg backdrop-blur-sm ${className}`}>
      <div className="mb-4 flex flex-wrap items-baseline justify-between gap-2 border-b border-slate-800/80 pb-3">
        <div>
          <h3 className="text-base font-semibold text-slate-100 flex items-center gap-2">
            <span>Form 4 Insider Trading Activity</span>
            <span className="rounded-full bg-slate-800 text-slate-300 font-mono text-[10px] font-bold px-2 py-0.5">
              SEC XML PARSED
            </span>
          </h3>
          <p className="text-xs text-slate-400">
            Open-market officer, director, and 10% owner transactions (excludes Form 13F)
          </p>
        </div>
        <div className="text-right text-[11px] text-slate-500">
          CIK: {trades.cik}
        </div>
      </div>

      {/* 90-Day Net Buying Metric */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 mb-4">
        <div className="rounded-lg border border-slate-800 bg-slate-950/70 p-3">
          <div className="text-[11px] text-slate-400">90-Day Net Buy Ratio [-1.0 to +1.0]</div>
          <div className={`mt-1 font-mono text-xl font-bold ${ratioColor}`}>
            {ratio !== null && ratio !== undefined ? `${ratio >= 0 ? '+' : ''}${ratio.toFixed(2)}` : 'N/A (No P/S)'}
          </div>
          <div className="text-[10px] text-slate-500">{trades.eligible_transaction_count} eligible trades</div>
        </div>

        <div className="rounded-lg border border-slate-800 bg-slate-950/70 p-3">
          <div className="text-[11px] text-slate-400">90D Open-Market Purchases (P)</div>
          <div className="mt-1 font-mono text-xl font-bold text-emerald-400">
            ${trades.p_notional_sum_90d.toLocaleString(undefined, { maximumFractionDigits: 0 })}
          </div>
          <div className="text-[10px] text-slate-500">Code P total notional</div>
        </div>

        <div className="rounded-lg border border-slate-800 bg-slate-950/70 p-3">
          <div className="text-[11px] text-slate-400">90D Open-Market Sales (S)</div>
          <div className="mt-1 font-mono text-xl font-bold text-rose-400">
            ${trades.s_notional_sum_90d.toLocaleString(undefined, { maximumFractionDigits: 0 })}
          </div>
          <div className="text-[10px] text-slate-500">Code S total notional</div>
        </div>
      </div>

      {/* Transaction Table */}
      {trades.transactions && trades.transactions.length > 0 ? (
        <div className="overflow-x-auto rounded-lg border border-slate-800 bg-slate-950/50">
          <table className="w-full text-left text-xs">
            <thead className="border-b border-slate-800 bg-slate-900/80 text-[11px] font-semibold text-slate-400">
              <tr>
                <th className="px-3 py-2">Date</th>
                <th className="px-3 py-2">Reporting Owner</th>
                <th className="px-3 py-2">Title</th>
                <th className="px-3 py-2 text-center">Type</th>
                <th className="px-3 py-2 text-right">Shares</th>
                <th className="px-3 py-2 text-right">Price</th>
                <th className="px-3 py-2 text-right">Notional</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60 font-mono text-[11px] text-slate-300">
              {trades.transactions.map((tx, idx) => {
                const isBuy = tx.transaction_code === 'P'
                const isSale = tx.transaction_code === 'S'
                return (
                  <tr key={`${tx.accession_number}-${idx}`} className="hover:bg-slate-900/40">
                    <td className="px-3 py-2 text-slate-400">{tx.transaction_date}</td>
                    <td className="px-3 py-2 font-sans font-medium text-slate-200">
                      <div>{tx.reporting_owner}</div>
                      {tx.is_amendment && (
                        <span className="text-[10px] text-amber-400">Form 4/A (Amendment)</span>
                      )}
                    </td>
                    <td className="px-3 py-2 font-sans text-slate-400">
                      {tx.officer_title || (tx.is_director ? 'Director' : tx.is_ten_percent_owner ? '10% Owner' : 'Insider')}
                    </td>
                    <td className="px-3 py-2 text-center">
                      <span className={`inline-block rounded px-2 py-0.5 text-[10px] font-bold ${
                        isBuy ? 'bg-emerald-950 text-emerald-400 border border-emerald-800' : isSale ? 'bg-rose-950 text-rose-400 border border-rose-800' : 'bg-slate-800 text-slate-400'
                      }`}>
                        {tx.transaction_code} ({isBuy ? 'Buy' : isSale ? 'Sale' : 'Other'})
                      </span>
                    </td>
                    <td className="px-3 py-2 text-right font-medium text-slate-200">
                      {tx.shares ? tx.shares.toLocaleString() : '-'}
                    </td>
                    <td className="px-3 py-2 text-right text-slate-300">
                      {tx.price_per_share ? `$${tx.price_per_share.toFixed(2)}` : '-'}
                    </td>
                    <td className="px-3 py-2 text-right font-bold text-slate-100">
                      {tx.notional_usd ? `$${tx.notional_usd.toLocaleString(undefined, { maximumFractionDigits: 0 })}` : '-'}
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      ) : (
        <div className="py-4 text-center text-xs text-slate-500">
          No individual Form 4 transactions recorded in the recent window.
        </div>
      )}

      {/* Strict Invariant Callout */}
      <div className="mt-3 rounded-lg border border-slate-800/80 bg-slate-950/40 p-2.5 text-[11px] text-slate-400">
        ℹ️ <strong>Form 4 XML Contract:</strong> Sourced directly from parsed Form 4 XML ownership documents. Form 13F institutional whale holdings are strictly segregated and excluded. 90-Day Net Buying includes only eligible common-share open-market purchases (P) and sales (S).
      </div>
    </div>
  )
}
