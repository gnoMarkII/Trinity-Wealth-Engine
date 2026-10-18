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
      <div className={`rounded-2xl border border-edge/60 bg-surface/70 p-6 text-center ${className}`}>
        <h4 className="text-sm font-semibold text-zinc-700">SEC Form 4 Insider Trades ({symbol})</h4>
        <p className="mt-2 text-xs text-zinc-500">No Form 4 insider transactions recorded for {symbol}.</p>
      </div>
    )
  }

  const ratio = trades.net_buy_ratio_90d
  const ratioColor =
    ratio === null || ratio === undefined
      ? 'text-zinc-400'
      : ratio > 0.1
      ? 'text-emerald-700'
      : ratio < -0.1
      ? 'text-rose-700'
      : 'text-amber-700'

  return (
    <div className={`flow-panel rounded-2xl border border-edge/80 p-6 shadow-sm transition-all animate-card-in ${className}`}>
      <div className="mb-4 flex flex-wrap items-baseline justify-between gap-2 border-b border-edge/60 pb-3">
        <div>
          <h3 className="text-base font-semibold text-zinc-900 flex items-center gap-2">
            <span>Form 4 Insider Trading Activity</span>
            <span className="rounded-full bg-sky-100 text-sky-800 border border-sky-200 font-mono text-[10px] font-bold px-2 py-0.5">
              SEC XML PARSED
            </span>
          </h3>
          <p className="text-xs text-zinc-500 mt-0.5">
            Open-market officer, director, and 10% owner transactions (excludes Form 13F)
          </p>
        </div>
        <div className="text-right text-[11px] text-zinc-400">
          CIK: {trades.cik}
        </div>
      </div>

      {/* 90-Day Net Buying Metric */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 mb-4">
        <div className="rounded-xl border border-edge/60 bg-surface/80 p-3">
          <div className="text-[11px] font-medium text-zinc-500">90-Day Net Buy Ratio [-1.0 to +1.0]</div>
          <div className={`mt-1 font-mono text-xl font-bold ${ratioColor}`}>
            {ratio !== null && ratio !== undefined ? `${ratio >= 0 ? '+' : ''}${ratio.toFixed(2)}` : 'N/A (No P/S)'}
          </div>
          <div className="text-[10px] text-zinc-400 mt-0.5">{trades.eligible_transaction_count} eligible trades</div>
        </div>

        <div className="rounded-xl border border-edge/60 bg-surface/80 p-3">
          <div className="text-[11px] font-medium text-zinc-500">90D Open-Market Purchases (P)</div>
          <div className="mt-1 font-mono text-xl font-bold text-emerald-700">
            ${trades.p_notional_sum_90d.toLocaleString(undefined, { maximumFractionDigits: 0 })}
          </div>
          <div className="text-[10px] text-zinc-400 mt-0.5">Code P total notional</div>
        </div>

        <div className="rounded-xl border border-edge/60 bg-surface/80 p-3">
          <div className="text-[11px] font-medium text-zinc-500">90D Open-Market Sales (S)</div>
          <div className="mt-1 font-mono text-xl font-bold text-rose-700">
            ${trades.s_notional_sum_90d.toLocaleString(undefined, { maximumFractionDigits: 0 })}
          </div>
          <div className="text-[10px] text-zinc-400 mt-0.5">Code S total notional</div>
        </div>
      </div>

      {/* Transaction Table */}
      {trades.transactions && trades.transactions.length > 0 ? (
        <div className="overflow-x-auto rounded-xl border border-edge/60 bg-surface/50">
          <table className="w-full text-left text-xs">
            <thead className="border-b border-edge/60 bg-surface-strong/80 text-[11px] font-semibold text-zinc-600">
              <tr>
                <th className="px-3.5 py-2.5">Date</th>
                <th className="px-3.5 py-2.5">Reporting Owner</th>
                <th className="px-3.5 py-2.5">Title</th>
                <th className="px-3.5 py-2.5 text-center">Type</th>
                <th className="px-3.5 py-2.5 text-right">Shares</th>
                <th className="px-3.5 py-2.5 text-right">Price</th>
                <th className="px-3.5 py-2.5 text-right">Notional</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-edge/40 font-mono text-[11px] text-zinc-700">
              {trades.transactions.map((tx, idx) => {
                const isBuy = tx.transaction_code === 'P'
                const isSale = tx.transaction_code === 'S'
                return (
                  <tr key={`${tx.accession_number}-${idx}`} className="hover:bg-surface/80 transition-colors">
                    <td className="px-3.5 py-2 text-zinc-500">{tx.transaction_date}</td>
                    <td className="px-3.5 py-2 font-sans font-medium text-zinc-900">
                      <div>{tx.reporting_owner}</div>
                      {tx.is_amendment && (
                        <span className="text-[10px] text-amber-700 font-semibold">Form 4/A (Amendment)</span>
                      )}
                    </td>
                    <td className="px-3.5 py-2 font-sans text-zinc-600">
                      {tx.officer_title || (tx.is_director ? 'Director' : tx.is_ten_percent_owner ? '10% Owner' : 'Insider')}
                    </td>
                    <td className="px-3.5 py-2 text-center">
                      <span className={`inline-block rounded px-2 py-0.5 text-[10px] font-bold border ${
                        isBuy ? 'bg-emerald-50 text-emerald-700 border-emerald-200' : isSale ? 'bg-rose-50 text-rose-700 border-rose-200' : 'bg-zinc-100 text-zinc-600 border-zinc-200'
                      }`}>
                        {tx.transaction_code} ({isBuy ? 'Buy' : isSale ? 'Sale' : 'Other'})
                      </span>
                    </td>
                    <td className="px-3.5 py-2 text-right font-medium text-zinc-800">
                      {tx.shares ? tx.shares.toLocaleString() : '-'}
                    </td>
                    <td className="px-3.5 py-2 text-right text-zinc-700">
                      {tx.price_per_share ? `$${tx.price_per_share.toFixed(2)}` : '-'}
                    </td>
                    <td className="px-3.5 py-2 text-right font-bold text-zinc-900">
                      {tx.notional_usd ? `$${tx.notional_usd.toLocaleString(undefined, { maximumFractionDigits: 0 })}` : '-'}
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      ) : (
        <div className="py-5 text-center text-xs text-zinc-400">
          No individual Form 4 transactions recorded in the recent window.
        </div>
      )}

      {/* Strict Invariant Callout */}
      <div className="mt-3 rounded-xl border border-edge/50 bg-surface-strong/40 p-3 text-[11px] text-zinc-500">
        ℹ️ <strong>Form 4 XML Contract:</strong> Sourced directly from parsed Form 4 XML ownership documents. Form 13F institutional whale holdings are strictly segregated and excluded. 90-Day Net Buying includes only eligible common-share open-market purchases (P) and sales (S).
      </div>
    </div>
  )
}
