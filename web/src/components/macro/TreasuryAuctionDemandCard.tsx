import React, { useState } from 'react'
import type { AuctionDemandSnapshotDTO } from '../../api/types'

interface TreasuryAuctionDemandCardProps {
  demandNote?: AuctionDemandSnapshotDTO | null
  demandBill?: AuctionDemandSnapshotDTO | null
  className?: string
}

export const TreasuryAuctionDemandCard: React.FC<TreasuryAuctionDemandCardProps> = ({
  demandNote,
  demandBill,
  className = '',
}) => {
  const [selectedTab, setSelectedTab] = useState<'Note' | 'Bill'>('Note')
  const activeDemand = selectedTab === 'Note' ? demandNote : demandBill

  if (!demandNote && !demandBill) return null

  const btc = activeDemand?.latest_bid_to_cover_ratio
  const priorMean = activeDemand?.prior_mean_bid_to_cover
  const delta = activeDemand?.demand_delta

  return (
    <div className={`rounded-2xl border border-sky-100 bg-white/80 p-5 shadow-[0_8px_25px_rgba(14,165,233,0.06)] backdrop-blur-md ${className}`}>
      <div className="mb-4 flex flex-wrap items-center justify-between gap-3 border-b border-sky-100/70 pb-3">
        <div>
          <h3 className="text-base font-bold text-zinc-900 tracking-tight flex items-center gap-2">
            <span>US Treasury Auction Demand</span>
            <span className="rounded-full bg-sky-100 text-sky-800 font-mono text-[10px] font-bold px-2.5 py-0.5">
              FISCAL DATA
            </span>
          </h3>
          <p className="text-xs text-zinc-500">
            Latest auction demand vs prior 8 completed auctions (Moving Average)
          </p>
        </div>

        <div className="inline-flex rounded-lg border border-sky-200/80 bg-slate-100/80 p-1">
          <button
            type="button"
            className={`rounded px-3 py-1 text-xs font-semibold transition-colors ${
              selectedTab === 'Note' ? 'bg-white text-sky-800 shadow-xs' : 'text-zinc-600 hover:text-zinc-900'
            }`}
            onClick={() => setSelectedTab('Note')}
          >
            10-Year Note
          </button>
          <button
            type="button"
            className={`rounded px-3 py-1 text-xs font-semibold transition-colors ${
              selectedTab === 'Bill' ? 'bg-white text-sky-800 shadow-xs' : 'text-zinc-600 hover:text-zinc-900'
            }`}
            onClick={() => setSelectedTab('Bill')}
          >
            13-Week Bill
          </button>
        </div>
      </div>

      {activeDemand ? (
        <div className="space-y-4">
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            <div className="rounded-xl border border-slate-100 bg-slate-50/80 p-3.5 transition-all hover:bg-white hover:shadow-xs hover:border-sky-200">
              <div className="text-[11px] font-medium text-zinc-500 uppercase tracking-wider">Latest Bid-to-Cover</div>
              <div className="mt-1 font-mono text-xl font-bold text-zinc-900">
                {btc !== null && btc !== undefined ? `${btc.toFixed(2)}x` : 'N/A'}
              </div>
              <div className="mt-0.5 text-[10px] text-zinc-400">As of {activeDemand.latest_auction_date}</div>
            </div>

            <div className="rounded-xl border border-slate-100 bg-slate-50/80 p-3.5 transition-all hover:bg-white hover:shadow-xs hover:border-sky-200">
              <div className="text-[11px] font-medium text-zinc-500 uppercase tracking-wider">Prior 8-Auction Mean</div>
              <div className="mt-1 font-mono text-xl font-bold text-zinc-700">
                {priorMean !== null && priorMean !== undefined ? `${priorMean.toFixed(2)}x` : 'N/A (< 3)'}
              </div>
              <div className="mt-0.5 text-[10px] text-zinc-400">{activeDemand.sample_count} samples</div>
            </div>

            <div className="rounded-xl border border-slate-100 bg-slate-50/80 p-3.5 transition-all hover:bg-white hover:shadow-xs hover:border-sky-200">
              <div className="text-[11px] font-medium text-zinc-500 uppercase tracking-wider">Demand Delta</div>
              <div className={`mt-1 font-mono text-xl font-bold ${
                (delta ?? 0) >= 0 ? 'text-emerald-600' : 'text-rose-600'
              }`}>
                {delta !== null && delta !== undefined ? `${delta >= 0 ? '+' : ''}${delta.toFixed(2)}x` : 'N/A'}
              </div>
              <div className="mt-0.5 text-[10px] text-zinc-400">Latest - Prior Mean</div>
            </div>

            <div className="rounded-xl border border-slate-100 bg-slate-50/80 p-3.5 transition-all hover:bg-white hover:shadow-xs hover:border-sky-200">
              <div className="text-[11px] font-medium text-zinc-500 uppercase tracking-wider">
                {activeDemand.security_type === 'Bill' ? 'Investment Rate' : 'High Yield'}
              </div>
              <div className="mt-1 font-mono text-xl font-bold text-sky-700">
                {activeDemand.security_type === 'Bill'
                  ? (activeDemand.latest_high_investment_rate !== null && activeDemand.latest_high_investment_rate !== undefined
                      ? `${activeDemand.latest_high_investment_rate.toFixed(3)}%`
                      : 'N/A')
                  : (activeDemand.latest_high_yield !== null && activeDemand.latest_high_yield !== undefined
                      ? `${activeDemand.latest_high_yield.toFixed(3)}%`
                      : 'N/A')}
              </div>
              <div className="mt-0.5 text-[10px] text-zinc-400">Awarded rate</div>
            </div>
          </div>

          <div className="flex flex-wrap items-center justify-between text-xs text-zinc-500 border-t border-sky-100/70 pt-3">
            <div>
              Offering: <span className="font-mono font-semibold text-zinc-800">${(activeDemand.latest_offering_amount_usd ?? 0).toLocaleString()}</span> • Accepted: <span className="font-mono font-semibold text-zinc-800">${(activeDemand.latest_total_accepted_usd ?? 0).toLocaleString()}</span>
            </div>
            <div className="text-zinc-500">
              Source: <span className="font-medium text-zinc-700">{activeDemand.source}</span>
            </div>
          </div>

          <div className="rounded-xl border border-amber-200/80 bg-amber-50/70 p-3 text-[11px] text-amber-900 flex items-start gap-2">
            <span className="shrink-0 text-base leading-none">⚠️</span>
            <span className="leading-relaxed">
              <strong>Contract Invariant:</strong> Auction tail is intentionally excluded as US Treasury Fiscal Data does not provide When-Issued (WI) market yields. Moving average is computed across identical security type and term only.
            </span>
          </div>
        </div>
      ) : (
        <div className="py-4 text-center text-xs text-zinc-500">
          No auction demand snapshot available for {selectedTab}.
        </div>
      )}
    </div>
  )
}
