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
    <div className={`rounded-xl border border-slate-800 bg-slate-900/90 p-5 shadow-lg backdrop-blur-sm ${className}`}>
      <div className="mb-4 flex flex-wrap items-center justify-between gap-3 border-b border-slate-800/80 pb-3">
        <div>
          <h3 className="text-base font-semibold text-slate-100 flex items-center gap-2">
            <span>US Treasury Auction Demand</span>
            <span className="rounded-full bg-slate-800 text-slate-300 font-mono text-[10px] font-bold px-2 py-0.5">
              FISCAL DATA
            </span>
          </h3>
          <p className="text-xs text-slate-400">
            Latest auction demand vs prior 8 completed auctions (Moving Average)
          </p>
        </div>

        <div className="inline-flex rounded-lg border border-slate-800 bg-slate-950 p-1">
          <button
            type="button"
            className={`rounded px-3 py-1 text-xs font-semibold transition-colors ${
              selectedTab === 'Note' ? 'bg-slate-800 text-slate-100' : 'text-slate-400 hover:text-slate-200'
            }`}
            onClick={() => setSelectedTab('Note')}
          >
            10-Year Note
          </button>
          <button
            type="button"
            className={`rounded px-3 py-1 text-xs font-semibold transition-colors ${
              selectedTab === 'Bill' ? 'bg-slate-800 text-slate-100' : 'text-slate-400 hover:text-slate-200'
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
            <div className="rounded-lg border border-slate-800 bg-slate-950/70 p-3">
              <div className="text-[11px] text-slate-400">Latest Bid-to-Cover</div>
              <div className="mt-1 font-mono text-xl font-bold text-slate-100">
                {btc !== null && btc !== undefined ? `${btc.toFixed(2)}x` : 'N/A'}
              </div>
              <div className="text-[10px] text-slate-500">As of {activeDemand.latest_auction_date}</div>
            </div>

            <div className="rounded-lg border border-slate-800 bg-slate-950/70 p-3">
              <div className="text-[11px] text-slate-400">Prior 8-Auction Mean</div>
              <div className="mt-1 font-mono text-xl font-bold text-slate-300">
                {priorMean !== null && priorMean !== undefined ? `${priorMean.toFixed(2)}x` : 'N/A (< 3)'}
              </div>
              <div className="text-[10px] text-slate-500">{activeDemand.sample_count} samples</div>
            </div>

            <div className="rounded-lg border border-slate-800 bg-slate-950/70 p-3">
              <div className="text-[11px] text-slate-400">Demand Delta</div>
              <div className={`mt-1 font-mono text-xl font-bold ${
                (delta ?? 0) >= 0 ? 'text-emerald-400' : 'text-rose-400'
              }`}>
                {delta !== null && delta !== undefined ? `${delta >= 0 ? '+' : ''}${delta.toFixed(2)}x` : 'N/A'}
              </div>
              <div className="text-[10px] text-slate-500">Latest - Prior Mean</div>
            </div>

            <div className="rounded-lg border border-slate-800 bg-slate-950/70 p-3">
              <div className="text-[11px] text-slate-400">
                {activeDemand.security_type === 'Bill' ? 'Investment Rate' : 'High Yield'}
              </div>
              <div className="mt-1 font-mono text-xl font-bold text-sky-400">
                {activeDemand.security_type === 'Bill'
                  ? (activeDemand.latest_high_investment_rate !== null && activeDemand.latest_high_investment_rate !== undefined
                      ? `${activeDemand.latest_high_investment_rate.toFixed(3)}%`
                      : 'N/A')
                  : (activeDemand.latest_high_yield !== null && activeDemand.latest_high_yield !== undefined
                      ? `${activeDemand.latest_high_yield.toFixed(3)}%`
                      : 'N/A')}
              </div>
              <div className="text-[10px] text-slate-500">Awarded rate</div>
            </div>
          </div>

          <div className="flex flex-wrap items-center justify-between text-xs text-slate-400 border-t border-slate-800/80 pt-3">
            <div>
              Offering: <span className="font-mono text-slate-200">${(activeDemand.latest_offering_amount_usd ?? 0).toLocaleString()}</span> • Accepted: <span className="font-mono text-slate-200">${(activeDemand.latest_total_accepted_usd ?? 0).toLocaleString()}</span>
            </div>
            <div className="text-slate-500">
              Source: {activeDemand.source}
            </div>
          </div>

          <div className="rounded-lg border border-amber-900/40 bg-amber-950/20 p-2.5 text-[11px] text-amber-300 flex items-start gap-2">
            <span className="shrink-0">⚠️</span>
            <span>
              <strong>Contract Invariant:</strong> Auction tail is intentionally excluded as US Treasury Fiscal Data does not provide When-Issued (WI) market yields. Moving average is computed across identical security type and term only.
            </span>
          </div>
        </div>
      ) : (
        <div className="py-4 text-center text-xs text-slate-500">
          No auction demand snapshot available for {selectedTab}.
        </div>
      )}
    </div>
  )
}
