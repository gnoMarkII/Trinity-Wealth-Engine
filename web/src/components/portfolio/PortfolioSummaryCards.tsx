import type { ActualSummaryDTO } from '../../api/types'

interface Props {
  summary: ActualSummaryDTO | null
  lastUpdated: string | null
  loading?: boolean
  refreshingPrices?: boolean
  priceRefreshInfo?: Record<string, string> | null
  onRefreshPrices?: () => void
}

export function formatTHB(val: number): string {
  return new Intl.NumberFormat('th-TH', {
    style: 'currency',
    currency: 'THB',
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  }).format(val)
}

export default function PortfolioSummaryCards({
  summary,
  lastUpdated,
  loading = false,
  refreshingPrices = false,
  priceRefreshInfo = null,
  onRefreshPrices,
}: Props) {
  if (loading || !summary) {
    return (
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {[1, 2, 3, 4].map((i) => (
          <div key={i} className="h-28 animate-pulse rounded-2xl border border-sky-100 bg-panel/60 p-5 shadow-sm" />
        ))}
      </div>
    )
  }

  const nav = summary.total_value_thb
  const unrealizedPnL = summary.total_unrealized_profit
  const costBasis = summary.total_cost_basis_thb ?? 0
  const pnlPct = costBasis > 0 ? (unrealizedPnL / costBasis) * 100 : null
  const isPositive = unrealizedPnL >= 0

  return (
    <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
      {/* Card 1: Total NAV */}
      <div className="flex flex-col justify-between rounded-2xl border border-sky-100 bg-gradient-to-br from-panel via-panel to-sky-50/40 p-5 shadow-sm transition-all hover:shadow-md">
        <div className="flex items-center justify-between text-xs font-semibold uppercase tracking-wider text-zinc-500">
          <span>Total Portfolio NAV</span>
          <span className="rounded-full bg-flow-cyan/10 px-2 py-0.5 text-[10px] text-flow-blue">THB</span>
        </div>
        <div className="mt-2 text-2xl font-bold tracking-tight text-zinc-900 sm:text-3xl">
          {formatTHB(nav)}
        </div>
        <div className="mt-2 flex items-center text-xs text-zinc-400">
          <span>มูลค่าสุทธิปัจจุบัน (รวมเงินสด)</span>
        </div>
      </div>

      {/* Card 2: Total Unrealized Profit */}
      <div className="flex flex-col justify-between rounded-2xl border border-sky-100 bg-gradient-to-br from-panel via-panel to-sky-50/40 p-5 shadow-sm transition-all hover:shadow-md">
        <div className="flex items-center justify-between text-xs font-semibold uppercase tracking-wider text-zinc-500">
          <span>Unrealized Profit/Loss</span>
          <span
            className={`rounded-full px-2 py-0.5 text-[10px] font-bold ${
              pnlPct === null
                ? 'bg-zinc-100 text-zinc-500'
                : isPositive
                  ? 'bg-emerald-50 text-emerald-700'
                  : 'bg-rose-50 text-rose-700'
            }`}
          >
            {pnlPct === null ? '—' : `${isPositive ? '+' : ''}${pnlPct.toFixed(2)}%`}
          </span>
        </div>
        <div
          className={`mt-2 text-2xl font-bold tracking-tight sm:text-3xl ${
            isPositive ? 'text-emerald-600' : 'text-rose-600'
          }`}
        >
          {isPositive ? '+' : ''}
          {formatTHB(unrealizedPnL)}
        </div>
        <div className="mt-2 flex items-center text-xs text-zinc-400">
          <span>กำไร/ขาดทุนที่ยังไม่รับรู้</span>
        </div>
      </div>

      {/* Card 3: Passive Income YTD */}
      <div className="flex flex-col justify-between rounded-2xl border border-sky-100 bg-gradient-to-br from-panel via-panel to-sky-50/40 p-5 shadow-sm transition-all hover:shadow-md">
        <div className="flex items-center justify-between text-xs font-semibold uppercase tracking-wider text-zinc-500">
          <span>Passive Income YTD</span>
          <span className="rounded-full bg-amber-50 px-2 py-0.5 text-[10px] font-semibold text-amber-700">
            เงินปันผลสะสม
          </span>
        </div>
        <div className="mt-2 text-2xl font-bold tracking-tight text-zinc-900 sm:text-3xl">
          {formatTHB(summary.passive_income_ytd)}
        </div>
        <div className="mt-2 flex items-center text-xs text-zinc-400">
          <span>เงินปันผลที่ได้รับตั้งแต่ต้นปี</span>
        </div>
      </div>

      {/* Card 4: Last Updated & Refresh */}
      <div className="flex flex-col justify-between rounded-2xl border border-sky-100 bg-gradient-to-br from-panel via-panel to-sky-50/40 p-5 shadow-sm transition-all hover:shadow-md">
        <div className="flex items-center justify-between text-xs font-semibold uppercase tracking-wider text-zinc-500">
          <span>Market Prices Status</span>
          {refreshingPrices && (
            <span className="inline-flex items-center gap-1 rounded-full bg-sky-100 px-2 py-0.5 text-[10px] font-medium text-sky-800 animate-pulse">
              Refreshing...
            </span>
          )}
        </div>
        <div className="mt-1 flex flex-col justify-center">
          <span className="text-sm font-semibold text-zinc-800">
            {lastUpdated ? new Date(lastUpdated).toLocaleString('th-TH') : 'N/A'}
          </span>
          <span className="mt-0.5 text-xs text-zinc-400">อัปเดตราคาล่าสุดจาก yfinance</span>
        </div>
        <div className="mt-2 pt-1">
          <button
            type="button"
            onClick={onRefreshPrices}
            disabled={refreshingPrices}
            className="flex w-full items-center justify-center gap-1.5 rounded-xl border border-sky-200 bg-flow-cyan/10 px-3 py-1.5 text-xs font-semibold text-flow-blue transition-all hover:bg-flow-cyan/20 disabled:cursor-not-allowed disabled:opacity-50"
          >
            <svg
              className={`h-3.5 w-3.5 ${refreshingPrices ? 'animate-spin' : ''}`}
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
              strokeWidth={2}
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                d="M16.023 9.348h4.992v-.001M2.985 19.644v-4.992m0 0h4.992m-4.993 0l3.181 3.183a8.25 8.25 0 0013.803-3.7M4.031 9.865a8.25 8.25 0 0113.803-3.7l3.181 3.182m0-4.991v4.99"
              />
            </svg>
            {refreshingPrices ? 'กำลังอัปเดตราคา...' : 'อัปเดตราคาตลาด (Refresh)'}
          </button>
        </div>
        {priceRefreshInfo && Object.keys(priceRefreshInfo).length > 0 && (
          <div className="mt-2 max-h-16 overflow-y-auto rounded-lg bg-zinc-50 p-1.5 text-[10px] space-y-0.5 border border-zinc-200">
            {Object.entries(priceRefreshInfo).map(([sym, status]) => (
              <div key={sym} className="flex justify-between items-center font-mono">
                <span className="font-bold text-zinc-700">{sym}:</span>
                <span
                  className={
                    status.toLowerCase().includes('ok') || status.toLowerCase().includes('updated')
                      ? 'text-emerald-600'
                      : 'text-amber-600'
                  }
                >
                  {status}
                </span>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
