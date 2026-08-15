import { useState, useEffect, useMemo } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { api } from '../api/client'
import type { EquitySummaryDTO, EquityDetailDTO } from '../api/types'
import type { StockOption } from '../components/equity/RunEquityAnalysisModal'
import { EquityDetail } from '../components/equity/EquityDetail'
import { sentimentClass } from '../lib/sentiment'
import TextInput from '../components/ui/TextInput'
import ScoreRing from '../components/equity/ScoreRing'
import RunEquityAnalysisModal from '../components/equity/RunEquityAnalysisModal'
import Toast from '../components/common/Toast'

// For explicit mock mode
const isMockMode = import.meta.env.VITE_EQUITY_MOCK === 'true' && import.meta.env.DEV

const STAGGER_STEP_MS = 30
const STAGGER_CAP_MS = 240

export function normalizeTicker(sym: string): string {
  return sym.trim().toUpperCase().replace(/\.(BK|TH)$/i, '')
}

export interface SidebarStockItem {
  symbol: string
  normalizedSymbol: string
  label?: string
  market: 'US' | 'TH'
  source: 'portfolio' | 'watchlist' | 'other'
  hasReport: boolean
  report?: EquitySummaryDTO
}

export default function Equity() {
  const { ticker } = useParams<{ ticker?: string }>()
  const navigate = useNavigate()

  const [summaries, setSummaries] = useState<EquitySummaryDTO[]>([])
  const [loadingList, setLoadingList] = useState(true)
  const [listError, setListError] = useState<string | null>(null)
  const [searchQuery, setSearchQuery] = useState('')
  const [sortBy, setSortBy] = useState<'ticker' | 'score_desc'>('ticker')
  const [activeFilter, setActiveFilter] = useState<'all' | 'portfolio' | 'watchlist'>('all')

  const [portfolioStocks, setPortfolioStocks] = useState<StockOption[]>([])
  const [watchlistStocks, setWatchlistStocks] = useState<StockOption[]>([])

  const [detailData, setDetailData] = useState<EquityDetailDTO | undefined>(undefined)
  const [detailStatus, setDetailStatus] = useState<'idle' | 'loading' | 'success' | 'error' | 'not-found'>('idle')
  const [detailError, setDetailError] = useState<string | undefined>(undefined)

  const [modalTicker, setModalTicker] = useState<string | undefined>(undefined)
  const [modalMarket, setModalMarket] = useState<'US' | 'TH' | undefined>(undefined)
  const [isModalOpen, setIsModalOpen] = useState(false)
  const [toastState, setToastState] = useState<{ message: string; actionLabel?: string; onAction?: () => void } | null>(null)

  // 1. Fetch analyzed equity summaries
  useEffect(() => {
    const fetchList = async () => {
      try {
        setLoadingList(true)
        if (isMockMode) {
          const { mockEquitySummary } = await import('../mocks/equity')
          setSummaries(mockEquitySummary)
        } else {
          const data = await api.getEquityLatest()
          setSummaries(data)
        }
        setListError(null)
      } catch (err: any) {
        console.error('Failed to fetch equity list:', err)
        setListError(err.message || 'Failed to load list')
      } finally {
        setLoadingList(false)
      }
    }
    fetchList()
  }, [])

  // 2. Fetch portfolio & watchlist stocks (fail-soft)
  useEffect(() => {
    let isMounted = true

    Promise.allSettled([
      api.getActualPortfolioState(),
      api.getActualWatchlist(),
    ]).then(([pResult, wResult]) => {
      if (!isMounted) return

      if (pResult.status === 'fulfilled' && pResult.value?.holdings) {
        const stocks: StockOption[] = pResult.value.holdings
          .filter((h) => h.asset_type === 'Stock' || h.asset_type?.toLowerCase() === 'stock')
          .map((h) => {
            const isUSD = (h.avg_cost_usd !== null && h.avg_cost_usd !== undefined) || (h.current_price_usd !== null && h.current_price_usd !== undefined)
            return {
              symbol: h.symbol,
              market: (isUSD ? 'US' : 'TH') as 'US' | 'TH',
              label: h.company_name || h.symbol,
              source: 'portfolio' as const,
            }
          })
        setPortfolioStocks(stocks)
      }

      if (wResult.status === 'fulfilled' && wResult.value?.items) {
        const stocks: StockOption[] = wResult.value.items
          .filter((item) => item.asset_type === 'Stock' || item.asset_type?.toLowerCase() === 'stock')
          .map((item) => {
            const upper = item.symbol.trim().toUpperCase()
            const isTH = upper.endsWith('.BK') || upper.endsWith('.TH')
            return {
              symbol: item.symbol,
              market: (isTH ? 'TH' : 'US') as 'US' | 'TH',
              label: item.notes || item.symbol,
              source: 'watchlist' as const,
            }
          })
        setWatchlistStocks(stocks)
      }
    })

    return () => {
      isMounted = false
    }
  }, [])

  // 3. Fetch detail when route ticker param changes
  useEffect(() => {
    const fetchDetail = async () => {
      if (!ticker) {
        setDetailStatus('idle')
        setDetailData(undefined)
        return
      }

      try {
        setDetailStatus('loading')
        if (isMockMode) {
          const { mockEquityDetailAAPL } = await import('../mocks/equity')
          if (ticker.toUpperCase() === 'AAPL') {
            setDetailData(mockEquityDetailAAPL)
            setDetailStatus('success')
          } else {
            setDetailStatus('not-found')
          }
        } else {
          const data = await api.getEquityDetail(ticker)
          setDetailData(data)
          setDetailStatus('success')
        }
      } catch (err: any) {
        console.error('Failed to fetch equity detail:', err)
        if (err.status === 404) {
          setDetailStatus('not-found')
        } else {
          setDetailStatus('error')
          setDetailError(err.message || 'Error loading detail')
        }
      }
    }

    fetchDetail()
  }, [ticker])

  const openAnalysisModal = (sym?: string, mkt?: 'US' | 'TH') => {
    setModalTicker(sym)
    setModalMarket(mkt)
    setIsModalOpen(true)
  }

  // 4. Derive categorized stock items with normalized symbol matching
  const { portfolioItems, watchlistItems, otherItems, totalAvailableCount } = useMemo(() => {
    const query = searchQuery.trim().toLowerCase()

    // Map portfolio holdings
    const pItems: SidebarStockItem[] = portfolioStocks.map((st) => {
      const norm = normalizeTicker(st.symbol)
      const report = summaries.find((s) => normalizeTicker(s.ticker) === norm)
      return {
        symbol: st.symbol,
        normalizedSymbol: norm,
        label: st.label,
        market: report?.market || st.market,
        source: 'portfolio',
        hasReport: Boolean(report),
        report,
      }
    })

    // Portfolio normalized symbol set for deduplication
    const portfolioNormSet = new Set(pItems.map((p) => p.normalizedSymbol))

    // Map watchlist items (deduplicated against portfolio)
    const wItems: SidebarStockItem[] = watchlistStocks
      .filter((st) => !portfolioNormSet.has(normalizeTicker(st.symbol)))
      .map((st) => {
        const norm = normalizeTicker(st.symbol)
        const report = summaries.find((s) => normalizeTicker(s.ticker) === norm)
        return {
          symbol: st.symbol,
          normalizedSymbol: norm,
          label: st.label,
          market: report?.market || st.market,
          source: 'watchlist',
          hasReport: Boolean(report),
          report,
        }
      })

    const allMyNormSet = new Set([...portfolioNormSet, ...wItems.map((w) => w.normalizedSymbol)])

    // Map other analyzed summaries (not in portfolio and not in watchlist)
    const oItems: SidebarStockItem[] = summaries
      .filter((s) => !allMyNormSet.has(normalizeTicker(s.ticker)))
      .map((s) => ({
        symbol: s.ticker,
        normalizedSymbol: normalizeTicker(s.ticker),
        label: s.company_name || s.ticker,
        market: s.market,
        source: 'other',
        hasReport: true,
        report: s,
      }))

    // Filter helper
    const matchesQuery = (item: SidebarStockItem) => {
      if (!query) return true
      const symMatch = item.symbol.toLowerCase().includes(query)
      const normMatch = item.normalizedSymbol.toLowerCase().includes(query)
      const labelMatch = item.label ? item.label.toLowerCase().includes(query) : false
      const compMatch = item.report?.company_name ? item.report.company_name.toLowerCase().includes(query) : false
      return symMatch || normMatch || labelMatch || compMatch
    }

    // Sort helper
    const sortItems = (a: SidebarStockItem, b: SidebarStockItem) => {
      if (sortBy === 'score_desc') {
        const scoreA = a.report?.composite_score ?? -1
        const scoreB = b.report?.composite_score ?? -1
        if (scoreA !== scoreB) {
          return scoreB - scoreA
        }
      }
      return a.symbol.localeCompare(b.symbol)
    }

    const filteredP = pItems.filter(matchesQuery).sort(sortItems)
    const filteredW = wItems.filter(matchesQuery).sort(sortItems)
    const filteredO = oItems.filter(matchesQuery).sort(sortItems)

    return {
      portfolioItems: filteredP,
      watchlistItems: filteredW,
      otherItems: filteredO,
      totalAvailableCount: pItems.length + wItems.length + oItems.length,
    }
  }, [summaries, portfolioStocks, watchlistStocks, searchQuery, sortBy])

  const renderStockItem = (item: SidebarStockItem, index: number) => {
    const isSelected = ticker ? normalizeTicker(ticker) === item.normalizedSymbol : false

    if (item.hasReport && item.report) {
      const report = item.report
      return (
        <li key={`${item.source}-${item.symbol}`}>
          <button
            type="button"
            onClick={() => navigate(`/equity/${report.ticker.toLowerCase()}`)}
            style={{ animationDelay: `${Math.min(index * STAGGER_STEP_MS, STAGGER_CAP_MS)}ms` }}
            className={`animate-card-in w-full text-left p-3 rounded-xl border transition-all flex items-center gap-3 group ${
              isSelected
                ? 'bg-sky-50/90 border-sky-300 shadow-sm ring-1 ring-sky-300/60'
                : 'bg-panel/70 border-edge hover:border-sky-200 hover:bg-surface-strong'
            }`}
          >
            <div className="flex-1 min-w-0">
              <div className="flex items-center justify-between gap-1 mb-1">
                <div className="flex items-center gap-1.5 min-w-0">
                  <span className="font-extrabold text-zinc-900 font-mono text-sm sm:text-base truncate group-hover:text-sky-700 transition-colors">
                    {item.symbol}
                  </span>
                  <span
                    className={`rounded px-1.5 py-0.5 text-[10px] font-bold border uppercase ${
                      item.market === 'TH'
                        ? 'bg-emerald-50 text-emerald-700 border-emerald-200'
                        : 'bg-blue-50 text-blue-700 border-blue-200'
                    }`}
                  >
                    {item.market}
                  </span>
                </div>
                <span className="text-[11px] text-zinc-500 shrink-0">
                  {new Date(report.evaluated_at).toLocaleDateString('th-TH', { month: 'short', day: 'numeric' })}
                </span>
              </div>
              <div className="flex items-center gap-2 text-xs">
                {item.label && item.label !== item.symbol && (
                  <span className="text-zinc-600 truncate max-w-[120px]" title={item.label}>
                    {item.label}
                  </span>
                )}
                <span className={`px-2 py-0.5 rounded-full border text-[11px] font-semibold whitespace-nowrap ml-auto ${sentimentClass(report.market_sentiment)}`}>
                  {report.market_sentiment}
                </span>
              </div>
            </div>
            <div className="shrink-0 flex items-center justify-center">
              <ScoreRing score={report.composite_score} size={42} textSizeClass="text-xs font-bold" />
            </div>
          </button>
        </li>
      )
    }

    // Unanalyzed item (from portfolio or watchlist)
    return (
      <li key={`${item.source}-${item.symbol}`}>
        <div
          style={{ animationDelay: `${Math.min(index * STAGGER_STEP_MS, STAGGER_CAP_MS)}ms` }}
          className="animate-card-in w-full text-left p-3 rounded-xl border border-dashed border-amber-200/90 bg-amber-50/40 hover:bg-amber-50/70 transition-all flex items-center justify-between gap-2.5"
        >
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-1.5 mb-1">
              <span className="font-extrabold text-zinc-900 font-mono text-sm sm:text-base truncate">
                {item.symbol}
              </span>
              <span
                className={`rounded px-1.5 py-0.5 text-[10px] font-bold border uppercase ${
                  item.market === 'TH'
                    ? 'bg-emerald-50 text-emerald-700 border-emerald-200'
                    : 'bg-blue-50 text-blue-700 border-blue-200'
                }`}
              >
                {item.market}
              </span>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-xs text-zinc-500 truncate max-w-[140px]" title={item.label || item.symbol}>
                {item.label || (item.source === 'portfolio' ? 'พอร์ตลงทุน' : 'Watchlist')}
              </span>
              <span className="rounded-full bg-amber-100/80 px-2 py-0.5 text-[10px] font-bold text-amber-800 border border-amber-200 shrink-0">
                ⏳ ยังไม่วิเคราะห์
              </span>
            </div>
          </div>

          <button
            type="button"
            onClick={() => openAnalysisModal(item.symbol, item.market)}
            className="shrink-0 rounded-lg bg-gradient-to-r from-sky-500 to-blue-600 hover:from-sky-600 hover:to-blue-700 text-white text-xs font-bold px-3 py-1.5 shadow-sm hover:shadow transition-all flex items-center gap-1 active:scale-95"
            title={`สั่งวิเคราะห์หุ้น ${item.symbol} (${item.market})`}
          >
            <span>🚀</span>
            <span>วิเคราะห์</span>
          </button>
        </div>
      </li>
    )
  }

  return (
    <div className="animate-page-in flex h-[calc(100vh-4rem)] flex-col md:flex-row">
      {/* Sidebar List */}
      <div className={`w-full md:w-1/3 md:min-w-[320px] md:max-w-[420px] border-r border-edge overflow-y-auto bg-surface p-4 ${ticker ? 'hidden md:block' : 'block'}`}>
        <div className="flex items-center justify-between mb-3">
          <div>
            <h2 className="text-xl font-bold text-zinc-900 flex items-center gap-2">
              <span>Equity Analysis</span>
              {isMockMode && <span className="text-xs bg-purple-100 text-purple-800 px-2 py-0.5 rounded-full">MOCK</span>}
            </h2>
            <p className="text-xs text-zinc-500 mt-0.5">ผลวิเคราะห์หุ้น ปัจจัยพื้นฐาน และ Valuation</p>
          </div>
          <button
            type="button"
            onClick={() => openAnalysisModal()}
            className="rounded-xl bg-sky-500 hover:bg-sky-600 text-white p-2 text-xs font-bold shadow-sm transition-all flex items-center gap-1 active:scale-95 shrink-0"
            title="วิเคราะห์หุ้นใหม่"
          >
            <span>+ วิเคราะห์หุ้น</span>
          </button>
        </div>

        {/* Filter Switcher Chips */}
        <div className="flex items-center gap-1 p-1 bg-surface-strong/80 rounded-xl border border-edge mb-3 text-xs font-semibold">
          <button
            type="button"
            onClick={() => setActiveFilter('all')}
            className={`flex-1 py-1.5 px-2 rounded-lg text-center transition-all ${
              activeFilter === 'all'
                ? 'bg-panel text-sky-700 shadow-2xs font-bold border border-sky-100'
                : 'text-zinc-500 hover:text-zinc-800'
            }`}
          >
            ทั้งหมด ({totalAvailableCount})
          </button>
          <button
            type="button"
            onClick={() => setActiveFilter('portfolio')}
            className={`flex-1 py-1.5 px-2 rounded-lg text-center transition-all ${
              activeFilter === 'portfolio'
                ? 'bg-panel text-sky-700 shadow-2xs font-bold border border-sky-100'
                : 'text-zinc-500 hover:text-zinc-800'
            }`}
          >
            💼 ในพอร์ต ({portfolioStocks.length})
          </button>
          <button
            type="button"
            onClick={() => setActiveFilter('watchlist')}
            className={`flex-1 py-1.5 px-2 rounded-lg text-center transition-all ${
              activeFilter === 'watchlist'
                ? 'bg-panel text-amber-700 shadow-2xs font-bold border border-amber-100'
                : 'text-zinc-500 hover:text-zinc-800'
            }`}
          >
            ⭐ Watch ({watchlistStocks.length})
          </button>
        </div>

        {/* Search & Sort Controls */}
        <div className="mb-4 space-y-2.5">
          <TextInput
            placeholder="ค้นหาหุ้น (เช่น AAPL, PTT)..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full"
          />
          <div className="flex items-center gap-2">
            <span className="text-xs text-zinc-500 whitespace-nowrap font-medium">เรียงตาม:</span>
            <select
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value as 'ticker' | 'score_desc')}
              className="w-full rounded-lg border border-edge bg-surface px-3 py-1.5 text-xs text-zinc-900 focus:border-sky-500 focus:outline-none focus:ring-1 focus:ring-sky-500 font-medium"
            >
              <option value="ticker">ชื่อหุ้น (A-Z)</option>
              <option value="score_desc">คะแนนประเมิน (มากไปน้อย)</option>
            </select>
          </div>
        </div>

        {loadingList && summaries.length === 0 ? (
          <div className="space-y-2">
            {Array.from({ length: 4 }).map((_, i) => (
              <div key={i} className="animate-shimmer h-16 rounded-xl border border-edge" />
            ))}
          </div>
        ) : listError ? (
          <div className="text-xs text-red-500 p-3 bg-red-50 border border-red-200 rounded-xl">{listError}</div>
        ) : (
          <div className="space-y-4">
            {/* 1. Portfolio Section */}
            {(activeFilter === 'all' || activeFilter === 'portfolio') && portfolioItems.length > 0 && (
              <div>
                <div className="flex items-center justify-between mb-2 px-1">
                  <span className="text-xs font-bold uppercase tracking-wider text-sky-800 flex items-center gap-1.5">
                    <span>💼 หุ้นในพอร์ต</span>
                    <span className="rounded-full bg-sky-100 text-sky-700 px-1.5 py-0.2 text-[10px] font-extrabold">
                      {portfolioItems.length}
                    </span>
                  </span>
                </div>
                <ul className="space-y-2">
                  {portfolioItems.map((item, idx) => renderStockItem(item, idx))}
                </ul>
              </div>
            )}

            {/* 2. Watchlist Section */}
            {(activeFilter === 'all' || activeFilter === 'watchlist') && watchlistItems.length > 0 && (
              <div>
                <div className="flex items-center justify-between mb-2 px-1">
                  <span className="text-xs font-bold uppercase tracking-wider text-amber-800 flex items-center gap-1.5">
                    <span>⭐ Watchlist</span>
                    <span className="rounded-full bg-amber-100 text-amber-700 px-1.5 py-0.2 text-[10px] font-extrabold">
                      {watchlistItems.length}
                    </span>
                  </span>
                </div>
                <ul className="space-y-2">
                  {watchlistItems.map((item, idx) => renderStockItem(item, idx))}
                </ul>
              </div>
            )}

            {/* 3. Other Analyzed Equities Section */}
            {activeFilter === 'all' && otherItems.length > 0 && (
              <div>
                <div className="flex items-center justify-between mb-2 px-1">
                  <span className="text-xs font-bold uppercase tracking-wider text-zinc-500 flex items-center gap-1.5">
                    <span>📊 วิเคราะห์แล้วอื่นๆ</span>
                    <span className="rounded-full bg-zinc-100 text-zinc-600 px-1.5 py-0.2 text-[10px] font-extrabold">
                      {otherItems.length}
                    </span>
                  </span>
                </div>
                <ul className="space-y-2">
                  {otherItems.map((item, idx) => renderStockItem(item, idx))}
                </ul>
              </div>
            )}

            {/* Empty State */}
            {portfolioItems.length === 0 && watchlistItems.length === 0 && otherItems.length === 0 && (
              <div className="text-xs text-zinc-500 text-center py-6 bg-surface-strong border border-edge rounded-xl">
                {searchQuery ? 'ไม่พบหุ้นที่ตรงกับคำค้นหา' : 'ยังไม่มีข้อมูลหุ้นในส่วนนี้'}
              </div>
            )}
          </div>
        )}
      </div>

      {/* Main Detail Area */}
      <div className={`flex-1 overflow-y-auto p-4 md:p-8 ${!ticker ? 'hidden md:block' : 'block'}`}>
        {!ticker ? (
          <div className="flex flex-col items-center justify-center h-full text-zinc-400 gap-3">
            <svg className="w-12 h-12 text-sky-300" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden="true">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
            </svg>
            <span className="text-sm font-medium">เลือกหุ้นจากรายการด้านซ้ายเพื่อดูรายละเอียดการวิเคราะห์</span>
          </div>
        ) : (
          <div className="max-w-5xl mx-auto">
            <div className="md:hidden mb-4">
              <button
                onClick={() => navigate('/equity')}
                className="flex items-center text-sm font-semibold text-sky-600 hover:text-sky-800"
              >
                <svg className="w-4 h-4 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 19l-7-7m0 0l7-7m-7 7h18" />
                </svg>
                กลับไปหน้ารายการ
              </button>
            </div>
            <EquityDetail
              status={detailStatus}
              data={detailData}
              errorMessage={detailError}
              onOpenAnalysisModal={(t) => {
                openAnalysisModal(t)
              }}
            />
          </div>
        )}
      </div>

      {isModalOpen && (
        <RunEquityAnalysisModal
          initialTicker={modalTicker}
          initialMarket={modalMarket}
          portfolioOptions={portfolioStocks}
          watchlistOptions={watchlistStocks}
          onClose={() => setIsModalOpen(false)}
          onDispatched={(_jobId, _cardId, dispatchedTicker) => {
            setToastState({
              message: `สั่งงานวิเคราะห์หุ้น ${dispatchedTicker} และดึงข่าวเรียบร้อย`,
              actionLabel: 'ดูสถานะใน Kanban',
              onAction: () => navigate('/kanban'),
            })
          }}
        />
      )}

      {toastState && (
        <Toast
          message={toastState.message}
          actionLabel={toastState.actionLabel}
          onAction={toastState.onAction}
          onClose={() => setToastState(null)}
        />
      )}
    </div>
  )
}
