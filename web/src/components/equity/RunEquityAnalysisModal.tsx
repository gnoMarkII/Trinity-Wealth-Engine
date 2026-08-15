import React, { useState, useEffect } from 'react'
import Modal from '../ui/Modal'
import TextInput from '../ui/TextInput'
import Button from '../ui/Button'
import { api, ApiError } from '../../api/client'

export interface StockOption {
  symbol: string
  market: 'US' | 'TH'
  label?: string
  source: 'portfolio' | 'watchlist'
}

export interface RunEquityAnalysisModalProps {
  initialTicker?: string
  initialMarket?: 'US' | 'TH'
  portfolioOptions?: StockOption[]
  watchlistOptions?: StockOption[]
  onClose: () => void
  onDispatched?: (jobId: string, cardId: string, ticker: string) => void
}

export const RunEquityAnalysisModal: React.FC<RunEquityAnalysisModalProps> = ({
  initialTicker = '',
  initialMarket,
  portfolioOptions: propPortfolioOptions,
  watchlistOptions: propWatchlistOptions,
  onClose,
  onDispatched,
}) => {
  const [ticker, setTicker] = useState(initialTicker)
  const [market, setMarket] = useState<'US' | 'TH'>(initialMarket || 'US')
  const [userTouchedMarket, setUserTouchedMarket] = useState(Boolean(initialMarket))
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const [portfolioOptions, setPortfolioOptions] = useState<StockOption[]>(propPortfolioOptions || [])
  const [watchlistOptions, setWatchlistOptions] = useState<StockOption[]>(propWatchlistOptions || [])
  const [loadingStocks, setLoadingStocks] = useState(false)

  // Fetch portfolio & watchlist stocks on mount if not provided via props (fail-soft)
  useEffect(() => {
    if (propPortfolioOptions !== undefined && propWatchlistOptions !== undefined) {
      setPortfolioOptions(propPortfolioOptions)
      setWatchlistOptions(propWatchlistOptions)
      return
    }

    let isMounted = true
    setLoadingStocks(true)

    Promise.allSettled([
      api.getActualPortfolioState(),
      api.getActualWatchlist(),
    ]).then(([pResult, wResult]) => {
      if (!isMounted) return

      if (pResult.status === 'fulfilled' && pResult.value?.holdings) {
        const stocks = pResult.value.holdings
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
        setPortfolioOptions(stocks)
      }

      if (wResult.status === 'fulfilled' && wResult.value?.items) {
        const stocks = wResult.value.items
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
        setWatchlistOptions(stocks)
      }
    }).finally(() => {
      if (isMounted) setLoadingStocks(false)
    })

    return () => {
      isMounted = false
    }
  }, [propPortfolioOptions, propWatchlistOptions])

  // Auto-detect market based on ticker format if user hasn't explicitly set it
  useEffect(() => {
    if (userTouchedMarket) return
    const upper = ticker.trim().toUpperCase()
    if (upper.endsWith('.BK') || upper.endsWith('.TH')) {
      setMarket('TH')
    } else if (upper && !upper.includes('.')) {
      setMarket('US')
    }
  }, [ticker, userTouchedMarket])

  const handleSelectOption = (opt: StockOption) => {
    setTicker(opt.symbol.toUpperCase())
    setMarket(opt.market)
    setUserTouchedMarket(true)
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    const cleanTicker = ticker.trim().toUpperCase()
    if (!cleanTicker) {
      setError('กรุณาระบุ Ticker หุ้น (เช่น AAPL หรือ PTT.BK)')
      return
    }

    setLoading(true)
    setError(null)

    const cardTitle = `วิเคราะห์หุ้น ${cleanTicker} (${market})`
    const instruction = `วิเคราะห์หุ้น ${cleanTicker} (${market}) และดึงข่าวล่าสุดพร้อมประเมิน Valuation`

    let cardId: string | undefined
    try {
      const { card } = await api.createKanbanCard(cardTitle, 'manager', instruction, 'both')
      cardId = card.card_id
    } catch (err) {
      console.error('Failed to create equity kanban card:', err)
      setError(err instanceof ApiError ? err.message : 'เกิดข้อผิดพลาดในการสร้างการ์ดวิเคราะห์หุ้น')
      setLoading(false)
      return
    }

    try {
      const job = await api.dispatchJob(instruction, cardId, 'manager', 'both')
      onDispatched?.(job.job_id, cardId, cleanTicker)
      onClose()
    } catch (err) {
      console.error('Failed to dispatch equity job:', err)
      setError(
        (err instanceof ApiError ? err.message : 'เกิดข้อผิดพลาดในการเริ่มงานวิเคราะห์') +
          ' — สร้างการ์ดไว้ที่ Backlog แล้ว กด dispatch เองในการ์ดนั้นได้'
      )
    } finally {
      setLoading(false)
    }
  }

  const hasOptions = portfolioOptions.length > 0 || watchlistOptions.length > 0

  return (
    <Modal titleId="run-equity-analysis-title" onClose={onClose}>
      <form onSubmit={handleSubmit} className="space-y-4">
        <div className="flex items-center justify-between border-b border-zinc-100 pb-3">
          <h3 id="run-equity-analysis-title" className="text-base font-semibold text-zinc-900">
            📊 วิเคราะห์หุ้นและดึงข่าวล่าสุด
          </h3>
          <button
            type="button"
            onClick={onClose}
            aria-label="Close"
            className="text-zinc-400 hover:text-zinc-600 transition-colors"
          >
            ✕
          </button>
        </div>

        {error && (
          <div className="rounded-xl border border-rose-200 bg-rose-50 p-3 text-xs text-rose-700">
            {error}
          </div>
        )}

        <div className="space-y-3">
          {loadingStocks && (
            <div className="text-xs text-zinc-400 animate-pulse">กำลังโหลดรายชื่อหุ้นจาก พอร์ต และ Watchlist...</div>
          )}

          {hasOptions && (
            <div>
              <label htmlFor="run-equity-select-opt" className="block text-xs font-medium text-zinc-700 mb-1">
                เลือกหุ้นจาก Portfolio / Watchlist
              </label>
              <select
                id="run-equity-select-opt"
                aria-label="เลือกหุ้นจากระบบ"
                defaultValue=""
                onChange={(e) => {
                  const val = e.target.value
                  if (!val) return
                  const [src, sym] = val.split(':')
                  const list = src === 'portfolio' ? portfolioOptions : watchlistOptions
                  const match = list.find((opt) => opt.symbol === sym)
                  if (match) {
                    handleSelectOption(match)
                  }
                  e.target.value = ''
                }}
                className="w-full rounded-xl border border-sky-200 bg-panel px-3 py-2 text-sm text-zinc-900 outline-none shadow-sm transition-colors focus:border-flow-cyan focus:ring-2 focus:ring-flow-cyan/20"
              >
                <option value="">-- เลือกหุ้นจาก Portfolio หรือ Watchlist --</option>
                {portfolioOptions.length > 0 && (
                  <optgroup label="💼 พอร์ตลงทุน (Portfolio)">
                    {portfolioOptions.map((opt) => (
                      <option key={`p-${opt.symbol}`} value={`portfolio:${opt.symbol}`}>
                        {opt.symbol} {opt.label && opt.label !== opt.symbol ? `— ${opt.label}` : ''} ({opt.market})
                      </option>
                    ))}
                  </optgroup>
                )}
                {watchlistOptions.length > 0 && (
                  <optgroup label="⭐ รายการตามติด (Watchlist)">
                    {watchlistOptions.map((opt) => (
                      <option key={`w-${opt.symbol}`} value={`watchlist:${opt.symbol}`}>
                        {opt.symbol} {opt.label && opt.label !== opt.symbol ? `— ${opt.label}` : ''} ({opt.market})
                      </option>
                    ))}
                  </optgroup>
                )}
              </select>

              <div className="flex flex-wrap gap-1.5 mt-2">
                {portfolioOptions.slice(0, 6).map((opt) => (
                  <button
                    key={`pill-p-${opt.symbol}`}
                    type="button"
                    onClick={() => handleSelectOption(opt)}
                    className={`rounded-lg px-2.5 py-1 text-xs font-medium transition-colors border ${
                      ticker === opt.symbol.toUpperCase()
                        ? 'bg-sky-500 text-white border-sky-600'
                        : 'bg-sky-50 text-sky-800 border-sky-200 hover:bg-sky-100'
                    }`}
                  >
                    💼 {opt.symbol}
                  </button>
                ))}
                {watchlistOptions.slice(0, 6).map((opt) => (
                  <button
                    key={`pill-w-${opt.symbol}`}
                    type="button"
                    onClick={() => handleSelectOption(opt)}
                    className={`rounded-lg px-2.5 py-1 text-xs font-medium transition-colors border ${
                      ticker === opt.symbol.toUpperCase()
                        ? 'bg-amber-500 text-white border-amber-600'
                        : 'bg-amber-50 text-amber-800 border-amber-200 hover:bg-amber-100'
                    }`}
                  >
                    ⭐ {opt.symbol}
                  </button>
                ))}
              </div>
            </div>
          )}

          <div>
            <label htmlFor="run-equity-ticker" className="block text-xs font-medium text-zinc-700 mb-1">
              Ticker Symbol (สัญลักษณ์หุ้น)
            </label>
            <TextInput
              id="run-equity-ticker"
              value={ticker}
              onChange={(e) => setTicker(e.target.value.toUpperCase())}
              placeholder="เช่น AAPL, NVDA, PTT.BK"
              className="w-full"
              autoFocus
            />
          </div>

          <div>
            <label htmlFor="run-equity-market" className="block text-xs font-medium text-zinc-700 mb-1">
              ตลาด (Market)
            </label>
            <select
              id="run-equity-market"
              value={market}
              onChange={(e) => {
                setMarket(e.target.value as 'US' | 'TH')
                setUserTouchedMarket(true)
              }}
              className="w-full rounded-xl border border-sky-200 bg-panel px-3 py-2 text-sm text-zinc-900 outline-none shadow-sm transition-colors focus:border-flow-cyan focus:ring-2 focus:ring-flow-cyan/20"
            >
              <option value="US">🇺🇸 สหรัฐฯ (US - S&P 500 / NASDAQ)</option>
              <option value="TH">🇹🇭 ไทย (TH - SET Index)</option>
            </select>
          </div>

          <div className="rounded-xl border border-sky-100 bg-sky-50/50 p-3 text-xs text-sky-800 leading-relaxed">
            💡 ระบบจะทำการสร้างการ์ดลง Kanban Board บันทึกลง Obsidian Vault และเริ่มงานวิเคราะห์ Quant + ข่าวพร้อมกันทันที
          </div>
        </div>

        <div className="flex justify-end gap-2 pt-2 border-t border-zinc-100">
          <Button type="button" variant="secondary" onClick={onClose} disabled={loading}>
            ยกเลิก
          </Button>
          <Button type="submit" variant="primary" disabled={loading || !ticker.trim()}>
            {loading ? 'กำลังส่งงาน...' : '🚀 สร้างการ์ดและเริ่มวิเคราะห์'}
          </Button>
        </div>
      </form>
    </Modal>
  )
}

export default RunEquityAnalysisModal

