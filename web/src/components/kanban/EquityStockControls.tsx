import { useState, useEffect } from 'react'
import { api } from '../../api/client'
import TextInput from '../ui/TextInput'

export interface StockOption {
  symbol: string
  market: 'US' | 'TH'
  label?: string
  source: 'portfolio' | 'watchlist'
}

export interface ParsedEquityStock {
  ticker: string
  market: 'US' | 'TH'
}

export function parseEquityStock(prompt: string, title?: string): ParsedEquityStock {
  const text = `${title || ''} ${prompt || ''}`
  const matchWithMarket = text.match(/วิเคราะห์หุ้น\s+([A-Z0-9.\-_]+)\s*\((US|TH)\)/i)
  if (matchWithMarket && matchWithMarket[1] && matchWithMarket[2]) {
    return {
      ticker: matchWithMarket[1].toUpperCase(),
      market: matchWithMarket[2].toUpperCase() as 'US' | 'TH',
    }
  }

  const matchTicker = text.match(/วิเคราะห์หุ้น\s+([A-Z0-9.\-_]+)/i)
  if (matchTicker && matchTicker[1]) {
    const ticker = matchTicker[1].toUpperCase()
    const isTH = ticker.endsWith('.BK') || ticker.endsWith('.TH')
    return {
      ticker,
      market: isTH ? 'TH' : 'US',
    }
  }

  return { ticker: 'AAPL', market: 'US' }
}

export function updateEquityPromptAndTitle(
  originalPrompt: string,
  originalTitle: string,
  newTicker: string,
  newMarket: 'US' | 'TH'
): { title: string; prompt: string } {
  const cleanTicker = newTicker.trim().toUpperCase()

  let updatedTitle = originalTitle
  if (originalTitle && /วิเคราะห์หุ้น\s+[A-Z0-9.\-_]+/i.test(originalTitle)) {
    updatedTitle = originalTitle.replace(
      /วิเคราะห์หุ้น\s+[A-Z0-9.\-_]+(?:\s*\((?:US|TH)\))?/i,
      `วิเคราะห์หุ้น ${cleanTicker} (${newMarket})`
    )
  } else {
    updatedTitle = `วิเคราะห์หุ้น ${cleanTicker} (${newMarket})`
  }

  let updatedPrompt = originalPrompt
  if (originalPrompt && /วิเคราะห์หุ้น\s+[A-Z0-9.\-_]+/i.test(originalPrompt)) {
    updatedPrompt = originalPrompt.replace(
      /วิเคราะห์หุ้น\s+[A-Z0-9.\-_]+(?:\s*\((?:US|TH)\))?/i,
      `วิเคราะห์หุ้น ${cleanTicker} (${newMarket})`
    )
  } else {
    updatedPrompt = `วิเคราะห์หุ้น ${cleanTicker} (${newMarket}) และดึงข่าวล่าสุดพร้อมประเมิน Valuation`
  }

  return { title: updatedTitle, prompt: updatedPrompt }
}

interface Props {
  prompt: string
  title?: string
  onChange: (updated: { title: string; prompt: string }) => void
  disabled?: boolean
  className?: string
}

export default function EquityStockControls({
  prompt,
  title = '',
  onChange,
  disabled = false,
  className = '',
}: Props) {
  const current = parseEquityStock(prompt, title)
  const [ticker, setTicker] = useState(current.ticker)
  const [market, setMarket] = useState<'US' | 'TH'>(current.market)
  const [userTouchedMarket, setUserTouchedMarket] = useState(false)

  const [portfolioOptions, setPortfolioOptions] = useState<StockOption[]>([])
  const [watchlistOptions, setWatchlistOptions] = useState<StockOption[]>([])
  const [loadingStocks, setLoadingStocks] = useState(false)

  // Keep state in sync if prompt/title changes externally
  useEffect(() => {
    const updated = parseEquityStock(prompt, title)
    setTicker(updated.ticker)
    if (!userTouchedMarket) {
      setMarket(updated.market)
    }
  }, [prompt, title, userTouchedMarket])

  // Load portfolio & watchlist stocks on mount (fail-soft)
  useEffect(() => {
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
            const isTH = h.avg_cost_thb != null
            return {
              symbol: h.symbol,
              market: (isTH ? 'TH' : 'US') as 'US' | 'TH',
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
  }, [])

  const applyStockChange = (newTicker: string, newMarket: 'US' | 'TH') => {
    setTicker(newTicker)
    setMarket(newMarket)
    const result = updateEquityPromptAndTitle(prompt, title, newTicker, newMarket)
    onChange(result)
  }

  const handleSelectOption = (opt: StockOption) => {
    const cleanSym = opt.symbol.toUpperCase()
    setUserTouchedMarket(true)
    applyStockChange(cleanSym, opt.market)
  }

  const handleManualTickerChange = (newSym: string) => {
    const cleanSym = newSym.toUpperCase()
    setTicker(cleanSym)
    let nextMarket = market
    if (!userTouchedMarket) {
      if (cleanSym.endsWith('.BK') || cleanSym.endsWith('.TH')) {
        nextMarket = 'TH'
      } else if (cleanSym && !cleanSym.includes('.')) {
        nextMarket = 'US'
      }
    }
    const result = updateEquityPromptAndTitle(prompt, title, cleanSym, nextMarket)
    onChange(result)
  }

  const handleMarketChange = (newMarket: 'US' | 'TH') => {
    setMarket(newMarket)
    setUserTouchedMarket(true)
    const result = updateEquityPromptAndTitle(prompt, title, ticker, newMarket)
    onChange(result)
  }

  const hasOptions = portfolioOptions.length > 0 || watchlistOptions.length > 0

  return (
    <div className={`rounded-xl border border-sky-200/80 bg-gradient-to-br from-sky-50/70 to-sky-50/30 p-3.5 shadow-sm ${className}`}>
      <div className="flex items-center justify-between border-b border-sky-100 pb-2.5">
        <div className="flex items-center gap-1.5">
          <span className="text-base">📊</span>
          <span id="equity-stock-control-label" className="text-xs font-semibold text-sky-900">
            เลือกรุ่นหุ้นสำหรับวิเคราะห์และดึงข่าว
          </span>
        </div>
        {loadingStocks && (
          <span className="text-[11px] text-sky-600 animate-pulse">กำลังโหลดหุ้น...</span>
        )}
      </div>

      <div className="mt-3 space-y-2.5">
        {hasOptions && (
          <div>
            <label htmlFor="equity-select-stock" className="block text-[11px] font-semibold text-sky-800 mb-1">
              เลือกจาก Portfolio / Watchlist
            </label>
            <select
              id="equity-select-stock"
              disabled={disabled}
              aria-label="เลือกหุ้นจากระบบ"
              value=""
              onChange={(e) => {
                const val = e.target.value
                if (!val) return
                const [src, sym] = val.split(':')
                const list = src === 'portfolio' ? portfolioOptions : watchlistOptions
                const match = list.find((opt) => opt.symbol === sym)
                if (match) {
                  handleSelectOption(match)
                }
              }}
              className="w-full rounded-lg border border-sky-200 bg-white px-2.5 py-1.5 text-xs text-zinc-900 outline-none shadow-sm transition-colors focus:border-flow-cyan focus:ring-1 focus:ring-flow-cyan/30 disabled:opacity-50"
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
                  disabled={disabled}
                  onClick={() => handleSelectOption(opt)}
                  className={`rounded-lg px-2.5 py-1 text-xs font-medium transition-all ${
                    ticker === opt.symbol.toUpperCase()
                      ? 'bg-sky-600 text-white shadow-sm shadow-sky-500/20'
                      : 'border border-sky-200 bg-white text-sky-800 hover:bg-sky-100/60'
                  }`}
                >
                  💼 {opt.symbol}
                </button>
              ))}
              {watchlistOptions.slice(0, 6).map((opt) => (
                <button
                  key={`pill-w-${opt.symbol}`}
                  type="button"
                  disabled={disabled}
                  onClick={() => handleSelectOption(opt)}
                  className={`rounded-lg px-2.5 py-1 text-xs font-medium transition-all ${
                    ticker === opt.symbol.toUpperCase()
                      ? 'bg-amber-600 text-white shadow-sm shadow-amber-500/20'
                      : 'border border-amber-200 bg-white text-amber-800 hover:bg-amber-100/60'
                  }`}
                >
                  ⭐ {opt.symbol}
                </button>
              ))}
            </div>
          </div>
        )}

        <div className="grid grid-cols-1 gap-2 sm:grid-cols-3">
          <div className="sm:col-span-2">
            <label htmlFor="equity-ticker-input" className="block text-[11px] font-semibold text-sky-800 mb-1">
              Ticker Symbol
            </label>
            <TextInput
              id="equity-ticker-input"
              disabled={disabled}
              uiSize="sm"
              value={ticker}
              onChange={(e) => handleManualTickerChange(e.target.value)}
              placeholder="เช่น AAPL, NVDA, PTT.BK"
              className="w-full bg-white"
            />
          </div>
          <div>
            <label htmlFor="equity-market-select" className="block text-[11px] font-semibold text-sky-800 mb-1">
              ตลาด (Market)
            </label>
            <select
              id="equity-market-select"
              disabled={disabled}
              value={market}
              onChange={(e) => handleMarketChange(e.target.value as 'US' | 'TH')}
              className="w-full rounded-xl border border-sky-200 bg-white px-2 py-1.5 text-xs font-medium text-zinc-900 outline-none shadow-sm transition-colors focus:border-flow-cyan focus:ring-1 focus:ring-flow-cyan/30 disabled:opacity-50"
            >
              <option value="US">🇺🇸 US</option>
              <option value="TH">🇹🇭 TH</option>
            </select>
          </div>
        </div>
      </div>
    </div>
  )
}
