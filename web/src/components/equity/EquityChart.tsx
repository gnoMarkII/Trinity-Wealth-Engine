import React, { useEffect, useRef, useState, useCallback } from 'react'
import type {
  CorporateActionEventDTO,
  CorporateActionsMetadataDTO,
  OHLCVCandleDTO,
  ChartInterval,
  IndicatorWarmupDetailDTO,
} from '../../api/types'
import {
  initKlineChart,
  type KlineChartInstance,
  type IndicatorConfig,
  type OverlayIndicatorType,
  type MomentumIndicatorType,
  type DrawingToolType,
} from './klineAdapter'

export type ChartRange = string

export const INTERVAL_OPTIONS: { key: ChartInterval; label: string }[] = [
  { key: '15m', label: '15m' },
  { key: '1h', label: '1H' },
  { key: '1d', label: '1D' },
  { key: '1wk', label: '1W' },
  { key: '1mo', label: '1M' },
]

export const INTERVAL_RANGES: Record<ChartInterval, { key: string; label: string }[]> = {
  '15m': [
    { key: '5d', label: '5D' },
    { key: '1mo', label: '1M' },
  ],
  '1h': [
    { key: '1mo', label: '1M' },
    { key: '3mo', label: '3M' },
    { key: '6mo', label: '6M' },
    { key: '1y', label: '1Y' },
    { key: '2y', label: '2Y' },
  ],
  '1d': [
    { key: '1mo', label: '1M' },
    { key: '3mo', label: '3M' },
    { key: '6mo', label: '6M' },
    { key: '1y', label: '1Y' },
    { key: '5y', label: '5Y' },
    { key: 'max', label: 'MAX' },
  ],
  '1wk': [
    { key: '1y', label: '1Y' },
    { key: '5y', label: '5Y' },
    { key: 'max', label: 'MAX' },
  ],
  '1mo': [
    { key: '5y', label: '5Y' },
    { key: 'max', label: 'MAX' },
  ],
}

interface EquityChartProps {
  ticker: string
  companyName?: string
  currentPrice: number | null
  priceChange: number | null
  priceChangePct: number | null
  priceAsOf?: string | null
  currency: 'USD' | 'THB'
  candles: OHLCVCandleDTO[]
  selectedInterval?: ChartInterval
  onIntervalChange?: (interval: ChartInterval) => void
  selectedRange: ChartRange
  onRangeChange: (range: ChartRange) => void
  allowedRanges?: string[]
  isLoading: boolean
  displayStartTimestamp?: number | null
  availableWarmupBars?: number
  requiredWarmupBars?: number
  warmupStatus?: string
  indicatorWarmup?: Record<string, IndicatorWarmupDetailDTO>
  events?: CorporateActionEventDTO[]
  eventsMetadata?: CorporateActionsMetadataDTO | null
  week52High?: number | null
  week52Low?: number | null
  week52CoverageDays?: number
  valuationTargets?: import('../../api/types').ValuationTargetsDTO | null
  insiderFilings?: import('../../api/types').InsiderFilingDTO[]
  market?: 'US' | 'TH'
  analystContext?: import('../../api/types').AnalystContextDTO | null
}

const OVERLAY_OPTIONS: { key: OverlayIndicatorType; label: string }[] = [
  { key: 'BOLL', label: 'BOLL' },
  { key: 'EMA', label: 'EMA' },
  { key: 'SMA', label: 'SMA' },
  { key: 'NONE', label: 'None' },
]

const MOMENTUM_OPTIONS: { key: MomentumIndicatorType; label: string }[] = [
  { key: 'RSI', label: 'RSI' },
  { key: 'MACD', label: 'MACD' },
  { key: 'NONE', label: 'None' },
]


const CONFIG_STORAGE_KEY = 'equity_chart_indicator_config:v2'
const DEFAULT_CONFIG: IndicatorConfig = {
  overlay: 'BOLL',
  volume: false,
  momentum: 'NONE',
  atr: false,
  events: true,
  week52: false,
  dcf: false,
  insider: false,
  vwap: false,
  consensus: true,
}

function loadIndicatorConfig(): IndicatorConfig {
  try {
    const raw = localStorage.getItem(CONFIG_STORAGE_KEY)
    if (!raw) return DEFAULT_CONFIG
    const parsed = JSON.parse(raw)
    const validOverlays: OverlayIndicatorType[] = ['BOLL', 'EMA', 'SMA', 'NONE']
    const validMomentums: MomentumIndicatorType[] = ['RSI', 'MACD', 'NONE']

    const overlay = validOverlays.includes(parsed.overlay) ? parsed.overlay : DEFAULT_CONFIG.overlay
    const volume = typeof parsed.volume === 'boolean' ? parsed.volume : DEFAULT_CONFIG.volume
    const momentum = validMomentums.includes(parsed.momentum)
      ? parsed.momentum
      : DEFAULT_CONFIG.momentum
    const atr = typeof parsed.atr === 'boolean' ? parsed.atr : DEFAULT_CONFIG.atr
    const events = typeof parsed.events === 'boolean' ? parsed.events : DEFAULT_CONFIG.events
    const week52 = typeof parsed.week52 === 'boolean' ? parsed.week52 : DEFAULT_CONFIG.week52
    const dcf = typeof parsed.dcf === 'boolean' ? parsed.dcf : DEFAULT_CONFIG.dcf
    const insider = typeof parsed.insider === 'boolean' ? parsed.insider : DEFAULT_CONFIG.insider
    const vwap = typeof parsed.vwap === 'boolean' ? parsed.vwap : DEFAULT_CONFIG.vwap
    const consensus = typeof parsed.consensus === 'boolean' ? parsed.consensus : DEFAULT_CONFIG.consensus

    return { overlay, volume, momentum, atr, events, week52, dcf, insider, vwap, consensus }
  } catch {
    return DEFAULT_CONFIG
  }
}



function saveIndicatorConfig(config: IndicatorConfig) {
  try {
    localStorage.setItem(CONFIG_STORAGE_KEY, JSON.stringify(config))
  } catch {
    // Ignore storage quota or disabled errors
  }
}

export const EquityChart: React.FC<EquityChartProps> = ({
  ticker,
  companyName,
  currentPrice,
  priceChange,
  priceChangePct,
  priceAsOf,
  currency,
  candles,
  selectedInterval = '1d',
  onIntervalChange,
  selectedRange,
  onRangeChange,
  allowedRanges,
  isLoading,
  displayStartTimestamp,
  availableWarmupBars = 0,
  requiredWarmupBars: _requiredWarmupBars = 200,
  warmupStatus = 'unknown',
  indicatorWarmup,
  events = [],
  eventsMetadata,
  week52High,
  week52Low,
  week52CoverageDays = 0,
  valuationTargets,
  insiderFilings,
  market,
  analystContext,
}) => {
  const chartContainerRef = useRef<HTMLDivElement>(null)
  const chartInstanceRef = useRef<KlineChartInstance | null>(null)
  const [config, setConfig] = useState<IndicatorConfig>(loadIndicatorConfig)
  const [activeDrawing, setActiveDrawing] = useState<DrawingToolType>(null)
  const [hoveredEvent, setHoveredEvent] = useState<{
    event: CorporateActionEventDTO
    pos: { x: number; y: number }
  } | null>(null)
  const [hoveredInsider, setHoveredInsider] = useState<{
    filing: import('../../api/types').InsiderMarkerHoverDTO
    pos: { x: number; y: number }
  } | null>(null)
  const [pinnedInsider, setPinnedInsider] = useState<import('../../api/types').InsiderMarkerHoverDTO | null>(null)

  const currSign = currency === 'THB' ? '฿' : '$'
  const isPositive = (priceChange ?? 0) >= 0
  const changeColorClass = isPositive ? 'text-emerald-600' : 'text-rose-600'
  const changeBadgeBg = isPositive
    ? 'bg-emerald-50 border-emerald-200'
    : 'bg-rose-50 border-rose-200'

  const updateConfig = useCallback((newConfig: Partial<IndicatorConfig>) => {
    setConfig((prev) => {
      const updated = { ...prev, ...newConfig }
      saveIndicatorConfig(updated)
      return updated
    })
  }, [])

  const handleCorporateHover = useCallback(
    (ev: CorporateActionEventDTO | null, pos: { x: number; y: number } | null) => {
      if (ev && pos) {
        setHoveredEvent({ event: ev, pos })
      } else {
        setHoveredEvent(null)
      }
    },
    []
  )

  const handleInsiderHover = useCallback(
    (
      payload: {
        filing: import('../../api/types').InsiderMarkerHoverDTO
        pos: { x: number; y: number }
      } | null,
      isHover: boolean
    ) => {
      if (isHover) {
        setHoveredInsider(payload)
      } else {
        // Clicked -> pin
        if (payload) {
          setPinnedInsider(payload.filing)
        }
      }
    },
    []
  )

  const isIntraday = selectedInterval === '15m' || selectedInterval === '1h'

  // Initialize or atomic update of chart data & parameters
  useEffect(() => {
    if (!chartContainerRef.current) return

    const intervalMap: Record<ChartInterval, { span: number; type: 'minute' | 'hour' | 'day' | 'week' | 'month' }> = {
      '15m': { span: 15, type: 'minute' },
      '1h': { span: 1, type: 'hour' },
      '1d': selectedRange === '5y' ? { span: 1, type: 'week' } : { span: 1, type: 'day' },
      '1wk': { span: 1, type: 'week' },
      '1mo': { span: 1, type: 'month' },
    }
    const period = intervalMap[selectedInterval || '1d'] || { span: 1, type: 'day' }

    if (!chartInstanceRef.current) {
      chartInstanceRef.current = initKlineChart(
        chartContainerRef.current,
        candles,
        ticker,
        period,
        config,
        displayStartTimestamp
      )
    }

    chartInstanceRef.current.updateChart({
      symbol: ticker,
      period,
      bars: candles,
      displayStartTimestamp,
      events,
      week52High,
      week52Low,
      latestClose: currentPrice,
      currency,
      coverageCalendarDays: week52CoverageDays,
      valuationTargets,
      insiderFilings,
      analystContext,
    })
  }, [
    ticker,
    candles,
    selectedInterval,
    selectedRange,
    displayStartTimestamp,
    events,
    week52High,
    week52Low,
    currentPrice,
    currency,
    week52CoverageDays,
    valuationTargets,
    insiderFilings,
    analystContext,
  ])

  // Sync indicator config changes atomically to KlineChartInstance
  useEffect(() => {
    if (!chartInstanceRef.current) return
    chartInstanceRef.current.setOverlay?.(config.overlay)
    chartInstanceRef.current.setVolume?.(config.volume)
    chartInstanceRef.current.setMomentum?.(config.momentum)
    chartInstanceRef.current.setATR?.(config.atr)
    chartInstanceRef.current.setEvents?.(events, config.events, handleCorporateHover)
    chartInstanceRef.current.set52WLevels?.({
      high: week52High ?? null,
      low: week52Low ?? null,
      latestClose: currentPrice,
      currency,
      coverageCalendarDays: week52CoverageDays,
      visible: config.week52,
    })
    chartInstanceRef.current.setValuationTargets?.({
      targets: valuationTargets ?? null,
      visible: config.dcf,
    })
    chartInstanceRef.current.setInsiderFilings?.(insiderFilings ?? [], config.insider, handleInsiderHover)
    chartInstanceRef.current.resize?.()
  }, [
    config,
    events,
    week52High,
    week52Low,
    currentPrice,
    currency,
    week52CoverageDays,
    valuationTargets,
    insiderFilings,
    handleCorporateHover,
    handleInsiderHover,
  ])

  // VWAP Lifecycle Effect: automatically enabled only on intraday (15m, 1h)
  useEffect(() => {
    if (!chartInstanceRef.current) return
    const tz =
      analystContext?.exchange_tz ??
      (market === 'TH' || currency === 'THB' ? 'Asia/Bangkok' : 'America/New_York')
    chartInstanceRef.current.setVWAP?.(config.vwap && isIntraday, tz)
  }, [selectedInterval, config.vwap, analystContext?.exchange_tz, market, currency, isIntraday])

  // Street Consensus Lifecycle Effect: synchronizes ray overlay on price or context change
  useEffect(() => {
    if (!chartInstanceRef.current) return
    chartInstanceRef.current.setAnalystContext?.({
      ctx: analystContext ?? null,
      visible: config.consensus,
      currentPrice,
      currency,
    })
  }, [analystContext, config.consensus, currentPrice, currency, candles])

  // Handle Escape key to cancel active drawing mode
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        chartInstanceRef.current?.cancelActiveDrawing()
        setActiveDrawing(null)
      }
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [])

  // ResizeObserver and destroy cleanup
  useEffect(() => {
    if (!chartContainerRef.current) return

    let ro: ResizeObserver | null = null
    if (typeof ResizeObserver !== 'undefined') {
      ro = new ResizeObserver(() => {
        chartInstanceRef.current?.resize()
      })
      ro.observe(chartContainerRef.current)
    }

    return () => {
      ro?.disconnect()
      if (chartInstanceRef.current) {
        chartInstanceRef.current.destroy()
        chartInstanceRef.current = null
      }
    }
  }, [])

  // Toolbar Keyboard Navigation handler
  const handleToolbarKeyDown = (e: React.KeyboardEvent<HTMLDivElement>) => {
    if (['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown'].includes(e.key)) {
      e.preventDefault()
      const buttons = Array.from(
        e.currentTarget.querySelectorAll<HTMLButtonElement>('button:not([disabled])')
      )
      const currentIndex = buttons.indexOf(document.activeElement as HTMLButtonElement)
      if (currentIndex === -1) return

      let nextIndex = currentIndex
      if (e.key === 'ArrowRight' || e.key === 'ArrowDown') {
        nextIndex = (currentIndex + 1) % buttons.length
      } else if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') {
        nextIndex = (currentIndex - 1 + buttons.length) % buttons.length
      }
      buttons[nextIndex]?.focus()
    }
  }

  const handleDrawingToolClick = (tool: DrawingToolType) => {
    if (activeDrawing === tool) {
      chartInstanceRef.current?.cancelActiveDrawing()
      setActiveDrawing(null)
    } else {
      setActiveDrawing(tool)
      chartInstanceRef.current?.enableDrawing(tool)
    }
  }

  const handleClearDrawings = () => {
    chartInstanceRef.current?.clearDrawings()
    setActiveDrawing(null)
  }

  // Dynamic Height calculation: base 440px + 85px per active subpane (Volume, Momentum, ATR)
  const subpaneCount =
    (config.volume ? 1 : 0) + (config.momentum !== 'NONE' ? 1 : 0) + (config.atr ? 1 : 0)
  const containerHeightClass =
    subpaneCount === 0
      ? 'h-[440px]'
      : subpaneCount === 1
        ? 'h-[525px]'
        : subpaneCount === 2
          ? 'h-[610px]'
          : 'h-[695px]'

  // Quality Warning Banner visibility
  const showQualityWarning = warmupStatus === 'insufficient' && config.overlay === 'EMA'

  return (
    <div className="flex flex-col rounded-2xl border border-edge bg-panel p-5 shadow-sm">
      {/* Header Bar */}
      <div className="flex flex-wrap items-start justify-between gap-4 border-b border-edge/60 pb-4">
        <div>
          <div className="flex flex-wrap items-center gap-2">
            <h3 className="font-serif text-2xl font-bold tracking-tight text-zinc-900">
              {ticker}
            </h3>
            {companyName && (
              <span className="text-sm font-medium text-zinc-500 truncate max-w-[200px] sm:max-w-xs">
                {companyName}
              </span>
            )}

            {/* Execution Axis Badges */}
            <div className="flex flex-wrap items-center gap-1.5 ml-1">
              {config.overlay === 'BOLL' && (
                <span className="rounded-full bg-sky-50 px-2 py-0.5 text-xs font-semibold text-sky-700 border border-sky-200">
                  BOLL (20, 2)
                </span>
              )}
              {config.overlay === 'EMA' && (
                <span className="rounded-full bg-indigo-50 px-2 py-0.5 text-xs font-semibold text-indigo-700 border border-indigo-200">
                  EMA (20, 50, 200)
                </span>
              )}
              {config.overlay === 'SMA' && (
                <span className="rounded-full bg-amber-50 px-2 py-0.5 text-xs font-semibold text-amber-700 border border-amber-200">
                  SMA (20, 50)
                </span>
              )}
              {config.vwap && isIntraday && (
                <span className="rounded-full bg-indigo-50 px-2 py-0.5 text-xs font-semibold text-indigo-700 border border-indigo-200">
                  VWAP
                </span>
              )}
              {config.volume && (
                <span className="rounded-full bg-slate-100 px-2 py-0.5 text-xs font-semibold text-slate-700 border border-slate-300">
                  VOL (MA20)
                </span>
              )}
              {config.momentum === 'RSI' && (
                <span className="rounded-full bg-teal-50 px-2 py-0.5 text-xs font-semibold text-teal-700 border border-teal-200">
                  RSI (14)
                </span>
              )}
              {config.momentum === 'MACD' && (
                <span className="rounded-full bg-purple-50 px-2 py-0.5 text-xs font-semibold text-purple-700 border border-purple-200">
                  MACD (12, 26, 9)
                </span>
              )}
              {config.atr && (
                <span className="rounded-full bg-amber-50 px-2 py-0.5 text-xs font-semibold text-amber-700 border border-amber-200">
                  ATR (14)
                </span>
              )}
              {config.week52 && week52High && (
                <span className="rounded-full bg-emerald-50 px-2 py-0.5 text-xs font-semibold text-emerald-700 border border-emerald-200">
                  52W Levels
                </span>
              )}
              {config.dcf && valuationTargets?.status === 'available' && valuationTargets.comparability_status === 'comparable' && (
                <span className="rounded-full bg-teal-50 px-2 py-0.5 text-xs font-semibold text-teal-700 border border-teal-200">
                  DCF ({valuationTargets.valuation_verdict})
                </span>
              )}
              {config.consensus && analystContext?.target_mean && (
                <span className="rounded-full bg-amber-50 px-2 py-0.5 text-xs font-semibold text-amber-700 border border-amber-200">
                  Street (${analystContext.target_mean.toFixed(2)})
                </span>
              )}
              {config.insider && (insiderFilings?.length ?? 0) > 0 && (
                <span className="rounded-full bg-purple-50 px-2 py-0.5 text-xs font-semibold text-purple-700 border border-purple-200">
                  Insider ({insiderFilings?.length} filings)
                </span>
              )}
            </div>
          </div>

          {/* Current Price and Daily Change */}
          <div className="mt-1 flex flex-wrap items-baseline gap-2.5">
            <span className="font-mono text-3xl font-bold tracking-tight text-zinc-900">
              {currentPrice !== null ? `${currSign}${currentPrice.toFixed(2)}` : '—'}
            </span>

            {priceChange !== null && priceChangePct !== null && (
              <div
                className={`inline-flex items-center gap-1 rounded-lg border px-2 py-0.5 text-xs font-semibold ${changeColorClass} ${changeBadgeBg}`}
              >
                <span>{isPositive ? '▲' : '▼'}</span>
                <span>
                  {isPositive ? '+' : ''}
                  {priceChange.toFixed(2)} ({isPositive ? '+' : ''}
                  {priceChangePct.toFixed(2)}%)
                </span>
                <span className="text-[10px] text-zinc-400 font-normal ml-0.5">วันนี้</span>
              </div>
            )}

            {priceAsOf && (
              <span className="text-[11px] text-zinc-400">
                ณ{' '}
                {new Date(priceAsOf).toLocaleTimeString('th-TH', {
                  hour: '2-digit',
                  minute: '2-digit',
                })}
              </span>
            )}

            {/* Next Earnings Date Countdown Badge */}
            {analystContext?.next_earnings_date && (
              <span
                className="inline-flex items-center gap-1 rounded-md bg-blue-50 border border-blue-200 px-2 py-0.5 text-[10px] font-medium text-blue-700"
                title={`Next Earnings Date: ${analystContext.next_earnings_date}`}
              >
                📅 Earnings · {analystContext.days_to_earnings !== null && analystContext.days_to_earnings !== undefined ? `${analystContext.days_to_earnings} days` : analystContext.next_earnings_date}
              </span>
            )}

            {/* Stale Consensus Warning Badge */}
            {config.consensus && analystContext?.data_status === 'stale' && (
              <span
                className="inline-flex items-center gap-1 rounded-md bg-amber-50 border border-amber-200 px-2 py-0.5 text-[10px] font-medium text-amber-700"
                title={`Consensus target price is stale. As of: ${analystContext.source_as_of || 'N/A'}`}
              >
                ⚠️ Street Stale (as of {analystContext.source_as_of?.slice(0, 10)})
              </span>
            )}

            {eventsMetadata?.status === 'partial' && (
              <span
                className="inline-flex items-center gap-1 rounded-md bg-amber-50 border border-amber-200 px-2 py-0.5 text-[10px] font-medium text-amber-700"
                title={`Missing: ${eventsMetadata.missing_sources.join(', ')} | As of: ${eventsMetadata.as_of || 'N/A'}`}
              >
                ⚠️ Partial Events
              </span>
            )}
            {eventsMetadata?.status === 'unavailable' && (
              <span
                className="inline-flex items-center gap-1 rounded-md bg-zinc-100 border border-zinc-200 px-2 py-0.5 text-[10px] font-medium text-zinc-500"
                title="Corporate actions data temporarily unavailable"
              >
                Events unavailable
              </span>
            )}
            {/* Active Indicator Warmup / Burn-In Badge */}
            {config.overlay === 'EMA' && indicatorWarmup?.EMA200?.status === 'partial' && (
              <span
                className="inline-flex items-center gap-1 rounded-md bg-amber-50 border border-amber-200 px-2 py-0.5 text-[10px] font-medium text-amber-700"
                title={`EMA200 Burn-in: ${indicatorWarmup.EMA200.burn_in_bars_remaining} bars remaining until mathematical stability`}
              >
                ⚠️ EMA200 Burn-in (+{indicatorWarmup.EMA200.burn_in_bars_remaining} bars)
              </span>
            )}
            {config.overlay === 'EMA' && indicatorWarmup?.EMA200?.status === 'unavailable' && (
              <span
                className="inline-flex items-center gap-1 rounded-md bg-rose-50 border border-rose-200 px-2 py-0.5 text-[10px] font-medium text-rose-700"
                title="Total available bars insufficient for EMA200 calculation"
              >
                ⚠️ EMA200 Insufficient Data
              </span>
            )}
            {/* DCF Valuation Status & Comparability Warnings */}
            {config.dcf && valuationTargets?.status === 'available' && valuationTargets.comparability_status === 'comparable' && valuationTargets.scenarios.length > 0 && (
              <span
                className="inline-flex items-center gap-1 rounded-md bg-teal-50 border border-teal-200 px-2 py-0.5 text-[10px] font-medium text-teal-800"
                title={`DCF Model: ${valuationTargets.model_version} | ${valuationTargets.as_of_label} | WACC: ${valuationTargets.wacc_pct ? valuationTargets.wacc_pct.toFixed(1) + '%' : 'N/A'}`}
              >
                🎯 {valuationTargets.scenarios[0]?.label}: {currSign}{valuationTargets.scenarios[0]?.target_price.toFixed(2)}
              </span>
            )}
            {config.dcf && valuationTargets?.comparability_status === 'not_comparable' && (
              <span
                className="inline-flex items-center gap-1 rounded-md bg-amber-50 border border-amber-200 px-2 py-0.5 text-[10px] font-medium text-amber-800"
                title={valuationTargets.comparability_reasons.join(' | ')}
              >
                ⚠️ DCF Suppressed (Not comparable)
              </span>
            )}
          </div>
        </div>

        {/* Multi-Timeframe & Dynamic Range Selector Group */}
        <div className="flex flex-wrap items-center gap-2">
          {/* Interval Selector Pills (15m, 1H, 1D, 1W, 1M) */}
          <div
            role="group"
            aria-label="Timeframe interval selector"
            className="flex items-center rounded-xl bg-surface p-1 border border-edge/80"
          >
            {INTERVAL_OPTIONS.map(({ key, label }) => {
              const isSelected = (selectedInterval || '1d') === key
              return (
                <button
                  key={key}
                  type="button"
                  aria-pressed={isSelected}
                  onClick={() => onIntervalChange?.(key)}
                  className={`rounded-lg px-2.5 py-1 text-xs font-semibold transition-all ${
                    isSelected
                      ? 'bg-white text-indigo-700 shadow-sm border border-indigo-200'
                      : 'text-zinc-500 hover:text-zinc-900'
                  }`}
                >
                  {label}
                </button>
              )
            })}
          </div>

          {/* Dynamic Range Selector Pills */}
          <div
            role="group"
            aria-label="Timeframe range selector"
            className="flex items-center rounded-xl bg-surface p-1 border border-edge/80"
          >
            {(INTERVAL_RANGES[selectedInterval || '1d'] || []).map(({ key, label }) => {
              const isSelected = selectedRange === key
              const isAllowed = !allowedRanges || allowedRanges.includes(key)
              return (
                <button
                  key={key}
                  type="button"
                  aria-pressed={isSelected}
                  disabled={!isAllowed}
                  onClick={() => onRangeChange(key)}
                  className={`rounded-lg px-2.5 py-1 text-xs font-semibold transition-all ${
                    isSelected
                      ? 'bg-white text-sky-700 shadow-sm border border-sky-200'
                      : isAllowed
                      ? 'text-zinc-500 hover:text-zinc-900'
                      : 'text-zinc-300 cursor-not-allowed'
                  }`}
                >
                  {label}
                </button>
              )
            })}
          </div>
        </div>
      </div>

      {/* Accessible Indicator & Drawing Toolbar */}
      <div
        role="toolbar"
        aria-label="Technical Indicators and Drawing Toolbar"
        onKeyDown={handleToolbarKeyDown}
        className="mt-3 flex flex-wrap items-center justify-between gap-3 border-b border-edge/40 pb-3"
      >
        <div className="flex flex-wrap items-center gap-3 text-xs">
          {/* Overlay Group */}
          <div role="group" aria-label="Overlay indicator" className="flex items-center gap-1">
            <span className="font-medium text-zinc-400 mr-1">Overlay:</span>
            <div className="flex items-center rounded-lg bg-surface p-0.5 border border-edge/60">
              {OVERLAY_OPTIONS.map(({ key, label }) => {
                const isActive = config.overlay === key
                return (
                  <button
                    key={key}
                    type="button"
                    aria-pressed={isActive}
                    onClick={() => updateConfig({ overlay: key })}
                    className={`rounded-md px-2 py-0.5 text-xs font-medium transition-all ${
                      isActive
                        ? 'bg-white text-sky-700 shadow-sm border border-sky-200'
                        : 'text-zinc-500 hover:text-zinc-800'
                    }`}
                  >
                    {label}
                  </button>
                )
              })}
            </div>
          </div>

          {/* Volume & VWAP Toggle Group */}
          <div role="group" aria-label="Volume and VWAP indicators" className="flex items-center gap-1">
            <button
              type="button"
              aria-pressed={config.volume}
              onClick={() => updateConfig({ volume: !config.volume })}
              className={`rounded-lg border px-2.5 py-0.5 text-xs font-medium transition-all ${
                config.volume
                  ? 'bg-slate-800 text-white border-slate-800 shadow-sm'
                  : 'bg-surface text-zinc-500 border-edge/80 hover:text-zinc-900'
              }`}
            >
              VOL
            </button>
            <button
              type="button"
              aria-pressed={config.vwap}
              disabled={!isIntraday}
              title={!isIntraday ? 'VWAP ใช้งานได้เฉพาะกราฟ Intraday (15m, 1H)' : 'Session VWAP'}
              onClick={() => updateConfig({ vwap: !config.vwap })}
              className={`rounded-lg border px-2.5 py-0.5 text-xs font-medium transition-all ${
                config.vwap && isIntraday
                  ? 'bg-indigo-600 text-white border-indigo-600 shadow-sm'
                  : isIntraday
                  ? 'bg-surface text-zinc-500 border-edge/80 hover:text-zinc-900'
                  : 'bg-surface text-zinc-300 border-edge/40 cursor-not-allowed opacity-60'
              }`}
            >
              VWAP
            </button>
          </div>

          {/* Momentum Group */}
          <div role="group" aria-label="Momentum indicator" className="flex items-center gap-1">
            <span className="font-medium text-zinc-400 mr-1">Momentum:</span>
            <div className="flex items-center rounded-lg bg-surface p-0.5 border border-edge/60">
              {MOMENTUM_OPTIONS.map(({ key, label }) => {
                const isActive = config.momentum === key
                return (
                  <button
                    key={key}
                    type="button"
                    aria-pressed={isActive}
                    onClick={() => updateConfig({ momentum: key })}
                    className={`rounded-md px-2 py-0.5 text-xs font-medium transition-all ${
                      isActive
                        ? 'bg-white text-teal-700 shadow-sm border border-teal-200'
                        : 'text-zinc-500 hover:text-zinc-800'
                    }`}
                  >
                    {label}
                  </button>
                )
              })}
            </div>
          </div>

          {/* Volatility & Risk Group (ATR, 52W, Events, DCF, Street, Insider) */}
          <div
            role="group"
            aria-label="Volatility and events toggles"
            className="flex items-center gap-1.5"
          >
            <span className="font-medium text-zinc-400 mr-0.5">Context:</span>
            <button
              type="button"
              aria-pressed={config.atr}
              onClick={() => updateConfig({ atr: !config.atr })}
              className={`rounded-lg border px-2 py-0.5 text-xs font-medium transition-all ${
                config.atr
                  ? 'bg-amber-600 text-white border-amber-600 shadow-sm'
                  : 'bg-surface text-zinc-500 border-edge/80 hover:text-zinc-900'
              }`}
            >
              ATR(14)
            </button>
            <button
              type="button"
              aria-pressed={config.week52}
              onClick={() => updateConfig({ week52: !config.week52 })}
              className={`rounded-lg border px-2 py-0.5 text-xs font-medium transition-all ${
                config.week52
                  ? 'bg-emerald-600 text-white border-emerald-600 shadow-sm'
                  : 'bg-surface text-zinc-500 border-edge/80 hover:text-zinc-900'
              }`}
            >
              52W
            </button>
            <button
              type="button"
              aria-pressed={config.events}
              onClick={() => updateConfig({ events: !config.events })}
              className={`rounded-lg border px-2 py-0.5 text-xs font-medium transition-all ${
                config.events
                  ? 'bg-indigo-600 text-white border-indigo-600 shadow-sm'
                  : 'bg-surface text-zinc-500 border-edge/80 hover:text-zinc-900'
              }`}
            >
              Events
            </button>
            <button
              type="button"
              aria-pressed={config.dcf}
              onClick={() => updateConfig({ dcf: !config.dcf })}
              className={`rounded-lg border px-2 py-0.5 text-xs font-medium transition-all ${
                config.dcf
                  ? 'bg-teal-700 text-white border-teal-700 shadow-sm'
                  : 'bg-surface text-zinc-500 border-edge/80 hover:text-zinc-900'
              }`}
            >
              DCF
            </button>
            <button
              type="button"
              aria-pressed={config.consensus}
              disabled={!analystContext?.target_mean || analystContext.data_status === 'unavailable'}
              title={
                !analystContext?.target_mean || analystContext.data_status === 'unavailable'
                  ? 'ไม่มีข้อมูล Consensus Target Price'
                  : `Street Consensus: $${analystContext.target_mean.toFixed(2)}`
              }
              onClick={() => updateConfig({ consensus: !config.consensus })}
              className={`rounded-lg border px-2 py-0.5 text-xs font-medium transition-all ${
                config.consensus && analystContext?.target_mean && analystContext.data_status !== 'unavailable'
                  ? 'bg-amber-600 text-white border-amber-600 shadow-sm'
                  : analystContext?.target_mean && analystContext.data_status !== 'unavailable'
                  ? 'bg-surface text-zinc-500 border-edge/80 hover:text-zinc-900'
                  : 'bg-surface text-zinc-300 border-edge/40 cursor-not-allowed opacity-60'
              }`}
            >
              Street
            </button>
            <button
              type="button"
              aria-pressed={config.insider}
              onClick={() => updateConfig({ insider: !config.insider })}
              className={`rounded-lg border px-2 py-0.5 text-xs font-medium transition-all ${
                config.insider
                  ? 'bg-purple-700 text-white border-purple-700 shadow-sm'
                  : 'bg-surface text-zinc-500 border-edge/80 hover:text-zinc-900'
              }`}
            >
              Insider
            </button>
          </div>
        </div>

        {/* Drawing Tools Group */}
        <div
          role="group"
          aria-label="Drawing tools"
          className="flex items-center gap-1 rounded-lg bg-surface p-0.5 border border-edge/60 text-xs"
        >
          <span className="text-[10px] font-semibold text-zinc-400 px-1">Draw:</span>
          <button
            type="button"
            title="Trendline (Line)"
            aria-pressed={activeDrawing === 'straightLine'}
            onClick={() => handleDrawingToolClick('straightLine')}
            className={`rounded px-1.5 py-0.5 transition-all ${
              activeDrawing === 'straightLine'
                ? 'bg-sky-600 text-white shadow-sm font-semibold'
                : 'text-zinc-600 hover:text-zinc-900'
            }`}
          >
            📏 Line
          </button>
          <button
            type="button"
            title="Horizontal Line"
            aria-pressed={activeDrawing === 'horizontalStraightLine'}
            onClick={() => handleDrawingToolClick('horizontalStraightLine')}
            className={`rounded px-1.5 py-0.5 transition-all ${
              activeDrawing === 'horizontalStraightLine'
                ? 'bg-sky-600 text-white shadow-sm font-semibold'
                : 'text-zinc-600 hover:text-zinc-900'
            }`}
          >
            〰️ Horiz
          </button>
          <button
            type="button"
            title="Fibonacci Retracement"
            aria-pressed={activeDrawing === 'fibonacciLine'}
            onClick={() => handleDrawingToolClick('fibonacciLine')}
            className={`rounded px-1.5 py-0.5 transition-all ${
              activeDrawing === 'fibonacciLine'
                ? 'bg-sky-600 text-white shadow-sm font-semibold'
                : 'text-zinc-600 hover:text-zinc-900'
            }`}
          >
            🔶 Fib
          </button>
          <button
            type="button"
            title="Clear All User Drawings"
            onClick={handleClearDrawings}
            className="rounded px-1.5 py-0.5 text-zinc-400 hover:text-rose-600 hover:bg-rose-50 transition-all"
          >
            🗑️
          </button>
        </div>
      </div>

      {/* Quality Axis Warning Banner */}
      {showQualityWarning && (
        <div
          role="alert"
          data-testid="quality-warning-banner"
          className="mt-3 flex items-center gap-2 rounded-xl border border-amber-200 bg-amber-50 px-3.5 py-2 text-xs font-medium text-amber-800 animate-fade-in"
        >
          <span>⚠️</span>
          <span>
            ประวัติก่อนแสดงผลมี {availableWarmupBars} แท่ง (EMA 200 อาจยังไม่เสถียร)
          </span>
        </div>
      )}

      {/* Chart Viewport Container */}
      <div
        className={`relative mt-4 w-full transition-all duration-200 ${containerHeightClass} ${activeDrawing ? 'cursor-crosshair' : ''}`}
      >
        <div
          ref={chartContainerRef}
          className="h-full w-full rounded-xl overflow-hidden"
          data-testid="kline-chart-viewport"
        />

        {/* Accessible Floating Corporate Action Tooltip with EPS Surprise */}
        {hoveredEvent && (
          <div
            data-testid="corporate-event-tooltip"
            style={{
              position: 'absolute',
              left: `${Math.max(10, Math.min(hoveredEvent.pos.x - 100, (chartContainerRef.current?.clientWidth || 500) - 240))}px`,
              top: `${Math.max(10, hoveredEvent.pos.y - 75)}px`,
              pointerEvents: 'none',
              zIndex: 30,
            }}
            className="rounded-lg bg-zinc-900/95 backdrop-blur-sm px-3 py-2 text-xs text-white shadow-xl border border-zinc-700 max-w-xs animate-fade-in"
          >
            <div className="font-semibold text-[11px] text-zinc-300">
              {hoveredEvent.event.event_type.toUpperCase()} • {hoveredEvent.event.date_str}
            </div>
            <div className="mt-0.5 text-[11px] leading-relaxed text-zinc-100">
              {hoveredEvent.event.tooltip}
              {(() => {
                const ev = hoveredEvent.event
                if (ev.event_type === 'earnings' && typeof ev.eps_actual === 'number' && typeof ev.eps_estimate === 'number' && ev.eps_estimate !== 0) {
                  const surp = ((ev.eps_actual - ev.eps_estimate) / Math.abs(ev.eps_estimate)) * 100
                  return (
                    <div className="mt-1 font-medium text-amber-400">
                      Surprise: {surp >= 0 ? '+' : ''}{surp.toFixed(1)}%
                    </div>
                  )
                }
                return null
              })()}
            </div>
          </div>
        )}

        {/* Accessible Floating Insider Popover (Hover Preview Mode) */}
        {hoveredInsider && !pinnedInsider && (
          <div
            data-testid="insider-event-preview-popover"
            style={{
              position: 'absolute',
              left: `${Math.max(10, Math.min(hoveredInsider.pos.x - 120, (chartContainerRef.current?.clientWidth || 500) - 260))}px`,
              top: `${Math.max(10, hoveredInsider.pos.y - 80)}px`,
              pointerEvents: 'none',
              zIndex: 30,
            }}
            className="rounded-lg bg-zinc-900/95 backdrop-blur-sm px-3.5 py-2.5 text-xs text-white shadow-xl border border-zinc-700 max-w-xs animate-fade-in"
          >
            <div className="flex items-center justify-between gap-2">
              <span className="font-semibold text-[11px] text-purple-300">
                {hoveredInsider.filing.label}
              </span>
              <span className="text-[10px] text-zinc-400">Click to pin 📌</span>
            </div>
            <div className="mt-1 font-medium text-[11px] text-zinc-100">
              {hoveredInsider.filing.insider_name}
            </div>
            <div className="text-[10px] text-zinc-300">
              {hoveredInsider.filing.shares.toLocaleString()} หุ้น @ ${hoveredInsider.filing.price.toFixed(2)}
            </div>
          </div>
        )}

        {/* Interactive Pinned Insider Popover (Pinned Mode with Link and Close) */}
        {pinnedInsider && (
          <div
            data-testid="insider-event-pinned-popover"
            style={{
              position: 'absolute',
              right: '16px',
              top: '16px',
              pointerEvents: 'auto',
              zIndex: 40,
            }}
            className="rounded-xl bg-white/95 backdrop-blur-md p-4 text-xs text-zinc-900 shadow-2xl border border-purple-200 max-w-sm animate-fade-in"
          >
            <div className="flex items-start justify-between gap-3 border-b border-zinc-100 pb-2">
              <div>
                <span className="inline-flex items-center gap-1 rounded bg-purple-50 px-1.5 py-0.5 text-[10px] font-bold text-purple-700 border border-purple-200">
                  {pinnedInsider.label}
                </span>
                <h5 className="mt-1 font-bold text-zinc-900 text-sm">
                  {pinnedInsider.insider_name}
                </h5>
                {pinnedInsider.officer_title && (
                  <p className="text-[11px] text-zinc-500">{pinnedInsider.officer_title}</p>
                )}
              </div>
              <button
                type="button"
                onClick={() => setPinnedInsider(null)}
                className="rounded-lg p-1 text-zinc-400 hover:bg-zinc-100 hover:text-zinc-700 transition-colors"
                title="ปิด"
              >
                ✕
              </button>
            </div>

            <div className="mt-3 space-y-1.5 text-[11px]">
              <div className="flex justify-between text-zinc-600">
                <span>จำนวนหุ้นรวม:</span>
                <span className="font-semibold text-zinc-900">{pinnedInsider.shares.toLocaleString()} หุ้น</span>
              </div>
              <div className="flex justify-between text-zinc-600">
                <span>ราคาเฉลี่ย:</span>
                <span className="font-semibold text-zinc-900">${pinnedInsider.price.toFixed(2)}</span>
              </div>
              {pinnedInsider.all_filers && pinnedInsider.all_filers.length > 0 && (
                <div className="mt-2 pt-2 border-t border-zinc-100">
                  <span className="text-[10px] font-semibold text-zinc-400">ผู้รายงานในรอบนี้:</span>
                  <ul className="mt-1 space-y-1 text-[10px] text-zinc-600">
                    {pinnedInsider.all_filers.map((f, i) => (
                      <li key={i} className="flex justify-between">
                        <span className="truncate max-w-[150px]">{f.name}</span>
                        <span>{f.shares.toLocaleString()} หุ้น</span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>

            <div className="mt-3 pt-2.5 border-t border-zinc-100 flex items-center justify-between">
              <a
                href={pinnedInsider.filing_url}
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-1 text-[11px] font-semibold text-purple-600 hover:text-purple-800 hover:underline"
              >
                <span>ดูเอกสาร SEC EDGAR</span>
                <span>↗</span>
              </a>
              <button
                type="button"
                onClick={() => setPinnedInsider(null)}
                className="rounded-lg bg-zinc-100 px-2.5 py-1 text-[10px] font-semibold text-zinc-600 hover:bg-zinc-200"
              >
                เสร็จสิ้น
              </button>
            </div>
          </div>
        )}

        {isLoading && (
          <div className="absolute inset-0 flex items-center justify-center bg-white/60 backdrop-blur-[1px] transition-opacity">
            <div className="flex items-center gap-2 rounded-xl bg-white px-3 py-1.5 shadow-md border border-edge text-xs font-medium text-zinc-600">
              <svg
                className="h-4 w-4 animate-spin text-sky-600"
                xmlns="http://www.w3.org/2000/svg"
                fill="none"
                viewBox="0 0 24 24"
              >
                <circle
                  className="opacity-25"
                  cx="12"
                  cy="12"
                  r="10"
                  stroke="currentColor"
                  strokeWidth="4"
                />
                <path
                  className="opacity-75"
                  fill="currentColor"
                  d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                />
              </svg>
              <span>กำลังโหลดข้อมูลกราฟ...</span>
            </div>
          </div>
        )}
      </div>

      {/* Static Context & Navigation Hint */}
      <div className="mt-2.5 flex flex-wrap items-center justify-between gap-2 text-[11px] text-zinc-400 px-1">
        <span>
          ℹ️ กราฟแสดงราคาปรับสิทธิ (Adjusted Basis) • เหตุการณ์ (E, XD, S) แสดงบริบทเหตุการณ์สำคัญ
        </span>
        {activeDrawing && (
          <span className="text-sky-600 font-medium animate-pulse">
            โหมดวาด: คลิก 2 จุดเพื่อวางเส้น (กด Escape เพื่อยกเลิก)
          </span>
        )}
      </div>
    </div>
  )
}
