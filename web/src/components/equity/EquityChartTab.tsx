import React, { useState, useEffect, useCallback } from 'react'
import { api } from '../../api/client'
import type {
  OHLCVResponseDTO,
  ChartInterval,
  ValuationTargetsDTO,
  InsiderFilingDTO,
  AnalystContextDTO,
} from '../../api/types'
import { EquityChart, INTERVAL_RANGES } from './EquityChart'
import { SupportResistanceTable } from './SupportResistanceTable'

interface EquityChartTabProps {
  ticker: string
  companyName?: string
  market?: 'TH' | 'US'
  currentPrice?: number | null
}

const DEFAULT_RANGE_FOR_INTERVAL: Record<ChartInterval, string> = {
  '15m': '5d',
  '1h': '3mo',
  '1d': '6mo',
  '1wk': '1y',
  '1mo': '5y',
}

export const EquityChartTab: React.FC<EquityChartTabProps> = ({
  ticker,
  companyName,
  market = 'US',
  currentPrice: initialCurrentPrice,
}) => {
  const [selectedInterval, setSelectedInterval] = useState<ChartInterval>('1d')
  const [selectedRange, setSelectedRange] = useState<string>('6mo')
  const [data, setData] = useState<OHLCVResponseDTO | null>(null)
  const [valuationTargets, setValuationTargets] = useState<ValuationTargetsDTO | null>(null)
  const [insiderFilings, setInsiderFilings] = useState<InsiderFilingDTO[]>([])
  const [analystContext, setAnalystContext] = useState<AnalystContextDTO | null>(null)
  const [status, setStatus] = useState<'loading' | 'success' | 'error'>('loading')
  const [errorMessage, setErrorMessage] = useState<string | null>(null)

  const fetchData = useCallback(
    (range: string, interval: ChartInterval, signal?: AbortSignal) => {
      setStatus('loading')
      setErrorMessage(null)

      api
        .getEquityOHLCV(ticker, range, interval, signal)
        .then((res) => {
          setData(res)
          setStatus('success')
        })
        .catch((err) => {
          if (err.name === 'AbortError' || err.message?.includes('aborted')) {
            return
          }
          setStatus('error')
          setErrorMessage(err.message || 'ไม่สามารถโหลดข้อมูลกราฟราคาได้')
        })
    },
    [ticker]
  )

  useEffect(() => {
    const controller = new AbortController()
    fetchData(selectedRange, selectedInterval, controller.signal)

    return () => {
      controller.abort()
    }
  }, [ticker, selectedRange, selectedInterval, fetchData])

  useEffect(() => {
    const controller = new AbortController()
    api
      .getValuationTargets(ticker, controller.signal)
      .then((res) => {
        setValuationTargets(res)
      })
      .catch(() => {
        setValuationTargets(null)
      })

    return () => {
      controller.abort()
    }
  }, [ticker])

  useEffect(() => {
    const controller = new AbortController()
    api
      .getInsiderFilings(ticker, selectedRange, selectedInterval, controller.signal)
      .then((res) => {
        setInsiderFilings(res?.filings || [])
      })
      .catch(() => {
        setInsiderFilings([])
      })

    return () => {
      controller.abort()
    }
  }, [ticker, selectedRange, selectedInterval])

  useEffect(() => {
    const controller = new AbortController()
    api
      .getAnalystContext(ticker, controller.signal)
      .then((res) => {
        setAnalystContext(res)
      })
      .catch(() => {
        setAnalystContext(null)
      })

    return () => {
      controller.abort()
    }
  }, [ticker])

  const handleIntervalChange = (newInterval: ChartInterval) => {
    if (newInterval === selectedInterval) return
    setSelectedInterval(newInterval)
    const validRanges = INTERVAL_RANGES[newInterval]?.map((r) => r.key) || []
    if (!validRanges.includes(selectedRange)) {
      setSelectedRange(DEFAULT_RANGE_FOR_INTERVAL[newInterval] || '1mo')
    }
  }

  const handleRangeChange = (newRange: string) => {
    if (newRange === selectedRange) return
    setSelectedRange(newRange)
  }

  if (status === 'error' && !data) {
    return (
      <div className="flex flex-col items-center justify-center rounded-2xl border border-rose-200 bg-rose-50/70 p-8 text-center">
        <span className="text-3xl mb-2">⚠️</span>
        <h4 className="text-base font-bold text-rose-900">เกิดข้อผิดพลาดในการโหลดกราฟ</h4>
        <p className="text-xs text-rose-600 mt-1 max-w-md">{errorMessage}</p>
        <button
          type="button"
          onClick={() => fetchData(selectedRange, selectedInterval)}
          className="mt-4 rounded-xl bg-rose-600 px-4 py-2 text-xs font-semibold text-white shadow-sm hover:bg-rose-700 transition-colors"
        >
          ลองใหม่อีกครั้ง
        </button>
      </div>
    )
  }

  const currency = data?.currency ?? (market === 'TH' ? 'THB' : 'USD')
  const currentPrice = data?.current_price ?? initialCurrentPrice ?? null

  return (
    <div className="space-y-6 animate-fade-in">
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-12 items-start">
        {/* Left Column: Interactive Candlestick Chart (7/12) */}
        <div className="lg:col-span-7">
          <EquityChart
            ticker={ticker}
            companyName={companyName}
            currentPrice={currentPrice}
            priceChange={data?.price_change ?? null}
            priceChangePct={data?.price_change_pct ?? null}
            priceAsOf={data?.price_as_of ?? null}
            currency={currency}
            candles={data?.candles ?? []}
            selectedInterval={selectedInterval}
            onIntervalChange={handleIntervalChange}
            selectedRange={selectedRange}
            onRangeChange={handleRangeChange}
            allowedRanges={data?.allowed_ranges}
            isLoading={status === 'loading'}
            displayStartTimestamp={data?.display_start_timestamp ?? null}
            availableWarmupBars={data?.available_warmup_bars ?? 0}
            requiredWarmupBars={data?.required_warmup_bars ?? 200}
            warmupStatus={data?.warmup_status ?? 'unknown'}
            indicatorWarmup={data?.indicator_warmup}
            events={data?.events ?? []}
            eventsMetadata={data?.events_metadata ?? null}
            week52High={data?.week52_high ?? null}
            week52Low={data?.week52_low ?? null}
            week52CoverageDays={data?.week52_coverage_calendar_days ?? 0}
            valuationTargets={valuationTargets}
            insiderFilings={insiderFilings}
            market={market}
            analystContext={analystContext}
          />
        </div>

        {/* Right Column: Support / Resistance Pivot Table (5/12) */}
        <div className="lg:col-span-5">
          <SupportResistanceTable
            pivotLevels={data?.pivot_levels ?? null}
            pivotPeriod={data?.pivot_period ?? null}
            pivotAsOf={data?.pivot_as_of ?? null}
            currency={currency}
            currentPrice={currentPrice}
          />
        </div>
      </div>
    </div>
  )
}



