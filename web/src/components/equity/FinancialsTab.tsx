import { useState, useEffect, useMemo, useCallback } from 'react'
import { api, ApiError } from '../../api/client'
import type {
  FinancialStatementsDTO,
  FinancialStatementCategoryDTO,
  FinancialRatioPointDTO,
} from '../../api/types'
import { FinancialSummaryChart } from './FinancialSummaryChart'

interface FinancialsTabProps {
  ticker: string
  market?: 'US' | 'TH'
  currency?: string
}

type StatementType = 'income' | 'balance_sheet' | 'cash_flow' | 'ratios'
type PeriodFrequency = 'quarterly' | 'annual'
type UnitScale = 'auto' | 'millions' | 'billions' | 'raw'

function formatRelativeTime(isoString?: string | null): { formatted: string; relative: string } {
  if (!isoString) return { formatted: 'ไม่ระบุเวลา', relative: '' }
  try {
    const d = new Date(isoString)
    if (isNaN(d.getTime())) return { formatted: isoString, relative: '' }

    const formatted = d.toLocaleString('th-TH', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
    })

    const diffSeconds = Math.max(0, Math.floor((Date.now() - d.getTime()) / 1000))
    let relative = ''
    if (diffSeconds < 60) {
      relative = 'เมื่อสักครู่'
    } else if (diffSeconds < 3600) {
      const mins = Math.floor(diffSeconds / 60)
      relative = `${mins} นาทีที่แล้ว`
    } else if (diffSeconds < 86400) {
      const hours = Math.floor(diffSeconds / 3600)
      relative = `${hours} ชั่วโมงที่แล้ว`
    } else {
      const days = Math.floor(diffSeconds / 86400)
      relative = `${days} วันที่แล้ว`
    }

    return { formatted, relative }
  } catch {
    return { formatted: isoString, relative: '' }
  }
}

export function FinancialsTab({ ticker, market = 'US', currency }: FinancialsTabProps) {
  const [data, setData] = useState<FinancialStatementsDTO | null>(null)
  const [loading, setLoading] = useState<boolean>(true)
  const [error, setError] = useState<string | null>(null)
  const [refreshing, setRefreshing] = useState<boolean>(false)
  const [elapsedSeconds, setElapsedSeconds] = useState<number>(0)

  const [statementType, setStatementType] = useState<StatementType>('income')
  const [periodFrequency, setPeriodFrequency] = useState<PeriodFrequency>('quarterly')
  const [unitScale, setUnitScale] = useState<UnitScale>('auto')
  const [showExpandedItems, setShowExpandedItems] = useState<boolean>(true)

  useEffect(() => {
    let timer: ReturnType<typeof setInterval> | null = null
    if (loading || refreshing) {
      setElapsedSeconds(0)
      timer = setInterval(() => {
        setElapsedSeconds((s) => s + 1)
      }, 1000)
    } else {
      setElapsedSeconds(0)
    }
    return () => {
      if (timer) clearInterval(timer)
    }
  }, [loading, refreshing])

  const EXPANDED_LINE_ITEMS = [
    'long_term_investments',
    'deferred_contract_costs',
    'deferred_tax_assets',
    'goodwill',
    'other_intangible_assets',
    'deferred_revenue_current',
    'deferred_revenue_noncurrent',
    'income_tax_liabilities',
    'noncontrolling_interests',
    'reported_free_cash_flow',
    'free_cash_flow_adjustments',
  ]

  // Fetch financial statements
  const fetchData = useCallback(async (forceRefresh = false) => {
    if (forceRefresh) {
      setRefreshing(true)
    } else {
      setLoading(true)
    }
    setError(null)

    const controller = new AbortController()
    try {
      const res = await api.getEquityFinancials(ticker, market, forceRefresh, controller.signal)
      setData(res)
    } catch (err: any) {
      if (err.name !== 'AbortError') {
        setError(err instanceof ApiError ? err.message : (err?.message || 'Failed to load financial statements'))
      }
    } finally {
      setLoading(false)
      setRefreshing(false)
    }
    return () => controller.abort()
  }, [ticker, market])

  useEffect(() => {
    fetchData(false)
  }, [fetchData])

  const currencySymbol = useMemo(() => {
    if (currency) return currency === 'THB' ? '฿' : '$'
    if (data?.currency) return data.currency === 'THB' ? '฿' : '$'
    return market === 'TH' ? '฿' : '$'
  }, [currency, data?.currency, market])

  const syncInfo = useMemo(() => {
    return formatRelativeTime(data?.synced_at)
  }, [data?.synced_at])

  // Active Category or Ratios
  const activeCategory: FinancialStatementCategoryDTO | undefined = useMemo(() => {
    if (!data) return undefined
    const list = periodFrequency === 'quarterly' ? data.quarterly : data.annual
    return list.find((c) => c.statement_type === statementType)
  }, [data, periodFrequency, statementType])

  const activeRatios: FinancialRatioPointDTO[] = useMemo(() => {
    if (!data) return []
    return periodFrequency === 'quarterly' ? data.ratios_quarterly : data.ratios_annual
  }, [data, periodFrequency])

  const activeChartPoints = useMemo(() => {
    if (!data) return []
    return periodFrequency === 'quarterly' ? data.summary_chart_quarterly : data.summary_chart_annual
  }, [data, periodFrequency])

  // Auto Scaling Divisor across all items in active statement
  const autoDivisor = useMemo(() => {
    if (!activeCategory || activeCategory.periods.length === 0) return 1_000_000
    let maxVal = 0
    activeCategory.periods.forEach((p) => {
      Object.values(p.items).forEach((c) => {
        if (c.value && Math.abs(c.value) > maxVal) {
          maxVal = Math.abs(c.value)
        }
      })
    })
    if (maxVal >= 1_000_000_000) return 1_000_000_000
    if (maxVal >= 1_000_000) return 1_000_000
    return 1
  }, [activeCategory])

  const scaleConfig = useMemo(() => {
    if (unitScale === 'billions') return { divisor: 1_000_000_000, suffix: 'B' }
    if (unitScale === 'millions') return { divisor: 1_000_000, suffix: 'M' }
    if (unitScale === 'raw') return { divisor: 1, suffix: '' }
    // Auto
    if (autoDivisor === 1_000_000_000) return { divisor: 1_000_000_000, suffix: 'B' }
    if (autoDivisor === 1_000_000) return { divisor: 1_000_000, suffix: 'M' }
    return { divisor: 1, suffix: '' }
  }, [unitScale, autoDivisor])

  // Cell number formatting helper
  const formatCellValue = (
    val: number | null | undefined,
    unitType: 'currency' | 'per_share' | 'shares' | 'ratio' | 'percentage'
  ) => {
    if (val === null || val === undefined) return '-'

    if (unitType === 'per_share') {
      return `${currencySymbol}${val.toFixed(2)}`
    }
    if (unitType === 'ratio') {
      return `${val.toFixed(2)}x`
    }
    if (unitType === 'percentage') {
      return `${val.toFixed(2)}%`
    }
    if (unitType === 'shares') {
      if (Math.abs(val) >= 1_000_000_000) return `${(val / 1_000_000_000).toFixed(2)}B`
      if (Math.abs(val) >= 1_000_000) return `${(val / 1_000_000).toFixed(2)}M`
      return val.toLocaleString()
    }

    // Currency values scale according to scaleConfig
    const scaled = val / scaleConfig.divisor
    const absScaled = Math.abs(scaled)
    const decimals = absScaled >= 100 ? 1 : 2
    const formatted = scaled.toLocaleString(undefined, {
      minimumFractionDigits: decimals,
      maximumFractionDigits: decimals,
    })
    return `${currencySymbol}${formatted}${scaleConfig.suffix ? ` ${scaleConfig.suffix}` : ''}`
  }

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Top Controls Toolbar */}
      <div className="bg-panel border border-edge/80 rounded-2xl p-4 shadow-sm flex flex-col md:flex-row md:items-center justify-between gap-4">
        {/* Statement Selector Pills */}
        <div className="flex flex-wrap items-center gap-1.5 p-1 bg-surface rounded-xl border border-edge/60">
          <button
            onClick={() => setStatementType('income')}
            className={`px-3.5 py-1.5 rounded-lg text-xs font-semibold transition-all ${
              statementType === 'income'
                ? 'bg-panel text-primary shadow-sm border border-edge'
                : 'text-muted hover:text-primary'
            }`}
          >
            Income Statement
          </button>
          <button
            onClick={() => setStatementType('balance_sheet')}
            className={`px-3.5 py-1.5 rounded-lg text-xs font-semibold transition-all ${
              statementType === 'balance_sheet'
                ? 'bg-panel text-primary shadow-sm border border-edge'
                : 'text-muted hover:text-primary'
            }`}
          >
            Balance Sheet
          </button>
          <button
            onClick={() => setStatementType('cash_flow')}
            className={`px-3.5 py-1.5 rounded-lg text-xs font-semibold transition-all ${
              statementType === 'cash_flow'
                ? 'bg-panel text-primary shadow-sm border border-edge'
                : 'text-muted hover:text-primary'
            }`}
          >
            Cash Flow
          </button>
          <button
            onClick={() => setStatementType('ratios')}
            className={`px-3.5 py-1.5 rounded-lg text-xs font-semibold transition-all ${
              statementType === 'ratios'
                ? 'bg-panel text-primary shadow-sm border border-edge'
                : 'text-muted hover:text-primary'
            }`}
          >
            Key Ratios
          </button>
        </div>

        {/* Right Controls: Data Quality Badge, Period Frequency, Unit Scale & Refresh */}
        <div className="flex flex-wrap items-center gap-3">
          {/* Data Quality & Coverage Status Badges */}
          {data && (
            <div className="flex flex-wrap items-center gap-1.5">
              {/* Coverage Ratio Pill */}
              {(data.core_coverage_pct !== undefined || data.expanded_coverage_pct !== undefined) && (
                <span
                  className="inline-flex items-center gap-1 px-2.5 py-1 rounded-lg bg-surface text-muted border border-edge text-[11px] font-mono font-medium"
                  title={`Core Coverage: ${data.core_coverage_pct ?? 100}% | Expanded Coverage: ${data.expanded_coverage_pct ?? 0}%`}
                >
                  <span>Core: {data.core_coverage_pct ?? 100}%</span>
                  <span className="text-edge">·</span>
                  <span>Expanded: {data.expanded_coverage_pct ?? 0}%</span>
                </span>
              )}

              {/* Core Badge */}
              {data.core_coverage_status === 'complete' || (data.data_status === 'ok' && data.coverage_status === 'complete') ? (
                <span
                  className="inline-flex items-center gap-1 px-2.5 py-1 rounded-lg bg-emerald-500/10 text-emerald-500 border border-emerald-500/20 text-[11px] font-semibold cursor-help"
                  title={`SEC Core Data Verified (${data.quarterly[0]?.periods.length || 0}Q / ${data.annual[0]?.periods.length || 0}Y periods, 0 core missing items, 0 accounting gaps)`}
                >
                  <span>✓</span>
                  <span>{data.provider === 'edgartools' ? 'SEC Core Data Verified' : 'Core Verified'}</span>
                </span>
              ) : data.data_status === 'stale' ? (
                <span
                  className="inline-flex items-center gap-1 px-2.5 py-1 rounded-lg bg-orange-500/10 text-orange-500 border border-orange-500/20 text-[11px] font-semibold cursor-help"
                  title={`Stale Cache (Synced: ${data.synced_at || 'N/A'}) — Live refresh failed`}
                >
                  <span>⏳</span>
                  <span>Stale Cache</span>
                </span>
              ) : data.data_status === 'empty' ? (
                <span
                  className="inline-flex items-center gap-1 px-2.5 py-1 rounded-lg bg-rose-500/10 text-rose-500 border border-rose-500/20 text-[11px] font-semibold"
                  title="Financial statements refresh required"
                >
                  <span>❌</span>
                  <span>Refresh Required</span>
                </span>
              ) : (
                <span
                  className="inline-flex items-center gap-1 px-2.5 py-1 rounded-lg bg-amber-500/10 text-amber-500 border border-amber-500/20 text-[11px] font-semibold cursor-help"
                  title={
                    data.missing_required_items && data.missing_required_items.length > 0
                      ? `Core Partial — Missing (${data.missing_required_items.length}): ${data.missing_required_items.slice(0, 5).join(', ')}`
                      : 'Core Partial'
                  }
                >
                  <span>⚠️</span>
                  <span>Core Partial</span>
                </span>
              )}

              {/* Expanded Data Badge */}
              {data.expanded_coverage_status === 'complete' ? (
                <span
                  className="inline-flex items-center gap-1 px-2.5 py-1 rounded-lg bg-emerald-500/10 text-emerald-500 border border-emerald-500/20 text-[11px] font-semibold cursor-help"
                  title="Expanded Balance Sheet items and Company-Reported Non-GAAP FCF complete"
                >
                  <span>★</span>
                  <span>Expanded Data Verified</span>
                </span>
              ) : data.expanded_coverage_status === 'partial' ? (
                <span
                  className="inline-flex items-center gap-1 px-2.5 py-1 rounded-lg bg-indigo-500/10 text-indigo-400 border border-indigo-500/20 text-[11px] font-semibold cursor-help"
                  title={
                    data.missing_expanded_items && data.missing_expanded_items.length > 0
                      ? `Expanded Partial (${data.missing_expanded_items.length} items omitted in filings): ${data.missing_expanded_items.slice(0, 4).join(', ')}`
                      : 'Expanded Partial'
                  }
                >
                  <span>★</span>
                  <span>Expanded Data Partial</span>
                </span>
              ) : null}
            </div>
          )}

          {/* Period Frequency Selector */}
          <div className="flex items-center p-1 bg-surface rounded-xl border border-edge/60 text-xs">
            <button
              onClick={() => setPeriodFrequency('quarterly')}
              className={`px-3 py-1 rounded-lg font-medium transition-all ${
                periodFrequency === 'quarterly'
                  ? 'bg-panel text-primary shadow-sm border border-edge'
                  : 'text-muted hover:text-primary'
              }`}
            >
              Quarterly (8Q)
            </button>
            <button
              onClick={() => setPeriodFrequency('annual')}
              className={`px-3 py-1 rounded-lg font-medium transition-all ${
                periodFrequency === 'annual'
                  ? 'bg-panel text-primary shadow-sm border border-edge'
                  : 'text-muted hover:text-primary'
              }`}
            >
              Annual (5Y)
            </button>
          </div>

          {/* Unit Scale Selector */}
          {statementType !== 'ratios' && (
            <div className="flex items-center gap-1.5 text-xs">
              <span className="text-muted font-medium">Unit:</span>
              <select
                value={unitScale}
                onChange={(e) => setUnitScale(e.target.value as UnitScale)}
                className="bg-surface border border-edge rounded-xl px-3 py-1.5 text-xs text-primary font-medium focus:outline-none focus:ring-1 focus:ring-accent"
              >
                <option value="auto">
                  Auto ({scaleConfig.suffix ? `${currencySymbol}${scaleConfig.suffix}` : currencySymbol})
                </option>
                <option value="millions">{currencySymbol}M (Millions)</option>
                <option value="billions">{currencySymbol}B (Billions)</option>
                <option value="raw">Raw (Full)</option>
              </select>
            </div>
          )}

          {/* Refresh Button */}
          <button
            onClick={() => fetchData(true)}
            disabled={refreshing || loading}
            className="flex items-center gap-1.5 px-3.5 py-1.5 rounded-xl border border-edge bg-surface hover:bg-panel text-xs font-semibold text-primary transition-all shadow-sm disabled:opacity-50"
            title="Force refresh live financial statements"
          >
            <span className={refreshing ? 'animate-spin' : ''}>🔄</span>
            <span>{refreshing ? `Refreshing (${elapsedSeconds}s)...` : 'Refresh'}</span>
          </button>
        </div>
      </div>

      {/* Top Status Bar: Synced At & Provider details */}
      {data && (
        <div className="flex flex-wrap items-center justify-between gap-3 px-4 py-2.5 rounded-xl bg-surface/50 border border-edge/60 text-xs">
          <div className="flex flex-wrap items-center gap-2 text-muted">
            <span className="flex h-2 w-2 relative">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
              <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
            </span>
            <span className="font-medium text-zinc-300">
              แหล่งข้อมูล:{' '}
              <span className="text-primary font-semibold">
                {data.provider === 'edgartools' ? 'SEC EDGAR (10-K / 10-Q / 8-K)' : (data.provider || 'Yahoo Finance')}
              </span>
            </span>
            <span className="text-edge">·</span>
            <span>
              ดึงข้อมูลเมื่อ:{' '}
              <span className="text-primary font-semibold font-mono" title={data.synced_at || undefined}>
                {syncInfo.formatted}
              </span>
              {syncInfo.relative && <span className="text-zinc-400 ml-1">({syncInfo.relative})</span>}
            </span>
          </div>

          <div className="flex items-center gap-2 text-[11px] text-zinc-400 font-mono">
            <span className="px-2 py-0.5 rounded-md bg-panel border border-edge/60">
              {data.market === 'US' ? 'US Equity' : 'Thai SET'}
            </span>
            <span className="px-2 py-0.5 rounded-md bg-panel border border-edge/60">
              {data.provider === 'edgartools' ? 'TTL: 7 Days' : 'TTL: 1 Hour'}
            </span>
          </div>
        </div>
      )}

      {/* Prominent Live Refreshing Banner */}
      {refreshing && (
        <div className="relative overflow-hidden rounded-2xl border border-accent/40 bg-gradient-to-r from-accent/15 via-panel to-accent/10 p-5 shadow-lg animate-fade-in">
          <div className="flex flex-col sm:flex-row items-center gap-4">
            <div className="relative flex items-center justify-center">
              <div className="w-12 h-12 rounded-2xl bg-accent/20 flex items-center justify-center text-accent animate-spin text-xl">
                🔄
              </div>
            </div>
            <div className="flex-1 text-center sm:text-left space-y-1">
              <div className="flex flex-wrap items-center justify-center sm:justify-start gap-2">
                <h4 className="text-sm font-bold text-primary">กำลังดึงข้อมูลงบการเงินล่าสุดจาก SEC EDGAR...</h4>
                <span className="px-2 py-0.5 rounded-full bg-accent/20 text-accent font-mono text-[11px] font-semibold">
                  {elapsedSeconds}s
                </span>
              </div>
              <p className="text-xs text-muted">
                กำลังดาวน์โหลดและวิเคราะห์งบ 10-K, 10-Q ย้อนหลัง 8 ไตรมาส, ตรวจสอบ XBRL Tags, และสแกน 8-K Press Releases เพื่อ Reconcile Non-GAAP FCF
              </p>
            </div>
            <div className="flex items-center gap-1.5 text-xs text-accent font-medium bg-accent/10 px-3 py-1.5 rounded-xl border border-accent/20">
              <span className="inline-block w-2 h-2 rounded-full bg-accent animate-ping" />
              <span>Live Querying</span>
            </div>
          </div>
          {/* Pulsing Progress Bar */}
          <div className="mt-3.5 w-full bg-surface/80 h-1.5 rounded-full overflow-hidden">
            <div className="h-full bg-accent animate-pulse w-full" />
          </div>
        </div>
      )}

      {/* Warnings & Status Banners */}
      {data?.warnings && data.warnings.length > 0 && (
        <div className="space-y-2">
          {data.warnings.map((warn, idx) => {
            const isYf = warn.toLowerCase().includes('yfinance')
            const isStale = warn.toLowerCase().includes('stale')
            return (
              <div
                key={idx}
                className={`p-3.5 rounded-2xl border text-xs flex items-center gap-2.5 shadow-sm ${
                  isYf
                    ? 'bg-blue-500/10 border-blue-500/30 text-blue-400'
                    : isStale
                    ? 'bg-amber-500/10 border-amber-500/30 text-amber-400'
                    : 'bg-surface border-edge text-muted'
                }`}
              >
                <span className="text-sm">{isYf ? 'ℹ️' : isStale ? '⚠️' : '📢'}</span>
                <span className="font-medium">{warn}</span>
              </div>
            )
          })}
        </div>
      )}

      {/* Loading Skeleton */}
      {loading && !refreshing && (
        <div className="space-y-4">
          <div className="p-8 bg-panel border border-edge/80 rounded-2xl text-center space-y-3 shadow-sm animate-pulse">
            <div className="w-12 h-12 mx-auto rounded-2xl bg-surface border border-edge flex items-center justify-center text-xl text-muted animate-spin">
              ⏳
            </div>
            <div className="space-y-1">
              <h4 className="text-sm font-bold text-primary">กำลังโหลดข้อมูลงบการเงิน...</h4>
              <p className="text-xs text-muted">
                กำลังค้นหาข้อมูลจาก Cache หรือเชื่อมต่อ SEC EDGAR ({elapsedSeconds}s)
              </p>
            </div>
            <div className="max-w-xs mx-auto h-1.5 bg-surface rounded-full overflow-hidden">
              <div className="h-full bg-accent/60 animate-pulse w-3/4 rounded-full" />
            </div>
          </div>
          <div className="h-64 bg-panel border border-edge rounded-2xl animate-pulse" />
          <div className="h-96 bg-panel border border-edge rounded-2xl animate-pulse" />
        </div>
      )}

      {/* Error State */}
      {error && !loading && (
        <div className="p-8 bg-rose-500/10 border border-rose-500/30 rounded-2xl text-center">
          <div className="text-2xl mb-2">⚠️</div>
          <p className="text-sm font-bold text-rose-400 mb-1">Error Loading Financials</p>
          <p className="text-xs text-muted mb-4">{error}</p>
          <button
            onClick={() => fetchData(true)}
            className="px-4 py-2 rounded-xl bg-surface border border-edge text-xs font-semibold text-primary hover:bg-panel shadow-sm"
          >
            Try Again
          </button>
        </div>
      )}

      {/* Empty State */}
      {!loading && !error && data?.data_status === 'empty' && (
        <div className="p-12 bg-panel border border-edge rounded-2xl text-center shadow-sm">
          <div className="text-3xl mb-3">📑</div>
          <p className="text-sm font-bold text-primary mb-1">No Financial Statements Available</p>
          <p className="text-xs text-muted max-w-md mx-auto mb-4">
            Financial statements for {ticker} could not be retrieved from SEC EDGAR or Yahoo Finance at this time. (Refresh Required)
          </p>
          <button
            onClick={() => fetchData(true)}
            className="px-4 py-2 rounded-xl bg-surface border border-edge text-xs font-semibold text-primary hover:bg-panel shadow-sm"
          >
            Retry Fetch
          </button>
        </div>
      )}

      {/* Main Content */}
      {!loading && !error && data && data.data_status !== 'empty' && (
        <div className={`space-y-6 transition-opacity duration-300 ${refreshing ? 'opacity-50 pointer-events-none' : ''}`}>
          {/* Top Modern Summary Chart */}
          <FinancialSummaryChart data={activeChartPoints} currency={data.currency} />

          {/* Key Ratios Table View */}
          {statementType === 'ratios' ? (
            <div className="bg-panel border border-edge/80 rounded-2xl shadow-sm overflow-hidden">
              <div className="p-4 border-b border-edge flex items-center justify-between bg-surface/30">
                <div>
                  <h3 className="text-sm font-bold text-primary">Financial & Operational Ratios</h3>
                  <p className="text-xs text-muted">Core profitability, debt solvency, and liquidity metrics across periods</p>
                </div>
              </div>

              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs border-collapse">
                  <thead>
                    <tr className="bg-surface/70 border-b border-edge text-muted text-[11px]">
                      <th className="p-3.5 font-semibold text-primary sticky left-0 bg-surface z-20 border-r border-edge min-w-[220px] shadow-[4px_0_10px_rgba(0,0,0,0.03)]">
                        Metric Ratio
                      </th>
                      {activeRatios.map((r) => (
                        <th key={r.period_key} className="p-3.5 text-right font-semibold min-w-[120px]">
                          <div className="font-mono font-bold text-primary">{r.period_key}</div>
                          <div className="text-[10px] text-muted font-mono">{r.period_end_date}</div>
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-edge/60">
                    <tr className="hover:bg-surface/40 transition-colors">
                      <td className="p-3.5 font-medium text-primary sticky left-0 bg-panel z-20 border-r border-edge shadow-[4px_0_10px_rgba(0,0,0,0.03)]">
                        Gross Margin (%)
                      </td>
                      {activeRatios.map((r) => (
                        <td key={r.period_key} className="p-3.5 text-right font-mono font-semibold tabular-nums">
                          {r.gross_margin_pct !== null && r.gross_margin_pct !== undefined ? `${r.gross_margin_pct}%` : '-'}
                        </td>
                      ))}
                    </tr>
                    <tr className="hover:bg-surface/40 transition-colors bg-surface/20">
                      <td className="p-3.5 font-semibold text-primary sticky left-0 bg-panel z-20 border-r border-edge shadow-[4px_0_10px_rgba(0,0,0,0.03)]">
                        Operating Margin (%)
                      </td>
                      {activeRatios.map((r) => (
                        <td key={r.period_key} className="p-3.5 text-right font-mono font-bold tabular-nums text-amber-500">
                          {r.operating_margin_pct !== null && r.operating_margin_pct !== undefined ? `${r.operating_margin_pct}%` : '-'}
                        </td>
                      ))}
                    </tr>
                    <tr className="hover:bg-surface/40 transition-colors">
                      <td className="p-3.5 font-semibold text-primary sticky left-0 bg-panel z-20 border-r border-edge shadow-[4px_0_10px_rgba(0,0,0,0.03)]">
                        Net Profit Margin (%)
                      </td>
                      {activeRatios.map((r) => (
                        <td key={r.period_key} className="p-3.5 text-right font-mono font-bold tabular-nums text-emerald-500">
                          {r.net_margin_pct !== null && r.net_margin_pct !== undefined ? `${r.net_margin_pct}%` : '-'}
                        </td>
                      ))}
                    </tr>
                    <tr className="hover:bg-surface/40 transition-colors bg-surface/20">
                      <td className="p-3.5 font-semibold text-primary sticky left-0 bg-panel z-20 border-r border-edge shadow-[4px_0_10px_rgba(0,0,0,0.03)]">
                        Free Cash Flow Margin (%)
                      </td>
                      {activeRatios.map((r) => (
                        <td key={r.period_key} className="p-3.5 text-right font-mono font-bold tabular-nums text-indigo-400">
                          {r.fcf_margin_pct !== null && r.fcf_margin_pct !== undefined ? `${r.fcf_margin_pct}%` : '-'}
                        </td>
                      ))}
                    </tr>
                    <tr className="hover:bg-surface/40 transition-colors">
                      <td className="p-3.5 font-medium text-primary sticky left-0 bg-panel z-20 border-r border-edge shadow-[4px_0_10px_rgba(0,0,0,0.03)]">
                        Current Ratio (Assets / Liab)
                      </td>
                      {activeRatios.map((r) => (
                        <td key={r.period_key} className="p-3.5 text-right font-mono tabular-nums">
                          {r.current_ratio !== null && r.current_ratio !== undefined ? `${r.current_ratio}x` : '-'}
                        </td>
                      ))}
                    </tr>
                    <tr className="hover:bg-surface/40 transition-colors bg-surface/20">
                      <td className="p-3.5 font-medium text-primary sticky left-0 bg-panel z-20 border-r border-edge shadow-[4px_0_10px_rgba(0,0,0,0.03)]">
                        Debt-to-Equity Ratio
                      </td>
                      {activeRatios.map((r) => (
                        <td key={r.period_key} className="p-3.5 text-right font-mono tabular-nums">
                          {r.debt_to_equity !== null && r.debt_to_equity !== undefined ? `${r.debt_to_equity}x` : '-'}
                        </td>
                      ))}
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>
          ) : (
            /* Multi-Period Statement Table View */
            <div className="bg-panel border border-edge/80 rounded-2xl shadow-sm overflow-hidden">
              <div className="p-4 border-b border-edge flex flex-wrap items-center justify-between gap-3 bg-surface/30">
                <div>
                  <h3 className="text-sm font-bold text-primary capitalize">
                    {statementType.replace('_', ' ')} Statement
                  </h3>
                  <p className="text-xs text-muted">
                    {statementType === 'balance_sheet'
                      ? 'Instant snapshot balance as of period end date'
                      : periodFrequency === 'quarterly'
                      ? 'Quarterly historical duration periods (Latest on left)'
                      : 'Annual historical duration periods (Latest on left)'}
                  </p>
                </div>

                {/* Extended Items Toggle Button */}
                <div className="flex items-center gap-2">
                  <button
                    onClick={() => setShowExpandedItems(!showExpandedItems)}
                    className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-surface/80 hover:bg-surface border border-edge/80 text-xs font-semibold text-primary transition-all shadow-sm"
                    title="Toggle display of extended SEC balance sheet and cash flow items"
                  >
                    <span>{showExpandedItems ? 'Hide Extended Items' : 'Show Extended Items'}</span>
                    <span className="text-[10px] text-muted">{showExpandedItems ? '▲' : '▼'}</span>
                  </button>
                </div>
              </div>

              {activeCategory && activeCategory.periods.length > 0 ? (
                <div className="overflow-x-auto">
                  <table className="w-full text-left text-xs border-collapse">
                    <thead>
                      <tr className="bg-surface/70 border-b border-edge text-muted text-[11px]">
                        {/* Sticky First Column for Line Item Labels */}
                        <th className="p-3.5 font-semibold text-primary sticky left-0 bg-surface z-20 border-r border-edge min-w-[260px] shadow-[4px_0_10px_rgba(0,0,0,0.03)]">
                          Line Item
                        </th>
                        {activeCategory.periods.map((period) => (
                          <th key={period.period_key} className="p-3.5 text-right font-semibold min-w-[140px]">
                            <div className="flex items-center justify-end gap-1.5">
                              <span className="font-mono font-bold text-primary text-xs">{period.period_key}</span>
                            </div>
                            <div className="text-[10px] text-muted font-mono mt-0.5">
                              {statementType === 'balance_sheet' ? `As of ${period.period_end_date}` : period.period_end_date}
                            </div>
                            {/* SEC EDGAR Filing Link */}
                            {data.provider === 'edgartools' && period.filing_url && (
                              <a
                                href={period.filing_url}
                                target="_blank"
                                rel="noreferrer"
                                className="inline-flex items-center gap-1 text-[10px] text-sky-500 hover:text-sky-400 hover:underline mt-1 font-medium"
                                title={`View SEC ${period.form_type} filing`}
                              >
                                <span>↗ {period.form_type}</span>
                              </a>
                            )}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-edge/60">
                      {activeCategory.line_items
                        .filter((itemMeta) => {
                          if (!showExpandedItems && EXPANDED_LINE_ITEMS.includes(itemMeta.canonical_key)) {
                            return false
                          }
                          return true
                        })
                        .map((itemMeta) => {
                          const isPrimary = itemMeta.is_primary_highlight
                          const isCalcFCF = itemMeta.canonical_key === 'calculated_free_cash_flow' || itemMeta.canonical_key === 'free_cash_flow'
                          const isRepFCF = itemMeta.canonical_key === 'reported_free_cash_flow'
                          const isTotalEq = itemMeta.canonical_key === 'total_equity'
                          const isExpandedItem = EXPANDED_LINE_ITEMS.includes(itemMeta.canonical_key)
                          return (
                            <tr
                              key={itemMeta.canonical_key}
                              className={`hover:bg-surface/50 transition-colors ${
                                isPrimary ? 'bg-surface/25' : ''
                              }`}
                            >
                              {/* Sticky Item Label */}
                              <td
                                className={`p-3.5 sticky left-0 bg-panel z-20 border-r border-edge text-primary shadow-[4px_0_10px_rgba(0,0,0,0.03)] ${
                                  isPrimary ? 'font-bold' : isExpandedItem ? 'font-normal pl-7 text-muted/80' : 'font-normal pl-5 text-muted/90'
                                }`}
                              >
                                <div className="flex items-center justify-between gap-2">
                                  <span title={
                                    isCalcFCF
                                      ? 'Calculated FCF: Operating Cash Flow - |CapEx| (Standard GAAP)'
                                      : isRepFCF
                                      ? 'Company-Reported Non-GAAP FCF from 8-K Press Release reconciliation'
                                      : isTotalEq
                                      ? 'Total Equity: Stockholders\' Equity + Non-controlling Interests'
                                      : isExpandedItem
                                      ? 'Expanded line item (SEC EDGAR XBRL taxonomy)'
                                      : undefined
                                  }>
                                    {itemMeta.display_label}
                                    {isCalcFCF && <span className="text-indigo-400 ml-1 text-[10px] font-normal">*</span>}
                                    {isRepFCF && <span className="text-purple-400 ml-1 text-[10px] font-normal">†</span>}
                                    {isExpandedItem && !isRepFCF && <span className="text-sky-400/70 ml-1 text-[9px] font-normal">◆</span>}
                                  </span>
                                  {itemMeta.unit_type === 'per_share' && (
                                    <span className="text-[10px] text-muted font-mono font-normal">/share</span>
                                  )}
                                </div>
                              </td>

                              {/* Cells per period */}
                              {activeCategory.periods.map((period) => {
                                const cell = period.items[itemMeta.canonical_key]
                                const val = cell?.value
                                const yoy = cell?.yoy_growth_pct
                                const isDerivedCell = cell?.is_derived || cell?.source_type === 'derived'
                                const isUnavailable = cell?.source_type === 'unavailable'
                                const isNotApplicable = cell?.source_type === 'not_applicable'

                                return (
                                  <td
                                    key={period.period_key}
                                    className="p-3.5 text-right font-mono tabular-nums whitespace-nowrap group/cell relative"
                                  >
                                    <div className="flex items-center justify-end gap-1">
                                      {isDerivedCell && (
                                        <span
                                          className="text-[9px] px-1 py-0.2 rounded bg-purple-500/10 text-purple-400 border border-purple-500/20 font-medium cursor-help"
                                          title={cell?.formula ? `Formula: ${cell.formula}` : cell?.derivation ? `Derivation: ${cell.derivation}` : 'Derived / Calculated'}
                                        >
                                          *
                                        </span>
                                      )}
                                      <span
                                        className={`text-xs ${
                                          isPrimary
                                            ? 'font-bold text-primary'
                                            : isNotApplicable
                                            ? 'text-muted/50 italic'
                                            : isUnavailable
                                            ? 'text-muted/50'
                                            : 'font-medium text-primary/85'
                                        }`}
                                        title={
                                          isNotApplicable
                                            ? `Not Applicable: ${cell?.derivation || 'No separate item applicable'}`
                                            : isUnavailable
                                            ? `Unavailable: ${cell?.unavailable_reason || cell?.derivation || 'Unavailable from SEC filing'}`
                                            : cell?.formula
                                            ? `Formula: ${cell.formula}`
                                            : cell?.source_concept
                                            ? `SEC Source: ${cell.source_concept}`
                                            : cell?.derivation
                                            ? `Derivation: ${cell.derivation}`
                                            : undefined
                                        }
                                      >
                                        {isNotApplicable ? '— (N/A)' : formatCellValue(val, itemMeta.unit_type)}
                                      </span>
                                    </div>

                                    {/* Modern YoY Growth Pill */}
                                    {!isNotApplicable && !isUnavailable && yoy !== null && yoy !== undefined && (
                                      <div className="mt-1">
                                        <span
                                          className={`inline-flex items-center gap-0.5 px-1.5 py-0.5 rounded-md text-[10px] font-bold ${
                                            yoy > 0
                                              ? 'bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border border-emerald-500/20'
                                              : yoy < 0
                                              ? 'bg-rose-500/10 text-rose-600 dark:text-rose-400 border border-rose-500/20'
                                              : 'bg-surface text-muted border border-edge'
                                          }`}
                                        >
                                          <span>{yoy > 0 ? '▲' : yoy < 0 ? '▼' : '•'}</span>
                                          <span>{yoy > 0 ? `+${yoy}%` : `${yoy}%`} YoY</span>
                                        </span>
                                      </div>
                                    )}
                                  </td>
                                )
                              })}
                            </tr>
                          )
                        })}
                    </tbody>
                  </table>

                  {/* Footnotes */}
                  <div className="p-3 border-t border-edge/60 bg-surface/20 text-[11px] text-muted space-y-1">
                    <p>
                      <span className="font-semibold text-indigo-400">* Calculated Free Cash Flow:</span> Operating Cash Flow − |CapEx| (Standard GAAP definition).
                    </p>
                    <p>
                      <span className="font-semibold text-purple-400">† Company-Reported Free Cash Flow:</span> Non-GAAP metric parsed directly from 8-K Press Release reconciliation tables (e.g. IP litigation settlements).
                    </p>
                    <p>
                      <span className="font-semibold text-sky-400/80">◆ Extended Line Item:</span> Detailed breakdown items extracted from SEC XBRL taxonomy (e.g. Deferred Contract Costs, NCI, Tax Liabilities).
                    </p>
                    <p>
                      <span className="font-semibold text-purple-400">* Cell Asterisk:</span> Indicates derived mathematical values (e.g. Q4 = FY − (Q1+Q2+Q3), Total Equity = Stockholders' Equity + NCI).
                    </p>
                  </div>
                </div>
              ) : (
                <div className="p-8 text-center text-xs text-muted">
                  No line item records for this statement view.
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  )
}
