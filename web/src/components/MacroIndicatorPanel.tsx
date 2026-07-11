import { useEffect, useState } from 'react'
import { api, ApiError } from '../api/client'
import type { MacroIndicatorDTO, MacroIndicatorSeriesDTO, MacroSeriesPointDTO } from '../api/types'

type RangeKey = '1m' | '3m' | '1y'

const RANGE_LABELS: Record<RangeKey, string> = {
  '1m': '1M',
  '3m': '3M',
  '1y': '1Y',
}

function pointPosition(points: MacroSeriesPointDTO[], index: number): { x: number; y: number } {
  const width = 560
  const height = 180
  const marginX = 32
  const marginY = 18
  const values = points.map((point) => point.value)
  const min = Math.min(...values)
  const max = Math.max(...values)
  const span = max - min || Math.max(Math.abs(max) * 0.1, 1)
  return {
    x: marginX + (index / Math.max(points.length - 1, 1)) * (width - marginX * 2),
    y: height - marginY - ((points[index].value - min) / span) * (height - marginY * 2),
  }
}

function chartPath(points: MacroSeriesPointDTO[]): string {
  return points
    .map((_, index) => {
      const { x, y } = pointPosition(points, index)
      return `${index === 0 ? 'M' : 'L'} ${x} ${y}`
    })
    .join(' ')
}

function dateLabel(value: string): string {
  const parsed = new Date(`${value}T00:00:00`)
  return Number.isNaN(parsed.valueOf())
    ? value
    : new Intl.DateTimeFormat('th-TH', { month: 'short', day: 'numeric' }).format(parsed)
}

function fredChartUrl(seriesId: string): string {
  return `https://fred.stlouisfed.org/series/${seriesId}`
}

function externalChartUrl(indicator: MacroIndicatorDTO): string | null {
  const lookup = `${indicator.series_key} ${indicator.label}`.toLowerCase()
  if (lookup.includes('us10y') || lookup.includes('us_10y') || lookup.includes('10y yield')) return 'https://www.tradingview.com/chart/?symbol=TVC%3AUS10Y'
  if (lookup.includes('10y_2y') || lookup.includes('10y-2y')) return fredChartUrl('T10Y2Y')
  if (lookup.includes('vix')) return 'https://www.tradingview.com/chart/?symbol=TVC%3AVIX'
  if (lookup.includes('hyg') && lookup.includes('lqd')) return 'https://www.tradingview.com/chart/?symbol=AMEX%3AHYG'
  if (lookup.includes('wti') || lookup.includes('crude') || lookup.includes('cl_f')) return 'https://www.tradingview.com/chart/?symbol=NYMEX%3ACL1%21'
  if (lookup.includes('sp500') || lookup.includes('s&p') || lookup.includes('erp_gspc')) return 'https://www.tradingview.com/chart/?symbol=SP%3ASPX'
  if (lookup.includes('dxy') || lookup.includes('dollar index')) return 'https://www.tradingview.com/chart/?symbol=TVC%3ADXY'
  if (lookup.includes('gold')) return 'https://www.tradingview.com/chart/?symbol=TVC%3AGOLD'
  if (lookup.includes('fedfunds') || lookup.includes('fed funds')) return fredChartUrl('FEDFUNDS')
  if (lookup.includes('core_pce') || lookup.includes('core pce')) return fredChartUrl('PCEPILFE')
  if (lookup.includes('cpi')) return fredChartUrl('CPIAUCSL')
  if (lookup.includes('gdp')) return fredChartUrl('GDP')
  if (lookup.includes('unrate') || lookup.includes('unemployment')) return fredChartUrl('UNRATE')
  if (lookup.includes('tbill') || lookup.includes('3mo')) return fredChartUrl('DGS3MO')
  if (lookup.includes('dfii10')) return fredChartUrl('DFII10')
  if (lookup.includes('hy_spread') || lookup.includes('high yield spread')) return fredChartUrl('BAMLH0A0HYM2')
  return null
}

interface Props {
  indicators: MacroIndicatorDTO[]
}

export default function MacroIndicatorPanel({ indicators }: Props) {
  const [selectedId, setSelectedId] = useState(indicators[0]?.indicator_id ?? '')
  const [range, setRange] = useState<RangeKey>('3m')
  const [series, setSeries] = useState<MacroIndicatorSeriesDTO | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [hoveredPoint, setHoveredPoint] = useState<MacroSeriesPointDTO | null>(null)

  const selected = indicators.find((indicator) => indicator.indicator_id === selectedId) ?? indicators[0]
  const chartPoints = series?.points ?? []
  const minValue = chartPoints.length ? Math.min(...chartPoints.map((point) => point.value)) : 0
  const maxValue = chartPoints.length ? Math.max(...chartPoints.map((point) => point.value)) : 0

  useEffect(() => {
    if (!selected) return
    setSelectedId((current) => (indicators.some((indicator) => indicator.indicator_id === current) ? current : selected.indicator_id))
  }, [indicators, selected])

  useEffect(() => {
    if (!selected?.chart_available) {
      setSeries(null)
      setError(null)
      return
    }
    let isCurrent = true
    setIsLoading(true)
    setError(null)
    setHoveredPoint(null)
    api
      .getMacroIndicatorSeries(selected.indicator_id, range)
      .then((nextSeries) => {
        if (isCurrent) setSeries(nextSeries)
      })
      .catch((requestError) => {
        if (!isCurrent) return
        setSeries(null)
        setError(requestError instanceof ApiError ? requestError.message : 'ไม่สามารถโหลดข้อมูลกราฟได้')
      })
      .finally(() => {
        if (isCurrent) setIsLoading(false)
      })
    return () => {
      isCurrent = false
    }
  }, [range, selected?.chart_available, selected?.indicator_id])

  if (!selected) return null

  const displayedPoint = hoveredPoint ?? chartPoints[chartPoints.length - 1]
  const externalChart = externalChartUrl(selected)

  return (
    <section className="rounded-2xl border border-zinc-200/80 bg-white p-5 shadow-sm shadow-black/5">
      <div className="mb-4 flex flex-col gap-3 lg:flex-row lg:items-start lg:justify-between">
        <div>
          <p className="text-xs font-bold uppercase tracking-wider text-zinc-500">Key Macro Indicators</p>
          <h2 className="mt-1 text-lg font-semibold text-zinc-900">ตัวชี้วัดที่ใช้ประกอบการตัดสินใจ</h2>
          <p className="mt-1 text-xs text-zinc-500">เลือกตัวชี้วัดเพื่อดูประวัติจาก snapshot ของรายงานที่ตรวจสอบแล้ว</p>
        </div>
        <div className="flex items-center gap-1 rounded-lg border border-zinc-200 bg-zinc-50 p-1" aria-label="ช่วงเวลาของกราฟ">
          {(Object.keys(RANGE_LABELS) as RangeKey[]).map((rangeKey) => (
            <button
              key={rangeKey}
              onClick={() => setRange(rangeKey)}
              aria-pressed={range === rangeKey}
              className={`rounded-md px-2.5 py-1 text-xs font-semibold transition-colors ${
                range === rangeKey ? 'bg-zinc-900 text-white shadow-sm' : 'text-zinc-600 hover:text-zinc-900'
              }`}
            >
              {RANGE_LABELS[rangeKey]}
            </button>
          ))}
        </div>
      </div>

      <div className="mb-4 flex flex-wrap items-end justify-between gap-3 rounded-xl border border-zinc-200 bg-zinc-50/70 p-3">
        <label className="grid gap-1 text-xs font-semibold text-zinc-600" htmlFor="macro-indicator-select">
          เลือกตัวชี้วัด
          <select
            id="macro-indicator-select"
            value={selected.indicator_id}
            onChange={(event) => setSelectedId(event.target.value)}
            className="min-w-60 rounded-lg border border-zinc-300 bg-white px-3 py-2 text-sm font-medium text-zinc-900 shadow-sm outline-none transition-colors focus:border-zinc-900"
          >
            {indicators.map((indicator) => (
              <option key={indicator.indicator_id} value={indicator.indicator_id}>
                {indicator.label}{indicator.display_value ? ` — ${indicator.display_value}` : ''}
              </option>
            ))}
          </select>
        </label>
        <span className="text-xs text-zinc-500">เลือกเพื่อเปลี่ยนกราฟโดยไม่ต้องแสดงรายการทั้งหมด</span>
      </div>

      <div className="space-y-4">
        <div className="hidden">
          {indicators.map((indicator) => {
            const isSelected = indicator.indicator_id === selected.indicator_id
            return (
              <button
                key={indicator.indicator_id}
                onClick={() => setSelectedId(indicator.indicator_id)}
                aria-pressed={isSelected}
                className={`rounded-xl border p-3 text-left transition-colors ${
                  isSelected ? 'border-zinc-900 bg-zinc-900 text-white shadow-sm' : 'border-zinc-200 bg-white hover:border-zinc-400'
                }`}
              >
                <div className={`text-xs font-semibold ${isSelected ? 'text-zinc-200' : 'text-zinc-900'}`}>{indicator.label}</div>
                <div className="mt-1 flex items-end justify-between gap-2">
                  <span className={`font-mono text-sm font-bold ${isSelected ? 'text-white' : 'text-zinc-700'}`}>{indicator.display_value || '—'}</span>
                  {!indicator.is_valid && <span className="text-[10px] font-semibold text-amber-600">STALE</span>}
                </div>
              </button>
            )
          })}
        </div>

        <div className="rounded-xl border border-zinc-200 bg-zinc-50/70 p-4">
          <div className="flex flex-wrap items-start justify-between gap-2">
            <div>
              <h3 className="font-semibold text-zinc-900">{selected.label}</h3>
              <p className="mt-0.5 text-xs text-zinc-500">{selected.provider || 'Macro data'} · ณ {selected.observed_at || '—'}</p>
            </div>
            <div className="text-right">
              <div className="font-mono text-base font-bold text-zinc-900">{selected.display_value || '—'}</div>
              <div className="text-[11px] text-zinc-500">{selected.unit}</div>
            </div>
          </div>

          {selected.stale_reason && (
            <p className="mt-3 rounded-lg border border-amber-200 bg-amber-50 px-2.5 py-2 text-xs text-amber-800">{selected.stale_reason}</p>
          )}

          <div className="mt-3 min-h-52">
            {isLoading && <div className="animate-shimmer h-52 rounded-lg border border-zinc-200" />}
            {!isLoading && error && <p className="rounded-lg bg-red-50 p-3 text-xs text-red-700">{error}</p>}
            {!isLoading && !error && !selected.chart_available && (
              <p className="rounded-lg border border-zinc-200 bg-white p-3 text-sm text-zinc-500">ตัวชี้วัดนี้ยังไม่มีข้อมูลเชิงตัวเลขสำหรับสร้างกราฟ</p>
            )}
            {!isLoading && !error && selected.chart_available && chartPoints.length < 2 && (
              <p className="rounded-lg border border-zinc-200 bg-white p-3 text-sm text-zinc-500">เริ่มเก็บข้อมูลแล้ว จะสร้างกราฟได้เมื่อมี snapshot รายงานอย่างน้อย 2 จุด</p>
            )}
            {!isLoading && !error && chartPoints.length >= 2 && (
              <div>
                <div className="mb-2 flex items-center justify-between text-xs text-zinc-500">
                  <span>{minValue.toLocaleString()}</span>
                  <span>{displayedPoint ? `${displayedPoint.value.toLocaleString()} · ${dateLabel(displayedPoint.observed_at)}` : ''}</span>
                  <span>{maxValue.toLocaleString()}</span>
                </div>
                <svg viewBox="0 0 560 180" className="h-52 w-full" role="img" aria-label={`กราฟ ${selected.label}`}>
                  {[48, 90, 132].map((y) => <line key={y} x1="32" x2="528" y1={y} y2={y} stroke="#e4e4e7" strokeDasharray="3 4" />)}
                  <path d={chartPath(chartPoints)} fill="none" stroke="#18181b" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" />
                  {chartPoints.map((point, index) => {
                    const { x, y } = pointPosition(chartPoints, index)
                    return (
                      <circle
                        key={`${point.observed_at}-${point.value}`}
                        cx={x}
                        cy={y}
                        r="4.5"
                        fill="#f59e0b"
                        stroke="#ffffff"
                        strokeWidth="2"
                        tabIndex={0}
                        onMouseEnter={() => setHoveredPoint(point)}
                        onFocus={() => setHoveredPoint(point)}
                      >
                        <title>{`${dateLabel(point.observed_at)}: ${point.value.toLocaleString()} ${selected.unit}`}</title>
                      </circle>
                    )
                  })}
                </svg>
                <div className="flex justify-between text-[11px] text-zinc-400">
                  <span>{dateLabel(chartPoints[0].observed_at)}</span>
                  <span>{dateLabel(chartPoints[chartPoints.length - 1].observed_at)}</span>
                </div>
              </div>
            )}
          </div>
          <div className="mt-2 flex flex-wrap items-center justify-between gap-2">
            <p className="text-[11px] text-zinc-400">แหล่งข้อมูล: {selected.source_file || selected.provider || 'ไม่ระบุ'}</p>
            {externalChart && (
              <a
              href={externalChart}
              target="_blank"
              rel="noopener noreferrer"
              className="text-xs font-semibold text-zinc-700 underline underline-offset-2 transition-colors hover:text-amber-700"
            >
              เปิดกราฟภายนอก ↗
              </a>
            )}
          </div>
        </div>
      </div>
    </section>
  )
}
