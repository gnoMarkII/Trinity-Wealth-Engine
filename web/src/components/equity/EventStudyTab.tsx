import React, { useState, useEffect, useMemo, useCallback, useRef } from 'react'
import { api } from '../../api/client'
import type {
  EventPresetItemDTO,
  EventStudyResponseDTO,
} from '../../api/types'

interface EventStudyTabProps {
  ticker: string
  market?: 'US' | 'TH'
}

export const EventStudyTab: React.FC<EventStudyTabProps> = ({ ticker, market = 'US' }) => {
  const currency = market === 'TH' ? '฿' : '$'
  const activeRequestIdRef = useRef(0)

  const [presets, setPresets] = useState<EventPresetItemDTO[]>([])
  const [selectedPreset, setSelectedPreset] = useState<string>('earnings')
  const [customDatesInput, setCustomDatesInput] = useState<string>('')
  const [customEventName, setCustomEventName] = useState<string>('')
  const [showCustomInput, setShowCustomInput] = useState<boolean>(false)

  const [activeHorizon, setActiveHorizon] = useState<number>(20)
  const [hoveredEventId, setHoveredEventId] = useState<string | null>(null)

  const [studyData, setStudyData] = useState<EventStudyResponseDTO | null>(null)
  const [loading, setLoading] = useState<boolean>(true)
  const [error, setError] = useState<string | null>(null)

  // Fetch preset catalogs on mount
  useEffect(() => {
    let mounted = true
    api.getEventStudyPresets()
      .then((data) => {
        if (mounted) {
          setPresets(data)
          // Default to ticker-specific preset if available, else earnings
          const tickerUpper = ticker.toUpperCase()
          const matched = data.find(
            (p) => p.recommended_ticker && p.recommended_ticker.toUpperCase() === tickerUpper
          )
          if (matched) {
            setSelectedPreset(matched.preset_id)
          } else {
            setSelectedPreset('earnings')
          }
        }
      })
      .catch((err) => {
        console.error('Failed to load event presets:', err)
      })
    return () => {
      mounted = false
    }
  }, [ticker])

  const { recommendedPresets, otherPresets } = useMemo(() => {
    const tickerUpper = ticker.toUpperCase()
    const recommended: EventPresetItemDTO[] = []
    const others: EventPresetItemDTO[] = []
    presets.forEach((p) => {
      if (!p.recommended_ticker || p.recommended_ticker.toUpperCase() === tickerUpper) {
        recommended.push(p)
      } else {
        others.push(p)
      }
    })
    return { recommendedPresets: recommended, otherPresets: others }
  }, [presets, ticker])

  // Run analysis when preset or ticker changes
  const executeAnalysis = useCallback(
    async (presetIdOverride?: string) => {
      const pId = presetIdOverride || selectedPreset
      const currentReqId = ++activeRequestIdRef.current
      setLoading(true)
      setError(null)
      try {
        let customDates: string[] | undefined = undefined
        if (pId === 'custom' && customDatesInput.trim()) {
          customDates = customDatesInput
            .split(/[\n,;]+/)
            .map((s) => s.trim())
            .filter((s) => /^\d{4}-\d{2}-\d{2}$/.test(s))
        }

        const res = await api.postEventStudy(ticker, {
          preset_id: pId,
          custom_dates: customDates,
          custom_event_name: pId === 'custom' ? customEventName : undefined,
          horizons: [5, 10, 20, 60],
        })
        if (currentReqId === activeRequestIdRef.current) {
          setStudyData(res)
        }
      } catch (err: any) {
        if (currentReqId === activeRequestIdRef.current) {
          setError(err.message || 'Failed to execute event study.')
        }
      } finally {
        if (currentReqId === activeRequestIdRef.current) {
          setLoading(false)
        }
      }
    },
    [ticker, selectedPreset, customDatesInput, customEventName]
  )

  useEffect(() => {
    return () => {
      activeRequestIdRef.current++
    }
  }, [ticker])

  useEffect(() => {
    if (selectedPreset) {
      executeAnalysis(selectedPreset)
    }
  }, [ticker, selectedPreset])

  const handlePresetChange = (pId: string) => {
    setSelectedPreset(pId)
    setShowCustomInput(pId === 'custom')
  }

  // Chart Dimensions & Scales
  const chartWidth = 760
  const chartHeight = 320
  const padLeft = 55
  const padRight = 30
  const padTop = 25
  const padBottom = 35
  const plotWidth = chartWidth - padLeft - padRight
  const plotHeight = chartHeight - padTop - padBottom

  const maxDays = useMemo(() => {
    return activeHorizon
  }, [activeHorizon])

  const { minReturn, maxReturn, eventsToPlot, avgPathToPlot } = useMemo(() => {
    if (!studyData || !studyData.events || studyData.events.length === 0) {
      return { minReturn: -5, maxReturn: 5, eventsToPlot: [], avgPathToPlot: [] }
    }

    let minR = 0
    let maxR = 0

    const evts = studyData.events.map((ev) => {
      const trajectory = ev.daily_trajectory.filter((p) => p.day_offset <= maxDays)
      trajectory.forEach((p) => {
        if (p.cumulative_return_pct < minR) minR = p.cumulative_return_pct
        if (p.cumulative_return_pct > maxR) maxR = p.cumulative_return_pct
      })
      const isWin = ev.is_win_by_horizon[String(activeHorizon)] ?? false
      return {
        ...ev,
        filteredTrajectory: trajectory,
        isWinAtHorizon: isWin,
      }
    })

    const avgTraj = (studyData.average_trajectory || []).filter((p) => p.day_offset <= maxDays)
    avgTraj.forEach((p) => {
      if (p.avg_return_pct < minR) minR = p.avg_return_pct
      if (p.avg_return_pct > maxR) maxR = p.avg_return_pct
    })

    // Add 15% padding above and below
    const range = Math.max(Math.abs(minR), Math.abs(maxR), 2)
    const paddedLimit = Math.ceil(range * 1.15)
    return {
      minReturn: -paddedLimit,
      maxReturn: paddedLimit,
      eventsToPlot: evts,
      avgPathToPlot: avgTraj,
    }
  }, [studyData, maxDays, activeHorizon])

  const scaleX = useCallback(
    (day: number) => {
      if (maxDays <= 1) return padLeft
      return padLeft + ((day - 1) / (maxDays - 1)) * plotWidth
    },
    [maxDays, plotWidth, padLeft]
  )

  const scaleY = useCallback(
    (retPct: number) => {
      const span = maxReturn - minReturn || 1
      const normalized = (retPct - minReturn) / span
      return padTop + plotHeight - normalized * plotHeight
    },
    [minReturn, maxReturn, padTop, plotHeight]
  )

  return (
    <div className="space-y-6">
      {/* 1. Header & Controls */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 p-4 rounded-xl border border-zinc-200/80 bg-zinc-50/70 shadow-sm">
        <div className="space-y-1">
          <div className="flex items-center gap-2">
            <span className="text-xs font-bold uppercase tracking-wider text-sky-600">Event Study Engine</span>
            <span className="text-xs px-2 py-0.5 rounded-full bg-sky-100 text-sky-800 font-semibold border border-sky-200">
              Quantitative Backtest
            </span>
          </div>
          <h2 className="text-lg font-bold text-zinc-900">
            {studyData?.event_name || 'Historical Event Analysis'}
          </h2>
          <p className="text-xs text-zinc-500">
            วิเคราะห์ผลตอบแทนไปข้างหน้า (Forward Returns) นับจากราคาเปิดวันถัดไป ($Open_{'{t+1}'}$) เพื่อกำจัด Look-Ahead Bias
          </p>
        </div>

        {/* Preset Selector */}
        <div className="flex items-center gap-2">
          <label htmlFor="preset-select" className="text-xs font-semibold text-zinc-600 whitespace-nowrap">
            เลือกเหตุการณ์:
          </label>
          <select
            id="preset-select"
            value={selectedPreset}
            onChange={(e) => handlePresetChange(e.target.value)}
            className="text-xs font-medium px-3 py-1.5 rounded-lg border border-zinc-300 bg-white text-zinc-800 shadow-sm focus:outline-none focus:ring-2 focus:ring-sky-500"
          >
            {recommendedPresets.length > 0 && (
              <optgroup label={`เหตุการณ์ของ ${ticker.toUpperCase()} & เหตุการณ์สากล`}>
                {recommendedPresets.map((p) => (
                  <option key={p.preset_id} value={p.preset_id}>
                    {p.name} {p.recommended_ticker ? `(${p.recommended_ticker})` : ''}
                  </option>
                ))}
              </optgroup>
            )}
            {otherPresets.length > 0 && (
              <optgroup label="เหตุการณ์เฉพาะบริษัทอื่น (Cross-Asset Study)">
                {otherPresets.map((p) => (
                  <option key={p.preset_id} value={p.preset_id}>
                    {p.name} ({p.recommended_ticker})
                  </option>
                ))}
              </optgroup>
            )}
          </select>
          <button
            onClick={() => executeAnalysis()}
            disabled={loading}
            className="px-3 py-1.5 rounded-lg text-xs font-bold bg-sky-600 text-white hover:bg-sky-700 disabled:opacity-50 transition-colors shadow-sm"
          >
            {loading ? 'กำลังวิเคราะห์...' : 'รีเฟรช'}
          </button>
        </div>
      </div>

      {/* Custom Dates Drawer if 'custom' is selected */}
      {showCustomInput && (
        <div className="p-4 rounded-xl border border-sky-200 bg-sky-50/50 space-y-3">
          <div className="flex items-center justify-between">
            <span className="text-xs font-bold text-sky-900">กำหนดรายการวันที่เหตุการณ์เอง (Custom Event Dates)</span>
            <span className="text-xs text-sky-700">รูปแบบ: YYYY-MM-DD (คั่นด้วยจุลภาคหรือบรรทัดใหม่)</span>
          </div>
          <input
            type="text"
            placeholder="ชื่อเหตุการณ์ เช่น FDA Approvals, Product Launch..."
            value={customEventName}
            onChange={(e) => setCustomEventName(e.target.value)}
            className="w-full text-xs px-3 py-2 rounded-lg border border-zinc-300 bg-white text-zinc-900 focus:outline-none focus:ring-2 focus:ring-sky-500"
          />
          <textarea
            rows={3}
            placeholder="2024-01-15, 2023-06-20, 2022-11-10..."
            value={customDatesInput}
            onChange={(e) => setCustomDatesInput(e.target.value)}
            className="w-full text-xs font-mono px-3 py-2 rounded-lg border border-zinc-300 bg-white text-zinc-900 focus:outline-none focus:ring-2 focus:ring-sky-500"
          />
          <button
            onClick={() => executeAnalysis('custom')}
            disabled={loading || !customDatesInput.trim()}
            className="px-4 py-2 rounded-lg text-xs font-bold bg-sky-600 text-white hover:bg-sky-700 disabled:opacity-50"
          >
            คำนวณเหตุการณ์ที่กำหนด
          </button>
        </div>
      )}

      {/* Warnings Banner */}
      {studyData?.warnings && studyData.warnings.length > 0 && (
        <div className="space-y-2">
          {studyData.warnings.map((w, idx) => (
            <div
              key={idx}
              className="flex items-start gap-2.5 p-3.5 rounded-lg border border-amber-200 bg-amber-50 text-amber-900 text-xs"
            >
              <span className="text-base leading-none">⚠️</span>
              <div>
                <strong className="font-bold uppercase tracking-wide mr-1.5">{w.code}:</strong>
                {w.message}
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Loading Skeleton */}
      {loading && !studyData && (
        <div className="p-8 text-center space-y-3 rounded-xl border border-zinc-200 bg-white">
          <div className="inline-block w-8 h-8 border-4 border-sky-600 border-t-transparent rounded-full animate-spin" />
          <p className="text-xs font-medium text-zinc-500">กำลังประมวลผลแท่งเทียนและ Forward Returns ย้อนหลัง...</p>
        </div>
      )}

      {/* Error View */}
      {error && (
        <div className="p-4 rounded-xl border border-rose-200 bg-rose-50 text-rose-800 text-xs">
          <strong>เกิดข้อผิดพลาด:</strong> {error}
        </div>
      )}

      {/* Data Loaded */}
      {studyData && !loading && (
        <>
          {/* 2. Four Horizon Summary Cards */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3.5">
            {[5, 10, 20, 60].map((h) => {
              const stats = studyData.horizon_stats[String(h)]
              const isSelected = activeHorizon === h
              const avg = stats?.avg_return_pct ?? 0
              const hitRate = stats?.hit_rate_pct ?? 0
              const isPositive = avg >= 0

              return (
                <button
                  key={h}
                  onClick={() => setActiveHorizon(h)}
                  className={`text-left p-4 rounded-xl border transition-all ${
                    isSelected
                      ? 'border-sky-500 bg-sky-50/50 shadow-md ring-2 ring-sky-500/20'
                      : 'border-zinc-200 bg-white hover:border-zinc-300 hover:bg-zinc-50/50 shadow-sm'
                  }`}
                >
                  <div className="flex items-center justify-between text-xs text-zinc-500 font-semibold mb-1">
                    <span>ขอบเขต {h} วันทำการ</span>
                    <span className="text-[11px] font-mono">N={stats?.sample_size ?? 0}</span>
                  </div>

                  <div className="flex items-baseline gap-1.5 my-1">
                    <span
                      className={`text-2xl font-black tracking-tight ${
                        isPositive ? 'text-emerald-600' : 'text-rose-600'
                      }`}
                    >
                      {isPositive ? '+' : ''}
                      {avg.toFixed(2)}%
                    </span>
                    <span className="text-[11px] font-medium text-zinc-400">เฉลี่ย</span>
                  </div>

                  <div className="flex items-center justify-between text-xs mt-2.5 pt-2 border-t border-zinc-100">
                    <span className="text-zinc-500 font-medium">Hit Rate:</span>
                    <span
                      className={`px-2 py-0.5 rounded-md font-bold text-xs ${
                        hitRate >= 50
                          ? 'bg-emerald-100 text-emerald-800'
                          : 'bg-rose-100 text-rose-800'
                      }`}
                    >
                      {hitRate.toFixed(1)}%
                    </span>
                  </div>

                  <div className="flex items-center justify-between text-[11px] text-zinc-400 mt-1">
                    <span>มัธยฐาน (Median):</span>
                    <span className="font-mono text-zinc-700 font-semibold">
                      {stats?.median_return_pct ? `${stats.median_return_pct > 0 ? '+' : ''}${stats.median_return_pct}%` : '0%'}
                    </span>
                  </div>
                </button>
              )
            })}
          </div>

          {/* 3. Interactive PnL Spaghetti Path Chart */}
          <div className="p-5 rounded-xl border border-zinc-200/90 bg-white shadow-sm space-y-3">
            <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-2 border-b border-zinc-100 pb-3">
              <div>
                <h3 className="text-sm font-bold text-zinc-900">
                  PnL Cumulative Path Chart (เส้นทางผลตอบแทนสะสม {activeHorizon} วัน)
                </h3>
                <p className="text-xs text-zinc-500">
                  เส้นสีเขียว = จบด้วยกำไร ณ วันที่ {activeHorizon} | เส้นสีแดง = จบด้วยขาดทุน | เส้นประสีฟ้าหนา = ค่าเฉลี่ย
                </p>
              </div>

              {/* Horizon Tabs */}
              <div className="flex items-center gap-1 bg-zinc-100 p-1 rounded-lg border border-zinc-200">
                {[5, 10, 20, 60].map((h) => (
                  <button
                    key={h}
                    onClick={() => setActiveHorizon(h)}
                    className={`px-2.5 py-1 text-xs font-bold rounded-md transition-all ${
                      activeHorizon === h
                        ? 'bg-white text-zinc-900 shadow-xs'
                        : 'text-zinc-600 hover:text-zinc-900'
                    }`}
                  >
                    {h}D
                  </button>
                ))}
              </div>
            </div>

            {/* SVG Chart */}
            <div className="relative w-full overflow-x-auto">
              <svg
                viewBox={`0 0 ${chartWidth} ${chartHeight}`}
                className="w-full h-auto select-none overflow-visible"
              >
                {/* Horizontal Grid lines */}
                {[-10, -5, 0, 5, 10].map((val) => {
                  if (val < minReturn || val > maxReturn) return null
                  const y = scaleY(val)
                  const isZero = val === 0
                  return (
                    <g key={val}>
                      <line
                        x1={padLeft}
                        y1={y}
                        x2={chartWidth - padRight}
                        y2={y}
                        stroke={isZero ? '#94A3B8' : '#E2E8F0'}
                        strokeWidth={isZero ? 1.5 : 0.8}
                        strokeDasharray={isZero ? '4,4' : undefined}
                      />
                      <text
                        x={padLeft - 8}
                        y={y + 3.5}
                        textAnchor="end"
                        className="text-[10px] font-mono fill-zinc-400 font-medium"
                      >
                        {val > 0 ? `+${val}%` : `${val}%`}
                      </text>
                    </g>
                  )
                })}

                {/* X-axis Day ticks */}
                {Array.from({ length: maxDays }, (_, i) => i + 1).map((d) => {
                  // Only show label at day 1, 5, 10, 20, 30, 40, 50, 60 or maxDays
                  const showLabel = d === 1 || d === 5 || d === 10 || d === 20 || d === 40 || d === 60 || d === maxDays
                  const x = scaleX(d)
                  return (
                    <g key={d}>
                      <line
                        x1={x}
                        y1={padTop + plotHeight}
                        x2={x}
                        y2={padTop + plotHeight + 4}
                        stroke="#CBD5E1"
                        strokeWidth={1}
                      />
                      {showLabel && (
                        <text
                          x={x}
                          y={padTop + plotHeight + 16}
                          textAnchor="middle"
                          className="text-[10px] font-mono fill-zinc-400 font-medium"
                        >
                          D+{d}
                        </text>
                      )}
                    </g>
                  )
                })}

                {/* Event Individual Paths (Spaghetti Lines) */}
                {eventsToPlot.map((ev) => {
                  if (ev.filteredTrajectory.length < 2) return null
                  const isHovered = hoveredEventId === ev.event_id
                  const isDimmed = hoveredEventId !== null && !isHovered

                  const pathD = ev.filteredTrajectory.reduce((acc, pt, i) => {
                    const x = scaleX(pt.day_offset)
                    const y = scaleY(pt.cumulative_return_pct)
                    return i === 0 ? `M ${x} ${y}` : `${acc} L ${x} ${y}`
                  }, '')

                  const strokeColor = ev.isWinAtHorizon ? '#10B981' : '#F43F5E'

                  return (
                    <g key={ev.event_id}>
                      <path
                        d={pathD}
                        fill="none"
                        stroke={strokeColor}
                        strokeWidth={isHovered ? 3 : 1.4}
                        strokeOpacity={isDimmed ? 0.18 : isHovered ? 1.0 : 0.65}
                        className="cursor-pointer transition-all duration-150"
                        onMouseEnter={() => setHoveredEventId(ev.event_id)}
                        onMouseLeave={() => setHoveredEventId(null)}
                      />
                      {/* End dot */}
                      {(() => {
                        const lastPt = ev.filteredTrajectory[ev.filteredTrajectory.length - 1]
                        if (!lastPt) return null
                        return (
                          <circle
                            cx={scaleX(lastPt.day_offset)}
                            cy={scaleY(lastPt.cumulative_return_pct)}
                            r={isHovered ? 4.5 : 2.5}
                            fill={strokeColor}
                            fillOpacity={isDimmed ? 0.2 : 0.85}
                            onMouseEnter={() => setHoveredEventId(ev.event_id)}
                            onMouseLeave={() => setHoveredEventId(null)}
                            className="cursor-pointer"
                          />
                        )
                      })()}
                    </g>
                  )
                })}

                {/* Average Trajectory Path (Thick dashed Blue Line) */}
                {avgPathToPlot.length >= 2 && (
                  <path
                    d={avgPathToPlot.reduce((acc, pt, i) => {
                      const x = scaleX(pt.day_offset)
                      const y = scaleY(pt.avg_return_pct)
                      return i === 0 ? `M ${x} ${y}` : `${acc} L ${x} ${y}`
                    }, '')}
                    fill="none"
                    stroke="#0284C7"
                    strokeWidth={2.8}
                    strokeDasharray="5,4"
                    strokeLinecap="round"
                  />
                )}
              </svg>

              {/* Hover Legend Callout */}
              {hoveredEventId && (
                <div className="absolute top-2 right-2 bg-zinc-900/90 text-white text-xs p-2.5 rounded-lg shadow-lg backdrop-blur-xs max-w-xs space-y-1">
                  {(() => {
                    const ev = studyData.events.find((e) => e.event_id === hoveredEventId)
                    if (!ev) return null
                    const retAtH = ev.forward_returns[String(activeHorizon)]
                    return (
                      <>
                        <div className="font-bold text-sky-300">{ev.description || ev.event_date}</div>
                        <div className="text-[11px] text-zinc-300">
                          วันที่ประกาศ: <span className="font-mono text-white">{ev.event_date}</span>
                        </div>
                        <div className="text-[11px] text-zinc-300">
                          เข้าซื้อ ({ev.entry_date}): <span className="font-mono text-white">{currency}{ev.entry_price}</span>
                        </div>
                        <div className="text-[11px] font-bold">
                          ผลตอบแทน D+{activeHorizon}:{' '}
                          <span
                            className={
                              retAtH != null && retAtH >= 0 ? 'text-emerald-400 font-mono' : 'text-rose-400 font-mono'
                            }
                          >
                            {retAtH != null ? `${retAtH > 0 ? '+' : ''}${retAtH}%` : 'N/A'}
                          </span>
                        </div>
                      </>
                    )
                  })()}
                </div>
              )}
            </div>

            <div className="flex items-center justify-between text-xs text-zinc-500 pt-2 border-t border-zinc-100">
              <div className="flex items-center gap-4">
                <span className="flex items-center gap-1.5">
                  <span className="w-3 h-1 bg-emerald-500 rounded-full" /> รอบที่กำไร
                </span>
                <span className="flex items-center gap-1.5">
                  <span className="w-3 h-1 bg-rose-500 rounded-full" /> รอบที่ขาดทุน
                </span>
                <span className="flex items-center gap-1.5">
                  <span className="w-3 h-1 border-t-2 border-dashed border-sky-600" /> ค่าเฉลี่ยทุกลูป
                </span>
              </div>
              <span className="text-[11px] text-zinc-400">ชี้เมาส์ที่เส้นเพื่อดูรายละเอียดแต่ละเหตุการณ์</span>
            </div>
          </div>

          {/* 4. Horizon Distribution Stats Table */}
          <div className="p-4 rounded-xl border border-zinc-200 bg-white shadow-sm space-y-3">
            <h3 className="text-sm font-bold text-zinc-900">
              สถิติการกระจายตัวของผลตอบแทน (Statistical Distribution Summary)
            </h3>
            <div className="overflow-x-auto">
              <table className="w-full text-xs text-left">
                <thead className="bg-zinc-50 text-zinc-600 font-bold border-b border-zinc-200">
                  <tr>
                    <th className="py-2.5 px-3">ขอบเขต</th>
                    <th className="py-2.5 px-3">จำนวนรอบ (N)</th>
                    <th className="py-2.5 px-3">Hit Rate (%)</th>
                    <th className="py-2.5 px-3">เฉลี่ย (Mean)</th>
                    <th className="py-2.5 px-3">มัธยฐาน (Median)</th>
                    <th className="py-2.5 px-3 text-emerald-700">ดีที่สุด (Best)</th>
                    <th className="py-2.5 px-3 text-rose-700">แย่ที่สุด (Worst)</th>
                    <th className="py-2.5 px-3">Avg Win</th>
                    <th className="py-2.5 px-3">Avg Loss</th>
                    <th className="py-2.5 px-3 font-extrabold text-sky-800">Risk/Reward</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-zinc-100 font-mono">
                  {[5, 10, 20, 60].map((h) => {
                    const st = studyData.horizon_stats[String(h)]
                    if (!st) return null
                    return (
                      <tr
                        key={h}
                        className={`hover:bg-zinc-50/80 transition-colors ${
                          activeHorizon === h ? 'bg-sky-50/40 font-semibold' : ''
                        }`}
                      >
                        <td className="py-2.5 px-3 font-sans font-bold text-zinc-900">{h} วัน</td>
                        <td className="py-2.5 px-3">{st.sample_size}</td>
                        <td className="py-2.5 px-3">
                          <span
                            className={`px-2 py-0.5 rounded font-bold ${
                              st.hit_rate_pct >= 50
                                ? 'bg-emerald-100 text-emerald-800'
                                : 'bg-rose-100 text-rose-800'
                            }`}
                          >
                            {st.hit_rate_pct.toFixed(1)}%
                          </span>
                        </td>
                        <td className={`py-2.5 px-3 font-bold ${st.avg_return_pct >= 0 ? 'text-emerald-600' : 'text-rose-600'}`}>
                          {st.avg_return_pct > 0 ? '+' : ''}{st.avg_return_pct.toFixed(2)}%
                        </td>
                        <td className="py-2.5 px-3 text-zinc-700">
                          {st.median_return_pct > 0 ? '+' : ''}{st.median_return_pct.toFixed(2)}%
                        </td>
                        <td className="py-2.5 px-3 text-emerald-600">
                          {st.best_return_pct != null ? `+${st.best_return_pct.toFixed(2)}%` : '-'}
                        </td>
                        <td className="py-2.5 px-3 text-rose-600">
                          {st.worst_return_pct != null ? `${st.worst_return_pct.toFixed(2)}%` : '-'}
                        </td>
                        <td className="py-2.5 px-3 text-emerald-700">
                          {st.avg_win_pct != null ? `+${st.avg_win_pct.toFixed(2)}%` : '-'}
                        </td>
                        <td className="py-2.5 px-3 text-rose-700">
                          {st.avg_loss_pct != null ? `${st.avg_loss_pct.toFixed(2)}%` : '-'}
                        </td>
                        <td className="py-2.5 px-3 font-bold text-sky-700">
                          {st.risk_reward_ratio != null ? `${st.risk_reward_ratio.toFixed(2)}x` : '-'}
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          </div>

          {/* 5. Event Instance Breakdown Ledger */}
          <div className="p-4 rounded-xl border border-zinc-200 bg-white shadow-sm space-y-3">
            <h3 className="text-sm font-bold text-zinc-900">
              รายละเอียดเหตุการณ์ในอดีต (Event Instances Ledger)
            </h3>
            <div className="overflow-x-auto">
              <table className="w-full text-xs text-left">
                <thead className="bg-zinc-50 text-zinc-600 font-bold border-b border-zinc-200">
                  <tr>
                    <th className="py-2.5 px-3">วันที่ประกาศ</th>
                    <th className="py-2.5 px-3">ชื่อเหตุการณ์ / หัวข้อ</th>
                    <th className="py-2.5 px-3">วันเข้าซื้อ (t+1)</th>
                    <th className="py-2.5 px-3">ราคาเข้า ({currency})</th>
                    <th className="py-2.5 px-3">5D (%)</th>
                    <th className="py-2.5 px-3">10D (%)</th>
                    <th className="py-2.5 px-3">20D (%)</th>
                    <th className="py-2.5 px-3">60D (%)</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-zinc-100 font-mono">
                  {studyData.events.map((ev) => {
                    const isHovered = hoveredEventId === ev.event_id
                    return (
                      <tr
                        key={ev.event_id}
                        onMouseEnter={() => setHoveredEventId(ev.event_id)}
                        onMouseLeave={() => setHoveredEventId(null)}
                        className={`transition-colors cursor-pointer ${
                          isHovered ? 'bg-sky-50 font-semibold' : 'hover:bg-zinc-50'
                        }`}
                      >
                        <td className="py-2 px-3 text-zinc-900 font-sans">{ev.event_date}</td>
                        <td className="py-2 px-3 text-zinc-700 font-sans">{ev.description || '-'}</td>
                        <td className="py-2 px-3 text-zinc-500 font-sans">{ev.entry_date}</td>
                        <td className="py-2 px-3 text-zinc-900">{currency}{ev.entry_price.toFixed(2)}</td>
                        {[5, 10, 20, 60].map((h) => {
                          const ret = ev.forward_returns[String(h)]
                          if (ret == null) {
                            return <td key={h} className="py-2 px-3 text-zinc-300">-</td>
                          }
                          const isWin = ret >= 0
                          return (
                            <td
                              key={h}
                              className={`py-2 px-3 font-bold ${
                                isWin ? 'text-emerald-600' : 'text-rose-600'
                              }`}
                            >
                              {isWin ? '+' : ''}{ret.toFixed(2)}%
                            </td>
                          )
                        })}
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}
    </div>
  )
}

export default EventStudyTab
