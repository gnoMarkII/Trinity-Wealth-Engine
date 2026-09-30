import React, { useState } from 'react'

export interface HeatmapDay {
  date: string // YYYY-MM-DD
  value: number // PnL or score
  count?: number // number of trades
  label?: string
}

export interface CalendarHeatmapProps {
  data: HeatmapDay[]
  title: string
  subtitle?: string
  unit?: string
  colorScheme?: 'pnl' | 'activity'
  onSelectDate?: (day: HeatmapDay) => void
  className?: string
}

export const CalendarHeatmap: React.FC<CalendarHeatmapProps> = ({
  data,
  title,
  subtitle,
  unit = '$',
  colorScheme = 'pnl',
  onSelectDate,
  className = '',
}) => {
  const [hoveredDay, setHoveredDay] = useState<HeatmapDay | null>(null)

  // Map for fast date lookup
  const dayMap = React.useMemo(() => {
    const map = new Map<string, HeatmapDay>()
    data.forEach((d) => map.set(d.date, d))
    return map
  }, [data])

  // Summary statistics
  const stats = React.useMemo(() => {
    if (!data.length) return { totalPnl: 0, winRate: 0, totalTrades: 0, profitDays: 0 }
    const profitDays = data.filter((d) => d.value > 0).length
    const lossDays = data.filter((d) => d.value < 0).length
    const totalPnl = data.reduce((acc, d) => acc + d.value, 0)
    const totalTrades = data.reduce((acc, d) => acc + (d.count || 0), 0)
    const activeDays = profitDays + lossDays
    const winRate = activeDays > 0 ? (profitDays / activeDays) * 100 : 0
    return { totalPnl, winRate, totalTrades, profitDays }
  }, [data])

  // Build last 12-16 weeks (up to 84-112 days)
  const daysGrid = React.useMemo(() => {
    const days: { dateStr: string; dayOfWeek: number; monthStr: string; item?: HeatmapDay }[] = []
    const now = new Date()
    // Past 84 days (12 weeks)
    const totalDays = 84
    for (let i = totalDays - 1; i >= 0; i--) {
      const d = new Date(now)
      d.setDate(d.getDate() - i)
      const dateStr = d.toISOString().split('T')[0] ?? ''
      const dayOfWeek = d.getDay() // 0 = Sun, 1 = Mon ...
      const monthStr = d.toLocaleDateString('en-US', { month: 'short' })
      days.push({
        dateStr,
        dayOfWeek,
        monthStr,
        item: dayMap.get(dateStr),
      })
    }
    return days
  }, [dayMap])

  // Color resolver
  const getCellBg = (item?: HeatmapDay) => {
    if (!item || item.value === 0) return 'bg-slate-100 hover:border-slate-300'

    if (colorScheme === 'pnl') {
      if (item.value > 0) {
        if (item.value > 2000) return 'bg-emerald-600 hover:ring-2 hover:ring-emerald-400'
        if (item.value > 800) return 'bg-emerald-500 hover:ring-2 hover:ring-emerald-300'
        if (item.value > 200) return 'bg-emerald-400 hover:ring-2 hover:ring-emerald-200'
        return 'bg-emerald-200 hover:ring-2 hover:ring-emerald-100'
      } else {
        if (item.value < -2000) return 'bg-rose-600 hover:ring-2 hover:ring-rose-400'
        if (item.value < -800) return 'bg-rose-500 hover:ring-2 hover:ring-rose-300'
        if (item.value < -200) return 'bg-rose-400 hover:ring-2 hover:ring-rose-200'
        return 'bg-rose-200 hover:ring-2 hover:ring-rose-100'
      }
    } else {
      // Activity / Volume style
      if (item.value > 80) return 'bg-sky-600'
      if (item.value > 40) return 'bg-sky-400'
      return 'bg-sky-200'
    }
  }

  // Group into columns of 7 days
  const columns = React.useMemo(() => {
    const cols: typeof daysGrid[] = []
    let current: typeof daysGrid = []
    daysGrid.forEach((d) => {
      current.push(d)
      if (current.length === 7) {
        cols.push(current)
        current = []
      }
    })
    if (current.length > 0) cols.push(current)
    return cols
  }, [daysGrid])

  return (
    <div
      className={`rounded-2xl border border-sky-100 bg-white/80 p-5 shadow-[0_8px_25px_rgba(14,165,233,0.06)] backdrop-blur-md ${className}`}
    >
      <div className="flex flex-wrap items-center justify-between gap-4 border-b border-sky-100/60 pb-3">
        <div>
          <h3 className="font-semibold text-zinc-900 tracking-tight">{title}</h3>
          {subtitle && <p className="text-xs text-zinc-500">{subtitle}</p>}
        </div>

        {/* Mini stats badges */}
        <div className="flex items-center gap-3">
          <div className="rounded-lg bg-sky-50 px-2.5 py-1 text-right">
            <span className="block text-[10px] uppercase font-bold text-sky-600">Total PnL</span>
            <span
              className={`font-mono text-xs font-bold ${
                stats.totalPnl >= 0 ? 'text-emerald-600' : 'text-rose-600'
              }`}
            >
              {stats.totalPnl >= 0 ? '+' : ''}
              {unit}
              {stats.totalPnl.toLocaleString()}
            </span>
          </div>

          <div className="rounded-lg bg-emerald-50 px-2.5 py-1 text-right">
            <span className="block text-[10px] uppercase font-bold text-emerald-700">Win Rate</span>
            <span className="font-mono text-xs font-bold text-emerald-800">
              {stats.winRate.toFixed(1)}%
            </span>
          </div>

          {stats.totalTrades > 0 && (
            <div className="rounded-lg bg-slate-50 px-2.5 py-1 text-right">
              <span className="block text-[10px] uppercase font-bold text-zinc-500">Trades</span>
              <span className="font-mono text-xs font-bold text-zinc-700">
                {stats.totalTrades}
              </span>
            </div>
          )}
        </div>
      </div>

      {/* Heatmap Grid */}
      <div className="mt-4 overflow-x-auto pb-2">
        <div className="min-w-[500px]">
          {/* Day of week labels + columns */}
          <div className="flex gap-1.5 items-start">
            {/* Day of week indicator */}
            <div className="flex flex-col gap-1.5 pt-1 text-[10px] font-medium text-zinc-400 w-6 shrink-0">
              <span className="h-3 leading-3">M</span>
              <span className="h-3 leading-3">W</span>
              <span className="h-3 leading-3">F</span>
            </div>

            {/* Weeks columns */}
            <div className="flex gap-1.5">
              {columns.map((col, colIdx) => (
                <div key={colIdx} className="flex flex-col gap-1.5">
                  {col.map((cell) => {
                    const isHovered = hoveredDay?.date === cell.dateStr
                    return (
                      <button
                        key={cell.dateStr}
                        type="button"
                        aria-label={`Date: ${cell.dateStr}${cell.item ? `, Value: ${cell.item.value}` : ''}`}
                        onMouseEnter={() => setHoveredDay(cell.item || { date: cell.dateStr, value: 0 })}
                        onMouseLeave={() => setHoveredDay(null)}
                        onClick={() => cell.item && onSelectDate && onSelectDate(cell.item)}
                        className={`h-3.5 w-3.5 rounded-sm transition-all duration-150 ${getCellBg(
                          cell.item
                        )} ${isHovered ? 'scale-125 z-10 shadow-sm ring-2 ring-sky-400' : ''}`}
                      />
                    )
                  })}
                </div>
              ))}
            </div>
          </div>

          {/* Legend and Active Day Detail */}
          <div className="mt-4 flex flex-wrap items-center justify-between gap-2 border-t border-sky-100/60 pt-3 text-xs text-zinc-500">
            {/* Hover tooltip bar */}
            <div className="min-h-[20px]">
              {hoveredDay ? (
                <span className="font-medium text-zinc-800">
                  <span className="font-mono font-semibold text-sky-700 mr-2">{hoveredDay.date}</span>
                  {hoveredDay.value !== 0 ? (
                    <span
                      className={`font-mono font-bold ${
                        hoveredDay.value > 0 ? 'text-emerald-600' : 'text-rose-600'
                      }`}
                    >
                      {hoveredDay.value > 0 ? '+' : ''}
                      {unit}
                      {hoveredDay.value.toLocaleString()}{' '}
                      {hoveredDay.count !== undefined ? `(${hoveredDay.count} trades)` : ''}
                    </span>
                  ) : (
                    <span className="text-zinc-400">No trading activity</span>
                  )}
                </span>
              ) : (
                <span className="text-zinc-400 italic">Hover over any day to inspect performance</span>
              )}
            </div>

            {/* Color Legend */}
            <div className="flex items-center gap-1.5 text-[11px]">
              <span>Loss</span>
              <span className="h-2.5 w-2.5 rounded-sm bg-rose-500" />
              <span className="h-2.5 w-2.5 rounded-sm bg-rose-300" />
              <span className="h-2.5 w-2.5 rounded-sm bg-slate-200" />
              <span className="h-2.5 w-2.5 rounded-sm bg-emerald-300" />
              <span className="h-2.5 w-2.5 rounded-sm bg-emerald-500" />
              <span>Gain</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
