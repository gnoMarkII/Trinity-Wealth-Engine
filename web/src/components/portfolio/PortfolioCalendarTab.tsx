import { useState, useEffect, useMemo } from 'react'
import { useNavigate } from 'react-router-dom'
import { api } from '../../api/client'
import type { CalendarEventDTO, PortfolioCalendarDTO } from '../../api/types'

interface Props {
  selectedPortfolioId?: string
}

export default function PortfolioCalendarTab({ selectedPortfolioId = 'default' }: Props) {
  const navigate = useNavigate()
  const [data, setData] = useState<PortfolioCalendarDTO | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [currentDate, setCurrentDate] = useState(() => new Date())
  const [sidebarOpen, setSidebarOpen] = useState(false)

  useEffect(() => {
    let isMounted = true
    const fetchCalendar = async () => {
      try {
        setLoading(true)
        setError(null)
        const res = await api.getPortfolioCalendar(selectedPortfolioId)
        if (isMounted) {
          setData(res)
        }
      } catch (err: any) {
        if (isMounted) {
          setError(err.message || 'Failed to load calendar events')
        }
      } finally {
        if (isMounted) {
          setLoading(false)
        }
      }
    }
    fetchCalendar()
    return () => {
      isMounted = false
    }
  }, [selectedPortfolioId])

  // Month navigation helpers
  const year = currentDate.getFullYear()
  const month = currentDate.getMonth()

  const monthLabel = currentDate.toLocaleDateString('th-TH', {
    month: 'long',
    year: 'numeric',
  })

  const prevMonth = () => {
    setCurrentDate(new Date(year, month - 1, 1))
  }

  const nextMonth = () => {
    setCurrentDate(new Date(year, month + 1, 1))
  }

  const resetToToday = () => {
    setCurrentDate(new Date())
  }

  // Helper to format Date in local time YYYY-MM-DD (prevents UTC shift bugs)
  const formatLocalDate = (d: Date): string => {
    const y = d.getFullYear()
    const m = String(d.getMonth() + 1).padStart(2, '0')
    const day = String(d.getDate()).padStart(2, '0')
    return `${y}-${m}-${day}`
  }

  // Days grid generation for current month view
  const gridDays = useMemo(() => {
    const firstDayOfMonth = new Date(year, month, 1)
    const lastDayOfMonth = new Date(year, month + 1, 0)

    const startingDayOfWeek = firstDayOfMonth.getDay() // 0 = Sun, 1 = Mon...
    const daysInMonth = lastDayOfMonth.getDate()

    const days: { date: Date; dateStr: string; isCurrentMonth: boolean }[] = []

    // Previous month padding
    const prevMonthLastDay = new Date(year, month, 0).getDate()
    for (let i = startingDayOfWeek - 1; i >= 0; i--) {
      const d = new Date(year, month - 1, prevMonthLastDay - i)
      days.push({ date: d, dateStr: formatLocalDate(d), isCurrentMonth: false })
    }

    // Current month days
    for (let day = 1; day <= daysInMonth; day++) {
      const d = new Date(year, month, day)
      days.push({ date: d, dateStr: formatLocalDate(d), isCurrentMonth: true })
    }

    // Next month padding to complete 35 or 42 cells grid
    const remainingCells = (7 - (days.length % 7)) % 7
    for (let day = 1; day <= remainingCells; day++) {
      const d = new Date(year, month + 1, day)
      days.push({ date: d, dateStr: formatLocalDate(d), isCurrentMonth: false })
    }

    return days
  }, [year, month])

  // Group events by date string "YYYY-MM-DD"
  const eventsByDate = useMemo(() => {
    const map: Record<string, CalendarEventDTO[]> = {}
    if (!data?.events) return map

    for (const evt of data.events) {
      (map[evt.event_date] ??= []).push(evt)
    }
    return map
  }, [data])

  // Upcoming events sorted ascending by days_until
  const upcomingEvents = useMemo(() => {
    if (!data?.events) return []
    return [...data.events]
      .filter((e) => e.days_until >= 0)
      .sort((a, b) => a.days_until - b.days_until)
  }, [data])

  const todayStr = formatLocalDate(new Date())

  if (loading) {
    return (
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <div className="h-8 w-48 bg-slate-200 rounded animate-pulse" />
          <div className="h-8 w-32 bg-slate-200 rounded animate-pulse" />
        </div>
        <div className="grid grid-cols-7 gap-2">
          {Array.from({ length: 35 }).map((_, i) => (
            <div key={i} className="h-28 bg-slate-100 border border-edge rounded-lg p-2 animate-pulse" />
          ))}
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="rounded-xl border border-red-200 bg-red-50 p-6 text-center text-red-700">
        <p className="font-semibold">เกิดข้อผิดพลาดในการโหลดปฏิทิน</p>
        <p className="text-sm mt-1">{error}</p>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Header & Controls */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h2 className="text-lg font-bold text-slate-900 flex items-center gap-2">
            <span>📅 Corporate Events Calendar</span>
          </h2>
          <p className="text-xs text-slate-500 mt-0.5">
            วันประกาศงบ (Earnings Date) และวันขึ้น XD ของหุ้นใน Holdings และ Watchlist
          </p>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={prevMonth}
            className="rounded-lg border border-edge bg-panel px-3 py-1.5 text-xs font-medium text-slate-700 hover:bg-surface"
          >
            ◀ เดือนก่อนหน้า
          </button>

          <span className="min-w-[140px] text-center font-semibold text-sm text-slate-800">
            {monthLabel}
          </span>

          <button
            onClick={nextMonth}
            className="rounded-lg border border-edge bg-panel px-3 py-1.5 text-xs font-medium text-slate-700 hover:bg-surface"
          >
            เดือนถัดไป ▶
          </button>

          <button
            onClick={resetToToday}
            className="rounded-lg bg-sky-600 px-3 py-1.5 text-xs font-medium text-white hover:bg-sky-500 transition-colors"
          >
            เดือนนี้
          </button>
        </div>
      </div>

      {/* Warnings for failed tickers */}
      {data?.tickers_failed && data.tickers_failed.length > 0 && (
        <div className="rounded-lg border border-amber-300 bg-amber-50 p-3 text-xs text-amber-800">
          ⚠️ ไม่สามารถดึงข้อมูล Calendar สำหรับหุ้นบางตัวได้: {data.tickers_failed.join(', ')}
        </div>
      )}

      {/* Legend */}
      <div className="flex flex-wrap items-center gap-4 text-xs">
        <div className="flex items-center gap-1.5">
          <span className="w-3 h-3 rounded bg-violet-100 border border-violet-400" />
          <span className="text-slate-600">Holdings (หุ้นในพอร์ต)</span>
        </div>
        <div className="flex items-center gap-1.5">
          <span className="w-3 h-3 rounded bg-sky-100 border border-sky-400" />
          <span className="text-slate-600">Watchlist</span>
        </div>
        <div className="flex items-center gap-1.5">
          <span>📈</span>
          <span className="text-slate-600">ประกาศงบการเงิน (Earnings)</span>
        </div>
        <div className="flex items-center gap-1.5">
          <span>🪙</span>
          <span className="text-slate-600">ขึ้นเครื่องหมาย XD (Ex-Dividend)</span>
        </div>
        <div className="flex items-center gap-1.5">
          <span className="text-red-500 font-bold">🔔</span>
          <span className="text-slate-600">เตือนเหตุการณ์ภายใน 7 วัน</span>
        </div>
      </div>

      {/* Calendar + Sidebar wrapper */}
      <div className="flex gap-4 items-start">

        {/* ─── Upcoming Events Sidebar ───────────────────────────── */}
        <div
          style={{
            width: sidebarOpen ? 280 : 0,
            minWidth: sidebarOpen ? 280 : 0,
            opacity: sidebarOpen ? 1 : 0,
            overflow: 'hidden',
            transition: 'width 0.28s ease, min-width 0.28s ease, opacity 0.22s ease',
          }}
        >
          <div
            style={{ width: 280 }}
            className="rounded-xl border border-edge bg-panel shadow-sm flex flex-col"
          >
            {/* Sidebar header */}
            <div className="flex items-center justify-between px-3 py-2.5 border-b border-edge bg-surface rounded-t-xl">
              <span className="text-xs font-bold text-slate-700 tracking-wide uppercase">📋 Upcoming Events</span>
              <button
                onClick={() => setSidebarOpen(false)}
                className="text-slate-500 hover:text-slate-700 text-sm leading-none transition-colors"
                title="ปิด Panel"
              >
                ✕
              </button>
            </div>

            {/* Event list */}
            <div className="overflow-y-auto flex-1" style={{ maxHeight: 620 }}>
              {upcomingEvents.length === 0 ? (
                <div className="py-10 text-center text-xs text-slate-500">
                  ไม่มีกิจกรรมที่กำลังจะเกิดขึ้น
                </div>
              ) : (
                <ul className="divide-y divide-edge">
                  {upcomingEvents.map((evt, idx) => {
                    const isHolding = evt.bucket === 'holding'
                    const isToday = evt.days_until === 0
                    const isUrgent = evt.days_until <= 7

                    return (
                      <li key={`sidebar-${evt.ticker}-${evt.event_type}-${idx}`}>
                        <button
                          onClick={() => navigate(`/equity/${evt.ticker}`)}
                          className="w-full text-left px-3 py-2.5 hover:bg-surface transition-colors flex items-start gap-2.5"
                        >
                          {/* Event type icon */}
                          <span className="text-base leading-none mt-0.5 flex-shrink-0">
                            {evt.event_type === 'earnings' ? '📈' : '🪙'}
                          </span>

                          <div className="flex-1 min-w-0">
                            {/* Ticker + type badge */}
                            <div className="flex items-center gap-1.5 flex-wrap">
                              <span className="text-xs font-bold text-slate-900">{evt.ticker}</span>
                              <span
                                className={`text-[11px] px-1.5 py-0.5 rounded-full font-medium ${
                                  isHolding
                                    ? 'bg-violet-100 text-violet-700'
                                    : 'bg-sky-100 text-sky-700'
                                }`}
                              >
                                {isHolding ? 'ถือ' : 'Watch'}
                              </span>
                              <span className="text-[11px] px-1 py-0.5 rounded bg-slate-100 text-slate-600">
                                {evt.event_type === 'earnings' ? 'ประกาศงบ' : 'ขึ้น XD'}
                              </span>
                            </div>

                            {/* Company name (if available) */}
                            {evt.company_name && (
                              <p className="text-[11px] text-slate-500 truncate mt-0.5">{evt.company_name}</p>
                            )}

                            {/* Date + days_until */}
                            <div className="flex items-center gap-1.5 mt-1">
                              <span className="text-[11px] text-slate-500">{evt.event_date}</span>
                              <span
                                className={`text-[11px] font-bold px-1.5 py-0.5 rounded-full ${
                                  isToday
                                    ? 'bg-red-600 text-white'
                                    : isUrgent
                                    ? 'bg-red-100 text-red-700'
                                    : 'bg-slate-100 text-slate-600'
                                }`}
                              >
                                {isToday ? 'วันนี้!' : `อีก ${evt.days_until} วัน`}
                              </span>
                            </div>
                          </div>
                        </button>
                      </li>
                    )
                  })}
                </ul>
              )}
            </div>

            {/* Footer count */}
            <div className="px-3 py-2 border-t border-edge bg-surface rounded-b-xl">
              <p className="text-[11px] text-slate-500 text-center">
                {upcomingEvents.length} กิจกรรม upcoming
              </p>
            </div>
          </div>
        </div>

        {/* ─── Monthly Grid Calendar ──────────────────────────────── */}
        <div className="flex-1 min-w-0 rounded-xl border border-edge bg-panel shadow-sm overflow-hidden">
          {/* Toolbar: Toggle button */}
          <div className="flex items-center gap-2 px-2 py-1.5 border-b border-edge bg-surface">
            <button
              id="calendar-sidebar-toggle"
              onClick={() => setSidebarOpen((v) => !v)}
              title={sidebarOpen ? 'ปิด Upcoming Events' : 'เปิด Upcoming Events'}
              className={`flex items-center gap-1.5 rounded-lg px-2 py-1 text-xs font-medium transition-colors ${
                sidebarOpen
                  ? 'bg-sky-100 text-sky-700'
                  : 'text-slate-500 hover:bg-slate-100 hover:text-slate-700'
              }`}
            >
              <span>{sidebarOpen ? '◀' : '☰'}</span>
              <span>{sidebarOpen ? 'ซ่อน Upcoming' : 'Upcoming Events'}</span>
              {!sidebarOpen && upcomingEvents.length > 0 && (
                <span className="bg-red-500 text-white text-[11px] font-bold rounded-full px-1.5 py-0.5">
                  {upcomingEvents.length}
                </span>
              )}
            </button>
          </div>

          {/* Days Header */}
          <div className="grid grid-cols-7 border-b border-edge bg-surface text-center text-xs font-semibold text-slate-500 py-2.5">
            {['อา.', 'จ.', 'อ.', 'พ.', 'พฤ.', 'ศ.', 'ส.'].map((d) => (
              <div key={d}>{d}</div>
            ))}
          </div>

          {/* Days Grid */}
          <div className="grid grid-cols-7 divide-x divide-y divide-edge">
            {gridDays.map(({ date: dayDate, dateStr, isCurrentMonth }, cellIdx) => {
              const isToday = dateStr === todayStr
              const cellEvents = eventsByDate[dateStr] || []

              return (
                <div
                  key={`${dateStr}-${cellIdx}`}
                  className={`min-h-[110px] p-1.5 flex flex-col justify-start transition-colors ${
                    !isCurrentMonth
                      ? 'bg-slate-50/50 text-slate-500'
                      : 'bg-panel text-slate-800'
                  } ${isToday ? 'ring-2 ring-sky-500 ring-inset z-10' : ''}`}
                >
                  <div className="flex items-center justify-between mb-1">
                    <span
                      className={`text-xs font-semibold px-1.5 py-0.5 rounded-full ${
                        isToday ? 'bg-sky-600 text-white' : ''
                      }`}
                    >
                      {dayDate.getDate()}
                    </span>
                    {cellEvents.length > 0 && (
                      <span className="text-[11px] text-slate-500 font-medium">
                        {cellEvents.length} กิจกรรม
                      </span>
                    )}
                  </div>

                  {/* Event Chips */}
                  <div className="space-y-1 overflow-y-auto max-h-[120px]">
                    {cellEvents.map((evt, idx) => {
                      const isHolding = evt.bucket === 'holding'
                      const isUpcoming = evt.days_until >= 0 && evt.days_until <= 7

                      const tooltipText = [
                        `${evt.ticker} - ${evt.event_type === 'earnings' ? 'ประกาศงบการเงิน' : 'วันขึ้น XD (ปันผล)'}`,
                        evt.company_name ? `บริษัท: ${evt.company_name}` : null,
                        evt.eps_estimate !== null && evt.eps_estimate !== undefined
                          ? `EPS Est: $${evt.eps_estimate} (Low: $${evt.eps_low ?? '-'}, High: $${evt.eps_high ?? '-'})`
                          : null,
                        `วันทำการ: ${evt.event_date} (${evt.days_until === 0 ? 'วันนี้!' : `${evt.days_until} วัน`})`,
                      ]
                        .filter(Boolean)
                        .join('\n')

                      return (
                        <button
                          key={`${evt.ticker}-${evt.event_type}-${idx}`}
                          title={tooltipText}
                          onClick={() => navigate(`/equity/${evt.ticker}`)}
                          className={`w-full text-left rounded px-1.5 py-1 text-[11px] font-medium border transition-transform hover:scale-[1.02] flex items-center justify-between gap-1 ${
                            isHolding
                              ? 'bg-violet-50 text-violet-900 border-violet-200 hover:bg-violet-100'
                              : 'bg-sky-50 text-sky-900 border-sky-200 hover:bg-sky-100'
                          }`}
                        >
                          <span className="truncate">
                            {evt.event_type === 'earnings' ? '📈' : '🪙'} <strong>{evt.ticker}</strong>{' '}
                            <span className="text-[11px] font-normal opacity-80">
                              ({evt.event_type === 'earnings' ? 'ประกาศงบ' : 'ขึ้น XD'})
                            </span>
                          </span>

                          {isUpcoming && (
                            <span className="text-[11px] font-bold text-red-600 bg-red-100 px-1 rounded flex-shrink-0">
                              {evt.days_until === 0 ? 'วันนี้' : `อีก ${evt.days_until}d`}
                            </span>
                          )}
                        </button>
                      )
                    })}
                  </div>
                </div>
              )
            })}
          </div>
        </div>
      </div>
    </div>
  )
}
