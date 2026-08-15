import { useState, useMemo } from 'react'
import type { ActualPortfolioStateDTO, DividendRoundDTO, SyncDividendsResponseDTO } from '../../api/types'
import { api } from '../../api/client'
import { formatTHB } from '../../utils/formatters'
import { IncomeIcon } from './icons/PortfolioIcons'

interface Props {
  state: ActualPortfolioStateDTO | null
  selectedPortfolioId: string
  onSuccess: (state: ActualPortfolioStateDTO) => void
  onOpenIncomeModal?: () => void
}

function getDaysUntil(dateStr: string | null | undefined): { text: string; isPast: boolean; isToday: boolean } {
  if (!dateStr) return { text: '—', isPast: false, isToday: false }
  try {
    const target = new Date(dateStr.slice(0, 10))
    const today = new Date()
    today.setHours(0, 0, 0, 0)
    target.setHours(0, 0, 0, 0)
    const diffTime = target.getTime() - today.getTime()
    const diffDays = Math.round(diffTime / (1000 * 60 * 60 * 24))
    if (diffDays === 0) return { text: 'วันนี้ 🔔', isPast: false, isToday: true }
    if (diffDays > 0) return { text: `อีก ${diffDays} วัน`, isPast: false, isToday: false }
    return { text: `ผ่านมาแล้ว ${Math.abs(diffDays)} วัน`, isPast: true, isToday: false }
  } catch {
    return { text: dateStr, isPast: false, isToday: false }
  }
}

export default function PortfolioIncomesTab({
  state,
  selectedPortfolioId,
  onSuccess,
  onOpenIncomeModal,
}: Props) {
  const [activeSubTab, setActiveSubTab] = useState<'received' | 'upcoming'>('received')
  const [syncing, setSyncing] = useState(false)
  const [syncError, setSyncError] = useState<string | null>(null)
  const [syncResult, setSyncResult] = useState<SyncDividendsResponseDTO | null>(null)
  const [searchQuery, setSearchQuery] = useState('')
  const [sourceFilter, setSourceFilter] = useState<'ALL' | 'synced' | 'manual' | 'none'>('ALL')
  const [selectedSymbolRounds, setSelectedSymbolRounds] = useState<{
    symbol: string
    rounds: DividendRoundDTO[]
  } | null>(null)

  const stockHoldings = useMemo(
    () => (state?.holdings || []).filter((h) => h.asset_type !== 'Cash'),
    [state?.holdings]
  )

  const totalAccDividend = state?.summary.total_accumulated_dividend ?? 0

  const handleSyncDividends = async () => {
    setSyncing(true)
    setSyncError(null)
    try {
      const res = await api.syncDividends(selectedPortfolioId)
      setSyncResult(res)
      // Reload updated portfolio state
      const freshState = await api.getActualPortfolioState(false, false, selectedPortfolioId)
      onSuccess(freshState)
    } catch (err: any) {
      setSyncError(err?.message || 'เกิดข้อผิดพลาดในการซิงค์ข้อมูลเงินปันผล')
    } finally {
      setSyncing(false)
    }
  }

  // Aggregate all upcoming rounds across holdings
  const allUpcomingRounds = useMemo(() => {
    const list: Array<DividendRoundDTO & { company_name?: string | null }> = []
    for (const h of stockHoldings) {
      const rounds = syncResult?.details?.[h.symbol] || h.dividend_rounds || []
      for (const r of rounds) {
        if (r.status === 'upcoming') {
          list.push({ ...r, company_name: h.company_name })
        }
      }
    }
    // Sort chronological: upcoming earliest date first
    list.sort((a, b) => {
      const dateA = a.pay_date || a.ex_date
      const dateB = b.pay_date || b.ex_date
      return dateA.localeCompare(dateB)
    })
    return list
  }, [stockHoldings, syncResult])

  const totalUpcomingThb = useMemo(() => {
    return allUpcomingRounds.reduce((acc, r) => acc + (r.net_thb || 0), 0)
  }, [allUpcomingRounds])

  const totalUpcomingUsd = useMemo(() => {
    return allUpcomingRounds
      .filter((r) => r.currency === 'USD')
      .reduce((acc, r) => acc + (r.net_native || 0), 0)
  }, [allUpcomingRounds])

  // Filtered holdings list for Received tab
  const filteredHoldings = useMemo(() => {
    return stockHoldings
      .filter((h) => {
        if (!searchQuery.trim()) return true
        const q = searchQuery.trim().toUpperCase()
        return (
          h.symbol.toUpperCase().includes(q) ||
          (h.company_name && h.company_name.toUpperCase().includes(q))
        )
      })
      .filter((h) => {
        if (sourceFilter === 'ALL') return true
        if (sourceFilter === 'synced') return h.dividend_source === 'synced'
        if (sourceFilter === 'manual') return h.dividend_source === 'manual'
        if (sourceFilter === 'none') return !h.dividend_source
        return true
      })
  }, [stockHoldings, searchQuery, sourceFilter])

  // Filtered upcoming rounds for Upcoming tab
  const filteredUpcomingRounds = useMemo(() => {
    if (!searchQuery.trim()) return allUpcomingRounds
    const q = searchQuery.trim().toUpperCase()
    return allUpcomingRounds.filter(
      (r) =>
        r.symbol.toUpperCase().includes(q) ||
        (r.company_name && r.company_name.toUpperCase().includes(q))
    )
  }, [allUpcomingRounds, searchQuery])

  // Count stats
  const syncedCount = stockHoldings.filter((h) => h.dividend_source === 'synced').length
  const manualCount = stockHoldings.filter((h) => h.dividend_source === 'manual').length
  const unsyncedCount = stockHoldings.filter((h) => !h.dividend_source).length

  // Calculate total USD received dividend across holdings
  const totalUsdReceived = useMemo(() => {
    return stockHoldings
      .filter((h) => h.avg_cost_usd !== null || h.current_price_usd !== null)
      .reduce((acc, h) => acc + (h.accumulated_dividend_native || 0), 0)
  }, [stockHoldings])

  return (
    <div className="space-y-6">
      {/* Top Banner / Sync Controls */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 rounded-2xl border border-sky-200 bg-gradient-to-r from-sky-50 via-white to-cyan-50/60 p-5 sm:p-6 shadow-sm">
        <div className="flex items-center gap-3.5">
          <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-flow-blue/10 border border-flow-blue/20 text-flow-blue shadow-inner">
            <IncomeIcon className="h-6 w-6" />
          </div>
          <div>
            <h2 className="text-base sm:text-lg font-bold text-zinc-900 tracking-tight flex items-center gap-2">
              ประวัติเงินปันผลและรายรับ (Dividends & Passive Income)
            </h2>
            <p className="text-xs text-zinc-500">
              แยกส่วนได้รับแล้วสะสม และรอรับเงินปันผล พร้อมคำนวณสกุลเงินดั้งเดิมอัตโนมัติ
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2.5">
          {onOpenIncomeModal && (
            <button
              type="button"
              onClick={onOpenIncomeModal}
              className="inline-flex items-center justify-center gap-1.5 rounded-xl border border-sky-200 bg-white hover:bg-sky-50 px-3.5 py-2 text-xs font-semibold text-sky-800 shadow-sm transition-all cursor-pointer"
            >
              <span>✏️</span> บันทึกปันผลเอง
            </button>
          )}

          <button
            type="button"
            onClick={handleSyncDividends}
            disabled={syncing}
            className="inline-flex items-center justify-center gap-2 rounded-xl bg-flow-blue hover:bg-sky-600 px-4 py-2 text-xs font-bold text-white shadow-sm hover:shadow transition-all active:scale-[0.98] disabled:opacity-50 cursor-pointer"
          >
            <span className={syncing ? 'animate-spin' : ''}>🔄</span>
            <span>{syncing ? 'กำลังซิงค์ประวัติปันผล...' : 'ซิงค์เงินปันผลอัตโนมัติ'}</span>
          </button>
        </div>
      </div>

      {/* Sync Error Alert */}
      {syncError && (
        <div className="rounded-xl border border-rose-200 bg-rose-50 p-4 text-xs text-rose-900 flex items-center justify-between shadow-sm animate-fade-in">
          <div className="flex items-center gap-2">
            <span>🚨</span>
            <span className="font-medium">{syncError}</span>
          </div>
          <button
            type="button"
            onClick={() => setSyncError(null)}
            className="text-rose-500 hover:text-rose-700 font-bold cursor-pointer"
          >
            ✕
          </button>
        </div>
      )}

      {/* Sync Result Banner */}
      {syncResult && (
        <div className="rounded-xl border border-emerald-200 bg-emerald-50/90 p-4 space-y-2 text-xs shadow-sm animate-fade-in">
          <div className="flex items-center justify-between text-emerald-900 font-semibold">
            <div className="flex items-center gap-2">
              <span className="text-base">✅</span>
              <span>
                ซิงค์สำเร็จ {syncResult.synced_symbols} สินทรัพย์ รวม {syncResult.total_rounds} รอบการจ่าย
                — ได้รับแล้ว {formatTHB(syncResult.total_dividend_thb)}
                {syncResult.total_upcoming_thb ? ` • รอจ่าย ${formatTHB(syncResult.total_upcoming_thb)}` : ''}
              </span>
            </div>
            <button
              type="button"
              onClick={() => setSyncResult(null)}
              className="text-emerald-700 hover:text-emerald-900 font-bold cursor-pointer"
            >
              ✕
            </button>
          </div>

          {syncResult.skipped_manual.length > 0 && (
            <div className="flex items-center gap-2 text-amber-900 bg-amber-50 border border-amber-200 rounded-lg px-3 py-1.5">
              <span>⚠️</span>
              <span>
                ข้าม {syncResult.skipped_manual.join(', ')} เนื่องจากเคยถูกแก้ไขแบบกำหนดเอง (Manual Protection)
              </span>
            </div>
          )}
        </div>
      )}

      {/* SUB-TABS SWITCHER */}
      <div className="flex items-center gap-3 border-b border-sky-100 pb-2">
        <button
          type="button"
          onClick={() => setActiveSubTab('received')}
          className={`flex items-center gap-2.5 rounded-xl px-5 py-3 text-sm sm:text-base font-bold transition-all cursor-pointer shadow-sm ${
            activeSubTab === 'received'
              ? 'bg-flow-blue text-white shadow-md'
              : 'border border-sky-200 bg-white text-zinc-600 hover:bg-sky-50'
          }`}
        >
          <span>💰</span>
          <span>ได้รับแล้ว (Received)</span>
          <span
            className={`rounded-full px-2.5 py-0.5 text-xs font-mono font-bold ${
              activeSubTab === 'received' ? 'bg-white/20 text-white' : 'bg-sky-100 text-sky-800'
            }`}
          >
            {formatTHB(totalAccDividend)}
          </span>
        </button>

        <button
          type="button"
          onClick={() => setActiveSubTab('upcoming')}
          className={`flex items-center gap-2.5 rounded-xl px-5 py-3 text-sm sm:text-base font-bold transition-all cursor-pointer shadow-sm ${
            activeSubTab === 'upcoming'
              ? 'bg-flow-blue text-white shadow-md'
              : 'border border-sky-200 bg-white text-zinc-600 hover:bg-sky-50'
          }`}
        >
          <span>🕒</span>
          <span>รอรับเงิน (Upcoming)</span>
          <span
            className={`rounded-full px-2.5 py-0.5 text-xs font-mono font-bold ${
              activeSubTab === 'upcoming' ? 'bg-white/20 text-white' : 'bg-amber-100 text-amber-900'
            }`}
          >
            {allUpcomingRounds.length} รอบ ({formatTHB(totalUpcomingThb)})
          </span>
        </button>
      </div>

      {/* SUBTAB 1: RECEIVED DIVIDENDS */}
      {activeSubTab === 'received' && (
        <div className="space-y-6 animate-fade-in">
          {/* KPI Summary Cards */}
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            {/* Card 1: Total Received */}
            <div className="flex flex-col justify-between rounded-2xl border border-sky-100 bg-gradient-to-br from-panel via-panel to-sky-50/40 p-5 sm:p-6 shadow-sm transition-all hover:shadow-md relative overflow-hidden">
              <div className="flex items-center justify-between text-xs sm:text-sm font-bold uppercase tracking-wider text-zinc-500">
                <span>เงินปันผลสะสมทั้งหมด (Total Accumulated)</span>
                <span className="rounded-full bg-emerald-50 border border-emerald-200 px-2.5 py-0.5 text-xs font-bold text-emerald-700">NET RECEIVED</span>
              </div>
              <div className="mt-2.5 text-2xl sm:text-3xl lg:text-4xl font-extrabold font-mono text-emerald-600 flex items-baseline gap-2">
                <span>{formatTHB(totalAccDividend)}</span>
                {totalUsdReceived > 0 && (
                  <span className="text-sm font-bold text-zinc-500 font-sans">
                    (${totalUsdReceived.toFixed(2)} USD)
                  </span>
                )}
              </div>
              <div className="mt-1.5 text-xs text-zinc-500 font-medium">
                คำนวณสุทธิหลังหักภาษี ณ ที่จ่าย (WHT 10% / 15%)
              </div>
            </div>

            {/* Card 2: Sync Status */}
            <div className="flex flex-col justify-between rounded-2xl border border-sky-100 bg-gradient-to-br from-panel via-panel to-sky-50/40 p-5 sm:p-6 shadow-sm transition-all hover:shadow-md relative overflow-hidden">
              <div className="flex items-center justify-between text-xs sm:text-sm font-bold uppercase tracking-wider text-zinc-500">
                <span>สถานะการเชื่อมต่อ (Sync Status)</span>
                <span className="rounded-full bg-sky-50 border border-sky-200 px-2.5 py-0.5 text-xs font-bold text-flow-blue">HOLDINGS</span>
              </div>
              <div className="mt-2.5 flex items-baseline gap-2">
                <span className="text-2xl sm:text-3xl lg:text-4xl font-extrabold font-mono text-zinc-900">{syncedCount}</span>
                <span className="text-sm font-semibold text-zinc-500">/ {stockHoldings.length} สินทรัพย์</span>
              </div>
              <div className="mt-1.5 flex items-center gap-2 text-xs font-medium">
                <span className="text-emerald-700 font-bold bg-emerald-50 px-2 py-0.5 rounded-md border border-emerald-200">🤖 {syncedCount} Synced</span>
                <span className="text-zinc-400">•</span>
                <span className="text-amber-700 font-bold bg-amber-50 px-2 py-0.5 rounded-md border border-amber-200">✏️ {manualCount} Manual</span>
                {unsyncedCount > 0 && (
                  <>
                    <span className="text-zinc-400">•</span>
                    <span className="text-zinc-500">{unsyncedCount} ยังไม่ซิงค์</span>
                  </>
                )}
              </div>
            </div>

            {/* Card 3: Passive Income YTD */}
            <div className="flex flex-col justify-between rounded-2xl border border-sky-100 bg-gradient-to-br from-panel via-panel to-sky-50/40 p-5 sm:p-6 shadow-sm transition-all hover:shadow-md relative overflow-hidden">
              <div className="flex items-center justify-between text-xs sm:text-sm font-bold uppercase tracking-wider text-zinc-500">
                <span>Passive Income YTD</span>
                <span className="rounded-full bg-cyan-50 border border-cyan-200 px-2.5 py-0.5 text-xs font-bold text-cyan-700">YTD</span>
              </div>
              <div className="mt-2.5 text-2xl sm:text-3xl lg:text-4xl font-extrabold font-mono text-flow-blue">
                {formatTHB(state?.summary.passive_income_ytd ?? 0)}
              </div>
              <div className="mt-1 text-[11px] text-zinc-500">
                รายรับเงินปันผลและดอกเบี้ยตั้งแต่ต้นปี
              </div>
            </div>
          </div>

          {/* Table Filter Bar */}
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pt-2">
            <div className="flex items-center gap-2">
              <div className="relative">
                <input
                  type="text"
                  placeholder="🔍 ค้นหาสัญลักษณ์หุ้น..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="w-64 rounded-xl border border-sky-200 bg-white px-3.5 py-2 text-xs text-zinc-900 placeholder-zinc-400 focus:border-flow-blue focus:outline-none focus:ring-1 focus:ring-flow-blue shadow-sm"
                />
              </div>
              {searchQuery && (
                <button
                  type="button"
                  onClick={() => setSearchQuery('')}
                  className="rounded-lg border border-sky-200 bg-white px-2.5 py-1.5 text-xs text-zinc-500 hover:text-zinc-900 shadow-sm cursor-pointer"
                >
                  ล้าง
                </button>
              )}
            </div>

            <div className="flex items-center gap-1.5">
              {(['ALL', 'synced', 'manual', 'none'] as const).map((filterKey) => {
                const labels: Record<string, string> = {
                  ALL: 'ทั้งหมด',
                  synced: '🤖 Synced',
                  manual: '✏️ Manual',
                  none: 'ยังไม่ซิงค์',
                }
                const isActive = sourceFilter === filterKey
                return (
                  <button
                    key={filterKey}
                    type="button"
                    onClick={() => setSourceFilter(filterKey)}
                    className={`rounded-xl px-3 py-1.5 text-xs font-semibold transition-all cursor-pointer shadow-sm ${
                      isActive
                        ? 'bg-flow-blue text-white shadow-md'
                        : 'border border-sky-200 bg-white text-zinc-600 hover:bg-sky-50'
                    }`}
                  >
                    {labels[filterKey]}
                  </button>
                )
              })}
            </div>
          </div>

          {/* Holdings Received Dividend Table */}
          <div className="rounded-2xl border border-sky-100 bg-panel shadow-sm overflow-hidden">
            <div className="overflow-x-auto">
              <table className="w-full text-left text-sm min-w-[950px]">
                <thead className="border-b border-sky-100 bg-sky-50/80 text-xs sm:text-sm font-bold uppercase tracking-wider text-zinc-700">
                  <tr>
                    <th className="px-4 py-3.5">สัญลักษณ์</th>
                    <th className="px-3 py-3.5">ประเภท</th>
                    <th className="px-3 py-3.5 text-right">จำนวนหน่วย (Units)</th>
                    <th className="px-3 py-3.5 text-right">ปันผลต่อหุ้น (DPS)</th>
                    <th className="px-3 py-3.5 text-right">Dividend Yield</th>
                    <th className="px-3 py-3.5 text-right font-bold text-emerald-700">เงินปันผลที่ได้รับ (Net Received)</th>
                    <th className="px-3 py-3.5 text-center">สถานะที่มา (Source)</th>
                    <th className="px-4 py-3.5 text-center">รายละเอียด</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-sky-100/60 font-sans">
                  {filteredHoldings.length === 0 ? (
                    <tr>
                      <td colSpan={8} className="py-12 text-center text-zinc-500 font-medium">
                        ไม่พบข้อมูลสินทรัพย์ตามเงื่อนไขที่ค้นหา
                      </td>
                    </tr>
                  ) : (
                    filteredHoldings.map((h, idx) => {
                      const symbolRounds = syncResult?.details?.[h.symbol] || h.dividend_rounds || []
                      const hasRounds = symbolRounds.length > 0
                      const isUSD = h.avg_cost_usd !== null || h.current_price_usd !== null

                      return (
                        <tr key={`${h.symbol}-${idx}`} className="hover:bg-sky-50/50 transition-colors">
                          <td className="px-4 py-3.5 font-sans">
                            <div className="font-extrabold text-zinc-900 flex items-center gap-2 text-base font-mono">
                              <span>{h.symbol}</span>
                              {h.company_name && (
                                <span className="text-xs text-zinc-500 font-sans font-medium truncate max-w-[160px]">
                                  {h.company_name}
                                </span>
                              )}
                            </div>
                          </td>

                          <td className="px-3 py-3.5 font-sans">
                            <span className="rounded-md bg-sky-50 border border-sky-100 px-2 py-0.5 text-xs font-bold text-sky-800">
                              {h.asset_type}
                            </span>
                          </td>

                          <td className="px-3 py-3.5 text-right font-mono font-semibold text-zinc-900 text-sm sm:text-base">
                            {h.units.toLocaleString('en-US', { minimumFractionDigits: 2 })}
                          </td>

                          <td className="px-3 py-3.5 text-right font-mono font-semibold text-zinc-800 text-sm sm:text-base">
                            {h.dividend_per_share !== null && h.dividend_per_share !== undefined
                              ? isUSD
                                ? `$${h.dividend_per_share.toFixed(2)}`
                                : `฿${h.dividend_per_share.toFixed(2)}`
                              : '-'}
                          </td>

                          <td className="px-3 py-3.5 text-right font-mono font-semibold text-zinc-800 text-sm sm:text-base">
                            {h.dividend_yield !== null && h.dividend_yield !== undefined
                              ? `${(h.dividend_yield * 100).toFixed(2)}%`
                              : '-'}
                          </td>

                          <td className="px-3 py-3.5 text-right font-mono">
                            {isUSD ? (
                              <div className="flex flex-col items-end">
                                <span className="font-extrabold text-emerald-600 text-sm sm:text-base">
                                  ${(h.accumulated_dividend_native ?? 0).toFixed(2)}
                                </span>
                                <span className="text-xs text-zinc-500 font-sans font-medium">
                                  ({formatTHB(h.accumulated_dividend_thb ?? 0)})
                                </span>
                              </div>
                            ) : (
                              <span className="font-extrabold text-emerald-600 text-sm sm:text-base">
                                {formatTHB(h.accumulated_dividend_thb ?? 0)}
                              </span>
                            )}
                          </td>

                          <td className="px-3 py-3.5 text-center font-sans">
                            {h.dividend_source === 'synced' ? (
                              <span className="inline-flex items-center gap-1 rounded-full bg-emerald-50 border border-emerald-200 px-2.5 py-1 text-xs font-bold text-emerald-700">
                                <span>🤖</span> Synced
                              </span>
                            ) : h.dividend_source === 'manual' ? (
                              <span
                                title="สินทรัพย์นี้ถูกแก้ไขแบบ Manual — ระบบจะไม่ทับค่าอัตโนมัติ"
                                className="inline-flex items-center gap-1 rounded-full bg-amber-50 border border-amber-200 px-2.5 py-1 text-xs font-bold text-amber-700"
                              >
                                <span>✏️</span> Manual
                              </span>
                            ) : (
                              <span className="inline-flex items-center rounded-full bg-zinc-100 border border-zinc-200 px-2.5 py-1 text-xs font-medium text-zinc-500">
                                — Not Synced
                              </span>
                            )}
                          </td>

                          <td className="px-4 py-3.5 text-center font-sans">
                            {hasRounds ? (
                              <button
                                type="button"
                                onClick={() =>
                                  setSelectedSymbolRounds({
                                    symbol: h.symbol,
                                    rounds: symbolRounds,
                                  })
                                }
                                className="rounded-xl border border-sky-200 bg-white hover:bg-sky-50 px-3 py-1.5 text-xs sm:text-sm font-bold text-sky-800 shadow-sm transition-all cursor-pointer hover:scale-105"
                              >
                                🔍 ดู {symbolRounds.length} รอบ
                              </button>
                            ) : (
                              <span className="text-xs text-zinc-400 font-medium">—</span>
                            )}
                          </td>
                        </tr>
                      )
                    })
                  )}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {/* SUBTAB 2: UPCOMING DIVIDENDS */}
      {activeSubTab === 'upcoming' && (
        <div className="space-y-6 animate-fade-in">
          {/* Upcoming KPI Cards */}
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            {/* Card 1: Total Upcoming */}
            <div className="flex flex-col justify-between rounded-2xl border border-amber-100 bg-gradient-to-br from-panel via-panel to-amber-50/40 p-5 sm:p-6 shadow-sm transition-all hover:shadow-md relative overflow-hidden">
              <div className="flex items-center justify-between text-xs sm:text-sm font-bold uppercase tracking-wider text-zinc-500">
                <span>เงินปันผลที่รอรับทั้งหมด (Total Upcoming)</span>
                <span className="rounded-full bg-amber-50 border border-amber-200 px-2.5 py-0.5 text-xs font-bold text-amber-800">PENDING PAY</span>
              </div>
              <div className="mt-2.5 text-2xl sm:text-3xl lg:text-4xl font-extrabold font-mono text-amber-600 flex items-baseline gap-2">
                <span>{formatTHB(totalUpcomingThb)}</span>
                {totalUpcomingUsd > 0 && (
                  <span className="text-sm font-bold text-zinc-500 font-sans">
                    (${totalUpcomingUsd.toFixed(2)} USD)
                  </span>
                )}
              </div>
              <div className="mt-1.5 text-xs text-zinc-500 font-medium">
                เงินจะถูกบันทึกเป็น "ได้รับแล้ว" อัตโนมัติเมื่อถึงวันจ่ายเงิน (Payment Date)
              </div>
            </div>

            {/* Card 2: Upcoming Rounds Count */}
            <div className="flex flex-col justify-between rounded-2xl border border-sky-100 bg-gradient-to-br from-panel via-panel to-sky-50/40 p-5 sm:p-6 shadow-sm transition-all hover:shadow-md relative overflow-hidden">
              <div className="flex items-center justify-between text-xs sm:text-sm font-bold uppercase tracking-wider text-zinc-500">
                <span>รอบการจ่ายที่รออยู่ (Upcoming Rounds)</span>
                <span className="rounded-full bg-sky-50 border border-sky-200 px-2.5 py-0.5 text-xs font-bold text-flow-blue">SCHEDULE</span>
              </div>
              <div className="mt-2.5 text-2xl sm:text-3xl lg:text-4xl font-extrabold font-mono text-zinc-900">
                {allUpcomingRounds.length} <span className="text-sm font-semibold text-zinc-500">รายการ</span>
              </div>
              <div className="mt-1.5 text-xs text-zinc-500 font-medium">
                ตามการขึ้นเครื่องหมาย XD และปฏิทินจ่ายเงิน
              </div>
            </div>

            {/* Card 3: Next Payout */}
            <div className="flex flex-col justify-between rounded-2xl border border-sky-100 bg-gradient-to-br from-panel via-panel to-sky-50/40 p-5 sm:p-6 shadow-sm transition-all hover:shadow-md relative overflow-hidden">
              <div className="flex items-center justify-between text-xs sm:text-sm font-bold uppercase tracking-wider text-zinc-500">
                <span>กำหนดจ่ายถัดไป (Next Payout)</span>
                <span className="rounded-full bg-cyan-50 border border-cyan-200 px-2.5 py-0.5 text-xs font-bold text-cyan-700">NEXT</span>
              </div>
              <div className="mt-2.5 text-xl sm:text-2xl font-extrabold text-flow-blue">
                {allUpcomingRounds.length > 0 && allUpcomingRounds[0]
                  ? `${allUpcomingRounds[0].pay_date || allUpcomingRounds[0].ex_date} (${allUpcomingRounds[0].symbol})`
                  : 'ไม่มีรายการรอจ่าย'}
              </div>
              <div className="mt-1.5 text-xs text-zinc-600 font-mono font-medium">
                {allUpcomingRounds.length > 0 && allUpcomingRounds[0]
                  ? getDaysUntil(allUpcomingRounds[0].pay_date || allUpcomingRounds[0].ex_date).text
                  : 'ทุกรอบถูกจ่ายครบแล้ว'}
              </div>
            </div>
          </div>

          {/* Table Filter Bar for Upcoming */}
          <div className="flex items-center gap-2 pt-2">
            <input
              type="text"
              placeholder="🔍 ค้นหาสัญลักษณ์หุ้น..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-64 rounded-xl border border-sky-200 bg-white px-3.5 py-2 text-xs sm:text-sm text-zinc-900 placeholder-zinc-400 focus:border-flow-blue focus:outline-none focus:ring-1 focus:ring-flow-blue shadow-sm"
            />
            {searchQuery && (
              <button
                type="button"
                onClick={() => setSearchQuery('')}
                className="rounded-xl border border-sky-200 bg-white px-3 py-2 text-xs sm:text-sm font-bold text-zinc-600 hover:text-zinc-900 shadow-sm cursor-pointer"
              >
                ล้าง
              </button>
            )}
          </div>

          {/* Upcoming Schedule Table */}
          <div className="rounded-2xl border border-amber-100 bg-panel shadow-sm overflow-hidden">
            <div className="overflow-x-auto">
              <table className="w-full text-left text-sm min-w-[950px]">
                <thead className="border-b border-amber-100 bg-amber-50/80 text-xs sm:text-sm font-bold uppercase tracking-wider text-zinc-700">
                  <tr>
                    <th className="px-4 py-3.5">สัญลักษณ์</th>
                    <th className="px-3 py-3.5">วัน XD (Ex-Date)</th>
                    <th className="px-3 py-3.5">วันจ่ายเงินจริง (Pay Date)</th>
                    <th className="px-3 py-3.5 text-center">นับถอยหลัง</th>
                    <th className="px-3 py-3.5 text-right">จำนวนหน่วย</th>
                    <th className="px-3 py-3.5 text-right">DPS (ปันผล/หุ้น)</th>
                    <th className="px-3 py-3.5 text-right">WHT</th>
                    <th className="px-4 py-3.5 text-right font-bold text-amber-700">สุทธิคาดการณ์ (Est. Net)</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-amber-100/60 font-sans">
                  {filteredUpcomingRounds.length === 0 ? (
                    <tr>
                      <td colSpan={8} className="py-12 text-center text-zinc-500 font-medium">
                        🎉 ไม่มีรายการเงินปันผลที่รอจ่ายในขณะนี้
                      </td>
                    </tr>
                  ) : (
                    filteredUpcomingRounds.map((r, idx) => {
                      const isUSD = r.currency === 'USD'
                      const countdown = getDaysUntil(r.pay_date || r.ex_date)

                      return (
                        <tr key={`upcoming-${r.symbol}-${r.ex_date}-${idx}`} className="hover:bg-amber-50/40 transition-colors">
                          <td className="px-4 py-3.5 font-sans">
                            <div className="font-extrabold text-zinc-900 flex items-center gap-2 text-base font-mono">
                              <span>{r.symbol}</span>
                              {r.company_name && (
                                <span className="text-xs text-zinc-500 font-sans font-medium truncate max-w-[160px]">
                                  {r.company_name}
                                </span>
                              )}
                            </div>
                          </td>

                          <td className="px-3 py-3.5 font-mono font-semibold text-zinc-700 text-sm sm:text-base">{r.ex_date}</td>

                          <td className="px-3 py-3.5 font-mono font-bold text-zinc-900 text-sm sm:text-base">
                            {r.pay_date || '—'}
                          </td>

                          <td className="px-3 py-3.5 text-center">
                            <span
                              className={`inline-flex items-center rounded-full px-3 py-1 text-xs font-bold ${
                                countdown.isToday
                                  ? 'bg-rose-100 text-rose-800 border border-rose-200 animate-pulse'
                                  : countdown.isPast
                                  ? 'bg-zinc-100 text-zinc-600 border border-zinc-200'
                                  : 'bg-amber-100 text-amber-800 border border-amber-200'
                              }`}
                            >
                              {countdown.text}
                            </span>
                          </td>

                          <td className="px-3 py-3.5 text-right font-mono font-semibold text-zinc-800 text-sm sm:text-base">
                            {r.units_held.toLocaleString('en-US', { minimumFractionDigits: 2 })}
                          </td>

                          <td className="px-3 py-3.5 text-right font-mono font-semibold text-zinc-800 text-sm sm:text-base">
                            {isUSD ? `$${r.dps.toFixed(4)}` : `฿${r.dps.toFixed(4)}`}
                          </td>

                          <td className="px-3 py-3.5 text-right font-mono font-bold text-rose-600 text-sm sm:text-base">
                            {(r.tax_rate * 100).toFixed(0)}%
                          </td>

                          <td className="px-4 py-3.5 text-right font-mono">
                            {isUSD ? (
                              <div className="flex flex-col items-end">
                                <span className="font-extrabold text-amber-600 text-sm sm:text-base">
                                  ${(r.net_native ?? 0).toFixed(2)}
                                </span>
                                <span className="text-xs text-zinc-500 font-sans font-medium">
                                  ({formatTHB(r.net_thb)})
                                </span>
                              </div>
                            ) : (
                              <span className="font-extrabold text-amber-600 text-sm sm:text-base">{formatTHB(r.net_thb)}</span>
                            )}
                          </td>
                        </tr>
                      )
                    })
                  )}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {/* Detailed Rounds Modal */}
      {selectedSymbolRounds && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-zinc-900/60 backdrop-blur-sm p-4 animate-fade-in">
          <div className="w-full max-w-4xl rounded-2xl border border-sky-100 bg-panel p-6 shadow-2xl space-y-4">
            <div className="flex items-center justify-between border-b border-sky-100 pb-3">
              <div>
                <h3 className="text-lg font-bold text-zinc-900 flex items-center gap-2">
                  <span>📅</span> รายละเอียดรอบเงินปันผล: {selectedSymbolRounds.symbol}
                </h3>
                <p className="text-xs text-zinc-500 font-medium">
                  รวม {selectedSymbolRounds.rounds.length} รอบตามประวัติการถือครอง
                </p>
              </div>
              <button
                type="button"
                onClick={() => setSelectedSymbolRounds(null)}
                className="rounded-xl border border-sky-200 bg-white hover:bg-sky-50 p-2 text-zinc-500 hover:text-zinc-900 transition-colors shadow-sm cursor-pointer"
              >
                ✕
              </button>
            </div>

            <div className="overflow-x-auto max-h-[60vh]">
              <table className="w-full text-left text-sm">
                <thead className="border-b border-sky-100 bg-sky-50/80 text-xs sm:text-sm font-bold uppercase tracking-wider text-zinc-700">
                  <tr>
                    <th className="px-3 py-3">รอบ</th>
                    <th className="px-3 py-3">สถานะ</th>
                    <th className="px-3 py-3">วัน XD</th>
                    <th className="px-3 py-3">วันจ่ายเงิน (Pay Date)</th>
                    <th className="px-3 py-3 text-right">DPS</th>
                    <th className="px-3 py-3 text-right">หน่วยที่ถือ</th>
                    <th className="px-3 py-3 text-right">WHT</th>
                    <th className="px-3 py-3 text-right">สุทธิ (Native)</th>
                    <th className="px-3 py-3 text-right">FX Rate</th>
                    <th className="px-3 py-3 text-right font-bold text-emerald-700">สุทธิ (THB)</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-sky-100/60 font-mono">
                  {selectedSymbolRounds.rounds.map((r, idx) => {
                    const isUSD = r.currency === 'USD'
                    const isUpcoming = r.status === 'upcoming'

                    return (
                      <tr key={`${r.symbol}-${r.ex_date}-${idx}`} className="hover:bg-sky-50/40 transition-colors">
                        <td className="px-3 py-3 text-zinc-500 text-xs">#{selectedSymbolRounds.rounds.length - idx}</td>

                        <td className="px-3 py-3 font-sans">
                          {isUpcoming ? (
                            <span className="inline-flex items-center rounded-full bg-amber-50 border border-amber-200 px-2.5 py-0.5 text-xs font-bold text-amber-700">
                              ⏳ รอจ่าย
                            </span>
                          ) : (
                            <span className="inline-flex items-center rounded-full bg-emerald-50 border border-emerald-200 px-2.5 py-0.5 text-xs font-bold text-emerald-700">
                              ✅ ได้รับแล้ว
                            </span>
                          )}
                        </td>

                        <td className="px-3 py-3 text-zinc-900 font-bold">{r.ex_date}</td>

                        <td className="px-3 py-3 text-zinc-700 font-semibold">{r.pay_date || '—'}</td>

                        <td className="px-3 py-3 text-right text-zinc-800 font-semibold">{r.dps.toFixed(4)}</td>

                        <td className="px-3 py-3 text-right text-zinc-800 font-semibold">{r.units_held.toLocaleString('en-US')}</td>

                        <td className="px-3 py-3 text-right text-rose-600 font-bold">
                          {(r.tax_rate * 100).toFixed(0)}%
                        </td>

                        <td className="px-3 py-3 text-right font-bold text-zinc-900">
                          {isUSD ? `$${(r.net_native ?? (r.net_thb / (r.fx_rate || 1))).toFixed(2)}` : formatTHB(r.net_thb)}
                        </td>

                        <td className="px-3 py-3 text-right text-zinc-600">
                          {isUSD ? r.fx_rate.toFixed(4) : '-'}
                        </td>

                        <td className="px-3 py-2 text-right font-bold text-emerald-600">
                          {formatTHB(r.net_thb)}
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>

            <div className="flex justify-end pt-3 border-t border-sky-100">
              <button
                type="button"
                onClick={() => setSelectedSymbolRounds(null)}
                className="rounded-xl border border-sky-200 bg-white hover:bg-sky-50 px-4 py-2 text-xs font-semibold text-zinc-700 transition-colors shadow-sm cursor-pointer"
              >
                ปิด
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
