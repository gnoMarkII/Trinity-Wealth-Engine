import { useState, useMemo, Fragment } from 'react'
import type { ActualHoldingDTO, AllocationTargetDTO, ActualPortfolioStateDTO, JournalEntryDTO } from '../../api/types'
import { formatTHB } from './PortfolioSummaryCards'
import { api } from '../../api/client'
import BatchAssignBucketModal from './Modals/BatchAssignBucketModal'
import HoldingCorrectionModal from './Modals/HoldingCorrectionModal'
import JournalModal from './Modals/JournalModal'
import Modal from '../ui/Modal'

interface Props {
  holdings: ActualHoldingDTO[]
  selectedBucket: string | null
  onClearBucketFilter: () => void
  targets?: AllocationTargetDTO[]
  onSuccess?: (state: ActualPortfolioStateDTO) => void
  journalRows?: JournalEntryDTO[]
  journalKeyword?: string
  onChangeJournalKeyword?: (kw: string) => void
  onSuccessJournal?: (entries: JournalEntryDTO[]) => void
}

type SortField = 'symbol' | 'asset_type' | 'units' | 'market_value_thb' | 'unrealized_pnl_percent' | 'pe_ratio' | 'yield_on_cost'

export default function PortfolioHoldingsTab({
  holdings,
  selectedBucket,
  onClearBucketFilter,
  targets = [],
  onSuccess,
  journalRows = [],
  journalKeyword = '',
  onChangeJournalKeyword,
  onSuccessJournal,
}: Props) {
  const [sortField, setSortField] = useState<SortField>('market_value_thb')
  const [sortAsc, setSortAsc] = useState<boolean>(false)
  const [selectedSymbols, setSelectedSymbols] = useState<Set<string>>(new Set())

  const [expandedSymbol, setExpandedSymbol] = useState<string | null>(null)
  const [journalModalOpen, setJournalModalOpen] = useState(false)
  const [journalModalSymbol, setJournalModalSymbol] = useState<string | undefined>(undefined)
  const [allEntriesModalOpen, setAllEntriesModalOpen] = useState(false)

  const [batchAssignModalOpen, setBatchAssignModalOpen] = useState(false)
  const [editingHolding, setEditingHolding] = useState<ActualHoldingDTO | null>(null)
  const [batchDeleteLoading, setBatchDeleteLoading] = useState(false)

  const handleBatchDelete = async () => {
    if (!onSuccess) return
    if (!window.confirm(`คุณแน่ใจหรือไม่ว่าต้องการลบสินทรัพย์ที่เลือก ${selectedSymbols.size} รายการนี้ออกจากพอร์ต?`)) return
    setBatchDeleteLoading(true)
    try {
      const state = await api.batchRemoveHoldings({ symbols: Array.from(selectedSymbols) })
      setSelectedSymbols(new Set())
      onSuccess(state)
    } catch (err: any) {
      alert(err?.message || 'ลบรายการไม่สำเร็จ')
    } finally {
      setBatchDeleteLoading(false)
    }
  }

  const handleRemoveSingle = async (symbol: string) => {
    if (!onSuccess) return
    if (!window.confirm(`คุณแน่ใจหรือไม่ว่าต้องการลบ ${symbol} ออกจากพอร์ต?`)) return
    try {
      const state = await api.removeHolding(symbol)
      onSuccess(state)
    } catch (err: any) {
      alert(err?.message || 'ลบรายการไม่สำเร็จ')
    }
  }

  // Filter holdings by URL/selected bucket
  const filteredHoldings = useMemo(() => {
    if (!selectedBucket) return holdings
    return holdings.filter((h) => h.bucket_id === selectedBucket)
  }, [holdings, selectedBucket])

  // Sort holdings
  const sortedHoldings = useMemo(() => {
    return [...filteredHoldings].sort((a, b) => {
      let valA: any = a[sortField]
      let valB: any = b[sortField]

      if (valA === null || valA === undefined) valA = -Infinity
      if (valB === null || valB === undefined) valB = -Infinity

      if (typeof valA === 'string' && typeof valB === 'string') {
        return sortAsc ? valA.localeCompare(valB) : valB.localeCompare(valA)
      }
      return sortAsc ? valA - valB : valB - valA
    })
  }, [filteredHoldings, sortField, sortAsc])

  const handleSort = (field: SortField) => {
    if (sortField === field) {
      setSortAsc(!sortAsc)
    } else {
      setSortField(field)
      setSortAsc(false)
    }
  }

  const toggleSelectAll = () => {
    if (selectedSymbols.size === sortedHoldings.length) {
      setSelectedSymbols(new Set())
    } else {
      setSelectedSymbols(new Set(sortedHoldings.map((h) => h.symbol)))
    }
  }

  const toggleSelect = (symbol: string) => {
    const next = new Set(selectedSymbols)
    if (next.has(symbol)) {
      next.delete(symbol)
    } else {
      next.add(symbol)
    }
    setSelectedSymbols(next)
  }

  const formatPrice = (usd: number | null, thb: number | null) => {
    if (usd !== null && usd !== undefined) return `$${usd.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`
    if (thb !== null && thb !== undefined) return `฿${thb.toLocaleString('th-TH', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`
    return 'N/A'
  }

  const renderSortIndicator = (field: SortField) => {
    if (sortField !== field) return <span className="opacity-20 ml-1">⇅</span>
    return <span className="text-flow-blue ml-1 font-bold">{sortAsc ? '▲' : '▼'}</span>
  }

  return (
    <div className="space-y-4">
      {/* Bucket Filter Active Banner */}
      {selectedBucket && (
        <div className="flex items-center justify-between rounded-xl border border-sky-200 bg-sky-50/80 px-4 py-3 text-sm text-sky-900 shadow-sm animate-fade-in">
          <div className="flex items-center gap-2">
            <span className="font-bold">🔍 กำลังกรองเฉพาะ Strategy Bucket:</span>
            <span className="rounded-lg bg-flow-blue px-2.5 py-0.5 font-mono text-xs font-semibold text-white">
              {selectedBucket}
            </span>
            <span className="text-xs text-sky-700">({filteredHoldings.length} รายการ)</span>
          </div>
          <button
            type="button"
            onClick={onClearBucketFilter}
            className="flex items-center gap-1 rounded-lg border border-sky-300 bg-white px-3 py-1 text-xs font-semibold text-sky-700 transition-colors hover:bg-sky-100"
          >
            <span>ล้างตัวกรอง</span>
            <span>✕</span>
          </button>
        </div>
      )}

      {/* Floating Action Bar (Visible when selections made) */}
      {selectedSymbols.size > 0 && (
        <div className="sticky top-4 z-20 flex items-center justify-between rounded-xl border border-sky-200 bg-zinc-900 px-5 py-3 text-white shadow-xl animate-bounce-short">
          <div className="flex items-center gap-3">
            <span className="rounded-full bg-flow-cyan px-3 py-0.5 text-xs font-extrabold text-zinc-900">
              {selectedSymbols.size} Selected
            </span>
            <span className="text-xs text-zinc-300">
              เลือกสินทรัพย์เพื่อดำเนินการแบบกลุ่ม (Batch Actions)
            </span>
          </div>
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={() => setBatchAssignModalOpen(true)}
              className="rounded-lg bg-flow-blue px-3 py-1.5 text-xs font-bold text-white shadow-md hover:bg-sky-600 transition-colors"
            >
              📁 เปลี่ยน Bucket ({selectedSymbols.size})
            </button>
            <button
              type="button"
              onClick={handleBatchDelete}
              disabled={batchDeleteLoading || !onSuccess}
              className="rounded-lg bg-rose-600 px-3 py-1.5 text-xs font-bold text-white shadow-md hover:bg-rose-700 disabled:opacity-50 transition-colors"
            >
              {batchDeleteLoading ? 'กำลังลบ...' : `🗑️ ลบที่เลือก (${selectedSymbols.size})`}
            </button>
            <button
              type="button"
              onClick={() => setSelectedSymbols(new Set())}
              className="rounded-lg bg-zinc-800 px-2.5 py-1.5 text-xs font-medium text-zinc-300 hover:bg-zinc-700"
            >
              ยกเลิก
            </button>
          </div>
        </div>
      )}

      {/* Table Container with Overflow & 2-line Stacked Grid */}
      <div className="rounded-2xl border border-sky-100 bg-panel shadow-sm overflow-hidden">
        <div className="border-b border-sky-100 bg-sky-50/40 px-6 py-4 flex flex-col sm:flex-row sm:items-center justify-between gap-3">
          <div>
            <h3 className="text-base font-bold text-zinc-900">Portfolio Holdings ({filteredHoldings.length})</h3>
            <p className="text-xs text-zinc-500">
              ตารางแสดง 12-Column ครบถ้วน พร้อมรายละเอียด 2-line Stacked Grid (ข้อมูลหลักและข้อมูลพื้นฐานบริษัท)
            </p>
          </div>
          <div className="flex items-center gap-2">
            {onSuccessJournal && (
              <button
                type="button"
                onClick={() => setAllEntriesModalOpen(true)}
                className="rounded-xl bg-flow-blue px-3.5 py-1.5 text-xs font-bold text-white shadow-sm hover:bg-sky-600 transition-colors flex items-center gap-1.5"
              >
                <span>📓</span>
                <span>Trading Journal ทั่วไป ({journalRows.length})</span>
              </button>
            )}
            <div className="text-xs text-zinc-500 hidden md:block">
              คลิกหัวตารางเพื่อเรียงลำดับ | เลือก Checkbox เพื่อจัดการ
            </div>
          </div>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs min-w-[1000px]">
            <thead>
              <tr className="border-b border-sky-100 bg-zinc-50/80 font-semibold uppercase tracking-wider text-zinc-600">
                <th className="w-10 px-4 py-3 text-center">
                  <input
                    type="checkbox"
                    checked={sortedHoldings.length > 0 && selectedSymbols.size === sortedHoldings.length}
                    onChange={toggleSelectAll}
                    className="h-4 w-4 rounded border-sky-300 text-flow-blue focus:ring-flow-cyan"
                  />
                </th>
                <th className="px-4 py-3 cursor-pointer select-none hover:text-flow-blue" onClick={() => handleSort('symbol')}>
                  Symbol / Company {renderSortIndicator('symbol')}
                </th>
                <th className="px-4 py-3 cursor-pointer select-none hover:text-flow-blue" onClick={() => handleSort('asset_type')}>
                  Type / Bucket {renderSortIndicator('asset_type')}
                </th>
                <th className="px-4 py-3 text-right cursor-pointer select-none hover:text-flow-blue" onClick={() => handleSort('units')}>
                  Shares / Avg Cost {renderSortIndicator('units')}
                </th>
                <th className="px-4 py-3 text-right">Current Price</th>
                <th className="px-4 py-3 text-right cursor-pointer select-none hover:text-flow-blue" onClick={() => handleSort('market_value_thb')}>
                  Market Value (THB) {renderSortIndicator('market_value_thb')}
                </th>
                <th className="px-4 py-3 text-right cursor-pointer select-none hover:text-flow-blue" onClick={() => handleSort('unrealized_pnl_percent')}>
                  Unrealized PnL {renderSortIndicator('unrealized_pnl_percent')}
                </th>
                <th className="px-4 py-3 text-right cursor-pointer select-none hover:text-flow-blue" onClick={() => handleSort('pe_ratio')}>
                  P/E & EPS {renderSortIndicator('pe_ratio')}
                </th>
                <th className="px-4 py-3 text-right cursor-pointer select-none hover:text-flow-blue" onClick={() => handleSort('yield_on_cost')}>
                  Yield on Cost & Div {renderSortIndicator('yield_on_cost')}
                </th>
                <th className="px-4 py-3 text-center">Tier & Updated</th>
                <th className="px-3 py-3 text-center">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-sky-50">
              {sortedHoldings.map((h) => {
                const isSelected = selectedSymbols.has(h.symbol)
                const isPos = (h.unrealized_pnl_percent ?? 0) >= 0
                const symbolEntries = (journalRows || []).filter((e) => e.content.includes(h.symbol))

                return (
                  <Fragment key={h.symbol}>
                    <tr
                      className={`transition-colors ${
                        isSelected ? 'bg-flow-cyan/10' : 'hover:bg-sky-50/50'
                      }`}
                    >
                    {/* Checkbox */}
                    <td className="px-4 py-4 text-center align-top">
                      <input
                        type="checkbox"
                        checked={isSelected}
                        onChange={() => toggleSelect(h.symbol)}
                        className="h-4 w-4 rounded border-sky-300 text-flow-blue focus:ring-flow-cyan"
                      />
                    </td>

                    {/* Col 1: Symbol (Line 1) + Company Name (Line 2) */}
                    <td className="px-4 py-4 align-top">
                      <div className="font-bold text-zinc-900 text-sm tracking-tight">{h.symbol}</div>
                      <div className="mt-0.5 text-[11px] text-zinc-500 line-clamp-1">
                        {h.company_name || 'N/A'}
                      </div>
                    </td>

                    {/* Col 2: Asset Type (Line 1) + Bucket ID (Line 2) */}
                    <td className="px-4 py-4 align-top">
                      <span className="inline-block rounded-md bg-sky-50 px-2 py-0.5 text-xs font-semibold text-sky-800">
                        {h.asset_type}
                      </span>
                      <div className="mt-1 font-mono text-[11px] text-zinc-400">
                        {h.bucket_id || 'unassigned'}
                      </div>
                    </td>

                    {/* Col 3: Shares/Units (Line 1) + Avg Cost (Line 2) */}
                    <td className="px-4 py-4 text-right align-top font-mono">
                      <div className="font-bold text-zinc-900">
                        {h.units.toLocaleString('en-US', { maximumFractionDigits: 4 })}
                      </div>
                      <div className="mt-0.5 text-[11px] text-zinc-500">
                        Avg: {formatPrice(h.avg_cost_usd, h.avg_cost_thb)}
                      </div>
                    </td>

                    {/* Col 4: Current Price (Line 1) + Currency tag (Line 2) */}
                    <td className="px-4 py-4 text-right align-top font-mono">
                      <div className="font-bold text-zinc-900">
                        {formatPrice(h.current_price_usd, h.current_price_thb)}
                      </div>
                      <div className="mt-0.5 text-[11px] text-zinc-400">
                        {h.current_price_usd !== null && h.current_price_usd !== undefined ? 'USD' : 'THB'}
                      </div>
                    </td>

                    {/* Col 5: Market Value THB (Line 1) + % of NAV placeholder (Line 2) */}
                    <td className="px-4 py-4 text-right align-top font-mono">
                      <div className="font-extrabold text-zinc-900 text-sm">
                        {formatTHB(h.market_value_thb)}
                      </div>
                      <div className="mt-0.5 text-[11px] text-zinc-400">
                        Market Cap:{' '}
                        {h.market_cap_value
                          ? (h.current_price_usd !== null && h.current_price_usd !== undefined
                              ? formatPrice(h.market_cap_value / 1e9, null) + 'B'
                              : formatPrice(null, h.market_cap_value / 1e9) + 'B')
                          : 'N/A'}
                      </div>
                    </td>

                    {/* Col 6: Unrealized PnL % (Line 1) + Unrealized PnL Value (Line 2) */}
                    <td className="px-4 py-4 text-right align-top font-mono">
                      {h.unrealized_pnl_percent !== null ? (
                        <>
                          <div className={`font-bold ${isPos ? 'text-emerald-600' : 'text-rose-600'}`}>
                            {isPos ? '+' : ''}
                            {h.unrealized_pnl_percent.toFixed(2)}%
                          </div>
                          <div className={`mt-0.5 text-[11px] font-semibold ${isPos ? 'text-emerald-700' : 'text-rose-700'}`}>
                            {isPos ? '+' : ''}
                            {formatTHB(h.unrealized_pnl_value ?? 0)}
                          </div>
                        </>
                      ) : (
                        <span className="text-zinc-400">-</span>
                      )}
                    </td>

                    {/* Col 7: P/E Ratio (Line 1) + EPS & Payout Ratio (Line 2) */}
                    <td className="px-4 py-4 text-right align-top font-mono">
                      <div className="font-semibold text-zinc-800">
                        PE: {h.pe_ratio ? `${h.pe_ratio.toFixed(1)}x` : 'N/A'}
                      </div>
                      <div className="mt-0.5 text-[11px] text-zinc-500">
                        EPS: {h.eps ? `$${h.eps.toFixed(2)}` : 'N/A'} {h.payout_ratio ? `(${h.payout_ratio.toFixed(0)}% PO)` : ''}
                      </div>
                    </td>

                    {/* Col 8: Yield on Cost (Line 1) + Div Yield & DPS (Line 2) */}
                    <td className="px-4 py-4 text-right align-top font-mono">
                      <div className="font-bold text-amber-700">
                        YoC: {h.yield_on_cost ? `${h.yield_on_cost.toFixed(2)}%` : 'N/A'}
                      </div>
                      <div className="mt-0.5 text-[11px] text-zinc-500">
                        Div: {h.dividend_yield ? `${h.dividend_yield.toFixed(2)}%` : 'N/A'} {h.dividend_per_share ? `($${h.dividend_per_share.toFixed(2)})` : ''}
                      </div>
                    </td>

                    {/* Col 9: Market Cap Tier (Line 1) + Fundamentals Updated At (Line 2) */}
                    <td className="px-4 py-4 text-center align-top">
                      <span className="inline-block rounded-full border border-sky-200 bg-white px-2.5 py-0.5 text-[11px] font-semibold text-zinc-700 shadow-2xs">
                        {h.market_cap_tier || 'N/A'}
                      </span>
                      <div className="mt-1 text-[10px] text-zinc-400">
                        {h.fundamentals_updated_at
                          ? new Date(h.fundamentals_updated_at * 1000).toLocaleDateString('th-TH', { month: 'short', day: 'numeric' })
                          : 'Not cached'}
                      </div>
                    </td>

                    {/* Col 10: Actions */}
                    <td className="px-3 py-4 text-center align-top space-y-1">
                      <div className="flex flex-col gap-1 items-center justify-center">
                        <button
                          type="button"
                          onClick={() => setExpandedSymbol(expandedSymbol === h.symbol ? null : h.symbol)}
                          className={`w-16 rounded-lg border px-2 py-1 text-[11px] font-bold transition-colors ${
                            expandedSymbol === h.symbol
                              ? 'border-flow-blue bg-flow-blue text-white'
                              : 'border-sky-200 bg-white text-zinc-700 hover:bg-sky-50'
                          }`}
                          title="ดู/เขียน Trading Journal สำหรับหุ้นนี้"
                        >
                          📓 {symbolEntries.length}
                        </button>
                        <button
                          type="button"
                          onClick={() => setEditingHolding(h)}
                          className="w-16 rounded-lg border border-sky-200 bg-sky-50 px-2 py-1 text-[11px] font-bold text-flow-blue hover:bg-flow-blue hover:text-white transition-colors"
                          title="แก้ไข / ปรับปรุง"
                        >
                          ✏️ แก้ไข
                        </button>
                        {onSuccess && (
                          <button
                            type="button"
                            onClick={() => handleRemoveSingle(h.symbol)}
                            className="w-16 rounded-lg border border-rose-200 bg-rose-50 px-2 py-1 text-[11px] font-bold text-rose-600 hover:bg-rose-600 hover:text-white transition-colors"
                            title="ลบ Holding"
                          >
                            🗑️ ลบ
                          </button>
                        )}
                      </div>
                    </td>
                  </tr>

                  {expandedSymbol === h.symbol && (
                    <tr className="bg-sky-50/70 border-b border-sky-200 animate-fade-in">
                      <td colSpan={11} className="px-6 py-5">
                        <div className="rounded-xl border border-sky-200 bg-white p-4 shadow-sm space-y-4 text-left">
                          <div className="flex flex-col sm:flex-row sm:items-center justify-between border-b border-sky-100 pb-3 gap-2">
                            <div className="flex items-center gap-2">
                              <span className="text-lg">📓</span>
                              <h4 className="font-bold text-zinc-900 text-sm">
                                Trading Journal & Activity Log สำหรับ [{h.symbol}] ({symbolEntries.length} บันทึก)
                              </h4>
                            </div>
                            {onSuccessJournal && (
                              <button
                                type="button"
                                onClick={() => {
                                  setJournalModalSymbol(h.symbol)
                                  setJournalModalOpen(true)
                                }}
                                className="rounded-lg bg-flow-blue px-3 py-1.5 text-xs font-bold text-white shadow-2xs hover:bg-sky-600 transition-colors shrink-0"
                              >
                                + เขียนบันทึกสำหรับ [{h.symbol}]
                              </button>
                            )}
                          </div>

                          <div className="space-y-2 max-h-60 overflow-y-auto pr-1">
                            {symbolEntries.map((entry, idx) => (
                              <div
                                key={`${entry.timestamp}-${idx}`}
                                className="rounded-xl border border-sky-100 bg-sky-50/40 p-3.5 font-sans text-xs text-zinc-800"
                              >
                                <div className="flex items-center justify-between text-[11px] text-zinc-400 font-mono mb-1.5">
                                  <span>🕒 {entry.timestamp}</span>
                                  <span className="rounded bg-white px-1.5 py-0.5 border border-sky-200 text-sky-700">Wikilink Obsidian</span>
                                </div>
                                <div className="whitespace-pre-wrap leading-relaxed">{entry.content}</div>
                              </div>
                            ))}
                            {symbolEntries.length === 0 && (
                              <div className="py-8 text-center text-xs text-zinc-400 italic">
                                ยังไม่มีบันทึก Trading Journal ที่กล่าวถึงหุ้น [{h.symbol}] ในขณะนี้
                              </div>
                            )}
                          </div>
                        </div>
                      </td>
                    </tr>
                  )}
                </Fragment>
              )
            })}
              {sortedHoldings.length === 0 && (
                <tr>
                  <td colSpan={11} className="py-16 text-center text-zinc-400">
                    ไม่พบรายการ Holding ตามเงื่อนไขที่เลือก
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>

      {batchAssignModalOpen && onSuccess && (
        <BatchAssignBucketModal
          symbols={Array.from(selectedSymbols)}
          targets={targets}
          onClose={() => setBatchAssignModalOpen(false)}
          onSuccess={(state) => {
            setSelectedSymbols(new Set())
            onSuccess(state)
          }}
        />
      )}

      {editingHolding && onSuccess && (
        <HoldingCorrectionModal
          holding={editingHolding}
          targets={targets}
          onClose={() => setEditingHolding(null)}
          onSuccess={(state) => {
            onSuccess(state)
          }}
        />
      )}

      {allEntriesModalOpen && onSuccessJournal && (
        <Modal
          titleId="all-journal-entries-title"
          onClose={() => setAllEntriesModalOpen(false)}
          panelClassName="max-w-3xl rounded-2xl border border-sky-100 bg-white p-6 shadow-2xl space-y-4"
        >
          <div className="flex items-center justify-between border-b border-sky-100 pb-3">
            <h3 id="all-journal-entries-title" className="text-base font-bold text-zinc-900 flex items-center gap-2">
              <span>📓 Trading Journal ทั่วไป (All Entries)</span>
              <span className="rounded-full bg-sky-100 px-2.5 py-0.5 text-xs text-flow-blue">
                {journalRows.length} บันทึก
              </span>
            </h3>
            <button
              type="button"
              onClick={() => setAllEntriesModalOpen(false)}
              className="text-zinc-400 hover:text-zinc-600"
            >
              ✕
            </button>
          </div>

          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 bg-sky-50/50 p-3 rounded-xl border border-sky-100">
            <div className="relative flex-1">
              <input
                type="text"
                placeholder="ค้นหาข้อความในบันทึกทั้งหมด (เช่น AAPL, DEPOSIT, MISTAKE...)"
                value={journalKeyword || ''}
                onChange={(e) => onChangeJournalKeyword?.(e.target.value)}
                className="w-full rounded-xl border border-sky-200 bg-white px-3.5 py-1.5 text-xs text-zinc-800 focus:border-flow-blue focus:outline-none"
              />
              {journalKeyword && (
                <button
                  type="button"
                  onClick={() => onChangeJournalKeyword?.('')}
                  className="absolute right-3 top-2 text-xs text-zinc-400 hover:text-zinc-600"
                >
                  ✕
                </button>
              )}
            </div>
            <button
              type="button"
              onClick={() => {
                setJournalModalSymbol(undefined)
                setJournalModalOpen(true)
              }}
              className="rounded-xl bg-flow-blue px-3.5 py-1.5 text-xs font-bold text-white shadow-md hover:bg-sky-600 transition-colors shrink-0"
            >
              + เขียนบันทึก Journal ใหม่
            </button>
          </div>

          <div className="space-y-3 max-h-96 overflow-y-auto pr-1">
            {journalRows.map((entry, idx) => (
              <div
                key={`${entry.timestamp}-${idx}`}
                className="rounded-xl border border-sky-100 bg-panel p-4 shadow-2xs hover:shadow-sm transition-shadow text-left"
              >
                <div className="flex items-center justify-between text-xs text-zinc-400 font-mono mb-2">
                  <span>🕒 {entry.timestamp}</span>
                  <span className="rounded-md bg-sky-50 px-2 py-0.5 text-[10px] text-sky-700 font-semibold">Obsidian Journal</span>
                </div>
                <div className="text-xs text-zinc-800 leading-relaxed whitespace-pre-wrap font-sans">
                  {entry.content}
                </div>
              </div>
            ))}
            {journalRows.length === 0 && (
              <div className="rounded-2xl border border-dashed border-sky-200 p-12 text-center text-zinc-400 text-xs">
                ไม่พบบันทึก Trading Journal ที่ตรงกับคำค้นหา
              </div>
            )}
          </div>

          <div className="flex justify-end pt-2 border-t border-sky-100">
            <button
              type="button"
              onClick={() => setAllEntriesModalOpen(false)}
              className="rounded-xl border border-zinc-300 bg-white px-4 py-2 text-xs font-semibold text-zinc-700 hover:bg-zinc-50"
            >
              ปิดหน้าต่าง
            </button>
          </div>
        </Modal>
      )}

      {journalModalOpen && onSuccessJournal && (
        <JournalModal
          initialSymbol={journalModalSymbol}
          onClose={() => {
            setJournalModalOpen(false)
            setJournalModalSymbol(undefined)
          }}
          onSuccess={(entries) => {
            onSuccessJournal(entries)
          }}
        />
      )}
    </div>
  )
}
