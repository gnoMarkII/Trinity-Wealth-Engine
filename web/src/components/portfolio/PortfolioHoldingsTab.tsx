import { useState, useMemo, Fragment, useEffect } from 'react'
import type { ActualHoldingDTO, AllocationTargetDTO, ActualPortfolioStateDTO, JournalEntryDTO } from '../../api/types'
import { formatTHB } from '../../utils/formatters'
import { api } from '../../api/client'
import BatchAssignBucketModal from './Modals/BatchAssignBucketModal'
import HoldingCorrectionModal from './Modals/HoldingCorrectionModal'
import JournalModal from './Modals/JournalModal'
import Modal from '../ui/Modal'
import { TradeIcon, EditIcon, DeleteIcon, FolderIcon, JournalIcon } from './icons/PortfolioIcons'
import { isPvdHolding, isValidIsoDate } from '../../lib/pvdPresentation'

interface Props {
  portfolioId: string
  holdings: ActualHoldingDTO[]
  selectedBucket: string | null
  onClearBucketFilter: () => void
  targets?: AllocationTargetDTO[]
  allocationPolicy?: ActualPortfolioStateDTO['allocation_policy']
  onSuccess?: (state: ActualPortfolioStateDTO) => void
  journalRows?: JournalEntryDTO[]
  journalKeyword?: string
  onChangeJournalKeyword?: (kw: string) => void
  onSuccessJournal?: (entries: JournalEntryDTO[]) => void
  onOpenTradeModal?: () => void
  onViewTransactions?: (symbol: string) => void
}

type SortField = 'symbol' | 'asset_type' | 'bucket_id' | 'units' | 'market_value_thb' | 'unrealized_pnl_percent' | 'pe_ratio' | 'yield_on_cost'

export default function PortfolioHoldingsTab({
  portfolioId,
  holdings,
  selectedBucket,
  onClearBucketFilter,
  targets = [],
  allocationPolicy,
  onSuccess,
  journalRows = [],
  journalKeyword = '',
  onChangeJournalKeyword,
  onSuccessJournal,
  onOpenTradeModal,
  onViewTransactions,
}: Props) {
  const [sortField, setSortField] = useState<SortField>('market_value_thb')
  const [sortAsc, setSortAsc] = useState<boolean>(false)
  const [selectedSymbols, setSelectedSymbols] = useState<Set<string>>(new Set())
  const [filterOnlyUnassigned, setFilterOnlyUnassigned] = useState(false)

  const [expandedSymbol, setExpandedSymbol] = useState<string | null>(null)
  const [journalModalOpen, setJournalModalOpen] = useState(false)
  const [journalModalSymbol, setJournalModalSymbol] = useState<string | undefined>(undefined)
  const [allEntriesModalOpen, setAllEntriesModalOpen] = useState(false)

  const [batchAssignModalOpen, setBatchAssignModalOpen] = useState(false)
  const [editingHolding, setEditingHolding] = useState<ActualHoldingDTO | null>(null)
  const [batchDeleteLoading, setBatchDeleteLoading] = useState(false)
  const [selectionNotice, setSelectionNotice] = useState<string | null>(null)

  const currentHoldingBySymbol = useMemo(
    () => new Map(holdings.map((holding) => [holding.symbol, holding])),
    [holdings],
  )

  useEffect(() => {
    const invalidSelectedSymbols = Array.from(selectedSymbols).filter((symbol) => {
      const currentHolding = currentHoldingBySymbol.get(symbol)
      return !currentHolding
    })
    if (invalidSelectedSymbols.length > 0) {
      setSelectedSymbols(new Set())
      setBatchAssignModalOpen(false)
      setSelectionNotice('ยกเลิกการเลือกแล้ว เพราะรายการในพอร์ตเปลี่ยน')
    }

    if (editingHolding) {
      const currentHolding = currentHoldingBySymbol.get(editingHolding.symbol)
      if (!currentHolding || isPvdHolding(currentHolding)) setEditingHolding(null)
    }
    if (batchAssignModalOpen && Array.from(selectedSymbols).some((symbol) => !currentHoldingBySymbol.get(symbol))) {
      setBatchAssignModalOpen(false)
    }
  }, [holdings, currentHoldingBySymbol, selectedSymbols, editingHolding, batchAssignModalOpen])

  const getValidatedSelectedSymbolsForAssign = (): string[] | null => {
    const symbols = Array.from(selectedSymbols)
    const invalidSymbols = symbols.filter((symbol) => !currentHoldingBySymbol.get(symbol))
    if (symbols.length === 0 || invalidSymbols.length > 0) {
      setSelectionNotice('รายการที่เลือกเปลี่ยนไป กรุณาเลือกสินทรัพย์ใหม่ก่อนทำรายการ')
      return null
    }
    return symbols
  }

  const getValidatedSelectedSymbolsForDelete = (): string[] | null => {
    const symbols = Array.from(selectedSymbols)
    const invalidSymbols = symbols.filter((symbol) => {
      const holding = currentHoldingBySymbol.get(symbol)
      return !holding || isPvdHolding(holding)
    })
    if (symbols.length === 0 || invalidSymbols.length > 0) {
      setSelectionNotice('รายการที่เลือกมี PVD รวมอยู่ ซึ่งอัปเดตจาก Statement จึงลบรายกองไม่ได้')
      return null
    }
    return symbols
  }

  const openBatchAssignModal = () => {
    if (getValidatedSelectedSymbolsForAssign()) setBatchAssignModalOpen(true)
  }

  const handleBatchDelete = async () => {
    if (!onSuccess) return
    const symbols = getValidatedSelectedSymbolsForDelete()
    if (!symbols) return
    if (!window.confirm(`คุณแน่ใจหรือไม่ว่าต้องการลบสินทรัพย์ที่เลือก ${symbols.length} รายการนี้ออกจากพอร์ต?`)) return
    setBatchDeleteLoading(true)
    try {
      const state = await api.batchRemoveHoldings({ symbols }, portfolioId)
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
    const holding = currentHoldingBySymbol.get(symbol)
    if (!holding || isPvdHolding(holding)) {
      setSelectionNotice('ยอด PVD อัปเดตจาก Statement จึงลบรายกองไม่ได้')
      return
    }
    if (!window.confirm(`คุณแน่ใจหรือไม่ว่าต้องการลบ ${symbol} ออกจากพอร์ต?`)) return
    try {
      const state = await api.removeHolding(symbol, portfolioId)
      onSuccess(state)
    } catch (err: any) {
      alert(err?.message || 'ลบรายการไม่สำเร็จ')
    }
  }

  // Unassigned holdings calculation (WP13 Task 9)
  const unassignedHoldings = useMemo(
    () => holdings.filter((h) => !h.bucket_id || h.bucket_id.trim() === ''),
    [holdings]
  )
  const assignableUnassignedHoldings = useMemo(
    () => unassignedHoldings,
    [unassignedHoldings],
  )
  const unassignedTotalValue = useMemo(
    () => unassignedHoldings.reduce((sum, h) => sum + (h.market_value_thb || 0), 0),
    [unassignedHoldings]
  )

  // Filter holdings by URL/selected bucket and unassigned toggle
  const filteredHoldings = useMemo(() => {
    let list = holdings
    if (selectedBucket) {
      list = list.filter((h) => h.bucket_id === selectedBucket)
    }
    if (filterOnlyUnassigned) {
      list = list.filter((h) => !h.bucket_id || h.bucket_id.trim() === '')
    }
    return list
  }, [holdings, selectedBucket, filterOnlyUnassigned])

  useEffect(() => {
    const visibleSymbols = new Set(filteredHoldings.map((holding) => holding.symbol))
    if (Array.from(selectedSymbols).some((symbol) => !visibleSymbols.has(symbol))) {
      setSelectedSymbols(new Set())
      setBatchAssignModalOpen(false)
      setSelectionNotice('ยกเลิกการเลือกแล้ว เพราะตัวกรองรายการเปลี่ยน')
    }
  }, [filteredHoldings, selectedSymbols])

  // Sort holdings
  const sortedHoldings = useMemo(() => {
    return [...filteredHoldings].sort((a, b) => {
      let valA: any = sortField === 'unrealized_pnl_percent' && isPvdHolding(a) ? null : a[sortField]
      let valB: any = sortField === 'unrealized_pnl_percent' && isPvdHolding(b) ? null : b[sortField]

      if (valA === null || valA === undefined) valA = -Infinity
      if (valB === null || valB === undefined) valB = -Infinity

      if (typeof valA === 'string' && typeof valB === 'string') {
        return sortAsc ? valA.localeCompare(valB) : valB.localeCompare(valA)
      }
      return sortAsc ? valA - valB : valB - valA
    })
  }, [filteredHoldings, sortField, sortAsc])

  const selectableHoldings = useMemo(
    () => sortedHoldings,
    [sortedHoldings],
  )
  const selectedVisibleCount = selectableHoldings.filter((holding) => selectedSymbols.has(holding.symbol)).length
  const allVisibleSelectableSelected = selectableHoldings.length > 0 && selectedVisibleCount === selectableHoldings.length

  const handleSort = (field: SortField) => {
    if (sortField === field) {
      setSortAsc(!sortAsc)
    } else {
      setSortField(field)
      setSortAsc(false)
    }
  }

  const toggleSelectAll = () => {
    if (allVisibleSelectableSelected) {
      setSelectedSymbols(new Set())
    } else {
      setSelectedSymbols(new Set(selectableHoldings.map((h) => h.symbol)))
    }
    setSelectionNotice(null)
  }

  const toggleSelect = (symbol: string) => {
    const holding = currentHoldingBySymbol.get(symbol)
    if (!holding) return
    const next = new Set(selectedSymbols)
    if (next.has(symbol)) {
      next.delete(symbol)
    } else {
      next.add(symbol)
    }
    setSelectedSymbols(next)
    setSelectionNotice(null)
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
      {/* Unassigned Holdings Banner (WP13 Task 9) */}
      {unassignedHoldings.length > 0 && !selectedBucket && (
        <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 rounded-2xl border border-amber-300 bg-gradient-to-r from-amber-50/90 via-white to-orange-50/70 p-4 shadow-sm animate-fade-in">
          <div className="flex items-start gap-3">
            <span className="text-2xl shrink-0 mt-0.5">⚠️</span>
            <div>
              <div className="text-sm font-bold text-amber-950 flex items-center gap-2">
                <span>พบสินทรัพย์ที่ยังไม่ได้จัดหมวดหมู่ (Unassigned)</span>
                <span className="rounded-full bg-amber-200/90 px-2 py-0.5 text-xs font-bold text-amber-900">
                  {unassignedHoldings.length} รายการ
                </span>
              </div>
              <p className="text-xs text-amber-800/90 mt-0.5">
                มีสินทรัพย์มูลค่ารวม <strong className="font-bold text-amber-950">{formatTHB(unassignedTotalValue)}</strong> ที่ยังไม่ได้กำหนด Strategy Bucket
                {unassignedHoldings.some(isPvdHolding) && <span className="block mt-1">รายการ PVD เป็นแบบอ่านอย่างเดียวและอัปเดตผ่าน Statement</span>}
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2 shrink-0 w-full sm:w-auto justify-end">
            <button
              type="button"
              onClick={() => setFilterOnlyUnassigned(!filterOnlyUnassigned)}
              className={`rounded-xl px-3 py-1.5 text-xs font-bold transition-all border ${
                filterOnlyUnassigned
                  ? 'bg-amber-200 border-amber-400 text-amber-950'
                  : 'bg-white border-amber-300 text-amber-900 hover:bg-amber-100'
              }`}
            >
              {filterOnlyUnassigned ? 'แสดงทั้งหมด' : 'กรองเฉพาะ Unassigned'}
            </button>
            <button
              type="button"
              onClick={() => {
                const symbols = assignableUnassignedHoldings.map((holding) => holding.symbol)
                if (symbols.length === 0) return
                setSelectedSymbols(new Set(symbols))
                setSelectionNotice(null)
                setBatchAssignModalOpen(true)
              }}
              disabled={assignableUnassignedHoldings.length === 0}
              title={assignableUnassignedHoldings.length === 0 ? 'ไม่มีสินทรัพย์ทั่วไปที่จัดหมวดหมู่ได้' : undefined}
              className="rounded-xl bg-amber-600 px-3.5 py-1.5 text-xs font-bold text-white shadow-sm hover:bg-amber-700 active:scale-98 transition-all flex items-center gap-1.5 disabled:cursor-not-allowed disabled:opacity-50"
            >
              <FolderIcon className="h-3.5 w-3.5" />
              <span>จัดหมวดหมู่สินทรัพย์ทั่วไป ({assignableUnassignedHoldings.length})</span>
            </button>
          </div>
        </div>
      )}

      {selectionNotice && (
        <div role="status" className="rounded-xl border border-amber-200 bg-amber-50 px-4 py-2 text-xs font-semibold text-amber-900">
          {selectionNotice}
        </div>
      )}

      {/* Bucket Filter Active Banner */}
      {selectedBucket && (
        <div className="flex items-center justify-between rounded-xl border border-sky-200 bg-sky-50/80 px-4 py-3 text-sm text-sky-900 shadow-sm animate-fade-in">
          <div className="flex items-center gap-2">
            <span className="font-bold">🔍 กำลังกรองเฉพาะ Strategy Bucket:</span>
            <span className="rounded-lg bg-flow-blue px-2.5 py-0.5 font-mono tabular-nums text-xs font-semibold text-white">
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
      {selectedVisibleCount > 0 && (
        <div className="sticky top-4 z-20 flex items-center justify-between rounded-xl border border-sky-200 bg-zinc-900 px-5 py-3 text-white shadow-xl animate-bounce-short">
          <div className="flex items-center gap-3">
            <span className="rounded-full bg-flow-cyan px-3 py-0.5 text-xs font-extrabold text-zinc-900">
              {selectedVisibleCount} Selected
            </span>
            <span className="text-xs text-zinc-300">
              เลือกสินทรัพย์ทั่วไปเพื่อดำเนินการแบบกลุ่ม (Batch Actions)
            </span>
          </div>
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={openBatchAssignModal}
              className="rounded-lg bg-flow-blue px-3 py-1.5 text-xs font-bold text-white shadow-md hover:bg-sky-600 transition-colors flex items-center gap-1.5"
            >
              <FolderIcon className="h-3.5 w-3.5" />
              <span>เปลี่ยน Bucket ({selectedVisibleCount})</span>
            </button>
            <button
              type="button"
              onClick={handleBatchDelete}
              disabled={batchDeleteLoading || !onSuccess}
              className="rounded-lg bg-rose-600 px-3 py-1.5 text-xs font-bold text-white shadow-md hover:bg-rose-700 disabled:opacity-50 transition-colors flex items-center gap-1.5"
            >
              <DeleteIcon className="h-3.5 w-3.5" />
              <span>{batchDeleteLoading ? 'กำลังลบ...' : `ลบที่เลือก (${selectedVisibleCount})`}</span>
            </button>
            <button
              type="button"
              onClick={() => { setSelectedSymbols(new Set()); setSelectionNotice(null) }}
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
            {holdings.some(isPvdHolding) && (
              <p className="mt-1 text-xs font-medium text-teal-800">
                Statement มียอดเงินต้นรวม จึงยังแสดงต้นทุนและผลตอบแทนจริงรายกองไม่ได้
              </p>
            )}
          </div>
          <div className="flex items-center gap-2">
            {onSuccessJournal && (
              <button
                type="button"
                onClick={() => setAllEntriesModalOpen(true)}
                className="rounded-xl bg-flow-blue px-3.5 py-1.5 text-xs font-bold text-white shadow-sm hover:bg-sky-600 transition-colors flex items-center gap-1.5"
              >
                <JournalIcon className="h-4 w-4" />
                <span>Trading Journal ทั่วไป ({journalRows.length})</span>
              </button>
            )}
            <div className="text-xs text-zinc-500 hidden md:block">
              คลิกหัวตารางเพื่อเรียงลำดับ | เลือก Checkbox เพื่อจัดการ
            </div>
          </div>
        </div>

        <div className="overflow-x-auto custom-scrollbar">
          <table className="w-full text-left text-sm min-w-[1180px]">
            <thead>
              <tr className="border-b border-sky-100 bg-zinc-50/90 font-bold uppercase tracking-wider text-xs text-zinc-700 whitespace-nowrap">
                <th className="w-10 px-4 py-3.5 text-center">
                  <input
                    type="checkbox"
                    aria-label="Select all holdings"
                    checked={allVisibleSelectableSelected}
                    ref={(input) => { if (input) input.indeterminate = selectedVisibleCount > 0 && !allVisibleSelectableSelected }}
                    disabled={selectableHoldings.length === 0}
                    onChange={toggleSelectAll}
                    title={selectableHoldings.length === 0 ? 'ไม่มีสินทรัพย์ทั่วไปที่เลือกได้' : undefined}
                    className="h-4 w-4 rounded border-sky-300 text-flow-blue focus:ring-flow-cyan"
                  />
                </th>
                <th className="px-4 py-3.5 cursor-pointer select-none hover:text-flow-blue whitespace-nowrap" onClick={() => handleSort('symbol')}>
                  Symbol / Company {renderSortIndicator('symbol')}
                </th>
                <th className="px-4 py-3.5 cursor-pointer select-none hover:text-flow-blue whitespace-nowrap" onClick={() => handleSort('asset_type')}>
                  Type {renderSortIndicator('asset_type')}
                </th>
                <th className="px-4 py-3.5 cursor-pointer select-none hover:text-flow-blue whitespace-nowrap" onClick={() => handleSort('bucket_id')}>
                  Strategy Bucket {renderSortIndicator('bucket_id')}
                </th>
                <th className="px-4 py-3.5 text-right cursor-pointer select-none hover:text-flow-blue whitespace-nowrap" onClick={() => handleSort('units')}>
                  Shares / Avg Cost {renderSortIndicator('units')}
                </th>
                <th className="px-4 py-3.5 text-right whitespace-nowrap">Current Price</th>
                <th className="px-4 py-3.5 text-right cursor-pointer select-none hover:text-flow-blue whitespace-nowrap" onClick={() => handleSort('market_value_thb')}>
                  Market Value (THB) {renderSortIndicator('market_value_thb')}
                </th>
                <th className="px-4 py-3.5 text-right cursor-pointer select-none hover:text-flow-blue whitespace-nowrap" onClick={() => handleSort('unrealized_pnl_percent')}>
                  Unrealized PnL {renderSortIndicator('unrealized_pnl_percent')}
                </th>
                <th className="px-4 py-3.5 text-right cursor-pointer select-none hover:text-flow-blue whitespace-nowrap" onClick={() => handleSort('pe_ratio')}>
                  P/E & EPS {renderSortIndicator('pe_ratio')}
                </th>
                <th className="px-4 py-3.5 text-right cursor-pointer select-none hover:text-flow-blue whitespace-nowrap" onClick={() => handleSort('yield_on_cost')}>
                  Yield on Cost & Div {renderSortIndicator('yield_on_cost')}
                </th>
                <th className="px-4 py-3.5 text-center whitespace-nowrap">Tier & Updated</th>
                <th className="px-3 py-3.5 text-center whitespace-nowrap">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-sky-50">
              {sortedHoldings.map((h) => {
                const isPvd = isPvdHolding(h)
                const isSelected = selectedSymbols.has(h.symbol)
                const isPos = (h.unrealized_pnl_percent ?? 0) >= 0
                const statementNav = Number(h.pvd_valuation?.statement_nav)
                const displayStatementNav = h.pvd_valuation?.statement_nav && Number.isFinite(statementNav)
                  ? statementNav.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 4 })
                  : 'N/A'
                const statementDate = isValidIsoDate(h.pvd_valuation?.as_of_date)
                  ? h.pvd_valuation.as_of_date
                  : 'ไม่ทราบวันที่ Statement'
                const symbolEntries = (journalRows || []).filter((e) => e.content.includes(h.symbol))
                const bucketTarget = targets.find((t) => t.bucket_id === h.bucket_id)
                const bucketDisplayName = bucketTarget ? bucketTarget.name : null

                return (
                  <Fragment key={h.symbol}>
                    <tr
                      className={`transition-colors ${isSelected ? 'bg-flow-cyan/10' : 'hover:bg-sky-50/50'
                        }`}
                    >
                      {/* Checkbox */}
                      <td className="px-4 py-4 text-center align-top">
                        <input
                          type="checkbox"
                          aria-label={`Select ${h.symbol}`}
                          checked={isSelected}
                          onChange={() => toggleSelect(h.symbol)}
                          className="h-4 w-4 rounded border-sky-300 text-flow-blue focus:ring-flow-cyan"
                        />
                      </td>

                      {/* Col 1: Symbol (Line 1) + Company Name (Line 2) */}
                      <td className="px-4 py-4 align-top">
                        <div className="font-extrabold text-zinc-900 text-base tracking-tight font-mono">{h.symbol}</div>
                        <div className="mt-0.5 text-xs text-zinc-500 line-clamp-1 font-medium">
                          {h.company_name || (h.asset_type === 'Gold' ? 'ARR Paotang Gold 99.99%' : 'N/A')}
                        </div>
                      </td>

                      {/* Col 2: Asset Type */}
                      <td className="px-4 py-4 align-top whitespace-nowrap">
                        <span className={`inline-block rounded-md px-2.5 py-1 text-xs font-bold ${
                          h.asset_type === 'Gold'
                            ? 'bg-amber-100/80 border border-amber-300 text-amber-900'
                            : h.asset_type === 'Fund'
                            ? 'bg-purple-50 border border-purple-100 text-purple-800'
                            : 'bg-sky-50 border border-sky-100 text-sky-800'
                        }`}>
                          {h.asset_type === 'Gold' ? '🥇 Gold' : h.asset_type}
                        </span>
                      </td>

                      {/* Col 3: Strategy Bucket */}
                      <td className="px-4 py-4 align-top whitespace-nowrap">
                        {h.bucket_id ? (
                          <div className="flex flex-col items-start gap-1">
                            <button
                              type="button"
                              onClick={() => {
                                setSelectedSymbols(new Set([h.symbol]))
                                setBatchAssignModalOpen(true)
                              }}
                              className="font-semibold text-zinc-900 text-xs sm:text-sm hover:text-flow-blue transition-colors text-left flex items-center gap-1 group"
                              title="คลิกเพื่อเปลี่ยน Bucket ให้กับสินทรัพย์นี้"
                            >
                              <span>{bucketDisplayName || h.bucket_id}</span>
                              <span className="text-[11px] text-zinc-400 group-hover:text-flow-blue">✎</span>
                            </button>
                            {(() => {
                              const pb = allocationPolicy?.purpose_buckets?.find(
                                (p) => p.bucket_id === h.bucket_id
                              )
                              if (
                                pb?.eligible_assets &&
                                pb.eligible_assets.length > 0 &&
                                !pb.eligible_assets.includes(h.asset_type) &&
                                !pb.eligible_assets.includes('All')
                              ) {
                                return (
                                  <span
                                    className="text-[10px] text-amber-800 bg-amber-100/80 rounded px-1.5 py-0.5 font-medium whitespace-nowrap"
                                    title={`ประเภทสินทรัพย์ ${h.asset_type} ไม่อยู่ในรายชื่อสินทรัพย์ที่แนะนำสำหรับ Bucket นี้`}
                                  >
                                    ⚠️ นอกเกณฑ์แนะนำ
                                  </span>
                                )
                              }
                              return null
                            })()}
                          </div>
                        ) : (
                          <button
                            type="button"
                            onClick={() => {
                              if (isPvd) {
                                setSelectedSymbols(new Set([h.symbol]))
                                setBatchAssignModalOpen(true)
                              } else {
                                setEditingHolding(h)
                              }
                            }}
                            className="inline-flex items-center gap-1 rounded bg-amber-100/90 hover:bg-amber-200 text-amber-900 px-2 py-0.5 text-[11px] font-bold shadow-2xs transition-colors whitespace-nowrap"
                            title="คลิกเพื่อกำหนด Bucket ให้กับสินทรัพย์นี้"
                          >
                            <span>⚠️ ยังไม่ระบุหมวด</span>
                          </button>
                        )}
                      </td>

                      {/* Col 3: Shares/Units (Line 1) + Avg Cost (Line 2) */}
                      <td className="px-4 py-4 text-right align-top tabular-nums">
                        <div className="font-extrabold text-zinc-900 text-base font-mono">
                          {h.units.toLocaleString('en-US', { maximumFractionDigits: 4 })}
                          {h.asset_type === 'Gold' && (
                            <span className="text-xs font-semibold text-amber-800 ml-1">Oz</span>
                          )}
                        </div>
                        <div className="mt-0.5 text-xs text-zinc-500 font-medium">
                          Avg: {isPvd ? (
                            <span className="font-bold text-zinc-400 font-mono" title="ไม่มีต้นทุนซื้อจริงรายกองจาก Statement">—</span>
                          ) : (
                            <span className="font-bold text-zinc-800 font-mono">{formatPrice(h.avg_cost_usd, h.avg_cost_thb)}</span>
                          )}
                        </div>
                      </td>

                      {/* Col 4: Current Price (Line 1) + Currency tag (Line 2) */}
                      <td className="px-4 py-4 text-right align-top tabular-nums">
                        <div className="font-extrabold text-zinc-900 text-base font-mono">
                          {isPvd
                            ? displayStatementNav
                            : formatPrice(h.current_price_usd, h.current_price_thb)}
                        </div>
                        <div className="mt-0.5 text-xs text-zinc-400 font-semibold">
                          {isPvd
                            ? `NAV ตาม Statement · ${statementDate}`
                            : h.current_price_usd !== null && h.current_price_usd !== undefined ? 'USD' : 'THB'}
                        </div>
                      </td>

                      {/* Col 5: Market Value THB (Line 1) + Market Cap (Line 2) */}
                      <td className="px-4 py-4 text-right align-top tabular-nums">
                        <div className="font-extrabold text-zinc-950 text-base font-mono">
                          {formatTHB(h.market_value_thb)}
                        </div>
                        <div className="mt-0.5 text-xs text-zinc-500 font-medium">
                          {isPvd ? (
                            <span>มูลค่าตาม Statement</span>
                          ) : (
                          <>
                          Cap:{' '}
                          <span className="text-zinc-600 font-mono">
                            {h.market_cap_value
                              ? (h.current_price_usd !== null && h.current_price_usd !== undefined
                                ? formatPrice(h.market_cap_value / 1e9, null) + 'B'
                                : formatPrice(null, h.market_cap_value / 1e9) + 'B')
                              : 'N/A'}
                          </span>
                          </>
                          )}
                        </div>
                      </td>

                      {/* Col 6: Unrealized PnL % (Line 1) + Unrealized PnL Value (Line 2) */}
                      <td className="px-4 py-4 text-right align-top tabular-nums">
                        {isPvd ? (
                          <span className="text-zinc-400 text-sm" title="ดูผลประโยชน์รวมตาม Statement">
                            <span className="block font-bold" aria-label="ไม่มีผลตอบแทนจริงรายกอง">—</span>
                            <span className="mt-0.5 block text-[11px]">ดูผลประโยชน์รวมตาม Statement</span>
                          </span>
                        ) : h.unrealized_pnl_percent !== null ? (
                          <>
                            <div className={`font-extrabold text-base font-mono ${isPos ? 'text-emerald-600' : 'text-rose-600'}`}>
                              {isPos ? '+' : ''}
                              {h.unrealized_pnl_percent.toFixed(2)}%
                            </div>
                            <div className={`mt-0.5 text-xs font-bold font-mono ${isPos ? 'text-emerald-700' : 'text-rose-700'}`}>
                              {isPos ? '+' : ''}
                              {formatTHB(h.unrealized_pnl_value ?? 0)}
                            </div>
                          </>
                        ) : (
                          <span className="text-zinc-400 text-sm">-</span>
                        )}
                      </td>

                      {/* Col 7: P/E Ratio (Line 1) + EPS & Payout Ratio (Line 2) */}
                      <td className="px-4 py-4 text-right align-top tabular-nums">
                        <div className="font-bold text-zinc-800 text-sm font-mono">
                          <span className="text-zinc-400 font-normal text-xs">PE: </span>
                          {h.pe_ratio ? `${h.pe_ratio.toFixed(1)}x` : 'N/A'}
                        </div>
                        <div className="mt-0.5 text-xs text-zinc-500 font-medium">
                          <span className="text-zinc-400">EPS: </span>
                          <span className="font-mono text-zinc-700">{h.eps ? `$${h.eps.toFixed(2)}` : 'N/A'}</span>{' '}
                          {h.payout_ratio ? <span className="text-zinc-400 font-mono">({h.payout_ratio.toFixed(0)}% PO)</span> : ''}
                        </div>
                      </td>

                      {/* Col 8: Yield on Cost (Line 1) + Div Yield & DPS (Line 2) */}
                      <td className="px-4 py-4 text-right align-top tabular-nums">
                        <div className="font-extrabold text-amber-700 text-sm font-mono">
                          <span className="text-amber-800/60 font-medium text-xs">YoC: </span>
                          {h.yield_on_cost ? `${h.yield_on_cost.toFixed(2)}%` : 'N/A'}
                        </div>
                        <div className="mt-0.5 text-xs text-zinc-600 font-medium">
                          <span className="text-zinc-400">Div: </span>
                          <span className="font-mono text-zinc-700">{h.dividend_yield ? `${h.dividend_yield.toFixed(2)}%` : 'N/A'}</span>{' '}
                          {h.dividend_per_share ? <span className="text-zinc-400 font-mono">(${h.dividend_per_share.toFixed(2)})</span> : ''}
                        </div>
                      </td>

                      {/* Col 9: Market Cap Tier (Line 1) + Fundamentals Updated At (Line 2) */}
                      <td className="px-4 py-4 text-center align-top">
                        <span className="inline-block rounded-full border border-sky-200 bg-white px-2.5 py-0.5 text-xs font-bold text-zinc-700 shadow-2xs">
                          {h.market_cap_tier || 'N/A'}
                        </span>
                        <div className="mt-1 text-xs text-zinc-400 font-medium">
                          {h.fundamentals_updated_at
                            ? new Date(h.fundamentals_updated_at * 1000).toLocaleDateString('th-TH', { month: 'short', day: 'numeric' })
                            : 'Not cached'}
                        </div>
                      </td>

                      {/* Col 10: Actions */}
                      <td className="px-3 py-3.5 text-center align-middle whitespace-nowrap">
                        <div className="flex items-center justify-center gap-1.5">
                          {/* Journal Button */}
                          <button
                            type="button"
                            onClick={() => setExpandedSymbol(expandedSymbol === h.symbol ? null : h.symbol)}
                            className={`h-8 px-2.5 rounded-lg border text-xs font-bold transition-all flex items-center justify-center gap-1 shadow-2xs ${
                              expandedSymbol === h.symbol
                                ? 'border-flow-blue bg-flow-blue text-white shadow-xs'
                                : 'border-sky-200 bg-sky-50 text-sky-700 hover:bg-sky-600 hover:text-white'
                            }`}
                            title="ดู/เขียน Trading Journal สำหรับหุ้นนี้"
                          >
                            <JournalIcon className="h-4 w-4" />
                            <span>{symbolEntries.length}</span>
                          </button>

                          {/* Transactions Button */}
                          {onViewTransactions && (
                            <button
                              type="button"
                              onClick={() => onViewTransactions(h.symbol)}
                              className="h-8 px-2.5 rounded-lg border border-emerald-200 bg-emerald-50 text-xs font-bold text-emerald-700 hover:bg-emerald-600 hover:text-white transition-all flex items-center justify-center gap-1 shadow-2xs"
                              title={isPvd ? `ดูประวัติ Statement ของ ${h.symbol}` : `ดูประวัติ Transactions ของ ${h.symbol}`}
                            >
                              <TradeIcon className="h-4 w-4" />
                              <span>{isPvd ? 'Statement' : 'เทรด'}</span>
                            </button>
                          )}

                          {/* Edit Button */}
                          {!isPvd && (
                            <button
                              type="button"
                              onClick={() => setEditingHolding(h)}
                              className="h-8 w-8 rounded-lg border border-sky-200 bg-sky-50 text-flow-blue hover:bg-flow-blue hover:text-white transition-all flex items-center justify-center shadow-2xs"
                              title="แก้ไข / ปรับปรุงข้อมูล"
                            >
                              <EditIcon className="h-4 w-4" />
                            </button>
                          )}

                          {/* Delete Button */}
                          {onSuccess && !isPvd && (
                            <button
                              type="button"
                              onClick={() => handleRemoveSingle(h.symbol)}
                              className="h-7 w-7 rounded-lg border border-rose-200 bg-rose-50 text-rose-600 hover:bg-rose-600 hover:text-white transition-all flex items-center justify-center shadow-2xs"
                              title="ลบ Holding ออกจากพอร์ต"
                            >
                              <DeleteIcon className="h-3.5 w-3.5" />
                            </button>
                          )}
                        </div>
                      </td>
                    </tr>

                    {expandedSymbol === h.symbol && (
                      <tr className="bg-sky-50/70 border-b border-sky-200 animate-fade-in">
                        <td colSpan={12} className="px-6 py-5" aria-label="Journal Details">
                          <div className="rounded-xl border border-sky-200 bg-white p-4 shadow-sm space-y-4 text-left">
                            <div className="flex flex-col sm:flex-row sm:items-center justify-between border-b border-sky-100 pb-3 gap-2">
                              <div className="flex items-center gap-2">
                                <span className="flex h-7 w-7 items-center justify-center rounded-lg bg-flow-blue/10 text-flow-blue">
                                  <JournalIcon className="h-4 w-4" />
                                </span>
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
                                  <div className="flex items-center justify-between text-[11px] text-zinc-400 font-mono tabular-nums mb-1.5">
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
                  <td colSpan={11} className="py-16 text-center">
                    {holdings.length === 0 ? (
                      <div className="mx-auto max-w-md flex flex-col items-center justify-center space-y-4 animate-fade-in">
                        <div className="flex h-16 w-16 items-center justify-center rounded-2xl bg-sky-50 border border-sky-100 text-flow-blue shadow-sm">
                          <TradeIcon className="h-8 w-8" />
                        </div>
                        <div className="space-y-1">
                          <h4 className="text-base font-extrabold text-zinc-900">ยังไม่มีสินทรัพย์ในพอร์ตการลงทุน</h4>
                          <p className="text-xs text-zinc-500 max-w-sm">
                            เริ่มต้นสร้างพอร์ตของคุณด้วยการบันทึกรายการเทรด ซื้อหุ้น หรือกองทุนแรกของคุณเข้าสู่ระบบ
                          </p>
                        </div>
                        {onOpenTradeModal && (
                          <button
                            type="button"
                            onClick={onOpenTradeModal}
                            className="inline-flex items-center gap-2 rounded-xl bg-flow-blue px-4 py-2.5 text-xs font-bold text-white shadow-md hover:bg-sky-600 transition-all hover:scale-105"
                          >
                            <TradeIcon className="h-4 w-4" />
                            <span>บันทึกเทรดแรกของคุณทันที</span>
                          </button>
                        )}
                      </div>
                    ) : (
                      <div className="mx-auto max-w-md flex flex-col items-center justify-center space-y-3 py-4 animate-fade-in">
                        <span className="text-3xl">🔍</span>
                        <div className="space-y-1">
                          <h4 className="text-sm font-bold text-zinc-800">ไม่พบรายการ Holding ตามเงื่อนไขตัวกรอง</h4>
                          <p className="text-xs text-zinc-500">
                            {selectedBucket ? `กำลังกรองเฉพาะ Strategy Bucket [${selectedBucket}]` : 'ไม่มีหุ้นหรือสินทรัพย์ที่ตรงกับการค้นหาปัจจุบัน'}
                          </p>
                        </div>
                        {selectedBucket && (
                          <button
                            type="button"
                            onClick={onClearBucketFilter}
                            className="rounded-xl border border-sky-300 bg-sky-50 px-3 py-1.5 text-xs font-bold text-flow-blue shadow-2xs hover:bg-sky-100 transition-colors"
                          >
                            🧹 ล้างตัวกรอง Bucket
                          </button>
                        )}
                      </div>
                    )}
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>

      {batchAssignModalOpen && onSuccess && (
        <BatchAssignBucketModal
          portfolioId={portfolioId}
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
          portfolioId={portfolioId}
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
                <div className="flex items-center justify-between text-xs text-zinc-400 font-mono tabular-nums mb-2">
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
          portfolioId={portfolioId}
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
