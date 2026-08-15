/* eslint-disable jsx-a11y/control-has-associated-label */
import { useState, useEffect, useMemo, useCallback } from 'react'
import type { TransactionItemDTO, TransactionSummaryDTO, ActualPortfolioStateDTO, ActualHoldingDTO } from '../../api/types'
import { api } from '../../api/client'
import { formatTHB } from '../../utils/formatters'
import { TradeIcon, EditIcon, PlusIcon, DeleteIcon } from './icons/PortfolioIcons'
import EditTransactionModal from './Modals/EditTransactionModal'
import Modal from '../ui/Modal'

interface Props {
  portfolioId: string
  initialSymbol?: string | null
  holdings?: ActualHoldingDTO[]
  onClearSymbolFilter?: () => void
  onOpenTradeModal?: () => void
  onSuccess?: (state: ActualPortfolioStateDTO) => void
}

type SortField = 'timestamp' | 'symbol' | 'units' | 'price' | 'cost_thb' | 'realized_pnl_thb'

export default function PortfolioTransactionsTab({
  portfolioId,
  initialSymbol,
  holdings,
  onClearSymbolFilter,
  onOpenTradeModal,
  onSuccess,
}: Props) {
  const [transactions, setTransactions] = useState<TransactionItemDTO[]>([])
  const [summary, setSummary] = useState<TransactionSummaryDTO | null>(null)
  const [loading, setLoading] = useState<boolean>(true)
  const [error, setError] = useState<string | null>(null)

  // Filters & Sorting
  const [searchQuery, setSearchQuery] = useState<string>(initialSymbol || '')
  const [actionFilter, setActionFilter] = useState<'ALL' | 'BUY' | 'SELL'>('ALL')
  const [sortField, setSortField] = useState<SortField>('timestamp')
  const [sortAsc, setSortAsc] = useState<boolean>(false)
  const [currencyView, setCurrencyView] = useState<'NATIVE' | 'THB'>('NATIVE')

  // Pagination
  const [pageSize, setPageSize] = useState<number>(50)
  const [currentPage, setCurrentPage] = useState<number>(1)

  // Full Edit Modal State
  const [editingTx, setEditingTx] = useState<TransactionItemDTO | null>(null)

  // Delete Dialog State
  const [deletingTx, setDeletingTx] = useState<TransactionItemDTO | null>(null)
  const [deleteAdjustCash, setDeleteAdjustCash] = useState<boolean>(true)
  const [deletingLoading, setDeletingLoading] = useState<boolean>(false)
  const [deletingError, setDeletingError] = useState<string | null>(null)

  // Inline Note Editor
  const [editingTxId, setEditingTxId] = useState<string | null>(null)
  const [editingNote, setEditingNote] = useState<string>('')
  const [savingNote, setSavingNote] = useState<boolean>(false)

  // Sync searchQuery when initialSymbol prop changes
  useEffect(() => {
    if (initialSymbol) {
      setSearchQuery(initialSymbol)
    }
  }, [initialSymbol])

  const loadTransactions = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const res = await api.getTransactions(portfolioId)
      setTransactions(res.transactions || [])
      setSummary(res.summary || null)
    } catch (err: any) {
      setError(err?.message || 'ไม่สามารถโหลดข้อมูล Transactions ได้')
    } finally {
      setLoading(false)
    }
  }, [portfolioId])

  useEffect(() => {
    loadTransactions()
  }, [loadTransactions])

  // Filtered & Sorted items
  const filteredItems = useMemo(() => {
    let list = [...transactions]

    if (searchQuery.trim()) {
      const q = searchQuery.trim().toUpperCase()
      list = list.filter(
        (t) => t.symbol.toUpperCase().includes(q) || (t.notes && t.notes.toUpperCase().includes(q))
      )
    }

    if (actionFilter !== 'ALL') {
      list = list.filter((t) => t.action.toUpperCase() === actionFilter)
    }

    list.sort((a, b) => {
      let valA: any
      let valB: any

      if (sortField === 'cost_thb') {
        const isBuyA = a.action.toUpperCase() === 'BUY'
        const isBuyB = b.action.toUpperCase() === 'BUY'
        valA = isBuyA ? a.cost_thb : a.cost_thb + (a.realized_pnl_thb ?? 0)
        valB = isBuyB ? b.cost_thb : b.cost_thb + (b.realized_pnl_thb ?? 0)
      } else {
        valA = a[sortField]
        valB = b[sortField]
      }

      if (valA === null || valA === undefined) valA = -Infinity
      if (valB === null || valB === undefined) valB = -Infinity

      if (typeof valA === 'string' && typeof valB === 'string') {
        return sortAsc ? valA.localeCompare(valB) : valB.localeCompare(valA)
      }
      return sortAsc ? valA - valB : valB - valA
    })

    return list
  }, [transactions, searchQuery, actionFilter, sortField, sortAsc])

  // Paginated items
  const totalPages = Math.max(1, Math.ceil(filteredItems.length / pageSize))
  const paginatedItems = useMemo(() => {
    const start = (currentPage - 1) * pageSize
    return filteredItems.slice(start, start + pageSize)
  }, [filteredItems, currentPage, pageSize])

  // Totals for filtered items
  const filteredTotals = useMemo(() => {
    let totalShares = 0
    let totalRealizedPnL = 0
    let totalBuyTHB = 0
    let totalSellTHB = 0

    for (const item of filteredItems) {
      totalShares += item.units
      const isBuy = item.action.toUpperCase() === 'BUY'
      const summTHB = isBuy ? item.cost_thb : item.cost_thb + (item.realized_pnl_thb ?? 0)
      if (isBuy) {
        totalBuyTHB += summTHB
      } else {
        totalSellTHB += summTHB
      }
      if (item.realized_pnl_thb !== null && item.realized_pnl_thb !== undefined) {
        totalRealizedPnL += item.realized_pnl_thb
      }
    }

    return {
      totalShares,
      totalRealizedPnL,
      totalBuyTHB,
      totalSellTHB,
      totalNetTHB: totalSellTHB - totalBuyTHB,
    }
  }, [filteredItems])

  const handleSort = (field: SortField) => {
    if (sortField === field) {
      setSortAsc(!sortAsc)
    } else {
      setSortField(field)
      setSortAsc(false)
    }
  }

  const startEditNote = (tx: TransactionItemDTO) => {
    setEditingTxId(tx.transaction_id)
    setEditingNote(tx.notes || '')
  }

  const cancelEditNote = () => {
    setEditingTxId(null)
    setEditingNote('')
  }

  const saveNote = async (txId: string) => {
    setSavingNote(true)
    try {
      const updated = await api.updateTransactionNote(txId, editingNote, portfolioId)
      setTransactions((prev) =>
        prev.map((t) => (t.transaction_id === txId ? { ...t, notes: updated.notes } : t))
      )
      setEditingTxId(null)
      setEditingNote('')
    } catch (err: any) {
      alert(err?.message || 'บันทึก Note ไม่สำเร็จ')
    } finally {
      setSavingNote(false)
    }
  }

  const handleEditSuccess = (updatedState: ActualPortfolioStateDTO) => {
    onSuccess?.(updatedState)
    setEditingTx(null)
    loadTransactions()
  }

  const handleDeleteTransaction = async () => {
    if (!deletingTx) return
    setDeletingLoading(true)
    setDeletingError(null)
    try {
      const updatedState = await api.deleteTransaction(
        deletingTx.transaction_id,
        { adjust_cash: deleteAdjustCash },
        portfolioId
      )
      onSuccess?.(updatedState)
      setDeletingTx(null)
      loadTransactions()
    } catch (err: any) {
      setDeletingError(err?.message || 'ลบรายการ Transaction ไม่สำเร็จ')
    } finally {
      setDeletingLoading(false)
    }
  }

  const formatPrice = (price: number, currency: string) => {
    if (currency === 'USD') {
      return `$${price.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`
    }
    return `฿${price.toLocaleString('th-TH', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`
  }

  const formatSumm = (item: TransactionItemDTO) => {
    const isBuy = item.action.toUpperCase() === 'BUY'
    const sign = isBuy ? '-' : '+'

    if (currencyView === 'THB') {
      const summTHB = isBuy
        ? item.cost_thb
        : item.cost_thb + (item.realized_pnl_thb ?? 0)
      return `${sign}${formatTHB(summTHB)}`
    }

    const nativeAmount = item.units * item.price
    if (item.currency === 'USD') {
      return `${sign}$${nativeAmount.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`
    }
    return `${sign}฿${nativeAmount.toLocaleString('th-TH', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`
  }

  const renderSortIndicator = (field: SortField) => {
    if (sortField !== field) return <span className="opacity-20 ml-1">⇅</span>
    return <span className="text-flow-blue ml-1 font-bold">{sortAsc ? '▲' : '▼'}</span>
  }

  return (
    <div className="space-y-4 animate-fade-in">
      {/* Top Filter Banner when Symbol Filter is active */}
      {initialSymbol && (
        <div className="flex items-center justify-between rounded-xl border border-sky-200 bg-sky-50/80 px-4 py-3 text-sm text-sky-900 shadow-sm animate-fade-in">
          <div className="flex items-center gap-2">
            <span className="font-bold">🔍 กำลังกรองเฉพาะสินทรัพย์:</span>
            <span className="rounded-lg bg-flow-blue px-2.5 py-0.5 font-mono tabular-nums text-xs font-semibold text-white">
              {initialSymbol}
            </span>
            <span className="text-xs text-sky-700">({filteredItems.length} รายการ)</span>
          </div>
          {onClearSymbolFilter && (
            <button
              type="button"
              onClick={() => {
                setSearchQuery('')
                onClearSymbolFilter()
              }}
              className="rounded-lg border border-sky-300 bg-white px-3 py-1 text-xs font-semibold text-sky-700 shadow-2xs hover:bg-sky-50 transition-colors"
            >
              ล้างตัวกรอง
            </button>
          )}
        </div>
      )}

      {/* Header Bar: Actions & Summary Stats */}
      <div className="flex flex-wrap items-center justify-between gap-3 bg-panel rounded-2xl border border-sky-100 p-4 shadow-sm backdrop-blur-sm">
        {/* Left Side: + Add button & Stat Pills */}
        <div className="flex flex-wrap items-center gap-2.5">
          {onOpenTradeModal && (
            <button
              type="button"
              onClick={onOpenTradeModal}
              className="inline-flex items-center gap-1.5 rounded-xl bg-flow-blue px-4 py-2 text-xs sm:text-sm font-bold text-white shadow-md hover:bg-sky-600 active:scale-95 transition-all"
            >
              <PlusIcon className="w-4 h-4" />
              <span>Add</span>
            </button>
          )}

          {/* Stat Pills */}
          {summary && (
            <div className="flex flex-wrap items-center gap-2">
              <div className="inline-flex items-center gap-1.5 rounded-xl border border-sky-100 bg-sky-50/70 px-3 py-1.5 text-xs sm:text-sm font-medium text-sky-900">
                <span className="h-2.5 w-2.5 rounded-full bg-emerald-500"></span>
                <span>Buy</span>
                <span className="font-mono font-bold text-zinc-900">{formatTHB(summary.total_buy_thb)}</span>
                <span className="text-xs text-sky-700">({summary.total_buy_count} trades)</span>
              </div>

              {summary.total_sell_count > 0 && (
                <div className="inline-flex items-center gap-1.5 rounded-xl border border-sky-100 bg-sky-50/70 px-3 py-1.5 text-xs sm:text-sm font-medium text-sky-900">
                  <span className="h-2.5 w-2.5 rounded-full bg-rose-500"></span>
                  <span>Sell</span>
                  <span className="font-mono font-bold text-zinc-900">{formatTHB(summary.total_sell_thb)}</span>
                  <span className="text-xs text-sky-700">({summary.total_sell_count} trades)</span>
                </div>
              )}

              {summary.total_realized_pnl_thb !== 0 && (
                <div className={`inline-flex items-center gap-1.5 rounded-xl border px-3 py-1.5 text-xs sm:text-sm font-semibold ${
                  summary.total_realized_pnl_thb >= 0
                    ? 'border-emerald-200 bg-emerald-50 text-emerald-800'
                    : 'border-rose-200 bg-rose-50 text-rose-800'
                }`}>
                  <span>Realized PnL:</span>
                  <span className="font-mono font-bold">
                    {summary.total_realized_pnl_thb >= 0 ? '+' : ''}
                    {formatTHB(summary.total_realized_pnl_thb)}
                  </span>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Right Side: Filters, Search, Currency Toggle */}
        <div className="flex flex-wrap items-center gap-2">
          {/* Action Filter */}
          <div className="flex rounded-xl border border-sky-200 bg-sky-50/50 p-0.5 text-xs sm:text-sm">
            <button
              type="button"
              onClick={() => {
                setActionFilter('ALL')
                setCurrentPage(1)
              }}
              className={`rounded-lg px-2.5 py-1 font-bold transition-all ${
                actionFilter === 'ALL' ? 'bg-white text-zinc-900 shadow-2xs' : 'text-zinc-500 hover:text-zinc-800'
              }`}
            >
              All
            </button>
            <button
              type="button"
              onClick={() => {
                setActionFilter('BUY')
                setCurrentPage(1)
              }}
              className={`rounded-lg px-2.5 py-1 font-bold transition-all ${
                actionFilter === 'BUY' ? 'bg-emerald-500 text-white shadow-2xs' : 'text-zinc-500 hover:text-zinc-800'
              }`}
            >
              Buy
            </button>
            <button
              type="button"
              onClick={() => {
                setActionFilter('SELL')
                setCurrentPage(1)
              }}
              className={`rounded-lg px-2.5 py-1 font-bold transition-all ${
                actionFilter === 'SELL' ? 'bg-rose-500 text-white shadow-2xs' : 'text-zinc-500 hover:text-zinc-800'
              }`}
            >
              Sell
            </button>
          </div>

          {/* Search Box */}
          <div className="relative">
            <input
              type="text"
              placeholder="ค้นหา Symbol / Note..."
              value={searchQuery}
              onChange={(e) => {
                setSearchQuery(e.target.value)
                setCurrentPage(1)
              }}
              className="w-44 sm:w-56 rounded-xl border border-sky-200 bg-white px-3 py-1.5 text-xs sm:text-sm text-zinc-800 focus:border-flow-blue focus:outline-none shadow-2xs"
            />
            {searchQuery && (
              <button
                type="button"
                onClick={() => {
                  setSearchQuery('')
                  onClearSymbolFilter?.()
                }}
                className="absolute right-2.5 top-2 text-xs text-zinc-400 hover:text-zinc-600"
              >
                ✕
              </button>
            )}
          </div>

          {/* Currency Toggle */}
          <div className="flex rounded-xl border border-sky-200 bg-sky-50/50 p-0.5 text-xs sm:text-sm">
            <button
              type="button"
              onClick={() => setCurrencyView('NATIVE')}
              className={`rounded-lg px-2.5 py-1 font-bold transition-all ${
                currencyView === 'NATIVE' ? 'bg-white text-zinc-900 shadow-2xs' : 'text-zinc-500 hover:text-zinc-800'
              }`}
              title="แสดงตามสกุลเงินดั้งเดิมของสินทรัพย์ (USD/THB)"
            >
              Native
            </button>
            <button
              type="button"
              onClick={() => setCurrencyView('THB')}
              className={`rounded-lg px-2.5 py-1 font-bold transition-all ${
                currencyView === 'THB' ? 'bg-white text-zinc-900 shadow-2xs' : 'text-zinc-500 hover:text-zinc-800'
              }`}
              title="แปลงมูลค่าเป็น THB ทั้งหมด"
            >
              THB
            </button>
          </div>
        </div>
      </div>

      {/* Error Banner */}
      {error && (
        <div className="rounded-xl border border-rose-200 bg-rose-50 p-3 text-xs sm:text-sm text-rose-700">
          ⚠️ {error}
        </div>
      )}

      {/* Main Table */}
      <div className="overflow-hidden rounded-2xl border border-sky-100 bg-panel shadow-sm backdrop-blur-sm">
        <div className="overflow-x-auto">
          <table className="w-full text-left text-sm border-collapse min-w-[1050px]">
            <thead>
              <tr className="border-b border-sky-100 bg-sky-50/80 text-zinc-700 font-bold uppercase tracking-wider text-xs sm:text-sm select-none">
                <th className="py-3.5 px-4 w-12 text-center">#</th>
                <th className="py-3.5 px-4 w-28">Operation</th>
                <th
                  className="py-3.5 px-4 cursor-pointer hover:text-flow-blue transition-colors"
                  onClick={() => handleSort('symbol')}
                >
                  Holding {renderSortIndicator('symbol')}
                </th>
                <th
                  className="py-3.5 px-4 cursor-pointer hover:text-flow-blue transition-colors"
                  onClick={() => handleSort('timestamp')}
                >
                  Date {renderSortIndicator('timestamp')}
                </th>
                <th
                  className="py-3.5 px-4 text-right cursor-pointer hover:text-flow-blue transition-colors"
                  onClick={() => handleSort('units')}
                >
                  Shares {renderSortIndicator('units')}
                </th>
                <th
                  className="py-3.5 px-4 text-right cursor-pointer hover:text-flow-blue transition-colors"
                  onClick={() => handleSort('price')}
                >
                  Price {renderSortIndicator('price')}
                </th>
                <th className="py-3.5 px-4 text-right">Fee / Tax</th>
                <th
                  className="py-3.5 px-4 text-right cursor-pointer hover:text-flow-blue transition-colors"
                  onClick={() => handleSort('cost_thb')}
                >
                  Summ {renderSortIndicator('cost_thb')}
                </th>
                <th
                  className="py-3.5 px-4 text-right cursor-pointer hover:text-flow-blue transition-colors"
                  onClick={() => handleSort('realized_pnl_thb')}
                >
                  Realized PnL {renderSortIndicator('realized_pnl_thb')}
                </th>
                <th className="py-3.5 px-4 min-w-[220px]">Note</th>
                <th className="py-3.5 px-4 w-16 text-center">Action</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-sky-50">
              {paginatedItems.map((tx, idx) => {
                const isBuy = tx.action.toUpperCase() === 'BUY'
                const isEditing = editingTxId === tx.transaction_id
                const rowNum = (currentPage - 1) * pageSize + idx + 1

                return (
                  <tr
                    key={tx.transaction_id || `${tx.timestamp}-${idx}`}
                    className="hover:bg-sky-50/40 transition-colors group"
                  >
                    <td className="py-3.5 px-4 text-center text-zinc-400 font-mono text-xs">
                      {rowNum}
                    </td>

                    {/* Operation */}
                    <td className="py-3.5 px-4">
                      <span
                        className={`inline-flex items-center gap-1.5 rounded-md px-2.5 py-1 text-xs font-bold ${
                          isBuy
                            ? 'bg-emerald-50 text-emerald-700 border border-emerald-200'
                            : 'bg-rose-50 text-rose-700 border border-rose-200'
                        }`}
                      >
                        <span
                          className={`h-2 w-2 rounded-full ${isBuy ? 'bg-emerald-500' : 'bg-rose-500'}`}
                        ></span>
                        {isBuy ? 'Buy' : 'Sell'}
                      </span>
                    </td>

                    {/* Holding Symbol */}
                    <td className="py-3.5 px-4">
                      <div className="flex flex-col">
                        <span className="font-extrabold text-zinc-900 font-mono text-base tracking-tight">
                          {tx.symbol}
                        </span>
                        <span className="text-xs text-zinc-400 font-medium">
                          {tx.currency} {tx.fx_rate ? `@ ${tx.fx_rate.toFixed(2)}` : ''}
                        </span>
                      </div>
                    </td>

                    {/* Date */}
                    <td className="py-3.5 px-4 text-zinc-600 font-mono tabular-nums text-xs sm:text-sm whitespace-nowrap">
                      {tx.timestamp.replace('T', ' ')}
                    </td>

                    {/* Shares */}
                    <td className="py-3.5 px-4 text-right font-mono tabular-nums font-bold text-zinc-900 text-sm sm:text-base">
                      {tx.units.toLocaleString(undefined, { maximumFractionDigits: 6 })}
                    </td>

                    {/* Price */}
                    <td className="py-3.5 px-4 text-right font-mono tabular-nums font-semibold text-zinc-800 text-sm sm:text-base">
                      {formatPrice(tx.price, tx.currency)}
                    </td>

                    {/* Fee / Tax */}
                    <td className="py-3.5 px-4 text-right font-mono tabular-nums text-zinc-400 text-xs">
                      {tx.currency === 'USD' ? '$0.00' : '฿0.00'}
                    </td>

                    {/* Summ */}
                    <td className="py-3.5 px-4 text-right font-mono tabular-nums font-extrabold text-sm sm:text-base">
                      <span className={isBuy ? 'text-zinc-900' : 'text-emerald-700'}>
                        {formatSumm(tx)}
                      </span>
                    </td>

                    {/* Realized PnL */}
                    <td className="py-3.5 px-4 text-right font-mono tabular-nums text-sm sm:text-base">
                      {tx.realized_pnl_thb !== null && tx.realized_pnl_thb !== undefined ? (
                        <span
                          className={`font-bold ${
                            tx.realized_pnl_thb >= 0 ? 'text-emerald-600' : 'text-rose-600'
                          }`}
                        >
                          {tx.realized_pnl_thb >= 0 ? '+' : ''}
                          {formatTHB(tx.realized_pnl_thb)}
                        </span>
                      ) : (
                        <span className="text-zinc-300">-</span>
                      )}
                    </td>

                    {/* Note (with inline edit) */}
                    <td className="py-3.5 px-4">
                      {isEditing ? (
                        <div className="flex items-center gap-1.5">
                          <input
                            type="text"
                            value={editingNote}
                            onChange={(e) => setEditingNote(e.target.value)}
                            onKeyDown={(e) => {
                              if (e.key === 'Enter') saveNote(tx.transaction_id)
                              if (e.key === 'Escape') cancelEditNote()
                            }}
                            autoFocus
                            disabled={savingNote}
                            className="w-full rounded-lg border border-flow-blue bg-white px-2.5 py-1 text-xs sm:text-sm text-zinc-800 focus:outline-none shadow-2xs"
                            placeholder="ระบุบันทึกช่วยจำ..."
                          />
                          <button
                            type="button"
                            onClick={() => saveNote(tx.transaction_id)}
                            disabled={savingNote}
                            className="rounded-md bg-emerald-600 p-1 text-white hover:bg-emerald-700 transition-colors"
                            title="บันทึก (Enter)"
                          >
                            ✓
                          </button>
                          <button
                            type="button"
                            onClick={cancelEditNote}
                            disabled={savingNote}
                            className="rounded-md bg-zinc-200 p-1 text-zinc-600 hover:bg-zinc-300 transition-colors"
                            title="ยกเลิก (Esc)"
                          >
                            ✕
                          </button>
                        </div>
                      ) : (
                        <button
                          type="button"
                          onClick={() => startEditNote(tx)}
                          className="w-full text-left cursor-pointer group/note rounded-lg px-2.5 py-1.5 hover:bg-sky-50 transition-colors"
                          title="คลิกเพื่อแก้ไข Note ด่วน"
                        >
                          {tx.notes ? (
                            <span className="text-xs sm:text-sm text-zinc-800 font-medium leading-snug break-words">
                              {tx.notes}
                            </span>
                          ) : (
                            <span className="text-xs text-zinc-400 italic group-hover/note:text-zinc-600">
                              + เพิ่มบันทึกช่วยจำ...
                            </span>
                          )}
                        </button>
                      )}
                    </td>

                    {/* Action Buttons */}
                    <td className="py-3.5 px-4 text-center">
                      {!isEditing && (
                        <div className="flex items-center justify-center gap-1.5">
                          <button
                            type="button"
                            onClick={() => setEditingTx(tx)}
                            className="text-zinc-400 hover:text-flow-blue transition-colors p-2 rounded-lg hover:bg-sky-100 shadow-2xs"
                            title="แก้ไขข้อมูล Transaction (วัน, จำนวนหุ้น, ราคา, Note)"
                          >
                            <EditIcon className="w-4 h-4" />
                          </button>
                          <button
                            type="button"
                            onClick={() => {
                              setDeletingTx(tx)
                              setDeleteAdjustCash(true)
                              setDeletingError(null)
                            }}
                            className="text-zinc-400 hover:text-rose-600 transition-colors p-2 rounded-lg hover:bg-rose-50 shadow-2xs"
                            title="ลบรายการ Transaction"
                          >
                            <DeleteIcon className="w-4 h-4" />
                          </button>
                        </div>
                      )}
                    </td>
                  </tr>
                )
              })}

              {/* Empty State */}
              {paginatedItems.length === 0 && !loading && (
                <tr>
                  <td colSpan={11} className="py-16 text-center">
                    <div className="mx-auto max-w-sm flex flex-col items-center justify-center space-y-3">
                      <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-sky-50 border border-sky-100 text-flow-blue">
                        <TradeIcon className="h-6 w-6" />
                      </div>
                      <div className="space-y-1">
                        <h4 className="text-sm font-bold text-zinc-900">
                          {searchQuery || actionFilter !== 'ALL'
                            ? 'ไม่พบรายการ Transactions ตามเงื่อนไข'
                            : 'ยังไม่มีประวัติการซื้อขายในพอร์ตนี้'}
                        </h4>
                        <p className="text-xs text-zinc-500">
                          {searchQuery || actionFilter !== 'ALL'
                            ? 'ลองล้างตัวกรองหรือคำค้นหาเพื่อดูรายการทั้งหมด'
                            : 'เริ่มต้นบันทึกการซื้อขายหุ้นหรือสินทรัพย์ตัวแรกเข้าสู่ระบบ'}
                        </p>
                      </div>
                      {onOpenTradeModal && !searchQuery && actionFilter === 'ALL' && (
                        <button
                          type="button"
                          onClick={onOpenTradeModal}
                          className="inline-flex items-center gap-1.5 rounded-xl bg-flow-blue px-4 py-2 text-xs font-bold text-white shadow-md hover:bg-sky-600 transition-all"
                        >
                          <PlusIcon className="w-3.5 h-3.5" />
                          <span>บันทึกการซื้อขายแรก</span>
                        </button>
                      )}
                    </div>
                  </td>
                </tr>
              )}
            </tbody>

            {/* Table Footer: Total Summary */}
            {filteredItems.length > 0 && (
              <tfoot>
                <tr className="border-t-2 border-sky-200 bg-sky-50/70 font-bold text-zinc-900">
                  <td className="py-3 px-4 text-center text-zinc-400">∑</td>
                  <td className="py-3 px-4" colSpan={3}>
                    Total ({filteredItems.length} รายการ)
                  </td>
                  <td className="py-3 px-4 text-right font-mono tabular-nums">
                    {filteredTotals.totalShares.toLocaleString(undefined, { maximumFractionDigits: 6 })}
                  </td>
                  <td className="py-3 px-4 text-right" colSpan={2} aria-hidden="true">&nbsp;</td>
                  <td className="py-3 px-4 text-right font-mono tabular-nums text-zinc-900">
                    {actionFilter === 'BUY'
                      ? `-${formatTHB(filteredTotals.totalBuyTHB)}`
                      : actionFilter === 'SELL'
                      ? `+${formatTHB(filteredTotals.totalSellTHB)}`
                      : filteredTotals.totalSellTHB === 0
                      ? `-${formatTHB(filteredTotals.totalBuyTHB)}`
                      : filteredTotals.totalBuyTHB === 0
                      ? `+${formatTHB(filteredTotals.totalSellTHB)}`
                      : `${filteredTotals.totalNetTHB >= 0 ? '+' : '-'}${formatTHB(Math.abs(filteredTotals.totalNetTHB))}`}
                  </td>
                  <td className="py-3 px-4 text-right font-mono tabular-nums">
                    {filteredTotals.totalRealizedPnL !== 0 ? (
                      <span
                        className={
                          filteredTotals.totalRealizedPnL >= 0 ? 'text-emerald-700' : 'text-rose-700'
                        }
                      >
                        {filteredTotals.totalRealizedPnL >= 0 ? '+' : ''}
                        {formatTHB(filteredTotals.totalRealizedPnL)}
                      </span>
                    ) : (
                      <span className="text-zinc-400">-</span>
                    )}
                  </td>
                  <td className="py-3 px-4" colSpan={2} aria-hidden="true">&nbsp;</td>
                </tr>
              </tfoot>
            )}
          </table>
        </div>

        {/* Pagination & Controls Bar */}
        {filteredItems.length > 0 && (
          <div className="flex flex-wrap items-center justify-between gap-3 border-t border-sky-100 bg-white/60 px-4 py-3 text-xs text-zinc-500">
            {/* Page Size */}
            <div className="flex items-center gap-2">
              <span>แสดงแถวต่อหน้า:</span>
              <select
                value={pageSize}
                onChange={(e) => {
                  setPageSize(Number(e.target.value))
                  setCurrentPage(1)
                }}
                className="rounded-lg border border-sky-200 bg-white px-2 py-1 text-xs text-zinc-800 focus:border-flow-blue focus:outline-none shadow-2xs"
              >
                <option value={10}>10</option>
                <option value={25}>25</option>
                <option value={50}>50</option>
                <option value={100}>100</option>
              </select>
            </div>

            {/* Page indicator & navigation buttons */}
            <div className="flex items-center gap-3">
              <span>
                See {Math.min((currentPage - 1) * pageSize + 1, filteredItems.length)}-
                {Math.min(currentPage * pageSize, filteredItems.length)} from {filteredItems.length}
              </span>
              <div className="flex items-center gap-1">
                <button
                  type="button"
                  disabled={currentPage <= 1}
                  onClick={() => setCurrentPage((p) => Math.max(1, p - 1))}
                  className="rounded-lg border border-sky-200 bg-white px-2.5 py-1 text-xs font-semibold text-zinc-700 shadow-2xs hover:bg-sky-50 disabled:opacity-40 disabled:cursor-not-allowed"
                >
                  ◀
                </button>
                <span className="px-2 font-mono font-bold text-zinc-800">
                  {currentPage} / {totalPages}
                </span>
                <button
                  type="button"
                  disabled={currentPage >= totalPages}
                  onClick={() => setCurrentPage((p) => Math.min(totalPages, p + 1))}
                  className="rounded-lg border border-sky-200 bg-white px-2.5 py-1 text-xs font-semibold text-zinc-700 shadow-2xs hover:bg-sky-50 disabled:opacity-40 disabled:cursor-not-allowed"
                >
                  ▶
                </button>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Edit Transaction Modal */}
      {editingTx && (
        <EditTransactionModal
          transaction={editingTx}
          selectedPortfolioId={portfolioId}
          holdings={holdings}
          onClose={() => setEditingTx(null)}
          onSuccess={handleEditSuccess}
        />
      )}

      {/* Delete Confirmation Modal */}
      {deletingTx && (
        <Modal
          titleId="delete-tx-title"
          onClose={() => setDeletingTx(null)}
          panelClassName="max-w-md rounded-2xl border border-rose-100 bg-white p-6 shadow-2xl"
        >
          <div className="flex items-center justify-between border-b border-zinc-100 pb-3">
            <h3 id="delete-tx-title" className="flex items-center gap-2 text-base font-bold text-zinc-900">
              <DeleteIcon className="w-5 h-5 text-rose-500" />
              <span>ยืนยันการลบรายการ</span>
            </h3>
            <button
              type="button"
              onClick={() => setDeletingTx(null)}
              className="rounded-lg p-1 text-zinc-400 hover:bg-zinc-100 hover:text-zinc-600 transition-colors"
            >
              ✕
            </button>
          </div>

          <div className="mt-4 space-y-4 text-xs">
            {deletingError && (
              <div className="rounded-xl border border-rose-200 bg-rose-50 p-3 font-semibold text-rose-800">
                ⚠️ {deletingError}
              </div>
            )}

            <div className="rounded-xl border border-zinc-200 bg-zinc-50/70 p-3 space-y-1.5">
              <div className="flex justify-between">
                <span className="text-zinc-500">สินทรัพย์:</span>
                <span className="font-bold text-zinc-900">{deletingTx.symbol} ({deletingTx.action})</span>
              </div>
              <div className="flex justify-between">
                <span className="text-zinc-500">จำนวน:</span>
                <span className="font-mono font-semibold text-zinc-900">{deletingTx.units} units</span>
              </div>
              <div className="flex justify-between">
                <span className="text-zinc-500">ราคา:</span>
                <span className="font-mono text-zinc-900">{formatPrice(deletingTx.price, deletingTx.currency)}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-zinc-500">วันที่:</span>
                <span className="font-mono text-zinc-600">{deletingTx.timestamp}</span>
              </div>
            </div>

            <div className="rounded-xl border border-amber-200 bg-amber-50/80 p-3 text-amber-900 text-[11px] leading-relaxed">
              ⚠️ <strong>ข้อควรระวัง:</strong> การลบรายการนี้จะทำให้ระบบทำการคำนวณย้อนหลัง (Replay) ต้นทุนถัวเฉลี่ยและกำไร/ขาดทุนสะสมของ {deletingTx.symbol} ใหม่ทั้งหมด
            </div>

            <div className="flex items-start gap-2 select-none">
              <input
                id="delete-tx-adjust-cash"
                type="checkbox"
                checked={deleteAdjustCash}
                onChange={(e) => setDeleteAdjustCash(e.target.checked)}
                className="mt-0.5 h-4 w-4 rounded border-zinc-300 text-flow-blue focus:ring-flow-blue cursor-pointer"
              />
              <div className="space-y-0.5">
                <label htmlFor="delete-tx-adjust-cash" className="text-xs font-semibold text-zinc-900 cursor-pointer block">
                  ปรับปรุงยอดเงินสด (CASH_{deletingTx.currency}) คืนกลับ/หักออก อัตโนมัติ
                </label>
                <p className="text-[11px] text-zinc-500">
                  {deletingTx.action.toUpperCase() === 'BUY'
                    ? 'คืนเงินสดที่เคยใช้ซื้อกลับเข้าพอร์ต'
                    : 'หักเงินสดที่เคยได้รับจากการขายออกจากพอร์ต'}
                </p>
              </div>
            </div>

            <div className="flex justify-end gap-2 border-t border-zinc-100 pt-4">
              <button
                type="button"
                onClick={() => setDeletingTx(null)}
                disabled={deletingLoading}
                className="rounded-xl border border-zinc-300 bg-white px-4 py-2 font-semibold text-zinc-700 hover:bg-zinc-50 transition-colors"
              >
                ยกเลิก
              </button>
              <button
                type="button"
                onClick={handleDeleteTransaction}
                disabled={deletingLoading}
                className="rounded-xl bg-rose-600 px-5 py-2 font-bold text-white shadow-md hover:bg-rose-700 transition-colors disabled:opacity-50"
              >
                {deletingLoading ? 'กำลังลบและ Replay...' : 'ยืนยันการลบ'}
              </button>
            </div>
          </div>
        </Modal>
      )}
    </div>
  )
}
