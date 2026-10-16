import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import type { KeyboardEvent } from 'react'
import type { ActualHoldingDTO, ActualPortfolioStateDTO, TransactionItemDTO, TransactionSummaryDTO } from '../../api/types'
import { api } from '../../api/client'
import { buildPvdStatementHistory, partitionTransactions, type TransactionView } from '../../lib/pvdStatementHistory'
import AssetTransactionsTable from './AssetTransactionsTable'
import PvdStatementsTable from './PvdStatementsTable'

interface Props {
  portfolioId: string
  activeView: TransactionView
  onViewChange: (view: TransactionView) => void
  initialSymbol?: string | null
  holdings?: ActualHoldingDTO[]
  onClearSymbolFilter?: () => void
  onOpenTradeModal?: () => void
  onSuccess?: (state: ActualPortfolioStateDTO) => void
}

export default function PortfolioTransactionsTab({
  portfolioId,
  activeView,
  onViewChange,
  initialSymbol,
  holdings,
  onClearSymbolFilter,
  onOpenTradeModal,
  onSuccess,
}: Props) {
  const [transactions, setTransactions] = useState<TransactionItemDTO[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const generation = useRef(0)
  const initialLoadStarted = useRef(false)
  const scopeRef = useRef({ portfolioId, mounted: true })
  const tabRefs = useRef<Array<HTMLButtonElement | null>>([])

  if (scopeRef.current.portfolioId !== portfolioId) {
    scopeRef.current = { portfolioId, mounted: true }
    initialLoadStarted.current = false
  }
  const scope = scopeRef.current

  const refreshTransactions = useCallback(async () => {
    const requestScope = scopeRef.current
    if (!requestScope.mounted) return
    const requestGeneration = ++generation.current
    setLoading(true)
    setError(null)
    try {
      const response = await api.getTransactions(requestScope.portfolioId)
      if (scopeRef.current !== requestScope || !requestScope.mounted || generation.current !== requestGeneration) return
      if (response.portfolio_id !== requestScope.portfolioId) throw new Error('Transactions ตอบกลับมาจากพอร์ตที่ไม่ตรงกัน')
      setTransactions(response.transactions || [])
    } catch (err) {
      if (scopeRef.current !== requestScope || !requestScope.mounted || generation.current !== requestGeneration) return
      setError(err instanceof Error ? err.message : 'ไม่สามารถโหลดข้อมูล Transactions ได้')
    } finally {
      if (scopeRef.current === requestScope && requestScope.mounted && generation.current === requestGeneration) setLoading(false)
    }
  }, [])

  useEffect(() => {
    const requestScope = scopeRef.current
    requestScope.mounted = true
    setTransactions([])
    if (!initialLoadStarted.current) {
      initialLoadStarted.current = true
      void refreshTransactions()
    }
    return () => {
      requestScope.mounted = false
    }
  }, [portfolioId, refreshTransactions])

  const requestSuccess = useCallback((state: ActualPortfolioStateDTO) => {
    if (scopeRef.current === scope && scope.mounted) onSuccess?.(state)
  }, [onSuccess, scope])

  const { assets, pvd } = useMemo(() => partitionTransactions(transactions), [transactions])
  const assetSummary = useMemo<TransactionSummaryDTO>(() => ({
    total_buy_count: assets.filter((tx) => tx.action.toUpperCase() === 'BUY').length,
    total_sell_count: assets.filter((tx) => tx.action.toUpperCase() === 'SELL').length,
    total_buy_thb: assets.filter((tx) => tx.action.toUpperCase() === 'BUY').reduce((sum, tx) => sum + tx.cost_thb, 0),
    total_sell_thb: assets.filter((tx) => tx.action.toUpperCase() === 'SELL').reduce((sum, tx) => sum + tx.cost_thb + (tx.realized_pnl_thb ?? 0), 0),
    total_realized_pnl_thb: assets.reduce((sum, tx) => sum + (tx.realized_pnl_thb ?? 0), 0),
  }), [assets])
  const pvdHistory = useMemo(() => buildPvdStatementHistory(portfolioId, pvd), [portfolioId, pvd])

  const selectView = (view: TransactionView) => onViewChange(view)
  const handleTabKeyDown = (event: KeyboardEvent<HTMLButtonElement>) => {
    const keys = ['ArrowLeft', 'ArrowRight', 'Home', 'End']
    if (!keys.includes(event.key)) return
    event.preventDefault()
    const nextIndex = event.key === 'Home' ? 0
      : event.key === 'End' ? 1
        : event.key === 'ArrowRight' ? (activeView === 'assets' ? 1 : 0)
          : (activeView === 'assets' ? 1 : 0)
    const nextView: TransactionView = nextIndex === 0 ? 'assets' : 'pvd'
    selectView(nextView)
    tabRefs.current[nextIndex]?.focus()
  }

  return (
    <div className="space-y-4">
      <div role="tablist" aria-label="ประเภทรายการ Transactions" className="flex w-fit rounded-xl border border-sky-200 bg-sky-50/70 p-1">
        {([
          { id: 'assets', label: 'สินทรัพย์' },
          { id: 'pvd', label: 'PVD' },
        ] as const).map(({ id, label }, index) => (
          <button
            key={id}
            ref={(element) => { tabRefs.current[index] = element }}
            id={`transactions-tab-${id}`}
            type="button"
            role="tab"
            aria-selected={activeView === id}
            aria-controls={`transactions-panel-${id}`}
            tabIndex={activeView === id ? 0 : -1}
            onClick={() => selectView(id)}
            onKeyDown={handleTabKeyDown}
            className={`rounded-lg px-4 py-2 text-sm font-bold transition-colors ${activeView === id ? 'bg-white text-flow-blue shadow-sm' : 'text-zinc-500 hover:text-zinc-800'}`}
          >
            {label}
          </button>
        ))}
      </div>

      {error && (
        <div role="alert" className="flex flex-wrap items-center justify-between gap-3 rounded-xl border border-rose-200 bg-rose-50 p-3 text-sm text-rose-800">
          <span>{error}</span>
          <button type="button" onClick={() => void refreshTransactions()} disabled={loading} className="rounded-lg border border-rose-300 bg-white px-3 py-1.5 text-xs font-bold disabled:opacity-50">ลองโหลดอีกครั้ง</button>
        </div>
      )}

      <section id="transactions-panel-assets" role="tabpanel" aria-labelledby="transactions-tab-assets" hidden={activeView !== 'assets'}>
        <AssetTransactionsTable
          portfolioId={portfolioId}
          transactions={assets}
          summary={assetSummary}
          loading={loading}
          active={activeView === 'assets'}
          onRefresh={refreshTransactions}
          initialSymbol={activeView === 'assets' ? initialSymbol : null}
          holdings={holdings}
          onClearSymbolFilter={onClearSymbolFilter}
          onOpenTradeModal={onOpenTradeModal}
          onSuccess={requestSuccess}
        />
      </section>

      <section id="transactions-panel-pvd" role="tabpanel" aria-labelledby="transactions-tab-pvd" hidden={activeView !== 'pvd'}>
        <PvdStatementsTable
          portfolioId={portfolioId}
          transactions={pvd}
          history={pvdHistory}
          active={activeView === 'pvd'}
          loading={loading}
          initialSymbol={activeView === 'pvd' ? initialSymbol : null}
          onClearSymbolFilter={onClearSymbolFilter}
          onRefresh={refreshTransactions}
          onSuccess={requestSuccess}
        />
      </section>
    </div>
  )
}
