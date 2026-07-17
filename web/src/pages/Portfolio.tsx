import { useState, useEffect, useCallback } from 'react'
import { useSearchParams } from 'react-router-dom'
import { api } from '../api/client'
import type {
  ActualPortfolioStateDTO,
  BucketAllocationResponseDTO,
  ActualWatchlistStateDTO,
  ActualGoalsResponseDTO,
  PerformanceSnapshotDTO,
  JournalEntryDTO,
} from '../api/types'
import SegmentedControl from '../components/ui/SegmentedControl'
import PortfolioSummaryCards from '../components/portfolio/PortfolioSummaryCards'
import PortfolioOverviewTab from '../components/portfolio/PortfolioOverviewTab'
import PortfolioHoldingsTab from '../components/portfolio/PortfolioHoldingsTab'
import PortfolioWatchlistTab from '../components/portfolio/PortfolioWatchlistTab'
import PortfolioGoalsTab from '../components/portfolio/PortfolioGoalsTab'
import PortfolioAnalyticsTab from '../components/portfolio/PortfolioAnalyticsTab'
import TradeModal from '../components/portfolio/Modals/TradeModal'
import CashFlowModal from '../components/portfolio/Modals/CashFlowModal'
import IncomeModal from '../components/portfolio/Modals/IncomeModal'
import ResetConfirmModal from '../components/portfolio/Modals/ResetConfirmModal'
import {
  TradeIcon,
  CashFlowIcon,
  IncomeIcon,
  ResetIcon,
  ChartBarIcon,
  BriefcaseIcon,
  WatchlistIcon,
  GoalIcon,
} from '../components/portfolio/icons/PortfolioIcons'

const TABS = [
  {
    key: 'overview',
    label: (
      <span className="flex items-center gap-1.5">
        <ChartBarIcon className="w-4 h-4 text-flow-blue" />
        <span>Strategy Buckets & Allocation</span>
      </span>
    ),
  },
  {
    key: 'holdings',
    label: (
      <span className="flex items-center gap-1.5">
        <BriefcaseIcon className="w-4 h-4 text-emerald-600" />
        <span>Holdings & Trading Journal</span>
      </span>
    ),
  },
  {
    key: 'watchlist',
    label: (
      <span className="flex items-center gap-1.5">
        <WatchlistIcon className="w-4 h-4 text-amber-500" />
        <span>Watchlist</span>
      </span>
    ),
  },
]

export default function Portfolio() {
  const [searchParams, setSearchParams] = useSearchParams()
  const activeTab = searchParams.get('tab') || 'overview'
  const selectedBucket = searchParams.get('bucket') || null

  // State
  const [portfolioState, setPortfolioState] = useState<ActualPortfolioStateDTO | null>(null)
  const [allocationsResponse, setAllocationsResponse] = useState<BucketAllocationResponseDTO | null>(null)
  const [watchlistState, setWatchlistState] = useState<ActualWatchlistStateDTO | null>(null)
  const [goalsResponse, setGoalsResponse] = useState<ActualGoalsResponseDTO | null>(null)
  const [performanceRows, setPerformanceRows] = useState<PerformanceSnapshotDTO[]>([])
  const [journalRows, setJournalRows] = useState<JournalEntryDTO[]>([])

  // Filters state
  const [daysRange, setDaysRange] = useState<number | undefined>(30)
  const [journalKeyword, setJournalKeyword] = useState<string>('')

  // Top-Level Executive Dashboard state
  const [executiveTab, setExecutiveTab] = useState<'goals' | 'performance'>('goals')
  const [executiveExpanded, setExecutiveExpanded] = useState<boolean>(true)

  // Modals state
  const [tradeModalOpen, setTradeModalOpen] = useState(false)
  const [cashFlowModalOpen, setCashFlowModalOpen] = useState(false)
  const [incomeModalOpen, setIncomeModalOpen] = useState(false)
  const [resetModalOpen, setResetModalOpen] = useState(false)

  const handlePortfolioStateSuccess = (newState: ActualPortfolioStateDTO) => {
    setPortfolioState(newState)
    api.getActualBucketAllocations().then((allocRes) => setAllocationsResponse(allocRes)).catch(() => {})
  }

  // Loading & Error
  const [loading, setLoading] = useState<boolean>(true)
  const [refreshingPrices, setRefreshingPrices] = useState<boolean>(false)
  const [error, setError] = useState<string | null>(null)

  const fetchAllData = useCallback(async (refreshPrices: boolean = false) => {
    try {
      if (refreshPrices) {
        setRefreshingPrices(true)
      } else {
        setLoading(true)
      }
      setError(null)

      const [pState, allocRes, wState, gRes] = await Promise.all([
        api.getActualPortfolioState(refreshPrices, false),
        api.getActualBucketAllocations(),
        api.getActualWatchlist(),
        api.getActualGoals(),
      ])

      setPortfolioState(pState)
      setAllocationsResponse(allocRes)
      setWatchlistState(wState)
      setGoalsResponse(gRes)

      if (refreshPrices) {
        api.getActualPerformance(daysRange).then((rows) => setPerformanceRows(rows)).catch(() => {})
      }
    } catch (err: any) {
      setError(err?.message || 'ไม่สามารถโหลดข้อมูล Actual Portfolio ได้')
    } finally {
      setLoading(false)
      setRefreshingPrices(false)
    }
  }, [])

  // Initial load
  useEffect(() => {
    void fetchAllData(false)
  }, [fetchAllData])

  // Load performance when daysRange changes or initial load
  useEffect(() => {
    api
      .getActualPerformance(daysRange)
      .then((rows) => setPerformanceRows(rows))
      .catch(() => setPerformanceRows([]))
  }, [daysRange])

  // Load journal when keyword changes or initial load
  useEffect(() => {
    const timer = setTimeout(() => {
      api
        .getActualJournal(365, journalKeyword || undefined, 100)
        .then((rows) => setJournalRows(rows))
        .catch(() => setJournalRows([]))
    }, 300)
    return () => clearTimeout(timer)
  }, [journalKeyword])

  // Auto-redirect legacy tabs to overview or holdings
  useEffect(() => {
    if (activeTab === 'goals' || activeTab === 'analytics') {
      const nextParams = new URLSearchParams(searchParams)
      nextParams.set('tab', 'overview')
      setSearchParams(nextParams, { replace: true })
    }
  }, [activeTab, searchParams, setSearchParams])

  const handleTabChange = (key: string) => {
    const nextParams = new URLSearchParams(searchParams)
    nextParams.set('tab', key)
    setSearchParams(nextParams)
  }

  const handleSelectBucket = (bucketId: string) => {
    const nextParams = new URLSearchParams(searchParams)
    nextParams.set('tab', 'holdings')
    nextParams.set('bucket', bucketId)
    setSearchParams(nextParams)
  }

  const handleClearBucketFilter = () => {
    const nextParams = new URLSearchParams(searchParams)
    nextParams.delete('bucket')
    setSearchParams(nextParams)
  }

  return (
    <div className="animate-page-in space-y-6 pb-14">
      {/* Page Header */}
      <div className="flex flex-col justify-between gap-4 sm:flex-row sm:items-center">
        <div>
          <h1 className="text-2xl font-extrabold tracking-tight text-zinc-900">Actual Portfolio Hub</h1>
          <p className="mt-0.5 text-sm text-zinc-500">
            ศูนย์กลางติดตามพอร์ตการลงทุนจริง (Actual Portfolio) พร้อมกลยุทธ์สัดส่วน การจัดกลุ่ม Bucket และระบบบันทึก Trading Journal
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <button
            type="button"
            onClick={() => setTradeModalOpen(true)}
            className="rounded-xl bg-flow-blue px-3.5 py-2 text-xs font-bold text-white shadow-md hover:bg-sky-600 transition-colors flex items-center gap-1.5"
          >
            <TradeIcon className="w-4 h-4" />
            <span>บันทึกเทรด (Trade)</span>
          </button>
          <button
            type="button"
            onClick={() => setCashFlowModalOpen(true)}
            className="rounded-xl border border-sky-200 bg-white px-3.5 py-2 text-xs font-bold text-zinc-700 shadow-sm hover:bg-sky-50 transition-colors flex items-center gap-1.5"
          >
            <CashFlowIcon className="w-4 h-4 text-emerald-600" />
            <span>ฝาก/ถอน (Cash)</span>
          </button>
          <button
            type="button"
            onClick={() => setIncomeModalOpen(true)}
            className="rounded-xl border border-sky-200 bg-white px-3.5 py-2 text-xs font-bold text-zinc-700 shadow-sm hover:bg-sky-50 transition-colors flex items-center gap-1.5"
          >
            <IncomeIcon className="w-4 h-4 text-emerald-600" />
            <span>ปันผล/รายรับ (Income)</span>
          </button>
          <button
            type="button"
            onClick={() => setResetModalOpen(true)}
            className="rounded-xl border border-rose-200 bg-rose-50 px-3 py-2 text-xs font-bold text-rose-600 shadow-sm hover:bg-rose-600 hover:text-white transition-colors ml-2 flex items-center gap-1"
            title="ล้างข้อมูลพอร์ตทั้งหมดกลับเป็นค่าเริ่มต้น"
          >
            <ResetIcon className="w-3.5 h-3.5" />
            <span>ล้างพอร์ต</span>
          </button>
        </div>
      </div>

      {/* Error Alert */}
      {error && (
        <div className="flex items-center justify-between rounded-2xl border border-rose-200 bg-rose-50 p-4 text-rose-900 shadow-sm animate-fade-in">
          <div className="flex items-center gap-3">
            <span className="text-xl">⚠️</span>
            <span className="text-sm font-semibold">{error}</span>
          </div>
          <button
            type="button"
            onClick={() => void fetchAllData(false)}
            className="rounded-xl bg-rose-600 px-3 py-1.5 text-xs font-semibold text-white hover:bg-rose-700"
          >
            ลองใหม่อีกครั้ง
          </button>
        </div>
      )}

      {/* Summary Cards */}
      <PortfolioSummaryCards
        summary={portfolioState?.summary ?? null}
        lastUpdated={portfolioState?.last_updated ?? null}
        loading={loading}
        refreshingPrices={refreshingPrices}
        priceRefreshInfo={portfolioState?.price_refresh_info ?? null}
        onRefreshPrices={() => void fetchAllData(true)}
        performanceRows={performanceRows}
      />

      {/* Top-Level Executive Section: Goals & Performance */}
      <div className="rounded-2xl border border-sky-100 bg-gradient-to-b from-sky-50/40 via-white to-panel p-5 shadow-sm transition-all">
        <div className="flex flex-col justify-between gap-4 sm:flex-row sm:items-center border-b border-sky-100 pb-4">
          <div className="flex items-center gap-3">
            <span className="flex h-10 w-10 items-center justify-center rounded-xl bg-flow-blue/10 text-flow-blue">
              <ChartBarIcon className="h-5 w-5" />
            </span>
            <div>
              <h2 className="text-base font-extrabold tracking-tight text-zinc-900 flex items-center gap-2">
                <span>Executive Dashboard: Goals & Performance</span>
                <span className="rounded-full bg-flow-cyan/20 px-2.5 py-0.5 text-[10px] font-bold text-flow-blue uppercase tracking-wider">
                  Top-Level Metrics
                </span>
              </h2>
              <p className="text-xs text-zinc-500 mt-0.5">
                ติดตามความคืบหน้าเป้าหมายพอร์ต และประวัติการเติบโตของ NAV Snapshot ได้ทันทีจากทุกมุมมอง
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={() => {
                setExecutiveTab('goals')
                if (!executiveExpanded) setExecutiveExpanded(true)
              }}
              className={`rounded-xl px-3.5 py-1.5 text-xs font-bold transition-all flex items-center gap-1.5 ${
                executiveTab === 'goals' && executiveExpanded
                  ? 'bg-flow-blue text-white shadow-md'
                  : 'bg-white text-zinc-600 hover:bg-sky-50 border border-sky-200'
              }`}
            >
              <GoalIcon className="h-4 w-4 shrink-0" />
              <span>Portfolio Goals ({goalsResponse?.goals?.length ?? 0})</span>
            </button>
            <button
              type="button"
              onClick={() => {
                setExecutiveTab('performance')
                if (!executiveExpanded) setExecutiveExpanded(true)
              }}
              className={`rounded-xl px-3.5 py-1.5 text-xs font-bold transition-all flex items-center gap-1.5 ${
                executiveTab === 'performance' && executiveExpanded
                  ? 'bg-flow-blue text-white shadow-md'
                  : 'bg-white text-zinc-600 hover:bg-sky-50 border border-sky-200'
              }`}
            >
              <ChartBarIcon className="h-4 w-4 shrink-0" />
              <span>Performance History ({performanceRows.length} points)</span>
            </button>
            <button
              type="button"
              onClick={() => setExecutiveExpanded(!executiveExpanded)}
              className="rounded-xl border border-sky-200 bg-white px-3 py-1.5 text-xs font-semibold text-zinc-500 hover:bg-sky-50 transition-colors ml-1"
              title={executiveExpanded ? 'ย่อแผงเป้าหมายและประวัติ NAV' : 'ขยายแผงเป้าหมายและประวัติ NAV'}
            >
              {executiveExpanded ? '▲ ย่อ' : '▼ ขยาย'}
            </button>
          </div>
        </div>

        {executiveExpanded && (
          <div className="pt-5 animate-fade-in">
            {executiveTab === 'goals' ? (
              <PortfolioGoalsTab
                goals={goalsResponse?.goals ?? []}
                generatedAt={goalsResponse?.generated_at ?? null}
                onSuccess={(gRes) => {
                  setGoalsResponse(gRes)
                  if (portfolioState) handlePortfolioStateSuccess(portfolioState)
                }}
              />
            ) : (
              <PortfolioAnalyticsTab
                performanceRows={performanceRows}
                daysRange={daysRange}
                onChangeDaysRange={setDaysRange}
              />
            )}
          </div>
        )}
      </div>

      {/* Tab Switcher */}
      <div className="flex items-center justify-between border-b border-sky-100 pb-4">
        <SegmentedControl options={TABS} value={activeTab} onChange={handleTabChange} />
        {selectedBucket && activeTab !== 'holdings' && (
          <button
            type="button"
            onClick={() => handleTabChange('holdings')}
            className="flex items-center gap-1.5 rounded-xl border border-sky-300 bg-sky-50 px-3 py-1.5 text-xs font-bold text-flow-blue shadow-2xs hover:bg-sky-100 animate-pulse"
          >
            <span>🔍 Active Filter: Bucket [{selectedBucket}]</span>
            <span>→ ไปที่ Holdings</span>
          </button>
        )}
      </div>

      {/* Tab Content */}
      <div className="pt-2">
        {activeTab === 'overview' && (
          <PortfolioOverviewTab
            targets={portfolioState?.allocation_targets ?? []}
            summaries={allocationsResponse?.summaries ?? []}
            warning={allocationsResponse?.warning ?? null}
            onSelectBucket={handleSelectBucket}
            onSuccess={handlePortfolioStateSuccess}
          />
        )}

        {activeTab === 'holdings' && (
          <PortfolioHoldingsTab
            holdings={portfolioState?.holdings ?? []}
            selectedBucket={selectedBucket}
            onClearBucketFilter={handleClearBucketFilter}
            targets={portfolioState?.allocation_targets ?? []}
            onSuccess={handlePortfolioStateSuccess}
            journalRows={journalRows}
            journalKeyword={journalKeyword}
            onChangeJournalKeyword={setJournalKeyword}
            onSuccessJournal={(entries) => setJournalRows(entries)}
          />
        )}

        {activeTab === 'watchlist' && (
          <PortfolioWatchlistTab
            items={watchlistState?.items ?? []}
            lastUpdated={watchlistState?.last_updated ?? null}
            onSuccess={(wState) => setWatchlistState(wState)}
          />
        )}
      </div>

      {tradeModalOpen && (
        <TradeModal
          targets={portfolioState?.allocation_targets ?? []}
          onClose={() => setTradeModalOpen(false)}
          onSuccess={handlePortfolioStateSuccess}
        />
      )}

      {cashFlowModalOpen && (
        <CashFlowModal
          onClose={() => setCashFlowModalOpen(false)}
          onSuccess={handlePortfolioStateSuccess}
        />
      )}

      {incomeModalOpen && (
        <IncomeModal
          holdingsSymbols={(portfolioState?.holdings ?? []).map((h) => h.symbol)}
          onClose={() => setIncomeModalOpen(false)}
          onSuccess={handlePortfolioStateSuccess}
        />
      )}

      {resetModalOpen && (
        <ResetConfirmModal
          onClose={() => setResetModalOpen(false)}
          onSuccess={handlePortfolioStateSuccess}
        />
      )}
    </div>
  )
}
