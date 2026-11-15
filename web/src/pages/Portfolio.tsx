import { lazy, Suspense, useState, useEffect, useCallback, useRef, useMemo } from 'react'
import { useSearchParams } from 'react-router-dom'
import { api } from '../api/client'
import type {
  PortfolioMetaDTO,
  ActualPortfolioStateDTO,
  BucketAllocationResponseDTO,
  ActualWatchlistStateDTO,
  ActualGoalsResponseDTO,
  PerformanceSnapshotDTO,
  JournalEntryDTO,
} from '../api/types'
import SegmentedControl from '../components/ui/SegmentedControl'
import Modal from '../components/ui/Modal'
import { Toast } from '../components/common/Toast'
import {
  AlertTriangleIcon,
  BoltStrategyIcon,
  CalendarIcon,
  ChartTrendIcon,
  ChevronDownIcon,
  ChevronUpIcon,
  CompassIcon,
  EditPenIcon,
  FolderIcon,
  MoreHorizontalIcon,
  TrashIcon,
} from '../components/ui/AppIcons'
import PortfolioSummaryCards from '../components/portfolio/PortfolioSummaryCards'
import PortfolioOverviewTab from '../components/portfolio/PortfolioOverviewTab'
import type { TransactionView } from '../lib/pvdStatementHistory'
import { getPvdValuationContext } from '../lib/pvdPresentation'
import { TopLoadingRibbon } from '../components/ui/FlowSkeleton'

const PortfolioHoldingsTab = lazy(() => import('../components/portfolio/PortfolioHoldingsTab'))
const PortfolioWatchlistTab = lazy(() => import('../components/portfolio/PortfolioWatchlistTab'))
const PortfolioGoalsTab = lazy(() => import('../components/portfolio/PortfolioGoalsTab'))
const TradeModal = lazy(() => import('../components/portfolio/Modals/TradeModal'))
const CashFlowModal = lazy(() => import('../components/portfolio/Modals/CashFlowModal'))
const IncomeModal = lazy(() => import('../components/portfolio/Modals/IncomeModal'))
const ResetConfirmModal = lazy(() => import('../components/portfolio/Modals/ResetConfirmModal'))
const PortfolioCalendarTab = lazy(() => import('../components/portfolio/PortfolioCalendarTab'))
const PortfolioTransactionsTab = lazy(() => import('../components/portfolio/PortfolioTransactionsTab'))
const PortfolioIncomesTab = lazy(() => import('../components/portfolio/PortfolioIncomesTab'))
const PortfolioAnalyticsTab = lazy(() => import('../components/portfolio/PortfolioAnalyticsTab'))
const InvestorPrinciplesTab = lazy(() => import('../components/portfolio/essence/InvestorPrinciplesTab'))
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
      <span className="flex items-center gap-1.5 text-sm font-semibold whitespace-nowrap">
        <ChartBarIcon className="w-4 h-4 text-flow-blue shrink-0" />
        <span>Strategy Buckets</span>
        <span className="hidden 2xl:inline">& Allocation</span>
      </span>
    ),
  },
  {
    key: 'principles',
    label: (
      <span className="flex items-center gap-1.5 text-sm font-semibold whitespace-nowrap">
        <CompassIcon className="w-4 h-4 text-section-accent shrink-0" />
        <span>หลักการลงทุน</span>
        <span className="hidden 2xl:inline"> (Principles)</span>
      </span>
    ),
  },
  {
    key: 'holdings',
    label: (
      <span className="flex items-center gap-1.5 text-sm font-semibold whitespace-nowrap">
        <BriefcaseIcon className="w-4 h-4 text-emerald-600 shrink-0" />
        <span>Holdings</span>
        <span className="hidden 2xl:inline"> & Trading Journal</span>
      </span>
    ),
  },
  {
    key: 'transactions',
    label: (
      <span className="flex items-center gap-1.5 text-sm font-semibold whitespace-nowrap">
        <TradeIcon className="w-4 h-4 text-sky-500 shrink-0" />
        <span>Transactions</span>
      </span>
    ),
  },
  {
    key: 'incomes',
    label: (
      <span className="flex items-center gap-1.5 text-sm font-semibold whitespace-nowrap">
        <IncomeIcon className="w-4 h-4 text-emerald-500 shrink-0" />
        <span>Dividends & Income</span>
      </span>
    ),
  },
  {
    key: 'analytics',
    label: (
      <span className="flex items-center gap-1.5 text-sm font-semibold whitespace-nowrap">
        <ChartTrendIcon className="w-4 h-4 text-section-accent shrink-0" />
        <span>Analytics</span>
        <span className="hidden 2xl:inline"> & History</span>
      </span>
    ),
  },
  {
    key: 'watchlist',
    label: (
      <span className="flex items-center gap-1.5 text-sm font-semibold whitespace-nowrap">
        <WatchlistIcon className="w-4 h-4 text-amber-500 shrink-0" />
        <span>Watchlist</span>
      </span>
    ),
  },
  {
    key: 'calendar',
    label: (
      <span className="flex items-center gap-1.5 text-sm font-semibold whitespace-nowrap">
        <CalendarIcon className="w-4 h-4 text-section-accent shrink-0" />
        <span>Calendar</span>
      </span>
    ),
  },
]

export default function Portfolio() {
  const [searchParams, setSearchParams] = useSearchParams()
  const rawTab = searchParams.get('tab') || 'overview'
  const activeTab = rawTab === 'essence' ? 'principles' : rawTab
  const selectedBucket = searchParams.get('bucket') || null
  const selectedSymbol = searchParams.get('symbol') || null
  const selectedPortfolioId = searchParams.get('portfolio_id') || 'default'
  const explicitTransactionView = searchParams.get('tx_view')
  const inferredTransactionView: TransactionView = selectedSymbol?.trim().toUpperCase().startsWith('PVD_') ? 'pvd' : 'assets'
  const selectedTransactionView: TransactionView = explicitTransactionView === 'assets' || explicitTransactionView === 'pvd'
    ? explicitTransactionView
    : inferredTransactionView
  const transactionFilterSymbol = selectedSymbol && ((selectedTransactionView === 'pvd') === selectedSymbol.trim().toUpperCase().startsWith('PVD_'))
    ? selectedSymbol
    : null
  const selectedPortfolioIdRef = useRef(selectedPortfolioId)
  selectedPortfolioIdRef.current = selectedPortfolioId

  // Tab Keep-Alive: Keep mounted tabs alive to prevent DOM teardown and enable 0ms instant tab switching
  const [mountedTabs, setMountedTabs] = useState<Record<string, boolean>>(() => ({
    [activeTab]: true,
  }))

  useEffect(() => {
    setMountedTabs((prev) => (prev[activeTab] ? prev : { ...prev, [activeTab]: true }))
  }, [activeTab])

  // Multi-Portfolio state
  const [portfolios, setPortfolios] = useState<PortfolioMetaDTO[]>([])
  const [createPortModalOpen, setCreatePortModalOpen] = useState(false)
  const [newPortName, setNewPortName] = useState('')
  const [creatingPort, setCreatingPort] = useState(false)
  const [renamePortModalOpen, setRenamePortModalOpen] = useState(false)
  const [editPortName, setEditPortName] = useState('')
  const [renamingPort, setRenamingPort] = useState(false)
  const [deletePortModalOpen, setDeletePortModalOpen] = useState(false)
  const [deletingPort, setDeletingPort] = useState(false)
  const [overflowMenuOpen, setOverflowMenuOpen] = useState(false)
  const overflowMenuRef = useRef<HTMLDivElement>(null)
  const [pageToast, setPageToast] = useState<{ message: string; type: 'success' | 'error' | 'info' } | null>(null)

  useEffect(() => {
    if (!overflowMenuOpen) return
    const handleOutside = (e: MouseEvent) => {
      if (overflowMenuRef.current && !overflowMenuRef.current.contains(e.target as Node)) {
        setOverflowMenuOpen(false)
      }
    }
    const handleKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setOverflowMenuOpen(false)
    }
    document.addEventListener('mousedown', handleOutside)
    document.addEventListener('keydown', handleKey)
    return () => {
      document.removeEventListener('mousedown', handleOutside)
      document.removeEventListener('keydown', handleKey)
    }
  }, [overflowMenuOpen])


  // Portfolio details state
  const [portfolioState, setPortfolioState] = useState<ActualPortfolioStateDTO | null>(null)
  const currentPortfolioMeta = portfolios.find((p) => p.id === selectedPortfolioId)
  const currentPortfolioName = currentPortfolioMeta?.name || portfolioState?.name || (selectedPortfolioId === 'default' ? 'พอร์ตลงทุนหลัก' : selectedPortfolioId)
  const pvdContext = getPvdValuationContext(portfolioState?.holdings ?? [], portfolioState?.pvd_statements ?? [])
  const [allocationsResponse, setAllocationsResponse] = useState<BucketAllocationResponseDTO | null>(null)
  const [watchlistState, setWatchlistState] = useState<ActualWatchlistStateDTO | null>(null)
  const [goalsResponse, setGoalsResponse] = useState<ActualGoalsResponseDTO | null>(null)
  const [performanceRows, setPerformanceRows] = useState<PerformanceSnapshotDTO[]>([])
  const [journalRows, setJournalRows] = useState<JournalEntryDTO[]>([])

  // Filters state
  const [daysRange, setDaysRange] = useState<number | undefined>(30)
  const [journalKeyword, setJournalKeyword] = useState<string>('')

  // Top-Level Executive Dashboard state
  const [executiveExpanded, setExecutiveExpanded] = useState<boolean>(true)

  // Modals state
  const [tradeModalOpen, setTradeModalOpen] = useState(false)
  const [cashFlowModalOpen, setCashFlowModalOpen] = useState(false)
  const [incomeModalOpen, setIncomeModalOpen] = useState(false)
  const [resetModalOpen, setResetModalOpen] = useState(false)

  const handlePortfolioStateSuccess = (newState: ActualPortfolioStateDTO) => {
    setPortfolioState(newState)
    const myFetchId = ++fetchIdRef.current
    api.listPortfolios().then((ports) => {
      if (fetchIdRef.current === myFetchId) setPortfolios(ports)
    }).catch(() => {})
    api.getActualBucketAllocations(selectedPortfolioId).then((allocRes) => {
      if (fetchIdRef.current === myFetchId) setAllocationsResponse(allocRes)
    }).catch(() => {})
    // Goals คำนวณจาก NAV/เงินสด/เงินปันผลของพอร์ต — ต้องรีเฟรชทุกครั้งที่พอร์ตมีการเปลี่ยนแปลง
    // ไม่กรองตามพอร์ตที่เลือกอยู่ (โชว์ทุกเป้าหมายจากทุกพอร์ตรวมกันเสมอ ดู badge ในการ์ด)
    api.getActualGoals().then((gRes) => {
      if (fetchIdRef.current === myFetchId) setGoalsResponse(gRes)
    }).catch(() => {})
  }

  // Loading & Error
  const [loading, setLoading] = useState<boolean>(true)
  const [refreshingPrices, setRefreshingPrices] = useState<boolean>(false)
  const [error, setError] = useState<string | null>(null)

  // นับรอบ fetch แบบ monotonic — กัน response ของพอร์ตเก่าที่มาช้ากว่ามาทับพอร์ตใหม่ที่โหลดเสร็จไปแล้ว
  // (fetchAllData เป็น useCallback ที่ถูกเรียกทั้งจาก effect และปุ่ม refresh เลยใช้ isMounted แบบ effect เดี่ยวไม่ได้)
  const fetchIdRef = useRef(0)

  const fetchAllData = useCallback(async (refreshPrices: boolean = false) => {
    const myFetchId = ++fetchIdRef.current
    try {
      if (refreshPrices) {
        setRefreshingPrices(true)
      } else {
        setLoading(true)
      }
      setError(null)

      const portPromise = api.listPortfolios()
        .then((ports) => {
          if (fetchIdRef.current === myFetchId) setPortfolios(ports)
        })
        .catch((err) => {
          console.warn('Failed to load portfolio list:', err)
        })

      const statePromise = api.getActualPortfolioState(refreshPrices, refreshPrices, selectedPortfolioId)
        .then((pState) => {
          if (fetchIdRef.current === myFetchId) {
            setPortfolioState(pState)
            if (!refreshPrices) {
              setLoading(false)
            }
          }
        })
        .catch((err) => {
          if (fetchIdRef.current === myFetchId) {
            setError(err?.message || 'ไม่สามารถโหลดข้อมูล Actual Portfolio ได้')
          }
        })

      const allocPromise = api.getActualBucketAllocations(selectedPortfolioId)
        .then((allocRes) => {
          if (fetchIdRef.current === myFetchId) setAllocationsResponse(allocRes)
        })
        .catch((err) => {
          console.warn('Failed to load bucket allocations:', err)
        })

      const watchlistPromise = Promise.resolve(api.getActualWatchlist(selectedPortfolioId))
        .then((wState) => {
          if (fetchIdRef.current !== myFetchId) return
          setWatchlistState(wState)
        })
        .catch(() => {})

      const goalsPromise = Promise.resolve(api.getActualGoals())
        .then((gRes) => {
          if (fetchIdRef.current !== myFetchId) return
          setGoalsResponse(gRes)
        })
        .catch(() => {})

      await Promise.allSettled([portPromise, statePromise, allocPromise, watchlistPromise, goalsPromise])

      if (refreshPrices) {
        try {
          const perf = await api.getActualPerformance(daysRange, selectedPortfolioId)
          if (fetchIdRef.current === myFetchId) setPerformanceRows(perf)
        } catch {
          // ignore
        }
      }
    } catch (err: any) {
      if (fetchIdRef.current !== myFetchId) return
      setError(err?.message || 'ไม่สามารถโหลดข้อมูล Actual Portfolio ได้')
    } finally {
      if (fetchIdRef.current === myFetchId) {
        setLoading(false)
        setRefreshingPrices(false)
      }
    }
  }, [daysRange, selectedPortfolioId])

  // Initial load & when portfolio_id changes
  useEffect(() => {
    void fetchAllData(false)
  }, [fetchAllData])

  // Load performance when daysRange or portfolio_id changes
  useEffect(() => {
    let isMounted = true
    api
      .getActualPerformance(daysRange, selectedPortfolioId)
      .then((rows) => { if (isMounted) setPerformanceRows(rows) })
      .catch(() => { if (isMounted) setPerformanceRows([]) })
    return () => {
      isMounted = false
    }
  }, [daysRange, selectedPortfolioId])

  // Load journal when keyword or portfolio_id changes
  useEffect(() => {
    let isMounted = true
    const timer = setTimeout(() => {
      api
        .getActualJournal(365, journalKeyword || undefined, 100, selectedPortfolioId)
        .then((rows) => { if (isMounted) setJournalRows(rows) })
        .catch(() => { if (isMounted) setJournalRows([]) })
    }, 300)
    return () => {
      isMounted = false
      clearTimeout(timer)
    }
  }, [journalKeyword, selectedPortfolioId])

  // Auto-redirect legacy tabs to overview or holdings
  useEffect(() => {
    if (activeTab === 'goals') {
      const nextParams = new URLSearchParams(searchParams)
      nextParams.set('tab', 'overview')
      setSearchParams(nextParams, { replace: true })
    }
  }, [activeTab, searchParams, setSearchParams])

  const handleTabChange = (key: string) => {
    const nextParams = new URLSearchParams(searchParams)
    nextParams.set('tab', key)
    nextParams.delete('symbol')
    if (key === 'transactions') nextParams.set('tx_view', 'assets')
    else nextParams.delete('tx_view')
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

  const handleViewTransactionsForSymbol = (symbol: string) => {
    const nextParams = new URLSearchParams(searchParams)
    nextParams.set('tab', 'transactions')
    nextParams.set('tx_view', symbol.trim().toUpperCase().startsWith('PVD_') ? 'pvd' : 'assets')
    nextParams.set('symbol', symbol)
    setSearchParams(nextParams)
  }

  const handleTransactionViewChange = (view: TransactionView) => {
    const nextParams = new URLSearchParams(searchParams)
    nextParams.set('tab', 'transactions')
    nextParams.set('tx_view', view)
    const symbol = nextParams.get('symbol')?.trim().toUpperCase()
    if (symbol && ((view === 'pvd') !== symbol.startsWith('PVD_'))) nextParams.delete('symbol')
    setSearchParams(nextParams)
  }

  const handleClearSymbolFilter = () => {
    const nextParams = new URLSearchParams(searchParams)
    nextParams.delete('symbol')
    setSearchParams(nextParams)
  }

  const refreshWatchlist = useCallback(async () => {
    try {
      const wState = await api.getActualWatchlist(selectedPortfolioId)
      if (selectedPortfolioIdRef.current === selectedPortfolioId) {
        setWatchlistState(wState)
      }
    } catch (e) {
      console.warn('Failed to refresh watchlist:', e)
    }
  }, [selectedPortfolioId])

  const watchlistedSymbols = useMemo(() => {
    return (watchlistState?.items ?? []).map((it) => it.symbol.toUpperCase())
  }, [watchlistState])

  const handleCreatePortfolio = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!newPortName.trim()) return
    setCreatingPort(true)
    try {
      const meta = await api.createPortfolio(newPortName.trim())
      setCreatePortModalOpen(false)
      setNewPortName('')
      setPageToast({ message: `สร้างพอร์ต "${meta.name}" เรียบร้อยแล้ว`, type: 'success' })
      const nextParams = new URLSearchParams(searchParams)
      nextParams.set('portfolio_id', meta.id)
      setSearchParams(nextParams)
    } catch (err: any) {
      setPageToast({ message: err?.message || 'ไม่สามารถสร้างพอร์ตใหม่ได้', type: 'error' })
    } finally {
      setCreatingPort(false)
    }
  }

  const handleOpenRenameModal = () => {
    const currentMeta = portfolios.find((p) => p.id === selectedPortfolioId)
    setEditPortName(currentMeta?.name || 'พอร์ตลงทุนหลัก')
    setRenamePortModalOpen(true)
  }

  const handleRenamePortfolio = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!editPortName.trim()) return
    setRenamingPort(true)
    try {
      const updatedMeta = await api.renamePortfolio(selectedPortfolioId, editPortName.trim())
      setPortfolios((prev) =>
        prev.map((p) => (p.id === updatedMeta.id ? { ...p, name: updatedMeta.name } : p))
      )
      setRenamePortModalOpen(false)
      setPageToast({ message: `เปลี่ยนชื่อพอร์ตเป็น "${updatedMeta.name}" เรียบร้อย`, type: 'success' })
    } catch (err: any) {
      setPageToast({ message: err?.message || 'ไม่สามารถแก้ไขชื่อพอร์ตได้', type: 'error' })
    } finally {
      setRenamingPort(false)
    }
  }

  const handleConfirmDeletePortfolio = async () => {
    if (selectedPortfolioId === 'default') return
    setDeletingPort(true)
    try {
      await api.deletePortfolio(selectedPortfolioId)
      setDeletePortModalOpen(false)
      setPageToast({ message: 'ลบพอร์ตเรียบร้อยแล้ว', type: 'success' })
      const nextParams = new URLSearchParams(searchParams)
      nextParams.set('portfolio_id', 'default')
      setSearchParams(nextParams)
    } catch (err: any) {
      setPageToast({ message: err?.message || 'ไม่สามารถลบพอร์ตได้', type: 'error' })
    } finally {
      setDeletingPort(false)
    }
  }

  return (
    <div className="animate-page-in space-y-6 pb-14">
      {pageToast && (
        <Toast
          message={pageToast.message}
          type={pageToast.type}
          onClose={() => setPageToast(null)}
        />
      )}

      {/* Background SWR Loading Indicator — Subtle, non-blocking */}
      {loading && portfolioState && (
        <div className="rounded-xl overflow-hidden shadow-sm">
          <TopLoadingRibbon label="กำลังอัปเดตข้อมูลพอร์ตล่าสุด..." />
        </div>
      )}

      {/* Page Header with Multi-Portfolio Selector — Hero Banner Card */}
      <div className="flow-panel rounded-2xl border border-emerald-100/80 p-5 md:p-6 shadow-sm bg-gradient-to-r from-white/95 via-emerald-50/35 to-teal-50/25 backdrop-blur-md flex flex-col justify-between gap-3 xl:flex-row xl:items-center">
        {/* Left: Branding & Title */}
        <div className="space-y-1.5 max-w-xl">
          <div className="flex items-center gap-2">
            <h1 className="text-xl sm:text-2xl lg:text-3xl font-extrabold tracking-tight text-slate-900">
              Actual Portfolio Hub
            </h1>
            {activeTab === 'principles' && (
              <span className="inline-flex items-center gap-1 rounded-full bg-emerald-100/80 px-2.5 py-0.5 text-[11px] font-bold text-emerald-900 whitespace-nowrap">
                <CompassIcon className="w-3 h-3 text-emerald-700" />
                <span>Principles & Purpose Buckets</span>
              </span>
            )}
          </div>
          <p className="text-xs md:text-sm text-slate-500 font-medium leading-relaxed">
            ศูนย์กลางติดตามพอร์ตการลงทุนจริง (Actual Portfolio) พร้อมกลยุทธ์สัดส่วน การจัดกลุ่ม Bucket และระบบบันทึก Trading Journal
          </p>
        </div>

        {/* Right: Portfolio Selector & Streamlined Action Toolbar */}
        <div className="flex flex-col items-stretch xl:items-end gap-2.5 w-full xl:w-auto min-w-0">
          {/* Portfolio Selector Pill + Overflow Management Menu */}
          <div
            ref={overflowMenuRef}
            className="relative flex flex-wrap sm:flex-nowrap w-full items-center justify-between gap-2 bg-white/95 backdrop-blur border border-emerald-200/80 shadow-sm rounded-xl px-3 py-1.5 transition-all hover:border-emerald-300"
          >
            <span className="text-xs sm:text-sm font-bold text-slate-600 flex items-center gap-1.5 shrink-0">
              <FolderIcon className="w-4 h-4 text-section-accent" />
              <span>พอร์ต:</span>
            </span>
            <select
              value={selectedPortfolioId}
              onChange={(e) => {
                const nextParams = new URLSearchParams(searchParams)
                nextParams.set('portfolio_id', e.target.value)
                setSearchParams(nextParams)
              }}
              className="flex-1 min-w-0 rounded-lg border border-emerald-200/80 bg-emerald-50/40 px-2.5 py-1 text-xs sm:text-sm font-bold text-slate-800 shadow-sm focus:outline-none focus:ring-2 focus:ring-section-accent/45 cursor-pointer hover:bg-white transition-colors"
            >
              {portfolios.length > 0 ? (
                portfolios.map((p) => (
                  <option key={p.id} value={p.id}>
                    {p.name} {p.is_default ? '(หลัก)' : ''}
                  </option>
                ))
              ) : (
                <option value="default">พอร์ตลงทุนหลัก (Default)</option>
              )}
            </select>

            <button
              type="button"
              aria-label="จัดการพอร์ตเพิ่มเติม"
              aria-expanded={overflowMenuOpen}
              onClick={() => setOverflowMenuOpen((prev) => !prev)}
              title="จัดการพอร์ต: สร้างใหม่ แก้ไขชื่อ ลบ หรือล้างพอร์ต"
              className="shrink-0 inline-flex items-center justify-center rounded-lg border border-slate-200 bg-white px-2 py-1 text-slate-600 hover:bg-slate-50 hover:text-slate-900 transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-section-accent/45"
            >
              <MoreHorizontalIcon className="w-4 h-4" />
            </button>

            {overflowMenuOpen && (
              <div
                role="menu"
                aria-label="เมนูจัดการพอร์ต"
                className="absolute right-0 top-full mt-1.5 z-40 w-52 rounded-xl border border-slate-200 bg-white/95 backdrop-blur-xl p-1.5 shadow-lg space-y-0.5 animate-fade-in"
              >
                <button
                  type="button"
                  role="menuitem"
                  onClick={() => {
                    setOverflowMenuOpen(false)
                    setCreatePortModalOpen(true)
                  }}
                  className="flex w-full items-center gap-2 rounded-lg px-3 py-2 text-left text-xs font-semibold text-emerald-700 hover:bg-emerald-50 transition-colors"
                >
                  <FolderIcon className="w-3.5 h-3.5" />
                  <span>+ สร้างพอร์ตใหม่</span>
                </button>
                <button
                  type="button"
                  role="menuitem"
                  onClick={() => {
                    setOverflowMenuOpen(false)
                    handleOpenRenameModal()
                  }}
                  className="flex w-full items-center gap-2 rounded-lg px-3 py-2 text-left text-xs font-semibold text-slate-700 hover:bg-slate-50 transition-colors"
                >
                  <EditPenIcon className="w-3.5 h-3.5" />
                  <span>แก้ไขชื่อพอร์ต</span>
                </button>
                {selectedPortfolioId !== 'default' && (
                  <button
                    type="button"
                    role="menuitem"
                    onClick={() => {
                      setOverflowMenuOpen(false)
                      setDeletePortModalOpen(true)
                    }}
                    className="flex w-full items-center gap-2 rounded-lg px-3 py-2 text-left text-xs font-semibold text-rose-600 hover:bg-rose-50 transition-colors"
                  >
                    <TrashIcon className="w-3.5 h-3.5" />
                    <span>ลบพอร์ตนี้</span>
                  </button>
                )}
                <div className="my-1 border-t border-slate-100" />
                <button
                  type="button"
                  role="menuitem"
                  onClick={() => {
                    setOverflowMenuOpen(false)
                    setResetModalOpen(true)
                  }}
                  className="flex w-full items-center gap-2 rounded-lg px-3 py-2 text-left text-xs font-semibold text-rose-600 hover:bg-rose-50 transition-colors"
                >
                  <ResetIcon className="w-3.5 h-3.5" />
                  <span>ล้างข้อมูลพอร์ต</span>
                </button>
              </div>
            )}
          </div>

          {/* Action Toolbar — 1 Primary Solid CTA + Calibrated Secondary Pills */}
          <div className="flex flex-wrap items-center justify-start xl:justify-end gap-2 w-full">
            <a
              href={`/advisory?portfolio_id=${encodeURIComponent(selectedPortfolioId)}`}
              className="rounded-xl border border-emerald-200/90 bg-white/90 px-3 sm:px-3.5 py-1.5 sm:py-2 text-xs sm:text-sm font-semibold text-slate-800 shadow-sm hover:bg-emerald-50/80 hover:border-emerald-300 active:scale-98 transition-all flex items-center gap-1.5 whitespace-nowrap focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-section-accent/45"
            >
              <BoltStrategyIcon className="w-4 h-4 text-section-accent shrink-0" />
              <span>Advisory Desk</span>
            </a>
            <button
              type="button"
              onClick={() => setTradeModalOpen(true)}
              className="rounded-xl bg-gradient-to-r from-emerald-600 to-teal-600 px-3.5 sm:px-4 py-1.5 sm:py-2 text-xs sm:text-sm font-bold text-white shadow-md shadow-emerald-600/20 hover:from-emerald-700 hover:to-teal-700 hover:shadow-lg hover:shadow-emerald-600/25 active:scale-98 transition-all flex items-center gap-1.5 whitespace-nowrap focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-section-accent/45"
            >
              <TradeIcon className="w-4 h-4 text-white shrink-0" />
              <span>บันทึกเทรด (Trade)</span>
            </button>
            <button
              type="button"
              onClick={() => setCashFlowModalOpen(true)}
              className="rounded-xl border border-slate-200/90 bg-white/90 px-3 sm:px-3.5 py-1.5 sm:py-2 text-xs sm:text-sm font-semibold text-slate-700 shadow-sm hover:bg-emerald-50/60 hover:border-emerald-200 active:scale-98 transition-all flex items-center gap-1.5 whitespace-nowrap focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-section-accent/45"
            >
              <CashFlowIcon className="w-4 h-4 text-emerald-600 shrink-0" />
              <span>ฝาก/ถอน (Cash)</span>
            </button>
            <button
              type="button"
              onClick={() => setIncomeModalOpen(true)}
              className="rounded-xl border border-slate-200/90 bg-white/90 px-3 sm:px-3.5 py-1.5 sm:py-2 text-xs sm:text-sm font-semibold text-slate-700 shadow-sm hover:bg-emerald-50/60 hover:border-emerald-200 active:scale-98 transition-all flex items-center gap-1.5 whitespace-nowrap focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-section-accent/45"
            >
              <IncomeIcon className="w-4 h-4 text-emerald-600 shrink-0" />
              <span>ปันผล/รายรับ (Income)</span>
            </button>
          </div>
        </div>
      </div>

      {/* Modal: Create Portfolio */}
      {createPortModalOpen && (
        <Modal
          titleId="create-portfolio-title"
          onClose={() => setCreatePortModalOpen(false)}
          panelClassName="max-w-sm rounded-2xl bg-white p-6 shadow-2xl border border-emerald-100"
        >
          <h3 id="create-portfolio-title" className="text-lg font-bold text-slate-900 mb-2 flex items-center gap-2">
            <FolderIcon className="w-5 h-5 text-section-accent" />
            <span>สร้างพอร์ตการลงทุนใหม่</span>
          </h3>
          <p className="text-xs text-slate-500 mb-4">
            เช่น "พอร์ตเงินสำรองฉุกเฉิน", "พอร์ตเกษียณ", "พอร์ตปันผล" เพื่อแยกสินทรัพย์ออกจากกันเด็ดขาด
          </p>
          <form onSubmit={handleCreatePortfolio} className="space-y-4">
            <div>
              <label htmlFor="create-port-name" className="block text-xs font-semibold text-slate-700 mb-1">
                ชื่อพอร์ต (Portfolio Name)
              </label>
              <input
                id="create-port-name"
                type="text"
                required
                value={newPortName}
                onChange={(e) => setNewPortName(e.target.value)}
                placeholder="e.g. พอร์ตเงินสำรองฉุกเฉิน"
                className="w-full rounded-xl border border-slate-200 bg-surface-2 px-3.5 py-2 text-sm font-bold text-slate-900 focus:outline-none focus:ring-2 focus:ring-section-accent/45"
              />
            </div>
            <div className="flex justify-end gap-2 pt-2">
              <button
                type="button"
                onClick={() => setCreatePortModalOpen(false)}
                className="rounded-xl px-4 py-2 text-xs font-bold text-slate-600 hover:bg-slate-100"
              >
                ยกเลิก
              </button>
              <button
                type="submit"
                disabled={creatingPort}
                className="rounded-xl bg-emerald-600 px-4 py-2 text-xs font-bold text-white shadow-md hover:bg-emerald-700 disabled:opacity-50"
              >
                {creatingPort ? 'กำลังสร้าง...' : '+ สร้างพอร์ต'}
              </button>
            </div>
          </form>
        </Modal>
      )}

      {/* Modal: Rename Portfolio */}
      {renamePortModalOpen && (
        <Modal
          titleId="rename-portfolio-title"
          onClose={() => setRenamePortModalOpen(false)}
          panelClassName="max-w-sm rounded-2xl bg-white p-6 shadow-2xl border border-emerald-100"
        >
          <h3 id="rename-portfolio-title" className="text-lg font-bold text-slate-900 mb-2 flex items-center gap-2">
            <EditPenIcon className="w-5 h-5 text-section-accent" />
            <span>แก้ไขชื่อพอร์ต</span>
          </h3>
          <p className="text-xs text-slate-500 mb-4">
            เปลี่ยนชื่อแสดงผลของพอร์ตการลงทุนที่เลือก
          </p>
          <form onSubmit={handleRenamePortfolio} className="space-y-4">
            <div>
              <label htmlFor="rename-port-name" className="block text-xs font-semibold text-slate-700 mb-1">
                ชื่อพอร์ตใหม่ (Portfolio Name)
              </label>
              <input
                id="rename-port-name"
                type="text"
                required
                value={editPortName}
                onChange={(e) => setEditPortName(e.target.value)}
                placeholder="เช่น พอร์ตเพื่อการเติบโต"
                className="w-full rounded-xl border border-slate-200 bg-surface-2 px-3.5 py-2 text-sm font-bold text-slate-900 focus:outline-none focus:ring-2 focus:ring-section-accent/45"
              />
            </div>
            <div className="flex justify-end gap-2 pt-2">
              <button
                type="button"
                onClick={() => setRenamePortModalOpen(false)}
                className="rounded-xl px-4 py-2 text-xs font-bold text-slate-600 hover:bg-slate-100"
              >
                ยกเลิก
              </button>
              <button
                type="submit"
                disabled={renamingPort}
                className="rounded-xl bg-emerald-600 px-4 py-2 text-xs font-bold text-white shadow-md hover:bg-emerald-700 disabled:opacity-50"
              >
                {renamingPort ? 'กำลังบันทึก...' : 'บันทึก'}
              </button>
            </div>
          </form>
        </Modal>
      )}

      {/* Modal: Delete Portfolio Confirmation */}
      {deletePortModalOpen && (
        <Modal
          titleId="delete-portfolio-title"
          onClose={() => setDeletePortModalOpen(false)}
          panelClassName="max-w-sm rounded-2xl bg-white p-6 shadow-2xl border border-rose-200"
        >
          <h3 id="delete-portfolio-title" className="text-lg font-bold text-rose-700 mb-2 flex items-center gap-2">
            <TrashIcon className="w-5 h-5" />
            <span>ยืนยันการลบพอร์ต</span>
          </h3>
          <p className="text-xs text-slate-600 leading-relaxed mb-4">
            คุณต้องการลบพอร์ต <strong className="text-slate-900">"{currentPortfolioName}"</strong> ใช่หรือไม่? การลบพอร์ตนี้ไม่สามารถย้อนกลับได้
          </p>
          <div className="flex justify-end gap-2">
            <button
              type="button"
              onClick={() => setDeletePortModalOpen(false)}
              className="rounded-xl px-4 py-2 text-xs font-bold text-slate-600 hover:bg-slate-100"
            >
              ยกเลิก
            </button>
            <button
              type="button"
              disabled={deletingPort}
              onClick={() => void handleConfirmDeletePortfolio()}
              className="rounded-xl bg-rose-600 px-4 py-2 text-xs font-bold text-white shadow-md hover:bg-rose-700 disabled:opacity-50"
            >
              {deletingPort ? 'กำลังลบ...' : 'ลบพอร์ตถาวร'}
            </button>
          </div>
        </Modal>
      )}

      {/* Error Alert */}
      {error && (
        <div className="flex items-center justify-between rounded-2xl border border-rose-200 bg-rose-50 p-4 text-rose-900 shadow-sm animate-fade-in">
          <div className="flex items-center gap-3">
            <AlertTriangleIcon className="w-5 h-5 text-rose-600 shrink-0" />
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

      {/* Summary Cards & Executive Dashboard */}
      <PortfolioSummaryCards
        summary={portfolioState?.summary ?? null}
        lastUpdated={portfolioState?.last_updated ?? null}
        holdings={portfolioState?.holdings ?? []}
        pvdContext={pvdContext}
        loading={loading}
        refreshingPrices={refreshingPrices}
        priceRefreshInfo={portfolioState?.price_refresh_info ?? null}
        onRefreshPrices={() => void fetchAllData(true)}
        onViewPvdStatements={() => handleTransactionViewChange('pvd')}
        performanceRows={performanceRows}
      />

      {/* Top-Level Executive Goals Section */}
      <div
        className={`rounded-2xl border border-emerald-100/80 bg-gradient-to-b from-emerald-50/35 via-white to-panel shadow-sm transition-all ${
          executiveExpanded ? 'p-5' : 'px-4 py-3'
        }`}
      >
        <div
          className={`flex flex-col justify-between gap-3 sm:flex-row sm:items-center ${
            executiveExpanded ? 'border-b border-emerald-100 pb-4' : ''
          }`}
        >
          <div className="flex items-center gap-3 min-w-0">
            <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-emerald-500/10 text-section-accent">
              <GoalIcon className="h-5 w-5" />
            </span>
            <div className="min-w-0">
              <h2 className="text-sm sm:text-base font-extrabold tracking-tight text-slate-900 flex items-center gap-2 flex-wrap">
                <span>Portfolio Goals & Targets</span>
                <span className="rounded-full bg-emerald-100/80 px-2.5 py-0.5 font-mono text-[11px] font-bold text-emerald-800 uppercase tracking-wider">
                  {goalsResponse?.goals.length ?? 0} Goals
                </span>
                {!executiveExpanded && goalsResponse?.goals[0] && (
                  <span className="text-xs font-medium text-slate-500 truncate">
                    · {goalsResponse.goals[0]?.name} ({goalsResponse.goals[0]?.progress_pct?.toFixed(0) ?? 0}%)
                  </span>
                )}
              </h2>
              {executiveExpanded && (
                <p className="text-xs text-slate-500 mt-0.5">
                  ติดตามความคืบหน้าเป้าหมายทางการเงินและสัดส่วนสินทรัพย์ตามกลยุทธ์
                </p>
              )}
            </div>
          </div>
          <div className="flex items-center gap-2 shrink-0">
            <button
              type="button"
              onClick={() => setExecutiveExpanded(!executiveExpanded)}
              aria-expanded={executiveExpanded}
              className="rounded-xl border border-emerald-200/80 bg-white px-3.5 py-1.5 text-xs font-bold text-slate-600 hover:bg-emerald-50 hover:text-slate-900 transition-colors flex items-center gap-1.5 shadow-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-section-accent/45"
              title={executiveExpanded ? 'ย่อส่วน Goals' : 'ขยายส่วน Goals'}
            >
              <GoalIcon className="w-3.5 h-3.5 text-section-accent" />
              <span>Goals ({goalsResponse?.goals.length ?? 0})</span>
              {executiveExpanded ? (
                <ChevronUpIcon className="w-3.5 h-3.5 text-slate-500" />
              ) : (
                <ChevronDownIcon className="w-3.5 h-3.5 text-slate-500" />
              )}
            </button>
          </div>
        </div>

        {executiveExpanded && (
          <div className="mt-5 animate-fade-in">
            <Suspense
              fallback={
                <div className="rounded-xl border border-edge bg-white/70 p-6 text-center text-xs text-slate-500 animate-pulse">
                  กำลังโหลดเป้าหมายการเงิน...
                </div>
              }
            >
              <PortfolioGoalsTab
                goals={goalsResponse?.goals ?? []}
                generatedAt={goalsResponse?.generated_at ?? null}
                portfolios={portfolios}
                selectedPortfolioId={selectedPortfolioId}
                allocationPolicy={portfolioState?.allocation_policy}
                bucketSummaries={allocationsResponse?.summaries ?? []}
                onSuccess={(res) => {
                  setGoalsResponse(res)
                  api.getActualGoals().then((gRes) => setGoalsResponse(gRes)).catch(() => {})
                }}
              />
            </Suspense>
          </div>
        )}
      </div>

      {/* Tabs Control */}
      <div className="pt-2">
        <SegmentedControl options={TABS} value={activeTab} onChange={handleTabChange} wrap={false} />
      </div>

      {/* Tab Panels with Keep-Alive */}
      <div className="mt-4">
        {mountedTabs['overview'] && (
          <div className={activeTab === 'overview' ? 'block' : 'hidden'}>
            <PortfolioOverviewTab
              portfolioId={selectedPortfolioId}
              loading={loading}
              targets={portfolioState?.allocation_targets ?? []}
              summaries={allocationsResponse?.summaries ?? []}
              warning={allocationsResponse?.warning ?? null}
              holdings={portfolioState?.holdings ?? []}
              allocationPolicy={portfolioState?.allocation_policy}
              allocationPlanOrigin={portfolioState?.allocation_plan_origin}
              performanceRows={performanceRows}
              daysRange={daysRange}
              onChangeDaysRange={setDaysRange}
              onSnapshotTriggered={() => {
                api.getActualPerformance(daysRange, selectedPortfolioId).then((rows) => {
                  setPerformanceRows(rows)
                }).catch(() => {})
              }}
              onSelectBucket={handleSelectBucket}
              onSuccess={handlePortfolioStateSuccess}
              onOpenTradeModal={() => setTradeModalOpen(true)}
              onNavigateToPrinciples={() => {
                const nextParams = new URLSearchParams(searchParams)
                nextParams.set('tab', 'principles')
                setSearchParams(nextParams)
              }}
            />
          </div>
        )}

        <Suspense
          fallback={
            <div className="flow-panel rounded-2xl border border-edge p-8 text-center text-xs text-slate-500 animate-pulse">
              กำลังโหลดข้อมูลแท็บพอร์ตโฟลิโอ...
            </div>
          }
        >
          {mountedTabs['principles'] && (
            <div className={activeTab === 'principles' ? 'block space-y-6' : 'hidden'}>
              <InvestorPrinciplesTab
                portfolioId={selectedPortfolioId}
                portfolioName={currentPortfolioName}
                onPortfolioChangeNeeded={() => void fetchAllData(false)}
                onNavigateToOverview={() => {
                  const nextParams = new URLSearchParams(searchParams)
                  nextParams.set('tab', 'overview')
                  setSearchParams(nextParams)
                }}
              />

              <div className="rounded-2xl border border-rose-200/70 bg-rose-50/40 p-4 sm:p-5 flex flex-col sm:flex-row sm:items-center justify-between gap-3">
                <div>
                  <h3 className="text-sm font-bold text-rose-900">Danger Zone — รีเซ็ตข้อมูลพอร์ต</h3>
                  <p className="text-xs text-rose-700/90 mt-0.5">
                    ล้างประวัติธุรกรรม การถือครอง และเป้าหมายของพอร์ตนี้กลับสู่ค่าเริ่มต้น
                  </p>
                </div>
                <button
                  type="button"
                  onClick={() => setResetModalOpen(true)}
                  className="shrink-0 rounded-xl border border-rose-200 bg-white px-3.5 py-2 text-xs sm:text-sm font-bold text-rose-600 shadow-sm hover:bg-rose-600 hover:text-white active:scale-98 transition-all flex items-center gap-1.5 whitespace-nowrap"
                  title="ล้างข้อมูลพอร์ตทั้งหมดกลับเป็นค่าเริ่มต้น"
                >
                  <ResetIcon className="w-4 h-4 shrink-0" />
                  <span>ล้างพอร์ต</span>
                </button>
              </div>
            </div>
          )}

          {mountedTabs['holdings'] && (
            <div className={activeTab === 'holdings' ? 'block' : 'hidden'}>
              <PortfolioHoldingsTab
                key={selectedPortfolioId}
                loading={loading}
                portfolioId={selectedPortfolioId}
                holdings={portfolioState?.holdings ?? []}
                selectedBucket={selectedBucket}
                onClearBucketFilter={handleClearBucketFilter}
                targets={portfolioState?.allocation_targets ?? []}
                allocationPolicy={portfolioState?.allocation_policy}
                onSuccess={(newState) => {
                  if (selectedPortfolioIdRef.current === selectedPortfolioId) handlePortfolioStateSuccess(newState)
                }}
                journalRows={journalRows}
                journalKeyword={journalKeyword}
                onChangeJournalKeyword={setJournalKeyword}
                onSuccessJournal={(entries) => setJournalRows(entries)}
                onOpenTradeModal={() => setTradeModalOpen(true)}
                onViewTransactions={handleViewTransactionsForSymbol}
                watchlistedSymbols={watchlistedSymbols}
                onWatchlistChanged={refreshWatchlist}
              />
            </div>
          )}

          {mountedTabs['transactions'] && (
            <div className={activeTab === 'transactions' ? 'block' : 'hidden'}>
              <PortfolioTransactionsTab
                key={selectedPortfolioId}
                portfolioId={selectedPortfolioId}
                activeView={selectedTransactionView}
                onViewChange={handleTransactionViewChange}
                initialSymbol={transactionFilterSymbol}
                holdings={portfolioState?.holdings}
                onClearSymbolFilter={handleClearSymbolFilter}
                onOpenTradeModal={() => setTradeModalOpen(true)}
                onSuccess={(newState) => {
                  if (selectedPortfolioIdRef.current === selectedPortfolioId) handlePortfolioStateSuccess(newState)
                }}
              />
            </div>
          )}

          {mountedTabs['incomes'] && (
            <div className={activeTab === 'incomes' ? 'block' : 'hidden'}>
              <PortfolioIncomesTab
                state={portfolioState}
                selectedPortfolioId={selectedPortfolioId}
                onSuccess={handlePortfolioStateSuccess}
                onOpenIncomeModal={() => setIncomeModalOpen(true)}
              />
            </div>
          )}

          {mountedTabs['analytics'] && (
            <div className={activeTab === 'analytics' ? 'block' : 'hidden'}>
              <PortfolioAnalyticsTab
                performanceRows={performanceRows}
                daysRange={daysRange}
                onChangeDaysRange={setDaysRange}
                holdings={portfolioState?.holdings ?? []}
              />
            </div>
          )}

          {mountedTabs['watchlist'] && (
            <div className={activeTab === 'watchlist' ? 'block' : 'hidden'}>
              <PortfolioWatchlistTab
                portfolioId={selectedPortfolioId}
                items={watchlistState?.items ?? []}
                lastUpdated={watchlistState?.last_updated ?? null}
                onSuccess={(wState) => setWatchlistState(wState)}
              />
            </div>
          )}

          {mountedTabs['calendar'] && (
            <div className={activeTab === 'calendar' ? 'block' : 'hidden'}>
              <PortfolioCalendarTab selectedPortfolioId={selectedPortfolioId} />
            </div>
          )}
        </Suspense>
      </div>

      <Suspense fallback={null}>
        {tradeModalOpen && (
          <TradeModal
            targets={portfolioState?.allocation_targets ?? []}
            holdings={portfolioState?.holdings ?? []}
            selectedPortfolioId={selectedPortfolioId}
            onClose={() => setTradeModalOpen(false)}
            onSuccess={handlePortfolioStateSuccess}
          />
        )}

        {cashFlowModalOpen && (
          <CashFlowModal
            selectedPortfolioId={selectedPortfolioId}
            onClose={() => setCashFlowModalOpen(false)}
            onSuccess={handlePortfolioStateSuccess}
          />
        )}

        {incomeModalOpen && (
          <IncomeModal
            holdingsSymbols={(portfolioState?.holdings ?? []).map((h) => h.symbol)}
            selectedPortfolioId={selectedPortfolioId}
            onClose={() => setIncomeModalOpen(false)}
            onSuccess={handlePortfolioStateSuccess}
          />
        )}

        {resetModalOpen && (
          <ResetConfirmModal
            selectedPortfolioId={selectedPortfolioId}
            onClose={() => setResetModalOpen(false)}
            onSuccess={handlePortfolioStateSuccess}
          />
        )}
      </Suspense>
    </div>
  )
}
