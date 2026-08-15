import { useState, useEffect, useCallback, useRef } from 'react'
import { createPortal } from 'react-dom'
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
import PortfolioCalendarTab from '../components/portfolio/PortfolioCalendarTab'
import PortfolioTransactionsTab from '../components/portfolio/PortfolioTransactionsTab'
import PortfolioIncomesTab from '../components/portfolio/PortfolioIncomesTab'
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
      <span className="flex items-center gap-2 text-sm sm:text-base font-semibold">
        <ChartBarIcon className="w-4 h-4 sm:w-5 sm:h-5 text-flow-blue shrink-0" />
        <span>Strategy Buckets & Allocation</span>
      </span>
    ),
  },
  {
    key: 'holdings',
    label: (
      <span className="flex items-center gap-2 text-sm sm:text-base font-semibold">
        <BriefcaseIcon className="w-4 h-4 sm:w-5 sm:h-5 text-emerald-600 shrink-0" />
        <span>Holdings & Trading Journal</span>
      </span>
    ),
  },
  {
    key: 'transactions',
    label: (
      <span className="flex items-center gap-2 text-sm sm:text-base font-semibold">
        <TradeIcon className="w-4 h-4 sm:w-5 sm:h-5 text-sky-500 shrink-0" />
        <span>Transactions</span>
      </span>
    ),
  },
  {
    key: 'incomes',
    label: (
      <span className="flex items-center gap-2 text-sm sm:text-base font-semibold">
        <IncomeIcon className="w-4 h-4 sm:w-5 sm:h-5 text-emerald-500 shrink-0" />
        <span>Dividends & Income</span>
      </span>
    ),
  },
  {
    key: 'watchlist',
    label: (
      <span className="flex items-center gap-2 text-sm sm:text-base font-semibold">
        <WatchlistIcon className="w-4 h-4 sm:w-5 sm:h-5 text-amber-500 shrink-0" />
        <span>Watchlist</span>
      </span>
    ),
  },
  {
    key: 'calendar',
    label: (
      <span className="flex items-center gap-2 text-sm sm:text-base font-semibold">
        <span className="text-base shrink-0">📅</span>
        <span>Corporate Calendar</span>
      </span>
    ),
  },
]

export default function Portfolio() {
  const [searchParams, setSearchParams] = useSearchParams()
  const activeTab = searchParams.get('tab') || 'overview'
  const selectedBucket = searchParams.get('bucket') || null
  const selectedSymbol = searchParams.get('symbol') || null
  const selectedPortfolioId = searchParams.get('portfolio_id') || 'default'

  // Multi-Portfolio state
  const [portfolios, setPortfolios] = useState<PortfolioMetaDTO[]>([])
  const [createPortModalOpen, setCreatePortModalOpen] = useState(false)
  const [newPortName, setNewPortName] = useState('')
  const [creatingPort, setCreatingPort] = useState(false)
  const [renamePortModalOpen, setRenamePortModalOpen] = useState(false)
  const [editPortName, setEditPortName] = useState('')
  const [renamingPort, setRenamingPort] = useState(false)


  // Portfolio details state
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
    const myFetchId = ++fetchIdRef.current
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

      const [ports, pState, allocRes, wState, gRes] = await Promise.all([
        api.listPortfolios(),
        api.getActualPortfolioState(refreshPrices, refreshPrices, selectedPortfolioId),
        api.getActualBucketAllocations(selectedPortfolioId),
        api.getActualWatchlist(selectedPortfolioId),
        // Goals ไม่กรองตามพอร์ตที่เลือกอยู่ — โชว์ทุกเป้าหมายจากทุกพอร์ตรวมกันเสมอ (ดู badge ในการ์ด)
        api.getActualGoals(),
      ])

      if (fetchIdRef.current !== myFetchId) return // มี fetch ใหม่กว่าแซงไปแล้ว ทิ้งผลลัพธ์นี้

      setPortfolios(ports)
      setPortfolioState(pState)
      setAllocationsResponse(allocRes)
      setWatchlistState(wState)
      setGoalsResponse(gRes)

      if (refreshPrices) {
        api.getActualPerformance(daysRange, selectedPortfolioId).then((rows) => {
          if (fetchIdRef.current === myFetchId) setPerformanceRows(rows)
        }).catch(() => {})
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

  const handleViewTransactionsForSymbol = (symbol: string) => {
    const nextParams = new URLSearchParams(searchParams)
    nextParams.set('tab', 'transactions')
    nextParams.set('symbol', symbol)
    setSearchParams(nextParams)
  }

  const handleClearSymbolFilter = () => {
    const nextParams = new URLSearchParams(searchParams)
    nextParams.delete('symbol')
    setSearchParams(nextParams)
  }

  const handleCreatePortfolio = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!newPortName.trim()) return
    setCreatingPort(true)
    try {
      const meta = await api.createPortfolio(newPortName.trim())
      setCreatePortModalOpen(false)
      setNewPortName('')
      const nextParams = new URLSearchParams(searchParams)
      nextParams.set('portfolio_id', meta.id)
      setSearchParams(nextParams)
    } catch (err: any) {
      alert(err?.message || 'ไม่สามารถสร้างพอร์ตใหม่ได้')
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
    } catch (err: any) {
      alert(err?.message || 'ไม่สามารถแก้ไขชื่อพอร์ตได้')
    } finally {
      setRenamingPort(false)
    }
  }

  const handleDeletePortfolio = async () => {
    if (selectedPortfolioId === 'default') return
    const currentMeta = portfolios.find((p) => p.id === selectedPortfolioId)
    if (!window.confirm(`คุณต้องการลบพอร์ต "${currentMeta?.name || selectedPortfolioId}" หรือไม่?`)) return
    try {
      await api.deletePortfolio(selectedPortfolioId)
      const nextParams = new URLSearchParams(searchParams)
      nextParams.set('portfolio_id', 'default')
      setSearchParams(nextParams)
    } catch (err: any) {
      alert(err?.message || 'ไม่สามารถลบพอร์ตได้')
    }
  }

  return (
    <div className="animate-page-in space-y-6 pb-14">
      {/* Page Header with Multi-Portfolio Selector — Hero Banner Card */}
      <div className="flow-panel rounded-2xl border border-sky-100/80 p-5 md:p-6 shadow-sm bg-gradient-to-r from-white/90 via-sky-50/40 to-blue-50/30 backdrop-blur-md flex flex-col justify-between gap-4 md:flex-row md:items-center">
        {/* Left: Branding & Title */}
        <div className="space-y-1.5 max-w-xl">
          <h1 className="text-2xl md:text-3xl lg:text-4xl font-extrabold tracking-tight text-zinc-900">
            Actual Portfolio Hub
          </h1>
          <p className="text-xs md:text-sm text-zinc-500 font-medium leading-relaxed">
            ศูนย์กลางติดตามพอร์ตการลงทุนจริง (Actual Portfolio) พร้อมกลยุทธ์สัดส่วน การจัดกลุ่ม Bucket และระบบบันทึก Trading Journal
          </p>
        </div>

        {/* Right: Portfolio Selector & Action Toolbar */}
        <div className="flex flex-col items-stretch md:items-end gap-3 shrink-0">
          {/* Portfolio Selector Pill */}
          <div className="flex w-full items-center justify-between gap-2 bg-white/95 backdrop-blur border border-sky-200/90 shadow-xs rounded-xl px-3.5 py-1.5 transition-all hover:border-sky-300">
            <span className="text-xs sm:text-sm font-bold text-zinc-600 flex items-center gap-1.5 shrink-0">
              <span>📁</span> พอร์ต:
            </span>
            <select
              value={selectedPortfolioId}
              onChange={(e) => {
                const nextParams = new URLSearchParams(searchParams)
                nextParams.set('portfolio_id', e.target.value)
                setSearchParams(nextParams)
              }}
              className="flex-1 min-w-0 rounded-lg border border-sky-200 bg-sky-50/60 px-3 py-1.5 text-xs sm:text-sm font-bold text-zinc-800 shadow-xs focus:outline-none focus:ring-2 focus:ring-sky-400 cursor-pointer hover:bg-white transition-colors"
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
              onClick={handleOpenRenameModal}
              className="shrink-0 rounded-lg bg-sky-100/80 px-2.5 py-1.5 text-xs sm:text-sm font-bold text-sky-700 hover:bg-sky-600 hover:text-white active:scale-95 transition-all flex items-center gap-1"
              title="แก้ไขชื่อพอร์ตนี้"
            >
              ✏️
            </button>
            <button
              type="button"
              onClick={() => setCreatePortModalOpen(true)}
              className="shrink-0 rounded-lg bg-emerald-600 px-3.5 py-1.5 text-xs sm:text-sm font-bold text-white shadow-xs hover:bg-emerald-700 active:scale-95 transition-all flex items-center gap-1"
              title="สร้างพอร์ตใหม่"
            >
              <span>+ พอร์ตใหม่</span>
            </button>
            {selectedPortfolioId !== 'default' && (
              <button
                type="button"
                onClick={handleDeletePortfolio}
                className="shrink-0 rounded-lg bg-rose-100/80 px-2.5 py-1.5 text-xs sm:text-sm font-bold text-rose-700 hover:bg-rose-600 hover:text-white active:scale-95 transition-all"
                title="ลบพอร์ตนี้"
              >
                🗑️
              </button>
            )}
          </div>

          {/* Action Toolbar */}
          <div className="flex flex-wrap items-center justify-between md:justify-end gap-2 sm:gap-2.5 w-full">
            <button
              type="button"
              onClick={() => setTradeModalOpen(true)}
              className="rounded-xl bg-gradient-to-r from-sky-500 to-blue-600 px-4 py-2 sm:py-2.5 text-xs sm:text-sm font-bold text-white shadow-md shadow-sky-500/20 hover:from-sky-600 hover:to-blue-700 hover:shadow-lg hover:shadow-sky-500/30 active:scale-98 transition-all flex items-center gap-1.5 sm:gap-2 whitespace-nowrap"
            >
              <TradeIcon className="w-4 h-4 text-white shrink-0" />
              <span>บันทึกเทรด (Trade)</span>
            </button>
            <button
              type="button"
              onClick={() => setCashFlowModalOpen(true)}
              className="rounded-xl border border-sky-200/90 bg-white/90 px-3.5 sm:px-4 py-2 sm:py-2.5 text-xs sm:text-sm font-bold text-zinc-700 shadow-xs hover:bg-sky-50/90 hover:border-sky-300 active:scale-98 transition-all flex items-center gap-1.5 whitespace-nowrap"
            >
              <CashFlowIcon className="w-4 h-4 text-emerald-600 shrink-0" />
              <span>ฝาก/ถอน (Cash)</span>
            </button>
            <button
              type="button"
              onClick={() => setIncomeModalOpen(true)}
              className="rounded-xl border border-sky-200/90 bg-white/90 px-3.5 sm:px-4 py-2 sm:py-2.5 text-xs sm:text-sm font-bold text-zinc-700 shadow-xs hover:bg-sky-50/90 hover:border-sky-300 active:scale-98 transition-all flex items-center gap-1.5 whitespace-nowrap"
            >
              <IncomeIcon className="w-4 h-4 text-emerald-600 shrink-0" />
              <span>ปันผล/รายรับ (Income)</span>
            </button>
            <div className="h-5 w-px bg-sky-200/60 mx-0.5 hidden sm:block" />
            <button
              type="button"
              onClick={() => setResetModalOpen(true)}
              className="rounded-xl border border-rose-200/70 bg-rose-50/60 px-3 sm:px-3.5 py-2 sm:py-2.5 text-xs sm:text-sm font-bold text-rose-600 shadow-xs hover:bg-rose-600 hover:text-white active:scale-98 transition-all flex items-center gap-1.5 whitespace-nowrap"
              title="ล้างข้อมูลพอร์ตทั้งหมดกลับเป็นค่าเริ่มต้น"
            >
              <ResetIcon className="w-4 h-4 shrink-0" />
              <span>ล้างพอร์ต</span>
            </button>
          </div>
        </div>
      </div>

      {/* Modal: Create Portfolio */}
      {createPortModalOpen &&
        createPortal(
          <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4 backdrop-blur-sm animate-fade-in">
            <div className="w-full max-w-sm rounded-2xl bg-white p-6 shadow-2xl border border-sky-100 animate-scale-up">
              <h3 className="text-lg font-bold text-zinc-900 mb-2">📁 สร้างพอร์ตการลงทุนใหม่</h3>
              <p className="text-xs text-zinc-500 mb-4">
                เช่น "พอร์ตเงินสำรองฉุกเฉิน", "พอร์ตเกษียณ", "พอร์ตปันผล" เพื่อแยกสินทรัพย์ออกจากกันเด็ดขาด
              </p>
              <form onSubmit={handleCreatePortfolio} className="space-y-4">
                <div>
                  <label htmlFor="create-port-name" className="block text-xs font-semibold text-zinc-700 mb-1">ชื่อพอร์ต (Portfolio Name)</label>
                  <input
                    id="create-port-name"
                    type="text"
                    required
                    value={newPortName}
                    onChange={(e) => setNewPortName(e.target.value)}
                    placeholder="e.g. พอร์ตเงินสำรองฉุกเฉิน"
                    className="w-full rounded-xl border border-sky-200 px-3.5 py-2 text-sm font-bold text-zinc-900 focus:outline-none focus:ring-2 focus:ring-flow-blue"
                  />
                </div>
                <div className="flex justify-end gap-2 pt-2">
                  <button
                    type="button"
                    onClick={() => setCreatePortModalOpen(false)}
                    className="rounded-xl px-4 py-2 text-xs font-bold text-zinc-600 hover:bg-zinc-100"
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
            </div>
          </div>,
          document.body
        )}

      {/* Modal: Rename Portfolio */}
      {renamePortModalOpen &&
        createPortal(
          <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4 backdrop-blur-sm animate-fade-in">
            <div className="w-full max-w-sm rounded-2xl bg-white p-6 shadow-2xl border border-sky-100 animate-scale-up">
              <h3 className="text-lg font-bold text-zinc-900 mb-2">✏️ แก้ไขชื่อพอร์ต</h3>
              <p className="text-xs text-zinc-500 mb-4">
                เปลี่ยนชื่อแสดงผลของพอร์ตการลงทุนที่เลือก
              </p>
              <form onSubmit={handleRenamePortfolio} className="space-y-4">
                <div>
                  <label htmlFor="rename-port-name" className="block text-xs font-semibold text-zinc-700 mb-1">ชื่อพอร์ตใหม่ (Portfolio Name)</label>
                  <input
                    id="rename-port-name"
                    type="text"
                    required
                    value={editPortName}
                    onChange={(e) => setEditPortName(e.target.value)}
                    placeholder="เช่น พอร์ตเพื่อการเติบโต"
                    className="w-full rounded-xl border border-sky-200 px-3.5 py-2 text-sm font-bold text-zinc-900 focus:outline-none focus:ring-2 focus:ring-flow-blue"
                  />
                </div>
                <div className="flex justify-end gap-2 pt-2">
                  <button
                    type="button"
                    onClick={() => setRenamePortModalOpen(false)}
                    className="rounded-xl px-4 py-2 text-xs font-bold text-zinc-600 hover:bg-zinc-100"
                  >
                    ยกเลิก
                  </button>
                  <button
                    type="submit"
                    disabled={renamingPort}
                    className="rounded-xl bg-sky-600 px-4 py-2 text-xs font-bold text-white shadow-md hover:bg-sky-700 disabled:opacity-50"
                  >
                    {renamingPort ? 'กำลังบันทึก...' : 'บันทึก'}
                  </button>
                </div>
              </form>
            </div>
          </div>,
          document.body
        )}

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
              <GoalIcon className="w-3.5 h-3.5" />
              <span>Goals ({goalsResponse?.goals.length ?? 0})</span>
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
              <ChartBarIcon className="w-3.5 h-3.5" />
              <span>Performance Analytics</span>
            </button>
            <button
              type="button"
              onClick={() => setExecutiveExpanded(!executiveExpanded)}
              className="rounded-xl border border-sky-200 bg-white p-1.5 text-zinc-500 hover:bg-sky-50 hover:text-zinc-900 transition-colors ml-1"
              title={executiveExpanded ? 'ย่อส่วน Executive Dashboard' : 'ขยายส่วน Executive Dashboard'}
            >
              {executiveExpanded ? '🔼' : '🔽'}
            </button>
          </div>
        </div>

        {executiveExpanded && (
          <div className="mt-5 animate-fade-in">
            {executiveTab === 'goals' ? (
              <PortfolioGoalsTab
                goals={goalsResponse?.goals ?? []}
                generatedAt={goalsResponse?.generated_at ?? null}
                portfolios={portfolios}
                selectedPortfolioId={selectedPortfolioId}
                onSuccess={(res) => setGoalsResponse(res)}
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

      {/* Tabs Control */}
      <div className="pt-2">
        <SegmentedControl options={TABS} value={activeTab} onChange={handleTabChange} />
      </div>

      {/* Tab Panels */}
      <div className="mt-4">
        {activeTab === 'overview' && (
          <PortfolioOverviewTab
            portfolioId={selectedPortfolioId}
            targets={portfolioState?.allocation_targets ?? []}
            summaries={allocationsResponse?.summaries ?? []}
            warning={allocationsResponse?.warning ?? null}
            onSelectBucket={handleSelectBucket}
            onSuccess={handlePortfolioStateSuccess}
            onOpenTradeModal={() => setTradeModalOpen(true)}
          />
        )}

        {activeTab === 'holdings' && (
          <PortfolioHoldingsTab
            portfolioId={selectedPortfolioId}
            holdings={portfolioState?.holdings ?? []}
            selectedBucket={selectedBucket}
            onClearBucketFilter={handleClearBucketFilter}
            targets={portfolioState?.allocation_targets ?? []}
            onSuccess={handlePortfolioStateSuccess}
            journalRows={journalRows}
            journalKeyword={journalKeyword}
            onChangeJournalKeyword={setJournalKeyword}
            onSuccessJournal={(entries) => setJournalRows(entries)}
            onOpenTradeModal={() => setTradeModalOpen(true)}
            onViewTransactions={handleViewTransactionsForSymbol}
          />
        )}

        {activeTab === 'transactions' && (
          <PortfolioTransactionsTab
            portfolioId={selectedPortfolioId}
            initialSymbol={selectedSymbol}
            holdings={portfolioState?.holdings}
            onClearSymbolFilter={handleClearSymbolFilter}
            onOpenTradeModal={() => setTradeModalOpen(true)}
            onSuccess={handlePortfolioStateSuccess}
          />
        )}

        {activeTab === 'incomes' && (
          <PortfolioIncomesTab
            state={portfolioState}
            selectedPortfolioId={selectedPortfolioId}
            onSuccess={handlePortfolioStateSuccess}
            onOpenIncomeModal={() => setIncomeModalOpen(true)}
          />
        )}

        {activeTab === 'watchlist' && (
          <PortfolioWatchlistTab
            portfolioId={selectedPortfolioId}
            items={watchlistState?.items ?? []}
            lastUpdated={watchlistState?.last_updated ?? null}
            onSuccess={(wState) => setWatchlistState(wState)}
          />
        )}

        {activeTab === 'calendar' && (
          <PortfolioCalendarTab selectedPortfolioId={selectedPortfolioId} />
        )}
      </div>

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
    </div>
  )
}
