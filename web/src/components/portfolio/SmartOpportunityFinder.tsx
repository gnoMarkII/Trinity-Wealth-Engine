import { useState, useEffect, useMemo } from 'react'
import { api } from '../../api/client'
import type {
  PortfolioOpportunitiesResponseDTO,
  CandidateEvaluationResultDTO,
  ActualWatchlistStateDTO,
} from '../../api/types'
import OpportunityCandidateCard from './OpportunityCandidateCard'
import WatchlistModal, { type WatchlistDraft } from './Modals/WatchlistModal'
import { useFocusTrap } from '../../hooks/useFocusTrap'

interface Props {
  isOpen: boolean
  portfolioId: string
  onClose: () => void
  onWatchlistUpdated?: (state: ActualWatchlistStateDTO) => void
}

export default function SmartOpportunityFinder({
  isOpen,
  portfolioId,
  onClose,
  onWatchlistUpdated,
}: Props) {
  const panelRef = useFocusTrap<HTMLDivElement>(isOpen, onClose)

  const [data, setData] = useState<PortfolioOpportunitiesResponseDTO | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // Filters
  const [searchQuery, setSearchQuery] = useState('')
  const [priorityFilter, setPriorityFilter] = useState<'all' | 'actionable' | 'p1'>('all')
  const [marketFilter, setMarketFilter] = useState<'all' | 'US' | 'TH'>('all')
  const [capacityFilter, setCapacityFilter] = useState<'all' | 'available' | 'constrained'>('all')

  // Watchlist modal state for prefilled draft
  const [draftModalOpen, setDraftModalOpen] = useState(false)
  const [activeDraft, setActiveDraft] = useState<WatchlistDraft | null>(null)

  const loadOpportunities = async (forceRefresh: boolean = false) => {
    setLoading(true)
    setError(null)
    try {
      const res = forceRefresh
        ? await api.getPortfolioOpportunities(portfolioId, true)
        : await api.getPortfolioOpportunities(portfolioId)
      setData(res)
    } catch (err: any) {
      setError(err?.message || 'ไม่สามารถโหลดข้อมูล Opportunity Finder ได้')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    if (isOpen) {
      loadOpportunities()
    }
  }, [isOpen, portfolioId])

  const handleOpenDraftModal = (result: CandidateEvaluationResultDTO) => {
    const candidate = result.candidate
    const draft: WatchlistDraft = {
      symbol: candidate.symbol,
      asset_type: candidate.asset_type === 'REIT' ? 'REIT' : 'Stock',
      target_price: result.target_price ?? null,
      notes: `[3D Finder P${result.research_priority}] Macro: ${result.macro_alignment}, Room: ${result.capacity_status}`,
      suggested_bucket_id: result.suggested_bucket_id ?? undefined,
      evaluation_ref: data?.evaluation_id,
      source_refs: candidate.source_refs,
      target_price_basis: result.target_price_basis ?? undefined,
    }
    setActiveDraft(draft)
    setDraftModalOpen(true)
  }

  const handleWatchlistSuccess = (state: ActualWatchlistStateDTO) => {
    if (onWatchlistUpdated) {
      onWatchlistUpdated(state)
    }
    // Update local candidates state so card reflects already_watchlisted = true
    if (data && activeDraft?.symbol) {
      setData({
        ...data,
        candidates: data.candidates.map((c) =>
          c.candidate.symbol === activeDraft.symbol
            ? { ...c, candidate: { ...c.candidate, already_watchlisted: true } }
            : c
        ),
      })
    }
  }

  // Filtered Candidates
  const filteredCandidates = useMemo(() => {
    if (!data?.candidates) return []
    return data.candidates.filter((item) => {
      const { candidate } = item

      // Search Query
      if (searchQuery.trim()) {
        const query = searchQuery.trim().toLowerCase()
        const matchesSym = candidate.symbol.toLowerCase().includes(query)
        const matchesName = candidate.company_name.toLowerCase().includes(query)
        const matchesSector = candidate.sector_name.toLowerCase().includes(query)
        if (!matchesSym && !matchesName && !matchesSector) return false
      }

      // Priority Filter
      if (priorityFilter === 'actionable' && !item.is_actionable) return false
      if (priorityFilter === 'p1' && item.research_priority !== 1) return false

      // Market Filter
      if (marketFilter !== 'all' && candidate.market !== marketFilter) return false

      // Capacity Filter
      if (capacityFilter === 'available' && item.capacity_status !== 'available') return false
      if (capacityFilter === 'constrained' && item.capacity_status !== 'constrained') return false

      return true
    })
  }, [data, searchQuery, priorityFilter, marketFilter, capacityFilter])

  if (!isOpen) return null

  return (
    <div className="fixed inset-0 z-50 overflow-hidden" role="region" aria-label="3D Smart Opportunity Finder Modal">
      {/* Backdrop */}
      <div
        aria-hidden="true"
        className="animate-fade-in fixed inset-0 bg-black/50 backdrop-blur-xs transition-opacity"
        onClick={onClose}
      />

      {/* Main Drawer Container */}
      <div className="fixed inset-y-0 right-0 flex max-w-full pl-4 sm:pl-10">
        <div
          ref={panelRef}
          role="dialog"
          aria-modal="true"
          className="w-screen max-w-3xl transform bg-white shadow-2xl transition-all duration-300 ease-out flex flex-col border-l border-sky-100"
        >
          {/* Header */}
          <div className="p-5 border-b border-zinc-200 bg-gradient-to-r from-sky-50/70 via-white to-indigo-50/40 flex items-center justify-between">
            <div className="space-y-1">
              <div className="flex items-center gap-2">
                <span className="text-xl">🌟</span>
                <h2 className="text-lg font-bold text-zinc-900 tracking-tight">
                  3-Dimensional Smart Opportunity Finder
                </h2>
                <span className="px-2 py-0.5 rounded-full text-xs font-bold bg-sky-100 text-sky-800 border border-sky-200">
                  {portfolioId}
                </span>
              </div>
              <p className="text-xs text-zinc-500">
                คัดกรองโอกาสลงทุน 3 มิติ: Macro Sector Rotation × Investor Principles × Look-Through Capacity
              </p>
            </div>
            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={() => loadOpportunities(true)}
                disabled={loading}
                className="p-2 rounded-xl text-zinc-500 hover:text-zinc-800 hover:bg-zinc-100 transition-colors"
                title="Refresh Candidates"
              >
                🔄
              </button>
              <button
                type="button"
                onClick={onClose}
                className="p-2 rounded-xl text-zinc-400 hover:text-zinc-700 hover:bg-zinc-100 transition-colors font-bold text-lg leading-none"
                aria-label="Close"
              >
                ✕
              </button>
            </div>
          </div>

          {/* Content Area */}
          <div className="flex-1 overflow-y-auto p-5 space-y-5 custom-scrollbar">
            {/* KPI Summary Banner */}
            {data && (
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 p-4 rounded-2xl bg-gradient-to-br from-indigo-50/80 via-white to-sky-50/80 border border-indigo-100/80 shadow-2xs">
                <div>
                  <div className="text-[11px] font-medium text-zinc-500 uppercase tracking-wider">Scanned Candidates</div>
                  <div className="text-xl font-extrabold text-zinc-900 font-mono mt-0.5">
                    {data.total_candidates_scanned}
                  </div>
                </div>
                <div>
                  <div className="text-[11px] font-medium text-zinc-500 uppercase tracking-wider">Passed Principles</div>
                  <div className="text-xl font-extrabold text-emerald-600 font-mono mt-0.5">
                    {data.passed_candidates_count}
                  </div>
                </div>
                <div>
                  <div className="text-[11px] font-medium text-zinc-500 uppercase tracking-wider">P1 Top Priority</div>
                  <div className="text-xl font-extrabold text-flow-blue font-mono mt-0.5">
                    {data.candidates.filter((c) => c.research_priority === 1).length}
                  </div>
                </div>
                <div>
                  <div className="text-[11px] font-medium text-zinc-500 uppercase tracking-wider">Readiness</div>
                  <div className="mt-0.5">
                    <span className={`inline-block px-2 py-0.5 rounded text-xs font-bold ${
                      data.readiness === 'ready'
                        ? 'bg-emerald-100 text-emerald-800'
                        : 'bg-amber-100 text-amber-800'
                    }`}>
                      {data.readiness.toUpperCase()}
                    </span>
                  </div>
                </div>
              </div>
            )}

            {/* Warnings or Missing Requirements */}
            {data?.warnings && data.warnings.length > 0 && (
              <div className="p-3.5 rounded-xl bg-amber-50/90 border border-amber-200 text-xs text-amber-800 space-y-1">
                <div className="font-bold flex items-center gap-1.5">
                  <span>⚠️</span> ข้อสังเกตความพร้อมข้อมูล:
                </div>
                <ul className="list-disc list-inside space-y-0.5 pl-1 text-[11px]">
                  {data.warnings.map((w, i) => (
                    <li key={i}>{w}</li>
                  ))}
                </ul>
              </div>
            )}

            {/* Filters Bar */}
            <div className="space-y-3 p-4 rounded-2xl bg-zinc-50/80 border border-zinc-200/80">
              <div className="flex flex-col sm:flex-row items-stretch sm:items-center gap-3">
                <div className="flex-1 relative">
                  <span className="absolute left-3 top-2.5 text-zinc-400 text-xs">🔍</span>
                  <input
                    type="text"
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    placeholder="ค้นหา Symbol, ชื่อหุ้น หรือ Sector..."
                    className="w-full pl-8 pr-3 py-1.5 rounded-xl border border-zinc-200 bg-white text-xs text-zinc-900 focus:outline-none focus:border-flow-blue"
                  />
                </div>

                <div className="flex items-center gap-2 overflow-x-auto pb-1 sm:pb-0">
                  {/* Priority Filter */}
                  <select
                    value={priorityFilter}
                    onChange={(e: any) => setPriorityFilter(e.target.value)}
                    className="px-2.5 py-1.5 rounded-xl border border-zinc-200 bg-white text-xs font-medium text-zinc-700"
                  >
                    <option value="all">Priority: ทั้งหมด</option>
                    <option value="actionable">Actionable เท่านั้น</option>
                    <option value="p1">P1 Top Priority เท่านั้น</option>
                  </select>

                  {/* Market Filter */}
                  <select
                    value={marketFilter}
                    onChange={(e: any) => setMarketFilter(e.target.value)}
                    className="px-2.5 py-1.5 rounded-xl border border-zinc-200 bg-white text-xs font-medium text-zinc-700"
                  >
                    <option value="all">ตลาด: ทั้งหมด</option>
                    <option value="US">US (สหรัฐฯ)</option>
                    <option value="TH">TH (ไทย)</option>
                  </select>

                  {/* Capacity Filter */}
                  <select
                    value={capacityFilter}
                    onChange={(e: any) => setCapacityFilter(e.target.value)}
                    className="px-2.5 py-1.5 rounded-xl border border-zinc-200 bg-white text-xs font-medium text-zinc-700"
                  >
                    <option value="all">Capacity: ทั้งหมด</option>
                    <option value="available">มีวงเงินว่าง (Room Available)</option>
                    <option value="constrained">ติดเพดาน (Constrained)</option>
                  </select>
                </div>
              </div>
            </div>

            {/* Candidates List or States */}
            {loading ? (
              <div className="py-20 flex flex-col items-center justify-center gap-3 text-zinc-400">
                <span className="animate-spin text-3xl">⏳</span>
                <span className="text-xs font-medium">กำลังสแกนและประเมินโอกาสลงทุน 3 มิติ...</span>
              </div>
            ) : error ? (
              <div className="p-6 rounded-2xl bg-rose-50 border border-rose-200 text-center space-y-3">
                <div className="text-rose-600 font-bold text-sm">เกิดข้อผิดพลาด</div>
                <div className="text-xs text-rose-500">{error}</div>
                <button
                  type="button"
                  onClick={() => loadOpportunities(false)}
                  className="px-4 py-1.5 rounded-xl bg-rose-600 text-white text-xs font-bold hover:bg-rose-700 transition-colors"
                >
                  ลองใหม่อีกครั้ง
                </button>
              </div>
            ) : filteredCandidates.length === 0 ? (
              <div className="py-16 text-center text-zinc-400 space-y-2">
                <span className="text-3xl block">🔍</span>
                <p className="text-sm font-semibold text-zinc-600">ไม่พบรายการโอกาสที่ตรงกับเงื่อนไข</p>
                <p className="text-xs text-zinc-400">ลองปรับเปลี่ยนตัวกรองค้นหา หรือขยายมิติการค้นหา</p>
              </div>
            ) : (
              <div className="space-y-4">
                <div className="flex items-center justify-between text-xs text-zinc-500 px-1">
                  <span>พบ {filteredCandidates.length} รายการ</span>
                  <span>จัดเรียงตาม Deterministic Priority & Score</span>
                </div>
                {filteredCandidates.map((res) => (
                  <OpportunityCandidateCard
                    key={res.candidate.security_id}
                    result={res}
                    onAddToWatchlist={handleOpenDraftModal}
                  />
                ))}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Prefilled Watchlist Modal */}
      {draftModalOpen && activeDraft && (
        <WatchlistModal
          portfolioId={portfolioId}
          initialDraft={activeDraft}
          onClose={() => {
            setDraftModalOpen(false)
            setActiveDraft(null)
          }}
          onSuccess={handleWatchlistSuccess}
        />
      )}
    </div>
  )
}
