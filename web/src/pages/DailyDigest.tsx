import { useEffect, useState, useMemo } from 'react'
import { Link } from 'react-router-dom'
import { api } from '../api/client'
import type {
  DailyDigestResponseDTO,
  DailyDigestItemDTO,
  WeeklyDigestResponseDTO,
  DigestCandidatesResponseDTO,
} from '../api/types'

export default function DailyDigest() {
  const [activeTab, setActiveTab] = useState<'daily' | 'weekly'>('daily')
  const [newsFilter, setNewsFilter] = useState<'all' | 'portfolio' | 'unread'>('all')

  // Data states
  const [dailyData, setDailyData] = useState<DailyDigestResponseDTO | null>(null)
  const [weeklyData, setWeeklyData] = useState<WeeklyDigestResponseDTO | null>(null)
  const [dailyCandidates, setDailyCandidates] = useState<DigestCandidatesResponseDTO | null>(null)
  const [weeklyCandidates, setWeeklyCandidates] = useState<DigestCandidatesResponseDTO | null>(null)

  // UI & Loading states
  const [loading, setLoading] = useState<boolean>(true)
  const [synthesizing, setSynthesizing] = useState<boolean>(false)
  const [candidatesLoading, setCandidatesLoading] = useState<boolean>(false)
  const [error, setError] = useState<string | null>(null)
  const [showSelectionGate, setShowSelectionGate] = useState<boolean>(false)

  // Selection Gate states
  const [selectedVideoIds, setSelectedVideoIds] = useState<Set<string>>(new Set())
  const [dailyExcludedNewsIds, setDailyExcludedNewsIds] = useState<Set<string>>(new Set())
  const [weeklyExcludedNewsIds, setWeeklyExcludedNewsIds] = useState<Set<string>>(new Set())
  const [customYoutubeUrl, setCustomYoutubeUrl] = useState<string>('')
  const [customYoutubeUrls, setCustomYoutubeUrls] = useState<string[]>([])
  const [customNewsInput, setCustomNewsInput] = useState<string>('')
  const [customNewsItems, setCustomNewsItems] = useState<string[]>([])

  // Computed candidates and exclusions for active tab
  const candidatesData = activeTab === 'daily' ? dailyCandidates : weeklyCandidates
  const excludedNewsIds = activeTab === 'daily' ? dailyExcludedNewsIds : weeklyExcludedNewsIds

  // Journal Modal states
  const [journalItem, setJournalItem] = useState<DailyDigestItemDTO | null>(null)
  const [journalContent, setJournalContent] = useState<string>('')
  const [journalSaving, setJournalSaving] = useState<boolean>(false)
  const [journalSuccess, setJournalSuccess] = useState<string | null>(null)

  // 1. Initial Load: Fetch latest Daily Digest and Candidates
  useEffect(() => {
    loadInitialData()
  }, [])

  const loadInitialData = async () => {
    setLoading(true)
    setError(null)
    try {
      const [dailyRes, candRes] = await Promise.allSettled([
        api.getDailyDigest(),
        api.getDigestCandidates(1, false, 36),
      ])

      if (dailyRes.status === 'fulfilled') {
        setDailyData(dailyRes.value)
        // Auto open selection gate if no digest or stale
        if (!dailyRes.value.has_digest || dailyRes.value.is_snapshot_stale) {
          setShowSelectionGate(true)
        }
      }

      if (candRes.status === 'fulfilled') {
        setDailyCandidates(candRes.value)
        // Default select top 3 most recent YouTube videos
        const initialSelected = new Set(
          candRes.value.youtube_candidates.slice(0, 3).map((y) => y.video_id)
        )
        setSelectedVideoIds(initialSelected)
      }
    } catch (err: any) {
      setError(err?.message || 'เกิดข้อผิดพลาดในการโหลดข้อมูล')
    } finally {
      setLoading(false)
    }
  }

  // Load weekly digest & candidates on tab switch if not loaded
  const handleTabChange = async (tab: 'daily' | 'weekly') => {
    setActiveTab(tab)
    if (tab === 'weekly') {
      if (!weeklyData) {
        try {
          const res = await api.getWeeklyDigest()
          setWeeklyData(res)
        } catch (err: any) {
          setError(err?.message || 'ไม่สามารถโหลดสรุปรายสัปดาห์ได้')
        }
      }
      if (!weeklyCandidates) {
        setCandidatesLoading(true)
        try {
          const res = await api.getDigestCandidates(7, false, 168)
          setWeeklyCandidates(res)
        } catch (err: any) {
          console.error('Failed to load weekly candidates', err)
        } finally {
          setCandidatesLoading(false)
        }
      }
    } else if (tab === 'daily') {
      if (!dailyCandidates) {
        setCandidatesLoading(true)
        try {
          const res = await api.getDigestCandidates(1, false, 36)
          setDailyCandidates(res)
        } catch (err: any) {
          console.error('Failed to load daily candidates', err)
        } finally {
          setCandidatesLoading(false)
        }
      }
    }
  }

  // Refresh candidates for current tab
  const handleRefreshCandidates = async () => {
    setCandidatesLoading(true)
    try {
      if (activeTab === 'daily') {
        const res = await api.getDigestCandidates(1, true, 36)
        setDailyCandidates(res)
        const initialSelected = new Set(
          res.youtube_candidates.slice(0, 3).map((y) => y.video_id)
        )
        setSelectedVideoIds(initialSelected)
      } else {
        const res = await api.getDigestCandidates(7, true, 168)
        setWeeklyCandidates(res)
        const initialSelected = new Set(
          res.youtube_candidates.slice(0, 3).map((y) => y.video_id)
        )
        setSelectedVideoIds(initialSelected)
      }
    } catch (err: any) {
      setError('ไม่สามารถรีเฟรชรายการข้อมูลก่อนสังเคราะห์ได้: ' + (err?.message || 'Unknown error'))
    } finally {
      setCandidatesLoading(false)
    }
  }

  // Toggle Video Selection
  const toggleVideoSelection = (videoId: string) => {
    setSelectedVideoIds((prev) => {
      const next = new Set(prev)
      if (next.has(videoId)) {
        next.delete(videoId)
      } else {
        next.add(videoId)
      }
      return next
    })
  }

  // Toggle News Exclusion per tab
  const toggleNewsExclusion = (candidateId: string) => {
    if (activeTab === 'daily') {
      setDailyExcludedNewsIds((prev) => {
        const next = new Set(prev)
        if (next.has(candidateId)) {
          next.delete(candidateId)
        } else {
          next.add(candidateId)
        }
        return next
      })
    } else {
      setWeeklyExcludedNewsIds((prev) => {
        const next = new Set(prev)
        if (next.has(candidateId)) {
          next.delete(candidateId)
        } else {
          next.add(candidateId)
        }
        return next
      })
    }
  }

  // Date Picker Select Handlers
  const handleSelectDailyDate = async (targetDate: string) => {
    setLoading(true)
    setError(null)
    try {
      const res = await api.getDailyDigest(targetDate)
      setDailyData(res)
    } catch (err: any) {
      setError('ไม่สามารถโหลดข้อมูลย้อนหลังได้: ' + (err?.message || ''))
    } finally {
      setLoading(false)
    }
  }

  const handleSelectWeeklyDate = async (targetDate: string) => {
    setLoading(true)
    setError(null)
    try {
      const res = await api.getWeeklyDigest(targetDate)
      setWeeklyData(res)
    } catch (err: any) {
      setError('ไม่สามารถโหลดข้อมูลย้อนหลังได้: ' + (err?.message || ''))
    } finally {
      setLoading(false)
    }
  }

  // Pre-LLM Selection Gate Quick Actions
  const handleSelectOnlyPortfolioNews = () => {
    if (!candidatesData?.news_candidates) return
    const newExcluded = new Set<string>()
    candidatesData.news_candidates.forEach((cand) => {
      const hasBadges = cand.portfolio_badges && cand.portfolio_badges.length > 0
      if (!hasBadges) {
        newExcluded.add(cand.candidate_id)
      }
    })
    if (activeTab === 'daily') {
      setDailyExcludedNewsIds(newExcluded)
    } else {
      setWeeklyExcludedNewsIds(newExcluded)
    }
  }

  const handleSelectAllNews = () => {
    if (activeTab === 'daily') {
      setDailyExcludedNewsIds(new Set())
    } else {
      setWeeklyExcludedNewsIds(new Set())
    }
  }

  const handleDeselectAllNews = () => {
    if (!candidatesData?.news_candidates) return
    const allIds = new Set(candidatesData.news_candidates.map((c) => c.candidate_id))
    if (activeTab === 'daily') {
      setDailyExcludedNewsIds(allIds)
    } else {
      setWeeklyExcludedNewsIds(allIds)
    }
  }

  const handleSelectAllVideos = () => {
    if (!candidatesData?.youtube_candidates) return
    setSelectedVideoIds(new Set(candidatesData.youtube_candidates.map((y) => y.video_id)))
  }

  const handleDeselectAllVideos = () => {
    setSelectedVideoIds(new Set())
  }

  // Add Custom YouTube URL
  const handleAddCustomYoutube = () => {
    if (!customYoutubeUrl.trim()) return
    setCustomYoutubeUrls((prev) => [...prev, customYoutubeUrl.trim()])
    setCustomYoutubeUrl('')
  }

  // Remove Custom YouTube URL
  const handleRemoveCustomYoutube = (idx: number) => {
    setCustomYoutubeUrls((prev) => prev.filter((_, i) => i !== idx))
  }

  // Add Custom News Item
  const handleAddCustomNews = () => {
    if (!customNewsInput.trim()) return
    setCustomNewsItems((prev) => [...prev, customNewsInput.trim()])
    setCustomNewsInput('')
  }

  // Remove Custom News Item
  const handleRemoveCustomNews = (idx: number) => {
    setCustomNewsItems((prev) => prev.filter((_, i) => i !== idx))
  }

  // Trigger One-Shot LLM Synthesis
  const handleSynthesizeDaily = async () => {
    setSynthesizing(true)
    setError(null)
    try {
      const res = await api.synthesizeDailyDigest({
        selected_video_ids: Array.from(selectedVideoIds),
        excluded_news_ids: Array.from(dailyExcludedNewsIds),
        custom_youtube_urls: customYoutubeUrls,
        custom_news_items: customNewsItems,
      })
      setDailyData(res)
      setShowSelectionGate(false)
    } catch (err: any) {
      setError('เกิดข้อผิดพลาดในการสังเคราะห์: ' + (err?.message || 'Unknown error'))
    } finally {
      setSynthesizing(false)
    }
  }

  // Trigger Weekly LLM Synthesis
  const handleSynthesizeWeekly = async () => {
    setSynthesizing(true)
    setError(null)
    try {
      const res = await api.synthesizeWeeklyDigest({
        days: 7,
        selected_video_ids: Array.from(selectedVideoIds),
        excluded_news_ids: Array.from(weeklyExcludedNewsIds),
        custom_youtube_urls: customYoutubeUrls,
        custom_news_items: customNewsItems,
      })
      setWeeklyData(res)
      setShowSelectionGate(false)
    } catch (err: any) {
      setError('เกิดข้อผิดพลาดในการสังเคราะห์รายสัปดาห์: ' + (err?.message || 'Unknown error'))
    } finally {
      setSynthesizing(false)
    }
  }

  // Toggle Read State for an Item
  const handleToggleRead = async (item: DailyDigestItemDTO) => {
    const itemId = item.source_candidate_id || item.link
    const nextRead = !item.is_read

    // Optimistic UI update across both daily and weekly
    setDailyData((prev) => {
      if (!prev || !prev.digest) return prev
      return {
        ...prev,
        digest: {
          ...prev.digest,
          items: prev.digest.items.map((it) =>
            (it.source_candidate_id || it.link) === itemId
              ? { ...it, is_read: nextRead, is_new: false }
              : it
          ),
        },
      }
    })

    setWeeklyData((prev) => {
      if (!prev || !prev.digest) return prev
      return {
        ...prev,
        digest: {
          ...prev.digest,
          items: (prev.digest.items || []).map((it) =>
            (it.source_candidate_id || it.link) === itemId
              ? { ...it, is_read: nextRead, is_new: false }
              : it
          ),
        },
      }
    })

    try {
      await api.updateDigestReadState({
        item_ids: [itemId],
        mark_as_read: nextRead,
      })
    } catch (err) {
      console.error('Failed to update read state', err)
    }
  }

  // Mark all filtered items as read
  const handleMarkAllRead = async () => {
    if (!dailyData?.digest?.items) return
    const idsToMark = filteredDailyItems
      .filter((it) => !it.is_read)
      .map((it) => it.source_candidate_id || it.link)

    if (idsToMark.length === 0) return

    setDailyData((prev) => {
      if (!prev || !prev.digest) return prev
      const idSet = new Set(idsToMark)
      return {
        ...prev,
        digest: {
          ...prev.digest,
          items: prev.digest.items.map((it) =>
            idSet.has(it.source_candidate_id || it.link)
              ? { ...it, is_read: true, is_new: false }
              : it
          ),
        },
      }
    })

    setWeeklyData((prev) => {
      if (!prev || !prev.digest) return prev
      const idSet = new Set(idsToMark)
      return {
        ...prev,
        digest: {
          ...prev.digest,
          items: (prev.digest.items || []).map((it) =>
            idSet.has(it.source_candidate_id || it.link)
              ? { ...it, is_read: true, is_new: false }
              : it
          ),
        },
      }
    })

    try {
      await api.updateDigestReadState({
        item_ids: idsToMark,
        mark_as_read: true,
      })
    } catch (err) {
      console.error('Failed to mark all as read', err)
    }
  }

  // Open Journal Modal
  const handleOpenJournalModal = (item: DailyDigestItemDTO) => {
    setJournalItem(item)
    const dateStr = new Date().toISOString().slice(0, 10)
    const tickerTag = item.affected_tickers.length > 0 ? ` [${item.affected_tickers.join(', ')}]` : ''
    const prefill = `### ${dateStr} - ${item.thai_headline}${tickerTag}\n\n**เหตุการณ์ (What Happened):**\n${item.what_happened}\n\n**ผลกระทบต่อพอร์ต & วิทยานิพนธ์ (Why It Matters):**\n${item.why_it_matters}\n\n**สัญญาณวิทยานิพนธ์:** ${item.thesis_signal}\n**ที่มา:** ${item.source} (${item.link})`
    setJournalContent(prefill)
  }

  // Save Journal Entry
  const handleSaveJournal = async () => {
    if (!journalContent.trim()) return
    setJournalSaving(true)
    try {
      await api.appendJournal({ entry: journalContent }, 'default')
      setJournalSuccess('บันทึกลงใน Investment Journal เรียบร้อยแล้ว')
      setTimeout(() => {
        setJournalItem(null)
        setJournalSuccess(null)
      }, 1500)
    } catch (err: any) {
      setError('ไม่สามารถบันทึก Journal ได้: ' + err.message)
    } finally {
      setJournalSaving(false)
    }
  }

  // Filtered daily items
  const filteredDailyItems = useMemo(() => {
    if (!dailyData?.digest?.items) return []
    let list = dailyData.digest.items
    if (newsFilter === 'portfolio') {
      list = list.filter(
        (it) =>
          it.impact_type === 'direct_holding' ||
          it.impact_type === 'fund_lookthrough' ||
          it.impact_type === 'watchlist' ||
          (it.portfolio_badges && it.portfolio_badges.length > 0)
      )
    } else if (newsFilter === 'unread') {
      list = list.filter((it) => !it.is_read)
    }
    return list
  }, [dailyData, newsFilter])

  // Counts for filter tabs
  const portfolioItemsCount = useMemo(() => {
    if (!dailyData?.digest?.items) return 0
    return dailyData.digest.items.filter(
      (it) =>
        it.impact_type === 'direct_holding' ||
        it.impact_type === 'fund_lookthrough' ||
        it.impact_type === 'watchlist' ||
        (it.portfolio_badges && it.portfolio_badges.length > 0)
    ).length
  }, [dailyData])

  const unreadItemsCount = useMemo(() => {
    if (!dailyData?.digest?.items) return 0
    return dailyData.digest.items.filter((it) => !it.is_read).length
  }, [dailyData])

  // Render signal badge
  const renderThesisBadge = (signal: string) => {
    switch (signal) {
      case 'thesis_breaker':
        return (
          <span className="shrink-0 px-2 py-0.5 rounded text-[11px] font-bold bg-rose-50 text-rose-700 border border-rose-200">
            🚨 THESIS BREAKER
          </span>
        )
      case 'buy_on_dip':
        return (
          <span className="shrink-0 px-2 py-0.5 rounded text-[11px] font-bold bg-emerald-50 text-emerald-700 border border-emerald-200">
            💎 BUY ON DIP
          </span>
        )
      case 'thesis_reinforcer':
        return (
          <span className="shrink-0 px-2 py-0.5 rounded text-[11px] font-bold bg-sky-50 text-sky-700 border border-sky-200">
            ✅ THESIS REINFORCER
          </span>
        )
      case 'noise':
      default:
        return (
          <span className="shrink-0 px-2 py-0.5 rounded text-[11px] font-medium bg-zinc-100 text-zinc-500 border border-zinc-200">
            💤 NOISE
          </span>
        )
    }
  }

  // Render freshness badge
  const renderFreshnessBadge = (tier: string, label: string) => {
    if (tier === 'ultra_fresh') {
      return (
        <span className="rounded bg-emerald-50 px-2 py-0.5 text-[10px] font-bold text-emerald-700 border border-emerald-200">
          🟢 {label || '< 3h'}
        </span>
      )
    }
    if (tier === 'fresh_today') {
      return (
        <span className="rounded bg-sky-50 px-2 py-0.5 text-[10px] font-semibold text-sky-700 border border-sky-200">
          🔵 {label || 'วันนี้'}
        </span>
      )
    }
    return (
      <span className="rounded bg-zinc-100 px-2 py-0.5 text-[10px] text-zinc-600 border border-zinc-200">
        🟡 {label || 'ล่าสุด'}
      </span>
    )
  }

  return (
    <div className="mx-auto max-w-7xl space-y-6 p-4 sm:p-6 lg:p-8">
      {/* ------------------------------------------------------------------ */}
      {/* 1. Header Command Bar & View Switcher                              */}
      {/* ------------------------------------------------------------------ */}
      <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between border-b border-edge pb-5">
        <div>
          <div className="flex items-center gap-3">
            <h1 className="text-2xl font-bold tracking-tight text-zinc-900 flex items-center gap-2">
              <span>Daily Market Intelligence Digest</span>
              <span className="text-xl">📰</span>
            </h1>
            <span className="rounded-full bg-sky-100 px-2.5 py-0.5 text-xs font-semibold text-sky-800 border border-sky-200">
              One-Shot Mega-Batch (ประหยัด AI 90%+)
            </span>
          </div>
          <p className="mt-1 text-sm text-zinc-500">
            กลั่นกรองข่าวรอบโลก วิเคราะห์คลิปสัมมนา/YouTube และเชื่อมโยงส่งตรงถึงหุ้นในพอร์ตโฟลิโอของคุณในรอบเดียว
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-2.5">
          {/* Daily / Weekly View Switcher */}
          <div className="flex rounded-xl bg-surface-strong p-1 border border-edge">
            <button
              onClick={() => handleTabChange('daily')}
              className={`rounded-lg px-3.5 py-1.5 text-sm font-semibold transition-colors ${
                activeTab === 'daily'
                  ? 'bg-panel text-sky-700 shadow-sm'
                  : 'text-zinc-600 hover:text-zinc-900'
              }`}
            >
              ☀️ Daily Digest
            </button>
            <button
              onClick={() => handleTabChange('weekly')}
              className={`rounded-lg px-3.5 py-1.5 text-sm font-semibold transition-colors ${
                activeTab === 'weekly'
                  ? 'bg-panel text-sky-700 shadow-sm'
                  : 'text-zinc-600 hover:text-zinc-900'
              }`}
            >
              📅 Weekly Matrix
            </button>
          </div>

          {/* History Date Dropdown */}
          {activeTab === 'daily' && dailyData?.available_dates && dailyData.available_dates.length > 0 && (
            <div className="flex items-center gap-1.5 text-xs text-zinc-600 bg-surface-strong px-3 py-1.5 rounded-xl border border-edge">
              <span>📅 ประวัติ:</span>
              <select
                aria-label="เลือกวันที่สรุปข่าว"
                value={dailyData.digest?.date || ''}
                onChange={(e) => handleSelectDailyDate(e.target.value)}
                className="bg-transparent font-semibold text-zinc-800 focus:outline-none cursor-pointer"
              >
                {dailyData.available_dates.map((d) => (
                  <option key={d} value={d}>
                    {d} {d === dailyData.available_dates?.[0] ? '(ล่าสุด)' : ''}
                  </option>
                ))}
              </select>
            </div>
          )}

          {activeTab === 'weekly' && weeklyData?.available_dates && weeklyData.available_dates.length > 0 && (
            <div className="flex items-center gap-1.5 text-xs text-zinc-600 bg-surface-strong px-3 py-1.5 rounded-xl border border-edge">
              <span>📅 ประวัติ:</span>
              <select
                aria-label="เลือกสัปดาห์สรุปข่าว"
                value={weeklyData.digest?.to_date || weeklyData.digest?.digest_id || ''}
                onChange={(e) => handleSelectWeeklyDate(e.target.value)}
                className="bg-transparent font-semibold text-zinc-800 focus:outline-none cursor-pointer"
              >
                {weeklyData.available_dates.map((d) => (
                  <option key={d} value={d}>
                    {d} {d === weeklyData.available_dates?.[0] ? '(ล่าสุด)' : ''}
                  </option>
                ))}
              </select>
            </div>
          )}

          {/* Toggle Pre-LLM Selection Gate Button */}
          <button
            onClick={() => setShowSelectionGate((prev) => !prev)}
            className={`inline-flex items-center gap-1.5 rounded-xl border px-3.5 py-1.5 text-xs font-semibold transition-colors ${
              showSelectionGate
                ? 'border-sky-300 bg-sky-50 text-sky-700'
                : 'border-edge bg-panel text-zinc-700 hover:bg-surface-strong'
            }`}
          >
            <span>🛠️ เลือกข่าว/คลิป ({selectedVideoIds.size} คลิป)</span>
            <span className="text-[10px]">{showSelectionGate ? '▲ ซ่อน' : '▼ เปิด'}</span>
          </button>
        </div>
      </div>

      {/* Error Alert */}
      {error && (
        <div className="rounded-xl border border-rose-200 bg-rose-50 p-4 text-xs text-rose-700 flex items-center justify-between">
          <span>{error}</span>
          <button onClick={() => setError(null)} className="font-bold hover:underline">
            ✕ ปิด
          </button>
        </div>
      )}

      {/* Staleness Warning Banner */}
      {activeTab === 'daily' && dailyData?.has_digest && dailyData?.is_snapshot_stale && !synthesizing && (
        <div className="rounded-xl border border-amber-200 bg-amber-50/80 p-4 shadow-sm flex flex-col sm:flex-row sm:items-center justify-between gap-3">
          <div className="flex items-center gap-2.5 text-xs text-amber-800">
            <span className="text-base">⚠️</span>
            <div>
              <span className="font-bold">ข้อมูลสรุปนี้ถูกสร้างเมื่อ {dailyData.snapshot_age_hours} ชั่วโมงที่แล้ว (Snapshot เก่า)</span>
              <p className="text-[11px] text-amber-700 mt-0.5">
                มีข่าวสารและคลิปวิดีโอใหม่ที่รอให้คุณคัดกรอง {dailyData.fresh_pending_count} รายการ เพื่ออัปเดตมุมมองล่าสุด
              </p>
            </div>
          </div>
          <button
            onClick={() => setShowSelectionGate(true)}
            className="shrink-0 rounded-lg bg-amber-600 hover:bg-amber-700 text-white px-3 py-1.5 text-xs font-semibold transition-colors"
          >
            เปิดกล่องคัดกรองเพื่อสังเคราะห์ใหม่
          </button>
        </div>
      )}

      {/* ------------------------------------------------------------------ */}
      {/* 2. Interactive Pre-LLM Selection Gate (The Gate Before AI)         */}
      {/* ------------------------------------------------------------------ */}
      {showSelectionGate && (
        <div className="rounded-2xl border-2 border-sky-200 bg-sky-50/30 p-5 shadow-sm space-y-5 animate-fade-in">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-sky-200/60 pb-3">
            <div>
              <div className="flex items-center gap-2">
                <span className="text-base">🛡️</span>
                <h3 className="text-sm font-bold uppercase tracking-wider text-sky-950">
                  {activeTab === 'daily'
                    ? 'Interactive Pre-LLM Selection Gate (กล่องคัดกรองข่าวประจำวันก่อนส่ง AI)'
                    : 'Interactive Pre-LLM Selection Gate (กล่องคัดกรองข่าวยุทธศาสตร์รอบสัปดาห์ก่อนส่ง AI)'}
                </h3>
                <span className="rounded bg-sky-100 text-sky-800 text-[10px] font-bold px-2 py-0.5 border border-sky-200">
                  ZERO AI COST PREVIEW
                </span>
              </div>
              <p className="mt-1 text-xs text-zinc-600">
                {activeTab === 'daily'
                  ? 'ประหยัดค่าใช้จ่าย LLM: เลือกเฉพาะคลิป YouTube และหัวข้อข่าวสดรอบวัน (24-36 ชม.) AI จะรวบประมวลผลทั้งหมดใน 1 API Call เดียว'
                  : 'ประหยัดค่าใช้จ่าย LLM: คัดเลือกข่าวสารสำคัญรอบ 7 วันและคลิปสัมมนา AI จะสังเคราะห์เป็น Weekly Intelligence Matrix ใน 1 API Call เดียว'}
              </p>
            </div>

            <div className="flex items-center gap-2">
              <button
                onClick={handleRefreshCandidates}
                disabled={candidatesLoading}
                className="rounded-lg border border-sky-200 bg-panel px-3 py-1 text-xs font-semibold text-sky-700 hover:bg-sky-50 disabled:opacity-50"
              >
                {candidatesLoading ? 'กำลังดึงข้อมูล...' : '🔄 ตรวจสอบข้อมูลใหม่'}
              </button>
            </div>
          </div>

          {/* Two-Column Grid: YouTube vs News */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
            {/* Left: YouTube Video Selection */}
            <div className="rounded-xl border border-edge bg-panel p-4 space-y-3">
              <div className="flex flex-wrap items-center justify-between border-b border-edge pb-2 gap-2">
                <span className="text-xs font-bold text-zinc-900 flex items-center gap-1.5">
                  <span className="text-red-600">▶</span>
                  <span>เลือกคลิป YouTube ({selectedVideoIds.size}/{candidatesData?.youtube_candidates.length || 0})</span>
                </span>
                <div className="flex items-center gap-2">
                  <button
                    type="button"
                    onClick={handleSelectAllVideos}
                    className="text-[11px] text-sky-600 hover:text-sky-800 hover:underline font-medium"
                  >
                    เลือกทั้งหมด
                  </button>
                  <span className="text-zinc-300">|</span>
                  <button
                    type="button"
                    onClick={handleDeselectAllVideos}
                    className="text-[11px] text-zinc-500 hover:text-zinc-800 hover:underline font-medium"
                  >
                    ยกเลิกทั้งหมด
                  </button>
                </div>
              </div>

              {/* YouTube Candidate List */}
              <div className="space-y-2 max-h-72 overflow-y-auto pr-1 custom-scrollbar">
                {candidatesData?.youtube_candidates.map((yt) => {
                  const isChecked = selectedVideoIds.has(yt.video_id)
                  return (
                    <div
                      key={yt.video_id}
                      onClick={() => toggleVideoSelection(yt.video_id)}
                      className={`flex items-start gap-3 rounded-lg border p-2.5 cursor-pointer transition-colors ${
                        isChecked
                          ? 'border-sky-300 bg-sky-50/40'
                          : 'border-edge bg-panel hover:bg-surface-strong opacity-80'
                      }`}
                    >
                      <input
                        type="checkbox"
                        checked={isChecked}
                        onChange={() => {}} // Handled by div click
                        className="mt-1 h-4 w-4 rounded border-zinc-300 text-sky-600 focus:ring-sky-500"
                      />
                      <div className="space-y-1 flex-1 min-w-0">
                        <div className="flex items-center justify-between gap-2">
                          <span className="text-[11px] font-bold text-red-700 truncate">
                            {yt.channel}
                          </span>
                          <span className="text-[10px] text-zinc-400 shrink-0">
                            {yt.freshness_label || `${yt.age_hours}h ago`}
                          </span>
                        </div>
                        <p className="text-xs font-medium text-zinc-900 leading-snug line-clamp-2">
                          {yt.title}
                        </p>
                        {yt.cached_summary_snippet && (
                          <p className="text-[11px] text-zinc-500 line-clamp-1 italic">
                            💡 {yt.cached_summary_snippet}
                          </p>
                        )}
                      </div>
                    </div>
                  )
                })}

                {(!candidatesData?.youtube_candidates || candidatesData.youtube_candidates.length === 0) && (
                  <p className="text-xs text-zinc-400 text-center py-6">
                    {candidatesLoading ? 'กำลังดึงคลิป YouTube...' : 'ไม่พบคลิป YouTube ใหม่ในรอบ 7 วัน'}
                  </p>
                )}
              </div>

              {/* Custom YouTube URL Adder */}
              <div className="pt-2 border-t border-edge space-y-2">
                <span className="text-[11px] font-semibold text-zinc-700">
                  + เพิ่มคลิป YouTube URL ด้วยตนเอง
                </span>
                <div className="flex gap-2">
                  <input
                    type="text"
                    value={customYoutubeUrl}
                    onChange={(e) => setCustomYoutubeUrl(e.target.value)}
                    placeholder="https://www.youtube.com/watch?v=..."
                    className="flex-1 rounded-lg border border-edge bg-surface px-2.5 py-1 text-xs text-zinc-800 placeholder-zinc-400 focus:outline-none focus:border-sky-400"
                  />
                  <button
                    onClick={handleAddCustomYoutube}
                    className="rounded-lg bg-zinc-800 hover:bg-zinc-900 text-white px-3 py-1 text-xs font-semibold transition-colors"
                  >
                    เพิ่ม
                  </button>
                </div>

                {customYoutubeUrls.length > 0 && (
                  <div className="space-y-1">
                    {customYoutubeUrls.map((url, idx) => (
                      <div key={idx} className="flex items-center justify-between rounded bg-surface-strong px-2 py-1 text-[11px]">
                        <span className="truncate flex-1 text-zinc-700">{url}</span>
                        <button onClick={() => handleRemoveCustomYoutube(idx)} className="text-rose-600 hover:underline ml-2">
                          ลบ
                        </button>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>

            {/* Right: Fresh News Candidate List */}
            <div className="rounded-xl border border-edge bg-panel p-4 space-y-3">
              <div className="flex flex-wrap items-center justify-between border-b border-edge pb-2 gap-2">
                <span className="text-xs font-bold text-zinc-900 flex items-center gap-1.5">
                  <span>📰</span>
                  <span>
                    {activeTab === 'daily'
                      ? `หัวข้อข่าวสดรอบวัน (${(candidatesData?.news_candidates.length || 0) - excludedNewsIds.size}/${candidatesData?.news_candidates.length || 0})`
                      : `หัวข้อข่าวสำคัญรอบสัปดาห์ (${(candidatesData?.news_candidates.length || 0) - excludedNewsIds.size}/${candidatesData?.news_candidates.length || 0})`}
                  </span>
                </span>
                <div className="flex items-center gap-2">
                  <button
                    type="button"
                    onClick={handleSelectOnlyPortfolioNews}
                    className="text-[11px] text-sky-700 bg-sky-100 hover:bg-sky-200 px-2 py-0.5 rounded font-semibold transition-colors"
                    title="เลือกเฉพาะข่าวที่มีหุ้นหรือกองทุนในพอร์ตเกี่ยวข้อง"
                  >
                    🎯 เฉพาะกระทบพอร์ต
                  </button>
                  <button
                    type="button"
                    onClick={handleSelectAllNews}
                    className="text-[11px] text-sky-600 hover:text-sky-800 hover:underline font-medium"
                  >
                    เลือกทั้งหมด
                  </button>
                  <span className="text-zinc-300">|</span>
                  <button
                    type="button"
                    onClick={handleDeselectAllNews}
                    className="text-[11px] text-zinc-500 hover:text-zinc-800 hover:underline font-medium"
                  >
                    ยกเลิกทั้งหมด
                  </button>
                </div>
              </div>

              {/* News Candidate List */}
              <div className="space-y-2 max-h-72 overflow-y-auto pr-1 custom-scrollbar">
                {candidatesData?.news_candidates.map((cand) => {
                  const isExcluded = excludedNewsIds.has(cand.candidate_id)
                  return (
                    <div
                      key={cand.candidate_id}
                      onClick={() => toggleNewsExclusion(cand.candidate_id)}
                      className={`flex items-start gap-3 rounded-lg border p-2.5 cursor-pointer transition-colors ${
                        isExcluded
                          ? 'border-zinc-200 bg-zinc-50 opacity-50'
                          : 'border-edge bg-panel hover:bg-surface-strong'
                      }`}
                    >
                      <input
                        type="checkbox"
                        checked={!isExcluded}
                        onChange={() => {}} // Handled by div click
                        className="mt-1 h-4 w-4 rounded border-zinc-300 text-sky-600 focus:ring-sky-500"
                      />
                      <div className="space-y-1 flex-1 min-w-0">
                        <div className="flex items-center justify-between gap-2">
                          <span className="text-[11px] font-semibold text-zinc-600">
                            {cand.source} {cand.sources_count > 1 && `(${cand.sources_count} แหล่ง)`}
                          </span>
                          {renderFreshnessBadge(cand.freshness_tier, cand.freshness_label)}
                        </div>
                        <p className={`text-xs font-medium text-zinc-900 leading-snug line-clamp-2 ${isExcluded ? 'line-through text-zinc-400' : ''}`}>
                          {cand.title}
                        </p>
                        {cand.portfolio_badges && cand.portfolio_badges.length > 0 && (
                          <div className="flex flex-wrap gap-1 pt-0.5">
                            {cand.portfolio_badges.map((b) => (
                              <span
                                key={b.symbol}
                                className="rounded bg-sky-50 text-sky-700 px-1.5 py-0.2 text-[10px] font-bold border border-sky-100"
                              >
                                {b.symbol} ({b.impact_type === 'direct_holding' ? 'พอร์ตตรง' : b.impact_type === 'fund_lookthrough' ? `กองทุน (${b.source_fund_codes && b.source_fund_codes.length > 0 ? b.source_fund_codes.join(', ') : 'ไส้ใน'})` : 'Watchlist'})
                              </span>
                            ))}
                          </div>
                        )}
                      </div>
                    </div>
                  )
                })}

                {(!candidatesData?.news_candidates || candidatesData.news_candidates.length === 0) && (
                  <p className="text-xs text-zinc-400 text-center py-6">
                    {candidatesLoading
                      ? 'กำลังดึงรายการข่าวสาร...'
                      : activeTab === 'daily'
                      ? 'ไม่พบข่าวสารใหม่ในรอบ 36 ชั่วโมง'
                      : 'ไม่พบข่าวสารในรอบ 7 วัน'}
                  </p>
                )}
              </div>

              {/* Custom News Input */}
              <div className="pt-2 border-t border-edge space-y-2">
                <span className="text-[11px] font-semibold text-zinc-700">
                  + เพิ่มประเด็นข่าว/หัวข้อด่วนด้วยตนเอง
                </span>
                <div className="flex gap-2">
                  <input
                    type="text"
                    value={customNewsInput}
                    onChange={(e) => setCustomNewsInput(e.target.value)}
                    placeholder="เช่น เฟดประกาศลดดอกเบี้ย 0.25% คืนนี้..."
                    className="flex-1 rounded-lg border border-edge bg-surface px-2.5 py-1 text-xs text-zinc-800 placeholder-zinc-400 focus:outline-none focus:border-sky-400"
                  />
                  <button
                    onClick={handleAddCustomNews}
                    className="rounded-lg bg-zinc-800 hover:bg-zinc-900 text-white px-3 py-1 text-xs font-semibold transition-colors"
                  >
                    เพิ่ม
                  </button>
                </div>

                {customNewsItems.length > 0 && (
                  <div className="space-y-1">
                    {customNewsItems.map((item, idx) => (
                      <div key={idx} className="flex items-center justify-between rounded bg-surface-strong px-2 py-1 text-[11px]">
                        <span className="truncate flex-1 text-zinc-700">{item}</span>
                        <button onClick={() => handleRemoveCustomNews(idx)} className="text-rose-600 hover:underline ml-2">
                          ลบ
                        </button>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Action Trigger Button */}
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pt-3 border-t border-sky-200/60">
            <div className="text-xs text-sky-900">
              พร้อมสังเคราะห์: <strong>{selectedVideoIds.size + customYoutubeUrls.length} คลิป YouTube</strong> และ{' '}
              <strong>{(candidatesData?.news_candidates.length || 0) - excludedNewsIds.size + customNewsItems.length} หัวข้อข่าว</strong>
            </div>

            <button
              onClick={activeTab === 'daily' ? handleSynthesizeDaily : handleSynthesizeWeekly}
              disabled={synthesizing}
              className="inline-flex items-center justify-center gap-2 rounded-xl bg-gradient-to-r from-sky-600 to-flow-blue px-6 py-2.5 text-sm font-bold text-white shadow-md shadow-sky-500/20 hover:from-sky-700 hover:to-sky-800 transition-all disabled:opacity-50"
            >
              {synthesizing ? (
                <>
                  <svg className="animate-spin h-4 w-4 text-white" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                  </svg>
                  <span>
                    {activeTab === 'daily'
                      ? 'กำลังสังเคราะห์รอบเดียว (One-Shot Mega-Batch)...'
                      : 'กำลังสังเคราะห์ Weekly Matrix (One-Shot)...'}
                  </span>
                </>
              ) : (
                <>
                  <span>
                    {activeTab === 'daily'
                      ? '⚡ ให้ AI สังเคราะห์ Daily Digest (One-Shot Mega-Batch)'
                      : '⚡ ให้ AI สังเคราะห์ Weekly Intelligence Matrix'}
                  </span>
                </>
              )}
            </button>
          </div>
        </div>
      )}

      {/* ------------------------------------------------------------------ */}
      {/* 3. Daily Digest View                                               */}
      {/* ------------------------------------------------------------------ */}
      {activeTab === 'daily' && (
        <div className="space-y-6">
          {dailyData?.has_digest && dailyData.digest ? (
            <>
              {/* Market Pulse Hero Bar */}
              <div className="rounded-2xl border border-edge bg-panel p-5 shadow-sm space-y-4">
                <div className="flex flex-wrap items-center justify-between gap-2 border-b border-edge pb-3">
                  <div className="flex items-center gap-2.5">
                    <span className="text-xs font-bold uppercase tracking-wider text-zinc-500">
                      30-Sec Market Pulse
                    </span>
                    <span className="text-zinc-300">•</span>
                    <span className="text-xs font-semibold text-zinc-700">
                      รอบประจำวันที่ {dailyData.digest.date} ({dailyData.digest.period})
                    </span>
                  </div>

                  {/* Market Mood Badge */}
                  <div className="flex items-center gap-2">
                    {dailyData.digest.market_mood === 'risk_on' && (
                      <span className="rounded-full bg-emerald-50 px-3 py-1 text-xs font-bold text-emerald-700 border border-emerald-200">
                        🟢 RISK-ON (เปิดรับความเสี่ยง)
                      </span>
                    )}
                    {dailyData.digest.market_mood === 'risk_off' && (
                      <span className="rounded-full bg-rose-50 px-3 py-1 text-xs font-bold text-rose-700 border border-rose-200">
                        🔴 RISK-OFF (ระมัดระวังความเสี่ยง)
                      </span>
                    )}
                    {dailyData.digest.market_mood === 'mixed' && (
                      <span className="rounded-full bg-amber-50 px-3 py-1 text-xs font-bold text-amber-700 border border-amber-200">
                        🟡 MIXED / NEUTRAL (ตลาดผสมผสาน)
                      </span>
                    )}
                  </div>
                </div>

                {/* Mood Summary & Transmission Chain */}
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div className="md:col-span-2 space-y-2">
                    <p className="text-sm font-medium text-zinc-900 leading-relaxed">
                      {dailyData.digest.market_mood_summary}
                    </p>
                    <div className="rounded-lg bg-surface-strong p-3 text-xs text-zinc-700 border border-edge">
                      <span className="font-bold text-zinc-900">🔗 สายส่งผลกระทบ (Transmission Chain): </span>
                      {dailyData.digest.transmission_chain}
                    </div>
                  </div>

                  <div className="rounded-xl border border-sky-100 bg-sky-50/50 p-3.5 space-y-1.5 flex flex-col justify-center">
                    <span className="text-[11px] font-bold uppercase tracking-wider text-sky-800">
                      🎯 โฟกัสหลักในพอร์ตวันนี้
                    </span>
                    <p className="text-xs text-sky-950 font-medium leading-relaxed">
                      {dailyData.digest.top_portfolio_focus}
                    </p>
                  </div>
                </div>
              </div>

              {/* News Items Section */}
              <div className="space-y-4">
                {/* Filter and Action Bar */}
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-edge pb-3">
                  <div className="flex items-center gap-1.5">
                    <button
                      onClick={() => setNewsFilter('all')}
                      className={`rounded-lg px-3 py-1 text-xs font-semibold transition-colors ${
                        newsFilter === 'all'
                          ? 'bg-zinc-800 text-white'
                          : 'bg-surface text-zinc-600 hover:bg-surface-strong'
                      }`}
                    >
                      ทั้งหมด ({dailyData.digest.items.length})
                    </button>
                    <button
                      onClick={() => setNewsFilter('portfolio')}
                      className={`rounded-lg px-3 py-1 text-xs font-semibold transition-colors ${
                        newsFilter === 'portfolio'
                          ? 'bg-sky-600 text-white'
                          : 'bg-surface text-zinc-600 hover:bg-surface-strong'
                      }`}
                    >
                      เฉพาะพอร์ตของเรา ⭐ ({portfolioItemsCount})
                    </button>
                    <button
                      onClick={() => setNewsFilter('unread')}
                      className={`rounded-lg px-3 py-1 text-xs font-semibold transition-colors ${
                        newsFilter === 'unread'
                          ? 'bg-flow-blue text-white'
                          : 'bg-surface text-zinc-600 hover:bg-surface-strong'
                      }`}
                    >
                      ยังไม่ได้อ่าน 🔵 ({unreadItemsCount})
                    </button>
                  </div>

                  <div className="flex items-center gap-2">
                    <button
                      onClick={handleMarkAllRead}
                      className="text-xs font-medium text-sky-600 hover:text-sky-700 hover:underline"
                    >
                      ✓ ทำเครื่องหมายอ่านแล้วทั้งหมด
                    </button>
                  </div>
                </div>

                {/* News Cards List */}
                <div className="grid grid-cols-1 gap-4">
                  {filteredDailyItems.map((item, idx) => {
                    return (
                      <div
                        key={item.source_candidate_id || item.link || idx}
                        className={`rounded-2xl border bg-panel p-5 shadow-sm transition-all ${
                          item.is_read
                            ? 'border-edge opacity-80'
                            : 'border-sky-200/80 shadow-[0_4px_20px_rgba(14,165,233,0.04)]'
                        }`}
                      >
                        <div className="space-y-3">
                          {/* Card Header: Badges & Read Toggle */}
                          <div className="flex items-start justify-between gap-3">
                            <div className="flex flex-wrap items-center gap-2">
                              {renderThesisBadge(item.thesis_signal)}
                              {renderFreshnessBadge(item.freshness_tier, item.freshness_label)}
                              {item.is_new && (
                                <span className="rounded bg-sky-500 text-white px-1.5 py-0.2 text-[10px] font-bold">
                                  NEW
                                </span>
                              )}
                            </div>

                            <button
                              onClick={() => handleToggleRead(item)}
                              title={item.is_read ? 'ทำเครื่องหมายว่ายังไม่ได้อ่าน' : 'ทำเครื่องหมายว่าอ่านแล้ว'}
                              className="text-xs font-semibold text-zinc-400 hover:text-sky-600 shrink-0 flex items-center gap-1"
                            >
                              {item.is_read ? (
                                <span className="text-zinc-400">✓ อ่านแล้ว</span>
                              ) : (
                                <span className="h-2.5 w-2.5 rounded-full bg-flow-blue" />
                              )}
                            </button>
                          </div>

                          {/* Thai Headline */}
                          <h3 className="text-base font-bold text-zinc-900 leading-snug">
                            {item.thai_headline}
                          </h3>

                          {/* What Happened */}
                          <div className="text-xs text-zinc-700 leading-relaxed">
                            <span className="font-semibold text-zinc-900">เกิดอะไรขึ้น: </span>
                            {item.what_happened}
                          </div>

                          {/* Why it Matters (The Crucial Alpha) */}
                          <div className="rounded-xl bg-sky-50/70 border border-sky-100 p-3 text-xs text-sky-950 leading-relaxed space-y-1">
                            <span className="font-bold text-sky-900">
                              💡 ทำไมจึงสำคัญต่อพอร์ต & วิทยานิพนธ์ (Why It Matters):
                            </span>
                            <p>{item.why_it_matters}</p>
                          </div>

                          {/* YouTube Corroboration if present */}
                          {item.youtube_corroboration && (
                            <div className="rounded-lg bg-red-50/60 border border-red-100 p-2.5 text-xs text-red-950 flex items-start gap-2">
                              <span className="text-red-600 font-bold">▶ YouTube:</span>
                              <span>{item.youtube_corroboration}</span>
                            </div>
                          )}

                          {/* Card Footer: Portfolio badges, Source link, and Journal button */}
                          <div className="flex flex-wrap items-center justify-between gap-3 pt-2 border-t border-edge">
                            <div className="flex flex-wrap items-center gap-1.5">
                              {item.affected_tickers.map((sym) => (
                                <Link
                                  key={sym}
                                  to={`/equity/${sym}`}
                                  className="rounded-md bg-surface-strong px-2 py-0.5 text-xs font-bold text-sky-700 hover:bg-sky-100 transition-colors"
                                >
                                  ${sym} ↗
                                </Link>
                              ))}

                              {item.portfolio_badges?.map((badge) => (
                                <span
                                  key={badge.symbol}
                                  className="rounded bg-sky-100/70 text-sky-800 text-[11px] font-semibold px-2 py-0.5 border border-sky-200"
                                >
                                  {badge.impact_type === 'direct_holding' ? 'พอร์ตตรง' : badge.impact_type === 'fund_lookthrough' ? `กองทุน (${badge.source_fund_codes && badge.source_fund_codes.length > 0 ? badge.source_fund_codes.join(', ') : 'ไส้ใน'})` : 'Watchlist'}{' '}
                                  {badge.weight_pct ? `(${badge.weight_pct}%)` : ''}
                                </span>
                              ))}

                              <span className="text-xs text-zinc-400 ml-1">
                                ที่มา: {item.source}
                              </span>
                            </div>

                            <div className="flex items-center gap-3">
                              <button
                                onClick={() => handleOpenJournalModal(item)}
                                className="inline-flex items-center gap-1 text-xs font-semibold text-zinc-600 hover:text-sky-700 hover:underline"
                              >
                                ✍️ บันทึกลง Journal
                              </button>

                              {item.link && (
                                <a
                                  href={item.link}
                                  target="_blank"
                                  rel="noopener noreferrer"
                                  className="inline-flex items-center gap-1 text-xs font-semibold text-sky-600 hover:text-sky-700 hover:underline"
                                >
                                  อ่านต้นฉบับ ↗
                                </a>
                              )}
                            </div>
                          </div>
                        </div>
                      </div>
                    )
                  })}

                  {filteredDailyItems.length === 0 && (
                    <div className="rounded-xl border border-edge bg-panel p-8 text-center text-xs text-zinc-400">
                      ไม่พบข่าวสารในหมวดหมู่นี้
                    </div>
                  )}
                </div>
              </div>

              {/* YouTube Highlights Section */}
              {dailyData.digest.youtube_highlights && dailyData.digest.youtube_highlights.length > 0 && (
                <div className="rounded-2xl border border-red-100 bg-red-50/20 p-5 space-y-4">
                  <div className="flex items-center justify-between border-b border-red-200/60 pb-3">
                    <h3 className="text-xs font-bold uppercase tracking-wider text-red-900 flex items-center gap-2">
                      <span className="text-red-600 text-sm">▶</span>
                      <span>YouTube Channel Highlights ({dailyData.digest.youtube_highlights.length} รายการที่วิเคราะห์)</span>
                    </h3>
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {dailyData.digest.youtube_highlights.map((yt, idx) => (
                      <div key={idx} className="rounded-xl border border-edge bg-panel p-4 space-y-2 shadow-2xs">
                        <div className="flex items-start justify-between gap-2">
                          <span className="text-[11px] font-bold text-red-700">
                            {yt.channel}
                          </span>
                          {yt.alignment_with_news === 'confirmed' && (
                            <span className="rounded bg-emerald-50 text-emerald-700 px-2 py-0.5 text-[10px] font-bold border border-emerald-200">
                              สอดคล้องกับข่าว
                            </span>
                          )}
                          {yt.alignment_with_news === 'contrarian' && (
                            <span className="rounded bg-amber-50 text-amber-700 px-2 py-0.5 text-[10px] font-bold border border-amber-200">
                              มุมมองสวนทาง (Contrarian)
                            </span>
                          )}
                        </div>
                        <h4 className="text-xs font-bold text-zinc-900 leading-snug">
                          {yt.thai_title}
                        </h4>
                        <ul className="list-disc pl-4 space-y-1 text-xs text-zinc-700 leading-relaxed">
                          {yt.key_takeaways.map((point, pIdx) => (
                            <li key={pIdx}>{point}</li>
                          ))}
                        </ul>
                        {yt.alignment_note && (
                          <p className="text-[11px] text-zinc-500 italic pt-1">
                            หมายเหตุ: {yt.alignment_note}
                          </p>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </>
          ) : loading ? (
            <div className="rounded-2xl border border-edge bg-panel p-12 text-center space-y-3">
              <div className="inline-block animate-spin h-6 w-6 border-2 border-sky-600 border-t-transparent rounded-full" />
              <p className="text-xs text-zinc-500">กำลังโหลดข้อมูล Daily Digest...</p>
            </div>
          ) : (
            /* No Daily Digest Yet Placeholder */
            <div className="rounded-2xl border border-dashed border-edge bg-surface p-12 text-center space-y-3">
              <span className="text-3xl">☕</span>
              <h3 className="text-base font-semibold text-zinc-800">
                ยังไม่มีการสังเคราะห์ Daily Digest ล่าสุด
              </h3>
              <p className="text-xs text-zinc-500 max-w-md mx-auto">
                เริ่มต้นวันของคุณโดยกดปุ่มด้านล่างเพื่อเลือกคลิป YouTube และให้ AI สังเคราะห์ข่าวสารรอบโลกใน 1 รอบเดียว
              </p>
              <button
                onClick={() => setShowSelectionGate(true)}
                className="rounded-xl bg-sky-600 hover:bg-sky-700 text-white px-5 py-2 text-xs font-semibold transition-colors shadow-sm"
              >
                🛠️ เปิดกล่องคัดกรองเพื่อสังเคราะห์รอบแรก
              </button>
            </div>
          )}
        </div>
      )}

      {/* ------------------------------------------------------------------ */}
      {/* 4. Weekly Intelligence Matrix View                                 */}
      {/* ------------------------------------------------------------------ */}
      {activeTab === 'weekly' && (
        <div className="space-y-6">
          {weeklyData?.has_digest && weeklyData.digest ? (
            <div className="space-y-6">
              {/* Executive Themes */}
              <div className="rounded-2xl border border-edge bg-panel p-5 shadow-sm space-y-3">
                <div className="flex items-center justify-between border-b border-edge pb-2">
                  <h3 className="text-xs font-bold uppercase tracking-wider text-sky-800">
                    🌍 Executive Macro & Sector Themes (ประเด็นมหภาครายสัปดาห์)
                  </h3>
                  <span className="text-xs text-zinc-400">
                    {weeklyData.digest.from_date} ถึง {weeklyData.digest.to_date}
                  </span>
                </div>
                <ul className="list-disc pl-5 space-y-1.5 text-xs text-zinc-700 leading-relaxed">
                  {weeklyData.digest.executive_themes.map((theme, idx) => (
                    <li key={idx} className="font-medium">
                      {theme}
                    </li>
                  ))}
                </ul>
              </div>

              {/* Portfolio Health Summary & Thesis Matrix */}
              <div className="rounded-2xl border border-edge bg-panel p-5 shadow-sm space-y-4">
                <div className="border-b border-edge pb-2">
                  <h3 className="text-xs font-bold uppercase tracking-wider text-sky-800">
                    📊 Portfolio Health & Thesis Matrix (สถานะหุ้นในพอร์ตรายตัว)
                  </h3>
                  <p className="text-xs text-zinc-600 mt-1">
                    {weeklyData.digest.portfolio_health_summary}
                  </p>
                </div>

                <div className="overflow-x-auto">
                  <table className="w-full text-left text-xs">
                    <thead>
                      <tr className="border-b border-edge bg-surface text-zinc-500">
                        <th className="py-2.5 px-3 font-semibold">Ticker</th>
                        <th className="py-2.5 px-3 font-semibold">Thesis Status</th>
                        <th className="py-2.5 px-3 font-semibold">Weekly Summary & Action Signal</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-edge">
                      {weeklyData.digest.portfolio_matrix.map((row) => (
                        <tr key={row.ticker} className="hover:bg-surface/50 transition-colors">
                          <td className="py-2.5 px-3 font-bold text-sky-700">
                            <Link to={`/equity/${row.ticker}`} className="hover:underline">
                              ${row.ticker}
                            </Link>
                          </td>
                          <td className="py-2.5 px-3">
                            {row.thesis_status === 'strong_hold' && (
                              <span className="rounded bg-emerald-50 text-emerald-700 px-2 py-0.5 font-bold border border-emerald-200">
                                {row.thesis_status_label || 'Strong Hold'}
                              </span>
                            )}
                            {row.thesis_status === 'watch_closely' && (
                              <span className="rounded bg-amber-50 text-amber-700 px-2 py-0.5 font-bold border border-amber-200">
                                {row.thesis_status_label || 'Watch Closely'}
                              </span>
                            )}
                            {row.thesis_status === 'accumulate_zone' && (
                              <span className="rounded bg-sky-50 text-sky-700 px-2 py-0.5 font-bold border border-sky-200">
                                {row.thesis_status_label || 'Accumulate Zone'}
                              </span>
                            )}
                            {row.thesis_status === 'review_thesis' && (
                              <span className="rounded bg-rose-50 text-rose-700 px-2 py-0.5 font-bold border border-rose-200">
                                {row.thesis_status_label || 'Review Thesis'}
                              </span>
                            )}
                          </td>
                          <td className="py-2.5 px-3 text-zinc-700 leading-relaxed">
                            {row.weekly_summary}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>

              {/* Narratives: Consensus vs Contrarian */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
                <div className="rounded-xl border border-edge bg-panel p-4 space-y-2">
                  <h4 className="text-xs font-bold uppercase tracking-wider text-zinc-900">
                    🏛️ ตลาดมองอย่างไร (Consensus Narratives)
                  </h4>
                  <ul className="list-disc pl-4 space-y-1 text-xs text-zinc-700 leading-relaxed">
                    {weeklyData.digest.consensus_narratives.map((c, i) => (
                      <li key={i}>{c}</li>
                    ))}
                  </ul>
                </div>

                <div className="rounded-xl border border-edge bg-panel p-4 space-y-2">
                  <h4 className="text-xs font-bold uppercase tracking-wider text-amber-800">
                    💡 มุมมองสวนทางและโอกาส (Contrarian & Pitches)
                  </h4>
                  <ul className="list-disc pl-4 space-y-1 text-xs text-zinc-700 leading-relaxed">
                    {weeklyData.digest.contrarian_and_pitches.map((c, i) => (
                      <li key={i}>{c}</li>
                    ))}
                  </ul>
                </div>
              </div>

              {/* Week Ahead Checklist */}
              {weeklyData.digest.week_ahead_checklist && (
                <div className="rounded-xl border border-sky-100 bg-sky-50/40 p-4 space-y-2">
                  <h4 className="text-xs font-bold uppercase tracking-wider text-sky-900">
                    🗓️ เช็กลิสต์สัปดาห์หน้า (Week-Ahead Checklist)
                  </h4>
                  <ul className="list-disc pl-5 space-y-1 text-xs text-sky-950 leading-relaxed">
                    {weeklyData.digest.week_ahead_checklist.map((item, idx) => (
                      <li key={idx}>{item}</li>
                    ))}
                  </ul>
                </div>
              )}

              {/* Weekly Strategic News & Key Developments */}
              {weeklyData.digest.items && weeklyData.digest.items.length > 0 && (
                <div className="space-y-4">
                  <div className="flex items-center justify-between border-b border-edge pb-2">
                    <h3 className="text-xs font-bold uppercase tracking-wider text-sky-800 flex items-center gap-2">
                      <span>📰 ข่าวสารยุทธศาสตร์สำคัญรอบสัปดาห์ (Weekly Key Developments & News)</span>
                      <span className="rounded bg-sky-100 text-sky-800 text-[10px] font-bold px-2 py-0.5">
                        {weeklyData.digest.items.length} ข่าว
                      </span>
                    </h3>
                  </div>

                  <div className="grid grid-cols-1 gap-4">
                    {weeklyData.digest.items.map((item, idx) => (
                      <div
                        key={item.source_candidate_id || item.link || idx}
                        className={`rounded-2xl border bg-panel p-5 shadow-sm transition-all ${
                          item.is_read
                            ? 'border-edge opacity-80'
                            : 'border-sky-200/80 shadow-[0_4px_20px_rgba(14,165,233,0.04)]'
                        }`}
                      >
                        <div className="space-y-3">
                          {/* Card Header: Badges & Read Toggle */}
                          <div className="flex items-start justify-between gap-3">
                            <div className="flex flex-wrap items-center gap-2">
                              {renderThesisBadge(item.thesis_signal)}
                              {renderFreshnessBadge(item.freshness_tier, item.freshness_label)}
                              {item.is_new && (
                                <span className="rounded bg-sky-500 text-white px-1.5 py-0.2 text-[10px] font-bold">
                                  NEW
                                </span>
                              )}
                            </div>

                            <button
                              onClick={() => handleToggleRead(item)}
                              title={item.is_read ? 'ทำเครื่องหมายว่ายังไม่ได้อ่าน' : 'ทำเครื่องหมายว่าอ่านแล้ว'}
                              className="text-xs font-semibold text-zinc-400 hover:text-sky-600 shrink-0 flex items-center gap-1"
                            >
                              {item.is_read ? (
                                <span className="text-zinc-400">✓ อ่านแล้ว</span>
                              ) : (
                                <span className="h-2.5 w-2.5 rounded-full bg-flow-blue" />
                              )}
                            </button>
                          </div>

                          {/* Thai Headline */}
                          <h3 className="text-base font-bold text-zinc-900 leading-snug">
                            {item.thai_headline}
                          </h3>

                          {/* What Happened */}
                          <div className="text-xs text-zinc-700 leading-relaxed">
                            <span className="font-semibold text-zinc-900">เกิดอะไรขึ้น: </span>
                            {item.what_happened}
                          </div>

                          {/* Why it Matters (The Crucial Alpha) */}
                          <div className="rounded-xl bg-sky-50/70 border border-sky-100 p-3 text-xs text-sky-950 leading-relaxed space-y-1">
                            <span className="font-bold text-sky-900">
                              💡 ทำไมจึงสำคัญต่อพอร์ต & วิทยานิพนธ์ (Why It Matters):
                            </span>
                            <p>{item.why_it_matters}</p>
                          </div>

                          {/* YouTube Corroboration if present */}
                          {item.youtube_corroboration && (
                            <div className="rounded-lg bg-red-50/60 border border-red-100 p-2.5 text-xs text-red-950 flex items-start gap-2">
                              <span className="text-red-600 font-bold">▶ YouTube:</span>
                              <span>{item.youtube_corroboration}</span>
                            </div>
                          )}

                          {/* Card Footer: Portfolio badges, Source link, and Journal button */}
                          <div className="flex flex-wrap items-center justify-between gap-3 pt-2 border-t border-edge">
                            <div className="flex flex-wrap items-center gap-1.5">
                              {item.affected_tickers.map((sym) => (
                                <Link
                                  key={sym}
                                  to={`/equity/${sym}`}
                                  className="rounded-md bg-surface-strong px-2 py-0.5 text-xs font-bold text-sky-700 hover:bg-sky-100 transition-colors"
                                >
                                  ${sym} ↗
                                </Link>
                              ))}

                              {item.portfolio_badges?.map((badge) => (
                                <span
                                  key={badge.symbol}
                                  className="rounded bg-sky-100/70 text-sky-800 text-[11px] font-semibold px-2 py-0.5 border border-sky-200"
                                >
                                  {badge.impact_type === 'direct_holding' ? 'พอร์ตตรง' : badge.impact_type === 'fund_lookthrough' ? `กองทุน (${badge.source_fund_codes && badge.source_fund_codes.length > 0 ? badge.source_fund_codes.join(', ') : 'ไส้ใน'})` : 'Watchlist'}{' '}
                                  {badge.weight_pct ? `(${badge.weight_pct}%)` : ''}
                                </span>
                              ))}

                              <span className="text-xs text-zinc-400 ml-1">
                                ที่มา: {item.source}
                              </span>
                            </div>

                            <div className="flex items-center gap-3">
                              <button
                                onClick={() => handleOpenJournalModal(item)}
                                className="inline-flex items-center gap-1 text-xs font-semibold text-zinc-600 hover:text-sky-700 hover:underline"
                              >
                                ✍️ บันทึกลง Journal
                              </button>

                              {item.link && (
                                <a
                                  href={item.link}
                                  target="_blank"
                                  rel="noopener noreferrer"
                                  className="inline-flex items-center gap-1 text-xs font-semibold text-sky-600 hover:text-sky-700 hover:underline"
                                >
                                  อ่านต้นฉบับ ↗
                                </a>
                              )}
                            </div>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          ) : (
            /* No Weekly Digest Yet Placeholder */
            <div className="rounded-2xl border border-dashed border-edge bg-surface p-12 text-center space-y-3">
              <span className="text-3xl">🗓️</span>
              <h3 className="text-base font-semibold text-zinc-800">
                ยังไม่มีการสังเคราะห์ Weekly Digest
              </h3>
              <p className="text-xs text-zinc-500 max-w-md mx-auto">
                ระบบจะรวบรวมข้อมูลตลอด 7 วันที่ผ่านมา และประเมินสถานะวิทยานิพนธ์ของหุ้นทุกตัวในพอร์ตให้โดยอัตโนมัติ
              </p>
              <button
                onClick={handleSynthesizeWeekly}
                disabled={synthesizing}
                className="rounded-xl bg-sky-600 hover:bg-sky-700 text-white px-5 py-2 text-xs font-semibold transition-colors shadow-sm disabled:opacity-50"
              >
                {synthesizing ? 'กำลังสังเคราะห์...' : '⚡ สังเคราะห์ Weekly Intelligence Matrix'}
              </button>
            </div>
          )}
        </div>
      )}

      {/* ------------------------------------------------------------------ */}
      {/* 5. Investment Journal Modal                                        */}
      {/* ------------------------------------------------------------------ */}
      {journalItem && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 backdrop-blur-xs p-4">
          <div className="rounded-2xl border border-edge bg-panel w-full max-w-2xl p-6 shadow-2xl space-y-4 animate-scale-up">
            <div className="flex items-center justify-between border-b border-edge pb-3">
              <h3 className="text-sm font-bold text-zinc-900 flex items-center gap-2">
                <span>✍️ บันทึกมุมมองลงใน Investment Journal</span>
              </h3>
              <button
                onClick={() => setJournalItem(null)}
                className="text-zinc-400 hover:text-zinc-600 text-sm font-bold"
              >
                ✕
              </button>
            </div>

            {journalSuccess ? (
              <div className="rounded-xl bg-emerald-50 border border-emerald-200 p-4 text-center text-xs font-bold text-emerald-700">
                ✓ {journalSuccess}
              </div>
            ) : (
              <>
                <p className="text-xs text-zinc-500">
                  คุณสามารถแก้ไขข้อความ Markdown ด้านล่างก่อนที่จะบันทึกลงสมุดบันทึกการลงทุน (Journal):
                </p>

                <textarea
                  value={journalContent}
                  onChange={(e) => setJournalContent(e.target.value)}
                  rows={10}
                  className="w-full rounded-xl border border-edge bg-surface p-3 text-xs font-mono text-zinc-800 focus:outline-none focus:border-sky-400"
                />

                <div className="flex items-center justify-end gap-2 pt-2">
                  <button
                    onClick={() => setJournalItem(null)}
                    className="rounded-lg border border-edge bg-surface px-4 py-2 text-xs font-semibold text-zinc-600 hover:bg-surface-strong"
                  >
                    ยกเลิก
                  </button>
                  <button
                    onClick={handleSaveJournal}
                    disabled={journalSaving}
                    className="rounded-lg bg-sky-600 hover:bg-sky-700 text-white px-5 py-2 text-xs font-semibold transition-colors disabled:opacity-50"
                  >
                    {journalSaving ? 'กำลังบันทึก...' : 'บันทึกลง Journal'}
                  </button>
                </div>
              </>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
