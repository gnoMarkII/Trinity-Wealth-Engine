import { useState } from 'react'
import type { MacroDashboardDTO } from '../api/types'
import {
  enrichIndicator,
  enrichSourceFile,
  type IndicatorCategory,
} from '../lib/macroReferences'
import { useFocusTrap } from '../hooks/useFocusTrap'
import {
  BookOpenIcon,
  ChartTrendIcon,
  CloseIcon,
  FolderIcon,
  ScrollHistoryIcon,
  TargetIcon,
} from './ui/AppIcons'

interface Props {
  data: MacroDashboardDTO
  isOpen: boolean
  onClose: () => void
}

type TabKey = 'observables' | 'registry' | 'source_files' | 'items'

const CATEGORY_STYLES: Record<IndicatorCategory, { bg: string; text: string; border: string }> = {
  Volatility: {
    bg: 'bg-sky-50',
    text: 'text-blue-700',
    border: 'border-sky-200',
  },
  Credit: {
    bg: 'bg-blue-50',
    text: 'text-blue-700',
    border: 'border-blue-200',
  },
  'Rates & Liquidity': {
    bg: 'bg-emerald-50',
    text: 'text-emerald-700',
    border: 'border-emerald-200',
  },
  'Commodities & Energy': {
    bg: 'bg-amber-50',
    text: 'text-amber-700',
    border: 'border-amber-200',
  },
  'Macro & Equities': {
    bg: 'bg-surface-strong',
    text: 'text-slate-700',
    border: 'border-edge',
  },
}

const STATUS_BADGES: Record<string, { bg: string; text: string; border: string; label: string }> = {
  verified: { bg: 'bg-emerald-50', text: 'text-emerald-700', border: 'border-emerald-200', label: 'Verified' },
  stale: { bg: 'bg-amber-50', text: 'text-amber-700', border: 'border-amber-200', label: 'Stale' },
  mock: { bg: 'bg-sky-50', text: 'text-blue-700', border: 'border-sky-200', label: 'Mock/Synthetic' },
  unavailable: { bg: 'bg-slate-100', text: 'text-slate-600', border: 'border-slate-200', label: 'Unavailable' },
}

interface NormalizedObservable {
  id: string
  name: string
  region: string
  provider: string
  source_type: string
  status: string
  value: string
  unit: string
  observed_at: string
  stale_reason?: string | null
  description?: string
}

export default function MacroReferenceDrawer({ data, isOpen, onClose }: Props) {
  const [activeTab, setActiveTab] = useState<TabKey>('observables')
  const [regionFilter, setRegionFilter] = useState<string>('All')
  const [statusFilter, setStatusFilter] = useState<string>('All')
  // focus trap ร่วมกับ Modal — เดิม drawer มีแค่ Escape + โฟกัสปุ่มปิด แต่ Tab ยังหลุด
  // ออกไปหน้า background ที่ถูก backdrop บังอยู่ได้
  const panelRef = useFocusTrap<HTMLDivElement>(isOpen, onClose)

  if (!isOpen) return null

  // Collect all observable refs across items (including Thailand market stance)
  const observableSet = new Set<string>()
  data.asset_allocation?.forEach((a) => {
    a.observable_refs?.forEach((r) => observableSet.add(r))
  })
  data.pair_trades?.forEach((pt) => {
    pt.observable_refs?.forEach((r) => observableSet.add(r))
  })
  data.regime_evidence?.forEach((re) => {
    re.observable_refs?.forEach((r) => observableSet.add(r))
  })
  data.thailand_market_stance?.observable_refs?.forEach((r) => observableSet.add(r))

  const enrichedIndicators = Array.from(observableSet).map((id) =>
    enrichIndicator(id, data)
  )

  const enrichedSources = (data.source_files || []).map((file) =>
    enrichSourceFile(file)
  )

  const reportSources = enrichedSources.filter((s) => s.type === 'report')
  const quantEngines = enrichedSources.filter((s) => s.type === 'quant_engine')

  // Normalized observable registry list
  const registryItems: NormalizedObservable[] = []
  if (data.observable_registry && Object.keys(data.observable_registry).length > 0) {
    Object.entries(data.observable_registry).forEach(([k, item]) => {
      if (!item) return
      registryItems.push({
        id: item.observable_id || k,
        name: item.canonical_name || k,
        region: item.region || 'Global',
        provider: item.provider || '—',
        source_type: item.source_type || 'derived',
        status: item.status || 'verified',
        value: item.value !== undefined && item.value !== null ? String(item.value) : '—',
        unit: item.unit || '',
        observed_at: item.observed_at || '—',
        stale_reason: item.stale_reason || null,
        description: item.family,
      })
    })
  } else if (data.dashboard_indicators && data.dashboard_indicators.length > 0) {
    data.dashboard_indicators.forEach((ind) => {
      registryItems.push({
        id: ind.indicator_id,
        name: ind.label,
        region: ind.region || 'Global',
        provider: ind.provider || '—',
        source_type: ind.source_type || 'derived',
        status: ind.status || 'verified',
        value: ind.display_value,
        unit: ind.unit || '',
        observed_at: ind.observed_at || '—',
        stale_reason: ind.status === 'stale' ? (ind.stale_reason || 'ข้อมูลขาดการอัปเดตเกินรอบสังเกตการณ์') : null,
        description: ind.source_file,
      })
    })
  }

  const filteredRegistry = registryItems.filter((item) => {
    const matchRegion = regionFilter === 'All' || item.region.toLowerCase() === regionFilter.toLowerCase()
    const matchStatus = statusFilter === 'All' || item.status.toLowerCase() === statusFilter.toLowerCase()
    return matchRegion && matchStatus
  })

  return (
    <div className="fixed inset-0 z-50 overflow-hidden">
      {/* Backdrop — ฉากหลังตกแต่ง ปิดด้วยคลิกสำหรับ mouse user (คีย์บอร์ดใช้ Escape) */}
      <div
        aria-hidden="true"
        className="animate-fade-in fixed inset-0 bg-black/40 backdrop-blur-sm transition-opacity"
        onClick={onClose}
      />

      {/* Slide-over Panel */}
      <div className="fixed inset-y-0 right-0 flex max-w-full pl-10">
        <div
          ref={panelRef}
          role="dialog"
          aria-modal="true"
          aria-labelledby="macro-reference-drawer-title"
          className="animate-drawer-in flex w-screen max-w-xl flex-col bg-panel shadow-2xl"
        >
          {/* Drawer Header */}
          <div className="flex items-center justify-between border-b border-sky-200/80 bg-gradient-to-r from-sky-50/90 via-blue-50/70 to-white px-6 py-4 text-slate-900">
            <div className="flex items-center gap-3">
              <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-blue-100/80 border border-blue-200 text-section-accent">
                <BookOpenIcon className="h-5 w-5" />
              </span>
              <div>
                <h2 id="macro-reference-drawer-title" className="text-base font-bold text-slate-900">
                  ศูนย์รวมอ้างอิงและตัวชี้วัดเศรษฐกิจ (Macro Reference Hub)
                </h2>
                <p className="text-xs text-slate-600">
                  ตัวชี้วัดตลาด รายงานเศรษฐกิจ และโมเดลคำนวณเชิงปริมาณที่ใช้ในรายงาน
                </p>
              </div>
            </div>
            <button
              type="button"
              onClick={onClose}
              className="inline-flex items-center gap-1.5 rounded-lg border border-slate-200 bg-white px-3 py-1.5 text-xs font-medium text-slate-700 transition-colors hover:bg-slate-100 hover:text-slate-900"
            >
              <CloseIcon className="h-3.5 w-3.5" />
              <span>ปิดหน้าต่าง</span>
            </button>
          </div>

          {/* Sub-Tabs Selector */}
          <div className="flex border-b border-edge bg-surface px-6 pt-3 overflow-x-auto">
            <button
              onClick={() => setActiveTab('observables')}
              aria-pressed={activeTab === 'observables'}
              className={`mr-6 inline-flex items-center gap-1.5 border-b-2 pb-3 text-xs font-semibold whitespace-nowrap transition-colors ${
                activeTab === 'observables'
                  ? 'border-section-accent text-section-accent'
                  : 'border-transparent text-slate-500 hover:text-slate-800'
              }`}
            >
              <ChartTrendIcon className="h-3.5 w-3.5" />
              <span>ตัวชี้วัดเศรษฐกิจ ({enrichedIndicators.length})</span>
            </button>
            <button
              onClick={() => setActiveTab('registry')}
              aria-pressed={activeTab === 'registry'}
              className={`mr-6 inline-flex items-center gap-1.5 border-b-2 pb-3 text-xs font-semibold whitespace-nowrap transition-colors ${
                activeTab === 'registry'
                  ? 'border-section-accent text-section-accent'
                  : 'border-transparent text-slate-500 hover:text-slate-800'
              }`}
            >
              <FolderIcon className="h-3.5 w-3.5" />
              <span>ทะเบียน Observables ({registryItems.length})</span>
            </button>
            <button
              onClick={() => setActiveTab('source_files')}
              aria-pressed={activeTab === 'source_files'}
              className={`mr-6 inline-flex items-center gap-1.5 border-b-2 pb-3 text-xs font-semibold whitespace-nowrap transition-colors ${
                activeTab === 'source_files'
                  ? 'border-section-accent text-section-accent'
                  : 'border-transparent text-slate-500 hover:text-slate-800'
              }`}
            >
              <ScrollHistoryIcon className="h-3.5 w-3.5" />
              <span>รายงานและโมเดลคำนวณ ({enrichedSources.length})</span>
            </button>
            <button
              onClick={() => setActiveTab('items')}
              aria-pressed={activeTab === 'items'}
              className={`inline-flex items-center gap-1.5 border-b-2 pb-3 text-xs font-semibold whitespace-nowrap transition-colors ${
                activeTab === 'items'
                  ? 'border-section-accent text-section-accent'
                  : 'border-transparent text-slate-500 hover:text-slate-800'
              }`}
            >
              <TargetIcon className="h-3.5 w-3.5" />
              <span>หลักฐานอ้างอิงรายสินทรัพย์</span>
            </button>
          </div>

          {/* Tab Contents */}
          <div className="custom-scrollbar flex-1 overflow-y-auto overscroll-contain p-6">
            {/* TAB: Observable Registry (All Feeds with Metadata & Filters) */}
            {activeTab === 'registry' && (
              <div className="space-y-4">
                <div className="rounded-xl border border-sky-200 bg-sky-50/70 p-3.5 text-xs leading-relaxed text-sky-950">
                  <span className="font-bold text-sky-900">ทะเบียนข้อมูลสังเกตการณ์ (Observable Registry):</span>{' '}
                  แสดงตัวแปรทั้งหมดในระบบ พร้อมระบุภูมิภาค ชนิดแหล่งข้อมูล วันสังเกตการณ์จริง และสถานะความสดใหม่
                </div>

                {/* Filter Controls: Region & Status */}
                <div className="space-y-2 rounded-xl border border-edge bg-surface p-3 text-xs">
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <div className="flex items-center gap-1.5 flex-wrap">
                      <span className="font-semibold text-slate-600 mr-1">ภูมิภาค:</span>
                      {['All', 'United States', 'Thailand', 'Global'].map((reg) => (
                        <button
                          key={reg}
                          type="button"
                          onClick={() => setRegionFilter(reg)}
                          className={`rounded-md px-2.5 py-0.5 text-[11px] font-medium transition-colors ${
                            regionFilter === reg
                              ? 'bg-section-accent text-white shadow-sm'
                              : 'border border-edge bg-panel text-slate-600 hover:bg-sky-50 hover:text-sky-900'
                          }`}
                        >
                          {reg}
                        </button>
                      ))}
                    </div>

                    <div className="flex items-center gap-1.5 flex-wrap">
                      <span className="font-semibold text-slate-600 mr-1">สถานะ:</span>
                      {['All', 'verified', 'stale', 'mock', 'unavailable'].map((st) => (
                        <button
                          key={st}
                          type="button"
                          onClick={() => setStatusFilter(st)}
                          className={`rounded-md px-2.5 py-0.5 text-[11px] font-medium transition-colors ${
                            statusFilter === st
                              ? 'bg-section-accent text-white shadow-sm'
                              : 'border border-edge bg-panel text-slate-600 hover:bg-sky-50 hover:text-sky-900'
                          }`}
                        >
                          {st}
                        </button>
                      ))}
                    </div>
                  </div>
                </div>

                {filteredRegistry.length === 0 ? (
                  <p className="text-xs italic text-slate-500 py-4 text-center">
                    ไม่พบตัวชี้วัดตามเงื่อนไขตัวกรอง ({regionFilter}, {statusFilter})
                  </p>
                ) : (
                  <div className="space-y-3">
                    {filteredRegistry.map((item) => {
                      const badge = STATUS_BADGES[item.status] || {
                        bg: 'bg-slate-100',
                        text: 'text-slate-600',
                        border: 'border-slate-200',
                        label: item.status,
                      }
                      return (
                        <div
                          key={item.id}
                          className="rounded-xl border border-edge bg-panel p-4 shadow-sm space-y-2 transition-all hover:border-slate-300"
                        >
                          <div className="flex flex-wrap items-start justify-between gap-2">
                            <div className="space-y-1">
                              <div className="flex items-center gap-1.5 flex-wrap">
                                <span className="rounded-md border border-slate-200 bg-slate-100 px-2 py-0.5 text-[11px] font-bold text-slate-700">
                                  {item.region}
                                </span>
                                <span className="rounded-md border border-slate-200 bg-slate-50 px-2 py-0.5 text-[11px] font-medium text-slate-500">
                                  {item.source_type}
                                </span>
                                <span
                                  className={`rounded-md border px-2 py-0.5 text-[11px] font-bold uppercase ${badge.bg} ${badge.text} ${badge.border}`}
                                >
                                  {badge.label}
                                </span>
                              </div>
                              <h4 className="font-semibold text-slate-900 text-xs">{item.name}</h4>
                              <code className="text-[11px] font-mono text-slate-500">{item.id}</code>
                            </div>

                            <div className="text-right">
                              <div className="font-mono text-sm font-bold text-slate-900">
                                {item.value} {item.unit}
                              </div>
                              <div className="text-[11px] text-slate-500">
                                สังเกตการณ์: {item.observed_at}
                              </div>
                            </div>
                          </div>

                          <div className="flex items-center justify-between text-[11px] text-slate-500 pt-1 border-t border-slate-100">
                            <span>
                              ผู้ให้บริการ: <strong className="text-slate-700">{item.provider}</strong>
                            </span>
                            {item.description && <span className="text-slate-500">{item.description}</span>}
                          </div>

                          {item.stale_reason && (
                            <div className="rounded-lg border border-amber-200 bg-amber-50/80 p-2 text-[11px] text-amber-900 flex items-center gap-1.5">
                              <span>⚠️</span>
                              <span>
                                <strong>เหตุผลที่ stale:</strong> {item.stale_reason}
                              </span>
                            </div>
                          )}
                        </div>
                      )
                    })}
                  </div>
                )}
              </div>
            )}
            {/* TAB 1: Enriched Economic & Market Indicators */}
            {activeTab === 'observables' && (
              <div className="space-y-4">
                <div className="rounded-xl border border-blue-200 bg-blue-50/70 p-3.5 text-xs leading-relaxed text-blue-950">
                  <span className="font-bold text-blue-900">เกณฑ์ประเมินสภาวะตลาด:</span>{' '}
                  ตัวชี้วัดด้านล่างถูกดึงค่าล่าสุดและวิเคราะห์ร่วมกันเพื่อประเมินความเสี่ยงและกำหนดสัดส่วนลงทุน (Asset Allocation)
                </div>

                {enrichedIndicators.length === 0 ? (
                  <p className="text-xs italic text-slate-500">ไม่มีตัวชี้วัดระบุไว้</p>
                ) : (
                  <div className="space-y-3">
                    {enrichedIndicators.map((ind, idx) => {
                      const style = CATEGORY_STYLES[ind.category]
                      return (
                        <div
                          key={idx}
                          className="rounded-xl border border-edge bg-panel p-4 shadow-sm transition-all hover:border-slate-300"
                        >
                          <div className="flex flex-wrap items-start justify-between gap-2">
                            <div className="space-y-1">
                              <div className="flex items-center gap-2">
                                <span
                                  className={`rounded-md border px-2 py-0.5 text-[11px] font-bold uppercase tracking-wider ${style.bg} ${style.text} ${style.border}`}
                                >
                                  {ind.category}
                                </span>
                                <span className="text-[11px] font-medium text-slate-500">
                                  ที่มา: {ind.sourceProvider}
                                </span>
                              </div>
                              <h3 className="font-semibold text-slate-900">{ind.name}</h3>
                            </div>

                            {ind.extractedValue && (
                              <div className="rounded-lg border border-blue-200 bg-blue-50/90 px-3 py-1.5 text-right shadow-sm">
                                <div className="text-[11px] uppercase tracking-wider text-blue-700 font-semibold">
                                  ค่าปัจจุบัน
                                </div>
                                <div className="font-mono text-xs font-bold text-blue-950">
                                  {ind.extractedValue}
                                </div>
                              </div>
                            )}
                          </div>

                          <p className="mt-2.5 text-xs leading-relaxed text-slate-600">
                            {ind.description}
                          </p>
                        </div>
                      )
                    })}
                  </div>
                )}
              </div>
            )}

            {/* TAB 2: Reports & Quant Models */}
            {activeTab === 'source_files' && (
              <div className="space-y-6">
                <div className="rounded-xl bg-surface-strong p-3.5 text-xs text-slate-600">
                  <span className="font-semibold text-slate-800">ระบบประเมินผลหลัก:</span>{' '}
                  {data.generated_by || 'Strategic Allocator / Macro Core Engine'}{' '}
                  <span className="text-slate-500">({data.evaluated_at})</span>
                </div>

                {/* Section 2.1: Macro Research Snapshots */}
                <div className="space-y-3">
                  <h3 className="flex items-center gap-2 text-xs font-bold uppercase tracking-wider text-slate-700">
                    <span>📑</span>
                    <span>รายงานภาพรวมเศรษฐกิจและบทวิเคราะห์ (Macro Research Snapshots)</span>
                  </h3>
                  {reportSources.length === 0 ? (
                    <p className="text-xs italic text-slate-500">ไม่มีรายงานเศรษฐกิจระบุไว้</p>
                  ) : (
                    <div className="space-y-2.5">
                      {reportSources.map((file, idx) => (
                        <div
                          key={idx}
                          className="rounded-xl border border-edge bg-panel p-3.5 shadow-sm transition-colors hover:border-slate-300"
                        >
                          <div className="flex items-start justify-between gap-2">
                            <div>
                              <div className="font-semibold text-slate-900">{file.title}</div>
                              <div className="mt-1 text-xs text-slate-500">{file.description}</div>
                              <div className="mt-2 font-mono text-[11px] text-slate-500">
                                ไฟล์ฐานข้อมูล: {file.filename}
                              </div>
                            </div>
                            <span className="shrink-0 rounded-md bg-blue-50 border border-blue-200 px-2 py-0.5 text-[11px] font-semibold text-blue-700">
                              {file.badgeText}
                            </span>
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>

                {/* Section 2.2: Quantitative Analytical Engines */}
                <div className="space-y-3">
                  <h3 className="flex items-center gap-2 text-xs font-bold uppercase tracking-wider text-slate-700">
                    <span>⚙️</span>
                    <span>ระเบียบวิธีประเมินผลเชิงปริมาณ (Quantitative Models & Engines)</span>
                  </h3>
                  {quantEngines.length === 0 ? (
                    <p className="text-xs italic text-slate-500">ไม่มีโมเดลเชิงปริมาณระบุไว้</p>
                  ) : (
                    <div className="space-y-2.5">
                      {quantEngines.map((file, idx) => (
                        <div
                          key={idx}
                          className="rounded-xl border border-edge bg-slate-50/60 p-3.5 shadow-sm transition-colors hover:border-slate-300"
                        >
                          <div className="flex items-start justify-between gap-2">
                            <div>
                              <div className="font-semibold text-slate-900">{file.title}</div>
                              <div className="mt-1 text-xs leading-relaxed text-slate-600">
                                {file.description}
                              </div>
                              <div className="mt-2 font-mono text-[11px] text-slate-500">
                                โมดูลคำนวณ: {file.filename}
                              </div>
                            </div>
                            <span className="shrink-0 rounded-md bg-sky-50 border border-sky-200 px-2 py-0.5 text-[11px] font-semibold text-blue-700">
                              {file.badgeText}
                            </span>
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            )}

            {/* TAB 3: Strategy Traceability & Supporting Evidence */}
            {activeTab === 'items' && (
              <div className="space-y-6">
                {/* Cross-Asset Allocations */}
                <div>
                  <h3 className="mb-3 text-xs font-bold uppercase tracking-wider text-slate-700">
                    หลักฐานอ้างอิง: สัดส่วนสินทรัพย์ลงทุน (Cross-Asset Allocations)
                  </h3>
                  <div className="space-y-3">
                    {data.asset_allocation?.map((a, idx) => {
                      const indList = (a.observable_refs || []).map((r) =>
                        enrichIndicator(r, data)
                      )
                      const srcList = (a.source_refs || []).map((s) =>
                        enrichSourceFile(s)
                      )
                      return (
                        <div
                          key={idx}
                          className="rounded-xl border border-edge bg-panel p-4 shadow-sm"
                        >
                          <div className="flex items-center justify-between">
                            <span className="font-semibold text-slate-900">{a.asset_class}</span>
                            <span className="rounded-md bg-surface-strong px-2.5 py-0.5 text-xs font-bold text-slate-800">
                              {a.stance}
                            </span>
                          </div>

                          {indList.length > 0 && (
                            <div className="mt-3 space-y-1.5">
                              <div className="text-[11px] font-semibold uppercase tracking-wide text-slate-500">
                                ตัวชี้วัดและตัวเลขที่ใช้ตัดสินใจ:
                              </div>
                              <div className="flex flex-wrap gap-1.5">
                                {indList.map((ind, k) => (
                                  <span
                                    key={k}
                                    className="inline-flex items-center gap-1.5 rounded-lg border border-edge bg-surface px-2.5 py-1 text-xs text-slate-700"
                                  >
                                    <span className="font-medium">{ind.name}</span>
                                    {ind.extractedValue && (
                                      <span className="rounded bg-blue-50 border border-blue-200 px-1.5 py-0.5 font-mono text-[11px] font-bold text-blue-950">
                                        {ind.extractedValue}
                                      </span>
                                    )}
                                  </span>
                                ))}
                              </div>
                            </div>
                          )}

                          {srcList.length > 0 && (
                            <div className="mt-2.5 text-xs text-slate-500">
                              <span className="font-semibold text-slate-700">รายงานอ้างอิง: </span>
                              {srcList.map((s) => s.title).join(', ')}
                            </div>
                          )}

                          {indList.length === 0 && srcList.length === 0 && (
                            <div className="mt-2 text-xs italic text-slate-500">
                              อ้างอิงจากบทวิเคราะห์สภาวะเศรษฐกิจรวมในหน้าหลัก
                            </div>
                          )}
                        </div>
                      )
                    })}
                  </div>
                </div>

                {/* Tactical Pair Trades */}
                {data.pair_trades && data.pair_trades.length > 0 && (
                  <div>
                    <h3 className="mb-3 text-xs font-bold uppercase tracking-wider text-slate-700">
                      หลักฐานอ้างอิง: กลยุทธ์จับคู่ Long/Short (Tactical Pair Trades)
                    </h3>
                    <div className="space-y-3">
                      {data.pair_trades.map((pt, idx) => {
                        const indList = (pt.observable_refs || []).map((r) =>
                          enrichIndicator(r, data)
                        )
                        const srcList = (pt.source_refs || []).map((s) =>
                          enrichSourceFile(s)
                        )
                        return (
                          <div
                            key={idx}
                            className="rounded-xl border border-edge bg-panel p-4 shadow-sm"
                          >
                            <div className="font-semibold text-slate-900">
                              Long <span className="text-emerald-600">{pt.long_leg}</span> / Short{' '}
                              <span className="text-red-600">{pt.short_leg}</span>
                            </div>

                            {indList.length > 0 && (
                              <div className="mt-3 space-y-1.5">
                                <div className="text-[11px] font-semibold uppercase tracking-wide text-slate-500">
                                  ตัวชี้วัดที่ติดตาม:
                                </div>
                                <div className="flex flex-wrap gap-1.5">
                                  {indList.map((ind, k) => (
                                    <span
                                      key={k}
                                      className="inline-flex items-center gap-1.5 rounded-lg border border-edge bg-surface px-2.5 py-1 text-xs text-slate-700"
                                    >
                                      <span className="font-medium">{ind.name}</span>
                                      {ind.extractedValue && (
                                        <span className="rounded bg-slate-900 px-1.5 py-0.5 font-mono text-[11px] font-bold text-white">
                                          {ind.extractedValue}
                                        </span>
                                      )}
                                    </span>
                                  ))}
                                </div>
                              </div>
                            )}

                            {srcList.length > 0 && (
                              <div className="mt-2.5 text-xs text-slate-500">
                                <span className="font-semibold text-slate-700">รายงานอ้างอิง: </span>
                                {srcList.map((s) => s.title).join(', ')}
                              </div>
                            )}
                          </div>
                        )
                      })}
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Drawer Footer */}
          <div className="border-t border-edge bg-surface px-6 py-3 text-right">
            <button
              onClick={onClose}
              className="rounded-lg bg-section-accent px-4 py-2 text-xs font-semibold text-white transition-colors hover:bg-blue-900"
            >
              ปิดหน้าต่างอ้างอิง
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}

