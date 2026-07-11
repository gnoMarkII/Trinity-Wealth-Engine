import { useState, useEffect, useRef } from 'react'
import type { MacroDashboardDTO } from '../api/types'
import {
  enrichIndicator,
  enrichSourceFile,
  type IndicatorCategory,
} from '../lib/macroReferences'

interface Props {
  data: MacroDashboardDTO
  isOpen: boolean
  onClose: () => void
}

type TabKey = 'observables' | 'source_files' | 'items'

const CATEGORY_STYLES: Record<IndicatorCategory, { bg: string; text: string; border: string }> = {
  Volatility: {
    bg: 'bg-purple-50',
    text: 'text-purple-700',
    border: 'border-purple-200',
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
    bg: 'bg-zinc-100',
    text: 'text-zinc-700',
    border: 'border-zinc-200',
  },
}

export default function MacroReferenceDrawer({ data, isOpen, onClose }: Props) {
  const [activeTab, setActiveTab] = useState<TabKey>('observables')
  const closeButtonRef = useRef<HTMLButtonElement>(null)

  useEffect(() => {
    if (isOpen) closeButtonRef.current?.focus()
  }, [isOpen])

  useEffect(() => {
    if (!isOpen) return
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [isOpen, onClose])

  if (!isOpen) return null

  // Collect all observable refs across items
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

  const enrichedIndicators = Array.from(observableSet).map((id) =>
    enrichIndicator(id, data)
  )

  const enrichedSources = (data.source_files || []).map((file) =>
    enrichSourceFile(file)
  )

  const reportSources = enrichedSources.filter((s) => s.type === 'report')
  const quantEngines = enrichedSources.filter((s) => s.type === 'quant_engine')

  return (
    <div className="fixed inset-0 z-50 overflow-hidden">
      {/* Backdrop */}
      <div
        className="animate-fade-in fixed inset-0 bg-black/40 backdrop-blur-sm transition-opacity"
        onClick={onClose}
      />

      {/* Slide-over Panel */}
      <div className="fixed inset-y-0 right-0 flex max-w-full pl-10">
        <div
          role="dialog"
          aria-modal="true"
          aria-labelledby="macro-reference-drawer-title"
          className="animate-drawer-in flex w-screen max-w-xl flex-col bg-white shadow-2xl"
        >
          {/* Drawer Header */}
          <div className="flex items-center justify-between border-b border-zinc-200 bg-zinc-900 px-6 py-4 text-white">
            <div className="flex items-center gap-3">
              <span className="text-2xl">📚</span>
              <div>
                <h2 id="macro-reference-drawer-title" className="text-base font-bold">
                  ศูนย์รวมอ้างอิงและตัวชี้วัดเศรษฐกิจ (Macro Reference Hub)
                </h2>
                <p className="text-xs text-zinc-400">
                  ตัวชี้วัดตลาด รายงานเศรษฐกิจ และโมเดลคำนวณเชิงปริมาณที่ใช้ในรายงาน
                </p>
              </div>
            </div>
            <button
              ref={closeButtonRef}
              onClick={onClose}
              className="rounded-lg bg-zinc-800 px-3 py-1.5 text-xs font-medium text-zinc-300 transition-colors hover:bg-zinc-700 hover:text-white"
            >
              ✕ ปิดหน้าต่าง
            </button>
          </div>

          {/* Sub-Tabs Selector */}
          <div className="flex border-b border-zinc-200 bg-zinc-50/80 px-6 pt-3">
            <button
              onClick={() => setActiveTab('observables')}
              aria-pressed={activeTab === 'observables'}
              className={`mr-6 border-b-2 pb-3 text-xs font-semibold transition-colors ${
                activeTab === 'observables'
                  ? 'border-zinc-900 text-zinc-900'
                  : 'border-transparent text-zinc-500 hover:text-zinc-800'
              }`}
            >
              📊 ตัวชี้วัดเศรษฐกิจ ({enrichedIndicators.length})
            </button>
            <button
              onClick={() => setActiveTab('source_files')}
              aria-pressed={activeTab === 'source_files'}
              className={`mr-6 border-b-2 pb-3 text-xs font-semibold transition-colors ${
                activeTab === 'source_files'
                  ? 'border-zinc-900 text-zinc-900'
                  : 'border-transparent text-zinc-500 hover:text-zinc-800'
              }`}
            >
              📑 รายงานและโมเดลคำนวณ ({enrichedSources.length})
            </button>
            <button
              onClick={() => setActiveTab('items')}
              aria-pressed={activeTab === 'items'}
              className={`border-b-2 pb-3 text-xs font-semibold transition-colors ${
                activeTab === 'items'
                  ? 'border-zinc-900 text-zinc-900'
                  : 'border-transparent text-zinc-500 hover:text-zinc-800'
              }`}
            >
              🎯 หลักฐานอ้างอิงรายสินทรัพย์
            </button>
          </div>

          {/* Tab Contents */}
          <div className="custom-scrollbar flex-1 overflow-y-auto overscroll-contain p-6">
            {/* TAB 1: Enriched Economic & Market Indicators */}
            {activeTab === 'observables' && (
              <div className="space-y-4">
                <div className="rounded-xl border border-blue-200 bg-blue-50/70 p-3.5 text-xs leading-relaxed text-blue-950">
                  <span className="font-bold text-blue-900">เกณฑ์ประเมินสภาวะตลาด:</span>{' '}
                  ตัวชี้วัดด้านล่างถูกดึงค่าล่าสุดและวิเคราะห์ร่วมกันเพื่อประเมินความเสี่ยงและกำหนดสัดส่วนลงทุน (Asset Allocation)
                </div>

                {enrichedIndicators.length === 0 ? (
                  <p className="text-xs italic text-zinc-400">ไม่มีตัวชี้วัดระบุไว้</p>
                ) : (
                  <div className="space-y-3">
                    {enrichedIndicators.map((ind, idx) => {
                      const style = CATEGORY_STYLES[ind.category]
                      return (
                        <div
                          key={idx}
                          className="rounded-xl border border-zinc-200/80 bg-white p-4 shadow-sm transition-all hover:border-zinc-300"
                        >
                          <div className="flex flex-wrap items-start justify-between gap-2">
                            <div className="space-y-1">
                              <div className="flex items-center gap-2">
                                <span
                                  className={`rounded-md border px-2 py-0.5 text-[10px] font-bold uppercase tracking-wider ${style.bg} ${style.text} ${style.border}`}
                                >
                                  {ind.category}
                                </span>
                                <span className="text-[11px] font-medium text-zinc-400">
                                  ที่มา: {ind.sourceProvider}
                                </span>
                              </div>
                              <h3 className="font-semibold text-zinc-900">{ind.name}</h3>
                            </div>

                            {ind.extractedValue && (
                              <div className="rounded-lg border border-zinc-900 bg-zinc-900 px-3 py-1.5 text-right shadow-sm">
                                <div className="text-[10px] uppercase tracking-wider text-zinc-400">
                                  ค่าปัจจุบัน
                                </div>
                                <div className="font-mono text-xs font-bold text-white">
                                  {ind.extractedValue}
                                </div>
                              </div>
                            )}
                          </div>

                          <p className="mt-2.5 text-xs leading-relaxed text-zinc-600">
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
                <div className="rounded-xl bg-zinc-100 p-3.5 text-xs text-zinc-600">
                  <span className="font-semibold text-zinc-800">ระบบประเมินผลหลัก:</span>{' '}
                  {data.generated_by || 'Strategic Allocator / Macro Core Engine'}{' '}
                  <span className="text-zinc-400">({data.evaluated_at})</span>
                </div>

                {/* Section 2.1: Macro Research Snapshots */}
                <div className="space-y-3">
                  <h3 className="flex items-center gap-2 text-xs font-bold uppercase tracking-wider text-zinc-700">
                    <span>📑</span>
                    <span>รายงานภาพรวมเศรษฐกิจและบทวิเคราะห์ (Macro Research Snapshots)</span>
                  </h3>
                  {reportSources.length === 0 ? (
                    <p className="text-xs italic text-zinc-400">ไม่มีรายงานเศรษฐกิจระบุไว้</p>
                  ) : (
                    <div className="space-y-2.5">
                      {reportSources.map((file, idx) => (
                        <div
                          key={idx}
                          className="rounded-xl border border-zinc-200 bg-white p-3.5 shadow-sm transition-colors hover:border-zinc-300"
                        >
                          <div className="flex items-start justify-between gap-2">
                            <div>
                              <div className="font-semibold text-zinc-900">{file.title}</div>
                              <div className="mt-1 text-xs text-zinc-500">{file.description}</div>
                              <div className="mt-2 font-mono text-[11px] text-zinc-400">
                                ไฟล์ฐานข้อมูล: {file.filename}
                              </div>
                            </div>
                            <span className="shrink-0 rounded-md bg-blue-50 border border-blue-200 px-2 py-0.5 text-[10px] font-semibold text-blue-700">
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
                  <h3 className="flex items-center gap-2 text-xs font-bold uppercase tracking-wider text-zinc-700">
                    <span>⚙️</span>
                    <span>ระเบียบวิธีประเมินผลเชิงปริมาณ (Quantitative Models & Engines)</span>
                  </h3>
                  {quantEngines.length === 0 ? (
                    <p className="text-xs italic text-zinc-400">ไม่มีโมเดลเชิงปริมาณระบุไว้</p>
                  ) : (
                    <div className="space-y-2.5">
                      {quantEngines.map((file, idx) => (
                        <div
                          key={idx}
                          className="rounded-xl border border-zinc-200 bg-zinc-50/60 p-3.5 shadow-sm transition-colors hover:border-zinc-300"
                        >
                          <div className="flex items-start justify-between gap-2">
                            <div>
                              <div className="font-semibold text-zinc-900">{file.title}</div>
                              <div className="mt-1 text-xs leading-relaxed text-zinc-600">
                                {file.description}
                              </div>
                              <div className="mt-2 font-mono text-[11px] text-zinc-400">
                                โมดูลคำนวณ: {file.filename}
                              </div>
                            </div>
                            <span className="shrink-0 rounded-md bg-purple-50 border border-purple-200 px-2 py-0.5 text-[10px] font-semibold text-purple-700">
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
                  <h3 className="mb-3 text-xs font-bold uppercase tracking-wider text-zinc-700">
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
                          className="rounded-xl border border-zinc-200/80 bg-white p-4 shadow-sm"
                        >
                          <div className="flex items-center justify-between">
                            <span className="font-semibold text-zinc-900">{a.asset_class}</span>
                            <span className="rounded-md bg-zinc-100 px-2.5 py-0.5 text-xs font-bold text-zinc-800">
                              {a.stance}
                            </span>
                          </div>

                          {indList.length > 0 && (
                            <div className="mt-3 space-y-1.5">
                              <div className="text-[11px] font-semibold uppercase tracking-wide text-zinc-500">
                                ตัวชี้วัดและตัวเลขที่ใช้ตัดสินใจ:
                              </div>
                              <div className="flex flex-wrap gap-1.5">
                                {indList.map((ind, k) => (
                                  <span
                                    key={k}
                                    className="inline-flex items-center gap-1.5 rounded-lg border border-zinc-200 bg-zinc-50 px-2.5 py-1 text-xs text-zinc-700"
                                  >
                                    <span className="font-medium">{ind.name}</span>
                                    {ind.extractedValue && (
                                      <span className="rounded bg-zinc-900 px-1.5 py-0.2 font-mono text-[11px] font-bold text-white">
                                        {ind.extractedValue}
                                      </span>
                                    )}
                                  </span>
                                ))}
                              </div>
                            </div>
                          )}

                          {srcList.length > 0 && (
                            <div className="mt-2.5 text-xs text-zinc-500">
                              <span className="font-semibold text-zinc-700">รายงานอ้างอิง: </span>
                              {srcList.map((s) => s.title).join(', ')}
                            </div>
                          )}

                          {indList.length === 0 && srcList.length === 0 && (
                            <div className="mt-2 text-xs italic text-zinc-400">
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
                    <h3 className="mb-3 text-xs font-bold uppercase tracking-wider text-zinc-700">
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
                            className="rounded-xl border border-zinc-200/80 bg-white p-4 shadow-sm"
                          >
                            <div className="font-semibold text-zinc-900">
                              Long <span className="text-emerald-600">{pt.long_leg}</span> / Short{' '}
                              <span className="text-red-600">{pt.short_leg}</span>
                            </div>

                            {indList.length > 0 && (
                              <div className="mt-3 space-y-1.5">
                                <div className="text-[11px] font-semibold uppercase tracking-wide text-zinc-500">
                                  ตัวชี้วัดที่ติดตาม:
                                </div>
                                <div className="flex flex-wrap gap-1.5">
                                  {indList.map((ind, k) => (
                                    <span
                                      key={k}
                                      className="inline-flex items-center gap-1.5 rounded-lg border border-zinc-200 bg-zinc-50 px-2.5 py-1 text-xs text-zinc-700"
                                    >
                                      <span className="font-medium">{ind.name}</span>
                                      {ind.extractedValue && (
                                        <span className="rounded bg-zinc-900 px-1.5 py-0.2 font-mono text-[11px] font-bold text-white">
                                          {ind.extractedValue}
                                        </span>
                                      )}
                                    </span>
                                  ))}
                                </div>
                              </div>
                            )}

                            {srcList.length > 0 && (
                              <div className="mt-2.5 text-xs text-zinc-500">
                                <span className="font-semibold text-zinc-700">รายงานอ้างอิง: </span>
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
          <div className="border-t border-zinc-200 bg-zinc-50 px-6 py-3 text-right">
            <button
              onClick={onClose}
              className="rounded-lg bg-zinc-900 px-4 py-2 text-xs font-semibold text-white transition-colors hover:bg-zinc-800"
            >
              ปิดหน้าต่างอ้างอิง
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}

