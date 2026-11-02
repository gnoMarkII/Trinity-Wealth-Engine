import { useState, useMemo, useEffect, useRef } from 'react'
import { createPortal } from 'react-dom'
import { useFocusTrap } from '../../hooks/useFocusTrap'
import TextInput from '../ui/TextInput'
import {
  type FavoritePromptItem,
  type FavoritePromptCategory,
  type CreateFavoritePromptInput,
  resolvePromptWithTicker,
} from '../../lib/favoritePromptsStorage'

export interface ManagerFavoritePromptsPopoverProps {
  isOpen: boolean
  onClose: () => void
  prompts: FavoritePromptItem[]
  activeTicker: string
  onActiveTickerChange: (ticker: string) => void
  onTogglePin: (id: string) => void
  onCreateCustom: (input: CreateFavoritePromptInput) => void
  onUpdateCustom: (id: string, input: CreateFavoritePromptInput) => void
  onDeleteCustom: (id: string) => void
  onSelectToEdit: (resolvedPrompt: string, resolvedContextRefs: string[]) => void
  onSelectAndSend: (resolvedPrompt: string, resolvedContextRefs: string[]) => void
}

type TabKey = 'pinned' | 'macro' | 'equity' | 'portfolio' | 'workflows'

const TABS: { key: TabKey; label: string; icon: string }[] = [
  { key: 'pinned', label: 'รายการโปรด', icon: '⭐' },
  { key: 'macro', label: 'เศรษฐกิจมหภาค', icon: '🌐' },
  { key: 'equity', label: 'หุ้นรายตัว', icon: '🔬' },
  { key: 'portfolio', label: 'พอร์ต & กลยุทธ์', icon: '💼' },
  { key: 'workflows', label: 'สั่งรันงาน & ความรู้', icon: '🚀' },
]

const QUICK_TICKERS = ['NVDA', 'AAPL', 'MSFT', 'GOOGL', 'TSM', 'ADVANC', 'DELTA']

const CATEGORY_STYLES: Record<
  FavoritePromptCategory,
  { badge: string; border: string; accent: string }
> = {
  macro: {
    badge: 'bg-sky-50 text-sky-700 border-sky-200',
    border: 'border-sky-100 hover:border-sky-300',
    accent: 'text-sky-800',
  },
  equity: {
    badge: 'bg-indigo-50 text-indigo-700 border-indigo-200',
    border: 'border-indigo-100 hover:border-indigo-300',
    accent: 'text-indigo-800',
  },
  portfolio: {
    badge: 'bg-emerald-50 text-emerald-700 border-emerald-200',
    border: 'border-emerald-100 hover:border-emerald-300',
    accent: 'text-emerald-800',
  },
  workflows: {
    badge: 'bg-amber-50 text-amber-700 border-amber-200',
    border: 'border-amber-100 hover:border-amber-300',
    accent: 'text-amber-800',
  },
}

export default function ManagerFavoritePromptsPopover({
  isOpen,
  onClose,
  prompts,
  activeTicker,
  onActiveTickerChange,
  onTogglePin,
  onCreateCustom,
  onUpdateCustom,
  onDeleteCustom,
  onSelectToEdit,
  onSelectAndSend,
}: ManagerFavoritePromptsPopoverProps) {
  const [activeTab, setActiveTab] = useState<TabKey>('pinned')
  const [search, setSearch] = useState('')
  const [isFormOpen, setIsFormOpen] = useState(false)
  const [editingId, setEditingId] = useState<string | null>(null)

  // Form states
  const [formTitle, setFormTitle] = useState('')
  const [formBadge, setFormBadge] = useState('')
  const [formPrompt, setFormPrompt] = useState('')
  const [formCategory, setFormCategory] = useState<FavoritePromptCategory>('equity')
  const [formContextRefs, setFormContextRefs] = useState<string[]>(['@Portfolio'])

  const searchInputRef = useRef<HTMLInputElement>(null)
  const dialogRef = useFocusTrap<HTMLDivElement>(isOpen, onClose)

  useEffect(() => {
    if (isOpen) {
      setTimeout(() => searchInputRef.current?.focus(), 50)
    } else {
      setIsFormOpen(false)
      setEditingId(null)
    }
  }, [isOpen])

  const pinnedCount = useMemo(() => prompts.filter((p) => p.isPinned).length, [prompts])

  const filteredPrompts = useMemo(() => {
    const q = search.trim().toLowerCase()
    return prompts.filter((item) => {
      // If searching, search globally across all prompts
      if (!q) {
        if (activeTab === 'pinned') {
          if (!item.isPinned) return false
        } else {
          if (item.category !== activeTab) return false
        }
      } else {
        const matchTitle = item.title.toLowerCase().includes(q)
        const matchPrompt = item.prompt.toLowerCase().includes(q)
        const matchBadge = item.dimensionBadge.toLowerCase().includes(q)
        if (!matchTitle && !matchPrompt && !matchBadge) return false
      }

      return true
    })
  }, [prompts, activeTab, search])

  const showTickerBar =
    activeTab === 'equity' ||
    filteredPrompts.some((p) => p.requiresTicker || p.prompt.includes('[TICKER]'))

  function openCreateForm() {
    setEditingId(null)
    setFormTitle('')
    setFormBadge('')
    setFormPrompt('')
    setFormCategory(activeTab === 'pinned' ? 'equity' : activeTab)
    setFormContextRefs(['@Portfolio'])
    setIsFormOpen(true)
  }

  function openEditForm(item: FavoritePromptItem) {
    setEditingId(item.id)
    setFormTitle(item.title)
    setFormBadge(item.dimensionBadge)
    setFormPrompt(item.prompt)
    setFormCategory(item.category)
    setFormContextRefs(item.contextRefs)
    setIsFormOpen(true)
  }

  function handleFormSubmit(e: React.FormEvent) {
    e.preventDefault()
    if (!formTitle.trim() || !formPrompt.trim()) return

    const inputData: CreateFavoritePromptInput = {
      title: formTitle,
      dimensionBadge: formBadge || 'Custom Insight',
      prompt: formPrompt,
      category: formCategory,
      contextRefs: formContextRefs,
      requiresTicker: formPrompt.includes('[TICKER]'),
    }

    if (editingId) {
      onUpdateCustom(editingId, inputData)
    } else {
      onCreateCustom(inputData)
    }
    setIsFormOpen(false)
    setEditingId(null)
  }

  function toggleFormContext(ref: string) {
    if (formContextRefs.includes(ref)) {
      setFormContextRefs(formContextRefs.filter((r) => r !== ref))
    } else {
      setFormContextRefs([...formContextRefs, ref])
    }
  }

  if (!isOpen) return null

  return createPortal(
    <div className="fixed inset-0 z-50 flex items-center justify-center p-3 sm:p-5 animate-fade-in">
      {/* Backdrop */}
      <div
        aria-hidden="true"
        onClick={onClose}
        className="absolute inset-0 bg-sky-950/40 backdrop-blur-sm"
      />

      {/* Main Dialog */}
      <div
        ref={dialogRef}
        role="dialog"
        aria-modal="true"
        aria-labelledby="favorite-prompts-title"
        className="relative flex h-[88vh] max-h-[820px] w-full max-w-4xl flex-col rounded-3xl border border-sky-100 bg-white/95 shadow-2xl shadow-sky-900/15 backdrop-blur-2xl animate-modal-in overflow-hidden"
      >
        {/* Top Header */}
        <div className="flex shrink-0 items-center justify-between border-b border-sky-100 px-5 py-3.5 bg-gradient-to-r from-sky-50/60 via-white to-cyan-50/40">
          <div className="flex items-center gap-2.5">
            <div className="flex h-9 w-9 items-center justify-center rounded-2xl bg-gradient-to-tr from-amber-400 to-orange-500 text-white shadow-sm">
              <span className="text-base">⭐</span>
            </div>
            <div>
              <h2 id="favorite-prompts-title" className="text-sm font-bold text-sky-950">
                คลังคำสั่งอัจฉริยะ (Favorite Prompts Library)
              </h2>
              <p className="text-[11px] text-zinc-500">
                คำสั่งครอบคลุมครบทุกมิติ Macro, Equity, Portfolio & Workflows
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={openCreateForm}
              className="flex items-center gap-1 rounded-xl border border-sky-200 bg-sky-50/80 px-2.5 py-1 text-xs font-semibold text-sky-800 shadow-sm hover:bg-sky-100 transition-colors"
            >
              <span>+</span>
              <span>สร้างคำสั่งใหม่</span>
            </button>
            <button
              type="button"
              onClick={onClose}
              className="flex h-8 w-8 items-center justify-center rounded-xl border border-sky-100 text-zinc-400 hover:bg-sky-50 hover:text-sky-800 transition-colors"
              title="ปิดหน้าต่าง (Esc)"
            >
              ✕
            </button>
          </div>
        </div>

        {/* Search & Tabs Controls */}
        <div className="shrink-0 space-y-2.5 border-b border-sky-100 bg-white/60 p-4 backdrop-blur-md">
          {/* Search bar */}
          <div className="flex items-center gap-2">
            <TextInput
              ref={searchInputRef}
              uiSize="sm"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="ค้นหาคำสั่ง เช่น DCF, Yield Curve, 3 ถังเงิน, Rebalance, มหภาค..."
              className="flex-1"
            />
            {search && (
              <button
                type="button"
                onClick={() => setSearch('')}
                className="rounded-lg px-2 py-1 text-xs text-zinc-400 hover:text-zinc-600"
              >
                ล้างคำค้น
              </button>
            )}
          </div>

          {/* Segmented Category Tabs */}
          <div className="flex flex-wrap items-center gap-1.5 overflow-x-auto no-scrollbar">
            {TABS.map((tab) => {
              const active = activeTab === tab.key
              const count =
                tab.key === 'pinned'
                  ? pinnedCount
                  : prompts.filter((p) => p.category === tab.key).length

              return (
                <button
                  key={tab.key}
                  type="button"
                  onClick={() => setActiveTab(tab.key)}
                  className={`flex items-center gap-1.5 rounded-xl px-3 py-1.5 text-xs font-bold transition-all ${
                    active
                      ? 'border border-sky-300 bg-sky-500 text-white shadow-sm shadow-sky-500/25'
                      : 'border border-sky-100 bg-panel text-zinc-600 hover:bg-sky-50 hover:text-sky-900'
                  }`}
                >
                  <span>{tab.icon}</span>
                  <span>{tab.label}</span>
                  <span
                    className={`rounded-full px-1.5 py-0.2 text-[9px] font-mono ${
                      active ? 'bg-white/25 text-white' : 'bg-sky-100 text-sky-800'
                    }`}
                  >
                    {count}
                  </span>
                </button>
              )
            })}
          </div>

          {/* Live Ticker Bar (when in Equity or when viewing Ticker prompts) */}
          {showTickerBar && (
            <div className="flex flex-wrap items-center gap-2 rounded-2xl border border-indigo-100 bg-indigo-50/50 p-2 text-xs">
              <span className="font-semibold text-indigo-950 flex items-center gap-1">
                <span>🎯</span>
                <span>หุ้นเป้าหมาย [TICKER]:</span>
              </span>
              <div className="flex flex-wrap items-center gap-1">
                {QUICK_TICKERS.map((t) => (
                  <button
                    key={t}
                    type="button"
                    onClick={() => onActiveTickerChange(t)}
                    className={`rounded-lg px-2 py-0.5 text-xs font-bold transition-all ${
                      activeTicker.toUpperCase() === t
                        ? 'bg-indigo-600 text-white shadow-sm'
                        : 'bg-white text-indigo-800 border border-indigo-200 hover:bg-indigo-100/60'
                    }`}
                  >
                    {t}
                  </button>
                ))}
              </div>
              <input
                type="text"
                value={activeTicker}
                onChange={(e) => onActiveTickerChange(e.target.value.toUpperCase())}
                placeholder="ระบุหุ้นอื่นๆ เช่น DELTA"
                className="w-28 rounded-lg border border-indigo-200 bg-white px-2 py-0.5 text-xs font-bold uppercase text-indigo-900 placeholder:normal-case placeholder-zinc-400 focus:outline-none focus:ring-1 focus:ring-indigo-400"
              />
            </div>
          )}
        </div>

        {/* Inline Form for Creating / Editing Custom Prompts */}
        {isFormOpen && (
          <form
            onSubmit={handleFormSubmit}
            className="shrink-0 border-b border-sky-200 bg-gradient-to-b from-sky-50/90 to-white p-4 space-y-3 animate-notice-in"
          >
            <div className="flex items-center justify-between">
              <h3 className="text-xs font-bold text-sky-950 flex items-center gap-1.5">
                <span>{editingId ? '✎ แก้ไขคำสั่งส่วนตัว' : '✨ สร้างคำสั่งส่วนตัวใหม่'}</span>
              </h3>
              <button
                type="button"
                onClick={() => setIsFormOpen(false)}
                className="text-xs text-zinc-400 hover:text-zinc-600"
              >
                ยกเลิก
              </button>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
              <input
                type="text"
                required
                value={formTitle}
                onChange={(e) => setFormTitle(e.target.value)}
                placeholder="ชื่อคำสั่ง เช่น ตรวจสอบงบกระแสเงินสด"
                className="rounded-xl border border-sky-200 bg-white px-3 py-1.5 text-xs text-sky-950 focus:outline-none focus:ring-1 focus:ring-sky-400"
              />
              <input
                type="text"
                value={formBadge}
                onChange={(e) => setFormBadge(e.target.value)}
                placeholder="ป้ายมิติ เช่น Cash Flow Quality (ไม่ระบุก็ได้)"
                className="rounded-xl border border-sky-200 bg-white px-3 py-1.5 text-xs text-sky-950 focus:outline-none focus:ring-1 focus:ring-sky-400"
              />
            </div>

            <textarea
              required
              rows={2}
              value={formPrompt}
              onChange={(e) => setFormPrompt(e.target.value)}
              placeholder="ข้อความคำสั่งที่ต้องการให้ AI ตอบ (ใส่ [TICKER] ได้เพื่อแทนค่าชื่อหุ้น)"
              className="w-full rounded-xl border border-sky-200 bg-white p-2.5 text-xs text-sky-950 focus:outline-none focus:ring-1 focus:ring-sky-400"
            />

            <div className="flex flex-wrap items-center justify-between gap-2 pt-1">
              <div className="flex items-center gap-2">
                <span className="text-[11px] text-zinc-500 font-medium">หมวดหมู่:</span>
                <select
                  value={formCategory}
                  onChange={(e) => setFormCategory(e.target.value as FavoritePromptCategory)}
                  className="rounded-lg border border-sky-200 bg-white px-2 py-1 text-xs text-sky-950"
                >
                  <option value="macro">เศรษฐกิจมหภาค (Macro)</option>
                  <option value="equity">หุ้นรายตัว (Equity)</option>
                  <option value="portfolio">พอร์ต & กลยุทธ์ (Portfolio)</option>
                  <option value="workflows">สั่งรันงาน & คลังความรู้</option>
                </select>
              </div>

              {/* Context toggles */}
              <div className="flex items-center gap-1">
                <span className="text-[11px] text-zinc-500">บริบท:</span>
                {['@Portfolio', '@Macro', '@Digest', '@Essence'].map((pill) => (
                  <button
                    key={pill}
                    type="button"
                    onClick={() => toggleFormContext(pill)}
                    className={`rounded-md px-2 py-0.5 text-[10px] font-semibold transition-all ${
                      formContextRefs.includes(pill)
                        ? 'bg-sky-500 text-white'
                        : 'bg-zinc-100 text-zinc-600 hover:bg-zinc-200'
                    }`}
                  >
                    {pill}
                  </button>
                ))}
              </div>

              <button
                type="submit"
                className="rounded-xl bg-sky-500 hover:bg-sky-600 px-4 py-1.5 text-xs font-bold text-white shadow-sm transition-all"
              >
                บันทึกคำสั่ง
              </button>
            </div>
          </form>
        )}

        {/* Cards Grid Body */}
        <div className="flex-1 min-h-0 overflow-y-auto p-4 custom-scrollbar">
          {filteredPrompts.length === 0 ? (
            <div className="flex h-48 flex-col items-center justify-center text-center text-zinc-400">
              <span className="text-3xl mb-1">🔍</span>
              <p className="text-xs">ไม่พบคำสั่งที่ตรงกับตัวกรองนี้</p>
              {activeTab === 'pinned' && (
                <p className="text-[11px] text-zinc-400 mt-1">
                  กดติดดาว ⭐ บนการ์ดคำสั่งที่คุณใช้บ่อยเพื่อรวมไว้ที่นี่
                </p>
              )}
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3 pb-4">
              {filteredPrompts.map((item, index) => {
                const resolved = resolvePromptWithTicker(item, activeTicker)
                const style = CATEGORY_STYLES[item.category]

                return (
                  <div
                    key={item.id}
                    style={{ animationDelay: `${Math.min(index * 20, 160)}ms` }}
                    className={`flex flex-col justify-between rounded-2xl border bg-panel p-3.5 shadow-sm transition-all hover:shadow-md animate-card-in ${style.border}`}
                  >
                    <div>
                      {/* Card Top Row: Badge + Star Pin Button */}
                      <div className="flex items-center justify-between gap-1.5 mb-2">
                        <div className="flex items-center gap-1.5 flex-wrap">
                          <span
                            className={`rounded-md border px-2 py-0.5 text-[10px] font-bold ${style.badge}`}
                          >
                            {item.dimensionBadge}
                          </span>
                          {item.isCustom && (
                            <span className="rounded-md border border-purple-200 bg-purple-50 px-1.5 py-0.5 text-[9px] font-semibold text-purple-700">
                              Custom
                            </span>
                          )}
                        </div>

                        <div className="flex items-center gap-1">
                          {item.isCustom && (
                            <>
                              <button
                                type="button"
                                onClick={() => openEditForm(item)}
                                title="แก้ไขคำสั่ง"
                                className="rounded p-1 text-zinc-400 hover:text-sky-700 hover:bg-sky-50"
                              >
                                ✎
                              </button>
                              <button
                                type="button"
                                onClick={() => onDeleteCustom(item.id)}
                                title="ลบคำสั่ง"
                                className="rounded p-1 text-zinc-400 hover:text-rose-600 hover:bg-rose-50"
                              >
                                🗑️
                              </button>
                            </>
                          )}
                          <button
                            type="button"
                            onClick={() => onTogglePin(item.id)}
                            title={item.isPinned ? 'ถอนหมุดจากรายการโปรด' : 'ปักหมุดเป็นรายการโปรด'}
                            className={`rounded-lg p-1 text-sm transition-transform active:scale-125 ${
                              item.isPinned
                                ? 'text-amber-500 animate-icon-pop'
                                : 'text-zinc-300 hover:text-amber-400'
                            }`}
                          >
                            ⭐
                          </button>
                        </div>
                      </div>

                      {/* Title */}
                      <h4 className="text-xs font-bold text-sky-950 mb-1 leading-snug">
                        {resolved.title}
                      </h4>

                      {/* Prompt Snippet */}
                      <p className="text-[11px] text-zinc-600 leading-relaxed line-clamp-3 mb-2.5">
                        {resolved.prompt}
                      </p>
                    </div>

                    {/* Card Footer: Context Pills + Action Buttons */}
                    <div className="pt-2 border-t border-sky-100/60 flex items-center justify-between gap-2 mt-auto">
                      <div className="flex items-center gap-1 flex-wrap">
                        {resolved.contextRefs.map((ref) => (
                          <span
                            key={ref}
                            className="rounded border border-sky-200/80 bg-sky-50 px-1.5 py-0.2 text-[9px] font-mono text-sky-800"
                          >
                            {ref}
                          </span>
                        ))}
                      </div>

                      <div className="flex items-center gap-1.5 shrink-0">
                        <button
                          type="button"
                          onClick={() => {
                            onSelectToEdit(resolved.prompt, resolved.contextRefs)
                            onClose()
                          }}
                          title="นำข้อความไปวางในช่องแชทเพื่อแก้ไขก่อนส่ง"
                          className="flex items-center gap-1 rounded-xl border border-sky-200 bg-white px-2 py-1 text-[11px] font-semibold text-sky-800 shadow-sm hover:bg-sky-50 transition-colors"
                        >
                          <span>✍️</span>
                          <span>แก้ไขก่อน</span>
                        </button>

                        <button
                          type="button"
                          onClick={() => {
                            onSelectAndSend(resolved.prompt, resolved.contextRefs)
                            onClose()
                          }}
                          title="ส่งคำสั่งนี้เข้าห้องสนทนากับ Trinity AI ทันที"
                          className="flex items-center gap-1 rounded-xl bg-sky-500 hover:bg-sky-600 px-2.5 py-1 text-[11px] font-bold text-white shadow-sm shadow-sky-500/20 active:scale-95 transition-all"
                        >
                          <span>⚡</span>
                          <span>ส่งทันที</span>
                        </button>
                      </div>
                    </div>
                  </div>
                )
              })}
            </div>
          )}
        </div>
      </div>
    </div>,
    document.body
  )
}
