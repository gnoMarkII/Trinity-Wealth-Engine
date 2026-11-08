import { useState, useRef, useEffect, type KeyboardEvent } from 'react'
import {
  loadFavoritePrompts,
  togglePinFavoritePrompt,
  addCustomFavoritePrompt,
  updateFavoritePrompt,
  deleteFavoritePrompt,
  FAVORITES_UPDATED_EVENT,
  type FavoritePromptItem,
  type CreateFavoritePromptInput,
} from '../../lib/favoritePromptsStorage'
import ManagerFavoritePromptsPopover from './ManagerFavoritePromptsPopover'

interface ManagerPromptComposerProps {
  onSubmit: (message: string, contextRefs: string[]) => void
  isStreaming: boolean
  onStop?: () => void
}

const CONTEXT_PILLS = [
  { label: '@Portfolio', desc: 'ข้อมูลพอร์ตจริง' },
  { label: '@Macro', desc: 'ภาวะเศรษฐกิจมหภาค' },
  { label: '@Digest', desc: 'สรุปข่าวรายวัน' },
  { label: '@Essence', desc: 'กฎ 5 เสาหลัก & 3 ถังเงิน' },
  { label: '$NVDA', desc: 'หุ้น Nvidia' },
  { label: '$AAPL', desc: 'หุ้น Apple' },
]

export default function ManagerPromptComposer({
  onSubmit,
  isStreaming,
  onStop,
}: ManagerPromptComposerProps) {
  const [input, setInput] = useState('')
  const [selectedPills, setSelectedPills] = useState<string[]>([])
  const [prompts, setPrompts] = useState<FavoritePromptItem[]>(loadFavoritePrompts)
  const [activeTicker, setActiveTicker] = useState('NVDA')
  const [isPopoverOpen, setIsPopoverOpen] = useState(false)
  const [justFilled, setJustFilled] = useState(false)

  const textareaRef = useRef<HTMLTextAreaElement>(null)

  // Listen to external favorite updates
  useEffect(() => {
    function handleUpdate(e: Event) {
      const customEvent = e as CustomEvent<FavoritePromptItem[]>
      if (customEvent.detail && Array.isArray(customEvent.detail)) {
        setPrompts(customEvent.detail)
      } else {
        setPrompts(loadFavoritePrompts())
      }
    }
    window.addEventListener(FAVORITES_UPDATED_EVENT, handleUpdate)
    return () => window.removeEventListener(FAVORITES_UPDATED_EVENT, handleUpdate)
  }, [])

  // Global Ctrl+K / Cmd+K listener
  useEffect(() => {
    function handleGlobalKey(e: globalThis.KeyboardEvent) {
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'k') {
        e.preventDefault()
        setIsPopoverOpen((prev) => !prev)
      }
    }
    window.addEventListener('keydown', handleGlobalKey)
    return () => window.removeEventListener('keydown', handleGlobalKey)
  }, [])

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto'
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 160)}px`
    }
  }, [input])

  function togglePill(pillLabel: string) {
    if (selectedPills.includes(pillLabel)) {
      setSelectedPills(selectedPills.filter((p) => p !== pillLabel))
    } else {
      setSelectedPills([...selectedPills, pillLabel])
    }
  }

  function handleFillInput(promptText: string, contextRefs: string[]) {
    setInput(promptText)
    // Merge contextRefs into selectedPills without duplicates
    setSelectedPills((prev) => Array.from(new Set([...prev, ...contextRefs])))
    setJustFilled(true)
    setTimeout(() => setJustFilled(false), 700)

    if (textareaRef.current) {
      textareaRef.current.focus()
      // Position cursor at the end
      setTimeout(() => {
        if (textareaRef.current) {
          textareaRef.current.selectionStart = textareaRef.current.value.length
          textareaRef.current.selectionEnd = textareaRef.current.value.length
        }
      }, 20)
    }
  }

  function handleKeyDown(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    } else if (e.key === '/' && input.trim() === '') {
      e.preventDefault()
      setIsPopoverOpen(true)
    }
  }

  function handleSend() {
    const trimmed = input.trim()
    if (!trimmed || isStreaming) return
    onSubmit(trimmed, selectedPills)
    setInput('')
    setSelectedPills([])
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto'
    }
  }

  function handleTogglePin(id: string) {
    const updated = togglePinFavoritePrompt(id)
    setPrompts(updated)
  }

  function handleCreateCustom(data: CreateFavoritePromptInput) {
    const updated = addCustomFavoritePrompt(data)
    setPrompts(updated)
  }

  function handleUpdateCustom(id: string, data: CreateFavoritePromptInput) {
    const updated = updateFavoritePrompt(id, data)
    setPrompts(updated)
  }

  function handleDeleteCustom(id: string) {
    const updated = deleteFavoritePrompt(id)
    setPrompts(updated)
  }

  return (
    <div className="w-full border-t border-sky-100/80 bg-white/70 p-3 sm:p-4 backdrop-blur-xl transition-all">
      <div className="mx-auto max-w-4xl">
        {/* Integrated Composer Box (Claude / ChatGPT Style) */}
        <div
          className={`relative flex flex-col rounded-3xl border bg-panel p-3 shadow-[0_8px_30px_rgba(14,165,233,0.06)] transition-all ${
            justFilled
              ? 'border-sky-400 ring-2 ring-sky-300/80 shadow-[0_0_15px_rgba(14,165,233,0.25)]'
              : 'border-sky-200/80 focus-within:border-sky-400 focus-within:ring-2 focus-within:ring-sky-200/50'
          }`}
        >
          {/* Active Context Chips (if any selected, shown at top of textarea) */}
          {selectedPills.length > 0 && (
            <div className="mb-2 flex flex-wrap items-center gap-1.5 px-1 animate-notice-in">
              <span className="text-xs font-medium text-sky-800/70">บริบทที่เลือก:</span>
              {selectedPills.map((pill) => (
                <span
                  key={pill}
                  className="inline-flex items-center gap-1 rounded-full border border-sky-300 bg-sky-100/90 px-2.5 py-0.5 text-xs font-semibold text-sky-800 shadow-sm"
                >
                  <span>{pill}</span>
                  <button
                    type="button"
                    onClick={() => togglePill(pill)}
                    className="ml-0.5 rounded-full p-0.5 text-sky-600 hover:bg-sky-200 hover:text-sky-900 transition-colors"
                    title={`ลบ ${pill}`}
                  >
                    ✕
                  </button>
                </span>
              ))}
              <button
                type="button"
                onClick={() => setSelectedPills([])}
                className="text-xs text-zinc-400 hover:text-zinc-600 underline ml-1"
              >
                ล้างทั้งหมด
              </button>
            </div>
          )}

          {/* Textarea */}
          <textarea
            ref={textareaRef}
            rows={1}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="ปรึกษาการลงทุน, วิเคราะห์พอร์ต, หรือพิมพ์ / เพื่อเปิดคลังคำสั่ง..."
            className="custom-scrollbar max-h-40 min-h-[48px] w-full resize-none bg-transparent px-2.5 py-1.5 text-[15px] sm:text-base text-sky-950 placeholder-zinc-400 focus:outline-none leading-relaxed"
          />

          {/* Integrated Bottom Action Bar */}
          <div className="mt-2 flex flex-wrap items-center justify-between gap-2 border-t border-sky-100/60 pt-2">
            {/* Left Tools & Context Controls */}
            <div className="flex flex-wrap items-center gap-1.5 min-w-0">
              {/* Command Library Button — Calm Minimal Style */}
              <button
                type="button"
                onClick={() => setIsPopoverOpen(true)}
                title="เปิดคลังคำสั่งโปรดครอบคลุมทุกมิติ (Ctrl+K หรือพิมพ์ /)"
                className="flex shrink-0 items-center gap-1.5 rounded-xl border border-sky-100 bg-sky-50/60 px-3 py-1.5 text-xs sm:text-sm font-medium text-sky-900 hover:border-sky-200 hover:bg-sky-100/70 transition-all active:scale-95"
              >
                <span className="text-sky-500">✨</span>
                <span>คลังคำสั่ง</span>
                <span className="hidden sm:inline font-mono text-[10px] text-sky-700/70 rounded bg-sky-100/80 px-1.5 py-0.5">
                  Ctrl+K
                </span>
              </button>

              <div className="h-4 w-px bg-sky-200/60 mx-1 hidden sm:block" />

              {/* Context Quick Pills */}
              <div className="flex items-center gap-1 overflow-x-auto no-scrollbar py-0.5">
                {CONTEXT_PILLS.map((pill) => {
                  const active = selectedPills.includes(pill.label)
                  return (
                    <button
                      key={pill.label}
                      type="button"
                      onClick={() => togglePill(pill.label)}
                      title={pill.desc}
                      className={`rounded-xl px-2.5 py-1 text-xs font-medium transition-all ${
                        active
                          ? 'border border-sky-400 bg-sky-100 text-sky-900 font-bold shadow-sm'
                          : 'border border-sky-100/80 bg-white/80 text-zinc-600 hover:border-sky-200 hover:bg-sky-50/70 hover:text-sky-800'
                      }`}
                    >
                      {pill.label}
                    </button>
                  )
                })}
              </div>
            </div>

            {/* Right Submit / Stop Button */}
            <div className="flex items-center gap-1.5 shrink-0 ml-auto">
              {isStreaming ? (
                <button
                  type="button"
                  onClick={onStop}
                  className="flex h-9 items-center gap-1.5 rounded-xl border border-rose-200 bg-rose-50 px-3.5 text-xs sm:text-sm font-semibold text-rose-600 shadow-sm transition-colors hover:bg-rose-100"
                >
                  <span className="h-2 w-2 rounded-full bg-rose-500 animate-pulse" />
                  <span>หยุดตอบ</span>
                </button>
              ) : (
                <button
                  type="button"
                  disabled={!input.trim()}
                  onClick={handleSend}
                  className={`flex h-9 items-center justify-center gap-1.5 rounded-xl px-4 text-xs sm:text-sm font-semibold shadow-sm transition-all ${
                    input.trim()
                      ? 'bg-sky-500 hover:bg-sky-600 text-white shadow-[0_4px_16px_rgba(14,165,233,0.3)] hover:brightness-105 active:scale-95'
                      : 'cursor-not-allowed bg-zinc-100 text-zinc-400'
                  }`}
                >
                  <span>ส่งคำถาม</span>
                  <span className="text-sm">➔</span>
                </button>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Popover Command Modal */}
      <ManagerFavoritePromptsPopover
        isOpen={isPopoverOpen}
        onClose={() => setIsPopoverOpen(false)}
        prompts={prompts}
        activeTicker={activeTicker}
        onActiveTickerChange={setActiveTicker}
        onTogglePin={handleTogglePin}
        onCreateCustom={handleCreateCustom}
        onUpdateCustom={handleUpdateCustom}
        onDeleteCustom={handleDeleteCustom}
        onSelectToEdit={handleFillInput}
        onSelectAndSend={(prompt, refs) => onSubmit(prompt, refs)}
      />
    </div>
  )
}
