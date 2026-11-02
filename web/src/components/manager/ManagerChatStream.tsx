import { useEffect, useRef, useState, useMemo } from 'react'
import type { AdvisorChatMessageDTO } from '../../api/types'
import AdvisorMarkdown from './AdvisorMarkdown'
import ManagerFluidOrb from './ManagerFluidOrb'
import ManagerResearchJobCard from './ManagerResearchJobCard'
import { extractPrimaryJobId } from '../../lib/jobDetection'
import {
  loadFavoritePrompts,
  resolvePromptWithTicker,
  type FavoritePromptCategory,
  type FavoritePromptItem,
} from '../../lib/favoritePromptsStorage'

interface ManagerChatStreamProps {
  messages: AdvisorChatMessageDTO[]
  streamingText: string
  isThinking: boolean
  isStreaming: boolean
  onQuickPromptClick?: (prompt: string, contextRefs?: string[]) => void
  onSelectJob?: (jobId: string) => void
  onOpenKanban?: () => void
}

type ShowcaseCategory = 'featured' | FavoritePromptCategory

const SHOWCASE_TABS: { key: ShowcaseCategory; label: string; icon: string }[] = [
  { key: 'featured', label: 'แนะนำ', icon: '⭐' },
  { key: 'macro', label: 'เศรษฐกิจมหภาค', icon: '🌐' },
  { key: 'equity', label: 'หุ้นรายตัว', icon: '🔬' },
  { key: 'portfolio', label: 'พอร์ตโฟลิโอ', icon: '💼' },
  { key: 'workflows', label: 'สั่งรันงาน', icon: '🚀' },
]

const QUICK_TICKERS = ['NVDA', 'AAPL', 'MSFT', 'TSM', 'ADVANC']

export default function ManagerChatStream({
  messages,
  streamingText,
  isThinking,
  isStreaming,
  onQuickPromptClick,
  onSelectJob,
  onOpenKanban,
}: ManagerChatStreamProps) {
  const bottomRef = useRef<HTMLDivElement>(null)
  const [prompts] = useState<FavoritePromptItem[]>(loadFavoritePrompts)
  const [showcaseTab, setShowcaseTab] = useState<ShowcaseCategory>('featured')
  const [showcaseTicker, setShowcaseTicker] = useState('NVDA')

  useEffect(() => {
    if (typeof bottomRef.current?.scrollIntoView === 'function') {
      bottomRef.current.scrollIntoView({ behavior: 'smooth' })
    }
  }, [messages, streamingText, isThinking])

  const isEmpty = messages.length === 0 && !streamingText && !isThinking

  const showcasePrompts = useMemo(() => {
    if (showcaseTab === 'featured') {
      return prompts.filter((p) => p.isPinned).slice(0, 4)
    }
    return prompts.filter((p) => p.category === showcaseTab).slice(0, 4)
  }, [prompts, showcaseTab])

  if (isEmpty) {
    return (
      <div className="flex h-full flex-col items-center justify-start overflow-y-auto p-4 sm:p-6 text-center animate-page-in custom-scrollbar">
        <div className="mx-auto max-w-2xl w-full flex flex-col items-center my-auto py-4">
          <div className="mb-3 flex h-14 w-14 items-center justify-center rounded-3xl bg-gradient-to-tr from-sky-400 via-cyan-400 to-sky-600 shadow-[0_10px_35px_rgba(14,165,233,0.18)]">
            <span className="text-2xl">✨</span>
          </div>
          <h2 className="mb-1 text-xl font-bold tracking-tight text-sky-950">
            ยินดีต้อนรับสู่ Trinity AI Advisor
          </h2>
          <p className="mb-5 max-w-md text-xs leading-relaxed text-zinc-500">
            ผู้จัดการกองทุนและที่ปรึกษาการลงทุนส่วนตัว เชื่อมโยงข้อมูลจริงครบทั้งพอร์ต เศรษฐกิจมหภาค และบทวิเคราะห์หุ้นรายตัว
          </p>

          {/* Interactive Category Tabs */}
          <div className="mb-3 flex flex-wrap items-center justify-center gap-1.5">
            {SHOWCASE_TABS.map((tab) => {
              const active = showcaseTab === tab.key
              return (
                <button
                  key={tab.key}
                  type="button"
                  onClick={() => setShowcaseTab(tab.key)}
                  className={`flex items-center gap-1 rounded-xl px-2.5 py-1 text-xs font-semibold transition-all ${
                    active
                      ? 'border border-sky-300 bg-sky-500 text-white shadow-sm shadow-sky-500/25'
                      : 'border border-sky-100 bg-panel text-zinc-600 hover:bg-sky-50 hover:text-sky-900'
                  }`}
                >
                  <span>{tab.icon}</span>
                  <span>{tab.label}</span>
                </button>
              )
            })}
          </div>

          {/* Mini Ticker Selector when in Equity tab */}
          {showcaseTab === 'equity' && (
            <div className="mb-3 flex items-center justify-center gap-1.5 rounded-xl border border-indigo-100 bg-indigo-50/60 px-3 py-1.5 text-xs animate-notice-in">
              <span className="font-semibold text-indigo-900 text-[11px]">หุ้น [TICKER]:</span>
              {QUICK_TICKERS.map((t) => (
                <button
                  key={t}
                  type="button"
                  onClick={() => setShowcaseTicker(t)}
                  className={`rounded px-1.5 py-0.5 text-[11px] font-bold transition-all ${
                    showcaseTicker === t
                      ? 'bg-indigo-600 text-white'
                      : 'bg-white text-indigo-800 border border-indigo-200 hover:bg-indigo-100'
                  }`}
                >
                  {t}
                </button>
              ))}
            </div>
          )}

          {/* Prompt Cards Grid */}
          <div className="grid w-full grid-cols-1 gap-2.5 sm:grid-cols-2 text-left">
            {showcasePrompts.map((item, index) => {
              const resolved = resolvePromptWithTicker(item, showcaseTicker)
              return (
                <button
                  key={item.id}
                  type="button"
                  style={{ animationDelay: `${index * 30}ms` }}
                  onClick={() => onQuickPromptClick?.(resolved.prompt, resolved.contextRefs)}
                  className="group relative flex flex-col justify-between rounded-2xl border border-sky-100 bg-panel p-3.5 shadow-sm transition-all hover:-translate-y-0.5 hover:border-sky-300 hover:bg-sky-50/70 hover:shadow-md animate-card-in"
                >
                  <div>
                    <div className="mb-1.5 flex items-center justify-between">
                      <span className="rounded-md border border-sky-200/80 bg-sky-50 px-1.5 py-0.2 text-[9px] font-bold text-sky-700">
                        {item.dimensionBadge}
                      </span>
                      <span className="text-[11px] text-zinc-400 group-hover:text-sky-600 transition-colors">
                        คลิกเพื่อเริ่ม →
                      </span>
                    </div>
                    <h3 className="text-xs font-bold text-sky-950 mb-1 group-hover:text-sky-700 transition-colors">
                      {resolved.title}
                    </h3>
                    <p className="text-[11px] text-zinc-500 line-clamp-2 leading-relaxed">
                      {resolved.prompt}
                    </p>
                  </div>

                  <div className="mt-2.5 flex items-center gap-1 flex-wrap pt-1.5 border-t border-sky-100/40">
                    {resolved.contextRefs.map((ref) => (
                      <span
                        key={ref}
                        className="rounded border border-sky-200/60 bg-white px-1 py-0.2 text-[9px] font-mono text-sky-800"
                      >
                        {ref}
                      </span>
                    ))}
                  </div>
                </button>
              )
            })}
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="custom-scrollbar h-full space-y-6 overflow-y-auto px-4 py-6 sm:px-6">
      <div className="mx-auto max-w-4xl space-y-5">
        {messages.map((msg) => {
          const isUser = msg.role === 'user'

          return (
            <div
              key={msg.id}
              className={`flex gap-3 animate-chat-turn-in ${
                isUser ? 'justify-end' : 'justify-start'
              }`}
            >
              {!isUser && <ManagerFluidOrb mode="avatar" />}

              <div
                className={`max-w-[85%] sm:max-w-[78%] rounded-2xl p-4 transition-all ${
                  isUser
                    ? 'border border-sky-300/60 bg-sky-500/10 text-sky-950 shadow-sm'
                    : 'border border-sky-100 bg-panel text-zinc-800 shadow-[0_8px_24px_rgba(14,165,233,0.06)] backdrop-blur-md'
                }`}
              >
                {/* Citations / Context Tags */}
                {msg.citations && msg.citations.length > 0 && (
                  <div className="mb-2 flex flex-wrap items-center gap-1">
                    {msg.citations.map((c) => (
                      <span
                        key={c}
                        className="rounded-full border border-sky-200 bg-sky-50 px-2 py-0.5 text-[10px] font-medium text-sky-700"
                      >
                        {c}
                      </span>
                    ))}
                  </div>
                )}

                {/* Message Body */}
                <AdvisorMarkdown content={msg.content} />

                {/* Job Card (if message spawned a research job) */}
                {(() => {
                  const matchedJobId = extractPrimaryJobId(msg.content)
                  if (!matchedJobId) return null
                  const tickerMatch = msg.content.match(/(?:สำหรับ|วิจัย|หุ้น)\s*([A-Z]{1,6})\b/)
                  const matchedTicker: string = tickerMatch && tickerMatch[1] ? tickerMatch[1] : ''
                  const isMacro = msg.content.toLowerCase().includes('macro') || msg.content.includes('มหภาค') || msg.content.includes('เศรษฐกิจ')
                  return (
                    <ManagerResearchJobCard
                      jobId={matchedJobId}
                      ticker={matchedTicker}
                      jobType={matchedTicker ? 'equity_intel' : (isMacro ? 'macro_intel' : 'equity_intel')}
                      status="queued"
                      onSelectJob={onSelectJob}
                      onOpenKanban={onOpenKanban}
                    />
                  )
                })()}
              </div>

              {isUser && (
                <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-xl border border-sky-200 bg-sky-100 text-xs font-bold text-sky-800">
                  คุณ
                </div>
              )}
            </div>
          )
        })}

        {/* Initial Thinking State: ManagerFluidOrb */}
        {isThinking && (
          <div className="flex justify-start gap-3 animate-chat-turn-in">
            <ManagerFluidOrb mode="avatar" />
            <div className="rounded-2xl border border-sky-100 bg-panel px-6 py-2 shadow-[0_8px_24px_rgba(14,165,233,0.06)] backdrop-blur-md">
              <ManagerFluidOrb size="sm" />
            </div>
          </div>
        )}

        {/* Real-time Streaming Response Turn: 28px Orb Avatar */}
        {isStreaming && streamingText && (
          <div className="flex justify-start gap-3 animate-chat-turn-in">
            <ManagerFluidOrb mode="avatar" />
            <div className="max-w-[85%] sm:max-w-[78%] rounded-2xl border border-sky-100 bg-panel p-4 text-zinc-800 shadow-[0_8px_24px_rgba(14,165,233,0.06)] backdrop-blur-md">
              <AdvisorMarkdown content={streamingText} />

              {/* Job Card (if streaming text already spawned or contains a research job) */}
              {(() => {
                const matchedJobId = extractPrimaryJobId(streamingText)
                if (!matchedJobId) return null
                const tickerMatch = streamingText.match(/(?:สำหรับ|วิจัย|หุ้น)\s*([A-Z]{1,6})\b/)
                const matchedTicker: string = tickerMatch && tickerMatch[1] ? tickerMatch[1] : ''
                const isMacro =
                  streamingText.toLowerCase().includes('macro') ||
                  streamingText.includes('มหภาค') ||
                  streamingText.includes('เศรษฐกิจ')
                return (
                  <ManagerResearchJobCard
                    jobId={matchedJobId}
                    ticker={matchedTicker}
                    jobType={matchedTicker ? 'equity_intel' : (isMacro ? 'macro_intel' : 'equity_intel')}
                    status="running"
                    onSelectJob={onSelectJob}
                    onOpenKanban={onOpenKanban}
                  />
                )
              })()}

              <div className="mt-2 flex items-center gap-1.5 text-xs text-sky-600">
                <span className="h-2 w-2 rounded-full bg-flow-blue animate-pulse" />
                <span className="font-medium text-[11px]">กำลังสร้างคำตอบ...</span>
              </div>
            </div>
          </div>
        )}

        <div ref={bottomRef} />
      </div>
    </div>
  )
}
