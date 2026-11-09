import { useEffect, useRef } from 'react'
import type { AdvisorChatMessageDTO } from '../../api/types'
import AdvisorMarkdown from './AdvisorMarkdown'
import ManagerFluidOrb from './ManagerFluidOrb'
import ManagerResearchJobCard from './ManagerResearchJobCard'
import { extractPrimaryJobId } from '../../lib/jobDetection'

interface ManagerChatStreamProps {
  messages: AdvisorChatMessageDTO[]
  streamingText: string
  isThinking: boolean
  isStreaming: boolean
  onQuickPromptClick?: (prompt: string, contextRefs?: string[]) => void
  onSelectJob?: (jobId: string) => void
  onOpenKanban?: () => void
}

const STARTER_PROMPTS = [
  {
    icon: '📊',
    title: 'วิเคราะห์ความเสี่ยงและสัดส่วนพอร์ต',
    prompt: 'ช่วยวิเคราะห์ความเสี่ยงและสัดส่วนพอร์ตปัจจุบัน พร้อมข้อเสนอแนะในการปรับสมดุล',
    contextRefs: ['@Portfolio'],
  },
  {
    icon: '🏛️',
    title: 'สรุปภาพรวมมหภาค & Morning Delta',
    prompt: 'สรุปประเด็นเศรษฐกิจมหภาคสำคัญและ Morning Delta ที่กระทบพอร์ตเช้านี้',
    contextRefs: ['@Macro', '@Portfolio'],
  },
  {
    icon: '🔬',
    title: 'วิเคราะห์ Valuation & ปัจจัยหุ้น NVDA',
    prompt: 'ช่วยวิเคราะห์ Valuation, DCF และความคาดหวังของตลาดสำหรับหุ้น NVDA',
    contextRefs: ['@Portfolio'],
  },
  {
    icon: '🛡️',
    title: 'ตรวจสอบ Guardrails 3 ถังการลงทุน',
    prompt: 'ตรวจสอบสถานะ Guardrails และเงินสำรองสภาพคล่อง 3 ถังว่าสอดคล้องกับสภาวะตลาดหรือไม่',
    contextRefs: ['@Portfolio', '@Essence'],
  },
]

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

  useEffect(() => {
    if (typeof bottomRef.current?.scrollIntoView === 'function') {
      bottomRef.current.scrollIntoView({ behavior: 'smooth' })
    }
  }, [messages, streamingText, isThinking])

  const isEmpty = messages.length === 0 && !streamingText && !isThinking

  if (isEmpty) {
    return (
      <div className="flex h-full flex-col items-center justify-center p-4 sm:p-6 text-center animate-page-in">
        <div className="mx-auto flex w-full max-w-xl flex-col items-center my-auto py-8">
          <h2 className="text-2xl sm:text-3xl font-semibold tracking-tight text-sky-950 mb-2">
            มีอะไรให้ช่วยวิเคราะห์วันนี้ไหมครับ?
          </h2>
          <p className="text-sm text-zinc-500 mb-6 max-w-md">
            ผู้จัดการกองทุนและที่ปรึกษา AI พร้อมเชื่อมโยงข้อมูลพอร์ตและสภาวะตลาด
          </p>

          {/* Clean 4 Starter Prompt Pills */}
          <div className="grid w-full grid-cols-1 gap-2.5 sm:grid-cols-2 text-left">
            {STARTER_PROMPTS.map((item) => (
              <button
                key={item.title}
                type="button"
                onClick={() => onQuickPromptClick?.(item.prompt, item.contextRefs)}
                className="group flex items-center justify-between rounded-xl border border-sky-100/80 bg-white/80 px-4 py-3 text-sm font-medium text-sky-950 shadow-xs hover:border-sky-300 hover:bg-sky-50/70 transition-all"
              >
                <span className="flex items-center gap-2.5 truncate">
                  <span className="text-base shrink-0">{item.icon}</span>
                  <span className="truncate group-hover:text-sky-700 transition-colors">
                    {item.title}
                  </span>
                </span>
                <span className="text-zinc-300 group-hover:translate-x-0.5 group-hover:text-sky-500 transition-all text-sm shrink-0 ml-1">
                  →
                </span>
              </button>
            ))}
          </div>

          <p className="mt-6 text-xs text-zinc-400">
            กด <kbd className="rounded bg-sky-100/70 px-1.5 py-0.5 font-mono text-xs text-sky-800">Ctrl+K</kbd> หรือพิมพ์ <kbd className="rounded bg-sky-100/70 px-1.5 py-0.5 font-mono text-xs text-sky-800">/</kbd> เพื่อเปิดคลังคำสั่งทั้งหมด
          </p>
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
                    ? 'border border-sky-300/60 bg-sky-500/10 text-sky-950 shadow-xs'
                    : 'border border-sky-100/70 bg-white/90 text-zinc-800 shadow-[0_4px_20px_rgba(14,165,233,0.04)] backdrop-blur-md'
                }`}
              >
                {/* Citations / Context Tags */}
                {msg.citations && msg.citations.length > 0 && (
                  <div className="mb-2.5 flex flex-wrap items-center gap-1.5">
                    {msg.citations.map((c) => (
                      <span
                        key={c}
                        className="rounded-full border border-sky-100 bg-sky-50/80 px-2.5 py-0.5 text-xs font-medium text-sky-700"
                      >
                        {c}
                      </span>
                    ))}
                  </div>
                )}

                {/* Message Body */}
                <AdvisorMarkdown content={msg.content} onActionClick={onQuickPromptClick} />

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
            <div className="rounded-2xl border border-sky-100/70 bg-white/90 px-6 py-2 shadow-[0_4px_20px_rgba(14,165,233,0.04)] backdrop-blur-md">
              <ManagerFluidOrb size="sm" />
            </div>
          </div>
        )}

        {/* Real-time Streaming Response Turn: 28px Orb Avatar */}
        {isStreaming && streamingText && (
          <div className="flex justify-start gap-3 animate-chat-turn-in">
            <ManagerFluidOrb mode="avatar" />
            <div className="max-w-[85%] sm:max-w-[78%] rounded-2xl border border-sky-100/70 bg-white/90 p-4 text-zinc-800 shadow-[0_4px_20px_rgba(14,165,233,0.04)] backdrop-blur-md">
              <AdvisorMarkdown content={streamingText} isStreaming onActionClick={onQuickPromptClick} />

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
