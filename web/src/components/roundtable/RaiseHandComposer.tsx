import { useMemo, useRef, useState } from 'react'
import type {
  CoreGuruIdType,
  DebateDepthType,
  DebateIntensityType,
  GuruPersonaDTO,
  SpeakerIdType,
} from '../../api/types'
import { getOrderedSeatIds, getSpeakerMeta } from './personas'
import { SlidersIcon } from '../ui/AppIcons'

interface RaiseHandComposerProps {
  mode: 'new' | 'active'
  personas?: GuruPersonaDTO[]
  // New session state
  agendaPrompt: string
  onAgendaPromptChange: (value: string) => void
  contextRefs: string[]
  onToggleContextRef: (refToken: string) => void
  onAddTickerRef: (ticker: string) => void
  debateIntensity?: DebateIntensityType
  onDebateIntensityChange?: (val: DebateIntensityType) => void
  debateDepth?: DebateDepthType
  onDebateDepthChange?: (val: DebateDepthType) => void
  onToggleSeatEnabled?: (speakerId: CoreGuruIdType) => void
  onOpenCustomization?: () => void
  onStartNewSession: () => void
  isCreating?: boolean
  // Active interjection state
  interjectText: string
  onInterjectTextChange: (value: string) => void
  targetSpeaker: SpeakerIdType | null
  onTargetSpeakerChange: (speaker: SpeakerIdType | null) => void
  runSandboxMath: boolean
  onRunSandboxMathChange: (val: boolean) => void
  onSubmitInterjection: () => void
  isInterjecting?: boolean
  isStreaming?: boolean
  canResumeDebate?: boolean
  compactHud?: boolean
  onResumeDebate?: () => void
}

const CONTEXT_PILL_OPTIONS = [
  { token: '@Portfolio', label: '@Portfolio', hint: 'แนบข้อมูลพอร์ตจริงของคุณ' },
  { token: '@Macro', label: '@Macro', hint: 'แนบมุมมองเศรษฐกิจมหภาค' },
  { token: '@Digest', label: '@Digest', hint: 'แนบสรุปข่าวล่าสุด' },
  { token: '@Essence', label: '@Essence', hint: 'แนบแก่นลงทุนและ Role Models ของคุณ' },
]

export function RaiseHandComposer({
  mode,
  personas,
  agendaPrompt,
  onAgendaPromptChange,
  contextRefs,
  onToggleContextRef,
  onAddTickerRef,
  debateIntensity = 'crossfire',
  onDebateIntensityChange,
  debateDepth = 'deep_youtube',
  onDebateDepthChange,
  onToggleSeatEnabled,
  onOpenCustomization,
  onStartNewSession,
  isCreating = false,
  interjectText,
  onInterjectTextChange,
  targetSpeaker,
  onTargetSpeakerChange,
  runSandboxMath,
  onRunSandboxMathChange,
  onSubmitInterjection,
  isInterjecting = false,
  isStreaming = false,
  canResumeDebate = false,
  compactHud = false,
  onResumeDebate,
}: RaiseHandComposerProps) {
  const [tickerInput, setTickerInput] = useState('')
  const [listening, setListening] = useState(false)
  const [hudMinimized, setHudMinimized] = useState(false)
  const recognitionRef = useRef<any>(null)

  const speechSupported = useMemo(() => {
    if (typeof window === 'undefined') return false
    return Boolean((window as any).SpeechRecognition || (window as any).webkitSpeechRecognition)
  }, [])

  const handleMicToggle = () => {
    if (!speechSupported) return
    if (listening && recognitionRef.current) {
      try {
        recognitionRef.current.stop()
      } catch {
        // ignore
      }
      setListening(false)
      return
    }
    const SpeechRec = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition
    if (!SpeechRec) return
    const rec = new SpeechRec()
    rec.lang = 'th-TH'
    rec.interimResults = false
    rec.maxAlternatives = 1
    rec.onresult = (event: any) => {
      const transcript = event?.results?.[0]?.[0]?.transcript
      if (transcript) {
        if (mode === 'new') {
          onAgendaPromptChange((agendaPrompt ? `${agendaPrompt} ` : '') + transcript)
        } else {
          onInterjectTextChange((interjectText ? `${interjectText} ` : '') + transcript)
        }
      }
    }
    rec.onend = () => setListening(false)
    rec.onerror = () => setListening(false)
    recognitionRef.current = rec
    setListening(true)
    try {
      rec.start()
    } catch {
      setListening(false)
    }
  }

  const handleAddTicker = () => {
    const cleaned = tickerInput.trim().toUpperCase().replace(/^\$/, '')
    if (!cleaned) return
    onAddTickerRef(`$${cleaned}`)
    setTickerInput('')
  }

  const tickerRefs = contextRefs.filter((r) => r.startsWith('$'))
  const hasPortfolio = contextRefs.some((r) => r.toLowerCase() === '@portfolio')

  const allPanelSeatIds = useMemo(
    () => getOrderedSeatIds(personas, true).filter((s) => s !== 'hermes_moderator'),
    [personas],
  )
  const enabledPanelSeatIds = useMemo(
    () => getOrderedSeatIds(personas, false).filter((s) => s !== 'hermes_moderator'),
    [personas],
  )
  const disabledPanelNames = useMemo(
    () =>
      allPanelSeatIds
        .filter((sid) => !enabledPanelSeatIds.includes(sid))
        .map((sid) => getSpeakerMeta(sid, personas).shortName),
    [allPanelSeatIds, enabledPanelSeatIds, personas],
  )

  if (compactHud && hudMinimized) {
    return (
      <div className="flex items-center justify-between gap-2 rounded-2xl border border-sky-200/85 bg-white/88 px-3 py-1.5 shadow-xl backdrop-blur-xl">
        <button
          type="button"
          onClick={() => setHudMinimized(false)}
          className="flex min-w-0 flex-1 items-center gap-2 text-left text-xs font-semibold text-slate-800 hover:text-sky-800"
        >
          <span>{mode === 'new' ? '📝 ตั้งวาระเปิดประชุมสภากูรูใหม่' : '✋ ยกมือแทรกถามสภากูรู'}</span>
          {mode === 'new' && agendaPrompt.trim() && (
            <span className="truncate text-[11px] font-normal text-slate-500">
              — {agendaPrompt.trim()}
            </span>
          )}
        </button>
        <div className="flex shrink-0 items-center gap-1.5">
          {mode === 'new' ? (
            <button
              type="button"
              disabled={!agendaPrompt.trim() || isCreating}
              onClick={onStartNewSession}
              className="rounded-xl bg-gradient-to-r from-sky-600 to-cyan-600 px-3 py-1 text-[11px] font-bold text-white shadow-sm disabled:opacity-50"
            >
              {isCreating ? 'กำลังเปิด...' : '🎙️ เริ่มประชุม'}
            </button>
          ) : (
            canResumeDebate &&
            !isStreaming &&
            onResumeDebate && (
              <button
                type="button"
                onClick={onResumeDebate}
                className="rounded-xl border border-sky-300 bg-sky-50 px-2.5 py-1 text-[11px] font-bold text-sky-900"
              >
                ▶️ ประชุมต่อ
              </button>
            )
          )}
          <button
            type="button"
            onClick={() => setHudMinimized(false)}
            className="rounded-lg border border-slate-200 bg-white px-2 py-1 text-[11px] font-semibold text-slate-700 hover:bg-sky-50"
          >
            ▴ กางแผง
          </button>
        </div>
      </div>
    )
  }

  if (mode === 'new') {
    return (
      <section
        aria-label="ตั้งวาระเปิดประชุมสภากูรูใหม่"
        className={`rounded-2xl border backdrop-blur-xl ${
          compactHud
            ? 'space-y-1.5 border-sky-200/85 bg-white/88 px-3 py-2 shadow-xl'
            : 'space-y-2.5 border-sky-200/80 bg-white/85 p-3.5 shadow-sm'
        }`}
      >
        <div className="flex flex-wrap items-center justify-between gap-2">
          <span className="text-xs font-bold tracking-wide text-slate-800">
            📝 ตั้งวาระเปิดประชุมสภากูรูใหม่
          </span>
          <div className="flex items-center gap-1.5">
            <span
              className={`rounded-full border px-2 py-0.5 text-[11px] font-semibold ${
                hasPortfolio
                  ? 'border-emerald-200 bg-emerald-50/90 text-emerald-800'
                  : 'border-slate-200 bg-slate-50/90 text-slate-500'
              }`}
            >
              {hasPortfolio ? '✓ แนบพอร์ตจริงแล้ว' : 'Neutral · ไม่ได้แนบพอร์ต'}
            </span>
            {compactHud && (
              <button
                type="button"
                onClick={() => setHudMinimized(true)}
                className="rounded-lg border border-slate-200 bg-white/90 px-2 py-0.5 text-[11px] font-semibold text-slate-600 hover:bg-slate-50 hover:text-slate-900"
              >
                ▾ ย่อแผง
              </button>
            )}
          </div>
        </div>

        {/* Context Ref Chips + $Ticker Adder */}
        <div className="flex flex-wrap items-center gap-1.5">
          {CONTEXT_PILL_OPTIONS.map((pill) => {
            const active = contextRefs.some(
              (r) => r.toLowerCase() === pill.token.toLowerCase(),
            )
            return (
              <button
                key={pill.token}
                type="button"
                title={pill.hint}
                onClick={() => onToggleContextRef(pill.token)}
                className={`rounded-lg border px-2 py-1 text-[11.5px] font-semibold transition-all ${
                  active
                    ? 'border-sky-400 bg-sky-100/90 text-sky-900 shadow-sm'
                    : 'border-slate-200/90 bg-white text-slate-600 hover:border-sky-200 hover:bg-sky-50/50'
                }`}
              >
                {active ? '✓ ' : '+ '}
                {pill.label}
              </button>
            )
          })}

          {tickerRefs.map((tok) => (
            <button
              key={tok}
              type="button"
              onClick={() => onToggleContextRef(tok)}
              title="คลิกเพื่อเอาออก"
              className="rounded-lg border border-amber-300 bg-amber-50 px-2 py-1 text-[11.5px] font-bold text-amber-900 hover:bg-amber-100"
            >
              {tok} ×
            </button>
          ))}

          <div className="inline-flex items-center gap-1 rounded-lg border border-slate-200/90 bg-slate-50/75 px-1.5 py-0.5">
            <input
              type="text"
              value={tickerInput}
              onChange={(e) => setTickerInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter') {
                  e.preventDefault()
                  handleAddTicker()
                }
              }}
              placeholder="เพิ่มหุ้น $NVDA..."
              aria-label="เพิ่มสัญลักษณ์หุ้น"
              className="w-24 rounded bg-transparent px-1 py-0.5 text-xs text-slate-800 focus:outline-none focus-visible:ring-2 focus-visible:ring-section-accent/45 placeholder:text-slate-400"
            />
            <button
              type="button"
              onClick={handleAddTicker}
              className="rounded-md bg-white px-2 py-0.5 text-[11px] font-semibold text-sky-700 shadow-sm hover:bg-sky-50"
            >
              + แนบหุ้น
            </button>
          </div>
        </div>

        {/* Compact Panel Lineup + Debate Mode Row */}
        {(onToggleSeatEnabled || onDebateIntensityChange || onDebateDepthChange) && (
          <div className="flex flex-wrap items-center justify-between gap-2 border-t border-slate-100 pt-2">
            {onToggleSeatEnabled && (
              <div className="flex flex-wrap items-center gap-1">
                <span className="mr-0.5 text-[11px] font-semibold text-slate-500">
                  ผู้ร่วมวง ({enabledPanelSeatIds.length}/5):
                </span>
                {allPanelSeatIds.map((sid) => {
                  const meta = getSpeakerMeta(sid, personas)
                  const active = meta.enabled
                  const lockMinimum = active && enabledPanelSeatIds.length <= 2
                  return (
                    <button
                      key={sid}
                      type="button"
                      disabled={lockMinimum}
                      aria-pressed={active}
                      onClick={() => onToggleSeatEnabled(sid as CoreGuruIdType)}
                      title={
                        lockMinimum
                          ? `ต้องเปิดใช้งานกูรูอย่างน้อย 2 ท่าน (${meta.focusSummary})`
                          : `${meta.displayName} · ${meta.roleLensLabel}`
                      }
                      className={`rounded-full border px-2 py-0.5 text-[11px] font-semibold transition-all disabled:cursor-not-allowed ${
                        active
                          ? 'border-sky-300 bg-sky-50/90 text-sky-900 hover:bg-sky-100'
                          : 'border-slate-200 bg-slate-100/80 text-slate-500 line-through hover:text-slate-600'
                      }`}
                    >
                      {meta.emoji} {meta.shortName}
                    </button>
                  )
                })}
                {onOpenCustomization && (
                  <button
                    type="button"
                    onClick={onOpenCustomization}
                    className="inline-flex items-center gap-1 rounded-full border border-slate-200 bg-white px-2 py-0.5 text-[11px] font-semibold text-sky-700 hover:bg-sky-50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-section-accent/45"
                  >
                    <SlidersIcon className="h-3 w-3" />
                    <span>เสียง</span>
                  </button>
                )}
                {disabledPanelNames.length > 0 && (
                  <span className="text-[11px] font-medium text-amber-700">
                    ⚠️ ปิด {disabledPanelNames.join(', ')}
                  </span>
                )}
              </div>
            )}

            {(onDebateIntensityChange || onDebateDepthChange) && (
              <div className="flex flex-wrap items-center gap-2">
                {onDebateIntensityChange && (
                  <div className="flex items-center gap-1" role="group" aria-label="ระดับความดุเดือด">
                    {(
                      [
                        { id: 'balanced', label: '⚖️ สมดุล', hint: 'ถกสมดุล' },
                        { id: 'heated', label: '🔥 เข้มข้น', hint: 'เข้มข้น 1 คู่ดวล' },
                        { id: 'crossfire', label: '⚔️ ปะทะเดือด', hint: 'ปะทะเดือด 2 คู่ดวล' },
                      ] as const
                    ).map((opt) => (
                      <button
                        key={opt.id}
                        type="button"
                        title={opt.hint}
                        aria-pressed={debateIntensity === opt.id}
                        onClick={() => onDebateIntensityChange(opt.id)}
                        className={`rounded-lg border px-2 py-0.5 text-[11px] font-semibold transition-all ${
                          debateIntensity === opt.id
                            ? 'border-rose-300 bg-rose-50 text-rose-900 shadow-sm'
                            : 'border-slate-200 bg-white text-slate-600 hover:bg-slate-50'
                        }`}
                      >
                        {opt.label}
                      </button>
                    ))}
                  </div>
                )}

                {onDebateDepthChange && (
                  <div className="flex items-center gap-1" role="group" aria-label="ความยาวและความลึก">
                    {(
                      [
                        { id: 'concise', label: '⚡ กระชับ' },
                        { id: 'deep_youtube', label: '🎬 YouTube ยาวสุด' },
                      ] as const
                    ).map((opt) => (
                      <button
                        key={opt.id}
                        type="button"
                        aria-pressed={debateDepth === opt.id}
                        onClick={() => onDebateDepthChange(opt.id)}
                        className={`rounded-lg border px-2 py-0.5 text-[11px] font-semibold transition-all ${
                          debateDepth === opt.id
                            ? 'border-amber-300 bg-amber-50 text-amber-900 shadow-sm'
                            : 'border-slate-200 bg-white text-slate-600 hover:bg-slate-50'
                        }`}
                      >
                        {opt.label}
                      </button>
                    ))}
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        {/* Agenda Input & Launch */}
        <div className="flex flex-col gap-2 sm:flex-row sm:items-end">
          <textarea
            rows={compactHud ? 1 : 2}
            value={agendaPrompt}
            onChange={(e) => onAgendaPromptChange(e.target.value)}
            placeholder="พิมพ์โจทย์ให้สภากูรูถกเถียง เช่น NVDA ราคานี้ยังมี Margin of Safety ไหม หรือควรรอจังหวะย่อ..."
            className={`flex-1 resize-none rounded-xl border border-sky-200 bg-white px-3 text-sm text-slate-900 outline-none focus:border-sky-400 focus:ring-2 focus:ring-sky-100 ${
              compactHud ? 'min-h-[38px] py-1.5' : 'min-h-[56px] py-2'
            }`}
          />
          <div className="flex items-center gap-2 shrink-0">
            {speechSupported && (
              <button
                type="button"
                onClick={handleMicToggle}
                className={`rounded-xl border px-3 py-2.5 text-xs font-semibold transition-colors ${
                  listening
                    ? 'border-rose-300 bg-rose-50 text-rose-700 animate-pulse'
                    : 'border-slate-200 bg-white text-slate-700 hover:bg-slate-50'
                }`}
                title="ใช้ไมโครโฟนพูดตั้งวาระ"
              >
                {listening ? '🔴 กำลังฟัง...' : '🎤 พูด'}
              </button>
            )}
            <button
              type="button"
              disabled={!agendaPrompt.trim() || isCreating}
              onClick={onStartNewSession}
              className="inline-flex items-center gap-1.5 rounded-xl bg-gradient-to-r from-sky-600 to-cyan-600 px-4 py-2.5 text-xs font-bold text-white shadow-sm transition-all hover:from-sky-700 hover:to-cyan-700 disabled:cursor-not-allowed disabled:opacity-50"
            >
              <span>🎙️</span>
              <span>{isCreating ? 'กำลังเปิดห้อง...' : 'เริ่มประชุมกูรู'}</span>
            </button>
          </div>
        </div>
      </section>
    )
  }

  return (
    <section
      className={`rounded-2xl border backdrop-blur-xl ${
        compactHud
          ? 'space-y-1.5 border-sky-200/85 bg-white/90 px-3 py-2 shadow-xl'
          : 'space-y-2 border-sky-200/90 bg-white/95 p-3 shadow-sm'
      }`}
    >
      {/* Mention Target Picker + Math Toggle */}
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex flex-wrap items-center gap-1">
          <span className="text-[11px] font-bold text-slate-500 mr-1">ถามระบุคน:</span>
          <button
            type="button"
            onClick={() => onTargetSpeakerChange(null)}
            className={`rounded-full border px-2 py-0.5 text-[11px] font-semibold transition-colors ${
              targetSpeaker === null
                ? 'border-sky-400 bg-sky-100 text-sky-900'
                : 'border-slate-200 bg-white text-slate-600 hover:bg-slate-50'
            }`}
          >
            @{getSpeakerMeta('hermes_moderator', personas).shortName} (สรุป/กลาง)
          </button>
          {enabledPanelSeatIds.map((sid) => {
            const meta = getSpeakerMeta(sid, personas)
            const active = targetSpeaker === sid
            return (
              <button
                key={sid}
                type="button"
                onClick={() => onTargetSpeakerChange(sid as SpeakerIdType)}
                className={`rounded-full border px-2 py-0.5 text-[11px] font-semibold transition-colors ${
                  active
                    ? 'border-sky-400 bg-sky-100 text-sky-900'
                    : 'border-slate-200 bg-white text-slate-600 hover:bg-slate-50'
                }`}
              >
                {meta.emoji} @{meta.shortName}
              </button>
            )
          })}
        </div>

        <div className="flex items-center gap-1.5">
          <label className="inline-flex cursor-pointer items-center gap-1.5 rounded-lg border border-sky-200 bg-sky-50/70 px-2 py-0.5 text-[11px] font-semibold text-sky-900">
            <input
              type="checkbox"
              checked={runSandboxMath}
              onChange={(e) => onRunSandboxMathChange(e.target.checked)}
              className="accent-sky-600"
            />
            <span>🧮 คำนวณ Decimal (What-If)</span>
          </label>
          {compactHud && (
            <button
              type="button"
              onClick={() => setHudMinimized(true)}
              className="rounded-lg border border-slate-200 bg-white/90 px-2 py-0.5 text-[11px] font-semibold text-slate-600 hover:bg-slate-50 hover:text-slate-900"
            >
              ▾ ย่อแผง
            </button>
          )}
        </div>
      </div>

      {/* Interject Composer Row */}
      <div className="flex flex-col gap-2 sm:flex-row sm:items-center">
        <input
          type="text"
          value={interjectText}
          onChange={(e) => onInterjectTextChange(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault()
              if (interjectText.trim() && !isInterjecting) onSubmitInterjection()
            }
          }}
          placeholder={
            isStreaming
              ? '✋ ยกมือแทรกกลางวง (หยุดเทิร์นถัดไปแล้วตอบคำถามคุณทันที)...'
              : 'ถามแทรกเพิ่มเติม หรือลองจำลองตัวเลข เช่น ถ้าซื้อเพิ่ม 10000 บาทที่ราคาลง 15% ต้นทุนเฉลี่ยเป็นเท่าไร...'
          }
          aria-label="ยกมือแทรกถามสภากูรู"
          className="h-10 flex-1 rounded-xl border border-sky-200 bg-white px-3 text-sm text-slate-900 outline-none focus:border-sky-400 focus:ring-2 focus:ring-sky-100"
        />

        <div className="flex items-center gap-1.5 shrink-0">
          {speechSupported && (
            <button
              type="button"
              onClick={handleMicToggle}
              className={`h-10 rounded-xl border px-2.5 text-xs font-semibold transition-colors ${
                listening
                  ? 'border-rose-300 bg-rose-50 text-rose-700 animate-pulse'
                  : 'border-slate-200 bg-white text-slate-700 hover:bg-slate-50'
              }`}
              title="พูดแทรกด้วยเสียง"
            >
              {listening ? '🔴 ฟัง...' : '🎤'}
            </button>
          )}

          <button
            type="button"
            disabled={!interjectText.trim() || isInterjecting}
            onClick={onSubmitInterjection}
            className="inline-flex h-10 items-center gap-1.5 rounded-xl bg-amber-500 px-3.5 text-xs font-bold text-white shadow-sm transition-colors hover:bg-amber-600 disabled:cursor-not-allowed disabled:opacity-50"
          >
            <span>✋</span>
            <span>{isInterjecting ? 'กำลังส่ง...' : isStreaming ? 'ยกมือแทรกทันที' : 'ส่งคำถาม'}</span>
          </button>

          {canResumeDebate && !isStreaming && onResumeDebate && (
            <button
              type="button"
              onClick={onResumeDebate}
              className="inline-flex h-10 items-center gap-1.5 rounded-xl border border-sky-300 bg-sky-50 px-3 text-xs font-bold text-sky-900 hover:bg-sky-100"
            >
              <span>▶️</span>
              <span>ประชุมต่อ</span>
            </button>
          )}
        </div>
      </div>
    </section>
  )
}

