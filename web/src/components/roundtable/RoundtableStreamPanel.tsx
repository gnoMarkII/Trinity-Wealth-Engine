import { useEffect, useRef, useState } from 'react'
import { motion } from 'motion/react'
import type { GuruPersonaDTO, RoundtableToolCallEventDTO, RoundtableTurnDTO } from '../../api/types'
import {
  getSpeakerMeta,
  parseEvidenceBullet,
  roundStageLabel,
  stanceBadgeClass,
} from './personas'
import type { PlaybackSpeed } from './useRoundtableVoicePlayer'
import {
  CommentThread,
  MessageActions,
  SegmentedControl,
  TextReveal,
  ThinkingLoader,
  ToolCallCard,
  type FeedbackKind,
  type ThreadMessage,
} from './ui'
import {
  MicStageIcon,
  PauseIcon,
  PlayIcon,
  SkipForwardIcon,
  VolumeOffIcon,
  VolumeOnIcon,
} from '../ui/AppIcons'

export interface InterjectionToolRecord {
  afterTurnId: string
  events: RoundtableToolCallEventDTO[]
}

interface RoundtableStreamPanelProps {
  turns: RoundtableTurnDTO[]
  personas?: GuruPersonaDTO[]
  revealTurnIds: Set<string>
  toolRecords: InterjectionToolRecord[]
  isStreaming: boolean
  thinkingSpeakerId: string | null
  thinkingStage: string | null
  activeTurnId: string | null
  voiceEnabled: boolean
  onToggleVoiceEnabled: (enabled: boolean) => void
  isPlaying: boolean
  isPaused: boolean
  isAwaitingTts?: boolean
  queueCount: number
  syncWithVoice?: boolean
  bufferedAheadCount?: number
  totalTurnCount?: number
  onToggleSyncWithVoice?: () => void
  playbackRate: PlaybackSpeed
  onChangePlaybackRate: (rate: PlaybackSpeed) => void
  onPauseOrResume: () => void
  onSkipAudio: () => void
  onPlayTurn: (turn: RoundtableTurnDTO) => void
  onCopyTurn?: (text?: string) => void
  onTurnFeedback?: (turn: RoundtableTurnDTO, kind: FeedbackKind | null, reason?: string) => void
  onQuickThreadReply?: (text: string) => Promise<void>
  onCollapseHud?: () => void
}

export function RoundtableStreamPanel({
  turns,
  personas,
  revealTurnIds,
  toolRecords,
  isStreaming,
  thinkingSpeakerId,
  thinkingStage,
  activeTurnId,
  voiceEnabled,
  onToggleVoiceEnabled,
  isPlaying,
  isPaused,
  isAwaitingTts = false,
  queueCount,
  syncWithVoice = true,
  bufferedAheadCount = 0,
  totalTurnCount,
  onToggleSyncWithVoice,
  playbackRate,
  onChangePlaybackRate,
  onPauseOrResume,
  onSkipAudio,
  onPlayTurn,
  onCopyTurn,
  onTurnFeedback,
  onQuickThreadReply,
  onCollapseHud,
}: RoundtableStreamPanelProps) {
  const [expandedEvidence, setExpandedEvidence] = useState<Set<string>>(() => new Set())
  const [showThreadDrawer, setShowThreadDrawer] = useState(false)
  const endRef = useRef<HTMLDivElement | null>(null)

  useEffect(() => {
    if (activeTurnId) {
      const activeEl = document.getElementById(`rt-turn-${activeTurnId}`)
      if (activeEl && typeof activeEl.scrollIntoView === 'function') {
        activeEl.scrollIntoView({ behavior: 'smooth', block: 'nearest' })
        return
      }
    }
    endRef.current?.scrollIntoView?.({ behavior: 'smooth', block: 'end' })
  }, [turns.length, thinkingSpeakerId, activeTurnId])

  const interjectionThreadMessages: ThreadMessage[] = turns
    .filter(
      (t) =>
        t.round_stage === 'user_interjection' || t.round_stage === 'interjection_reply',
    )
    .map((t) => {
      const m = getSpeakerMeta(t.speaker_id, personas)
      return {
        id: t.turn_id,
        author: {
          name: m.shortName,
          avatar: m.emoji,
          assistant: t.speaker_id !== 'user',
        },
        createdAt: (t.created_at || Date.now() / 1000) * 1000,
        text: t.spoken_dialogue,
        status: t.speaker_id !== 'user' ? `Stance: ${t.stance_tag}` : undefined,
        details: t.evidence_bullets.length ? t.evidence_bullets : undefined,
      }
    })

  const toggleEvidence = (turnId: string) => {
    setExpandedEvidence((prev) => {
      const next = new Set(prev)
      if (next.has(turnId)) next.delete(turnId)
      else next.add(turnId)
      return next
    })
  }

  const thinkingMeta = thinkingSpeakerId ? getSpeakerMeta(thinkingSpeakerId, personas) : null
  const thinkingStageLabel =
    thinkingStage === 'stage_intro'
      ? 'เปิดประเด็น Opening Hook'
      : thinkingStage === 'crossfire'
        ? 'ปะทะเดือด Crossfire ตัวต่อตัว'
        : thinkingStage === 'rebuttal'
          ? 'ซักค้านรอบที่ 2'
          : 'และหลักฐาน'
  const thinkingPhases = thinkingMeta
    ? [
        `${thinkingMeta.shortName} กำลังอ่านข้อมูล ${thinkingStageLabel}...`,
        `${thinkingMeta.shortName} กำลังสรุปจุดยืนและสัดส่วนความเสี่ยง...`,
        `${thinkingMeta.shortName} กำลังเรียบเรียงคำพูด...`,
      ]
    : ['กำลังเตรียมห้องประชุมสภากูรู...']

  return (
    <div className="flex min-h-0 flex-1 flex-col rounded-2xl border border-sky-200/80 bg-white/85 shadow-sm backdrop-blur-md overflow-hidden">
      {/* Voice Transport & Audio Queue Bar */}
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-sky-100 bg-sky-50/50 px-3.5 py-2">
        <div className="flex flex-wrap items-center gap-2">
          <button
            type="button"
            onClick={() => onToggleVoiceEnabled(!voiceEnabled)}
            className={`inline-flex items-center gap-1.5 rounded-lg border px-2.5 py-1 text-xs font-semibold transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-section-accent/45 ${
              voiceEnabled
                ? 'border-sky-300 bg-sky-100/80 text-sky-900'
                : 'border-slate-200 bg-white text-slate-500'
            }`}
          >
            {voiceEnabled ? (
              <VolumeOnIcon className="h-3.5 w-3.5 text-sky-700" />
            ) : (
              <VolumeOffIcon className="h-3.5 w-3.5 text-slate-500" />
            )}
            <span>{voiceEnabled ? 'Voice On' : 'Voice Muted'}</span>
          </button>

          {(isPlaying || isPaused || queueCount > 0) && (
            <div className="inline-flex items-center gap-1 rounded-lg border border-sky-200 bg-white px-2 py-0.5 text-xs">
              <button
                type="button"
                onClick={onPauseOrResume}
                className="inline-flex items-center gap-1 rounded px-1.5 py-0.5 font-semibold text-sky-700 hover:bg-sky-50"
              >
                {isPaused ? (
                  <>
                    <PlayIcon className="h-3 w-3" />
                    <span>เล่นต่อ</span>
                  </>
                ) : (
                  <>
                    <PauseIcon className="h-3 w-3" />
                    <span>พักเสียง</span>
                  </>
                )}
              </button>
              <button
                type="button"
                onClick={onSkipAudio}
                className="inline-flex items-center gap-1 rounded px-1.5 py-0.5 font-medium text-slate-600 hover:bg-slate-100"
                title="ข้ามเสียงพูดเทิร์นนี้"
              >
                <SkipForwardIcon className="h-3 w-3" />
                <span>ข้าม</span>
              </button>
              {queueCount > 0 && (
                <span className="text-[11px] text-slate-500">+{queueCount} ในคิว</span>
              )}
            </div>
          )}

          {isAwaitingTts && (
            <span className="inline-flex items-center gap-1 rounded-lg border border-amber-200 bg-amber-50 px-2 py-0.5 text-[11px] font-medium text-amber-800 animate-pulse">
              ⏳ กำลังสังเคราะห์เสียง Gemini...
            </span>
          )}

          {voiceEnabled && onToggleSyncWithVoice && (
            <button
              type="button"
              aria-pressed={syncWithVoice}
              onClick={onToggleSyncWithVoice}
              title="เปิดเมื่อต้องการให้ข้อความและการ์ดแสดงผลทยอยขึ้นตามจังหวะเสียงพูดจริง"
              className={`inline-flex items-center gap-1 rounded-lg border px-2 py-1 text-[11px] font-semibold transition-colors ${
                syncWithVoice
                  ? 'border-blue-300 bg-blue-50 text-blue-900'
                  : 'border-slate-200 bg-white text-slate-500 hover:bg-slate-50'
              }`}
            >
              <span>🎧</span>
              <span>{syncWithVoice ? 'ซิงค์ตามเสียง' : 'แสดงล่วงหน้า'}</span>
            </button>
          )}
        </div>

        <div className="flex items-center gap-2">
          {turns.length > 0 && (
            <button
              type="button"
              onClick={() => setShowThreadDrawer((s) => !s)}
              className={`rounded-lg border px-2 py-1 text-[11px] font-semibold transition-colors ${
                showThreadDrawer
                  ? 'border-sky-300 bg-sky-100 text-sky-900'
                  : 'border-slate-200 bg-white text-slate-600 hover:bg-slate-50'
              }`}
            >
              💬 Q&A ({interjectionThreadMessages.length})
            </button>
          )}
          <span className="text-[11px] font-medium text-slate-500">ความเร็ว:</span>
          <SegmentedControl<string>
            size="sm"
            aria-label="ความเร็วเสียงพูด"
            value={String(playbackRate)}
            onChange={(val) => onChangePlaybackRate(Number(val) as PlaybackSpeed)}
            options={[
              { value: '1', label: '1x' },
              { value: '1.25', label: '1.25x' },
              { value: '1.5', label: '1.5x' },
            ]}
          />
          {onCollapseHud && (
            <button
              type="button"
              onClick={onCollapseHud}
              title="ย่อกล่องบทสนทนาลงเพื่อดู 3D เต็มจอ"
              className="rounded-lg border border-slate-200 bg-white px-2 py-0.5 text-[11px] font-semibold text-slate-600 hover:bg-slate-50 hover:text-slate-900"
            >
              ▾ ย่อ
            </button>
          )}
        </div>
      </div>

      {showThreadDrawer && (
        <div className="border-b border-sky-100 bg-sky-50/40 p-3">
          <CommentThread
            title="เธรดถาม-ตอบแทรกระหว่างประชุม"
            messages={interjectionThreadMessages}
            currentUser={{ name: 'คุณ', avatar: '✋' }}
            assistant={{ name: 'Hermes', avatar: '🪶', assistant: true }}
            onReply={onQuickThreadReply}
            onClose={() => setShowThreadDrawer(false)}
          />
        </div>
      )}

      {/* Turn-by-Turn Dialogue Feed */}
      <div className="rt-perspective min-h-0 flex-1 overflow-y-auto px-4 py-3.5 space-y-3 custom-scrollbar">
        {turns.length === 0 && !isStreaming && (
          <div className="flex h-full min-h-[210px] flex-col items-center justify-center rounded-2xl border border-dashed border-sky-200/80 bg-gradient-to-b from-sky-50/40 to-white/60 p-6 text-center">
            <div
              aria-hidden="true"
              className="rt-perspective relative mb-3 flex h-16 w-16 items-center justify-center"
            >
              <span className="rt-orbit-ring-primary absolute inset-0 rounded-full border border-sky-400/55" />
              <span className="rt-orbit-ring-secondary absolute inset-1.5 rounded-full border border-cyan-400/45" />
              <span className="rt-crystal-levitate flex h-9 w-9 items-center justify-center rounded-2xl border border-sky-200 bg-white/95 shadow-sm">
                <MicStageIcon className="h-4 w-4 text-section-accent" />
              </span>
            </div>
            <h3 className="text-sm font-bold text-slate-900">พร้อมเปิดห้องประชุมสภากูรู</h3>
            <p className="mt-1 max-w-md text-xs leading-relaxed text-slate-500">
              เลือกบริบทเช่น <code className="rounded bg-white px-1 py-0.5 text-sky-800">$NVDA</code> หรือ{' '}
              <code className="rounded bg-white px-1 py-0.5 text-sky-800">@Portfolio</code> ด้านล่าง — กูรูจะเปิดมุมมอง ซักค้าน และสรุป Playbook อัตโนมัติ
            </p>
          </div>
        )}

        {turns.map((turn, idx) => {
          const prevTurn = idx > 0 ? turns[idx - 1] : null
          const showRoundDivider = !prevTurn || prevTurn.round_stage !== turn.round_stage
          const stageInfo = roundStageLabel(turn.round_stage)
          const meta = getSpeakerMeta(turn.speaker_id, personas)
          const isUser = turn.speaker_id === 'user'
          const isSpeakingThis = activeTurnId === turn.turn_id
          const shouldReveal = revealTurnIds.has(turn.turn_id)
          const evidenceOpen = expandedEvidence.has(turn.turn_id)
          const attachedTools = toolRecords.filter((rec) => rec.afterTurnId === turn.turn_id)

          return (
            <div key={turn.turn_id} className="space-y-2">
              {showRoundDivider && (
                <div className="flex items-center gap-2 py-1">
                  <div className="h-px flex-1 bg-slate-200/70" />
                  <span
                    className={`rounded-full border px-2.5 py-0.5 text-[11px] font-semibold ${stageInfo.toneClass}`}
                  >
                    {stageInfo.label}
                  </span>
                  <div className="h-px flex-1 bg-slate-200/70" />
                </div>
              )}

              <motion.article
                id={`rt-turn-${turn.turn_id}`}
                data-speaker-id={turn.speaker_id}
                data-ui-message=""
                initial={{ opacity: 0, y: 10, rotateX: 3 }}
                animate={{ opacity: 1, y: 0, rotateX: 0 }}
                transition={{ duration: 0.22, ease: 'easeOut' }}
                className={`rt-preserve-3d rounded-2xl border p-3.5 transition-shadow ${
                  isUser
                    ? 'ml-auto max-w-[88%] border-sky-300/80 bg-gradient-to-br from-sky-50 to-cyan-50/70 shadow-sm'
                    : isSpeakingThis
                      ? 'border-sky-400 bg-white shadow-md ring-1 ring-sky-300/50'
                      : 'border-slate-200/75 bg-white/95 shadow-sm hover:border-sky-200'
                }`}
              >
                {/* Minimal Turn Header */}
                <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
                  <div className="flex min-w-0 items-center gap-2">
                    <span
                      className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full border text-sm shadow-sm"
                      style={{
                        borderColor: `${meta.accentColor}88`,
                        backgroundColor: `${meta.accentColor}15`,
                      }}
                    >
                      {meta.emoji}
                    </span>
                    <div className="flex min-w-0 flex-wrap items-center gap-1.5">
                      <span className="text-xs font-bold text-slate-900">{meta.displayName}</span>
                      {!isUser && meta.roleLensLabel && (
                        <span
                          className="rounded-full border px-1.5 py-0 text-[11px] font-semibold"
                          style={{
                            borderColor: `${meta.accentColor}40`,
                            backgroundColor: `${meta.accentColor}10`,
                            color: meta.accentColor,
                          }}
                          title={meta.focusSummary}
                        >
                          {meta.roleLensLabel}
                        </span>
                      )}
                      {turn.replying_to && (
                        <span
                          className={`rounded-md px-1.5 py-0.5 text-[11px] font-semibold ${
                            turn.round_stage === 'crossfire' || turn.round_stage === 'rebuttal'
                              ? 'border border-rose-200 bg-rose-50 text-rose-800'
                              : 'bg-slate-100 text-slate-600'
                          }`}
                        >
                          {turn.round_stage === 'crossfire' || turn.round_stage === 'rebuttal'
                            ? '⚔️ ปะทะ'
                            : '↩'}{' '}
                          @{getSpeakerMeta(turn.replying_to, personas).shortName}
                        </span>
                      )}
                    </div>
                  </div>

                  <div className="flex flex-wrap items-center gap-1.5">
                    {isSpeakingThis && (
                      <span
                        aria-hidden="true"
                        className="inline-flex h-3 items-end gap-[2px] rounded-full bg-sky-50 px-1.5 py-0.5"
                      >
                        <span className="rt-eq-bar h-2 w-[2px] rounded-full bg-sky-600" />
                        <span
                          className="rt-eq-bar h-2.5 w-[2px] rounded-full bg-sky-600"
                          style={{ animationDelay: '140ms' }}
                        />
                        <span
                          className="rt-eq-bar h-1.5 w-[2px] rounded-full bg-sky-600"
                          style={{ animationDelay: '280ms' }}
                        />
                      </span>
                    )}
                    {!isUser && (
                      <span
                        className={`rounded-md border px-2 py-0.5 text-[11px] font-bold tracking-wide ${stanceBadgeClass(
                          turn.stance_tag,
                        )}`}
                      >
                        {turn.stance_tag}
                      </span>
                    )}
                    {!isUser && turn.sizing_posture && (
                      <span className="rounded-md border border-slate-200 bg-slate-50 px-1.5 py-0.5 text-[11px] font-medium text-slate-600">
                        {turn.sizing_posture}
                      </span>
                    )}
                  </div>
                </div>

                {/* Spoken Dialogue */}
                {shouldReveal && !isUser ? (
                  <TextReveal
                    text={turn.spoken_dialogue}
                    by="word"
                    stagger={12}
                    duration={320}
                    once
                    className="text-sm leading-relaxed text-slate-800"
                  />
                ) : (
                  <p dir="auto" className="text-sm leading-relaxed text-slate-800 whitespace-pre-line">
                    {turn.spoken_dialogue}
                  </p>
                )}

                {/* Expandable Evidence Bullets */}
                {!isUser && turn.evidence_bullets.length > 0 && (
                  <div className="mt-2.5 border-t border-slate-100 pt-2">
                    <button
                      type="button"
                      onClick={() => toggleEvidence(turn.turn_id)}
                      className="inline-flex items-center gap-1 text-[11px] font-semibold text-sky-700 hover:text-sky-900"
                    >
                      <span>{evidenceOpen ? '▾' : '▸'}</span>
                      <span>หลักฐานประกอบ ({turn.evidence_bullets.length} ข้อ)</span>
                    </button>
                    {evidenceOpen && (
                      <ul className="mt-1.5 space-y-1">
                        {turn.evidence_bullets.map((b, bIdx) => {
                          const parsed = parseEvidenceBullet(b)
                          return (
                            <li key={bIdx} className="flex items-start gap-1.5 text-xs text-slate-700">
                              <span
                                className={`mt-0.5 shrink-0 rounded border px-1 py-0 text-[11px] font-bold ${parsed.badgeClass}`}
                              >
                                {parsed.tag}
                              </span>
                              <span className="leading-snug">{parsed.text}</span>
                            </li>
                          )
                        })}
                      </ul>
                    )}
                  </div>
                )}

                {/* Message Actions Bar */}
                {!isUser && (
                  <div className="mt-2 border-t border-slate-100 pt-1">
                    <MessageActions
                      text={turn.spoken_dialogue}
                      actions={['copy', 'speak', 'feedback']}
                      speaking={isSpeakingThis}
                      quiet={false}
                      onCopy={onCopyTurn}
                      onSpeak={() => onPlayTurn(turn)}
                      onFeedback={(kind, reason) => onTurnFeedback?.(turn, kind, reason)}
                    />
                  </div>
                )}
              </motion.article>

              {/* Python Sandbox Decimal Tool Call Cards */}
              {attachedTools.map((rec, rIdx) =>
                rec.events.map((tc, tIdx) => (
                  <ToolCallCard
                    key={`${turn.turn_id}-tool-${rIdx}-${tIdx}`}
                    name={tc.tool_name}
                    summary={tc.summary}
                    status={
                      tc.status === 'completed'
                        ? 'success'
                        : tc.status === 'failed'
                          ? 'error'
                          : 'running'
                    }
                    duration={tc.elapsed_ms ?? undefined}
                    result={tc.detail_json ?? undefined}
                    defaultOpen={tc.status === 'completed'}
                  />
                )),
              )}
            </div>
          )
        })}

        {bufferedAheadCount > 0 && (
          <div className="flex flex-wrap items-center justify-between gap-2 rounded-xl border border-sky-200 bg-sky-50/75 px-3.5 py-2.5 text-xs text-sky-900">
            <div className="flex items-center gap-2">
              <span className="animate-pulse">🎧</span>
              <span>
                กำลังดำเนินรายการตามคิวเสียงพูด ({turns.length}
                {totalTurnCount ? `/${totalTurnCount}` : ''} ช่วง · เตรียมบทพากย์ล่วงหน้าอีก +{bufferedAheadCount} ช่วง)
              </span>
            </div>
            {onToggleSyncWithVoice && (
              <button
                type="button"
                onClick={onToggleSyncWithVoice}
                className="rounded-lg border border-sky-300 bg-white px-2.5 py-1 text-[11px] font-semibold text-sky-800 hover:bg-sky-100"
              >
                ⚡ แสดงทั้งหมดทันที
              </button>
            )}
          </div>
        )}

        {isStreaming && thinkingSpeakerId && bufferedAheadCount === 0 && (
          <div className="pt-1">
            <ThinkingLoader
              key={thinkingSpeakerId + (thinkingStage ?? '')}
              phases={thinkingPhases}
              interval={2000}
              size="md"
              variant="pill"
            />
          </div>
        )}

        <div ref={endRef} />
      </div>
    </div>
  )
}
