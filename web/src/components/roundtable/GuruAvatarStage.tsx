import { Suspense, lazy, useState } from 'react'
import { motion } from 'motion/react'
import type {
  DebatePillarCoverageDTO,
  GuruPersonaDTO,
  RoundtableStanceType,
  RoundtableTurnDTO,
} from '../../api/types'
import {
  getOrderedSeatIds,
  getSpeakerMeta,
  parseAnalyticalRubricLines,
  portfolioAttachmentLabel,
  stanceBadgeClass,
} from './personas'
import {
  ChevronDownIcon,
  CompassIcon,
  InfoCircleIcon,
  MicStageIcon,
  SlidersIcon,
  VolumeOffIcon,
  VolumeOnIcon,
} from '../ui/AppIcons'

const RoundtableRobloxStage3D = lazy(() => import('./RoundtableRobloxStage3D'))
const STAGE_VIEW_STORAGE_KEY = 'trinity_rt_stage_view_mode'

export type StageViewMode = '3d' | '2d'

interface GuruAvatarStageProps {
  personas?: GuruPersonaDTO[]
  turns: RoundtableTurnDTO[]
  prioritizedSpeakers: string[]
  userRoleModels?: string[]
  activeSpeakerId: string | null
  thinkingSpeakerId: string | null
  thinkingStage?: string | null
  mutedSpeakers: Set<string>
  primaryTicker?: string | null
  portfolioAttachmentState?: string
  finalRating?: RoundtableStanceType | string | null
  pillarCoverage?: DebatePillarCoverageDTO | null
  hasRecentSandboxMath?: boolean
  voiceEnabled?: boolean
  getVoiceEnergy?: () => number
  stageViewMode?: StageViewMode
  hudLayout?: boolean
  onChangeStageViewMode?: (mode: StageViewMode) => void
  onToggleVoiceEnabled?: (next: boolean) => void
  onToggleMute: (speakerId: string) => void
  onMentionSpeaker?: (speakerId: string) => void
  onCustomizeSpeaker?: (speakerId: string) => void
}

export function GuruAvatarStage({
  personas,
  turns,
  prioritizedSpeakers,
  userRoleModels = [],
  activeSpeakerId,
  thinkingSpeakerId,
  thinkingStage,
  mutedSpeakers,
  primaryTicker,
  portfolioAttachmentState,
  finalRating,
  pillarCoverage,
  hasRecentSandboxMath = false,
  voiceEnabled = true,
  getVoiceEnergy,
  stageViewMode: controlledViewMode,
  hudLayout = false,
  onChangeStageViewMode,
  onToggleVoiceEnabled,
  onToggleMute,
  onMentionSpeaker,
  onCustomizeSpeaker,
}: GuruAvatarStageProps) {
  const [showContextGuide, setShowContextGuide] = useState(false)
  const [guideSeatId, setGuideSeatId] = useState<string>('all')
  const [is3DExpanded, setIs3DExpanded] = useState(false)
  const [hudSeatDockOpen, setHudSeatDockOpen] = useState(true)
  const [internalViewMode, setInternalViewMode] = useState<StageViewMode>(() => {
    try {
      const saved = window.localStorage.getItem(STAGE_VIEW_STORAGE_KEY)
      return saved === '2d' ? '2d' : '3d'
    } catch {
      return '3d'
    }
  })
  const stageViewMode = controlledViewMode ?? internalViewMode

  const handleSetViewMode = (mode: StageViewMode) => {
    setInternalViewMode(mode)
    if (mode === '2d') setIs3DExpanded(false)
    try {
      window.localStorage.setItem(STAGE_VIEW_STORAGE_KEY, mode)
    } catch {
      // Ignore storage failures in restricted contexts
    }
    onChangeStageViewMode?.(mode)
  }

  const latestStanceBySpeaker: Record<string, RoundtableStanceType> = {}
  for (const t of turns) {
    if (t.speaker_id !== 'user' && t.stance_tag) {
      latestStanceBySpeaker[t.speaker_id] = t.stance_tag
    }
  }

  const orderedSeatIds = getOrderedSeatIds(personas, true)
  const activeSeatCount = getOrderedSeatIds(personas, false).length
  const visibleGuideSeatIds =
    guideSeatId !== 'all' && orderedSeatIds.includes(guideSeatId)
      ? [guideSeatId]
      : orderedSeatIds

  const spotlightSpeakerId = activeSpeakerId || thinkingSpeakerId
  const spotlightMeta = spotlightSpeakerId ? getSpeakerMeta(spotlightSpeakerId, personas) : null

  const renderSeatActions = (
    seatId: string,
    meta: ReturnType<typeof getSpeakerMeta>,
    isMuted: boolean,
    isDisabled: boolean,
    alwaysVisible = false,
    darkGlass = false,
  ) => (
    <div
      className={`flex shrink-0 items-center gap-0.5 rounded-lg px-0.5 py-0.5 transition-opacity ${
        darkGlass ? 'bg-white/10' : 'bg-slate-50/85'
      } ${
        alwaysVisible || (showContextGuide && guideSeatId === seatId) || isMuted
          ? 'opacity-100'
          : 'opacity-0 group-hover:opacity-100 focus-within:opacity-100'
      }`}
    >
      <button
        type="button"
        onClick={() => {
          if (showContextGuide && guideSeatId === seatId) {
            setShowContextGuide(false)
          } else {
            setGuideSeatId(seatId)
            setShowContextGuide(true)
          }
        }}
        aria-label={`ดูบริบท ${meta.shortName}`}
        title={`ดูกรอบคิดและเกณฑ์วิเคราะห์ของ ${meta.displayName}`}
        className={`rounded-md p-1 transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-500/45 ${
          showContextGuide && guideSeatId === seatId
            ? 'bg-sky-500/25 text-sky-300'
            : darkGlass
              ? 'text-slate-200 hover:bg-white/20 hover:text-white'
              : 'text-slate-500 hover:bg-white hover:text-sky-700'
        }`}
      >
        <InfoCircleIcon className="h-3 w-3" />
      </button>
      {onCustomizeSpeaker && (
        <button
          type="button"
          onClick={() => onCustomizeSpeaker(seatId)}
          aria-label={`ปรับแต่ง ${meta.shortName}`}
          title={`ปรับแต่ง ${meta.shortName} และเลือกเสียง (${meta.voiceName})`}
          className={`rounded-md p-1 transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-500/45 ${
            darkGlass
              ? 'text-slate-200 hover:bg-white/20 hover:text-white'
              : 'text-slate-500 hover:bg-white hover:text-sky-700'
          }`}
        >
          <SlidersIcon className="h-3 w-3" />
        </button>
      )}
      <button
        type="button"
        disabled={isDisabled}
        onClick={() => onToggleMute(seatId)}
        aria-label={isMuted ? `เปิดเสียง ${meta.shortName}` : `ปิดเสียง ${meta.shortName}`}
        title={isMuted ? 'ปิดเสียงอยู่ — คลิกเพื่อเปิดเสียง' : 'คลิกเพื่อปิดเสียงชั่วคราว'}
        className={`rounded-md p-1 transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-500/45 disabled:opacity-40 ${
          isMuted
            ? darkGlass
              ? 'bg-rose-500/30 text-rose-200 hover:bg-rose-500/40'
              : 'bg-rose-50 text-rose-600 hover:bg-rose-100'
            : darkGlass
              ? 'text-slate-200 hover:bg-white/20 hover:text-white'
              : 'text-slate-500 hover:bg-white hover:text-slate-700'
        }`}
      >
        {isMuted ? (
          <VolumeOffIcon className="h-3 w-3" />
        ) : (
          <VolumeOnIcon className="h-3 w-3" />
        )}
      </button>
    </div>
  )

  const isHudFullBleed = hudLayout && stageViewMode === '3d'

  const renderCompactSeatGrid = (darkGlass: boolean) => (
    <div className="grid grid-cols-2 gap-1.5 sm:grid-cols-3 lg:grid-cols-6">
      {orderedSeatIds.map((seatId) => {
        const meta = getSpeakerMeta(seatId, personas)
        const isSpeaking = activeSpeakerId === seatId
        const isThinking = thinkingSpeakerId === seatId
        const isMuted = mutedSpeakers.has(seatId)
        const isDisabled = !meta.enabled
        const latestStance = latestStanceBySpeaker[seatId]

        return (
          <div
            key={seatId}
            className={`group flex min-w-0 flex-col justify-between gap-0.5 rounded-xl border px-2 py-1 transition-colors ${
              darkGlass
                ? isDisabled
                  ? 'border-dashed border-white/15 bg-slate-900/45 opacity-60 backdrop-blur-md'
                  : isSpeaking
                    ? 'border-sky-400 bg-slate-900/80 shadow-md ring-1 ring-sky-400/60 backdrop-blur-md'
                    : isThinking
                      ? 'border-cyan-400/60 bg-slate-900/75 backdrop-blur-md'
                      : 'border-white/15 bg-slate-900/60 hover:border-sky-300/60 hover:bg-slate-900/75 backdrop-blur-md'
                : isDisabled
                  ? 'border-dashed border-slate-200 bg-slate-50/70 opacity-60'
                  : isSpeaking
                    ? 'border-sky-400 bg-sky-50/90 shadow-sm ring-1 ring-sky-300/60'
                    : isThinking
                      ? 'border-blue-300 bg-blue-50/75'
                      : 'border-slate-200/80 bg-white/90 hover:border-sky-300/90'
            }`}
          >
            <div className="flex items-center justify-between gap-1">
              <button
                type="button"
                disabled={isDisabled}
                onClick={() => {
                  if (!isDisabled) onMentionSpeaker?.(seatId)
                }}
                title={meta.focusSummary}
                className="flex min-w-0 flex-1 items-center gap-1 rounded text-left focus:outline-none focus-visible:ring-2 focus-visible:ring-sky-500/45 disabled:cursor-default"
              >
                <span aria-hidden="true" className="shrink-0 text-xs">
                  {meta.emoji}
                </span>
                <span
                  className={`truncate text-[11px] font-bold ${
                    darkGlass ? 'text-white' : 'text-slate-900'
                  }`}
                >
                  {meta.shortName}
                </span>
              </button>
              {renderSeatActions(seatId, meta, isMuted, isDisabled, true, darkGlass)}
            </div>

            <div className="flex flex-wrap items-center justify-between gap-1">
              <button
                type="button"
                disabled={isDisabled}
                onClick={() => {
                  if (!isDisabled) onMentionSpeaker?.(seatId)
                }}
                title={meta.focusSummary}
                className={`min-w-0 flex-1 truncate text-left text-[11px] font-medium focus:outline-none focus-visible:ring-2 focus-visible:ring-sky-500/45 disabled:cursor-default ${
                  darkGlass ? 'text-sky-200' : ''
                }`}
                style={darkGlass ? undefined : { color: meta.accentColor }}
              >
                {meta.roleLensLabel}
              </button>
              {latestStance && !isDisabled && (
                <span
                  className={`shrink-0 rounded border px-1 py-0 text-[11px] font-bold ${stanceBadgeClass(
                    latestStance,
                  )}`}
                >
                  {latestStance}
                </span>
              )}
            </div>
            {meta.presetId && meta.presetId !== 'default' && (
              <span
                className={`truncate rounded px-1 py-0 text-[11px] font-semibold ${
                  darkGlass ? 'bg-sky-500/25 text-sky-200' : 'bg-sky-50 text-sky-700'
                }`}
              >
                {meta.presetLabel || meta.presetId}
              </span>
            )}
          </div>
        )
      })}
    </div>
  )

  return (
    <section
      aria-label="เวทีสภากูรูการลงทุน"
      className={
        isHudFullBleed
          ? 'relative h-full w-full overflow-hidden'
          : 'relative overflow-hidden rounded-2xl border border-sky-200/75 bg-white/80 p-3 shadow-sm backdrop-blur-md'
      }
    >
      {/* Standard Card Header (hidden in 3D Full-Bleed HUD since Studio Bar floats cleanly at top) */}
      {!isHudFullBleed && (
      <div className="mb-2.5 flex flex-wrap items-center justify-between gap-2">
        <div className="flex min-w-0 items-center gap-2.5">
          {/* 3D Orbital Resonance Orb */}
          <div
            aria-hidden="true"
            className="rt-perspective relative flex h-8 w-8 shrink-0 items-center justify-center"
          >
            <span
              className="rt-orbit-ring-primary absolute inset-0 rounded-full border border-sky-400/55"
              style={{
                borderColor: spotlightMeta ? `${spotlightMeta.accentColor}88` : undefined,
                animationDuration: spotlightMeta ? '6s' : '16s',
              }}
            />
            <span
              className="rt-orbit-ring-secondary absolute inset-0.5 rounded-full border border-cyan-400/40"
              style={{
                animationDuration: spotlightMeta ? '8s' : '22s',
              }}
            />
            <span
              className={`relative flex h-4 w-4 items-center justify-center rounded-full shadow-sm transition-all duration-300 ${
                activeSpeakerId ? 'scale-110' : 'scale-100'
              }`}
              style={{
                background: spotlightMeta
                  ? `radial-gradient(circle at 30% 30%, #ffffff, ${spotlightMeta.accentColor})`
                  : 'radial-gradient(circle at 30% 30%, #e0f2fe, #0284c7)',
              }}
            >
              <span className="text-[11px] leading-none inline-flex items-center">
                {spotlightMeta ? (
                  spotlightMeta.emoji
                ) : (
                  <MicStageIcon className="h-3 w-3 text-sky-600" />
                )}
              </span>
            </span>
          </div>

          <div className="min-w-0 flex flex-wrap items-center gap-2">
            <span className="text-xs font-bold tracking-wide text-slate-800">
              ที่นั่งสภากูรู ({activeSeatCount} เสียง)
            </span>
            {spotlightMeta && (
              <span
                className="inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-[11px] font-semibold"
                style={{
                  borderColor: `${spotlightMeta.accentColor}50`,
                  backgroundColor: `${spotlightMeta.accentColor}12`,
                  color: spotlightMeta.accentColor,
                }}
              >
                <span className="h-1.5 w-1.5 rounded-full bg-current animate-pulse" />
                {activeSpeakerId
                  ? `${spotlightMeta.shortName} กำลังพูด`
                  : `${spotlightMeta.shortName} กำลังวิเคราะห์`}
              </span>
            )}
            {userRoleModels.length > 0 && (
              <span className="rounded-full border border-amber-200 bg-amber-50/90 px-2 py-0.5 text-[11px] font-semibold text-amber-800">
                ★ Role Models จาก Essence: {userRoleModels.join(', ')}
              </span>
            )}
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-1.5">
          {/* Segmented Switcher: 🎮 3D Roblox vs 🪪 ปกติ (2D) */}
          <div
            role="group"
            aria-label="สลับมุมมองเวทีสภากูรู"
            className="inline-flex items-center rounded-xl border border-sky-200/80 bg-sky-50/75 p-0.5"
          >
            <button
              type="button"
              aria-pressed={stageViewMode === '3d'}
              onClick={() => handleSetViewMode('3d')}
              className={`rounded-lg px-2 py-0.5 text-[11px] font-semibold transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-500/45 ${
                stageViewMode === '3d'
                  ? 'bg-sky-600 text-white shadow-sm'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              🎮 3D Roblox
            </button>
            <button
              type="button"
              aria-pressed={stageViewMode === '2d'}
              onClick={() => handleSetViewMode('2d')}
              className={`rounded-lg px-2 py-0.5 text-[11px] font-semibold transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-500/45 ${
                stageViewMode === '2d'
                  ? 'bg-sky-600 text-white shadow-sm'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              🪪 ปกติ (2D)
            </button>
          </div>

          <button
            type="button"
            aria-expanded={showContextGuide}
            onClick={() => {
              setShowContextGuide((prev) => !prev)
              setGuideSeatId('all')
            }}
            className={`rounded-xl border px-2.5 py-1 text-[11px] font-semibold transition-colors ${
              showContextGuide
                ? 'border-sky-400 bg-sky-100 text-sky-900'
                : 'border-slate-200/90 bg-white/90 text-slate-700 hover:border-sky-300 hover:bg-sky-50 hover:text-sky-900'
            }`}
          >
            {showContextGuide ? (
              <span className="inline-flex items-center gap-1">
                <ChevronDownIcon className="h-3.5 w-3.5 rotate-180 text-sky-600 transition-transform" />
                <span>ซ่อนกรอบคิดกูรู</span>
              </span>
            ) : (
              <span className="inline-flex items-center gap-1.5">
                <CompassIcon className="h-3.5 w-3.5 text-sky-600" />
                <span>ดูกรอบคิดกูรู 6 สาย</span>
              </span>
            )}
          </button>
        </div>
      </div>
      )}

      {/* Mode 1: 3D Roblox WebGL Room + Compact Seat Control Strip */}
      {stageViewMode === '3d' ? (
        <div className={isHudFullBleed ? 'relative h-full w-full' : 'space-y-2'}>
          <Suspense
            fallback={
              <div
                className={`flex items-center justify-center bg-[#122638] text-xs font-semibold text-sky-200 ${
                  isHudFullBleed
                    ? 'h-full w-full'
                    : 'h-[268px] rounded-2xl border border-slate-700/70 sm:h-[296px]'
                }`}
              >
                กำลังโหลดฉาก 3D Roblox Investment War Room...
              </div>
            }
          >
            <RoundtableRobloxStage3D
              personas={personas}
              turns={turns}
              prioritizedSpeakers={prioritizedSpeakers}
              activeSpeakerId={activeSpeakerId}
              thinkingSpeakerId={thinkingSpeakerId}
              thinkingStage={thinkingStage}
              mutedSpeakers={mutedSpeakers}
              primaryTicker={primaryTicker}
              portfolioAttachmentLabel={
                portfolioAttachmentState
                  ? portfolioAttachmentLabel(portfolioAttachmentState).label
                  : undefined
              }
              finalRating={finalRating}
              pillarCoverage={pillarCoverage}
              hasRecentSandboxMath={hasRecentSandboxMath}
              voiceEnabled={voiceEnabled}
              getVoiceEnergy={getVoiceEnergy}
              showContextGuide={showContextGuide}
              isExpanded={is3DExpanded}
              fullViewport={isHudFullBleed}
              onToggleExpanded={() => setIs3DExpanded((e) => !e)}
              onToggleVoiceEnabled={onToggleVoiceEnabled}
              onToggleContextGuide={() => {
                setShowContextGuide((prev) => !prev)
                setGuideSeatId('all')
              }}
              onToggleMute={onToggleMute}
              onMentionSpeaker={onMentionSpeaker}
              onInspectSpeaker={(sid) => {
                setGuideSeatId(sid)
                setShowContextGuide(true)
              }}
              onCustomizeSpeaker={onCustomizeSpeaker}
              onSwitchTo2D={() => handleSetViewMode('2d')}
            />
          </Suspense>

          {isHudFullBleed ? (
            <div className="pointer-events-none select-none absolute inset-x-3 top-[122px] z-10 flex flex-col gap-1.5">
              <div className="flex flex-wrap items-center justify-between gap-1.5">
                <div className="pointer-events-auto flex flex-wrap items-center gap-1.5">
                  <button
                    type="button"
                    aria-expanded={hudSeatDockOpen}
                    onClick={() => setHudSeatDockOpen((v) => !v)}
                    className="inline-flex items-center gap-1.5 rounded-full border border-white/15 bg-slate-900/70 px-2.5 py-0.5 text-[11px] font-semibold text-white backdrop-blur-md transition-colors hover:bg-slate-800/85"
                  >
                    <span>👥 ที่นั่งสภากูรู ({activeSeatCount} เสียง)</span>
                    <span className="text-sky-300">{hudSeatDockOpen ? '▴ ซ่อน' : '▾ กาง'}</span>
                  </button>
                  {spotlightMeta && (
                    <span
                      className="inline-flex items-center gap-1 rounded-full border border-sky-400/50 bg-slate-900/75 px-2.5 py-0.5 text-[11px] font-semibold text-sky-100 backdrop-blur-md"
                    >
                      <span className="h-1.5 w-1.5 rounded-full bg-sky-400 animate-pulse" />
                      {activeSpeakerId
                        ? `${spotlightMeta.shortName} กำลังพูด`
                        : `${spotlightMeta.shortName} กำลังวิเคราะห์`}
                    </span>
                  )}
                  {userRoleModels.length > 0 && (
                    <span className="hidden rounded-full border border-amber-300/40 bg-amber-500/20 px-2 py-0.5 text-[11px] font-semibold text-amber-100 backdrop-blur-md sm:inline-flex">
                      ★ Role Models: {userRoleModels.join(', ')}
                    </span>
                  )}
                </div>

                <button
                  type="button"
                  aria-expanded={showContextGuide}
                  onClick={() => {
                    setShowContextGuide((prev) => !prev)
                    setGuideSeatId('all')
                  }}
                  className={`pointer-events-auto rounded-full border px-2.5 py-0.5 text-[11px] font-semibold backdrop-blur-md transition-colors ${
                    showContextGuide
                      ? 'border-sky-300 bg-sky-500 text-white'
                      : 'border-white/15 bg-slate-900/70 text-sky-100 hover:bg-slate-800/85'
                  }`}
                >
                  {showContextGuide ? (
                    <span className="inline-flex items-center gap-1">
                      <ChevronDownIcon className="h-3 w-3 rotate-180 transition-transform" />
                      <span>ซ่อนกรอบคิดกูรู</span>
                    </span>
                  ) : (
                    <span className="inline-flex items-center gap-1.5">
                      <CompassIcon className="h-3 w-3 text-sky-300" />
                      <span>ดูกรอบคิดกูรู 6 สาย</span>
                    </span>
                  )}
                </button>
              </div>

              {hudSeatDockOpen && (
                <div className="pointer-events-auto">{renderCompactSeatGrid(true)}</div>
              )}
            </div>
          ) : (
            renderCompactSeatGrid(false)
          )}
        </div>
      ) : (
        /* Mode 2: 2D Classic Podium Grid — vertical hierarchy prevents horizontal text crush */
        <div className="rt-perspective grid grid-cols-2 gap-2 sm:grid-cols-3 lg:grid-cols-6">
        {orderedSeatIds.map((seatId) => {
          const meta = getSpeakerMeta(seatId, personas)
          const isSpeaking = activeSpeakerId === seatId
          const isThinking = thinkingSpeakerId === seatId
          const isMuted = mutedSpeakers.has(seatId)
          const isDisabled = !meta.enabled
          const isPrioritized = prioritizedSpeakers.includes(seatId)
          const latestStance = latestStanceBySpeaker[seatId]

          return (
            <motion.div
              key={seatId}
              initial={false}
              animate={
                isSpeaking
                  ? { y: -4, rotateX: 3, scale: 1.02 }
                  : { y: 0, rotateX: 0, rotateY: 0, scale: 1 }
              }
              whileHover={
                isDisabled
                  ? undefined
                  : { y: -3, rotateX: 4, rotateY: -3, scale: 1.015 }
              }
              transition={{ type: 'spring', stiffness: 300, damping: 24 }}
              className={`rt-preserve-3d group relative flex min-w-0 flex-col justify-between rounded-2xl border p-2.5 transition-colors ${
                isDisabled
                  ? 'border-dashed border-slate-200 bg-slate-50/70 opacity-55'
                  : isSpeaking
                    ? 'border-sky-400 bg-gradient-to-b from-white via-sky-50/90 to-cyan-50/70 shadow-md ring-1 ring-sky-300/60'
                    : isThinking
                      ? 'border-blue-300 bg-gradient-to-b from-white to-blue-50/75 shadow-sm'
                      : 'border-slate-200/80 bg-white/90 shadow-sm hover:border-sky-300/90 hover:shadow-sm'
              }`}
            >
              {/* Top Row: Avatar Orb (3D halo) + Corner Quiet Utility Icons */}
              <div className="flex items-start justify-between gap-1">
                <button
                  type="button"
                  disabled={isDisabled}
                  onClick={() => {
                    if (!isDisabled) onMentionSpeaker?.(seatId)
                  }}
                  title={
                    isDisabled
                      ? `${meta.shortName} ถูกปิดใช้งานสำหรับรอบนี้ (${meta.roleLensLabel})`
                      : `แท็กถาม @${meta.shortName} (${meta.archetype}) — ${meta.focusSummary}`
                  }
                  className="group/orb relative flex items-center gap-1.5 rounded-lg text-left focus:outline-none focus-visible:ring-2 focus-visible:ring-sky-500/45 disabled:cursor-default"
                >
                  <span
                    className={`relative flex h-9 w-9 shrink-0 items-center justify-center rounded-full border text-base transition-transform duration-300 ${
                      isSpeaking ? 'scale-105 shadow-sm' : 'group-hover/orb:scale-105'
                    }`}
                    style={{
                      borderColor: isSpeaking ? meta.accentColor : `${meta.accentColor}66`,
                      background: `radial-gradient(circle at 30% 28%, #ffffff, ${meta.accentColor}22)`,
                      boxShadow: isSpeaking
                        ? `0 0 0 3px ${meta.accentColor}26, 0 6px 14px -4px ${meta.accentColor}55`
                        : undefined,
                    }}
                  >
                    {meta.emoji}
                    {isSpeaking && (
                      <span
                        aria-hidden="true"
                        className="rt-orbit-ring-primary pointer-events-none absolute -inset-1 rounded-full border"
                        style={{ borderColor: `${meta.accentColor}99`, animationDuration: '4s' }}
                      />
                    )}
                  </span>

                  {isPrioritized && !isDisabled && (
                    <span
                      className="text-xs leading-none text-amber-500"
                      title="ตรงกับ Role Model ใน Essence ของคุณ"
                    >
                      ★
                    </span>
                  )}
                </button>

                {/* Quiet corner controls: progressive disclosure until hover/focus/active */}
                {renderSeatActions(seatId, meta, isMuted, isDisabled)}
              </div>

              {/* Middle Row: Full Name + Clean Lens Subtitle */}
              <button
                type="button"
                disabled={isDisabled}
                onClick={() => {
                  if (!isDisabled) onMentionSpeaker?.(seatId)
                }}
                className="mt-2 w-full min-w-0 rounded-lg text-left focus:outline-none focus-visible:ring-2 focus-visible:ring-sky-500/45 disabled:cursor-default"
              >
                <div className="flex items-center justify-between gap-1">
                  <span className="truncate text-[13px] font-bold tracking-tight text-slate-900">
                    {meta.shortName}
                  </span>
                  {meta.presetId && meta.presetId !== 'default' && (
                    <span className="max-w-[55%] truncate rounded bg-sky-50 px-1 py-0 text-[11px] font-semibold text-sky-700">
                      {meta.presetLabel || meta.presetId}
                    </span>
                  )}
                </div>
                <p
                  className="mt-0.5 line-clamp-1 text-[11px] font-medium"
                  style={{ color: meta.accentColor }}
                  title={meta.focusSummary}
                >
                  {meta.roleLensLabel}
                </p>
              </button>

              {/* Footer Row: Single Clean Status / Equalizer Pill */}
              <div className="mt-2 flex min-h-[20px] items-center justify-between gap-1">
                {isDisabled ? (
                  <span className="rounded-md border border-slate-200 bg-white/80 px-1.5 py-0.5 text-[11px] font-medium text-slate-500">
                    ไม่ได้ร่วมวง
                  </span>
                ) : latestStance ? (
                  <span
                    className={`rounded-md border px-1.5 py-0.5 text-[11px] font-bold tracking-wide ${stanceBadgeClass(
                      latestStance,
                    )}`}
                  >
                    {latestStance}
                  </span>
                ) : (
                  <span className="text-[11px] text-slate-500">รอความเห็น</span>
                )}

                {!isDisabled && isSpeaking && (
                  <span
                    aria-label="กำลังพูด"
                    className="inline-flex items-center gap-1 rounded-full bg-sky-600 px-1.5 py-0.5 text-[11px] font-semibold text-white shadow-sm"
                  >
                    <span className="inline-flex h-2.5 items-end gap-[2px]" aria-hidden="true">
                      <span className="rt-eq-bar h-2 w-[2px] rounded-full bg-white" />
                      <span
                        className="rt-eq-bar h-2.5 w-[2px] rounded-full bg-white"
                        style={{ animationDelay: '150ms' }}
                      />
                      <span
                        className="rt-eq-bar h-1.5 w-[2px] rounded-full bg-white"
                        style={{ animationDelay: '300ms' }}
                      />
                    </span>
                    <span>พูด</span>
                  </span>
                )}
                {!isDisabled && !isSpeaking && isThinking && (
                  <span className="inline-flex items-center gap-1 rounded-full bg-blue-100 px-1.5 py-0.5 text-[11px] font-semibold text-blue-700">
                    <span className="h-1.5 w-1.5 rounded-full bg-blue-500 animate-ping" />
                    <span>คิด...</span>
                  </span>
                )}
              </div>
            </motion.div>
          )
        })}
        </div>
      )}

      {/* Expandable Context Guide Drawer */}
      {showContextGuide && (
        <div
          aria-label="คำอธิบายกรอบคิดและจุดเน้นของกูรูแต่ละสาย"
          className={
            isHudFullBleed
              ? 'pointer-events-auto absolute inset-x-3 top-[198px] z-30 mx-auto max-h-[calc(100%-15rem)] max-w-5xl space-y-2 overflow-y-auto rounded-2xl border border-sky-200 bg-white/95 p-3 shadow-2xl backdrop-blur-xl custom-scrollbar'
              : 'mt-2.5 space-y-2 rounded-xl border border-sky-200 bg-sky-50/45 p-2.5'
          }
        >
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div className="flex flex-wrap items-center gap-1">
              <button
                type="button"
                onClick={() => setGuideSeatId('all')}
                className={`rounded-lg border px-2 py-0.5 text-[11px] font-semibold transition-colors ${
                  guideSeatId === 'all'
                    ? 'border-sky-400 bg-sky-600 text-white'
                    : 'border-slate-200 bg-white text-slate-700 hover:bg-slate-50'
                }`}
              >
                เทียบครบ {orderedSeatIds.length} สาย (ไม่ซ้ำกัน)
              </button>
              {orderedSeatIds.map((sid) => {
                const m = getSpeakerMeta(sid, personas)
                const active = guideSeatId === sid
                return (
                  <button
                    key={sid}
                    type="button"
                    onClick={() => setGuideSeatId(sid)}
                    className={`rounded-lg border px-2 py-0.5 text-[11px] font-semibold transition-colors ${
                      active
                        ? 'border-sky-400 bg-sky-100 text-sky-900'
                        : 'border-slate-200 bg-white text-slate-600 hover:bg-slate-50'
                    }`}
                  >
                    {m.emoji} {m.shortName}
                  </button>
                )
              })}
            </div>
            <button
              type="button"
              onClick={() => setShowContextGuide(false)}
              className="rounded px-1.5 py-0.5 text-[11px] font-medium text-slate-500 hover:bg-white hover:text-slate-800"
            >
              ✕ ปิด
            </button>
          </div>

          <div
            className={`grid gap-2 ${
              visibleGuideSeatIds.length === 1
                ? 'grid-cols-1'
                : 'grid-cols-1 sm:grid-cols-2 lg:grid-cols-3'
            }`}
          >
            {visibleGuideSeatIds.map((sid) => {
              const m = getSpeakerMeta(sid, personas)
              const rubricLines = parseAnalyticalRubricLines(m.analyticalRubric)
              return (
                <div
                  key={sid}
                  className="space-y-1.5 rounded-xl border border-slate-200/90 bg-white p-2.5 shadow-sm"
                >
                  <div className="flex flex-wrap items-start justify-between gap-1.5">
                    <div className="min-w-0">
                      <div className="flex flex-wrap items-center gap-1.5">
                        <span className="text-xs font-extrabold text-slate-900">
                          {m.emoji} {m.displayName}
                        </span>
                        <span
                          className="rounded-full border px-1.5 py-0 text-[11px] font-bold"
                          style={{
                            borderColor: `${m.accentColor}55`,
                            backgroundColor: `${m.accentColor}12`,
                            color: m.accentColor,
                          }}
                        >
                          {m.roleLensLabel}
                        </span>
                        {!m.enabled && (
                          <span className="rounded bg-slate-100 px-1.5 py-0 text-[11px] text-slate-500">
                            ปิดอยู่
                          </span>
                        )}
                      </div>
                      <p className="text-[11px] text-slate-500">{m.archetype}</p>
                    </div>
                    {onCustomizeSpeaker && (
                      <button
                        type="button"
                        onClick={() => onCustomizeSpeaker(sid)}
                        className="rounded border border-slate-200 bg-slate-50 px-1.5 py-0.5 text-[11px] font-semibold text-sky-700 hover:bg-sky-50"
                      >
                        ⚙️ ปรับแต่ง
                      </button>
                    )}
                  </div>

                  {m.focusSummary && (
                    <p className="text-[11px] leading-relaxed text-slate-700">{m.focusSummary}</p>
                  )}

                  {m.primaryEvidenceLabels.length > 0 && (
                    <div className="flex flex-wrap gap-1">
                      {m.primaryEvidenceLabels.map((lbl) => (
                        <span
                          key={lbl}
                          className="rounded-md border border-sky-200/80 bg-sky-50/70 px-1.5 py-0.5 text-[11px] font-medium text-sky-900"
                        >
                          {lbl}
                        </span>
                      ))}
                    </div>
                  )}

                  {rubricLines.length > 0 && (
                    <div className="space-y-0.5 border-t border-slate-100 pt-1.5 text-[11px] leading-snug text-slate-600">
                      {rubricLines.map((rl, idx) => (
                        <p key={`${sid}-rubric-${idx}`}>
                          {rl.title ? (
                            <>
                              <strong className="text-slate-800">{rl.title}:</strong> {rl.body}
                            </>
                          ) : (
                            rl.body
                          )}
                        </p>
                      ))}
                    </div>
                  )}

                  {m.customDirective && (
                    <p className="rounded-lg border border-amber-200/80 bg-amber-50/70 px-2 py-1 text-[11px] text-amber-900">
                      <strong>คำสั่งเสริมของคุณ:</strong> {m.customDirective}
                    </p>
                  )}
                </div>
              )
            })}
          </div>
        </div>
      )}
    </section>
  )
}
