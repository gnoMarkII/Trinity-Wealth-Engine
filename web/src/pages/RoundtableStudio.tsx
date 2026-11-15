import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { MotionConfig } from 'motion/react'
import { useSearchParams } from 'react-router-dom'
import { api, streamRoundtableDebate } from '../api/client'
import type {
  CoreGuruIdType,
  DebateDepthType,
  DebateIntensityType,
  GuruPersonaDTO,
  RoundtableGuruConfigCatalogDTO,
  RoundtableSessionDetailDTO,
  RoundtableSessionSummaryDTO,
  RoundtableTurnDTO,
  SpeakerIdType,
  UpdateGuruPersonaInputDTO,
  VoicePreviewRequestDTO,
} from '../api/types'
import { DecisionPlaybookPanel } from '../components/roundtable/DecisionPlaybookPanel'
import { ScrollHistoryIcon, SlidersIcon } from '../components/ui/AppIcons'
import { GuruAvatarStage, type StageViewMode } from '../components/roundtable/GuruAvatarStage'
import { GuruCustomizationModal } from '../components/roundtable/GuruCustomizationModal'
import { RaiseHandComposer } from '../components/roundtable/RaiseHandComposer'
import {
  RoundtableStreamPanel,
  type InterjectionToolRecord,
} from '../components/roundtable/RoundtableStreamPanel'
import {
  SPEAKER_META_MAP,
  STAGE_SEAT_ORDER,
  getSpeakerMeta,
  portfolioAttachmentLabel,
  stanceBadgeClass,
} from '../components/roundtable/personas'
import type { FeedbackKind } from '../components/roundtable/ui'
import { useRoundtableVoicePlayer } from '../components/roundtable/useRoundtableVoicePlayer'

function buildFallbackDefaultPersonas(): GuruPersonaDTO[] {
  return STAGE_SEAT_ORDER.map((sid) => {
    const meta = SPEAKER_META_MAP[sid]!
    return {
      speaker_id: meta.speakerId,
      display_name: meta.displayName,
      short_name: meta.shortName,
      archetype: meta.archetype,
      voice_name: meta.voiceName,
      accent_color: meta.accentColor,
      is_prioritized_role_model: false,
      emoji: meta.emoji,
      pitch: meta.pitch,
      rate: meta.rate,
      enabled: meta.enabled,
      debate_order: meta.debateOrder,
      preset_id: meta.presetId,
      preset_label: meta.presetLabel,
      speaking_style: meta.speakingStyle,
      custom_directive: meta.customDirective,
      role_lens_label: meta.roleLensLabel,
      focus_summary: meta.focusSummary,
      primary_evidence_labels: meta.primaryEvidenceLabels,
      analytical_rubric: meta.analyticalRubric,
    }
  })
}

const HISTORY_LOAD_TIMEOUT_MS = 25_000
const SESSION_DETAIL_TIMEOUT_MS = 20_000
const STAGE_VIEW_STORAGE_KEY = 'trinity_rt_stage_view_mode'

function isAbortError(err: unknown): boolean {
  return Boolean(err && typeof err === 'object' && (err as { name?: unknown }).name === 'AbortError')
}

function isTransientBootError(err: unknown): boolean {
  if (!err || typeof err !== 'object') return false
  const status = (err as { status?: unknown }).status
  return typeof status === 'number' && status >= 502 && status <= 504
}

async function runWithTimeout<T>(
  task: (signal: AbortSignal) => Promise<T>,
  timeoutMs: number,
  parentSignal?: AbortSignal,
): Promise<T> {
  const controller = new AbortController()
  const onParentAbort = () => controller.abort()
  if (parentSignal) {
    if (parentSignal.aborted) {
      controller.abort()
    } else {
      parentSignal.addEventListener('abort', onParentAbort, { once: true })
    }
  }
  const timer = window.setTimeout(() => controller.abort(), timeoutMs)
  try {
    return await task(controller.signal)
  } finally {
    window.clearTimeout(timer)
    parentSignal?.removeEventListener('abort', onParentAbort)
  }
}

function toOverridePayloads(personas: GuruPersonaDTO[]): UpdateGuruPersonaInputDTO[] {
  return STAGE_SEAT_ORDER.map((sid, idx) => {
    const meta = getSpeakerMeta(sid, personas)
    return {
      speaker_id: sid as CoreGuruIdType,
      display_name: meta.displayName,
      short_name: meta.shortName,
      archetype: meta.archetype,
      speaking_style: meta.speakingStyle,
      custom_directive: meta.customDirective,
      preset_id: meta.presetId,
      voice_name: meta.voiceName,
      pitch: meta.pitch,
      rate: meta.rate,
      accent_color: meta.accentColor,
      emoji: meta.emoji,
      enabled: sid === 'hermes_moderator' ? true : meta.enabled,
      debate_order: sid === 'hermes_moderator' ? 0 : meta.debateOrder || idx,
    }
  })
}

function mergeDraftOverridesIntoDtos(
  basePersonas: GuruPersonaDTO[],
  updates: UpdateGuruPersonaInputDTO[],
  presets?: RoundtableGuruConfigCatalogDTO['presets'],
): GuruPersonaDTO[] {
  const byId = new Map(updates.map((u) => [u.speaker_id as string, u]))
  return STAGE_SEAT_ORDER.map((sid) => {
    const base = basePersonas.find((p) => p.speaker_id === sid) ?? {
      ...getSpeakerMeta(sid),
      speaker_id: sid,
      display_name: getSpeakerMeta(sid).displayName,
      short_name: getSpeakerMeta(sid).shortName,
      archetype: getSpeakerMeta(sid).archetype,
      voice_name: getSpeakerMeta(sid).voiceName,
      accent_color: getSpeakerMeta(sid).accentColor,
      is_prioritized_role_model: false,
    }
    const patch = byId.get(sid)
    if (!patch) return base
    const nextPresetId = patch.preset_id ?? base.preset_id ?? 'default'
    const matchedPreset = presets?.find(
      (pr) => pr.speaker_id === sid && pr.preset_id === nextPresetId,
    )
    return {
      ...base,
      display_name: patch.display_name ?? base.display_name,
      short_name: patch.short_name ?? base.short_name,
      archetype: patch.archetype ?? base.archetype,
      speaking_style: patch.speaking_style ?? base.speaking_style,
      custom_directive: patch.custom_directive ?? base.custom_directive,
      preset_id: nextPresetId,
      preset_label:
        matchedPreset?.label_th || (nextPresetId === 'default' ? '' : base.preset_label),
      voice_name: patch.voice_name ?? base.voice_name,
      pitch: patch.pitch ?? base.pitch,
      rate: patch.rate ?? base.rate,
      accent_color: patch.accent_color ?? base.accent_color,
      emoji: patch.emoji ?? base.emoji,
      enabled: sid === 'hermes_moderator' ? true : (patch.enabled ?? base.enabled ?? true),
      debate_order:
        sid === 'hermes_moderator' ? 0 : (patch.debate_order ?? base.debate_order),
      role_lens_label: matchedPreset?.role_lens_label || base.role_lens_label,
      focus_summary: matchedPreset?.focus_summary || base.focus_summary,
      primary_evidence_labels:
        matchedPreset?.primary_evidence_labels && matchedPreset.primary_evidence_labels.length > 0
          ? matchedPreset.primary_evidence_labels
          : base.primary_evidence_labels,
      analytical_rubric: matchedPreset?.analytical_rubric || base.analytical_rubric,
    }
  }).sort((a, b) => (a.debate_order ?? 99) - (b.debate_order ?? 99))
}

export default function RoundtableStudio() {
  const [searchParams] = useSearchParams()

  const [sessions, setSessions] = useState<RoundtableSessionSummaryDTO[]>([])
  const [loadingSessions, setLoadingSessions] = useState(true)
  const [sessionsError, setSessionsError] = useState<string | null>(null)
  const [loadingSessionId, setLoadingSessionId] = useState<string | null>(null)
  const [showHistoryRail, setShowHistoryRail] = useState(true)
  const [mobileHistoryOpen, setMobileHistoryOpen] = useState(false)
  const [stageViewMode, setStageViewMode] = useState<StageViewMode>(() => {
    try {
      const saved = window.localStorage.getItem(STAGE_VIEW_STORAGE_KEY)
      return saved === '2d' ? '2d' : '3d'
    } catch {
      return '3d'
    }
  })
  const [rightHudExpanded, setRightHudExpanded] = useState(false)
  const [rightHudCollapsed, setRightHudCollapsed] = useState(false)
  const [streamHudCollapsed, setStreamHudCollapsed] = useState(false)
  const [activeDetail, setActiveDetail] = useState<RoundtableSessionDetailDTO | null>(null)
  const [isNewMode, setIsNewMode] = useState(true)

  const handleStageViewModeChange = useCallback((mode: StageViewMode) => {
    setStageViewMode(mode)
    try {
      window.localStorage.setItem(STAGE_VIEW_STORAGE_KEY, mode)
    } catch {
      // Ignore storage failures in restricted contexts
    }
  }, [])

  // Guru Customization & Voice Config State
  const [guruCatalog, setGuruCatalog] = useState<RoundtableGuruConfigCatalogDTO | null>(null)
  const [draftPersonasOverride, setDraftPersonasOverride] = useState<GuruPersonaDTO[] | null>(null)
  const [isCustomizeOpen, setIsCustomizeOpen] = useState(false)
  const [customizeInitialSeat, setCustomizeInitialSeat] = useState<string | null>('value_guru')
  const [isSavingProfiles, setIsSavingProfiles] = useState(false)
  const [isResettingProfiles, setIsResettingProfiles] = useState(false)

  const fallbackDefaultPersonas = useMemo(() => buildFallbackDefaultPersonas(), [])
  const baseConfiguredPersonas = useMemo(
    () => draftPersonasOverride ?? guruCatalog?.personas ?? fallbackDefaultPersonas,
    [draftPersonasOverride, guruCatalog?.personas, fallbackDefaultPersonas],
  )
  const effectivePersonas = useMemo(
    () =>
      !isNewMode && activeDetail?.personas?.length
        ? activeDetail.personas
        : baseConfiguredPersonas,
    [isNewMode, activeDetail?.personas, baseConfiguredPersonas],
  )

  const voicePlayer = useRoundtableVoicePlayer(effectivePersonas)

  // Draft new session composer
  const [agendaPrompt, setAgendaPrompt] = useState('')
  const [contextRefs, setContextRefs] = useState<string[]>(['@Portfolio', '@Macro'])
  const [debateIntensity, setDebateIntensity] = useState<DebateIntensityType>('crossfire')
  const [debateDepth, setDebateDepth] = useState<DebateDepthType>('deep_youtube')
  const [isCreating, setIsCreating] = useState(false)

  // Live debate & interjection state
  const [isStreaming, setIsStreaming] = useState(false)
  const [thinkingSpeakerId, setThinkingSpeakerId] = useState<string | null>(null)
  const [thinkingStage, setThinkingStage] = useState<string | null>(null)
  const [revealTurnIds, setRevealTurnIds] = useState<Set<string>>(() => new Set())
  const [toolRecords, setToolRecords] = useState<InterjectionToolRecord[]>([])
  const [syncWithVoice, setSyncWithVoice] = useState(true)
  const [pacedSessionId, setPacedSessionId] = useState<string | null>(null)

  const [interjectText, setInterjectText] = useState('')
  const [targetSpeaker, setTargetSpeaker] = useState<SpeakerIdType | null>(null)
  const [runSandboxMath, setRunSandboxMath] = useState(false)
  const [isInterjecting, setIsInterjecting] = useState(false)

  const [isSavingVault, setIsSavingVault] = useState(false)
  const [isUpdatingThesis, setIsUpdatingThesis] = useState(false)
  const [isExportingYoutube, setIsExportingYoutube] = useState(false)
  const [dismissedPrereqs, setDismissedPrereqs] = useState(false)
  const [dismissedDrift, setDismissedDrift] = useState(false)

  const [notice, setNotice] = useState<{ text: string; type: 'info' | 'success' | 'error' } | null>(
    null,
  )

  const mountedRef = useRef(true)
  const streamAbortRef = useRef<AbortController | null>(null)
  const sessionsAbortRef = useRef<AbortController | null>(null)
  const selectAbortRef = useRef<AbortController | null>(null)
  const selectSeqRef = useRef(0)

  const showNotice = useCallback(
    (text: string, type: 'info' | 'success' | 'error' = 'info') => {
      setNotice({ text, type })
    },
    [],
  )

  // Prefill from ?ticker=... or ?agenda=macro&refs=@Macro
  useEffect(() => {
    const qpTicker = searchParams.get('ticker')?.trim().toUpperCase()
    const qpAgenda = searchParams.get('agenda')?.trim().toLowerCase()
    const qpRefs = searchParams.get('refs')?.trim()

    if (qpTicker) {
      const tok = qpTicker.startsWith('$') ? qpTicker : `$${qpTicker}`
      setContextRefs([tok, '@Portfolio'])
      setAgendaPrompt(
        `ประชุมสภากูรูวิเคราะห์หุ้น ${qpTicker} รอบด้าน — มูลค่าเหมาะสม ความเสี่ยง และคำแนะนำถือ/ซื้อ/ขายสำหรับพอร์ต`,
      )
      setIsNewMode(true)
      setActiveDetail(null)
    } else if (qpAgenda === 'macro' || qpRefs) {
      const refs = qpRefs
        ? qpRefs
            .split(',')
            .map((r) => r.trim())
            .filter(Boolean)
        : ['@Macro', '@Digest']
      setContextRefs(refs.length ? refs : ['@Macro'])
      setAgendaPrompt(
        'ประเมินภาพรวมเศรษฐกิจมหภาค สภาพคล่องโลก และจังหวะการจัดสรรเงินลงทุนในระยะถัดไป',
      )
      setIsNewMode(true)
      setActiveDetail(null)
    }
  }, [searchParams])

  const loadGuruCatalog = useCallback(async (signal?: AbortSignal) => {
    for (let attempt = 0; attempt < 2; attempt += 1) {
      if (signal?.aborted || !mountedRef.current) return
      try {
        if (typeof api.getRoundtableGuruProfiles === 'function') {
          const catalog = await runWithTimeout(
            (reqSignal) => api.getRoundtableGuruProfiles({ signal: reqSignal }),
            HISTORY_LOAD_TIMEOUT_MS,
            signal,
          )
          if (mountedRef.current && catalog && Array.isArray(catalog.personas)) {
            setGuruCatalog(catalog)
          }
        }
        return
      } catch (err) {
        if (isAbortError(err) || signal?.aborted) return
        if (attempt === 0 && isTransientBootError(err)) {
          await new Promise((resolve) => window.setTimeout(resolve, 1000))
          continue
        }
        return
      }
    }
  }, [])

  const loadSessions = useCallback(async () => {
    sessionsAbortRef.current?.abort()
    const controller = new AbortController()
    sessionsAbortRef.current = controller

    setLoadingSessions(true)
    setSessionsError(null)
    try {
      for (let attempt = 0; attempt < 3; attempt += 1) {
        if (controller.signal.aborted || !mountedRef.current) return
        try {
          const list = await runWithTimeout(
            (reqSignal) => api.listRoundtableSessions(30, { signal: reqSignal }),
            HISTORY_LOAD_TIMEOUT_MS,
            controller.signal,
          )
          if (mountedRef.current && !controller.signal.aborted) {
            setSessions(list)
            setSessionsError(null)
          }
          return
        } catch (err: any) {
          if (controller.signal.aborted || !mountedRef.current) return
          if (attempt < 2 && isTransientBootError(err)) {
            await new Promise((resolve) => window.setTimeout(resolve, (attempt + 1) * 900))
            continue
          }
          const message = isAbortError(err)
            ? 'เซิร์ฟเวอร์ตอบสนองช้ากว่าปกติ กรุณากดลองโหลดใหม่'
            : err?.message || 'ไม่สามารถโหลดประวัติการประชุมได้'
          setSessionsError(message)
          showNotice(message, 'error')
          return
        }
      }
    } finally {
      if (mountedRef.current && sessionsAbortRef.current === controller) {
        setLoadingSessions(false)
      }
    }
  }, [showNotice])

  useEffect(() => {
    mountedRef.current = true
    const catalogAbort = new AbortController()
    void loadSessions()
    void loadGuruCatalog(catalogAbort.signal)
    return () => {
      mountedRef.current = false
      catalogAbort.abort()
      sessionsAbortRef.current?.abort()
      selectAbortRef.current?.abort()
      streamAbortRef.current?.abort()
    }
  }, [loadSessions, loadGuruCatalog])

  const startDebateStream = useCallback(
    async (
      sessionId: string,
      refreshContextOnDrift: boolean = false,
      debateDepthHint?: DebateDepthType,
    ) => {
      streamAbortRef.current?.abort()
      const ac = new AbortController()
      streamAbortRef.current = ac

      const initialDepth = debateDepthHint ?? activeDetail?.debate_depth ?? debateDepth
      setPacedSessionId(sessionId)
      setRightHudCollapsed(false)
      setStreamHudCollapsed(false)
      setIsStreaming(true)
      setThinkingSpeakerId('hermes_moderator')
      setThinkingStage(initialDepth === 'deep_youtube' ? 'stage_intro' : 'opening')

      await streamRoundtableDebate(
        sessionId,
        {
          voiceEnabled: voicePlayer.voiceEnabled,
          refreshContextOnDrift,
          signal: ac.signal,
        },
        {
          onSessionMeta: (meta) => {
            setActiveDetail((prev) =>
              prev && prev.session.session_id === sessionId
                ? {
                    ...prev,
                    prioritized_speakers: meta.prioritized_speakers,
                    missing_prerequisite_tickers: meta.missing_prerequisite_tickers,
                    portfolio_attachment_state: meta.portfolio_attachment_state,
                    packet_fingerprint_sha256: meta.packet_fingerprint_sha256,
                    context_drift_detected: meta.context_refreshed ? false : prev.context_drift_detected,
                    debate_intensity: meta.debate_intensity ?? prev.debate_intensity,
                    debate_depth: meta.debate_depth ?? prev.debate_depth,
                    personas: meta.personas?.length ? meta.personas : prev.personas,
                  }
                : prev,
            )
          },
          onSpeakerStarted: (ev) => {
            setThinkingSpeakerId(ev.speaker_id)
            setThinkingStage(ev.round_stage)
          },
          onTurnCompleted: (turn: RoundtableTurnDTO) => {
            setThinkingSpeakerId((curr) => (curr === turn.speaker_id ? null : curr))
            setRevealTurnIds((prev) => new Set(prev).add(turn.turn_id))
            setActiveDetail((prev) => {
              if (!prev || prev.session.session_id !== sessionId) return prev
              if (prev.turns.some((t) => t.turn_id === turn.turn_id)) return prev
              return { ...prev, turns: [...prev.turns, turn] }
            })
            voicePlayer.enqueueTurn(turn, voicePlayer.voiceEnabled)
          },
          onAudioReady: (ev) => {
            setActiveDetail((prev) => {
              if (!prev || prev.session.session_id !== sessionId) return prev
              return {
                ...prev,
                turns: prev.turns.map((t) =>
                  t.turn_id === ev.turn_id
                    ? {
                        ...t,
                        audio_cache_key: ev.audio_cache_key ?? t.audio_cache_key,
                        audio_status: ev.audio_status,
                      }
                    : t,
                ),
              }
            })
            voicePlayer.notifyAudioReady(ev)
          },
          onPlaybookUpdated: (playbook) => {
            setActiveDetail((prev) =>
              prev && prev.session.session_id === sessionId
                ? {
                    ...prev,
                    decision_playbook: playbook,
                    session: {
                      ...prev.session,
                      final_rating: playbook.final_rating,
                      status: 'completed',
                    },
                  }
                : prev,
            )
            setSessions((prev) =>
              prev.map((s) =>
                s.session_id === sessionId
                  ? { ...s, final_rating: playbook.final_rating, status: 'completed' }
                  : s,
              ),
            )
          },
          onPillarCoverage: (coverage) => {
            setActiveDetail((prev) =>
              prev && prev.session.session_id === sessionId
                ? {
                    ...prev,
                    pillar_coverage: coverage,
                  }
                : prev,
            )
          },
          onPaused: () => {
            setIsStreaming(false)
            setThinkingSpeakerId(null)
            showNotice('หยุดรอบการถกเถียงชั่วคราวเพื่อรับคำถามแทรกของคุณ', 'info')
          },
          onDone: () => {
            setIsStreaming(false)
            setThinkingSpeakerId(null)
          },
          onError: (msg) => {
            setIsStreaming(false)
            setThinkingSpeakerId(null)
            showNotice(msg, 'error')
          },
        },
      )
    },
    [activeDetail?.debate_depth, debateDepth, showNotice, voicePlayer],
  )

  const handleSelectSession = async (sessionId: string) => {
    streamAbortRef.current?.abort()
    selectAbortRef.current?.abort()
    const controller = new AbortController()
    selectAbortRef.current = controller
    const seq = ++selectSeqRef.current

    voicePlayer.clearQueue()
    setPacedSessionId(null)
    setIsStreaming(false)
    setThinkingSpeakerId(null)
    setDismissedPrereqs(false)
    setDismissedDrift(false)
    setToolRecords([])
    setLoadingSessionId(sessionId)

    const summary = sessions.find((s) => s.session_id === sessionId)
    const shouldCheckDrift = summary?.status === 'active'

    try {
      const detail = await runWithTimeout(
        (reqSignal) =>
          api.getRoundtableSession(sessionId, {
            checkDrift: shouldCheckDrift,
            signal: reqSignal,
          }),
        SESSION_DETAIL_TIMEOUT_MS,
        controller.signal,
      )
      if (!mountedRef.current || controller.signal.aborted || seq !== selectSeqRef.current) return
      setActiveDetail(detail)
      setIsNewMode(false)
      setRevealTurnIds(new Set())
      setMobileHistoryOpen(false)
    } catch (err: any) {
      if (controller.signal.aborted || seq !== selectSeqRef.current || !mountedRef.current) return
      const message = isAbortError(err)
        ? 'โหลดรายละเอียดห้องประชุมไม่ทันเวลา กรุณาลองอีกครั้ง'
        : err?.message || 'ไม่สามารถโหลดรายละเอียดห้องประชุมได้'
      showNotice(message, 'error')
    } finally {
      if (mountedRef.current && seq === selectSeqRef.current) {
        setLoadingSessionId(null)
      }
    }
  }

  const handleStartNewSession = async () => {
    const trimmed = agendaPrompt.trim()
    if (!trimmed) return

    streamAbortRef.current?.abort()
    voicePlayer.clearQueue()
    setIsCreating(true)
    setDismissedPrereqs(false)
    setDismissedDrift(false)
    setToolRecords([])
    setRevealTurnIds(new Set())

    try {
      const detail = await api.createRoundtableSession({
        agenda_prompt: trimmed,
        context_refs: contextRefs,
        debate_intensity: debateIntensity,
        debate_depth: debateDepth,
        persona_overrides: draftPersonasOverride
          ? toOverridePayloads(draftPersonasOverride)
          : undefined,
      })
      setActiveDetail(detail)
      setIsNewMode(false)
      setSessions((prev) => [
        detail.session,
        ...prev.filter((s) => s.session_id !== detail.session.session_id),
      ])

      if (detail.missing_prerequisite_tickers.length > 0) {
        showNotice(
          `พบหุ้นที่ยังไม่มีบทวิเคราะห์เจาะลึก (${detail.missing_prerequisite_tickers.join(', ')}) กรุณาเลือกส่งวิเคราะห์ก่อน หรือกดดำเนินการต่อด้านขวา`,
          'info',
        )
      } else {
        void startDebateStream(detail.session.session_id, false, detail.debate_depth)
      }
    } catch (err: any) {
      showNotice(err?.message || 'ไม่สามารถเปิดห้องประชุมสภากูรูได้', 'error')
    } finally {
      setIsCreating(false)
    }
  }

  const submitInterjectionText = async (rawText: string) => {
    if (!activeDetail) return
    const text = rawText.trim()
    if (!text) return

    streamAbortRef.current?.abort()
    voicePlayer.clearQueue()
    setPacedSessionId(null)
    setIsStreaming(false)
    setThinkingSpeakerId(targetSpeaker || 'hermes_moderator')
    setThinkingStage('interjection_reply')
    setIsInterjecting(true)

    try {
      const res = await api.interjectRoundtableSession(activeDetail.session.session_id, {
        message: text,
        target_speaker_id: targetSpeaker,
        voice_enabled: voicePlayer.voiceEnabled,
        run_sandbox_math: runSandboxMath,
        refresh_context: activeDetail.context_drift_detected && !dismissedDrift,
      })
      setRevealTurnIds((prev) => new Set(prev).add(res.reply_turn.turn_id))
      setActiveDetail((prev) => {
        if (!prev || prev.session.session_id !== activeDetail.session.session_id) return prev
        return {
          ...prev,
          context_drift_detected: res.context_refreshed ? false : prev.context_drift_detected,
          turns: [...prev.turns, res.user_turn, res.reply_turn],
        }
      })
      if (res.tool_calls.length > 0) {
        setToolRecords((prev) => [
          ...prev,
          { afterTurnId: res.reply_turn.turn_id, events: res.tool_calls },
        ])
      }
      setInterjectText('')
      voicePlayer.enqueueTurn(res.reply_turn, false)
    } catch (err: any) {
      showNotice(err?.message || 'ไม่สามารถส่งคำถามแทรกได้', 'error')
    } finally {
      setIsInterjecting(false)
      setThinkingSpeakerId(null)
    }
  }

  const handleDispatchTicker = async (ticker: string) => {
    try {
      const res = await api.dispatchRoundtablePrerequisite(ticker)
      showNotice(res.message || `ส่งงานวิเคราะห์หุ้น ${ticker} เรียบร้อย`, res.ok ? 'success' : 'error')
    } catch (err: any) {
      showNotice(err?.message || `ไม่สามารถส่งงานวิเคราะห์ ${ticker} ได้`, 'error')
    }
  }

  const handleSaveToVault = async () => {
    if (!activeDetail) return
    setIsSavingVault(true)
    try {
      const res = await api.saveRoundtableToVault(activeDetail.session.session_id, {})
      showNotice(
        res.vault_path ? `${res.message} (${res.vault_path})` : res.message,
        res.ok ? 'success' : 'error',
      )
    } catch (err: any) {
      showNotice(err?.message || 'บันทึกลง Obsidian Vault ไม่สำเร็จ', 'error')
    } finally {
      setIsSavingVault(false)
    }
  }

  const handleUpdateThesis = async () => {
    if (!activeDetail) return
    setIsUpdatingThesis(true)
    try {
      await api.updateThesisFromRoundtable(activeDetail.session.session_id, {})
      showNotice(
        `อัปเดต Active Stock Thesis ของ $${activeDetail.session.primary_ticker || ''} เรียบร้อยแล้ว`,
        'success',
      )
    } catch (err: any) {
      showNotice(err?.message || 'ไม่สามารถอัปเดต Active Thesis ได้', 'error')
    } finally {
      setIsUpdatingThesis(false)
    }
  }

  const handleExportYoutubePack = async () => {
    if (!activeDetail) return
    setIsExportingYoutube(true)
    try {
      const { blob, filename } = await api.downloadRoundtableYoutubePack(
        activeDetail.session.session_id,
      )
      const url = URL.createObjectURL(blob)
      const anchor = document.createElement('a')
      anchor.href = url
      anchor.download = filename
      document.body.appendChild(anchor)
      anchor.click()
      document.body.removeChild(anchor)
      window.setTimeout(() => URL.revokeObjectURL(url), 1000)
      showNotice(`ดาวน์โหลดชุดทำคลิป YouTube (${filename}) เรียบร้อยแล้ว`, 'success')
    } catch (err: any) {
      showNotice(err?.message || 'ไม่สามารถส่งออกชุดทำคลิป YouTube ได้', 'error')
    } finally {
      setIsExportingYoutube(false)
    }
  }

  const handleToggleContextRef = (token: string) => {
    setContextRefs((prev) =>
      prev.some((r) => r.toLowerCase() === token.toLowerCase())
        ? prev.filter((r) => r.toLowerCase() !== token.toLowerCase())
        : [...prev, token],
    )
  }

  const handleAddTickerRef = (token: string) => {
    setContextRefs((prev) =>
      prev.some((r) => r.toUpperCase() === token.toUpperCase()) ? prev : [token.toUpperCase(), ...prev],
    )
  }

  const handleMentionSpeaker = (speakerId: string) => {
    if (isNewMode || !activeDetail) return
    setTargetSpeaker(speakerId as SpeakerIdType)
    const meta = getSpeakerMeta(speakerId, activeDetail.personas)
    const mentionTag = `@${meta.shortName}`
    if (!interjectText.toLowerCase().includes(mentionTag.toLowerCase())) {
      setInterjectText((prev) => `${mentionTag} ${prev}`.trimStart())
    }
  }

  const openCustomizationFor = (speakerId: string = 'value_guru') => {
    if (!guruCatalog) {
      void loadGuruCatalog()
    }
    setCustomizeInitialSeat(speakerId)
    setIsCustomizeOpen(true)
  }

  const handleToggleSeatEnabled = (speakerId: CoreGuruIdType) => {
    if (speakerId === 'hermes_moderator') return
    const current = baseConfiguredPersonas
    const activePanelCount = current.filter(
      (p) => p.speaker_id !== 'hermes_moderator' && (p.enabled ?? true),
    ).length
    const target = current.find((p) => p.speaker_id === speakerId)
    const currentlyEnabled = target ? (target.enabled ?? true) : true
    if (currentlyEnabled && activePanelCount <= 2) {
      showNotice('ต้องเปิดใช้งานกูรูเข้าร่วมประชุมอย่างน้อย 2 ท่าน (ไม่รวม Hermes)', 'error')
      return
    }
    const nextDtos = current.map((p) =>
      p.speaker_id === speakerId ? { ...p, enabled: !currentlyEnabled } : p,
    )
    setDraftPersonasOverride(nextDtos)
  }

  const handleSaveProfilesAsDefault = async (profiles: UpdateGuruPersonaInputDTO[]) => {
    setIsSavingProfiles(true)
    try {
      const updatedCatalog = await api.updateRoundtableGuruProfiles(profiles)
      setGuruCatalog(updatedCatalog)
      setDraftPersonasOverride(null)
      setIsCustomizeOpen(false)
      showNotice('บันทึกการตั้งค่ากูรูและเสียงพากย์เป็นค่าเริ่มต้นเรียบร้อยแล้ว', 'success')
    } catch (err: any) {
      showNotice(err?.message || 'ไม่สามารถบันทึกการตั้งค่ากูรูได้', 'error')
    } finally {
      setIsSavingProfiles(false)
    }
  }

  const handleApplyProfilesForSessionOnly = (profiles: UpdateGuruPersonaInputDTO[]) => {
    const merged = mergeDraftOverridesIntoDtos(
      baseConfiguredPersonas,
      profiles,
      guruCatalog?.presets,
    )
    setDraftPersonasOverride(merged)
    setIsCustomizeOpen(false)
    showNotice(
      isNewMode || !activeDetail
        ? 'ปรับใช้โปรไฟล์กูรูและเสียงพากย์ชั่วคราวสำหรับการเปิดประชุมรอบนี้แล้ว'
        : 'เตรียมโปรไฟล์กูรูชั่วคราวไว้แล้ว — กด "+ เปิดประชุมใหม่" เพื่อเริ่มประชุมด้วยค่าที่ปรับ',
      'success',
    )
  }

  const handleResetProfiles = async (speakerId?: CoreGuruIdType) => {
    setIsResettingProfiles(true)
    try {
      const updatedCatalog = await api.resetRoundtableGuruProfiles(speakerId)
      setGuruCatalog(updatedCatalog)
      if (!speakerId) {
        setDraftPersonasOverride(null)
      } else if (draftPersonasOverride) {
        const resetSeat = updatedCatalog.personas.find((p) => p.speaker_id === speakerId)
        if (resetSeat) {
          setDraftPersonasOverride(
            draftPersonasOverride.map((p) => (p.speaker_id === speakerId ? resetSeat : p)),
          )
        }
      }
      showNotice(
        speakerId
          ? 'รีเซ็ตโปรไฟล์กูรูที่เลือกกลับเป็นค่าเริ่มต้นแล้ว'
          : 'รีเซ็ตโปรไฟล์กูรูทั้งหมดกลับเป็นค่าเริ่มต้นแล้ว',
        'success',
      )
    } catch (err: any) {
      showNotice(err?.message || 'ไม่สามารถรีเซ็ตการตั้งค่าได้', 'error')
    } finally {
      setIsResettingProfiles(false)
    }
  }

  const handlePreviewVoice = async (req: VoicePreviewRequestDTO) => {
    try {
      const res = await voicePlayer.previewVoice(req)
      if (res.audio_status === 'fallback_web_speech' && res.error_message) {
        showNotice(`เล่นเสียงตัวอย่างผ่านเบราว์เซอร์ (${res.voice_name})`, 'info')
      }
    } catch (err: any) {
      showNotice(err?.message || 'ไม่สามารถสังเคราะห์เสียงตัวอย่างได้', 'error')
    }
  }

  const handleTurnFeedback = (_turn: RoundtableTurnDTO, kind: FeedbackKind | null, reason?: string) => {
    if (kind === 'up') {
      showNotice('บันทึกเสียงตอบรับเชิงบวกแล้ว', 'success')
    } else if (kind === 'down' && reason) {
      showNotice(`บันทึกข้อเสนอแนะ "${reason}" เรียบร้อย`, 'info')
    }
  }

  const allTurns = activeDetail?.turns ?? []
  const activeVoiceIdx = voicePlayer.activeTurnId
    ? allTurns.findIndex((t) => t.turn_id === voicePlayer.activeTurnId)
    : -1
  const isPacingActive =
    syncWithVoice &&
    voicePlayer.voiceEnabled &&
    Boolean(activeDetail && pacedSessionId === activeDetail.session.session_id) &&
    activeVoiceIdx >= 0
  const visibleTurns = isPacingActive ? allTurns.slice(0, activeVoiceIdx + 1) : allTurns
  const bufferedAheadCount = allTurns.length - visibleTurns.length
  const canRevealPlaybook =
    !isPacingActive ||
    (bufferedAheadCount === 0 && !isStreaming) ||
    allTurns[activeVoiceIdx]?.round_stage === 'synthesis'
  const visibleDecisionPlaybook = canRevealPlaybook ? activeDetail?.decision_playbook : null
  const visiblePillarCoverage = canRevealPlaybook ? activeDetail?.pillar_coverage : null
  const visibleThinkingSpeakerId =
    isPacingActive && bufferedAheadCount > 0 ? null : thinkingSpeakerId

  const playbackActive =
    voicePlayer.activeTurnId !== null ||
    voicePlayer.isPaused ||
    voicePlayer.isAwaitingTts ||
    voicePlayer.queueCount > 0

  useEffect(() => {
    if (
      !isStreaming &&
      pacedSessionId &&
      activeDetail?.session.session_id === pacedSessionId &&
      allTurns.length > 0 &&
      !playbackActive
    ) {
      setPacedSessionId(null)
    }
  }, [
    isStreaming,
    pacedSessionId,
    activeDetail?.session.session_id,
    allTurns.length,
    playbackActive,
  ])

  const canResumeDebate =
    Boolean(activeDetail) &&
    !isStreaming &&
    !activeDetail?.decision_playbook &&
    (activeDetail?.turns.length ?? 0) >= 0

  const historyListContent = (
    <>
      <div className="mb-2.5 flex items-center justify-between gap-2">
        <div className="flex items-center gap-1.5">
          <h2 className="text-[11px] font-bold uppercase tracking-wider text-slate-500">
            ประวัติห้องประชุม
          </h2>
          {sessions.length > 0 && (
            <span className="rounded-full bg-slate-100 px-1.5 py-0 text-[11px] font-bold text-slate-600">
              {sessions.length}
            </span>
          )}
        </div>
        <button
          type="button"
          onClick={() => {
            void loadSessions()
            if (!guruCatalog) void loadGuruCatalog()
          }}
          disabled={loadingSessions}
          className="rounded-lg px-1.5 py-0.5 text-[11px] font-semibold text-sky-700 transition-colors hover:bg-sky-50 disabled:opacity-50"
        >
          {loadingSessions ? 'กำลังโหลด...' : 'รีเฟรช'}
        </button>
      </div>

      <div className="min-h-0 flex-1 space-y-1.5 overflow-y-auto pr-0.5 custom-scrollbar">
        {loadingSessions ? (
          <div className="space-y-2 p-1" aria-label="กำลังโหลดรายการประวัติ">
            {[0, 1, 2].map((idx) => (
              <div
                key={idx}
                className="h-12 animate-pulse rounded-xl border border-slate-100 bg-slate-100/70"
              />
            ))}
            <p className="text-center text-[11px] text-slate-500">กำลังโหลดรายการ...</p>
          </div>
        ) : sessionsError ? (
          <div className="space-y-2 rounded-xl border border-rose-200 bg-rose-50/70 p-2.5 text-center">
            <p className="text-[11px] font-medium leading-snug text-rose-800">{sessionsError}</p>
            <button
              type="button"
              onClick={() => {
                void loadSessions()
                if (!guruCatalog) void loadGuruCatalog()
              }}
              className="rounded-lg border border-rose-300 bg-white px-2.5 py-1 text-[11px] font-bold text-rose-800 transition-colors hover:bg-rose-50"
            >
              ↻ ลองโหลดใหม่
            </button>
          </div>
        ) : sessions.length === 0 ? (
          <p className="p-2 text-center text-xs text-slate-500">ยังไม่มีห้องประชุมก่อนหน้า</p>
        ) : (
          sessions.map((s) => {
            const selected = activeDetail?.session.session_id === s.session_id
            const loadingThis = loadingSessionId === s.session_id
            return (
              <button
                key={s.session_id}
                type="button"
                disabled={loadingThis}
                onClick={() => void handleSelectSession(s.session_id)}
                className={`w-full rounded-xl border p-2.5 text-left transition-all ${
                  selected
                    ? 'border-sky-300 bg-sky-50/90 shadow-sm ring-1 ring-sky-200/70'
                    : 'border-slate-100/90 bg-white/90 hover:border-sky-200 hover:bg-sky-50/40'
                }`}
              >
                <div className="flex items-center justify-between gap-1.5">
                  <span className="truncate text-xs font-bold text-slate-900">{s.title}</span>
                  <span
                    className={`shrink-0 rounded-md border px-1.5 py-0 text-[11px] font-bold ${stanceBadgeClass(
                      s.final_rating,
                    )}`}
                  >
                    {loadingThis ? '...' : s.final_rating}
                  </span>
                </div>
                <div className="mt-1 flex items-center justify-between gap-1.5">
                  <span className="truncate text-[11px] text-slate-500">
                    {s.primary_ticker ? `$${s.primary_ticker}` : s.agenda_prompt}
                  </span>
                  {s.settled_alpha_pct && (
                    <span className="shrink-0 text-[11px] font-semibold text-emerald-700">
                      Alpha {s.settled_alpha_pct}%
                    </span>
                  )}
                </div>
              </button>
            )
          })
        )}
      </div>
    </>
  )

  const resolvedPortfolioAttachmentState =
    activeDetail?.portfolio_attachment_state ??
    (contextRefs.some((r) => r.toLowerCase() === '@portfolio')
      ? 'ATTACHED_PRESENT'
      : 'NOT_ATTACHED_DO_NOT_ASSUME')

  const effectiveMissingPrerequisites = dismissedPrereqs
    ? []
    : (activeDetail?.missing_prerequisite_tickers ?? [])
  const effectiveContextDrift = dismissedDrift
    ? false
    : Boolean(activeDetail?.context_drift_detected)

  const hasRightHudContent = Boolean(
    visibleDecisionPlaybook ||
      activeDetail?.past_scorecard ||
      effectiveMissingPrerequisites.length > 0 ||
      effectiveContextDrift ||
      visiblePillarCoverage,
  )
  const is3DHudMode = stageViewMode === '3d'
  const showRightHudPanel = rightHudExpanded || (hasRightHudContent && !rightHudCollapsed)
  const hasActiveStreamFeed =
    isStreaming || visibleTurns.length > 0 || toolRecords.length > 0
  const portfolioGuardBadge = portfolioAttachmentLabel(resolvedPortfolioAttachmentState)

  const renderStudioHeaderBar = (floatingHud: boolean) => (
    <header
      className={`flex flex-wrap items-center justify-between gap-2 rounded-2xl border px-3.5 backdrop-blur-xl ${
        floatingHud
          ? 'border-sky-200/80 bg-white/86 py-1.5 shadow-lg'
          : 'border-sky-200/70 bg-white/80 py-2 shadow-sm'
      }`}
    >
      <div className="flex min-w-0 items-center gap-2.5">
        {/* Desktop rail collapse + Mobile drawer trigger */}
        <button
          type="button"
          aria-label="เปิดหรือซ่อนประวัติห้องประชุม"
          onClick={() => {
            if (typeof window !== 'undefined' && window.innerWidth < 1024) {
              setMobileHistoryOpen((prev) => !prev)
            } else {
              setShowHistoryRail((prev) => !prev)
            }
          }}
          className={`inline-flex shrink-0 items-center gap-1.5 rounded-xl border px-2.5 py-1 text-xs font-semibold transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-section-accent/45 ${
            showHistoryRail || mobileHistoryOpen
              ? 'border-sky-300 bg-sky-50 text-sky-900'
              : 'border-slate-200 bg-white text-slate-700 hover:bg-slate-50'
          }`}
        >
          <ScrollHistoryIcon className="h-3.5 w-3.5 text-section-accent" />
          <span className="hidden sm:inline">ประวัติ</span>
          {sessions.length > 0 && (
            <span className="rounded-full bg-white/90 px-1.5 text-[11px] font-mono font-bold text-sky-800 shadow-sm">
              {sessions.length}
            </span>
          )}
        </button>

        <div className="min-w-0">
          <div className="flex items-center gap-2">
            <h1 className="truncate text-sm font-extrabold tracking-tight text-slate-900 sm:text-base">
              🎙️ Trinity Guru Roundtable Studio
            </h1>
            {activeDetail?.session.primary_ticker && (
              <span className="rounded-lg border border-amber-200 bg-amber-50 px-2 py-0.5 font-mono text-[11px] font-bold text-amber-900">
                ${activeDetail.session.primary_ticker}
              </span>
            )}
          </div>
          {!floatingHud && (
            <p className="truncate text-[11px] text-slate-500">
              {activeDetail ? activeDetail.session.title : 'สภากูรู 5 มุมมอง + Hermes Moderator'}
            </p>
          )}
        </div>
      </div>

      <div className="flex flex-wrap items-center gap-1.5 sm:gap-2">
        {floatingHud && (
          <div
            role="group"
            aria-label="สลับมุมมองเวทีสภากูรู"
            className="inline-flex items-center rounded-xl border border-sky-200/80 bg-sky-50/80 p-0.5"
          >
            <button
              type="button"
              aria-pressed={stageViewMode === '3d'}
              onClick={() => handleStageViewModeChange('3d')}
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
              onClick={() => handleStageViewModeChange('2d')}
              className={`rounded-lg px-2 py-0.5 text-[11px] font-semibold transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-500/45 ${
                stageViewMode === '2d'
                  ? 'bg-sky-600 text-white shadow-sm'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              🪪 ปกติ (2D)
            </button>
          </div>
        )}

        <button
          type="button"
          onClick={() => openCustomizationFor('value_guru')}
          className="inline-flex items-center gap-1.5 rounded-xl border border-slate-200/90 bg-white px-2.5 py-1 text-xs font-semibold text-slate-700 transition-colors hover:border-sky-300 hover:bg-sky-50 hover:text-sky-900 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-section-accent/45"
        >
          <SlidersIcon className="h-3.5 w-3.5 text-slate-500" />
          <span>ปรับแต่งกูรู & เสียง</span>
        </button>
        <button
          type="button"
          onClick={() => {
            streamAbortRef.current?.abort()
            voicePlayer.clearQueue()
            setPacedSessionId(null)
            setIsStreaming(false)
            setIsNewMode(true)
            setActiveDetail(null)
            setToolRecords([])
            setRightHudExpanded(false)
            setRightHudCollapsed(false)
            setStreamHudCollapsed(false)
          }}
          className="rounded-xl border border-sky-300 bg-sky-50 px-3 py-1 text-xs font-bold text-sky-900 transition-colors hover:bg-sky-100"
        >
          + เปิดประชุมใหม่
        </button>
      </div>
    </header>
  )

  const stageElement = (
    <GuruAvatarStage
      personas={effectivePersonas}
      turns={visibleTurns}
      prioritizedSpeakers={activeDetail?.prioritized_speakers ?? []}
      userRoleModels={activeDetail?.user_role_models ?? []}
      activeSpeakerId={voicePlayer.activeSpeakerId}
      thinkingSpeakerId={visibleThinkingSpeakerId}
      thinkingStage={thinkingStage}
      mutedSpeakers={voicePlayer.mutedSpeakers}
      primaryTicker={activeDetail?.session.primary_ticker}
      portfolioAttachmentState={resolvedPortfolioAttachmentState}
      finalRating={visibleDecisionPlaybook?.final_rating ?? activeDetail?.session.final_rating}
      pillarCoverage={visiblePillarCoverage}
      hasRecentSandboxMath={toolRecords.length > 0 || (isInterjecting && runSandboxMath)}
      voiceEnabled={voicePlayer.voiceEnabled}
      getVoiceEnergy={voicePlayer.getVoiceEnergy}
      stageViewMode={stageViewMode}
      hudLayout={is3DHudMode}
      onChangeStageViewMode={handleStageViewModeChange}
      onToggleVoiceEnabled={voicePlayer.setVoiceEnabled}
      onToggleMute={voicePlayer.toggleSpeakerMute}
      onMentionSpeaker={handleMentionSpeaker}
      onCustomizeSpeaker={(sid) => openCustomizationFor(sid)}
    />
  )

  const streamElement = (
    <RoundtableStreamPanel
      turns={visibleTurns}
      personas={effectivePersonas}
      revealTurnIds={revealTurnIds}
      toolRecords={toolRecords}
      isStreaming={isStreaming}
      thinkingSpeakerId={visibleThinkingSpeakerId}
      thinkingStage={thinkingStage}
      activeTurnId={voicePlayer.activeTurnId}
      voiceEnabled={voicePlayer.voiceEnabled}
      onToggleVoiceEnabled={voicePlayer.setVoiceEnabled}
      isPlaying={voicePlayer.isPlaying}
      isPaused={voicePlayer.isPaused}
      isAwaitingTts={voicePlayer.isAwaitingTts}
      queueCount={voicePlayer.queueCount}
      syncWithVoice={syncWithVoice}
      bufferedAheadCount={bufferedAheadCount}
      totalTurnCount={allTurns.length}
      onToggleSyncWithVoice={() => setSyncWithVoice((v) => !v)}
      playbackRate={voicePlayer.playbackRate}
      onChangePlaybackRate={voicePlayer.setPlaybackRate}
      onPauseOrResume={voicePlayer.pauseOrResume}
      onSkipAudio={voicePlayer.skipCurrent}
      onPlayTurn={voicePlayer.playTurnNow}
      onCopyTurn={() => showNotice('คัดลอกข้อความกูรูแล้ว', 'success')}
      onTurnFeedback={handleTurnFeedback}
      onQuickThreadReply={submitInterjectionText}
      onCollapseHud={is3DHudMode ? () => setStreamHudCollapsed(true) : undefined}
    />
  )

  const composerElement = (
    <RaiseHandComposer
      mode={isNewMode || !activeDetail ? 'new' : 'active'}
      personas={effectivePersonas}
      agendaPrompt={agendaPrompt}
      onAgendaPromptChange={setAgendaPrompt}
      contextRefs={contextRefs}
      onToggleContextRef={handleToggleContextRef}
      onAddTickerRef={handleAddTickerRef}
      debateIntensity={debateIntensity}
      onDebateIntensityChange={setDebateIntensity}
      debateDepth={debateDepth}
      onDebateDepthChange={setDebateDepth}
      onToggleSeatEnabled={handleToggleSeatEnabled}
      onOpenCustomization={() => openCustomizationFor('value_guru')}
      onStartNewSession={() => {
        setRightHudCollapsed(false)
        setStreamHudCollapsed(false)
        void handleStartNewSession()
      }}
      isCreating={isCreating}
      interjectText={interjectText}
      onInterjectTextChange={setInterjectText}
      targetSpeaker={targetSpeaker}
      onTargetSpeakerChange={setTargetSpeaker}
      runSandboxMath={runSandboxMath}
      onRunSandboxMathChange={setRunSandboxMath}
      onSubmitInterjection={() => {
        setStreamHudCollapsed(false)
        void submitInterjectionText(interjectText)
      }}
      isInterjecting={isInterjecting}
      isStreaming={isStreaming}
      canResumeDebate={canResumeDebate}
      compactHud={is3DHudMode}
      onResumeDebate={() =>
        activeDetail && void startDebateStream(activeDetail.session.session_id)
      }
    />
  )

  const playbookElement = (
    <DecisionPlaybookPanel
      key={activeDetail?.session.session_id ?? 'empty'}
      personas={effectivePersonas}
      primaryTicker={activeDetail?.session.primary_ticker}
      portfolioAttachmentState={resolvedPortfolioAttachmentState}
      missingPrerequisiteTickers={effectiveMissingPrerequisites}
      contextDriftDetected={effectiveContextDrift}
      pastScorecard={activeDetail?.past_scorecard}
      decisionPlaybook={visibleDecisionPlaybook}
      pillarCoverage={visiblePillarCoverage}
      onFocusSpeakerTurn={(speakerId) => {
        if (is3DHudMode) setStreamHudCollapsed(false)
        const target = [...visibleTurns].reverse().find((turn) => turn.speaker_id === speakerId)
        if (target) {
          document
            .getElementById(`rt-turn-${target.turn_id}`)
            ?.scrollIntoView({ behavior: 'smooth', block: 'center' })
        }
      }}
      onDispatchTicker={(ticker) => void handleDispatchTicker(ticker)}
      onDismissMissingPrerequisites={() => {
        setDismissedPrereqs(true)
        if (activeDetail && activeDetail.turns.length === 0) {
          void startDebateStream(activeDetail.session.session_id)
        }
      }}
      onRefreshContextOnDrift={() => {
        setDismissedDrift(true)
        if (activeDetail) {
          void startDebateStream(activeDetail.session.session_id, true)
        }
      }}
      onDismissDrift={() => setDismissedDrift(true)}
      onSaveToVault={() => void handleSaveToVault()}
      onUpdateThesis={() => void handleUpdateThesis()}
      onExportYoutubePack={() => void handleExportYoutubePack()}
      hasTurns={Boolean(activeDetail && visibleTurns.length > 0)}
      isSavingVault={isSavingVault}
      isUpdatingThesis={isUpdatingThesis}
      isExportingYoutube={isExportingYoutube}
      onCollapseHud={
        is3DHudMode
          ? () => {
              setRightHudExpanded(false)
              setRightHudCollapsed(true)
            }
          : undefined
      }
    />
  )

  const mobileDrawerAndNotice = (
    <>
      {/* Mobile History Drawer (<lg) */}
      {mobileHistoryOpen && (
        <div
          className="fixed inset-0 z-40 flex lg:hidden"
          role="dialog"
          aria-modal="true"
          aria-label="ประวัติห้องประชุม"
        >
          <button
            type="button"
            aria-label="ปิดหน้าต่างประวัติห้องประชุม"
            className="fixed inset-0 bg-slate-900/25 backdrop-blur-sm"
            onClick={() => setMobileHistoryOpen(false)}
          />
          <aside className="relative z-10 flex h-full w-72 max-w-[85vw] flex-col border-r border-sky-200 bg-white/95 p-3 shadow-xl backdrop-blur-md">
            {historyListContent}
          </aside>
        </div>
      )}

      {/* Inline Toast Notice */}
      {notice && (
        <div
          role="status"
          className={`flex items-center justify-between gap-2 rounded-xl border px-3.5 py-2 text-xs font-medium ${
            is3DHudMode
              ? 'pointer-events-auto absolute inset-x-3 top-[60px] z-40 mx-auto max-w-xl shadow-lg backdrop-blur-xl'
              : 'shadow-sm'
          } ${
            notice.type === 'error'
              ? 'border-rose-200 bg-rose-50/95 text-rose-900'
              : notice.type === 'success'
                ? 'border-emerald-200 bg-emerald-50/95 text-emerald-900'
                : 'border-sky-200 bg-sky-50/95 text-sky-900'
          }`}
        >
          <span>{notice.text}</span>
          <button
            type="button"
            onClick={() => setNotice(null)}
            className="rounded px-1.5 py-0.5 text-xs opacity-70 hover:opacity-100"
          >
            ✕
          </button>
        </div>
      )}
    </>
  )

  return (
    <MotionConfig reducedMotion="user">
      {is3DHudMode ? (
        <div className="relative h-full w-full overflow-hidden bg-[#122638]">
          {/* Layer 0: Full-Bleed 3D Roblox Stage */}
          <div className="absolute inset-0 z-0">{stageElement}</div>

          {/* Layer 1: Floating Translucent Studio Top Bar */}
          <div className="pointer-events-none absolute inset-x-3 top-2.5 z-30">
            <div className="pointer-events-auto">{renderStudioHeaderBar(true)}</div>
          </div>

          {mobileDrawerAndNotice}

          {/* Layer 2 Left Stack: Floating History Rail + Active Live Transcript Feed */}
          {(showHistoryRail || hasActiveStreamFeed) && (
            <div className="pointer-events-none absolute bottom-[148px] left-3 top-[184px] z-20 flex w-[min(328px,calc(50vw-1rem))] flex-col gap-2 xl:w-[360px]">
              {showHistoryRail && (
                <aside
                  className={`pointer-events-auto hidden min-h-0 w-[224px] shrink-0 flex-col rounded-2xl border border-sky-200/80 bg-white/86 p-2.5 shadow-xl backdrop-blur-xl lg:flex ${
                    hasActiveStreamFeed && !streamHudCollapsed ? 'max-h-[38%]' : 'max-h-full'
                  }`}
                >
                  {historyListContent}
                </aside>
              )}

              {hasActiveStreamFeed &&
                (streamHudCollapsed ? (
                  <button
                    type="button"
                    onClick={() => setStreamHudCollapsed(false)}
                    className="pointer-events-auto inline-flex items-center gap-1.5 self-start rounded-xl border border-sky-200/85 bg-white/88 px-3 py-1.5 text-[11px] font-bold text-slate-800 shadow-lg backdrop-blur-xl hover:border-sky-300 hover:bg-white"
                  >
                    <span>💬 บทสนทนา ({visibleTurns.length})</span>
                    <span className="text-sky-700">▴ กาง</span>
                  </button>
                ) : (
                  <div className="pointer-events-auto flex min-h-0 flex-1 flex-col rounded-2xl shadow-xl">
                    {streamElement}
                  </div>
                ))}
            </div>
          )}

          {/* Layer 2 Right Dock / Floating Guard & Playbook Drawer */}
          {showRightHudPanel ? (
            <div className="pointer-events-auto absolute bottom-[148px] right-3 top-[184px] z-20 flex w-[min(328px,calc(50vw-1rem))] flex-col rounded-2xl border border-sky-200/80 bg-white/90 p-2.5 shadow-xl backdrop-blur-xl xl:w-[356px]">
              {playbookElement}
            </div>
          ) : (
            <div className="pointer-events-none absolute right-3 top-[184px] z-20">
              <button
                type="button"
                onClick={() => {
                  setRightHudCollapsed(false)
                  setRightHudExpanded(true)
                }}
                title={portfolioGuardBadge.detail}
                className="pointer-events-auto inline-flex items-center gap-2 rounded-xl border border-sky-200/85 bg-white/86 px-3 py-1.5 text-[11px] font-bold text-slate-800 shadow-lg backdrop-blur-xl transition-colors hover:border-sky-300 hover:bg-white"
              >
                <span className="text-slate-600">🛡️ {portfolioGuardBadge.label}</span>
                <span className="text-sky-700">⚖️ Playbook ▾</span>
              </button>
            </div>
          )}

          {/* Layer 2 Bottom Centered Command Dock */}
          <div className="pointer-events-none absolute inset-x-3 bottom-2.5 z-20 flex justify-center">
            <div className="pointer-events-auto w-full max-w-[900px]">{composerElement}</div>
          </div>
        </div>
      ) : (
        <div className="flex h-full min-h-0 flex-col gap-3 overflow-hidden p-3 sm:p-4">
          {renderStudioHeaderBar(false)}
          {mobileDrawerAndNotice}

          {/* Classic 3-Column 2D Workspace */}
          <div
            className={`grid min-h-0 flex-1 grid-cols-1 gap-3 ${
              showHistoryRail
                ? 'lg:grid-cols-[204px_minmax(0,1fr)_324px] xl:grid-cols-[216px_minmax(0,1fr)_350px]'
                : 'lg:grid-cols-[minmax(0,1fr)_340px] xl:grid-cols-[minmax(0,1fr)_372px]'
            }`}
          >
            {/* Left Sidebar: Collapsible Past Sessions */}
            {showHistoryRail && (
              <aside className="hidden min-h-0 flex-col rounded-2xl border border-sky-200/75 bg-white/80 p-3 shadow-sm backdrop-blur-md lg:flex">
                {historyListContent}
              </aside>
            )}

            {/* Center Stage: Avatars + Debate Stream + Raise Hand Composer */}
            <section className="flex min-h-0 flex-col gap-3">
              {stageElement}
              {streamElement}
              {composerElement}
            </section>

            {/* Right Panel: Guard Alerts, Scorecard & Decision Playbook */}
            {playbookElement}
          </div>
        </div>
      )}

      <GuruCustomizationModal
        isOpen={isCustomizeOpen}
        onClose={() => setIsCustomizeOpen(false)}
        personas={baseConfiguredPersonas}
        voices={guruCatalog?.voices ?? []}
        presets={guruCatalog?.presets ?? []}
        initialSpeakerId={customizeInitialSeat}
        isSaving={isSavingProfiles}
        isResetting={isResettingProfiles}
        isPreviewing={voicePlayer.isPreviewingVoice}
        onSaveAsDefault={handleSaveProfilesAsDefault}
        onApplyForSessionOnly={handleApplyProfilesForSessionOnly}
        onResetDefaults={handleResetProfiles}
        onPreviewVoice={handlePreviewVoice}
      />
    </MotionConfig>
  )
}

