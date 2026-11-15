import { useEffect, useMemo, useState } from 'react'
import type {
  CoreGuruIdType,
  GuruPersonaDTO,
  GuruPresetItemDTO,
  UpdateGuruPersonaInputDTO,
  VoiceCatalogItemDTO,
  VoicePreviewRequestDTO,
} from '../../api/types'
import {
  AlertTriangleIcon,
  CheckCircleIcon,
  CheckSmallIcon,
  CloseIcon,
  CompassIcon,
  MicStageIcon,
  RefreshIcon,
  SlidersIcon,
  SparklesIcon,
  VolumeOnIcon,
} from '../ui/AppIcons'
import {
  STAGE_SEAT_ORDER,
  getSpeakerMeta,
  parseAnalyticalRubricLines,
} from './personas'

interface GuruCustomizationModalProps {
  isOpen: boolean
  onClose: () => void
  personas: GuruPersonaDTO[]
  voices: VoiceCatalogItemDTO[]
  presets: GuruPresetItemDTO[]
  initialSpeakerId?: string | null
  isSaving?: boolean
  isResetting?: boolean
  isPreviewing?: boolean
  onSaveAsDefault: (profiles: UpdateGuruPersonaInputDTO[]) => Promise<void> | void
  onApplyForSessionOnly: (profiles: UpdateGuruPersonaInputDTO[]) => void
  onResetDefaults: (speakerId?: CoreGuruIdType) => Promise<void> | void
  onPreviewVoice: (req: VoicePreviewRequestDTO) => Promise<void> | void
}

const ACCENT_COLOR_SWATCHES = [
  '#0ea5e9',
  '#10b981',
  '#6366f1',
  '#f59e0b',
  '#ec4899',
  '#ef4444',
  '#14b8a6',
  '#8b5cf6',
]

const EMOJI_CHOICES = ['🪶', '🏛️', '💎', '🚀', '🌊', '🛡️', '⚖️', '📐', '🌱', '🔬', '⚡', '📈', '🌐', '🎯', '🔎', '⚠️']

interface EditableSeatDraft {
  speaker_id: CoreGuruIdType
  display_name: string
  short_name: string
  archetype: string
  speaking_style: string
  custom_directive: string
  preset_id: string
  voice_name: string
  pitch: number
  rate: number
  accent_color: string
  emoji: string
  enabled: boolean
  debate_order: number
}

function buildInitialDrafts(personas: GuruPersonaDTO[]): EditableSeatDraft[] {
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
  }).sort((a, b) => a.debate_order - b.debate_order)
}

function genderToneBadge(tone: string): { label: string; className: string } {
  if (tone === 'female') {
    return { label: 'เสียงหญิง', className: 'border-pink-200 bg-pink-50 text-pink-800' }
  }
  if (tone === 'male') {
    return { label: 'เสียงชาย', className: 'border-sky-200 bg-sky-50 text-sky-800' }
  }
  return { label: 'กลาง', className: 'border-slate-200 bg-slate-50 text-slate-700' }
}

export function GuruCustomizationModal({
  isOpen,
  onClose,
  personas,
  voices,
  presets,
  initialSpeakerId,
  isSaving = false,
  isResetting = false,
  isPreviewing = false,
  onSaveAsDefault,
  onApplyForSessionOnly,
  onResetDefaults,
  onPreviewVoice,
}: GuruCustomizationModalProps) {
  const [drafts, setDrafts] = useState<EditableSeatDraft[]>(() => buildInitialDrafts(personas))
  const [selectedSeatId, setSelectedSeatId] = useState<CoreGuruIdType>(
    (initialSpeakerId as CoreGuruIdType) || 'value_guru',
  )
  const [sampleText, setSampleText] = useState<string>('')
  const [customSampleEdited, setCustomSampleEdited] = useState(false)
  const [localError, setLocalError] = useState<string | null>(null)

  useEffect(() => {
    if (!isOpen) return
    setDrafts(buildInitialDrafts(personas))
    if (
      initialSpeakerId &&
      STAGE_SEAT_ORDER.includes(initialSpeakerId as (typeof STAGE_SEAT_ORDER)[number])
    ) {
      setSelectedSeatId(initialSpeakerId as CoreGuruIdType)
    }
    setCustomSampleEdited(false)
    setLocalError(null)
  }, [isOpen, personas, initialSpeakerId])

  useEffect(() => {
    if (!isOpen) return
    const onKeyDown = (ev: KeyboardEvent) => {
      if (ev.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [isOpen, onClose])

  const selectedSeat = useMemo(
    () => drafts.find((d) => d.speaker_id === selectedSeatId) ?? drafts[0]!,
    [drafts, selectedSeatId],
  )

  const selectedVoiceOption = useMemo(
    () => voices.find((v) => v.voice_name === selectedSeat.voice_name),
    [voices, selectedSeat.voice_name],
  )

  useEffect(() => {
    if (!isOpen || customSampleEdited) return
    setSampleText(
      selectedVoiceOption?.default_sample_text ||
        `สวัสดีครับ ผมคือ ${selectedSeat.display_name} พร้อมวิเคราะห์พอร์ตกับคุณแล้ว`,
    )
  }, [isOpen, customSampleEdited, selectedVoiceOption, selectedSeat.display_name])

  const seatPresets = useMemo(
    () => presets.filter((p) => p.speaker_id === selectedSeat.speaker_id),
    [presets, selectedSeat.speaker_id],
  )

  const selectedSeatContext = useMemo(() => {
    const baseMeta = getSpeakerMeta(selectedSeat.speaker_id, personas)
    const matchedPreset = seatPresets.find((p) => p.preset_id === selectedSeat.preset_id)
    return {
      roleLensLabel: matchedPreset?.role_lens_label || baseMeta.roleLensLabel,
      focusSummary: matchedPreset?.focus_summary || baseMeta.focusSummary,
      primaryEvidenceLabels:
        matchedPreset?.primary_evidence_labels && matchedPreset.primary_evidence_labels.length > 0
          ? matchedPreset.primary_evidence_labels
          : baseMeta.primaryEvidenceLabels,
      analyticalRubric: matchedPreset?.analytical_rubric || baseMeta.analyticalRubric,
      rubricLines: parseAnalyticalRubricLines(
        matchedPreset?.analytical_rubric || baseMeta.analyticalRubric,
      ),
    }
  }, [selectedSeat.speaker_id, selectedSeat.preset_id, seatPresets, personas])

  const activePanelCount = useMemo(
    () => drafts.filter((d) => d.speaker_id !== 'hermes_moderator' && d.enabled).length,
    [drafts],
  )

  const skepticSeat = drafts.find((d) => d.speaker_id === 'skeptic_guru')
  const skepticDisabled = skepticSeat ? !skepticSeat.enabled : false

  if (!isOpen) return null

  const updateSeat = (speakerId: CoreGuruIdType, patch: Partial<EditableSeatDraft>) => {
    setLocalError(null)
    setDrafts((prev) =>
      prev.map((d) => (d.speaker_id === speakerId ? { ...d, ...patch } : d)),
    )
  }

  const handleToggleEnabled = (speakerId: CoreGuruIdType) => {
    if (speakerId === 'hermes_moderator') return
    const target = drafts.find((d) => d.speaker_id === speakerId)
    if (!target) return
    if (target.enabled && activePanelCount <= 2) {
      setLocalError('ต้องเปิดใช้งานกูรูอย่างน้อย 2 ท่านเข้าร่วมวง (ไม่รวม Hermes Moderator)')
      return
    }
    updateSeat(speakerId, { enabled: !target.enabled })
  }

  const handleMoveSeat = (speakerId: CoreGuruIdType, direction: -1 | 1) => {
    if (speakerId === 'hermes_moderator') return
    setDrafts((prev) => {
      const hermes = prev.find((d) => d.speaker_id === 'hermes_moderator')
      const panel = prev
        .filter((d) => d.speaker_id !== 'hermes_moderator')
        .sort((a, b) => a.debate_order - b.debate_order)
      const idx = panel.findIndex((d) => d.speaker_id === speakerId)
      const swapIdx = idx + direction
      if (idx < 0 || swapIdx < 0 || swapIdx >= panel.length) return prev
      const nextPanel = [...panel]
      const tmp = nextPanel[idx]!
      nextPanel[idx] = nextPanel[swapIdx]!
      nextPanel[swapIdx] = tmp
      const normalized = nextPanel.map((item, i) => ({ ...item, debate_order: i + 1 }))
      return hermes ? [{ ...hermes, debate_order: 0 }, ...normalized] : normalized
    })
  }

  const handleApplyPreset = (preset: GuruPresetItemDTO) => {
    const matchedVoice = voices.find((v) => v.voice_name === preset.recommended_voice)
    setCustomSampleEdited(false)
    updateSeat(selectedSeat.speaker_id, {
      preset_id: preset.preset_id,
      display_name: preset.display_name,
      short_name: preset.short_name,
      archetype: preset.archetype,
      speaking_style: preset.speaking_style,
      custom_directive: preset.custom_directive,
      emoji: preset.emoji,
      voice_name: preset.recommended_voice,
      pitch: matchedVoice?.default_pitch ?? selectedSeat.pitch,
      rate: matchedVoice?.default_rate ?? selectedSeat.rate,
    })
  }

  const handleSelectVoice = (voice: VoiceCatalogItemDTO) => {
    setCustomSampleEdited(false)
    updateSeat(selectedSeat.speaker_id, {
      voice_name: voice.voice_name,
      pitch: voice.default_pitch,
      rate: voice.default_rate,
    })
  }

  const toUpdatePayloads = (): UpdateGuruPersonaInputDTO[] =>
    drafts.map((d) => ({
      speaker_id: d.speaker_id,
      display_name: d.display_name.trim() || d.short_name.trim(),
      short_name: d.short_name.trim() || d.display_name.trim(),
      archetype: d.archetype.trim(),
      speaking_style: d.speaking_style.trim(),
      custom_directive: d.custom_directive.trim(),
      preset_id: d.preset_id || 'default',
      voice_name: d.voice_name,
      pitch: d.pitch,
      rate: d.rate,
      accent_color: d.accent_color,
      emoji: d.emoji.trim() || '🎙️',
      enabled: d.speaker_id === 'hermes_moderator' ? true : d.enabled,
      debate_order: d.speaker_id === 'hermes_moderator' ? 0 : d.debate_order,
    }))

  const validateBeforeSubmit = (): boolean => {
    if (activePanelCount < 2) {
      setLocalError('ต้องเปิดใช้งานกูรูอย่างน้อย 2 ท่านเข้าร่วมวง (ไม่รวม Hermes Moderator)')
      return false
    }
    for (const d of drafts) {
      if (!d.display_name.trim() || !d.short_name.trim()) {
        setLocalError(`กรุณากรอกชื่อเต็มและชื่อย่อของ ${d.speaker_id}`)
        return false
      }
    }
    return true
  }

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/45 p-3 sm:p-5 backdrop-blur-sm"
      role="presentation"
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose()
      }}
    >
      <div
        role="dialog"
        aria-modal="true"
        aria-labelledby="guru-customization-title"
        className="flex max-h-[92vh] w-full max-w-5xl flex-col overflow-hidden rounded-2xl border border-sky-200 bg-white shadow-2xl"
      >
        {/* Modal Header */}
        <div className="flex flex-wrap items-center justify-between gap-2 border-b border-sky-100 bg-gradient-to-r from-sky-50 to-cyan-50/60 px-4 py-3">
          <div>
            <h2
              id="guru-customization-title"
              className="flex items-center gap-2 text-base font-extrabold text-sky-950"
            >
              <SlidersIcon className="h-4 w-4 text-sky-700 shrink-0" />
              <span>ปรับแต่งกูรู & เสียงพากย์ประจำสภา</span>
            </h2>
            <p className="text-xs text-slate-600">
              ปรับชื่อ บุคลิก แนวคำถามเฉพาะตัว และเลือกเสียง Gemini TTS จาก 12 โทนเสียง (เปิดใช้อยู่ {activePanelCount}/5 กูรู + Hermes)
            </p>
          </div>
          <button
            type="button"
            onClick={onClose}
            aria-label="ปิดหน้าต่างปรับแต่งกูรู"
            className="inline-flex items-center gap-1 rounded-lg border border-slate-200 bg-white px-2.5 py-1 text-xs font-semibold text-slate-600 hover:bg-slate-50 hover:text-slate-900"
          >
            <CloseIcon className="h-3.5 w-3.5" aria-hidden="true" />
            <span>ปิด</span>
          </button>
        </div>

        {/* Validation / Skeptic Safety Banners */}
        {(localError || skepticDisabled) && (
          <div className="space-y-1.5 border-b border-slate-100 bg-slate-50/70 px-4 py-2">
            {localError && (
              <div className="rounded-xl border border-rose-200 bg-rose-50 px-3 py-1.5 text-xs font-semibold text-rose-800">
                {localError}
              </div>
            )}
            {skepticDisabled && (
              <div className="flex items-start gap-2 rounded-xl border border-amber-200 bg-amber-50 px-3 py-1.5 text-xs font-medium text-amber-900">
                <AlertTriangleIcon className="mt-0.5 h-3.5 w-3.5 text-amber-600 shrink-0" />
                <div>
                  คุณกำลังปิดที่นั่ง <strong>Skeptic</strong> — ระบบจะให้{' '}
                  <strong>Hermes Moderator</strong> ทำหน้าที่เป็น Devil&apos;s Advocate ตรวจสอบคุณภาพงบ, หนี้สิน และความเสี่ยงขาลงให้โดยอัตโนมัติ
                </div>
              </div>
            )}
          </div>
        )}

        {/* Body: 6-Seat List + Active Seat Editor */}
        <div className="grid min-h-0 flex-1 grid-cols-1 overflow-hidden md:grid-cols-[260px_minmax(0,1fr)]">
          {/* Left Rail: 6 Seats */}
          <aside className="flex flex-col gap-1.5 overflow-y-auto border-b border-sky-100 bg-sky-50/30 p-3 md:border-b-0 md:border-r custom-scrollbar">
            <div className="mb-1 text-[11px] font-bold uppercase tracking-wider text-slate-500">
              ที่นั่งสภากูรู (ลำดับการพูด)
            </div>
            {drafts
              .slice()
              .sort((a, b) => a.debate_order - b.debate_order)
              .map((seat) => {
                const isSelected = seat.speaker_id === selectedSeat.speaker_id
                const isHermes = seat.speaker_id === 'hermes_moderator'
                const seatMeta = getSpeakerMeta(seat.speaker_id, personas)
                const liveSeatPreset = presets.find(
                  (pr) => pr.speaker_id === seat.speaker_id && pr.preset_id === seat.preset_id,
                )
                const liveRoleLens = liveSeatPreset?.role_lens_label || seatMeta.roleLensLabel
                return (
                  <div
                    key={seat.speaker_id}
                    className={`rounded-xl border p-2 transition-all ${
                      isSelected
                        ? 'border-sky-400 bg-white shadow-sm ring-1 ring-sky-200'
                        : seat.enabled
                          ? 'border-slate-200/90 bg-white/80 hover:border-sky-200'
                          : 'border-slate-200/70 bg-slate-100/70 opacity-65'
                    }`}
                  >
                    <div className="flex items-center justify-between gap-1.5">
                      <button
                        type="button"
                        onClick={() => {
                          setSelectedSeatId(seat.speaker_id)
                          setCustomSampleEdited(false)
                        }}
                        className="flex min-w-0 flex-1 items-center gap-2 text-left"
                      >
                        <span
                          className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full border text-sm"
                          style={{
                            borderColor: seat.accent_color,
                            backgroundColor: `${seat.accent_color}18`,
                          }}
                        >
                          {seat.emoji}
                        </span>
                        <div className="min-w-0">
                          <div className="truncate text-xs font-bold text-slate-900">
                            {isHermes ? '0.' : `${seat.debate_order}.`} {seat.short_name}
                          </div>
                          <div className="truncate text-[11px] font-medium text-sky-800/85">
                            {liveRoleLens}
                          </div>
                          <div className="truncate text-[11px] text-slate-500">
                            🎙️ {seat.voice_name}
                          </div>
                        </div>
                      </button>

                      <div className="flex shrink-0 items-center gap-1">
                        {!isHermes && (
                          <>
                            <button
                              type="button"
                              onClick={() => handleMoveSeat(seat.speaker_id, -1)}
                              disabled={seat.debate_order <= 1}
                              aria-label={`เลื่อน ${seat.short_name} ขึ้น`}
                              className="rounded border border-slate-200 bg-white px-1 py-0.5 text-[11px] text-slate-600 hover:bg-slate-50 disabled:opacity-35"
                              title="เลื่อนลำดับพูดก่อนหน้า"
                            >
                              ▲
                            </button>
                            <button
                              type="button"
                              onClick={() => handleMoveSeat(seat.speaker_id, 1)}
                              disabled={seat.debate_order >= 5}
                              aria-label={`เลื่อน ${seat.short_name} ลง`}
                              className="rounded border border-slate-200 bg-white px-1 py-0.5 text-[11px] text-slate-600 hover:bg-slate-50 disabled:opacity-35"
                              title="เลื่อนลำดับพูดถัดไป"
                            >
                              ▼
                            </button>
                            <input
                              type="checkbox"
                              checked={seat.enabled}
                              onChange={() => handleToggleEnabled(seat.speaker_id)}
                              aria-label={`เปิดใช้งาน ${seat.short_name}`}
                              className="ml-0.5 h-3.5 w-3.5 cursor-pointer accent-sky-600"
                            />
                          </>
                        )}
                        {isHermes && (
                          <span className="rounded bg-sky-100 px-1.5 py-0.5 text-[11px] font-semibold text-sky-800">
                            เปิดเสมอ
                          </span>
                        )}
                      </div>
                    </div>
                  </div>
                )
              })}
          </aside>

          {/* Right Panel: Selected Seat Customization */}
          <div className="min-h-0 flex-1 overflow-y-auto p-4 space-y-4 custom-scrollbar">
            {/* Presets Section */}
            {seatPresets.length > 0 && (
              <section className="space-y-1.5">
                <div className="flex items-center justify-between">
                  <h3 className="flex items-center gap-1.5 text-xs font-bold uppercase tracking-wider text-slate-600">
                    <SparklesIcon className="h-3.5 w-3.5 text-sky-600" />
                    <span>แม่แบบบุคลิกสำเร็จรูป ({seatPresets.length} รูปแบบ)</span>
                  </h3>
                  <span className="text-[11px] text-slate-500">คลิกเพื่อสวมบทบาททันที</span>
                </div>
                <div className="grid grid-cols-1 gap-2 sm:grid-cols-3">
                  {seatPresets.map((preset) => {
                    const active = selectedSeat.preset_id === preset.preset_id
                    return (
                      <button
                        key={preset.preset_id}
                        type="button"
                        onClick={() => handleApplyPreset(preset)}
                        className={`rounded-xl border p-2.5 text-left transition-all ${
                          active
                            ? 'border-sky-400 bg-sky-50/90 shadow-sm'
                            : 'border-slate-200 bg-white hover:border-sky-200 hover:bg-sky-50/30'
                        }`}
                      >
                        <div className="flex items-center justify-between gap-1">
                          <span className="text-xs font-bold text-slate-900">
                            {preset.emoji} {preset.short_name}
                          </span>
                          <span className="rounded bg-slate-100 px-1.5 py-0.5 text-[11px] font-medium text-slate-600">
                            {preset.recommended_voice}
                          </span>
                        </div>
                        <p className="mt-1 line-clamp-1 text-[11px] font-semibold text-slate-800">
                          {preset.label_th}
                        </p>
                        {preset.speaking_style && (
                          <p className="mt-0.5 line-clamp-2 text-[11px] leading-snug text-slate-500">
                            {preset.speaking_style}
                          </p>
                        )}
                      </button>
                    )
                  })}
                </div>
              </section>
            )}

            {/* Backend Analytical Context & Rubric Explanation */}
            <section
              aria-label="บริบทและกติกาวิเคราะห์ของกูรู"
              className="space-y-2 rounded-xl border border-blue-200/80 bg-blue-50/35 p-3"
            >
              <div className="flex flex-wrap items-center justify-between gap-2">
                <h3 className="flex items-center gap-1.5 text-xs font-bold uppercase tracking-wider text-blue-950">
                  <CompassIcon className="h-3.5 w-3.5 text-blue-700" />
                  <span>บริบทและกติกาวิเคราะห์ที่ระบบส่งให้กูรูนี้</span>
                </h3>
                {selectedSeatContext.roleLensLabel && (
                  <span
                    className="rounded-full border px-2 py-0.5 text-[11px] font-bold"
                    style={{
                      borderColor: `${selectedSeat.accent_color}55`,
                      backgroundColor: `${selectedSeat.accent_color}14`,
                      color: selectedSeat.accent_color,
                    }}
                  >
                    {selectedSeatContext.roleLensLabel}
                  </span>
                )}
              </div>

              {selectedSeatContext.focusSummary && (
                <p className="text-xs leading-relaxed text-slate-700">
                  {selectedSeatContext.focusSummary}
                </p>
              )}

              {selectedSeatContext.primaryEvidenceLabels.length > 0 && (
                <div className="flex flex-wrap items-center gap-1.5">
                  <span className="text-[11px] font-semibold text-slate-500">
                    แหล่งข้อมูลหลัก:
                  </span>
                  {selectedSeatContext.primaryEvidenceLabels.map((lbl) => (
                    <span
                      key={lbl}
                      className="rounded-md border border-blue-200 bg-white px-1.5 py-0.5 text-[11px] font-medium text-blue-900"
                    >
                      {lbl}
                    </span>
                  ))}
                </div>
              )}

              {selectedSeatContext.rubricLines.length > 0 && (
                <div className="rounded-lg border border-blue-100 bg-white/90 p-2.5 space-y-1 text-[11px] leading-snug text-slate-700">
                  {selectedSeatContext.rubricLines.map((rl, idx) => (
                    <p key={`modal-rubric-${idx}`}>
                      {rl.title ? (
                        <>
                          <strong className="text-blue-950">{rl.title}:</strong> {rl.body}
                        </>
                      ) : (
                        rl.body
                      )}
                    </p>
                  ))}
                </div>
              )}
            </section>

            {/* Identity & Visuals */}
            <section className="space-y-2.5 rounded-xl border border-slate-200/90 bg-slate-50/40 p-3">
              <h3 className="flex items-center gap-1.5 text-xs font-bold uppercase tracking-wider text-slate-600">
                <SparklesIcon className="h-3.5 w-3.5 text-sky-600" />
                <span>ชื่อ บทบาท และธีมการ์ด</span>
              </h3>
              <div className="grid grid-cols-1 gap-2.5 sm:grid-cols-2">
                <label className="space-y-1 text-xs">
                  <span className="font-semibold text-slate-700">ชื่อแสดงเต็ม (Display Name)</span>
                  <input
                    type="text"
                    maxLength={60}
                    value={selectedSeat.display_name}
                    onChange={(e) =>
                      updateSeat(selectedSeat.speaker_id, { display_name: e.target.value })
                    }
                    className="w-full rounded-lg border border-slate-200 bg-white px-2.5 py-1.5 text-xs text-slate-900 outline-none focus:border-sky-400"
                  />
                </label>
                <label className="space-y-1 text-xs">
                  <span className="font-semibold text-slate-700">
                    ชื่อย่อสำหรับ @แท็กถาม (Short Name)
                  </span>
                  <input
                    type="text"
                    maxLength={24}
                    value={selectedSeat.short_name}
                    onChange={(e) =>
                      updateSeat(selectedSeat.speaker_id, { short_name: e.target.value })
                    }
                    className="w-full rounded-lg border border-slate-200 bg-white px-2.5 py-1.5 text-xs text-slate-900 outline-none focus:border-sky-400"
                  />
                </label>
              </div>

              <label className="block space-y-1 text-xs">
                <span className="font-semibold text-slate-700">บทบาทหลัก (Archetype)</span>
                <input
                  type="text"
                  maxLength={80}
                  value={selectedSeat.archetype}
                  onChange={(e) =>
                    updateSeat(selectedSeat.speaker_id, { archetype: e.target.value })
                  }
                  className="w-full rounded-lg border border-slate-200 bg-white px-2.5 py-1.5 text-xs text-slate-900 outline-none focus:border-sky-400"
                />
              </label>

              <div className="flex flex-wrap items-center justify-between gap-3 pt-1">
                <div className="flex flex-wrap items-center gap-1">
                  <span className="mr-1 text-xs font-semibold text-slate-700">ไอคอน:</span>
                  {EMOJI_CHOICES.map((em) => (
                    <button
                      key={em}
                      type="button"
                      onClick={() => updateSeat(selectedSeat.speaker_id, { emoji: em })}
                      className={`h-7 w-7 rounded-lg border text-sm transition-colors ${
                        selectedSeat.emoji === em
                          ? 'border-sky-400 bg-sky-100'
                          : 'border-slate-200 bg-white hover:bg-slate-50'
                      }`}
                    >
                      {em}
                    </button>
                  ))}
                </div>
                <div className="flex items-center gap-1.5">
                  <span className="text-xs font-semibold text-slate-700">สีธีม:</span>
                  {ACCENT_COLOR_SWATCHES.map((hex) => (
                    <button
                      key={hex}
                      type="button"
                      onClick={() => updateSeat(selectedSeat.speaker_id, { accent_color: hex })}
                      aria-label={`เลือกสี ${hex}`}
                      className={`h-5 w-5 rounded-full border-2 transition-transform ${
                        selectedSeat.accent_color.toLowerCase() === hex.toLowerCase()
                          ? 'scale-110 border-slate-900'
                          : 'border-white shadow-sm'
                      }`}
                      style={{ backgroundColor: hex }}
                    />
                  ))}
                </div>
              </div>
            </section>

            {/* Persona Prompt & Style Overrides */}
            <section className="space-y-2.5 rounded-xl border border-slate-200/90 bg-slate-50/40 p-3">
              <h3 className="flex items-center gap-1.5 text-xs font-bold uppercase tracking-wider text-slate-600">
                <MicStageIcon className="h-3.5 w-3.5 text-sky-600" />
                <span>สไตล์การพูดและแนวคำสั่งเสริม</span>
              </h3>
              <label className="block space-y-1 text-xs">
                <span className="font-semibold text-slate-700">
                  โทนและจังหวะการพูด (Speaking Style)
                </span>
                <input
                  type="text"
                  maxLength={200}
                  value={selectedSeat.speaking_style}
                  onChange={(e) =>
                    updateSeat(selectedSeat.speaker_id, { speaking_style: e.target.value })
                  }
                  placeholder="เช่น สุขุม รอบคอบ เน้นเหตุผลเชิงตัวเลขและส่วนเผื่อความปลอดภัย"
                  className="w-full rounded-lg border border-slate-200 bg-white px-2.5 py-1.5 text-xs text-slate-900 outline-none focus:border-sky-400"
                />
              </label>
              <label className="block space-y-1 text-xs">
                <span className="font-semibold text-slate-700">
                  คำสั่งเพิ่มเติมสำหรับกูรูคนนี้ (Custom Directive — ไม่เกิน 500 ตัวอักษร)
                </span>
                <textarea
                  rows={2}
                  maxLength={500}
                  value={selectedSeat.custom_directive}
                  onChange={(e) =>
                    updateSeat(selectedSeat.speaker_id, { custom_directive: e.target.value })
                  }
                  placeholder="เช่น ให้ตั้งคำถามเรื่อง Free Cash Flow Yield และ Dilution เสมอก่อนสรุปจุดยืน"
                  className="w-full resize-none rounded-lg border border-slate-200 bg-white px-2.5 py-1.5 text-xs text-slate-900 outline-none focus:border-sky-400"
                />
                <span className="block text-[11px] text-slate-500">
                  หมายเหตุ: แม้เว้นช่องนี้ว่าง (เช่น ค่าเริ่มต้น) ระบบยังส่งกติกาและกรอบข้อมูลเฉพาะทางด้านบนให้กูรูนี้ครบทุกครั้ง — ช่องนี้ใช้เสริมคำสั่งเพิ่มเติมเฉพาะของคุณ
                </span>
              </label>
            </section>

            {/* Gemini TTS Voice Selection + Descriptions + Preview */}
            <section className="space-y-2.5 rounded-xl border border-sky-200/90 bg-sky-50/30 p-3">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <h3 className="flex items-center gap-1.5 text-xs font-bold uppercase tracking-wider text-sky-900">
                  <VolumeOnIcon className="h-3.5 w-3.5 text-sky-700" />
                  <span>เลือกเสียงพากย์ Gemini TTS (12 โทนเสียง)</span>
                </h3>
                <span className="rounded-full border border-sky-200 bg-white px-2 py-0.5 text-[11px] font-semibold text-sky-800">
                  เลือกอยู่: {selectedSeat.voice_name}
                </span>
              </div>

              <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
                {voices.map((v) => {
                  const active = selectedSeat.voice_name === v.voice_name
                  const tone = genderToneBadge(v.gender_tone)
                  return (
                    <button
                      key={v.voice_name}
                      type="button"
                      onClick={() => handleSelectVoice(v)}
                      className={`rounded-xl border p-2.5 text-left transition-all ${
                        active
                          ? 'border-sky-500 bg-white shadow-sm ring-1 ring-sky-200'
                          : 'border-slate-200 bg-white/90 hover:border-sky-300'
                      }`}
                    >
                      <div className="flex items-center justify-between gap-2">
                        <span className="inline-flex items-center gap-1 text-xs font-extrabold text-slate-900">
                          {active && <CheckSmallIcon className="h-3.5 w-3.5 text-sky-600" />}
                          <span>{v.voice_name}</span>
                        </span>
                        <span
                          className={`rounded-full border px-2 py-0 text-[11px] font-semibold ${tone.className}`}
                        >
                          {tone.label}
                        </span>
                      </div>
                      <p className="mt-1 text-[11px] leading-snug text-slate-700">{v.label_th}</p>
                      <p className="mt-1 text-[11px] font-medium text-sky-700">
                        เหมาะกับ: {v.recommended_for}
                      </p>
                    </button>
                  )
                })}
              </div>

              {/* Pitch / Rate Sliders + Voice Preview Player */}
              <div className="rounded-xl border border-sky-200 bg-white p-3 space-y-2.5">
                <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
                  <div className="space-y-1 text-xs">
                    <div className="flex items-center justify-between font-semibold text-slate-700">
                      <label htmlFor="guru-voice-pitch-slider">
                        ระดับเสียง Web Speech (Pitch)
                      </label>
                      <span className="font-mono text-sky-700">{selectedSeat.pitch.toFixed(2)}</span>
                    </div>
                    <input
                      id="guru-voice-pitch-slider"
                      type="range"
                      min={0.75}
                      max={1.25}
                      step={0.01}
                      value={selectedSeat.pitch}
                      onChange={(e) =>
                        updateSeat(selectedSeat.speaker_id, { pitch: Number(e.target.value) })
                      }
                      className="w-full accent-sky-600"
                    />
                  </div>
                  <div className="space-y-1 text-xs">
                    <div className="flex items-center justify-between font-semibold text-slate-700">
                      <label htmlFor="guru-voice-rate-slider">ความเร็วเสียงพูด (Rate)</label>
                      <span className="font-mono text-sky-700">{selectedSeat.rate.toFixed(2)}x</span>
                    </div>
                    <input
                      id="guru-voice-rate-slider"
                      type="range"
                      min={0.8}
                      max={1.25}
                      step={0.01}
                      value={selectedSeat.rate}
                      onChange={(e) =>
                        updateSeat(selectedSeat.speaker_id, { rate: Number(e.target.value) })
                      }
                      className="w-full accent-sky-600"
                    />
                  </div>
                </div>

                <div className="flex flex-col gap-2 sm:flex-row sm:items-center">
                  <input
                    type="text"
                    maxLength={280}
                    value={sampleText}
                    onChange={(e) => {
                      setCustomSampleEdited(true)
                      setSampleText(e.target.value)
                    }}
                    aria-label="ข้อความตัวอย่างสำหรับทดสอบเสียง"
                    placeholder="พิมพ์ข้อความทดสอบเสียง..."
                    className="h-9 flex-1 rounded-lg border border-slate-200 bg-slate-50/70 px-2.5 text-xs text-slate-800 outline-none focus:border-sky-400 focus:bg-white"
                  />
                  <button
                    type="button"
                    disabled={isPreviewing}
                    onClick={() =>
                      void onPreviewVoice({
                        speaker_id: selectedSeat.speaker_id,
                        voice_name: selectedSeat.voice_name,
                        sample_text: sampleText.trim() || undefined,
                        pitch: selectedSeat.pitch,
                        rate: selectedSeat.rate,
                      })
                    }
                    className="inline-flex h-9 shrink-0 items-center justify-center gap-1.5 rounded-lg border border-sky-400 bg-sky-600 px-3 text-xs font-bold text-white shadow-sm transition-colors hover:bg-sky-700 disabled:opacity-50"
                  >
                    {isPreviewing ? (
                      <>
                        <RefreshIcon className="h-3.5 w-3.5 animate-spin" />
                        <span>กำลังเล่นตัวอย่าง...</span>
                      </>
                    ) : (
                      <>
                        <VolumeOnIcon className="h-3.5 w-3.5" />
                        <span>ทดลองฟังเสียงตัวอย่าง</span>
                      </>
                    )}
                  </button>
                </div>
              </div>
            </section>
          </div>
        </div>

        {/* Modal Footer Actions */}
        <div className="flex flex-wrap items-center justify-between gap-2 border-t border-sky-100 bg-slate-50/90 px-4 py-3">
          <div className="flex flex-wrap items-center gap-2">
            <button
              type="button"
              disabled={isResetting || isSaving}
              onClick={() => void onResetDefaults(selectedSeat.speaker_id)}
              className="inline-flex items-center gap-1.5 rounded-xl border border-slate-300 bg-white px-3 py-1.5 text-xs font-semibold text-slate-700 hover:bg-slate-100 disabled:opacity-50"
            >
              <RefreshIcon className="h-3.5 w-3.5 text-slate-500" />
              <span>รีเซ็ตเฉพาะ {selectedSeat.short_name}</span>
            </button>
            <button
              type="button"
              disabled={isResetting || isSaving}
              onClick={() => void onResetDefaults(undefined)}
              className="inline-flex items-center gap-1.5 rounded-xl border border-slate-300 bg-white px-3 py-1.5 text-xs font-semibold text-slate-700 hover:bg-slate-100 disabled:opacity-50"
            >
              <RefreshIcon className="h-3.5 w-3.5 text-slate-500" />
              <span>รีเซ็ตทั้งหมดเป็นค่าเริ่มต้น</span>
            </button>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            <button
              type="button"
              onClick={() => {
                if (!validateBeforeSubmit()) return
                onApplyForSessionOnly(toUpdatePayloads())
              }}
              className="rounded-xl border border-sky-300 bg-white px-3.5 py-1.5 text-xs font-bold text-sky-900 hover:bg-sky-50"
            >
              ใช้เฉพาะห้องประชุมนี้
            </button>
            <button
              type="button"
              disabled={isSaving}
              onClick={() => {
                if (!validateBeforeSubmit()) return
                void onSaveAsDefault(toUpdatePayloads())
              }}
              className="inline-flex items-center gap-1.5 rounded-xl bg-gradient-to-r from-sky-600 to-cyan-600 px-4 py-1.5 text-xs font-bold text-white shadow-sm hover:from-sky-700 hover:to-cyan-700 disabled:opacity-50"
            >
              {isSaving ? (
                <>
                  <RefreshIcon className="h-3.5 w-3.5 animate-spin" />
                  <span>กำลังบันทึก...</span>
                </>
              ) : (
                <>
                  <CheckCircleIcon className="h-3.5 w-3.5" />
                  <span>บันทึกเป็นค่าเริ่มต้น</span>
                </>
              )}
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}
