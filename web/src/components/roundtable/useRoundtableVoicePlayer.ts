import { useCallback, useEffect, useRef, useState } from 'react'
import { api } from '../../api/client'
import type {
  AudioStatusType,
  GuruPersonaDTO,
  RoundtableTurnDTO,
  VoicePreviewRequestDTO,
  VoicePreviewResponseDTO,
} from '../../api/types'
import { getSpeakerMeta } from './personas'

export type PlaybackSpeed = 1 | 1.25 | 1.5
export const TTS_PENDING_WAIT_TIMEOUT_MS = 15000

export function getTtsPendingWaitMs(textLength: number): number {
  return Math.max(TTS_PENDING_WAIT_TIMEOUT_MS, Math.min(28000, 14000 + textLength * 18))
}

export function getMaxPlausibleWavDurationSec(textLength: number): number {
  return Math.max(45, Math.max(0, textLength) * 0.25 + 25)
}

const CYRILLIC_TO_LATIN: Record<string, string> = {
  а: 'a', б: 'b', в: 'v', г: 'g', д: 'd', е: 'e', ё: 'yo',
  ж: 'zh', з: 'z', и: 'i', й: 'y', к: 'k', л: 'l', м: 'm',
  н: 'n', о: 'o', п: 'p', р: 'r', с: 's', т: 't', у: 'u',
  ф: 'f', х: 'h', ц: 'ts', ч: 'ch', ш: 'sh', щ: 'shch', ъ: '',
  ы: 'y', ь: '', э: 'e', ю: 'yu', я: 'ya', і: 'i', ї: 'yi',
  є: 'ye', ґ: 'g',
}

/**
 * Clean markdown, stray Cyrillic homoglyphs, and awkward symbols before Web Speech synthesis.
 */
export function prepareTextForWebSpeech(text: string): string {
  let cleaned = (text || '')
    .replace(/```[\s\S]*?```/g, '')
    .replace(/[*_`#>|]+/g, '')
    .replace(/\[(?:FACT|MATH|INFERRED|DOCUMENTED|UNVERIFIED)[^\]]*\]/gi, '')
    .replace(/อ[гГ][рР][еЕ][сС]/g, 'อเกรส')
    .replace(/[\u0400-\u04FF]+/g, (seg) =>
      Array.from(seg)
        .map((ch) => CYRILLIC_TO_LATIN[ch.toLowerCase()] ?? '')
        .join(''),
    )
    .replace(/(?<!\w)@([A-Za-z\u0E00-\u0E7F_]+)/g, '$1')
    .replace(/\$([A-Z]{1,6}(?:\.[A-Z]{1,4})?)/g, '$1')
    .replace(/(\d+(?:,\d{3})*(?:\.\d+)?)\s*%/g, '$1 เปอร์เซ็นต์')
    .replace(/\s+/g, ' ')
    .trim()
  return cleaned
}

/**
 * Clamp pitch safely so Thai Web Speech synthesis avoids robotic/alien distortion.
 */
export function clampWebSpeechPitch(pitch: number, isThai: boolean): number {
  if (!Number.isFinite(pitch)) return 1.0
  return isThai
    ? Math.min(1.03, Math.max(0.97, pitch))
    : Math.min(1.15, Math.max(0.85, pitch))
}

/**
 * Split long dialogue into short sentence/clause chunks so Chromium Web Speech
 * neither stalls nor fires premature onend events on 400-1000+ char strings.
 */
export function splitTextIntoSpeechChunks(text: string, maxChunkChars = 180): string[] {
  const normalized = (text || '').replace(/\s+/g, ' ').trim()
  if (!normalized) return []
  if (normalized.length <= maxChunkChars) return [normalized]

  const segmenterCtor =
    typeof Intl !== 'undefined' && 'Segmenter' in Intl
      ? (Intl as unknown as { Segmenter: new (loc?: string, opts?: { granularity: string }) => { segment: (s: string) => Iterable<{ segment: string }> } }).Segmenter
      : null
  const wordSeg = segmenterCtor ? new segmenterCtor('th', { granularity: 'word' }) : null

  const rawSentences = normalized
    .split(/(?<=[.!?\n])\s+|\s+(?=แต่|ส่วน|ขณะที่|ดังนั้น|เพราะ|อย่างไรก็ตาม)/)
    .map((s) => s.trim())
    .filter(Boolean)

  const chunks: string[] = []
  let current = ''

  const pushPiecesOfOverlongUnit = (unit: string) => {
    const tokens = wordSeg
      ? Array.from(wordSeg.segment(unit), (x) => x.segment)
      : unit.split(/(\s+)/)
    for (const tok of tokens) {
      if (!tok) continue
      if (tok.length > maxChunkChars) {
        if (current.trim()) {
          chunks.push(current.trim())
          current = ''
        }
        for (let i = 0; i < tok.length; i += maxChunkChars) {
          chunks.push(tok.slice(i, i + maxChunkChars).trim())
        }
        continue
      }
      if ((current + tok).trim().length > maxChunkChars && current.trim()) {
        chunks.push(current.trim())
        current = tok.trimStart()
      } else {
        current += tok
      }
    }
  }

  for (const sent of rawSentences) {
    const candidate = current ? `${current} ${sent}` : sent
    if (candidate.length <= maxChunkChars) {
      current = candidate
    } else if (sent.length <= maxChunkChars) {
      if (current.trim()) chunks.push(current.trim())
      current = sent
    } else {
      if (current.trim()) {
        chunks.push(current.trim())
        current = ''
      }
      pushPiecesOfOverlongUnit(sent)
    }
  }
  if (current.trim()) chunks.push(current.trim())
  return chunks.filter(Boolean)
}

function pickNaturalSpeechVoice(langPrefix: string): SpeechSynthesisVoice | null {
  if (typeof window === 'undefined' || !('speechSynthesis' in window)) return null
  const getVoices = window.speechSynthesis.getVoices
  if (typeof getVoices !== 'function') return null
  const voices = getVoices.call(window.speechSynthesis) || []
  const prefix = langPrefix.toLowerCase()
  const matching = voices.filter((v) => (v.lang || '').toLowerCase().startsWith(prefix))
  if (matching.length === 0) return null
  const preferred = matching.find((v) =>
    /(natural|online|google|premwadee|niwat|pattara)/i.test(v.name || ''),
  )
  return preferred ?? matching[0] ?? null
}

interface QueueItem {
  turnId: string
  speakerId: string
  text: string
  audioCacheKey: string | null
  audioStatus: AudioStatusType
  awaitingTts: boolean
  pitchOverride?: number
  rateOverride?: number
  isPreview?: boolean
}

export function useRoundtableVoicePlayer(personas?: GuruPersonaDTO[]) {
  const [voiceEnabled, setVoiceEnabledState] = useState(true)
  const [playbackRate, setPlaybackRateState] = useState<PlaybackSpeed>(1)
  const [mutedSpeakers, setMutedSpeakers] = useState<Set<string>>(() => new Set())
  const [activeTurnId, setActiveTurnId] = useState<string | null>(null)
  const [activeSpeakerId, setActiveSpeakerId] = useState<string | null>(null)
  const [isPaused, setIsPaused] = useState(false)
  const [isPreviewingVoice, setIsPreviewingVoice] = useState(false)
  const [isAwaitingTts, setIsAwaitingTts] = useState(false)
  const [queueCount, setQueueCount] = useState(0)

  const queueRef = useRef<QueueItem[]>([])
  const currentItemRef = useRef<QueueItem | null>(null)
  const playingRef = useRef(false)
  const previewingRef = useRef(false)
  const pausedRef = useRef(false)
  const voiceEnabledRef = useRef(true)
  const rateRef = useRef<PlaybackSpeed>(1)
  const mutedRef = useRef<Set<string>>(new Set())
  const personasRef = useRef<GuruPersonaDTO[] | undefined>(personas)
  personasRef.current = personas

  const audioElRef = useRef<HTMLAudioElement | null>(null)
  const currentResolveRef = useRef<(() => void) | null>(null)
  const ttsWaitersRef = useRef<Map<string, () => void>>(new Map())
  const playbackEpochRef = useRef(0)

  const audioCtxRef = useRef<AudioContext | null>(null)
  const analyserRef = useRef<AnalyserNode | null>(null)
  const analyserDataRef = useRef<Uint8Array<ArrayBuffer> | null>(null)
  const mediaSourceRef = useRef<MediaElementAudioSourceNode | null>(null)
  const smoothedEnergyRef = useRef(0)

  const syncQueueCount = useCallback(() => {
    setQueueCount(queueRef.current.length)
  }, [])

  const detachAudioMeter = useCallback(() => {
    if (mediaSourceRef.current) {
      try {
        mediaSourceRef.current.disconnect()
      } catch {
        // ignore WebAudio disconnect error
      }
      mediaSourceRef.current = null
    }
    smoothedEnergyRef.current = 0
  }, [])

  const attachAudioMeter = useCallback((audio: HTMLAudioElement) => {
    if (typeof window === 'undefined') return
    const AudioCtxCtor =
      window.AudioContext ??
      (window as Window & { webkitAudioContext?: typeof AudioContext }).webkitAudioContext
    if (!AudioCtxCtor) return
    try {
      let ctx = audioCtxRef.current
      if (!ctx || ctx.state === 'closed') {
        ctx = new AudioCtxCtor()
        audioCtxRef.current = ctx
      }
      if (ctx.state === 'suspended') {
        void ctx.resume().catch(() => {})
      }
      let analyser = analyserRef.current
      if (!analyser || analyser.context !== ctx) {
        analyser = ctx.createAnalyser()
        analyser.fftSize = 256
        analyser.smoothingTimeConstant = 0.65
        analyser.connect(ctx.destination)
        analyserRef.current = analyser
        analyserDataRef.current = new Uint8Array(new ArrayBuffer(analyser.frequencyBinCount))
      }
      if (mediaSourceRef.current) {
        try {
          mediaSourceRef.current.disconnect()
        } catch {
          // ignore
        }
        mediaSourceRef.current = null
      }
      const source = ctx.createMediaElementSource(audio)
      source.connect(analyser)
      mediaSourceRef.current = source
    } catch {
      // Never block playback if WebAudio API is unavailable or restricted
    }
  }, [])

  const getVoiceEnergy = useCallback((): number => {
    if (pausedRef.current) {
      smoothedEnergyRef.current = 0
      return 0
    }
    const audio = audioElRef.current
    const analyser = analyserRef.current
    const buf = analyserDataRef.current
    if (audio && !audio.paused && !audio.ended && analyser && buf && mediaSourceRef.current) {
      analyser.getByteTimeDomainData(buf)
      let sumSq = 0
      for (let i = 0; i < buf.length; i++) {
        const centered = ((buf[i] ?? 128) - 128) / 128
        sumSq += centered * centered
      }
      const rms = Math.sqrt(sumSq / Math.max(1, buf.length))
      const raw = Math.min(1, rms * 4.4)
      const smoothed = smoothedEnergyRef.current * 0.55 + raw * 0.45
      smoothedEnergyRef.current = smoothed
      return smoothed
    }
    // Natural rhythmic syllable envelope when Web Speech synthesis is speaking
    // or when AudioContext metering is unavailable
    if (playingRef.current || previewingRef.current) {
      const t = (typeof performance !== 'undefined' ? performance.now() : Date.now()) / 1000
      const cadence = 0.44 + 0.34 * Math.sin(t * 14.2) * Math.sin(t * 4.9) + 0.18 * Math.sin(t * 22.7)
      return Math.max(0.08, Math.min(1, cadence))
    }
    smoothedEnergyRef.current = 0
    return 0
  }, [])

  const stopActivePlaybackOnly = useCallback(() => {
    detachAudioMeter()
    if (audioElRef.current) {
      try {
        audioElRef.current.pause()
        audioElRef.current.onended = null
        audioElRef.current.onerror = null
        audioElRef.current.src = ''
      } catch {
        // ignore audio teardown error
      }
      audioElRef.current = null
    }
    if (typeof window !== 'undefined' && 'speechSynthesis' in window) {
      try {
        window.speechSynthesis.cancel()
      } catch {
        // ignore speechSynthesis error
      }
    }
    const resolve = currentResolveRef.current
    currentResolveRef.current = null
    resolve?.()
  }, [detachAudioMeter])

  const clearQueue = useCallback(() => {
    playbackEpochRef.current += 1
    queueRef.current = []
    for (const waiter of ttsWaitersRef.current.values()) {
      waiter()
    }
    ttsWaitersRef.current.clear()
    pausedRef.current = false
    setIsPaused(false)
    setIsAwaitingTts(false)
    syncQueueCount()
    stopActivePlaybackOnly()
    playingRef.current = false
    currentItemRef.current = null
    setActiveTurnId(null)
    setActiveSpeakerId(null)
  }, [stopActivePlaybackOnly, syncQueueCount])

  useEffect(() => {
    return () => {
      clearQueue()
      if (audioCtxRef.current && audioCtxRef.current.state !== 'closed') {
        void audioCtxRef.current.close().catch(() => {})
      }
      audioCtxRef.current = null
      analyserRef.current = null
      analyserDataRef.current = null
    }
  }, [clearQueue])

  const setVoiceEnabled = useCallback(
    (enabled: boolean) => {
      voiceEnabledRef.current = enabled
      setVoiceEnabledState(enabled)
      if (!enabled) {
        clearQueue()
      }
    },
    [clearQueue],
  )

  const setPlaybackRate = useCallback((rate: PlaybackSpeed) => {
    rateRef.current = rate
    setPlaybackRateState(rate)
    if (audioElRef.current) {
      audioElRef.current.playbackRate = rate
    }
  }, [])

  const toggleSpeakerMute = useCallback((speakerId: string) => {
    setMutedSpeakers((prev) => {
      const next = new Set(prev)
      if (next.has(speakerId)) {
        next.delete(speakerId)
      } else {
        next.add(speakerId)
      }
      mutedRef.current = next
      return next
    })
  }, [])

  const speakWebSpeech = useCallback(async (item: QueueItem): Promise<void> => {
    if (
      typeof window === 'undefined' ||
      !('speechSynthesis' in window) ||
      typeof SpeechSynthesisUtterance === 'undefined'
    ) {
      return
    }
    const cleanedText = prepareTextForWebSpeech(item.text)
    if (!cleanedText) return
    const chunks = splitTextIntoSpeechChunks(cleanedText)
    if (chunks.length === 0) return

    const startEpoch = playbackEpochRef.current
    const meta = getSpeakerMeta(item.speakerId, personasRef.current)
    const isThai = /[ก-๙]/.test(cleanedText)
    const rawPitch = item.pitchOverride ?? meta.pitch
    const safePitch = clampWebSpeechPitch(rawPitch, isThai)
    const baseRate = item.rateOverride ?? meta.rate
    const safeRate = Math.min(2, Math.max(0.75, baseRate * rateRef.current))
    const selectedVoice = pickNaturalSpeechVoice(isThai ? 'th' : 'en')

    const speakChunk = (chunkText: string): Promise<void> =>
      new Promise((resolve) => {
        let settled = false
        let utter: SpeechSynthesisUtterance | null = null
        let timer = 0

        const finish = (timedOut: boolean = false) => {
          if (settled) return
          settled = true
          if (timer) {
            window.clearTimeout(timer)
            timer = 0
          }
          if (utter) {
            utter.onend = null
            utter.onerror = null
          }
          if (currentResolveRef.current === forceStop) {
            currentResolveRef.current = null
          }
          if (timedOut && typeof window !== 'undefined' && 'speechSynthesis' in window) {
            try {
              window.speechSynthesis.cancel()
            } catch {
              // ignore cancel error
            }
          }
          resolve()
        }
        const forceStop = () => finish(false)
        currentResolveRef.current = forceStop

        try {
          utter = new SpeechSynthesisUtterance(chunkText)
          utter.lang = /[ก-๙]/.test(chunkText) ? 'th-TH' : 'en-US'
          utter.pitch = safePitch
          utter.rate = safeRate
          if (selectedVoice) {
            utter.voice = selectedVoice
          }
          utter.onend = () => finish(false)
          utter.onerror = () => finish(false)
          window.speechSynthesis.speak(utter)
          const chunkMaxMs = Math.max(8000, Math.min(45000, Math.round((chunkText.length * 180) / safeRate)))
          timer = window.setTimeout(() => finish(true), chunkMaxMs)
        } catch {
          finish(false)
        }
      })

    for (const chunk of chunks) {
      if (playbackEpochRef.current !== startEpoch) break
      if (!item.isPreview && (!voiceEnabledRef.current || pausedRef.current)) break
      await speakChunk(chunk)
    }
  }, [])

  const playWavUrl = useCallback(
    (item: QueueItem): Promise<boolean> => {
      return new Promise((resolve) => {
        if (!item.audioCacheKey || typeof Audio === 'undefined') {
          resolve(false)
          return
        }
        // Defensively stop any lingering Web Speech synthesis before starting Gemini WAV
        if (typeof window !== 'undefined' && 'speechSynthesis' in window) {
          try {
            window.speechSynthesis.cancel()
          } catch {
            // ignore
          }
        }
        let settled = false
        let createdAudio: HTMLAudioElement | null = null
        const finish = (ok: boolean) => {
          if (settled) return
          settled = true
          if (createdAudio) {
            createdAudio.onloadedmetadata = null
            createdAudio.ontimeupdate = null
            createdAudio.onended = null
            createdAudio.onerror = null
          }
          if (currentResolveRef.current === forceResolve) {
            currentResolveRef.current = null
          }
          detachAudioMeter()
          audioElRef.current = null
          resolve(ok)
        }
        const forceResolve = () => finish(true)
        currentResolveRef.current = forceResolve

        try {
          const audio = new Audio(api.getRoundtableAudioUrl(item.audioCacheKey))
          createdAudio = audio
          audioElRef.current = audio
          audio.playbackRate = item.rateOverride ?? rateRef.current
          if ('preservesPitch' in audio) {
            ; (audio as HTMLAudioElement & { preservesPitch: boolean }).preservesPitch = true
          }
          const trimmedLen = item.text.trim().length
          const maxDurationSec = getMaxPlausibleWavDurationSec(trimmedLen)
          const abortIfCorrupt = () => {
            try {
              audio.pause()
            } catch {
              // ignore
            }
            finish(false)
          }
          audio.onloadedmetadata = () => {
            if (trimmedLen >= 40 && Number.isFinite(audio.duration) && audio.duration > maxDurationSec) {
              abortIfCorrupt()
            }
          }
          audio.ontimeupdate = () => {
            if (trimmedLen >= 40 && Number.isFinite(audio.currentTime) && audio.currentTime > maxDurationSec) {
              abortIfCorrupt()
            }
          }
          audio.onended = () => finish(true)
          audio.onerror = () => finish(false)
          attachAudioMeter(audio)
          const p = audio.play()
          if (p && typeof p.catch === 'function') {
            p.catch(() => finish(false))
          }
        } catch {
          finish(false)
        }
      })
    },
    [attachAudioMeter, detachAudioMeter],
  )

  const waitForTtsIfPending = useCallback(async (item: QueueItem) => {
    if (!item.awaitingTts) return
    const waitMs = getTtsPendingWaitMs(item.text.length)
    setIsAwaitingTts(true)
    try {
      await new Promise<void>((resolve) => {
        const done = () => {
          window.clearTimeout(timer)
          ttsWaitersRef.current.delete(item.turnId)
          resolve()
        }
        const timer = window.setTimeout(done, waitMs)
        ttsWaitersRef.current.set(item.turnId, done)
      })
    } finally {
      setIsAwaitingTts(false)
    }
  }, [])

  const pumpQueue = useCallback(async () => {
    if (playingRef.current) return
    const myEpoch = playbackEpochRef.current
    playingRef.current = true
    try {
      while (queueRef.current.length > 0) {
        if (playbackEpochRef.current !== myEpoch) break
        if (!voiceEnabledRef.current || pausedRef.current) {
          break
        }
        const next = queueRef.current.shift()
        syncQueueCount()
        if (!next) continue
        if (next.speakerId === 'user' || mutedRef.current.has(next.speakerId)) {
          continue
        }

        currentItemRef.current = next
        setActiveTurnId(next.turnId)
        setActiveSpeakerId(next.speakerId)

        if (next.awaitingTts) {
          await waitForTtsIfPending(next)
        }
        if (playbackEpochRef.current !== myEpoch) break
        if (!voiceEnabledRef.current || pausedRef.current) {
          queueRef.current.unshift(next)
          syncQueueCount()
          currentItemRef.current = null
          break
        }
        if (mutedRef.current.has(next.speakerId)) {
          currentItemRef.current = null
          continue
        }

        if (
          (next.audioStatus === 'ready' || next.audioStatus === 'cached') &&
          next.audioCacheKey
        ) {
          const ok = await playWavUrl(next)
          if (playbackEpochRef.current !== myEpoch) break
          if (!ok && voiceEnabledRef.current && !pausedRef.current) {
            await speakWebSpeech(next)
          }
        } else if (next.audioStatus !== 'disabled') {
          await speakWebSpeech(next)
        }
        if (playbackEpochRef.current !== myEpoch) break
        if (pausedRef.current && voiceEnabledRef.current) {
          queueRef.current.unshift(next)
          syncQueueCount()
          currentItemRef.current = null
          break
        }
        currentItemRef.current = null
      }
    } finally {
      if (playbackEpochRef.current === myEpoch) {
        currentItemRef.current = null
        playingRef.current = false
        setIsAwaitingTts(false)
        if (!pausedRef.current) {
          setActiveTurnId(null)
          setActiveSpeakerId(null)
        }
      }
    }
  }, [playWavUrl, speakWebSpeech, syncQueueCount, waitForTtsIfPending])

  const enqueueTurn = useCallback(
    (turn: RoundtableTurnDTO, awaitingTts: boolean = false) => {
      if (!voiceEnabledRef.current) return
      if (turn.speaker_id === 'user' || mutedRef.current.has(turn.speaker_id)) return
      if (turn.audio_status === 'disabled' && !awaitingTts) return

      queueRef.current.push({
        turnId: turn.turn_id,
        speakerId: turn.speaker_id,
        text: turn.spoken_dialogue,
        audioCacheKey: turn.audio_cache_key ?? null,
        audioStatus: turn.audio_status,
        awaitingTts: awaitingTts && !turn.audio_cache_key,
      })
      syncQueueCount()
      void pumpQueue()
    },
    [pumpQueue, syncQueueCount],
  )

  const notifyAudioReady = useCallback(
    (ev: {
      turn_id: string
      audio_status: AudioStatusType
      audio_cache_key?: string | null
    }) => {
      if (currentItemRef.current && currentItemRef.current.turnId === ev.turn_id) {
        currentItemRef.current.audioStatus = ev.audio_status
        currentItemRef.current.audioCacheKey = ev.audio_cache_key ?? null
        currentItemRef.current.awaitingTts = false
      }
      const queued = queueRef.current.find((q) => q.turnId === ev.turn_id)
      if (queued) {
        queued.audioStatus = ev.audio_status
        queued.audioCacheKey = ev.audio_cache_key ?? null
        queued.awaitingTts = false
      }
      const waiter = ttsWaitersRef.current.get(ev.turn_id)
      waiter?.()
    },
    [],
  )

  const playTurnNow = useCallback(
    (turn: RoundtableTurnDTO) => {
      if (turn.speaker_id === 'user') return
      if (activeTurnId === turn.turn_id) {
        clearQueue()
        return
      }
      clearQueue()
      voiceEnabledRef.current = true
      setVoiceEnabledState(true)
      queueRef.current.push({
        turnId: turn.turn_id,
        speakerId: turn.speaker_id,
        text: turn.spoken_dialogue,
        audioCacheKey: turn.audio_cache_key ?? null,
        audioStatus: turn.audio_status === 'disabled' ? 'fallback_web_speech' : turn.audio_status,
        awaitingTts: false,
      })
      syncQueueCount()
      void pumpQueue()
    },
    [activeTurnId, clearQueue, pumpQueue, syncQueueCount],
  )

  const pauseOrResume = useCallback(() => {
    if (pausedRef.current) {
      pausedRef.current = false
      setIsPaused(false)
      if (audioElRef.current && audioElRef.current.paused) {
        void audioElRef.current.play().catch(() => { })
      } else {
        void pumpQueue()
      }
    } else {
      pausedRef.current = true
      setIsPaused(true)
      if (audioElRef.current) {
        audioElRef.current.pause()
      }
      if (typeof window !== 'undefined' && 'speechSynthesis' in window) {
        try {
          window.speechSynthesis.cancel()
        } catch {
          // ignore
        }
      }
    }
  }, [pumpQueue])

  const skipCurrent = useCallback(() => {
    pausedRef.current = false
    setIsPaused(false)
    stopActivePlaybackOnly()
  }, [stopActivePlaybackOnly])

  const previewVoice = useCallback(
    async (req: VoicePreviewRequestDTO): Promise<VoicePreviewResponseDTO> => {
      clearQueue()
      previewingRef.current = true
      setIsPreviewingVoice(true)
      try {
        const res = await api.previewRoundtableVoice(req)
        const previewItem: QueueItem = {
          turnId: `preview_${req.speaker_id}_${req.voice_name}`,
          speakerId: req.speaker_id,
          text: res.sample_text,
          audioCacheKey: res.audio_cache_key ?? null,
          audioStatus: res.audio_status,
          awaitingTts: false,
          pitchOverride: res.pitch,
          rateOverride: res.rate,
          isPreview: true,
        }
        setActiveSpeakerId(req.speaker_id)
        if (
          (res.audio_status === 'ready' || res.audio_status === 'cached') &&
          res.audio_cache_key
        ) {
          const ok = await playWavUrl(previewItem)
          if (!ok) {
            await speakWebSpeech(previewItem)
          }
        } else {
          await speakWebSpeech(previewItem)
        }
        return res
      } finally {
        previewingRef.current = false
        setActiveSpeakerId(null)
        setIsPreviewingVoice(false)
      }
    },
    [clearQueue, playWavUrl, speakWebSpeech],
  )

  return {
    voiceEnabled,
    setVoiceEnabled,
    playbackRate,
    setPlaybackRate,
    mutedSpeakers,
    toggleSpeakerMute,
    activeTurnId,
    activeSpeakerId,
    isPlaying: activeTurnId !== null && !isPaused,
    isPaused,
    isPreviewingVoice,
    isAwaitingTts,
    queueCount,
    enqueueTurn,
    notifyAudioReady,
    playTurnNow,
    pauseOrResume,
    skipCurrent,
    clearQueue,
    previewVoice,
    getVoiceEnergy,
  }
}
