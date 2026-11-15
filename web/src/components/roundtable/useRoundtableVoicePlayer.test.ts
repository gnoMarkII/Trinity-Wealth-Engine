import { act, renderHook } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import {
  TTS_PENDING_WAIT_TIMEOUT_MS,
  clampWebSpeechPitch,
  getMaxPlausibleWavDurationSec,
  getTtsPendingWaitMs,
  prepareTextForWebSpeech,
  splitTextIntoSpeechChunks,
  useRoundtableVoicePlayer,
} from './useRoundtableVoicePlayer'
import type { RoundtableTurnDTO } from '../../api/types'

function makeTurn(turnId = 'turn_1'): RoundtableTurnDTO {
  return {
    turn_id: turnId,
    session_id: 'rt_voice_test',
    seq: 1,
    speaker_id: 'value_guru',
    round_stage: 'opening',
    replying_to: null,
    spoken_dialogue: 'ราคาปัจจุบันสูงกว่าส่วนเผื่อความปลอดภัยครับ',
    evidence_bullets: ['[FACT] Ref price'],
    stance_tag: 'HOLD',
    sizing_posture: null,
    audio_cache_key: null,
    audio_status: 'fallback_web_speech',
    created_at: 1730000001,
  }
}

describe('useRoundtableVoicePlayer TTS timeout handling & Web Speech safeguards', () => {
  let audioPlayUrls: string[] = []
  let speechSpeakMock: ReturnType<typeof vi.fn>
  let speechCancelMock: ReturnType<typeof vi.fn>
  let spokenUtterances: Array<{
    text: string
    pitch: number
    rate: number
    lang: string
    onend: (() => void) | null
    onerror: (() => void) | null
  }> = []

  beforeEach(() => {
    vi.useFakeTimers()
    audioPlayUrls = []
    spokenUtterances = []
    speechCancelMock = vi.fn()
    speechSpeakMock = vi.fn((utter) => {
      spokenUtterances.push(utter)
    })

    class MockAudio {
      src: string
      playbackRate = 1
      preservesPitch = true
      paused = false
      onended: (() => void) | null = null
      onerror: (() => void) | null = null
      constructor(url: string) {
        this.src = url
        audioPlayUrls.push(url)
      }
      play() {
        return Promise.resolve().then(() => {
          this.onended?.()
        })
      }
      pause() {
        this.paused = true
      }
    }

    vi.stubGlobal('Audio', MockAudio)
    vi.stubGlobal(
      'SpeechSynthesisUtterance',
      class MockUtterance {
        lang = 'th-TH'
        pitch = 1
        rate = 1
        voice: SpeechSynthesisVoice | null = null
        onend: (() => void) | null = null
        onerror: (() => void) | null = null
        text: string
        constructor(text: string) {
          this.text = text
        }
      },
    )
    Object.defineProperty(window, 'speechSynthesis', {
      configurable: true,
      value: {
        speak: speechSpeakMock,
        cancel: speechCancelMock,
        getVoices: () => [],
      },
    })
  })

  afterEach(() => {
    vi.useRealTimers()
    vi.unstubAllGlobals()
  })

  it('waits long enough (15-28s) so slower Gemini TTS (e.g. 17.5s Turn 1) plays WAV instead of falling back', async () => {
    expect(TTS_PENDING_WAIT_TIMEOUT_MS).toBe(15000)
    const { result } = renderHook(() => useRoundtableVoicePlayer())

    act(() => {
      result.current.enqueueTurn(makeTurn('turn_wav_slow'), true)
    })

    // Advance 12000ms (past the old 10s ceiling, still under wait budget): must NOT fall back yet
    await act(async () => {
      vi.advanceTimersByTime(12000)
      await Promise.resolve()
    })
    expect(speechSpeakMock).not.toHaveBeenCalled()
    expect(result.current.isAwaitingTts).toBe(true)

    // Gemini TTS completes -> plays WAV cleanly with zero Web Speech calls
    await act(async () => {
      result.current.notifyAudioReady({
        turn_id: 'turn_wav_slow',
        audio_status: 'ready',
        audio_cache_key: 'wav_key_slow',
      })
      await Promise.resolve()
      await Promise.resolve()
    })

    expect(audioPlayUrls.some((u) => u.includes('wav_key_slow'))).toBe(true)
    expect(speechSpeakMock).not.toHaveBeenCalled()
    expect(result.current.isAwaitingTts).toBe(false)
  })

  it('falls back cleanly to WebSpeech only after full pending timeout expires', async () => {
    const { result } = renderHook(() => useRoundtableVoicePlayer())
    const turn = makeTurn('turn_tts_timeout')

    act(() => {
      result.current.enqueueTurn(turn, true)
    })

    await act(async () => {
      vi.advanceTimersByTime(getTtsPendingWaitMs(turn.spoken_dialogue.length) + 50)
      await Promise.resolve()
    })

    expect(audioPlayUrls).toHaveLength(0)
    expect(speechSpeakMock).toHaveBeenCalledTimes(1)
  })

  it('scales TTS wait up to 28000ms for longer deep_youtube turns and resolves immediately on failure notification', async () => {
    const longText = 'การวิเคราะห์เจาะลึกระยะยาวประกอบด้วยปัจจัยด้านมูลค่าและคูเมืองธุรกิจ '.repeat(12)
    expect(getTtsPendingWaitMs(longText.length)).toBeGreaterThan(20000)
    expect(getTtsPendingWaitMs(longText.length)).toBeLessThanOrEqual(28000)

    const { result } = renderHook(() => useRoundtableVoicePlayer())
    const longTurn = { ...makeTurn('turn_long_yt'), spoken_dialogue: longText }

    act(() => {
      result.current.enqueueTurn(longTurn, true)
    })

    // Immediately resolves without waiting 28s if backend reports fallback_web_speech (e.g. offline/429)
    await act(async () => {
      result.current.notifyAudioReady({
        turn_id: 'turn_long_yt',
        audio_status: 'fallback_web_speech',
        audio_cache_key: null,
      })
      await Promise.resolve()
      await Promise.resolve()
    })

    expect(speechSpeakMock).toHaveBeenCalled()
  })

  it('clamps Thai Web Speech pitch to prevent alien distortion, chunks long text, and cancels on chunk timeout', async () => {
    expect(clampWebSpeechPitch(0.86, true)).toBe(0.97)
    expect(clampWebSpeechPitch(1.08, true)).toBe(1.03)
    expect(prepareTextForWebSpeech('ซื้อแบบอ\u0433\u0440\u0435\u0441ซีฟใน $NVDA 67.1%')).toBe(
      'ซื้อแบบอเกรสซีฟใน NVDA 67.1 เปอร์เซ็นต์',
    )

    const longThai = Array.from({ length: 12 }, (_, idx) => `ประโยคที่ ${idx + 1} หุ้นเทคโนโลยีเติบโตสูงมากครับ`).join(' ')
    const chunks = splitTextIntoSpeechChunks(longThai, 180)
    expect(chunks.length).toBeGreaterThan(1)
    expect(chunks.every((c) => c.length <= 180)).toBe(true)

    const { result } = renderHook(() => useRoundtableVoicePlayer())
    const skepticTurn: RoundtableTurnDTO = {
      ...makeTurn('turn_skeptic_chunked'),
      speaker_id: 'skeptic_guru', // default pitch 0.86 must be clamped for Thai Web Speech
      spoken_dialogue: longThai,
    }

    act(() => {
      result.current.enqueueTurn(skepticTurn, false)
    })

    expect(spokenUtterances.length).toBe(1)
    expect(spokenUtterances[0]?.pitch).toBe(0.97)

    // When chunk 1 times out without firing onend, synthesis.cancel() must run before advancing to chunk 2
    await act(async () => {
      vi.advanceTimersByTime(46000)
      await Promise.resolve()
    })
    expect(speechCancelMock).toHaveBeenCalled()
    expect(spokenUtterances.length).toBe(2)
  })

  it('guards pumpQueue against overlapping loops when clearQueue + playTurnNow interrupts waitForTtsIfPending', async () => {
    const { result } = renderHook(() => useRoundtableVoicePlayer())
    const t1 = makeTurn('turn_waiting')
    const t2: RoundtableTurnDTO = {
      ...makeTurn('turn_manual_play'),
      seq: 2,
      audio_status: 'ready',
      audio_cache_key: 'wav_manual_2',
    }

    act(() => {
      result.current.enqueueTurn(t1, true)
    })
    expect(result.current.activeTurnId).toBe('turn_waiting')

    await act(async () => {
      result.current.playTurnNow(t2)
      await Promise.resolve()
      await Promise.resolve()
    })

    expect(speechSpeakMock).not.toHaveBeenCalled()
    expect(audioPlayUrls.filter((u) => u.includes('wav_manual_2'))).toHaveLength(1)
  })

  it('aborts corrupt oversized WAV playback (> max plausible duration) and falls back cleanly to Web Speech', async () => {
    const longDialogue = 'การวิเคราะห์ความเสี่ยงเชิงโครงสร้างและต้นทุนเงินทุนของบริษัท '.repeat(15).trim()
    expect(getMaxPlausibleWavDurationSec(longDialogue.length)).toBeGreaterThan(200)
    expect(getMaxPlausibleWavDurationSec(longDialogue.length)).toBeLessThan(300)

    class CorruptDurationAudio {
      src: string
      playbackRate = 1
      preservesPitch = true
      paused = false
      duration = 655.13
      currentTime = 0
      onloadedmetadata: (() => void) | null = null
      ontimeupdate: (() => void) | null = null
      onended: (() => void) | null = null
      onerror: (() => void) | null = null
      constructor(url: string) {
        this.src = url
        audioPlayUrls.push(url)
      }
      play() {
        return Promise.resolve().then(() => {
          this.onloadedmetadata?.()
        })
      }
      pause() {
        this.paused = true
      }
    }
    vi.stubGlobal('Audio', CorruptDurationAudio)

    const { result } = renderHook(() => useRoundtableVoicePlayer())
    const turn: RoundtableTurnDTO = {
      ...makeTurn('turn_corrupt_wav'),
      spoken_dialogue: longDialogue,
      audio_status: 'ready',
      audio_cache_key: 'wav_corrupt_655s',
    }

    await act(async () => {
      result.current.enqueueTurn(turn, true)
      await Promise.resolve()
      await Promise.resolve()
    })

    expect(audioPlayUrls.some((u) => u.includes('wav_corrupt_655s'))).toBe(true)
    expect(speechSpeakMock).toHaveBeenCalled()
  })

  it('exposes bounded [0, 1] getVoiceEnergy amplitude during speech and resets cleanly on clearQueue', async () => {
    const { result } = renderHook(() => useRoundtableVoicePlayer())
    expect(result.current.getVoiceEnergy()).toBe(0)

    await act(async () => {
      result.current.enqueueTurn(makeTurn('turn_energy'), false)
      await Promise.resolve()
    })

    const activeEnergy = result.current.getVoiceEnergy()
    expect(Number.isFinite(activeEnergy)).toBe(true)
    expect(activeEnergy).toBeGreaterThan(0)
    expect(activeEnergy).toBeLessThanOrEqual(1)

    act(() => {
      result.current.clearQueue()
    })
    expect(result.current.getVoiceEnergy()).toBe(0)
  })
})

