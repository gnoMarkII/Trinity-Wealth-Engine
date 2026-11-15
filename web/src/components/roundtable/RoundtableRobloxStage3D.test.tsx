import { render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import * as THREE from 'three'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { GuruAvatarStage } from './GuruAvatarStage'
import { SPEAKER_META_MAP, STAGE_SEAT_ORDER, type SpeakerVisualMeta } from './personas'
import {
  ALL_STAGE_CHARACTER_IDS,
  PANEL_GURU_IDS,
  ROOM_LAYOUT,
  WORKSTATION_SLOTS,
  computeObstacleAwareStep,
  computeRoundtableChoreography,
} from './roundtableChoreography'
import type { RoundtableTurnDTO } from '../../api/types'
import { buildCentralHologramProjector, buildProfileFromOhlcv } from './roundtableHologram'
import { createRobloxR6Avatar, disposeRobloxR6Avatar } from './roundtableRobloxAvatars'
import { buildRobloxTycoonRoom } from './roundtableRobloxRoom'

function buildMetas(disabledIds: string[] = []): Record<string, SpeakerVisualMeta> {
  const out: Record<string, SpeakerVisualMeta> = {}
  for (const sid of ALL_STAGE_CHARACTER_IDS) {
    const base = SPEAKER_META_MAP[sid]!
    out[sid] = {
      ...base,
      enabled: disabledIds.includes(sid) ? false : base.enabled,
    }
  }
  return out
}

function makeTurn(overrides: Partial<RoundtableTurnDTO> = {}): RoundtableTurnDTO {
  return {
    turn_id: 'turn_1',
    session_id: 'rt_1',
    seq: 1,
    speaker_id: 'value_guru',
    round_stage: 'opening',
    replying_to: null,
    spoken_dialogue: '[FACT] ROIC แข็งแกร่งและ Margin of Safety อยู่ที่ 18%',
    evidence_bullets: ['[FACT] ROIC 24%'],
    stance_tag: 'HOLD',
    audio_cache_key: null,
    audio_status: 'ready',
    created_at: 1730000001,
    ...overrides,
  }
}

describe('Roundtable Roblox 3D Choreography & View Switcher', () => {
  beforeEach(() => {
    window.localStorage.clear()
  })

  it('positions idle panel gurus at Zone C workstations and places disabled seats in Bloxy Lounge', () => {
    const metasById = buildMetas(['macro_tactical_guru'])
    const choreo = computeRoundtableChoreography({
      metasById,
      orderedPanelIds: [...PANEL_GURU_IDS].filter((id) => id !== 'macro_tactical_guru'),
      turns: [],
      activeSpeakerId: null,
      thinkingSpeakerId: null,
      roundStage: null,
      hasActiveSandboxMath: false,
    })

    expect(choreo.effectiveStage).toBe('idle')
    expect(choreo.targetsById.value_guru?.targetPos).toEqual(WORKSTATION_SLOTS.value_guru!.chairPos)
    expect(choreo.targetsById.value_guru?.pose).toBe('sit_desk')
    expect(choreo.targetsById.hermes_moderator?.targetPos).toEqual(ROOM_LAYOUT.hermesStandPos)
    expect(choreo.targetsById.user?.targetPos).toEqual(ROOM_LAYOUT.userCeoSeatPos)

    // Disabled guru rests on lounge seat with zero bubbles
    const disabledTarget = choreo.targetsById.macro_tactical_guru!
    expect(disabledTarget.isEnabled).toBe(false)
    expect(disabledTarget.pose).toBe('lounge')
    expect(disabledTarget.showBubble).toBe(false)
  })

  it('moves crossfire opponent pair to Duel Pad and enforces max 2 simultaneous speech bubbles', () => {
    const metasById = buildMetas()
    const turns = [
      makeTurn({ turn_id: 't1', speaker_id: 'growth_guru', round_stage: 'opening', stance_tag: 'BUY' }),
      makeTurn({
        turn_id: 't2',
        speaker_id: 'skeptic_guru',
        round_stage: 'crossfire',
        replying_to: 'growth_guru',
        spoken_dialogue: 'Terminal Value เกิน 75% เสี่ยงหดตัวหากมาร์จิ้นชะลอ',
        stance_tag: 'UNDERWEIGHT',
      }),
    ]

    const choreo = computeRoundtableChoreography({
      metasById,
      orderedPanelIds: [...PANEL_GURU_IDS],
      turns,
      activeSpeakerId: 'skeptic_guru',
      thinkingSpeakerId: null,
      roundStage: 'crossfire',
      hasActiveSandboxMath: false,
    })

    expect(choreo.effectiveStage).toBe('crossfire')
    expect(choreo.targetsById.skeptic_guru?.targetPos).toEqual(ROOM_LAYOUT.duelLeftPos)
    expect(choreo.targetsById.growth_guru?.targetPos).toEqual(ROOM_LAYOUT.duelRightPos)
    expect(choreo.targetsById.skeptic_guru?.pose).toBe('duel')

    const visibleBubbleCount = Object.values(choreo.targetsById).filter((t) => t.showBubble).length
    expect(visibleBubbleCount).toBeLessThanOrEqual(2)
  })

  it('brings User avatar out of CEO room and routes Sandbox Math responder to Quant Server', () => {
    const metasById = buildMetas()
    const choreo = computeRoundtableChoreography({
      metasById,
      orderedPanelIds: [...PANEL_GURU_IDS],
      turns: [makeTurn({ speaker_id: 'user', round_stage: 'user_interjection' })],
      activeSpeakerId: null,
      thinkingSpeakerId: 'value_guru',
      roundStage: 'user_interjection',
      hasActiveSandboxMath: true,
    })

    expect(choreo.effectiveStage).toBe('user_interjection')
    expect(choreo.targetsById.user?.targetPos).toEqual(ROOM_LAYOUT.userInterjectPos)
    expect(choreo.targetsById.user?.pose).toBe('jump_interject')
    expect(choreo.targetsById.value_guru?.targetPos).toEqual(ROOM_LAYOUT.quantStandPos)
    expect(choreo.targetsById.value_guru?.pose).toBe('calc')
  })

  it('provides seamless 3D Roblox <-> 2D Classic mode toggle with localStorage persistence', async () => {
    const user = userEvent.setup()
    const onToggleMute = vi.fn()
    const onMentionSpeaker = vi.fn()

    render(
      <GuruAvatarStage
        turns={[]}
        prioritizedSpeakers={[]}
        activeSpeakerId={null}
        thinkingSpeakerId={null}
        mutedSpeakers={new Set()}
        onToggleMute={onToggleMute}
        onMentionSpeaker={onMentionSpeaker}
      />,
    )

    // Starts in 3D Roblox view mode by default and displays top stepper + toggles
    const viewGroup = screen.getByRole('group', { name: /สลับมุมมองเวทีสภากูรู/i })
    const btn3d = within(viewGroup).getByRole('button', { name: /3D Roblox/i })
    const btn2d = within(viewGroup).getByRole('button', { name: /ปกติ \(2D\)/i })
    expect(btn3d).toHaveAttribute('aria-pressed', 'true')
    expect(btn2d).toHaveAttribute('aria-pressed', 'false')

    const stageEl = await screen.findByTestId('roundtable-roblox-stage-3d')
    expect(stageEl).toBeInTheDocument()
    expect(stageEl).toHaveClass('select-none')
    expect(screen.getByRole('button', { name: /โหมดสาธิต/i })).toBeInTheDocument()

    // Click Demo Walkthrough mode and verify step switches cleanly
    await user.click(screen.getByRole('button', { name: /โหมดสาธิต/i }))
    expect(screen.getByRole('button', { name: /กลับไปดูงานจริง/i })).toBeInTheDocument()

    // Click Step 4 (Crossfire) in HUD Stepper
    await user.click(screen.getByRole('button', { name: /4/i }))
    expect(screen.getByText(/4\. ปะทะเดือด/)).toBeInTheDocument()

    // Switch back to classic 2D mode
    await user.click(btn2d)
    await waitFor(() => {
      expect(btn2d).toHaveAttribute('aria-pressed', 'true')
      expect(btn3d).toHaveAttribute('aria-pressed', 'false')
      expect(screen.queryByTestId('roundtable-roblox-stage-3d')).not.toBeInTheDocument()
    })
    expect(window.localStorage.getItem('trinity_rt_stage_view_mode')).toBe('2d')

    // Classic 2D mode preserves all seat controls and mention clicks
    expect(STAGE_SEAT_ORDER.length).toBe(6)
    await user.click(screen.getByRole('button', { name: /^Value\b/i }))
    expect(onMentionSpeaker).toHaveBeenCalledWith('value_guru')

    // Switch back to 3D Roblox mode cleanly
    await user.click(btn3d)
    expect(await screen.findByTestId('roundtable-roblox-stage-3d')).toBeInTheDocument()
    expect(window.localStorage.getItem('trinity_rt_stage_view_mode')).toBe('3d')

    // Defaults to Metropolitan Morning lighting ('ยามสาย') and toggles to Night ('กลางคืน')
    const lightBtn = screen.getByRole('button', { name: /ยามสาย/i })
    await user.click(lightBtn)
    expect(screen.getByRole('button', { name: /กลางคืน/i })).toBeInTheDocument()
  })

  it('assigns holo_point / tap_terminal poses and subtle head-tracking around Circular Holo-Console', () => {
    const metasById = buildMetas()
    const choreo = computeRoundtableChoreography({
      metasById,
      orderedPanelIds: [...PANEL_GURU_IDS],
      turns: [
        makeTurn({ speaker_id: 'growth_guru', round_stage: 'opening', stance_tag: 'BUY' }),
      ],
      activeSpeakerId: 'growth_guru',
      thinkingSpeakerId: 'quality_guru',
      roundStage: 'opening',
      hasActiveSandboxMath: false,
    })

    expect(choreo.targetsById.growth_guru?.pose).toBe('holo_point')
    expect(choreo.targetsById.quality_guru?.pose).toBe('tap_terminal')
    expect(choreo.targetsById.value_guru?.pose).toBe('sit_table')
    const headLook = choreo.targetsById.value_guru?.headLookYaw ?? 0
    expect(Math.abs(headLook)).toBeLessThanOrEqual(0.5)
  })

  it('builds and animates 3D candlestick hologram projector cleanly without WebGL errors', () => {
    const holo = buildCentralHologramProjector()
    expect(holo.group.name).toBe('central_holo_projector')
    expect(holo.group.userData).toMatchObject({ holoFocus: true })
    expect(holo.group.children.length).toBeGreaterThanOrEqual(7)

    const instancedParticles = holo.group.children.find(
      (child): child is THREE.InstancedMesh => child instanceof THREE.InstancedMesh,
    )
    expect(instancedParticles).toBeDefined()
    expect(instancedParticles?.count).toBe(36)

    holo.updateState({
      primaryTicker: 'NVDA',
      finalRating: 'OVERWEIGHT',
      stancesBySpeaker: {
        growth_guru: 'BUY',
        skeptic_guru: 'UNDERWEIGHT',
      },
      candles: [
        { timestamp: 1, open: 120, high: 125, low: 118, close: 124, volume: 1000 },
        { timestamp: 2, open: 124, high: 129, low: 122, close: 128, volume: 1200 },
      ],
    })
    holo.animate(1.5, true, false, false, 0.75)
    holo.animate(3.0, false, true, true, 0)
    holo.dispose()
  })

  it('builds normalized candlestick profile from live OHLCV bars and classifies bullish vs bearish delta', () => {
    const fallback = buildProfileFromOhlcv(null)
    expect(fallback.hasLiveData).toBe(false)
    expect(fallback.pctChange).toBeNull()
    expect(fallback.points.length).toBeGreaterThan(5)

    const sampleBars = [
      { timestamp: 1, open: 100, high: 104, low: 98, close: 103, volume: 1000 },
      { timestamp: 2, open: 103, high: 106, low: 101, close: 102, volume: 1200 },
      { timestamp: 3, open: 102, high: 112, low: 101, close: 110, volume: 1500 },
    ]
    const live = buildProfileFromOhlcv(sampleBars)
    expect(live.hasLiveData).toBe(true)
    expect(live.pctChange).toBeCloseTo(10, 4)
    expect(live.points).toHaveLength(13)
    expect(live.points[0]?.bull).toBe(true)
    expect(live.points[live.points.length - 1]?.bull).toBe(true)
    expect(live.points.some((p) => !p.bull)).toBe(true)
  })

  it('routes walk steps smoothly around the central holotable and through CEO glass doorway', () => {
    // 1. Walking directly across Circular Holotable should curve radially/tangentially outside HOLO_OBSTACLE_RADIUS
    let pos = { x: 0, y: 0, z: 1.05 }
    const destAcrossTable = { x: 0, y: 0, z: -4.2 }
    for (let i = 0; i < 6; i += 1) {
      pos = computeObstacleAwareStep(pos, destAcrossTable, 0.25)
    }
    const radialFromHoloCenter = Math.hypot(
      pos.x - ROOM_LAYOUT.councilCenter.x,
      pos.z - ROOM_LAYOUT.councilCenter.z,
    )
    expect(Math.abs(pos.x)).toBeGreaterThan(0.05)
    expect(radialFromHoloCenter).toBeGreaterThan(1.75)

    // 2. Full walk simulation for YOU exiting VIP CEO office and returning back inside without doorway oscillation
    let youPos = { ...ROOM_LAYOUT.userCeoSeatPos }
    const outSteps: typeof youPos[] = []
    for (let f = 0; f < 220; f += 1) {
      youPos = computeObstacleAwareStep(youPos, ROOM_LAYOUT.userInterjectPos, 0.075)
      outSteps.push(youPos)
      if (
        Math.hypot(
          youPos.x - ROOM_LAYOUT.userInterjectPos.x,
          youPos.z - ROOM_LAYOUT.userInterjectPos.z,
        ) <= 0.08
      ) {
        break
      }
    }
    expect(
      Math.hypot(
        youPos.x - ROOM_LAYOUT.userInterjectPos.x,
        youPos.z - ROOM_LAYOUT.userInterjectPos.z,
      ),
    ).toBeLessThanOrEqual(0.08)
    // Verify exit never clipped ceoDesk bounds (x in [-10.55, -8.65], z in [-6.35, -5.55])
    expect(
      outSteps.some((p) => p.x > -10.55 && p.x < -8.65 && p.z > -6.35 && p.z < -5.55),
    ).toBe(false)

    // Return back from podium interjection into CEO seat
    const backSteps: typeof youPos[] = []
    for (let f = 0; f < 220; f += 1) {
      youPos = computeObstacleAwareStep(youPos, ROOM_LAYOUT.userCeoSeatPos, 0.075)
      backSteps.push(youPos)
      if (
        Math.hypot(
          youPos.x - ROOM_LAYOUT.userCeoSeatPos.x,
          youPos.z - ROOM_LAYOUT.userCeoSeatPos.z,
        ) <= 0.08
      ) {
        break
      }
    }
    expect(
      Math.hypot(
        youPos.x - ROOM_LAYOUT.userCeoSeatPos.x,
        youPos.z - ROOM_LAYOUT.userCeoSeatPos.z,
      ),
    ).toBeLessThanOrEqual(0.08)
    expect(
      backSteps.some((p) => p.x > -10.55 && p.x < -8.65 && p.z > -6.35 && p.z < -5.55),
    ).toBe(false)
  })

  it('builds Architectural Glass-Tile PBR multi-map floor and switches day/twilight sunlight & clouds cleanly', () => {
    const room = buildRobloxTycoonRoom(buildMetas(), 8)
    const floorMat = room.floorMesh.material as THREE.MeshStandardMaterial
    expect(floorMat.map).toBeInstanceOf(THREE.CanvasTexture)
    expect(floorMat.roughnessMap).toBeInstanceOf(THREE.CanvasTexture)
    expect(floorMat.bumpMap).toBeInstanceOf(THREE.CanvasTexture)
    expect(floorMat.emissiveMap).toBeInstanceOf(THREE.CanvasTexture)
    expect(floorMat.emissiveIntensity).toBeCloseTo(0.42, 2)

    const sunFx = room.group.getObjectByName('morning_sunlight_fx')
    const clouds = room.group.getObjectByName('daytime_puffy_clouds')
    expect(sunFx?.visible).toBe(true)
    expect(clouds?.visible).toBe(true)
    expect(clouds?.children.length).toBeGreaterThanOrEqual(8)

    room.applyLightingTheme('twilight')
    expect(floorMat.emissiveIntensity).toBeCloseTo(0.74, 2)
    expect(sunFx?.visible).toBe(false)
    expect(clouds?.visible).toBe(false)

    room.applyLightingTheme('day')
    expect(floorMat.emissiveIntensity).toBeCloseTo(0.42, 2)
    expect(sunFx?.visible).toBe(true)
    expect(clouds?.visible).toBe(true)

    room.animateRoomEffects(1.25, true, true, false, 0.6)
    room.animateRoomEffects(2.5, false, false, true, 0)
    room.dispose()
  })

  it('disposes all four cached R6 avatar face mood textures on unmount cleanup', () => {
    const avatar = createRobloxR6Avatar(SPEAKER_META_MAP.value_guru!)
    const spies = Object.values(avatar.faceTextures).map((tex) => vi.spyOn(tex, 'dispose'))
    disposeRobloxR6Avatar(avatar)
    expect(avatar.faceMaterial.map).toBeNull()
    for (const spy of spies) {
      expect(spy).toHaveBeenCalledTimes(1)
    }
  })
})
