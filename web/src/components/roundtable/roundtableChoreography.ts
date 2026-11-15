import type { RoundtableStanceType, RoundtableTurnDTO } from '../../api/types'
import type { SpeakerVisualMeta } from './personas'
import { STAGE_SEAT_ORDER } from './personas'

export interface Vec3 {
  x: number
  y: number
  z: number
}

export type RobloxAvatarPose =
  | 'sit_desk'
  | 'sit_table'
  | 'walk'
  | 'stand_talk'
  | 'holo_point'
  | 'tap_terminal'
  | 'typing'
  | 'duel'
  | 'calc'
  | 'jump_interject'
  | 'lounge'
  | 'celebrate'

export type RoundtableStepperStepId =
  | 'stage_intro'
  | 'opening'
  | 'rebuttal'
  | 'crossfire'
  | 'user_interjection'
  | 'synthesis'

export interface StepperStepSpec {
  id: RoundtableStepperStepId
  stepNumber: number
  shortLabel: string
  fullLabel: string
  description: string
}

export const ROUNDTABLE_STEPPER_STEPS: readonly StepperStepSpec[] = [
  {
    id: 'stage_intro',
    stepNumber: 1,
    shortLabel: '1. เปิดประเด็น',
    fullLabel: '1. เปิดประเด็น & รวมตัว',
    description: 'Hermes เปิดวาระจากโพเดียม กูรูทุกสายเดินเข้าประจำโต๊ะสภา',
  },
  {
    id: 'opening',
    stepNumber: 2,
    shortLabel: '2. มุมมองหลัก',
    fullLabel: '2. รอบที่ 1 · เปิดมุมมองหลัก',
    description: 'กูรูแต่ละสายผลัดกันลุกยืนนำเสนอหลักฐานตามกรอบคิดของตน',
  },
  {
    id: 'rebuttal',
    stepNumber: 3,
    shortLabel: '3. ซักค้าน',
    fullLabel: '3. รอบที่ 2 · ซักค้านสัดส่วน',
    description: 'ตั้งคำถามทวนสมมติฐาน ชั่งน้ำหนัก ROIC vs ราคาและความเสี่ยง',
  },
  {
    id: 'crossfire',
    stepNumber: 4,
    shortLabel: '4. ปะทะเดือด',
    fullLabel: '4. Crossfire · ปะทะตัวต่อตัว',
    description: 'คู่กูรูเห็นต่างเดินออกมายืนประจันหน้ากลางแท่น Crossfire Arena',
  },
  {
    id: 'user_interjection',
    stepNumber: 5,
    shortLabel: '5. แทรกถาม',
    fullLabel: '5. เจ้าของพอร์ตแทรกถาม & Quant Math',
    description: 'อวาตาร์ของคุณเดินจากห้อง CEO มาหน้าวง พร้อมส่งคำนวณที่ตู้ Sandbox',
  },
  {
    id: 'synthesis',
    stepNumber: 6,
    shortLabel: '6. สรุปมติ CIO',
    fullLabel: '6. รอบที่ 3 · Hermes สรุปมติ',
    description: 'Hermes ตกผลึกมติพอร์ตขึ้นจอ LED ใหญ่พร้อม Kill-Switch',
  },
] as const

export const PANEL_GURU_IDS = [
  'value_guru',
  'quality_guru',
  'growth_guru',
  'macro_tactical_guru',
  'skeptic_guru',
] as const

export const ALL_STAGE_CHARACTER_IDS = [...STAGE_SEAT_ORDER, 'user'] as const

/** Single source of truth for spatial zones shared between Three.js room geometry & choreography */
export const ROOM_LAYOUT = {
  councilCenter: { x: 0, y: 0, z: -1.25 } as Vec3,
  councilSeatRadius: 2.58,
  councilTableRadius: 1.56,
  // Distributed around circular Holo-Console leaving front-right hero camera corridor open
  councilAngles: [-2.82, -1.98, -1.12, -0.22, 2.28] as readonly number[],
  duelPadCenter: { x: 0, y: 0, z: 2.45 } as Vec3,
  duelLeftPos: { x: -1.02, y: 0, z: 2.45 } as Vec3,
  duelRightPos: { x: 1.02, y: 0, z: 2.45 } as Vec3,
  hermesPodiumPos: { x: 0, y: 0, z: -6.05 } as Vec3,
  hermesStandPos: { x: 0, y: 0.14, z: -6.65 } as Vec3,
  mainWallScreenPos: { x: 0.6, y: 3.35, z: -9.05 } as Vec3,
  boardWallScreenPos: { x: 7.45, y: 2.9, z: -7.85 } as Vec3,
  ceoRoomCenter: { x: -9.6, y: 0, z: -6.3 } as Vec3,
  userCeoSeatPos: { x: -9.6, y: 0, z: -7.1 } as Vec3,
  userInterjectPos: { x: -2.65, y: 0, z: 1.35 } as Vec3,
  quantRackPos: { x: 10.2, y: 0, z: 0.1 } as Vec3,
  quantStandPos: { x: 8.85, y: 0, z: 0.1 } as Vec3,
  loungeCenter: { x: 9.8, y: 0, z: -4.8 } as Vec3,
  loungeSeatSlots: [
    { x: 9.15, y: 0, z: -4.8, rotY: -Math.PI / 2 },
    { x: 10.1, y: 0, z: -5.6, rotY: 0 },
    { x: 10.1, y: 0, z: -3.9, rotY: Math.PI },
    { x: 8.5, y: 0, z: -5.9, rotY: 0.35 },
    { x: 8.5, y: 0, z: -3.7, rotY: Math.PI - 0.35 },
  ] as ReadonlyArray<{ x: number; y: number; z: number; rotY: number }>,
} as const

export interface WorkstationSlotSpec {
  speakerId: string
  deskPos: Vec3
  chairPos: Vec3
  rotY: number
  propType: 'safe' | 'diamond' | 'rocket' | 'globe' | 'radar'
}

function buildWorkstationSlots(): Record<string, WorkstationSlotSpec> {
  const configs: ReadonlyArray<{
    speakerId: string
    x: number
    z: number
    rotY: number
    propType: WorkstationSlotSpec['propType']
  }> = [
    { speakerId: 'value_guru', x: -9.5, z: 5.0, rotY: -0.34, propType: 'safe' },
    { speakerId: 'quality_guru', x: -5.0, z: 6.2, rotY: -0.22, propType: 'diamond' },
    { speakerId: 'growth_guru', x: 0.0, z: 6.5, rotY: 0.0, propType: 'rocket' },
    { speakerId: 'macro_tactical_guru', x: 5.0, z: 6.2, rotY: 0.22, propType: 'globe' },
    { speakerId: 'skeptic_guru', x: 9.5, z: 5.0, rotY: 0.34, propType: 'radar' },
  ]

  const result: Record<string, WorkstationSlotSpec> = {}
  for (const cfg of configs) {
    // Chair sits 0.92 studs behind desk center along local -Z facing desk (+Z toward front-camera)
    const backX = -Math.sin(cfg.rotY) * 0.92
    const backZ = -Math.cos(cfg.rotY) * 0.92
    result[cfg.speakerId] = {
      speakerId: cfg.speakerId,
      deskPos: { x: cfg.x, y: 0, z: cfg.z },
      chairPos: { x: cfg.x + backX, y: 0, z: cfg.z + backZ },
      rotY: cfg.rotY,
      propType: cfg.propType,
    }
  }
  return result
}

export const WORKSTATION_SLOTS: Record<string, WorkstationSlotSpec> = buildWorkstationSlots()

export function getCouncilSeatTransform(seatIndex: number): { pos: Vec3; rotY: number } {
  const clampedIdx = Math.max(0, Math.min(ROOM_LAYOUT.councilAngles.length - 1, seatIndex))
  const angle = ROOM_LAYOUT.councilAngles[clampedIdx]!
  const x = ROOM_LAYOUT.councilCenter.x + Math.cos(angle) * ROOM_LAYOUT.councilSeatRadius
  const z = ROOM_LAYOUT.councilCenter.z + Math.sin(angle) * ROOM_LAYOUT.councilSeatRadius
  // Face inward toward the center of the horseshoe roundtable
  const rotY = yawToward({ x, y: 0, z }, ROOM_LAYOUT.councilCenter)
  return { pos: { x, y: 0, z }, rotY }
}

export function yawToward(from: Vec3, to: Vec3): number {
  return Math.atan2(to.x - from.x, to.z - from.z)
}

const HOLO_OBSTACLE_RADIUS = 2.24
// CEO glass box bounds in world XZ: right glass x = -6.8, front-left glass z = -4.2 (x <= -9.2).
// Doorway opening spans x in [-9.1, -6.9] at z = -4.2; ceoDesk sits at x in [-10.65, -8.55], z in [-6.43, -5.47].
const CEO_OUTER_APPROACH: Vec3 = { x: -8.0, y: 0, z: -3.5 }
const CEO_INNER_AISLE: Vec3 = { x: -7.85, y: 0, z: -5.15 }
const CEO_BEHIND_DESK_AISLE: Vec3 = { x: -7.85, y: 0, z: -7.05 }

function isInsideCeoGlassBox(p: Vec3): boolean {
  return p.x < -6.85 && p.z < -4.2
}

/**
 * Computes an obstacle-aware XZ walk step so characters detour smoothly around the central
 * Circular Holotable and navigate cleanly through the CEO glass room doorway & around ceoDesk
 * without doorway oscillation.
 */
export function computeObstacleAwareStep(
  current: Vec3,
  target: Vec3,
  maxStepDist: number,
): Vec3 {
  let effectiveTarget = target

  const startInCeo = isInsideCeoGlassBox(current)
  const endInCeo = isInsideCeoGlassBox(target)

  if (startInCeo && !endInCeo) {
    // Exiting CEO glass office: first step out to right aisle if behind ceoDesk, then exit through doorway
    if (current.z < -6.2 && current.x < -8.15) {
      effectiveTarget = CEO_BEHIND_DESK_AISLE
    } else {
      effectiveTarget = CEO_OUTER_APPROACH
    }
  } else if (!startInCeo && endInCeo) {
    // Entering CEO glass office: line up outside doorway first, then commit monotonically inward past z = -4.2
    if (current.x > -9.05 && current.x < -6.95 && current.z <= -3.35) {
      effectiveTarget = CEO_INNER_AISLE
    } else {
      effectiveTarget = CEO_OUTER_APPROACH
    }
  } else if (startInCeo && endInCeo) {
    // Navigating inside CEO glass box toward seat behind ceoDesk: walk along right aisle before turning left
    if (target.z < -6.4 && current.z > -6.75 && current.x > -8.35) {
      effectiveTarget = CEO_BEHIND_DESK_AISLE
    }
  }

  const dx = effectiveTarget.x - current.x
  const dz = effectiveTarget.z - current.z
  const dist = Math.hypot(dx, dz)
  if (dist <= 1e-4) return { x: target.x, y: target.y, z: target.z }

  const step = Math.min(dist, maxStepDist)
  let dirX = dx / dist
  let dirZ = dz / dist

  // Check if next straight step comes too close to the Circular Holotable core
  const ox = ROOM_LAYOUT.councilCenter.x
  const oz = ROOM_LAYOUT.councilCenter.z
  const nextX = current.x + dirX * step
  const nextZ = current.z + dirZ * step
  const toObsX = nextX - ox
  const toObsZ = nextZ - oz
  const distToCenter = Math.hypot(toObsX, toObsZ)

  if (distToCenter < HOLO_OBSTACLE_RADIUS) {
    // Choose tangent direction around the holotable closest to intended travel direction
    const radLen = Math.max(1e-4, Math.hypot(current.x - ox, current.z - oz))
    const rx = (current.x - ox) / radLen
    const rz = (current.z - oz) / radLen
    // Two unit tangents (-rz, rx) and (rz, -rx)
    const dotPos = -rz * dirX + rx * dirZ
    const sign = dotPos >= 0 ? 1 : -1
    const tanX = -rz * sign
    const tanZ = rx * sign
    // Blend outward push + tangent so character curves cleanly outside the rim
    dirX = tanX * 0.85 + rx * 0.35
    dirZ = tanZ * 0.85 + rz * 0.35
    const norm = Math.max(1e-4, Math.hypot(dirX, dirZ))
    dirX /= norm
    dirZ /= norm
  }

  return {
    x: current.x + dirX * step,
    y: target.y,
    z: current.z + dirZ * step,
  }
}

export function truncateBubbleText(raw: string, maxLength = 74): string {
  const cleaned = raw
    .replace(/```[\s\S]*?```/g, ' ')
    .replace(/\[(?:FACT|MATH|INFERRED)\]\s*/gi, '')
    .replace(/[*_`#>]/g, '')
    .replace(/\s+/g, ' ')
    .trim()
  if (!cleaned) return ''
  if (cleaned.length <= maxLength) return cleaned
  return `${cleaned.slice(0, maxLength - 1).trimEnd()}…`
}

export interface CharacterChoreographyTarget {
  speakerId: string
  targetPos: Vec3
  targetRotY: number
  headLookYaw?: number
  pose: RobloxAvatarPose
  isSpeaking: boolean
  isThinking: boolean
  isEnabled: boolean
  showBubble: boolean
  bubbleText: string
  stanceTag?: RoundtableStanceType
}

export interface ChoreographyInput {
  metasById: Record<string, SpeakerVisualMeta>
  orderedPanelIds: string[]
  turns: RoundtableTurnDTO[]
  activeSpeakerId: string | null
  thinkingSpeakerId: string | null
  roundStage: string | null
  hasActiveSandboxMath: boolean
  finalRating?: RoundtableStanceType | string | null
  demoStepIndex?: number | null
}

export interface ChoreographyOutput {
  effectiveStage: RoundtableStepperStepId | 'idle'
  activeStepIndex: number
  isDemoActive: boolean
  spotlightSpeakerId: string | null
  targetsById: Record<string, CharacterChoreographyTarget>
}

interface DemoStepPreset {
  stage: RoundtableStepperStepId
  activeSpeakerId: string
  thinkingSpeakerId?: string | null
  duelOpponentId?: string | null
  sandboxCalcSpeakerId?: string | null
  bubbles: Record<string, { text: string; stance?: RoundtableStanceType }>
}

const DEMO_STEP_PRESETS: readonly DemoStepPreset[] = [
  {
    stage: 'stage_intro',
    activeSpeakerId: 'hermes_moderator',
    bubbles: {
      hermes_moderator: {
        text: 'เปิดประชุมสภากูรู — วันนี้ชั่งน้ำหนักมูลค่า ความเสี่ยง และจังหวะเข้าพอร์ตกันครบทั้ง 5 มุมมอง',
        stance: 'REVIEW',
      },
    },
  },
  {
    stage: 'opening',
    activeSpeakerId: 'value_guru',
    thinkingSpeakerId: 'quality_guru',
    bubbles: {
      value_guru: {
        text: 'FCF Yield เทียบ WACC ยังต้องต่อรอง Margin of Safety 15–20% ก่อนเคาะไม้แรก',
        stance: 'HOLD',
      },
    },
  },
  {
    stage: 'rebuttal',
    activeSpeakerId: 'growth_guru',
    thinkingSpeakerId: 'macro_tactical_guru',
    bubbles: {
      growth_guru: {
        text: 'ฐานกำไรใหม่มี Operating Leverage หนุน EPS Revision 30 วันเด่นกว่าที่ตลาด Price-in',
        stance: 'BUY',
      },
    },
  },
  {
    stage: 'crossfire',
    activeSpeakerId: 'skeptic_guru',
    duelOpponentId: 'growth_guru',
    bubbles: {
      skeptic_guru: {
        text: 'ระวังสัดส่วน Terminal Value สูงเกิน 75% — ต้องล็อก Kill-Switch ทันทีที่ Gross Margin หลุดเกณฑ์',
        stance: 'UNDERWEIGHT',
      },
      growth_guru: {
        text: 'รับทราบเงื่อนไข Kill-Switch แต่ Backlog คำสั่งซื้อยังครอบคลุม 4 ไตรมาสถัดไป',
        stance: 'BUY',
      },
    },
  },
  {
    stage: 'user_interjection',
    activeSpeakerId: 'value_guru',
    sandboxCalcSpeakerId: 'value_guru',
    bubbles: {
      user: {
        text: '✋ ถ้าซื้อถัวเพิ่มอีก 1 ไม้ในโซนแนวรับ ต้นทุนเฉลี่ยพอร์ตจะเปลี่ยนเป็นเท่าไร?',
        stance: 'REVIEW',
      },
      value_guru: {
        text: '🧮 คำนวณ Sandbox Decimal แล้ว — แบ่งเข้า 2 ไม้คุมสัดส่วนรวมไม่เกินเพดานพอร์ต',
        stance: 'HOLD',
      },
    },
  },
  {
    stage: 'synthesis',
    activeSpeakerId: 'hermes_moderator',
    bubbles: {
      hermes_moderator: {
        text: 'สรุปมติ CIO: ทยอยสะสมตามแนวรับอย่างมีวินัย พร้อมบังคับใช้ Kill-Switch ตามมติสภา',
        stance: 'OVERWEIGHT',
      },
    },
  },
] as const

export function mapRawStageToStepperId(
  rawStage: string | null | undefined,
  hasTurns: boolean,
  hasFinalVerdict: boolean,
): RoundtableStepperStepId | 'idle' {
  switch (rawStage) {
    case 'stage_intro':
      return 'stage_intro'
    case 'opening':
      return 'opening'
    case 'rebuttal':
      return 'rebuttal'
    case 'crossfire':
      return 'crossfire'
    case 'user_interjection':
    case 'interjection_reply':
      return 'user_interjection'
    case 'synthesis':
      return 'synthesis'
    default:
      if (hasFinalVerdict && hasTurns) return 'synthesis'
      if (hasTurns) return 'opening'
      return 'idle'
  }
}

export function computeRoundtableChoreography(input: ChoreographyInput): ChoreographyOutput {
  const isDemoActive =
    typeof input.demoStepIndex === 'number' &&
    input.demoStepIndex >= 0 &&
    input.demoStepIndex < DEMO_STEP_PRESETS.length

  const demoPreset = isDemoActive ? DEMO_STEP_PRESETS[input.demoStepIndex!]! : null

  const latestTurn = input.turns.length > 0 ? input.turns[input.turns.length - 1]! : null
  const latestStanceBySpeaker: Record<string, RoundtableStanceType> = {}
  const latestTurnBySpeaker: Record<string, RoundtableTurnDTO> = {}
  for (const t of input.turns) {
    if (t.stance_tag) latestStanceBySpeaker[t.speaker_id] = t.stance_tag
    latestTurnBySpeaker[t.speaker_id] = t
  }

  const effectiveStage = demoPreset
    ? demoPreset.stage
    : mapRawStageToStepperId(
        input.roundStage ?? latestTurn?.round_stage,
        input.turns.length > 0,
        Boolean(input.finalRating && input.finalRating !== 'REVIEW'),
      )

  const activeStepIndex =
    effectiveStage === 'idle'
      ? -1
      : ROUNDTABLE_STEPPER_STEPS.findIndex((s) => s.id === effectiveStage)

  const effectiveActiveSpeaker = demoPreset
    ? demoPreset.activeSpeakerId
    : (input.activeSpeakerId ?? (effectiveStage !== 'idle' ? latestTurn?.speaker_id ?? null : null))

  const effectiveThinkingSpeaker = demoPreset
    ? (demoPreset.thinkingSpeakerId ?? null)
    : input.thinkingSpeakerId

  // Identify Crossfire duel pair when stage is crossfire
  let duelPair: [string, string] | null = null
  if (demoPreset?.stage === 'crossfire' && demoPreset.duelOpponentId) {
    duelPair = [demoPreset.activeSpeakerId, demoPreset.duelOpponentId]
  } else if (effectiveStage === 'crossfire') {
    const cfSpeaker = effectiveActiveSpeaker || latestTurn?.speaker_id || 'skeptic_guru'
    const opponent =
      (latestTurn?.speaker_id === cfSpeaker ? latestTurn.replying_to : null) ||
      (cfSpeaker === 'growth_guru' ? 'skeptic_guru' : 'growth_guru')
    if (
      PANEL_GURU_IDS.includes(cfSpeaker as (typeof PANEL_GURU_IDS)[number]) &&
      PANEL_GURU_IDS.includes(opponent as (typeof PANEL_GURU_IDS)[number]) &&
      cfSpeaker !== opponent
    ) {
      duelPair = [cfSpeaker, opponent]
    }
  }

  // Sandbox math speaker moves to Quant Rack when Decimal calculation is active
  const sandboxCalcSpeakerId = demoPreset?.sandboxCalcSpeakerId
    ? demoPreset.sandboxCalcSpeakerId
    : input.hasActiveSandboxMath &&
        (effectiveStage === 'user_interjection' || Boolean(effectiveThinkingSpeaker))
      ? effectiveThinkingSpeaker || effectiveActiveSpeaker
      : null

  // Enforce /ui-ux-pro-max rule: max 1-2 speech bubbles visible simultaneously
  const allowedBubbleSpeakers = new Set<string>()
  if (demoPreset) {
    for (const sid of Object.keys(demoPreset.bubbles).slice(0, 2)) {
      allowedBubbleSpeakers.add(sid)
    }
  } else if (effectiveStage === 'crossfire' && duelPair) {
    allowedBubbleSpeakers.add(duelPair[0])
    if (latestTurnBySpeaker[duelPair[1]]) allowedBubbleSpeakers.add(duelPair[1])
  } else if (effectiveStage === 'user_interjection') {
    if (latestTurnBySpeaker.user) allowedBubbleSpeakers.add('user')
    if (effectiveActiveSpeaker && effectiveActiveSpeaker !== 'user') {
      allowedBubbleSpeakers.add(effectiveActiveSpeaker)
    }
  } else if (effectiveActiveSpeaker) {
    allowedBubbleSpeakers.add(effectiveActiveSpeaker)
  }

  const orderedEnabledPanelIds = input.orderedPanelIds.filter(
    (sid) =>
      sid !== 'hermes_moderator' &&
      sid !== 'user' &&
      (input.metasById[sid]?.enabled ?? true),
  )

  // Track lounge slots for disabled gurus so each sits in its own seat
  let loungeCursor = 0
  const targetsById: Record<string, CharacterChoreographyTarget> = {}

  const resolveBubbleInfo = (sid: string): { text: string; stance?: RoundtableStanceType } => {
    if (demoPreset?.bubbles[sid]) {
      return {
        text: demoPreset.bubbles[sid]!.text,
        stance: demoPreset.bubbles[sid]!.stance,
      }
    }
    const speakerTurn = latestTurnBySpeaker[sid]
    if (!speakerTurn) {
      return { text: '', stance: latestStanceBySpeaker[sid] }
    }
    return {
      text: truncateBubbleText(speakerTurn.spoken_dialogue, 74),
      stance: speakerTurn.stance_tag,
    }
  }

  // First determine focal position so idle seated council members can turn heads toward active speaker
  const preliminarySpeakerPos = (sid: string | null): Vec3 => {
    if (!sid || sid === 'hermes_moderator') return ROOM_LAYOUT.hermesStandPos
    if (sid === 'user') return ROOM_LAYOUT.userInterjectPos
    if (duelPair && sid === duelPair[0]) return ROOM_LAYOUT.duelLeftPos
    if (duelPair && sid === duelPair[1]) return ROOM_LAYOUT.duelRightPos
    if (sandboxCalcSpeakerId === sid) return ROOM_LAYOUT.quantStandPos
    const panelIdx = orderedEnabledPanelIds.indexOf(sid)
    if (panelIdx >= 0) return getCouncilSeatTransform(panelIdx).pos
    return ROOM_LAYOUT.councilCenter
  }

  const focalPoint = effectiveActiveSpeaker
    ? preliminarySpeakerPos(effectiveActiveSpeaker)
    : effectiveThinkingSpeaker
      ? preliminarySpeakerPos(effectiveThinkingSpeaker)
      : ROOM_LAYOUT.hermesStandPos

  for (const sid of ALL_STAGE_CHARACTER_IDS) {
    const meta = input.metasById[sid]
    const isEnabled = sid === 'hermes_moderator' || sid === 'user' ? true : (meta?.enabled ?? true)
    const isSpeaking = effectiveActiveSpeaker === sid
    const isThinking = !isSpeaking && effectiveThinkingSpeaker === sid
    const bubbleInfo = resolveBubbleInfo(sid)
    const showBubble = allowedBubbleSpeakers.has(sid) && Boolean(bubbleInfo.text)

    if (sid === 'hermes_moderator') {
      const stepsForward = effectiveStage === 'stage_intro' || effectiveStage === 'synthesis'
      const pos: Vec3 = stepsForward
        ? { x: 0, y: 0.14, z: -5.9 }
        : ROOM_LAYOUT.hermesStandPos
      const pose: RobloxAvatarPose =
        effectiveStage === 'synthesis' && isSpeaking
          ? 'celebrate'
          : isSpeaking
            ? 'stand_talk'
            : isThinking
              ? 'typing'
              : 'stand_talk'
      targetsById[sid] = {
        speakerId: sid,
        targetPos: pos,
        targetRotY: effectiveActiveSpeaker && effectiveActiveSpeaker !== sid
          ? yawToward(pos, focalPoint)
          : 0,
        pose: isSpeaking || isThinking || effectiveStage !== 'idle' ? pose : 'stand_talk',
        isSpeaking,
        isThinking,
        isEnabled: true,
        showBubble,
        bubbleText: bubbleInfo.text,
        stanceTag: bubbleInfo.stance,
      }
      continue
    }

    if (sid === 'user') {
      const interjecting = effectiveStage === 'user_interjection'
      const pos = interjecting ? ROOM_LAYOUT.userInterjectPos : ROOM_LAYOUT.userCeoSeatPos
      const rotY = interjecting
        ? yawToward(ROOM_LAYOUT.userInterjectPos, ROOM_LAYOUT.councilCenter)
        : 0.28
      targetsById[sid] = {
        speakerId: sid,
        targetPos: pos,
        targetRotY: rotY,
        pose: interjecting ? 'jump_interject' : 'sit_desk',
        isSpeaking: interjecting && isSpeaking,
        isThinking: false,
        isEnabled: true,
        showBubble,
        bubbleText: bubbleInfo.text,
        stanceTag: bubbleInfo.stance,
      }
      continue
    }

    if (!isEnabled) {
      const slot =
        ROOM_LAYOUT.loungeSeatSlots[loungeCursor % ROOM_LAYOUT.loungeSeatSlots.length]!
      loungeCursor += 1
      targetsById[sid] = {
        speakerId: sid,
        targetPos: { x: slot.x, y: slot.y, z: slot.z },
        targetRotY: slot.rotY,
        pose: 'lounge',
        isSpeaking: false,
        isThinking: false,
        isEnabled: false,
        showBubble: false,
        bubbleText: '',
      }
      continue
    }

    // Idle state before meeting starts -> sit at Zone C 2-monitor research desks
    if (effectiveStage === 'idle') {
      const desk = WORKSTATION_SLOTS[sid]
      const targetPos = desk ? desk.chairPos : ROOM_LAYOUT.councilCenter
      const targetRotY = desk ? desk.rotY : 0
      targetsById[sid] = {
        speakerId: sid,
        targetPos,
        targetRotY,
        pose: isThinking ? 'typing' : 'sit_desk',
        isSpeaking,
        isThinking,
        isEnabled: true,
        showBubble,
        bubbleText: bubbleInfo.text,
        stanceTag: bubbleInfo.stance,
      }
      continue
    }

    // Crossfire duelists stand at the central duel pad facing one another
    if (duelPair && (sid === duelPair[0] || sid === duelPair[1])) {
      const isLeft = sid === duelPair[0]
      const pos = isLeft ? ROOM_LAYOUT.duelLeftPos : ROOM_LAYOUT.duelRightPos
      const oppPos = isLeft ? ROOM_LAYOUT.duelRightPos : ROOM_LAYOUT.duelLeftPos
      targetsById[sid] = {
        speakerId: sid,
        targetPos: pos,
        targetRotY: yawToward(pos, oppPos),
        pose: 'duel',
        isSpeaking,
        isThinking,
        isEnabled: true,
        showBubble,
        bubbleText: bubbleInfo.text,
        stanceTag: bubbleInfo.stance,
      }
      continue
    }

    // Quant Sandbox Math calculation -> walk to Quant Rack on the right floor
    if (sandboxCalcSpeakerId === sid) {
      targetsById[sid] = {
        speakerId: sid,
        targetPos: ROOM_LAYOUT.quantStandPos,
        targetRotY: yawToward(ROOM_LAYOUT.quantStandPos, ROOM_LAYOUT.quantRackPos),
        pose: 'calc',
        isSpeaking,
        isThinking,
        isEnabled: true,
        showBubble,
        bubbleText: bubbleInfo.text,
        stanceTag: bubbleInfo.stance,
      }
      continue
    }

    // Standard Council Table seating around the Zone A Circular Holo-Console
    const seatIdx = Math.max(0, orderedEnabledPanelIds.indexOf(sid))
    const councilTransform = getCouncilSeatTransform(seatIdx)
    const lookTarget =
      effectiveStage === 'synthesis'
        ? ROOM_LAYOUT.mainWallScreenPos
        : ROOM_LAYOUT.councilCenter

    const targetRotY = yawToward(councilTransform.pos, lookTarget)

    // Smoothly turn head toward whoever is currently presenting if seated neighbor is listening
    let headLookYaw = 0
    if (!isSpeaking && !isThinking && effectiveActiveSpeaker && effectiveActiveSpeaker !== sid) {
      let yawDelta = yawToward(councilTransform.pos, focalPoint) - targetRotY
      while (yawDelta > Math.PI) yawDelta -= Math.PI * 2
      while (yawDelta < -Math.PI) yawDelta += Math.PI * 2
      headLookYaw = Math.max(-0.5, Math.min(0.5, yawDelta))
    }

    let pose: RobloxAvatarPose = 'sit_table'
    if (isSpeaking) {
      pose = 'holo_point'
    } else if (isThinking) {
      pose = 'tap_terminal'
    } else if (
      effectiveStage === 'synthesis' &&
      (bubbleInfo.stance === 'BUY' || bubbleInfo.stance === 'OVERWEIGHT')
    ) {
      pose = 'celebrate'
    }

    targetsById[sid] = {
      speakerId: sid,
      targetPos: councilTransform.pos,
      targetRotY,
      headLookYaw,
      pose,
      isSpeaking,
      isThinking,
      isEnabled: true,
      showBubble,
      bubbleText: bubbleInfo.text,
      stanceTag: bubbleInfo.stance,
    }
  }

  return {
    effectiveStage,
    activeStepIndex,
    isDemoActive,
    spotlightSpeakerId: effectiveActiveSpeaker || effectiveThinkingSpeaker,
    targetsById,
  }
}
