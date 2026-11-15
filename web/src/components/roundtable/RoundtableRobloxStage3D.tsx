import { useEffect, useMemo, useRef, useState } from 'react'
import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import { api } from '../../api/client'
import type {
  DebatePillarCoverageDTO,
  OHLCVCandleDTO,
  RoundtableStanceType,
  RoundtableTurnDTO,
} from '../../api/types'
import { getOrderedSeatIds, getSpeakerMeta, stanceBadgeClass, type SpeakerVisualMeta } from './personas'
import type { GuruPersonaDTO } from '../../api/types'
import {
  ALL_STAGE_CHARACTER_IDS,
  ROUNDTABLE_STEPPER_STEPS,
  computeObstacleAwareStep,
  computeRoundtableChoreography,
} from './roundtableChoreography'
import {
  animateRobloxR6Avatar,
  createRobloxR6Avatar,
  disposeRobloxR6Avatar,
  type RobloxR6AvatarHandle,
} from './roundtableRobloxAvatars'
import {
  buildRobloxTycoonRoom,
  createStudioEnvironmentMap,
  type LightingThemeMode,
  type RobloxRoomHandle,
} from './roundtableRobloxRoom'
import {
  CompactLayoutIcon,
  ExpandIcon,
  PauseIcon,
  PlayIcon,
  SlidersIcon,
  SunIcon,
  VolumeOffIcon,
  VolumeOnIcon,
} from '../ui/AppIcons'

export interface RoundtableRobloxStage3DProps {
  personas?: GuruPersonaDTO[]
  turns: RoundtableTurnDTO[]
  prioritizedSpeakers: string[]
  activeSpeakerId: string | null
  thinkingSpeakerId: string | null
  thinkingStage?: string | null
  mutedSpeakers: Set<string>
  primaryTicker?: string | null
  portfolioAttachmentLabel?: string
  finalRating?: RoundtableStanceType | string | null
  pillarCoverage?: DebatePillarCoverageDTO | null
  hasRecentSandboxMath?: boolean
  voiceEnabled?: boolean
  getVoiceEnergy?: () => number
  showContextGuide?: boolean
  isExpanded?: boolean
  fullViewport?: boolean
  onToggleExpanded?: () => void
  onToggleVoiceEnabled?: (next: boolean) => void
  onToggleContextGuide?: () => void
  onToggleMute: (speakerId: string) => void
  onMentionSpeaker?: (speakerId: string) => void
  onInspectSpeaker?: (speakerId: string) => void
  onCustomizeSpeaker?: (speakerId: string) => void
  onSwitchTo2D?: () => void
}

type CameraPresetMode = 'overview' | 'council' | 'follow'

function canUseWebGL(): boolean {
  if (typeof window === 'undefined' || typeof document === 'undefined') return false
  // Vitest jsdom mocks getContext or lacks real WebGL shaders
  if (typeof navigator !== 'undefined' && /jsdom/i.test(navigator.userAgent)) return false
  try {
    const testCanvas = document.createElement('canvas')
    return Boolean(
      testCanvas.getContext('webgl2') ||
        testCanvas.getContext('webgl') ||
        testCanvas.getContext('experimental-webgl'),
    )
  } catch {
    return false
  }
}

function wrapAngleDelta(current: number, target: number): number {
  let diff = (target - current) % (Math.PI * 2)
  if (diff > Math.PI) diff -= Math.PI * 2
  if (diff < -Math.PI) diff += Math.PI * 2
  return diff
}

function getCameraPresetVectors(mode: CameraPresetMode): {
  pos: THREE.Vector3
  target: THREE.Vector3
} {
  if (mode === 'council') {
    return {
      pos: new THREE.Vector3(4.9, 4.7, 5.6),
      target: new THREE.Vector3(0, 1.45, -1.2),
    }
  }
  return {
    pos: new THREE.Vector3(8.8, 8.6, 13.4),
    target: new THREE.Vector3(0, 1.35, -1.25),
  }
}

function formatBangkokClock(): string {
  try {
    return new Intl.DateTimeFormat('th-TH', {
      timeZone: 'Asia/Bangkok',
      hour: '2-digit',
      minute: '2-digit',
      hour12: false,
    }).format(new Date())
  } catch {
    const now = new Date()
    return `${String(now.getHours()).padStart(2, '0')}:${String(now.getMinutes()).padStart(2, '0')}`
  }
}

export default function RoundtableRobloxStage3D({
  personas,
  turns,
  prioritizedSpeakers,
  activeSpeakerId,
  thinkingSpeakerId,
  thinkingStage,
  mutedSpeakers,
  primaryTicker,
  portfolioAttachmentLabel,
  finalRating,
  pillarCoverage,
  hasRecentSandboxMath = false,
  voiceEnabled = true,
  getVoiceEnergy,
  showContextGuide = false,
  isExpanded = false,
  fullViewport = false,
  onToggleExpanded,
  onToggleVoiceEnabled,
  onToggleContextGuide,
  onToggleMute,
  onMentionSpeaker,
  onInspectSpeaker,
  onCustomizeSpeaker,
  onSwitchTo2D,
}: RoundtableRobloxStage3DProps) {
  const containerRef = useRef<HTMLDivElement | null>(null)
  const overlayRefs = useRef<Record<string, HTMLDivElement | null>>({})

  // HUD interactive controls — start in Metropolitan Morning ('day') matching reference aesthetic
  const [showNameplates, setShowNameplates] = useState(true)
  const [showBubbles, setShowBubbles] = useState(true)
  const [lightingMode, setLightingMode] = useState<LightingThemeMode>('day')
  const [cameraMode, setCameraMode] = useState<CameraPresetMode>('overview')
  const [demoStepIndex, setDemoStepIndex] = useState<number | null>(null)
  const [demoAutoPlay, setDemoAutoPlay] = useState(false)
  const [clockLabel, setClockLabel] = useState(() => formatBangkokClock())
  const [webglReady] = useState(() => canUseWebGL())
  const [ohlcvCandles, setOhlcvCandles] = useState<OHLCVCandleDTO[] | undefined>(undefined)

  // Fetch live OHLCV bars for the central 3D candlestick hologram when WebGL & ticker are active
  useEffect(() => {
    const ticker = primaryTicker?.trim()
    if (!webglReady || !ticker) {
      setOhlcvCandles(undefined)
      return
    }
    const controller = new AbortController()
    api
      .getEquityOHLCV(ticker, '1mo', '1d', controller.signal)
      .then((res) => {
        if (!controller.signal.aborted) {
          setOhlcvCandles(res.candles)
        }
      })
      .catch(() => {
        if (!controller.signal.aborted) {
          setOhlcvCandles(undefined)
        }
      })
    return () => controller.abort()
  }, [webglReady, primaryTicker])

  // Update Bangkok clock every 30s
  useEffect(() => {
    const id = window.setInterval(() => setClockLabel(formatBangkokClock()), 30_000)
    return () => window.clearInterval(id)
  }, [])

  // Auto-advance walkthrough demo steps cleanly when user clicks "สาธิต"
  useEffect(() => {
    if (!demoAutoPlay || demoStepIndex === null) return
    const timer = window.setInterval(() => {
      setDemoStepIndex((prev) => {
        if (prev === null) return 0
        return (prev + 1) % ROUNDTABLE_STEPPER_STEPS.length
      })
    }, 3400)
    return () => window.clearInterval(timer)
  }, [demoAutoPlay, demoStepIndex])

  // Build resolved persona metadata map for all 7 stage characters
  const metasById = useMemo(() => {
    const map: Record<string, SpeakerVisualMeta> = {}
    for (const sid of ALL_STAGE_CHARACTER_IDS) {
      map[sid] = getSpeakerMeta(sid, personas)
    }
    return map
  }, [personas])

  const orderedPanelIds = useMemo(
    () => getOrderedSeatIds(personas, false).filter((sid) => sid !== 'hermes_moderator'),
    [personas],
  )

  const latestStancesBySpeaker = useMemo(() => {
    const map: Record<string, RoundtableStanceType> = {}
    for (const t of turns) {
      if (t.stance_tag) map[t.speaker_id] = t.stance_tag
    }
    return map
  }, [turns])

  // Compute deterministic Choreography targets for all 7 avatars
  const choreography = useMemo(
    () =>
      computeRoundtableChoreography({
        metasById,
        orderedPanelIds,
        turns,
        activeSpeakerId,
        thinkingSpeakerId,
        roundStage: thinkingStage ?? null,
        hasActiveSandboxMath: hasRecentSandboxMath,
        finalRating,
        demoStepIndex,
      }),
    [
      metasById,
      orderedPanelIds,
      turns,
      activeSpeakerId,
      thinkingSpeakerId,
      thinkingStage,
      hasRecentSandboxMath,
      finalRating,
      demoStepIndex,
    ],
  )

  // Keep latest choreography & mode refs readable inside RAF loop without tearing down WebGL scene
  const choreographyRef = useRef(choreography)
  choreographyRef.current = choreography
  const cameraModeRef = useRef(cameraMode)
  cameraModeRef.current = cameraMode
  const lightingModeRef = useRef(lightingMode)
  lightingModeRef.current = lightingMode
  const hasRecentSandboxMathRef = useRef(hasRecentSandboxMath)
  hasRecentSandboxMathRef.current = hasRecentSandboxMath
  const getVoiceEnergyRef = useRef(getVoiceEnergy)
  getVoiceEnergyRef.current = getVoiceEnergy
  const fullViewportRef = useRef(fullViewport)
  fullViewportRef.current = fullViewport
  const cameraTransitionRef = useRef<{
    active: boolean
    pos: THREE.Vector3
    target: THREE.Vector3
  }>({
    active: false,
    pos: new THREE.Vector3(),
    target: new THREE.Vector3(),
  })
  const callbacksRef = useRef({
    onMentionSpeaker,
    onInspectSpeaker,
  })
  callbacksRef.current = { onMentionSpeaker, onInspectSpeaker }

  const wallboardPayload = useMemo(() => {
    const activeStep =
      choreography.activeStepIndex >= 0
        ? ROUNDTABLE_STEPPER_STEPS[choreography.activeStepIndex]
        : null
    const demoStances: Record<string, RoundtableStanceType> = { ...latestStancesBySpeaker }
    if (choreography.isDemoActive) {
      for (const [sid, t] of Object.entries(choreography.targetsById)) {
        if (t.stanceTag) demoStances[sid] = t.stanceTag
      }
    }
    return {
      primaryTicker,
      portfolioLabel: portfolioAttachmentLabel,
      finalRating:
        choreography.isDemoActive && choreography.effectiveStage === 'synthesis'
          ? 'OVERWEIGHT'
          : finalRating,
      activeStageLabel: activeStep?.fullLabel ?? 'รอเปิดวาระประชุมสภากูรู',
      metas: ALL_STAGE_CHARACTER_IDS.filter((sid) => sid !== 'user').map((sid) => metasById[sid]!),
      stancesBySpeaker: demoStances,
      pillarCoverage,
      hasSandboxMathActive:
        hasRecentSandboxMath ||
        (choreography.isDemoActive && choreography.effectiveStage === 'user_interjection'),
      candles: ohlcvCandles,
    }
  }, [
    primaryTicker,
    portfolioAttachmentLabel,
    finalRating,
    choreography,
    latestStancesBySpeaker,
    metasById,
    pillarCoverage,
    hasRecentSandboxMath,
    ohlcvCandles,
  ])
  const wallboardPayloadRef = useRef(wallboardPayload)
  wallboardPayloadRef.current = wallboardPayload

  // Store Three.js handles cleanly across lighting / wallboard updates
  const sceneHandlesRef = useRef<{
    room: RobloxRoomHandle
    scene: THREE.Scene
    camera: THREE.PerspectiveCamera
    renderer: THREE.WebGLRenderer
    controls: OrbitControls
    avatars: Record<string, RobloxR6AvatarHandle>
    envRenderTarget: THREE.WebGLRenderTarget | null
    userDraggedCamera: boolean
  } | null>(null)

  // Sync LED Wallboards & lighting mode immediately on state changes
  useEffect(() => {
    const handles = sceneHandlesRef.current
    if (!handles) return
    handles.room.applyLightingTheme(lightingMode)
    const skyHex = lightingMode === 'day' ? '#A8D8F7' : '#0B1A2A'
    const fogHex = lightingMode === 'day' ? '#BAE6FD' : '#0B1A2A'
    handles.scene.background = new THREE.Color(skyHex)
    handles.scene.fog = new THREE.Fog(fogHex, 32, 72)

    handles.envRenderTarget?.dispose()
    const envRes = createStudioEnvironmentMap(handles.renderer, lightingMode)
    handles.envRenderTarget = envRes?.renderTarget ?? null
    handles.scene.environment = envRes?.texture ?? null
    handles.scene.environmentIntensity = lightingMode === 'day' ? 0.65 : 0.48
  }, [lightingMode])

  useEffect(() => {
    const handles = sceneHandlesRef.current
    if (!handles) return
    handles.room.updateWallboards(wallboardPayload)
  }, [wallboardPayload])

  // Smoothly animate camera when user switches camera preset button
  useEffect(() => {
    const handles = sceneHandlesRef.current
    if (!handles) return
    handles.userDraggedCamera = false
    if (cameraMode === 'follow') {
      cameraTransitionRef.current.active = false
      return
    }
    const { pos, target } = getCameraPresetVectors(cameraMode)
    const reducedMotion =
      typeof window !== 'undefined' &&
      typeof window.matchMedia === 'function' &&
      window.matchMedia('(prefers-reduced-motion: reduce)').matches
    if (reducedMotion) {
      cameraTransitionRef.current.active = false
      handles.camera.position.copy(pos)
      handles.controls.target.copy(target)
      handles.controls.update()
      return
    }
    cameraTransitionRef.current.pos.copy(pos)
    cameraTransitionRef.current.target.copy(target)
    cameraTransitionRef.current.active = true
  }, [cameraMode])

  // Main Three.js Scene Initialization & Lifecycle
  useEffect(() => {
    const container = containerRef.current
    if (!container || !webglReady) return

    const reducedMotion =
      typeof window !== 'undefined' &&
      typeof window.matchMedia === 'function' &&
      window.matchMedia('(prefers-reduced-motion: reduce)').matches

    const scene = new THREE.Scene()
    const initSkyHex = lightingModeRef.current === 'day' ? '#A8D8F7' : '#0B1A2A'
    const initFogHex = lightingModeRef.current === 'day' ? '#BAE6FD' : '#0B1A2A'
    scene.background = new THREE.Color(initSkyHex)
    scene.fog = new THREE.Fog(initFogHex, 32, 72)

    const width = Math.max(container.clientWidth, 320)
    const height = Math.max(container.clientHeight, 240)
    const camera = new THREE.PerspectiveCamera(38, width / height, 0.2, 85)
    const initialCam = getCameraPresetVectors(cameraModeRef.current)
    camera.position.copy(initialCam.pos)

    const renderer = new THREE.WebGLRenderer({
      antialias: true,
      powerPreference: 'high-performance',
    })
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2))
    renderer.setSize(width, height)
    renderer.shadowMap.enabled = true
    renderer.shadowMap.type = THREE.PCFSoftShadowMap
    renderer.toneMapping = THREE.ACESFilmicToneMapping
    renderer.toneMappingExposure = 1.06
    renderer.domElement.style.display = 'block'
    renderer.domElement.style.width = '100%'
    renderer.domElement.style.height = '100%'
    renderer.domElement.setAttribute('aria-hidden', 'true')
    container.appendChild(renderer.domElement)

    const initEnvRes = createStudioEnvironmentMap(renderer, lightingModeRef.current)
    scene.environment = initEnvRes?.texture ?? null
    scene.environmentIntensity = lightingModeRef.current === 'day' ? 0.65 : 0.48

    const controls = new OrbitControls(camera, renderer.domElement)
    controls.enableDamping = true
    controls.dampingFactor = 0.08
    controls.target.copy(initialCam.target)
    controls.minDistance = 5.5
    controls.maxDistance = 26
    controls.minPolarAngle = 0.28
    controls.maxPolarAngle = Math.PI / 2.22 // Prevent orbiting underground

    const room = buildRobloxTycoonRoom(metasById, renderer.capabilities.getMaxAnisotropy())
    room.applyLightingTheme(lightingModeRef.current)
    scene.add(room.group)

    // Spawn all 7 Roblox R6 character avatars cleanly at their initial choreography targets
    const avatars: Record<string, RobloxR6AvatarHandle> = {}
    const livePos: Record<string, THREE.Vector3> = {}
    const liveRotY: Record<string, number> = {}
    const initialTargets = choreographyRef.current.targetsById

    for (const sid of ALL_STAGE_CHARACTER_IDS) {
      const avatar = createRobloxR6Avatar(metasById[sid]!)
      const initT = initialTargets[sid]
      if (initT) {
        avatar.root.position.set(initT.targetPos.x, initT.targetPos.y, initT.targetPos.z)
        avatar.root.rotation.y = initT.targetRotY
      }
      scene.add(avatar.root)
      avatars[sid] = avatar
      livePos[sid] = avatar.root.position.clone()
      liveRotY[sid] = avatar.root.rotation.y
    }

    const handlesObj = {
      room,
      scene,
      camera,
      renderer,
      controls,
      avatars,
      envRenderTarget: initEnvRes?.renderTarget ?? null,
      userDraggedCamera: false,
    }
    sceneHandlesRef.current = handlesObj

    // Initial Wallboard draw right after mount
    room.updateWallboards(wallboardPayloadRef.current)

    // Raycast click detection on avatars, desks, & central hologram (distinguishes click vs drag)
    const raycaster = new THREE.Raycaster()
    const pointerDownScreen = new THREE.Vector2()
    const onPointerDown = (e: PointerEvent) => {
      pointerDownScreen.set(e.clientX, e.clientY)
      handlesObj.userDraggedCamera = true
      cameraTransitionRef.current.active = false
    }
    const onPointerUp = (e: PointerEvent) => {
      const movedPx = Math.hypot(e.clientX - pointerDownScreen.x, e.clientY - pointerDownScreen.y)
      if (movedPx > 6) return
      const rect = renderer.domElement.getBoundingClientRect()
      if (!rect.width || !rect.height) return
      const ndc = new THREE.Vector2(
        ((e.clientX - rect.left) / rect.width) * 2 - 1,
        -((e.clientY - rect.top) / rect.height) * 2 + 1,
      )
      raycaster.setFromCamera(ndc, camera)
      const hits = raycaster.intersectObjects(scene.children, true)
      for (const hit of hits) {
        let obj: THREE.Object3D | null = hit.object
        while (obj) {
          const ud = obj.userData as {
            speakerId?: string
            workstationSpeakerId?: string
            holoFocus?: boolean
          }
          if (ud.speakerId && ud.speakerId !== 'user') {
            callbacksRef.current.onMentionSpeaker?.(ud.speakerId)
            return
          }
          if (ud.workstationSpeakerId) {
            callbacksRef.current.onInspectSpeaker?.(ud.workstationSpeakerId)
            return
          }
          if (ud.holoFocus) {
            setCameraMode((prev) => (prev === 'council' ? 'overview' : 'council'))
            return
          }
          obj = obj.parent
        }
      }
    }
    renderer.domElement.addEventListener('pointerdown', onPointerDown)
    renderer.domElement.addEventListener('pointerup', onPointerUp)

    // Handle ResizeObserver dynamically when toggling Compact <-> Theatre or sidebar
    const handleResize = () => {
      if (!container) return
      const w = Math.max(container.clientWidth, 320)
      const h = Math.max(container.clientHeight, 240)
      camera.aspect = w / h
      camera.updateProjectionMatrix()
      renderer.setSize(w, h)
    }
    const ro = typeof ResizeObserver !== 'undefined' ? new ResizeObserver(handleResize) : null
    ro?.observe(container)

    const clock = new THREE.Clock()
    const anchorWorldPos = new THREE.Vector3()
    let rafId = 0

    const tick = () => {
      rafId = window.requestAnimationFrame(tick)
      const dt = Math.min(0.05, clock.getDelta())
      const elapsed = clock.elapsedTime
      const curChoreo = choreographyRef.current

      const curW = container.clientWidth
      const curH = container.clientHeight

      // Sample live voice audio amplitude once per frame (with gentle demo fallback during walkthrough)
      const meteredEnergy = getVoiceEnergyRef.current?.() ?? 0
      const demoFallbackEnergy =
        meteredEnergy <= 0.02 && curChoreo.isDemoActive && curChoreo.spotlightSpeakerId
          ? 0.32 + 0.44 * Math.abs(Math.sin(elapsed * 8.4))
          : 0
      const activeVoiceEnergy = Math.max(meteredEnergy, demoFallbackEnergy)

      // 1. Move & Animate each Roblox R6 character toward choreography targets
      for (const sid of ALL_STAGE_CHARACTER_IDS) {
        const avatar = avatars[sid]
        const target = curChoreo.targetsById[sid]
        const pos = livePos[sid]
        if (!avatar || !target || !pos) continue

        const dx = target.targetPos.x - pos.x
        const dz = target.targetPos.z - pos.z
        const distXZ = Math.hypot(dx, dz)

        let activePose = target.pose
        if (reducedMotion) {
          pos.set(target.targetPos.x, target.targetPos.y, target.targetPos.z)
          liveRotY[sid] = target.targetRotY
        } else if (distXZ > 0.08) {
          // Obstacle-aware R6 Walk Cycle interpolation toward destination (avoids Holotable & walks through CEO doorway)
          const nextPos = computeObstacleAwareStep(
            { x: pos.x, y: pos.y, z: pos.z },
            target.targetPos,
            dt * 4.4,
          )
          const stepDx = nextPos.x - pos.x
          const stepDz = nextPos.z - pos.z
          pos.set(nextPos.x, nextPos.y, nextPos.z)
          const walkYaw =
            Math.hypot(stepDx, stepDz) > 1e-4 ? Math.atan2(stepDx, stepDz) : Math.atan2(dx, dz)
          liveRotY[sid] =
            (liveRotY[sid] ?? 0) +
            wrapAngleDelta(liveRotY[sid] ?? 0, walkYaw) * Math.min(1, dt * 10)
          activePose = 'walk'
        } else {
          pos.set(target.targetPos.x, target.targetPos.y, target.targetPos.z)
          liveRotY[sid] =
            (liveRotY[sid] ?? 0) +
            wrapAngleDelta(liveRotY[sid] ?? 0, target.targetRotY) * Math.min(1, dt * 8)
        }

        avatar.root.position.copy(pos)
        avatar.root.rotation.y = liveRotY[sid] ?? 0

        const avatarSpeaking = target.isSpeaking && activePose !== 'walk'
        animateRobloxR6Avatar(
          avatar,
          activePose,
          elapsed,
          avatarSpeaking,
          target.isThinking && activePose !== 'walk',
          reducedMotion,
          target.headLookYaw ?? 0,
          dt,
          avatarSpeaking ? activeVoiceEnergy : 0,
        )

        // Direct DOM 3D-to-2D Screen Projection with camera depth zIndex & safe viewport margins
        const overlayEl = overlayRefs.current[sid]
        if (overlayEl && curW > 0 && curH > 0) {
          avatar.headAnchor.getWorldPosition(anchorWorldPos)
          anchorWorldPos.project(camera)
          if (anchorWorldPos.z >= -1 && anchorWorldPos.z <= 1) {
            const rawX = (anchorWorldPos.x * 0.5 + 0.5) * curW
            const rawY = (-anchorWorldPos.y * 0.5 + 0.5) * curH
            const padX = curW > 220 ? 74 : 24
            const minTopY = fullViewportRef.current ? 128 : 88
            const screenX = Math.max(padX, Math.min(curW - padX, rawX))
            const screenY = Math.max(minTopY, Math.min(Math.max(minTopY, curH - 16), rawY))
            const depthOrder = Math.max(
              1,
              Math.min(100, Math.round((1 - (anchorWorldPos.z * 0.5 + 0.5)) * 100)),
            )
            overlayEl.style.opacity = '1'
            overlayEl.style.zIndex = String(depthOrder)
            overlayEl.style.transform = `translate3d(${screenX.toFixed(1)}px, ${screenY.toFixed(1)}px, 0) translate(-50%, -100%)`
          } else {
            overlayEl.style.opacity = '0'
          }
        }
      }

      // 2. Smoothly ease camera between presets or gently follow active speaker
      const camTrans = cameraTransitionRef.current
      if (camTrans.active && cameraModeRef.current !== 'follow') {
        const alpha = Math.min(1, dt * 6.5)
        camera.position.lerp(camTrans.pos, alpha)
        controls.target.lerp(camTrans.target, alpha)
        if (
          camera.position.distanceTo(camTrans.pos) < 0.03 &&
          controls.target.distanceTo(camTrans.target) < 0.03
        ) {
          camera.position.copy(camTrans.pos)
          controls.target.copy(camTrans.target)
          camTrans.active = false
        }
      } else if (cameraModeRef.current === 'follow' && curChoreo.spotlightSpeakerId) {
        const focusPos = livePos[curChoreo.spotlightSpeakerId]
        if (focusPos) {
          const desiredTarget = new THREE.Vector3(focusPos.x, 1.15, focusPos.z)
          controls.target.lerp(desiredTarget, Math.min(1, dt * 3.5))
          if (!handlesObj.userDraggedCamera) {
            const desiredCam = new THREE.Vector3(focusPos.x + 5.2, 6.2, focusPos.z + 6.8)
            camera.position.lerp(desiredCam, Math.min(1, dt * 2.4))
          }
        }
      }

      // 3. Room ambient LED effects (Crossfire pad, Quant Sandbox rack LEDs, & voice-reactive hologram)
      room.animateRoomEffects(
        elapsed,
        curChoreo.effectiveStage === 'crossfire',
        hasRecentSandboxMathRef.current ||
          (curChoreo.isDemoActive && curChoreo.effectiveStage === 'user_interjection'),
        reducedMotion,
        curChoreo.spotlightSpeakerId ? activeVoiceEnergy : 0,
      )

      controls.update()
      renderer.render(scene, camera)
    }

    rafId = window.requestAnimationFrame(tick)

    return () => {
      window.cancelAnimationFrame(rafId)
      ro?.disconnect()
      renderer.domElement.removeEventListener('pointerdown', onPointerDown)
      renderer.domElement.removeEventListener('pointerup', onPointerUp)
      controls.dispose()
      handlesObj.envRenderTarget?.dispose()
      for (const sid of ALL_STAGE_CHARACTER_IDS) {
        const av = avatars[sid]
        if (av) disposeRobloxR6Avatar(av)
      }
      room.dispose()
      renderer.dispose()
      if (renderer.domElement.parentNode === container) {
        container.removeChild(renderer.domElement)
      }
      sceneHandlesRef.current = null
    }
    // Re-build Three scene only if WebGL availability changes or persona catalog identities reset
  }, [webglReady, metasById])

  const cycleCameraMode = () => {
    setCameraMode((prev) =>
      prev === 'overview' ? 'council' : prev === 'council' ? 'follow' : 'overview',
    )
  }

  const toggleDemoMode = () => {
    if (choreography.isDemoActive) {
      setDemoAutoPlay(false)
      setDemoStepIndex(null)
    } else {
      setDemoStepIndex(0)
      setDemoAutoPlay(true)
    }
  }

  const cameraModeLabel =
    cameraMode === 'overview' ? 'ทั้งห้อง' : cameraMode === 'council' ? 'โต๊ะสภา' : 'ตามคนพูด'

  const stageHeightClass = fullViewport
    ? 'h-full w-full'
    : isExpanded
      ? 'h-[430px] sm:h-[480px]'
      : 'h-[268px] sm:h-[296px]'

  return (
    <div
      data-testid="roundtable-roblox-stage-3d"
      data-full-viewport={fullViewport ? 'true' : 'false'}
      className={`relative w-full select-none overflow-hidden transition-all duration-300 ${
        fullViewport ? '' : 'rounded-2xl border shadow-inner'
      } ${stageHeightClass} ${
        lightingMode === 'day'
          ? 'border-sky-300/80 bg-gradient-to-b from-[#E2EEF8] to-[#F1F7FC]'
          : 'border-slate-700/80 bg-[#122638]'
      }`}
    >
      {/* WebGL Canvas Host */}
      <div ref={containerRef} className="absolute inset-0" />

      {/* Fallback Scenic Isometric Stage when running in headless jsdom / non-WebGL browsers */}
      {!webglReady && (
        <div
          data-testid="roundtable-webgl-fallback"
          className="absolute inset-0 flex flex-col items-center justify-center gap-2 px-4 text-center"
        >
          <p
            className={`text-xs font-semibold ${
              lightingMode === 'day' ? 'text-slate-700' : 'text-sky-200'
            }`}
          >
            🎮 โหมดเวที Roblox 3D พร้อมทำงานในเบราว์เซอร์จริง (ปิด WebGL ชั่วคราวในสภาพแวดล้อมทดสอบนี้)
          </p>
          <div className="flex flex-wrap items-center justify-center gap-2">
            {ALL_STAGE_CHARACTER_IDS.map((sid) => {
              const meta = metasById[sid]!
              const target = choreography.targetsById[sid]
              if (!target) return null
              return (
                <button
                  key={sid}
                  type="button"
                  disabled={sid === 'user' || !target.isEnabled}
                  onClick={() => {
                    if (sid !== 'user' && target.isEnabled) onMentionSpeaker?.(sid)
                  }}
                  className={`inline-flex items-center gap-1.5 rounded-xl border px-2.5 py-1 text-[11px] font-semibold transition-colors ${
                    target.isSpeaking
                      ? 'border-sky-400 bg-sky-500/25 text-white ring-1 ring-sky-300'
                      : 'border-slate-600/70 bg-slate-800/80 text-slate-100 hover:border-sky-400/70'
                  }`}
                >
                  <span>{meta.emoji}</span>
                  <span>{meta.shortName}</span>
                  {target.stanceTag && (
                    <span className="rounded bg-white/15 px-1 text-[11px] font-bold">
                      {target.stanceTag}
                    </span>
                  )}
                </button>
              )
            })}
          </div>
        </div>
      )}

      {/* 3D-to-2D Projected Roblox BillboardGui Nameplates & Chat Bubbles */}
      {webglReady && (
        <div className="pointer-events-none select-none absolute inset-0 overflow-hidden">
          {ALL_STAGE_CHARACTER_IDS.map((sid) => {
            const meta = metasById[sid]!
            const target = choreography.targetsById[sid]
            if (!target) return null
            const isMuted = mutedSpeakers.has(sid)
            const isPrioritized = prioritizedSpeakers.includes(sid)
            const visibleNameplate = showNameplates || target.isSpeaking || target.isThinking

            return (
              <div
                key={sid}
                ref={(el) => {
                  overlayRefs.current[sid] = el
                }}
                style={{ opacity: 0, left: 0, top: 0 }}
                className="absolute will-change-transform"
              >
                <div className="flex flex-col items-center gap-1">
                  {/* Speech Bubble (Clean white Roblox Chat Bubble matching Metropolitan style) */}
                  {showBubbles && target.showBubble && target.bubbleText && (
                    <div className="relative max-w-[218px] rounded-2xl border border-white bg-white/95 px-3 py-1.5 text-[11px] leading-snug text-[#18374c] shadow-xl ring-1 ring-cyan-500/30">
                      <div className="mb-0.5 flex items-center justify-between gap-1.5">
                        <span
                          className="font-extrabold"
                          style={{ color: meta.accentColor }}
                        >
                          {meta.emoji} {meta.shortName}
                        </span>
                        {target.stanceTag && (
                          <span
                            className={`rounded border px-1 py-0 text-[11px] font-bold ${stanceBadgeClass(
                              target.stanceTag,
                            )}`}
                          >
                            {target.stanceTag}
                          </span>
                        )}
                      </div>
                      <p className="line-clamp-2 font-medium text-slate-800">
                        {target.bubbleText}
                      </p>
                      <span
                        aria-hidden="true"
                        className="absolute -bottom-1.5 left-1/2 h-2.5 w-2.5 -translate-x-1/2 rotate-45 border-b border-r border-white bg-white/95"
                      />
                    </div>
                  )}

                  {/* Roblox Nameplate Pill */}
                  {visibleNameplate && (
                    <div className="pointer-events-auto group/pin flex items-center gap-1 rounded-full border border-white/20 bg-slate-900/85 px-2 py-0.5 text-[11px] font-semibold text-white shadow-md backdrop-blur-sm">
                      <button
                        type="button"
                        disabled={sid === 'user' || !target.isEnabled}
                        onClick={() => {
                          if (sid !== 'user' && target.isEnabled) onMentionSpeaker?.(sid)
                        }}
                        title={
                          sid === 'user'
                            ? 'อวาตาร์เจ้าของพอร์ต'
                            : `คลิกเพื่อแท็ก @${meta.shortName}`
                        }
                        className="flex items-center gap-1 rounded-full text-left focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-400 disabled:cursor-default"
                      >
                        <span
                          aria-hidden="true"
                          className={`h-2 w-2 shrink-0 rounded-full ${
                            target.isSpeaking ? 'animate-pulse' : ''
                          }`}
                          style={{ backgroundColor: meta.accentColor }}
                        />
                        <span>{meta.shortName}</span>
                        {isPrioritized && sid !== 'user' && (
                          <span className="text-amber-400" title="Role Model">
                            ★
                          </span>
                        )}
                        {!target.isEnabled && (
                          <span className="text-[11px] text-slate-300">💤</span>
                        )}
                      </button>

                      {sid !== 'user' && (
                        <span className="hidden items-center gap-0.5 border-l border-white/15 pl-1 group-hover/pin:inline-flex">
                          {onCustomizeSpeaker && (
                            <button
                              type="button"
                              aria-label={`ตั้งค่าเสียง ${meta.shortName}`}
                              onClick={() => onCustomizeSpeaker(sid)}
                              className="rounded p-0.5 text-slate-300 hover:bg-white/15 hover:text-white"
                            >
                              <SlidersIcon className="h-2.5 w-2.5" />
                            </button>
                          )}
                          <button
                            type="button"
                            aria-label={
                              isMuted ? `เปิดเสียง ${meta.shortName}` : `ปิดเสียง ${meta.shortName}`
                            }
                            onClick={() => onToggleMute(sid)}
                            className={`rounded p-0.5 ${
                              isMuted
                                ? 'text-rose-400 hover:bg-rose-500/20'
                                : 'text-slate-300 hover:bg-white/15 hover:text-white'
                            }`}
                          >
                            {isMuted ? (
                              <VolumeOffIcon className="h-2.5 w-2.5" />
                            ) : (
                              <VolumeOnIcon className="h-2.5 w-2.5" />
                            )}
                          </button>
                        </span>
                      )}
                    </div>
                  )}
                </div>
              </div>
            )
          })}
        </div>
      )}

      {/* TOP HUD BAR: Compact 6-Step Stepper + Clock + Demo / 2D Toggle */}
      <div
        className={`pointer-events-none absolute inset-x-0 top-0 flex flex-col gap-1.5 ${
          fullViewport ? 'px-3 pb-2 pt-[58px]' : 'p-2'
        }`}
      >
        <div className="pointer-events-auto flex flex-wrap items-center justify-between gap-1.5 rounded-xl border border-white/15 bg-slate-900/80 px-2 py-1 text-[11px] text-white backdrop-blur-md">
          {/* 6-Step Stepper */}
          <div
            role="group"
            aria-label="ขั้นตอนการประชุมสภากูรู"
            className="flex flex-wrap items-center gap-1"
          >
            {ROUNDTABLE_STEPPER_STEPS.map((step, idx) => {
              const isActive = choreography.activeStepIndex === idx
              const isPast = choreography.activeStepIndex > idx
              return (
                <button
                  key={step.id}
                  type="button"
                  onClick={() => {
                    setDemoAutoPlay(false)
                    setDemoStepIndex(idx)
                  }}
                  title={step.description}
                  className={`rounded-lg border px-1.5 py-0.5 text-[11px] font-semibold transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-400 ${
                    isActive
                      ? 'border-sky-400 bg-sky-500 text-white shadow-sm'
                      : isPast
                        ? 'border-emerald-400/40 bg-emerald-500/15 text-emerald-200 hover:bg-emerald-500/25'
                        : 'border-white/10 bg-white/5 text-slate-300 hover:bg-white/15 hover:text-white'
                  }`}
                >
                  <span className={isActive ? 'inline' : 'hidden xl:inline'}>
                    {step.shortLabel}
                  </span>
                  <span className={isActive ? 'hidden' : 'inline xl:hidden'}>
                    {step.stepNumber}
                  </span>
                </button>
              )
            })}
          </div>

          {/* Right HUD Info: Clock + Demo Mode + Switch to 2D */}
          <div className="flex flex-wrap items-center gap-1">
            <span className="hidden rounded-lg bg-white/10 px-2 py-0.5 font-mono text-[11px] text-sky-200 sm:inline">
              กรุงเทพฯ {clockLabel}
            </span>
            <button
              type="button"
              onClick={toggleDemoMode}
              className={`inline-flex items-center gap-1 rounded-lg border px-2 py-0.5 text-[11px] font-semibold transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-400 ${
                choreography.isDemoActive
                  ? 'border-amber-300/70 bg-amber-500/25 text-amber-100 hover:bg-amber-500/35'
                  : 'border-white/15 bg-white/10 text-slate-100 hover:bg-white/20'
              }`}
            >
              {choreography.isDemoActive ? (
                <>
                  <PauseIcon className="h-3 w-3" />
                  <span>กลับไปดูงานจริง</span>
                </>
              ) : (
                <>
                  <PlayIcon className="h-3 w-3" />
                  <span>โหมดสาธิต</span>
                </>
              )}
            </button>
            {onSwitchTo2D && !fullViewport && (
              <button
                type="button"
                onClick={onSwitchTo2D}
                title="สลับกลับเป็นมุมมองการ์ด 2D ปกติ"
                className="rounded-lg border border-white/15 bg-white/10 px-2 py-0.5 text-[11px] font-semibold text-slate-100 transition-colors hover:bg-white/20 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-400"
              >
                🪪 มุมมองปกติ
              </button>
            )}
          </div>
        </div>

        {/* Row 2: Compact Pill Toggles matching Reference HUD */}
        <div className="flex flex-wrap items-center justify-between gap-1.5">
          <div className="pointer-events-auto flex flex-wrap items-center gap-1">
            {onToggleContextGuide && (
              <button
                type="button"
                aria-pressed={showContextGuide}
                onClick={onToggleContextGuide}
                className={`rounded-full border px-2.5 py-0.5 text-[11px] font-semibold backdrop-blur-md transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-400 ${
                  showContextGuide
                    ? 'border-sky-300 bg-sky-500 text-white'
                    : 'border-white/15 bg-slate-900/70 text-slate-200 hover:bg-slate-800/85'
                }`}
              >
                แผงข้อมูล
              </button>
            )}
            <button
              type="button"
              aria-pressed={showNameplates}
              onClick={() => setShowNameplates((v) => !v)}
              className={`rounded-full border px-2.5 py-0.5 text-[11px] font-semibold backdrop-blur-md transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-400 ${
                showNameplates
                  ? 'border-sky-300 bg-sky-500 text-white'
                  : 'border-white/15 bg-slate-900/70 text-slate-200 hover:bg-slate-800/85'
              }`}
            >
              ป้ายชื่อ
            </button>
            <button
              type="button"
              aria-pressed={showBubbles}
              onClick={() => setShowBubbles((v) => !v)}
              className={`rounded-full border px-2.5 py-0.5 text-[11px] font-semibold backdrop-blur-md transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-400 ${
                showBubbles
                  ? 'border-sky-300 bg-sky-500 text-white'
                  : 'border-white/15 bg-slate-900/70 text-slate-200 hover:bg-slate-800/85'
              }`}
            >
              คำพูดลอย
            </button>
            {onToggleVoiceEnabled && (
              <button
                type="button"
                aria-pressed={voiceEnabled}
                onClick={() => onToggleVoiceEnabled(!voiceEnabled)}
                className={`inline-flex items-center gap-1 rounded-full border px-2.5 py-0.5 text-[11px] font-semibold backdrop-blur-md transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-400 ${
                  voiceEnabled
                    ? 'border-sky-300 bg-sky-500 text-white'
                    : 'border-white/15 bg-slate-900/70 text-slate-200 hover:bg-slate-800/85'
                }`}
              >
                {voiceEnabled ? (
                  <VolumeOnIcon className="h-3 w-3" />
                ) : (
                  <VolumeOffIcon className="h-3 w-3" />
                )}
                <span>เสียงพูด</span>
              </button>
            )}
          </div>

          <div className="pointer-events-auto flex flex-wrap items-center gap-1">
            {fullViewport && (
              <span className="hidden rounded-full border border-white/10 bg-slate-900/55 px-2.5 py-0.5 text-[11px] font-medium text-slate-200 backdrop-blur-md xl:inline-flex">
                ลากหมุน · ซูม · คลิกตัวละคร @แท็กถาม
              </span>
            )}
            <button
              type="button"
              onClick={cycleCameraMode}
              title="สลับมุมกล้อง 3D"
              className="rounded-full border border-white/15 bg-slate-900/70 px-2.5 py-0.5 text-[11px] font-semibold text-slate-100 backdrop-blur-md transition-colors hover:bg-slate-800/90 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-400"
            >
              📷 {cameraModeLabel}
            </button>
            <button
              type="button"
              onClick={() => setLightingMode((m) => (m === 'twilight' ? 'day' : 'twilight'))}
              title="สลับแสงห้อง Studio Twilight / Daylight"
              className="inline-flex items-center gap-1 rounded-full border border-white/15 bg-slate-900/70 px-2.5 py-0.5 text-[11px] font-semibold text-slate-100 backdrop-blur-md transition-colors hover:bg-slate-800/90 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-400"
            >
              <SunIcon className="h-3 w-3 text-amber-300" />
              <span>{lightingMode === 'twilight' ? 'กลางคืน' : 'ยามสาย'}</span>
            </button>
            {onToggleExpanded && !fullViewport && (
              <button
                type="button"
                onClick={onToggleExpanded}
                title={isExpanded ? 'ย่อความสูงเวที' : 'ขยายเวที 3D เต็มตา'}
                className="inline-flex items-center gap-1 rounded-full border border-white/15 bg-slate-900/70 px-2.5 py-0.5 text-[11px] font-semibold text-slate-100 backdrop-blur-md transition-colors hover:bg-slate-800/90 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-400"
              >
                {isExpanded ? (
                  <>
                    <CompactLayoutIcon className="h-3 w-3" />
                    <span>ย่อ</span>
                  </>
                ) : (
                  <>
                    <ExpandIcon className="h-3 w-3" />
                    <span>ขยาย</span>
                  </>
                )}
              </button>
            )}
          </div>
        </div>
      </div>

      {/* BOTTOM HUD CAPTION: Controls hint + Lighting Status (avoid floating bottom composer in fullViewport) */}
      {!fullViewport ? (
        <div className="pointer-events-none absolute inset-x-0 bottom-0 flex items-center justify-between gap-2 p-2">
          <span className="rounded-lg bg-slate-900/65 px-2.5 py-1 text-[11px] font-medium text-slate-200 backdrop-blur-sm">
            ลากเพื่อหมุน · เลื่อนเพื่อซูม · คลิกตัวละครเพื่อ @แท็กถาม · คลิกโต๊ะเพื่อดูกรอบคิด
          </span>
          {choreography.isDemoActive && (
            <span className="rounded-lg border border-amber-400/50 bg-amber-500/20 px-2 py-0.5 text-[11px] font-semibold text-amber-200 backdrop-blur-sm">
              ● โหมดสาธิต
            </span>
          )}
        </div>
      ) : (
        choreography.isDemoActive && (
          <div className="pointer-events-none absolute right-3 top-[122px] flex justify-end">
            <span className="rounded-full border border-amber-400/50 bg-amber-500/25 px-2.5 py-0.5 text-[11px] font-semibold text-amber-100 backdrop-blur-md">
              ● กำลังสาธิตขั้นตอนสภากูรู
            </span>
          </div>
        )
      )}
    </div>
  )
}
