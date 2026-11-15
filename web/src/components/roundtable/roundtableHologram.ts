import * as THREE from 'three'
import type { OHLCVCandleDTO, RoundtableStanceType } from '../../api/types'

export interface HologramUpdateState {
  primaryTicker?: string | null
  activeStageLabel?: string
  finalRating?: RoundtableStanceType | string | null
  stancesBySpeaker: Record<string, RoundtableStanceType>
  candles?: OHLCVCandleDTO[] | null
}

export interface RobloxHologramHandle {
  group: THREE.Group
  updateState: (state: HologramUpdateState) => void
  animate: (
    elapsedSec: number,
    isCrossfire: boolean,
    isSandboxMathActive: boolean,
    reducedMotion: boolean,
    voiceEnergy?: number,
  ) => void
  dispose: () => void
}

interface CandleRig {
  group: THREE.Group
  body: THREE.Mesh
  wick: THREE.Mesh
  baseY: number
  baseHeight: number
  wickHeight: number
  wickCenterY: number
  isBull: boolean
  phaseOffset: number
}

interface ParticleSeed {
  radius: number
  angle: number
  speed: number
  yOffset: number
  spinSpeed: number
  scaleBase: number
}

interface CandleProfilePoint {
  x: number
  baseY: number
  height: number
  wickCenterY: number
  wickHeight: number
  bull: boolean
}

const CANDLE_COUNT = 13
const PARTICLE_COUNT = 36

const DEFAULT_CANDLE_SEEDS: ReadonlyArray<{
  x: number
  baseY: number
  height: number
  bull: boolean
}> = [
  { x: -0.78, baseY: 1.02, height: 0.24, bull: true },
  { x: -0.65, baseY: 1.12, height: 0.20, bull: false },
  { x: -0.52, baseY: 1.06, height: 0.28, bull: true },
  { x: -0.39, baseY: 1.24, height: 0.34, bull: true },
  { x: -0.26, baseY: 1.38, height: 0.22, bull: false },
  { x: -0.13, baseY: 1.32, height: 0.30, bull: true },
  { x: 0.0, baseY: 1.48, height: 0.40, bull: true },
  { x: 0.13, baseY: 1.44, height: 0.26, bull: false },
  { x: 0.26, baseY: 1.36, height: 0.24, bull: false },
  { x: 0.39, baseY: 1.46, height: 0.36, bull: true },
  { x: 0.52, baseY: 1.64, height: 0.38, bull: true },
  { x: 0.65, baseY: 1.58, height: 0.24, bull: false },
  { x: 0.78, baseY: 1.72, height: 0.44, bull: true },
]

function buildDefaultProfile(): CandleProfilePoint[] {
  return DEFAULT_CANDLE_SEEDS.map((s) => ({
    x: s.x,
    baseY: s.baseY,
    height: s.height,
    wickCenterY: s.baseY,
    wickHeight: s.height + 0.22,
    bull: s.bull,
  }))
}

/** Converts real stock OHLCV candles into normalized 3D hologram coordinates */
export function buildProfileFromOhlcv(rawCandles?: OHLCVCandleDTO[] | null): {
  points: CandleProfilePoint[]
  pctChange: number | null
  hasLiveData: boolean
} {
  if (!rawCandles || rawCandles.length < 2) {
    return { points: buildDefaultProfile(), pctChange: null, hasLiveData: false }
  }

  const valid = rawCandles.filter(
    (c) =>
      Number.isFinite(c.open) &&
      Number.isFinite(c.high) &&
      Number.isFinite(c.low) &&
      Number.isFinite(c.close) &&
      c.high >= c.low,
  )
  if (valid.length < 2) {
    return { points: buildDefaultProfile(), pctChange: null, hasLiveData: false }
  }

  const slice = valid.slice(-CANDLE_COUNT)
  // If fewer than 13 bars, resample evenly across 13 slots
  const sampled: OHLCVCandleDTO[] = []
  for (let i = 0; i < CANDLE_COUNT; i += 1) {
    const idx = Math.min(
      slice.length - 1,
      Math.floor((i / Math.max(1, CANDLE_COUNT - 1)) * (slice.length - 1)),
    )
    sampled.push(slice[idx]!)
  }

  let minLow = Infinity
  let maxHigh = -Infinity
  for (const bar of sampled) {
    if (bar.low < minLow) minLow = bar.low
    if (bar.high > maxHigh) maxHigh = bar.high
  }
  const span = Math.max(1e-6, maxHigh - minLow)
  const yMin = 0.94
  const ySpan = 0.96
  const toY = (price: number) => yMin + ((price - minLow) / span) * ySpan

  const points: CandleProfilePoint[] = sampled.map((bar, i) => {
    const x = -0.78 + (i / (CANDLE_COUNT - 1)) * 1.56
    const openY = toY(bar.open)
    const closeY = toY(bar.close)
    const lowY = toY(bar.low)
    const highY = toY(bar.high)
    const bodyHeight = Math.max(0.06, Math.abs(closeY - openY))
    const bodyCenterY = (openY + closeY) / 2
    const wickHeight = Math.max(bodyHeight + 0.06, highY - lowY)
    const wickCenterY = (lowY + highY) / 2
    return {
      x,
      baseY: bodyCenterY,
      height: bodyHeight,
      wickCenterY,
      wickHeight,
      bull: bar.close >= bar.open,
    }
  })

  const firstOpen = slice[0]!.open
  const lastClose = slice[slice.length - 1]!.close
  const pctChange = firstOpen > 0 ? ((lastClose - firstOpen) / firstOpen) * 100 : null

  return { points, pctChange, hasLiveData: true }
}

function create3DArrowGeometry(pointsUp: boolean): THREE.ExtrudeGeometry {
  const s = new THREE.Shape()
  if (pointsUp) {
    s.moveTo(-0.075, -0.26)
    s.lineTo(0.075, -0.26)
    s.lineTo(0.075, 0.04)
    s.lineTo(0.21, 0.04)
    s.lineTo(0, 0.30)
    s.lineTo(-0.21, 0.04)
    s.lineTo(-0.075, 0.04)
    s.closePath()
  } else {
    s.moveTo(-0.075, 0.26)
    s.lineTo(0.075, 0.26)
    s.lineTo(0.075, -0.04)
    s.lineTo(0.21, -0.04)
    s.lineTo(0, -0.30)
    s.lineTo(-0.21, -0.04)
    s.lineTo(-0.075, -0.04)
    s.closePath()
  }
  const geom = new THREE.ExtrudeGeometry(s, {
    depth: 0.09,
    bevelEnabled: true,
    bevelThickness: 0.02,
    bevelSize: 0.015,
    bevelSegments: 2,
  })
  geom.center()
  return geom
}

function createHoloHaloTexture(): THREE.CanvasTexture {
  const canvas = document.createElement('canvas')
  canvas.width = 128
  canvas.height = 128
  const ctx = canvas.getContext('2d')
  if (ctx) {
    const grad = ctx.createRadialGradient(64, 64, 4, 64, 64, 62)
    grad.addColorStop(0, 'rgba(165, 243, 252, 0.92)')
    grad.addColorStop(0.35, 'rgba(34, 211, 238, 0.45)')
    grad.addColorStop(0.75, 'rgba(14, 165, 233, 0.12)')
    grad.addColorStop(1, 'rgba(14, 165, 233, 0.0)')
    ctx.fillStyle = grad
    ctx.fillRect(0, 0, 128, 128)
  }
  const tex = new THREE.CanvasTexture(canvas)
  tex.colorSpace = THREE.SRGBColorSpace
  tex.needsUpdate = true
  return tex
}

function drawTickerHologramHeader(
  ctx: CanvasRenderingContext2D,
  state: HologramUpdateState,
  sentimentScore: number,
  pctChange: number | null,
  hasLiveData: boolean,
): void {
  const W = 512
  const H = 128
  ctx.clearRect(0, 0, W, H)

  const bg = ctx.createLinearGradient(20, 0, W - 20, 0)
  bg.addColorStop(0, 'rgba(6, 78, 128, 0.0)')
  bg.addColorStop(0.18, 'rgba(14, 116, 144, 0.42)')
  bg.addColorStop(0.82, 'rgba(14, 116, 144, 0.42)')
  bg.addColorStop(1, 'rgba(6, 78, 128, 0.0)')
  ctx.fillStyle = bg
  ctx.fillRect(14, 14, W - 28, H - 28)

  // Glowing emerald/crimson geometric badge icon on left
  ctx.save()
  ctx.translate(92, 62)
  const badgePositive = pctChange !== null ? pctChange >= 0 : sentimentScore >= 0
  ctx.fillStyle = badgePositive ? '#4ADE80' : '#F87171'
  ctx.shadowColor = badgePositive ? '#22C55E' : '#EF4444'
  ctx.shadowBlur = 16
  ctx.beginPath()
  ctx.roundRect(-24, -22, 48, 44, 10)
  ctx.fill()
  ctx.strokeStyle = '#FFFFFF'
  ctx.lineWidth = 4
  ctx.beginPath()
  if (badgePositive) {
    ctx.moveTo(-12, 8)
    ctx.lineTo(-2, -4)
    ctx.lineTo(5, 3)
    ctx.lineTo(14, -10)
  } else {
    ctx.moveTo(-12, -8)
    ctx.lineTo(-2, 4)
    ctx.lineTo(5, -2)
    ctx.lineTo(14, 10)
  }
  ctx.stroke()
  ctx.restore()

  const rawTicker = state.primaryTicker?.trim().toUpperCase()
  const titleText = rawTicker ? `$${rawTicker}` : 'TRINITY HUB'
  ctx.save()
  ctx.fillStyle = '#FFFFFF'
  ctx.shadowColor = '#38BDF8'
  ctx.shadowBlur = 18
  ctx.font = '900 42px "Segoe UI", Tahoma, sans-serif'
  ctx.textBaseline = 'middle'
  ctx.fillText(titleText, 134, 54)
  ctx.restore()

  if (pctChange !== null && hasLiveData) {
    const sign = pctChange >= 0 ? '+' : ''
    ctx.fillStyle = pctChange >= 0 ? '#86EFAC' : '#FCA5A5'
    ctx.font = '900 22px "Segoe UI", Tahoma, sans-serif'
    ctx.fillText(`${sign}${pctChange.toFixed(1)}%`, 365, 56)
  }

  const subText =
    state.finalRating && state.finalRating !== 'REVIEW'
      ? `CIO VERDICT: ${state.finalRating}${hasLiveData ? ' · LIVE OHLCV' : ''}`
      : hasLiveData
        ? 'LIVE 1D CANDLESTICKS · CLICK TO ZOOM'
        : sentimentScore > 0.15
          ? 'BULLISH MOMENTUM ▲'
          : sentimentScore < -0.15
            ? 'SKEPTIC CAUTION ▼'
            : 'LIVE ROUNDTABLE TELEMETRY'

  ctx.fillStyle = sentimentScore < -0.15 ? '#FCA5A5' : '#A5F3FC'
  ctx.font = 'bold 16px "Segoe UI", Tahoma, sans-serif'
  ctx.fillText(subText, 136, 94)
}

function computeSentimentScore(stancesBySpeaker: Record<string, RoundtableStanceType>): number {
  const vals = Object.values(stancesBySpeaker)
  if (vals.length === 0) return 0.2
  let sum = 0
  for (const s of vals) {
    if (s === 'BUY') sum += 1
    else if (s === 'OVERWEIGHT') sum += 0.65
    else if (s === 'UNDERWEIGHT') sum -= 0.65
    else if (s === 'SELL') sum -= 1
  }
  return Math.max(-1, Math.min(1, sum / vals.length))
}

function buildTrendGeometries(profile: CandleProfilePoint[]): {
  innerGeom: THREE.TubeGeometry
  outerGeom: THREE.TubeGeometry
} {
  const curvePoints = profile.map(
    (c, idx) =>
      new THREE.Vector3(
        c.x,
        c.baseY + (idx % 2 === 0 ? 0.03 : -0.02),
        Math.sin(idx * 0.55) * 0.04,
      ),
  )
  const curve = new THREE.CatmullRomCurve3(curvePoints)
  return {
    innerGeom: new THREE.TubeGeometry(curve, 44, 0.022, 8, false),
    outerGeom: new THREE.TubeGeometry(curve, 44, 0.052, 8, false),
  }
}

/**
 * Builds the centerpiece 3D Volumetric Hologram Projector hovering above the circular Holo-Table:
 * - Supports live OHLCV stock candlesticks & dynamic moving-average trendline ribbon
 * - Reactive 3D Bull (Green Up) & Bear (Red Down) arrows
 * - Fake additive glow halos + expanding emitter scan rings
 * - GPU InstancedMesh for 36 rising holographic data particles (1 draw call)
 */
export function buildCentralHologramProjector(): RobloxHologramHandle {
  const group = new THREE.Group()
  group.name = 'central_holo_projector'
  group.userData = { holoFocus: true }

  const chartStage = new THREE.Group()
  chartStage.rotation.y = 0.46
  chartStage.userData = { holoFocus: true }
  group.add(chartStage)

  // Invisible hit-volume cylinder so raycasting click anywhere on Hologram triggers council zoom cleanly
  const hitCylinder = new THREE.Mesh(
    new THREE.CylinderGeometry(1.2, 1.2, 2.2, 12),
    new THREE.MeshBasicMaterial({ visible: false }),
  )
  hitCylinder.position.y = 1.65
  hitCylinder.userData = { holoFocus: true }
  group.add(hitCylinder)

  // 1. Additive Cyan Bloom Halos (emitter well disc glow & header backdrop glow)
  const haloTex = createHoloHaloTexture()
  const wellHaloMat = new THREE.MeshBasicMaterial({
    map: haloTex,
    transparent: true,
    opacity: 0.72,
    blending: THREE.AdditiveBlending,
    depthWrite: false,
    side: THREE.DoubleSide,
  })
  const wellHaloMesh = new THREE.Mesh(new THREE.PlaneGeometry(2.35, 2.35), wellHaloMat)
  wellHaloMesh.rotation.x = -Math.PI / 2
  wellHaloMesh.position.y = 0.645
  group.add(wellHaloMesh)

  const headerHaloMesh = new THREE.Mesh(
    new THREE.PlaneGeometry(2.55, 0.85),
    new THREE.MeshBasicMaterial({
      map: haloTex,
      transparent: true,
      opacity: 0.44,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
      side: THREE.DoubleSide,
    }),
  )
  headerHaloMesh.position.set(0, 2.40, 0.01)
  chartStage.add(headerHaloMesh)

  // 2. Translucent Cyan Projection Cone Aura
  const auraMat = new THREE.MeshBasicMaterial({
    color: '#22D3EE',
    transparent: true,
    opacity: 0.095,
    blending: THREE.AdditiveBlending,
    side: THREE.DoubleSide,
    depthWrite: false,
  })
  const auraCone = new THREE.Mesh(
    new THREE.CylinderGeometry(1.26, 0.94, 1.68, 36, 1, true),
    auraMat,
  )
  auraCone.position.y = 1.45
  group.add(auraCone)

  // 3. Twin Expanding Horizontal Scan Rings inside Emitter Well
  const scanRings: Array<{ mesh: THREE.Mesh; phase: number }> = []
  for (let i = 0; i < 2; i += 1) {
    const ringMat = new THREE.MeshBasicMaterial({
      color: '#67E8F9',
      transparent: true,
      opacity: 0.58,
      blending: THREE.AdditiveBlending,
      side: THREE.DoubleSide,
      depthWrite: false,
    })
    const ring = new THREE.Mesh(new THREE.RingGeometry(0.92, 0.98, 40), ringMat)
    ring.rotation.x = -Math.PI / 2
    ring.position.y = 0.655
    group.add(ring)
    scanRings.push({ mesh: ring, phase: i * 0.5 })
  }

  // 4. Holographic Trendline Tube & Outer Glow Tube
  let currentProfile = buildDefaultProfile()
  const initialTubes = buildTrendGeometries(currentProfile)
  const trendTube = new THREE.Mesh(
    initialTubes.innerGeom,
    new THREE.MeshBasicMaterial({
      color: '#E0F2FE',
      transparent: true,
      opacity: 0.95,
    }),
  )
  const trendGlowTube = new THREE.Mesh(
    initialTubes.outerGeom,
    new THREE.MeshBasicMaterial({
      color: '#22D3EE',
      transparent: true,
      opacity: 0.36,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    }),
  )
  chartStage.add(trendGlowTube, trendTube)

  // 5. 13 3D Volumetric Candlesticks sharing base BoxGeometry
  const candles: CandleRig[] = []
  const sharedBodyGeom = new THREE.BoxGeometry(0.082, 1, 0.082)
  const sharedWickGeom = new THREE.BoxGeometry(0.018, 1, 0.018)
  const bullColor = new THREE.Color('#22C55E')
  const bullEmissive = new THREE.Color('#10B981')
  const bullWickColor = new THREE.Color('#86EFAC')
  const bearColor = new THREE.Color('#EF4444')
  const bearEmissive = new THREE.Color('#DC2626')
  const bearWickColor = new THREE.Color('#FCA5A5')

  currentProfile.forEach((seed, idx) => {
    const cGroup = new THREE.Group()
    cGroup.position.set(seed.x, 0, idx % 2 === 0 ? 0.03 : -0.03)

    const bodyMat = new THREE.MeshStandardMaterial({
      color: seed.bull ? bullColor : bearColor,
      emissive: seed.bull ? bullEmissive : bearEmissive,
      emissiveIntensity: 0.88,
      transparent: true,
      opacity: 0.88,
      roughness: 0.22,
      metalness: 0.1,
    })
    const wickMat = new THREE.MeshBasicMaterial({
      color: seed.bull ? bullWickColor : bearWickColor,
      transparent: true,
      opacity: 0.84,
    })

    const body = new THREE.Mesh(sharedBodyGeom, bodyMat)
    body.scale.y = seed.height
    body.position.y = seed.baseY

    const wick = new THREE.Mesh(sharedWickGeom, wickMat)
    wick.scale.y = seed.wickHeight
    wick.position.y = seed.wickCenterY

    cGroup.add(wick, body)
    chartStage.add(cGroup)

    candles.push({
      group: cGroup,
      body,
      wick,
      baseY: seed.baseY,
      baseHeight: seed.height,
      wickHeight: seed.wickHeight,
      wickCenterY: seed.wickCenterY,
      isBull: seed.bull,
      phaseOffset: idx * 0.52,
    })
  })

  // 6. Giant 3D Bullish Green (Up) & Bearish Red (Down) Arrows
  const upArrowGeom = create3DArrowGeometry(true)
  const downArrowGeom = create3DArrowGeometry(false)

  const bullArrowMat = new THREE.MeshStandardMaterial({
    color: '#4ADE80',
    emissive: '#16A34A',
    emissiveIntensity: 0.92,
    transparent: true,
    opacity: 0.9,
    roughness: 0.2,
  })
  const bullArrow = new THREE.Mesh(upArrowGeom, bullArrowMat)
  bullArrow.position.set(-0.96, 1.72, 0.10)
  bullArrow.scale.setScalar(1.08)

  const bearArrowMat = new THREE.MeshStandardMaterial({
    color: '#F87171',
    emissive: '#DC2626',
    emissiveIntensity: 0.86,
    transparent: true,
    opacity: 0.88,
    roughness: 0.2,
  })
  const bearArrow = new THREE.Mesh(downArrowGeom, bearArrowMat)
  bearArrow.position.set(0.96, 1.48, -0.05)
  bearArrow.scale.setScalar(1.0)

  chartStage.add(bullArrow, bearArrow)

  // 7. Floating 3D Ticker & Rating Canvas Billboard above chart
  const headerCanvas = document.createElement('canvas')
  headerCanvas.width = 512
  headerCanvas.height = 128
  const headerTex = new THREE.CanvasTexture(headerCanvas)
  headerTex.colorSpace = THREE.SRGBColorSpace
  const headerMesh = new THREE.Mesh(
    new THREE.PlaneGeometry(2.22, 0.55),
    new THREE.MeshBasicMaterial({
      map: headerTex,
      transparent: true,
      depthWrite: false,
      side: THREE.DoubleSide,
    }),
  )
  headerMesh.position.set(0, 2.40, 0.04)
  chartStage.add(headerMesh)

  // 8. 36 Rising Holographic Data Voxels using a single GPU InstancedMesh (1 draw call)
  const particleGeom = new THREE.BoxGeometry(0.036, 0.036, 0.036)
  const particleMat = new THREE.MeshBasicMaterial({
    color: '#FFFFFF',
    transparent: true,
    opacity: 0.72,
    blending: THREE.AdditiveBlending,
    depthWrite: false,
  })
  const particleInstanced = new THREE.InstancedMesh(particleGeom, particleMat, PARTICLE_COUNT)
  particleInstanced.instanceMatrix.setUsage(THREE.DynamicDrawUsage)
  const particleSeeds: ParticleSeed[] = []
  const tmpColor = new THREE.Color()
  const tmpDummy = new THREE.Object3D()

  for (let i = 0; i < PARTICLE_COUNT; i += 1) {
    const colorHex = i % 5 === 0 ? '#86EFAC' : i % 7 === 0 ? '#FFFFFF' : '#67E8F9'
    tmpColor.set(colorHex)
    particleInstanced.setColorAt(i, tmpColor)
    particleSeeds.push({
      radius: 0.16 + ((i * 0.23) % 0.88),
      angle: (i / PARTICLE_COUNT) * Math.PI * 2,
      speed: 0.30 + (i % 4) * 0.085,
      yOffset: (i * 0.17) % 1,
      spinSpeed: 1.5 + (i % 3) * 0.4,
      scaleBase: 0.7 + (i % 3) * 0.25,
    })
  }
  if (particleInstanced.instanceColor) particleInstanced.instanceColor.needsUpdate = true
  group.add(particleInstanced)

  let sentimentScore = 0.2
  let lastHeaderSig = ''
  let lastCandleSig = ''

  const updateState = (state: HologramUpdateState) => {
    sentimentScore = computeSentimentScore(state.stancesBySpeaker)
    const liveDataResult = buildProfileFromOhlcv(state.candles)
    currentProfile = liveDataResult.points

    // Re-sync candle geometry targets when OHLCV candle dataset changes
    const firstTs = state.candles?.[0]?.timestamp ?? 0
    const lastTs = state.candles?.[state.candles.length - 1]?.timestamp ?? 0
    const cSig = `${state.primaryTicker ?? ''}:${state.candles?.length ?? 0}:${firstTs}:${lastTs}`
    if (cSig !== lastCandleSig) {
      lastCandleSig = cSig
      currentProfile.forEach((pt, idx) => {
        const rig = candles[idx]
        if (!rig) return
        rig.baseY = pt.baseY
        rig.baseHeight = pt.height
        rig.wickHeight = pt.wickHeight
        rig.wickCenterY = pt.wickCenterY
        rig.isBull = pt.bull
        const bMat = rig.body.material as THREE.MeshStandardMaterial
        bMat.color.copy(pt.bull ? bullColor : bearColor)
        bMat.emissive.copy(pt.bull ? bullEmissive : bearEmissive)
        const wMat = rig.wick.material as THREE.MeshBasicMaterial
        wMat.color.copy(pt.bull ? bullWickColor : bearWickColor)
      })

      // Update trendline curve smoothly & dispose old geometries
      const nextTubes = buildTrendGeometries(currentProfile)
      const prevInner = trendTube.geometry
      const prevOuter = trendGlowTube.geometry
      trendTube.geometry = nextTubes.innerGeom
      trendGlowTube.geometry = nextTubes.outerGeom
      prevInner.dispose()
      prevOuter.dispose()
    }

    const sig = JSON.stringify({
      t: state.primaryTicker || '',
      r: state.finalRating || '',
      s: Math.round(sentimentScore * 10),
      c: cSig,
    })
    if (sig === lastHeaderSig) return
    lastHeaderSig = sig
    const ctx = headerCanvas.getContext('2d')
    if (ctx) {
      drawTickerHologramHeader(
        ctx,
        state,
        sentimentScore,
        liveDataResult.pctChange,
        liveDataResult.hasLiveData,
      )
      headerTex.needsUpdate = true
    }
  }

  updateState({ primaryTicker: null, stancesBySpeaker: {} })

  const animate = (
    elapsedSec: number,
    isCrossfire: boolean,
    isSandboxMathActive: boolean,
    reducedMotion: boolean,
    voiceEnergy = 0,
  ) => {
    const t = reducedMotion ? 0 : elapsedSec
    const clampedVoice = reducedMotion ? 0 : Math.max(0, Math.min(1, voiceEnergy))

    chartStage.rotation.y = 0.46 + Math.sin(t * 0.45) * 0.09
    auraMat.opacity =
      (isCrossfire
        ? 0.12 + Math.sin(t * 6.5) * 0.03
        : isSandboxMathActive
          ? 0.11 + Math.sin(t * 4.5) * 0.02
          : 0.088) +
      clampedVoice * 0.045
    auraMat.color.set(isCrossfire ? '#F59E0B' : isSandboxMathActive ? '#10B981' : '#22D3EE')
    wellHaloMat.opacity = 0.65 + Math.sin(t * 2.2) * 0.1 + clampedVoice * 0.18

    scanRings.forEach((sr) => {
      const cycle = (t * 0.45 + sr.phase) % 1
      const scale = 0.22 + cycle * 0.88
      sr.mesh.scale.set(scale, scale, 1)
      const mat = sr.mesh.material as THREE.MeshBasicMaterial
      mat.opacity = (1 - cycle) * 0.58
      mat.color.set(isCrossfire ? '#F87171' : '#67E8F9')
    })

    candles.forEach((c, idx) => {
      const wave = Math.sin(t * 2.4 + c.phaseOffset) * 0.12
      const trendShift = sentimentScore * ((idx - 6) * 0.018)
      const h = Math.max(0.06, c.baseHeight * (1 + wave))
      c.body.scale.y = h
      c.wick.scale.y = c.wickHeight * (1 + wave * 0.35)
      const bobY = Math.sin(t * 1.8 + c.phaseOffset) * 0.025 + trendShift
      c.body.position.y = c.baseY + bobY
      c.wick.position.y = c.wickCenterY + bobY

      const bMat = c.body.material as THREE.MeshStandardMaterial
      bMat.emissiveIntensity =
        0.74 + Math.sin(t * 3.1 + c.phaseOffset) * 0.2 + clampedVoice * 0.28
    })

    trendTube.position.y = Math.sin(t * 1.8) * 0.02 + sentimentScore * 0.03
    trendGlowTube.position.y = trendTube.position.y

    const bullBias = Math.max(0.78, 1.0 + sentimentScore * 0.24)
    const bearBias = Math.max(0.78, 1.0 - sentimentScore * 0.24 + (isCrossfire ? 0.22 : 0))

    bullArrow.position.y = 1.72 + Math.sin(t * 2.3) * 0.08 + sentimentScore * 0.10
    bullArrow.scale.setScalar(bullBias + (isCrossfire ? Math.sin(t * 6) * 0.05 : 0))
    bullArrowMat.emissiveIntensity = 0.82 + Math.max(0, sentimentScore) * 0.5 + clampedVoice * 0.2

    bearArrow.position.y = 1.48 + Math.cos(t * 2.5) * 0.08 - sentimentScore * 0.10
    bearArrow.scale.setScalar(bearBias + (isCrossfire ? Math.cos(t * 6) * 0.07 : 0))
    bearArrowMat.emissiveIntensity = isCrossfire
      ? 1.15 + Math.sin(t * 8) * 0.3
      : 0.78 + Math.max(0, -sentimentScore) * 0.5

    headerMesh.position.y = 2.40 + Math.sin(t * 1.6) * 0.032
    headerHaloMesh.position.y = headerMesh.position.y

    for (let i = 0; i < PARTICLE_COUNT; i += 1) {
      const p = particleSeeds[i]!
      const progress = (t * p.speed + p.yOffset) % 1
      const y = 0.68 + progress * 1.68
      const spinAngle = p.angle + t * 0.45
      const fade = Math.sin(progress * Math.PI)
      tmpDummy.position.set(Math.cos(spinAngle) * p.radius, y, Math.sin(spinAngle) * p.radius)
      tmpDummy.rotation.set(spinAngle * p.spinSpeed, spinAngle * 2, 0)
      tmpDummy.scale.setScalar(Math.max(0.05, fade * p.scaleBase))
      tmpDummy.updateMatrix()
      particleInstanced.setMatrixAt(i, tmpDummy.matrix)
    }
    particleInstanced.instanceMatrix.needsUpdate = true
  }

  const dispose = () => {
    haloTex.dispose()
    headerTex.dispose()
    group.traverse((obj) => {
      const mesh = obj as THREE.Mesh
      if (mesh.geometry) mesh.geometry.dispose()
      const m = mesh.material
      if (Array.isArray(m)) m.forEach((x) => x.dispose())
      else if (m) m.dispose()
    })
  }

  return {
    group,
    updateState,
    animate,
    dispose,
  }
}
