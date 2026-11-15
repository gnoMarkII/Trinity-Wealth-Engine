import * as THREE from 'three'
import type {
  DebatePillarCoverageDTO,
  OHLCVCandleDTO,
  RoundtableStanceType,
} from '../../api/types'
import type { SpeakerVisualMeta } from './personas'
import {
  ROOM_LAYOUT,
  WORKSTATION_SLOTS,
  getCouncilSeatTransform,
} from './roundtableChoreography'
import { buildCentralHologramProjector, type RobloxHologramHandle } from './roundtableHologram'

export type LightingThemeMode = 'twilight' | 'day'

export interface WallboardState {
  primaryTicker?: string | null
  portfolioLabel?: string
  finalRating?: RoundtableStanceType | string | null
  activeStageLabel?: string
  metas: SpeakerVisualMeta[]
  stancesBySpeaker: Record<string, RoundtableStanceType>
  pillarCoverage?: DebatePillarCoverageDTO | null
  hasSandboxMathActive: boolean
  candles?: OHLCVCandleDTO[] | null
}

export interface RobloxRoomHandle {
  group: THREE.Group
  ambientLight: THREE.AmbientLight
  hemiLight: THREE.HemisphereLight
  keyLight: THREE.DirectionalLight
  rimLight: THREE.PointLight
  floorMesh: THREE.Mesh
  carpetMesh: THREE.Mesh
  carpetBorderMesh: THREE.Mesh
  duelRingMesh: THREE.Mesh
  wallMeshes: THREE.Mesh[]
  skylineGroup: THREE.Group
  quantLedMeshes: THREE.Mesh[]
  monitorGlowMeshes: THREE.Mesh[]
  hologram: RobloxHologramHandle
  applyLightingTheme: (mode: LightingThemeMode) => void
  updateWallboards: (state: WallboardState) => void
  animateRoomEffects: (
    elapsedSec: number,
    isCrossfire: boolean,
    isSandboxMathActive: boolean,
    reducedMotion: boolean,
    voiceEnergy?: number,
  ) => void
  dispose: () => void
}

/**
 * Generates a lightweight procedural equirectangular PMREM reflection cubemap
 * so PBR metallic consoles, grid floor, and glass skyscrapers catch crisp highlights.
 */
export function createStudioEnvironmentMap(
  renderer: THREE.WebGLRenderer,
  mode: LightingThemeMode,
): { renderTarget: THREE.WebGLRenderTarget; texture: THREE.Texture } | null {
  try {
    const canvas = document.createElement('canvas')
    canvas.width = 128
    canvas.height = 64
    const ctx = canvas.getContext('2d')
    if (!ctx) return null

    const grad = ctx.createLinearGradient(0, 0, 0, 64)
    if (mode === 'day') {
      grad.addColorStop(0, '#38BDF8')
      grad.addColorStop(0.42, '#BAE6FD')
      grad.addColorStop(0.52, '#E0F2FE')
      grad.addColorStop(0.60, '#334155')
      grad.addColorStop(1, '#1E293B')
    } else {
      grad.addColorStop(0, '#0F172A')
      grad.addColorStop(0.42, '#1E3A5F')
      grad.addColorStop(0.52, '#22D3EE')
      grad.addColorStop(0.62, '#0B1A2A')
      grad.addColorStop(1, '#020617')
    }
    ctx.fillStyle = grad
    ctx.fillRect(0, 0, 128, 64)

    // Bright cyan cove reflection band & studio key light hotspot
    ctx.fillStyle = mode === 'day' ? 'rgba(255, 251, 235, 0.85)' : 'rgba(103, 232, 249, 0.7)'
    ctx.beginPath()
    ctx.arc(92, 18, 10, 0, Math.PI * 2)
    ctx.fill()

    const equirect = new THREE.CanvasTexture(canvas)
    equirect.mapping = THREE.EquirectangularReflectionMapping
    equirect.colorSpace = THREE.SRGBColorSpace
    equirect.needsUpdate = true

    const pmrem = new THREE.PMREMGenerator(renderer)
    pmrem.compileEquirectangularShader()
    const renderTarget = pmrem.fromEquirectangular(equirect)
    equirect.dispose()
    pmrem.dispose()
    return { renderTarget, texture: renderTarget.texture }
  } catch {
    return null
  }
}

function createSkyDomeShaderMaterial(): THREE.ShaderMaterial {
  return new THREE.ShaderMaterial({
    side: THREE.BackSide,
    depthWrite: false,
    fog: false,
    uniforms: {
      uZenithColor: { value: new THREE.Color('#0EA5E9') },
      uHorizonColor: { value: new THREE.Color('#E0F2FE') },
      uBottomColor: { value: new THREE.Color('#64748B') },
      uSunDir: { value: new THREE.Vector3(-0.62, 0.34, -0.71).normalize() },
      uSunIntensity: { value: 1.0 },
    },
    vertexShader: `
      varying vec3 vWorldPos;
      void main() {
        vec4 wp = modelMatrix * vec4(position, 1.0);
        vWorldPos = wp.xyz;
        gl_Position = projectionMatrix * viewMatrix * wp;
      }
    `,
    fragmentShader: `
      uniform vec3 uZenithColor;
      uniform vec3 uHorizonColor;
      uniform vec3 uBottomColor;
      uniform vec3 uSunDir;
      uniform float uSunIntensity;
      varying vec3 vWorldPos;
      void main() {
        vec3 dir = normalize(vWorldPos);
        float h = dir.y;
        vec3 col = h >= 0.0
          ? mix(uHorizonColor, uZenithColor, pow(clamp(h, 0.0, 1.0), 0.52))
          : mix(uHorizonColor, uBottomColor, pow(clamp(-h * 2.5, 0.0, 1.0), 0.8));

        // Fresh crisp morning solar corona + warm horizon bloom (daytime only)
        if (uSunIntensity > 0.01) {
          float sunDot = max(0.0, dot(dir, normalize(uSunDir)));
          float halo = pow(sunDot, 10.0) * 0.42 + pow(sunDot, 64.0) * 0.55;
          float core = smoothstep(0.9965, 0.9993, sunDot) * 0.85;
          vec3 morningGlow = vec3(1.0, 0.97, 0.84);
          vec3 coreWhite = vec3(1.0, 0.995, 0.94);
          col += morningGlow * halo * uSunIntensity;
          col = mix(col, coreWhite, clamp(core * uSunIntensity, 0.0, 1.0));
        }

        gl_FragColor = vec4(col, 1.0);
      }
    `,
  })
}

function createSunbeamShaderMaterial(): THREE.ShaderMaterial {
  return new THREE.ShaderMaterial({
    transparent: true,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
    side: THREE.DoubleSide,
    uniforms: {
      uTime: { value: 0 },
      uCloudFactor: { value: 1.0 },
      uColor: { value: new THREE.Color('#FFF7D6') },
    },
    vertexShader: `
      varying vec2 vUv;
      varying vec3 vWorldNormal;
      varying vec3 vWorldPos;
      void main() {
        vUv = uv;
        vWorldNormal = normalize(mat3(modelMatrix) * normal);
        vec4 wp = modelMatrix * vec4(position, 1.0);
        vWorldPos = wp.xyz;
        gl_Position = projectionMatrix * viewMatrix * wp;
      }
    `,
    fragmentShader: `
      uniform float uTime;
      uniform float uCloudFactor;
      uniform vec3 uColor;
      varying vec2 vUv;
      varying vec3 vWorldNormal;
      varying vec3 vWorldPos;
      void main() {
        vec3 viewDir = normalize(cameraPosition - vWorldPos);
        float rimFade = pow(abs(dot(normalize(vWorldNormal), viewDir)), 1.55);
        // Bright near upper window source (vUv.y -> 1), soft atmospheric tail near floor (vUv.y -> 0)
        float lenFade = smoothstep(0.0, 0.32, vUv.y) * smoothstep(1.0, 0.72, vUv.y);
        float shimmer = 0.88 + 0.12 * sin(vUv.x * 18.84 + uTime * 0.65);
        float alpha = rimFade * lenFade * shimmer * 0.115 * uCloudFactor;
        gl_FragColor = vec4(uColor, alpha);
      }
    `,
  })
}

function createSunFloorPoolTexture(): THREE.CanvasTexture {
  const canvas = document.createElement('canvas')
  canvas.width = 128
  canvas.height = 128
  const ctx = canvas.getContext('2d')
  if (ctx) {
    const grad = ctx.createRadialGradient(64, 64, 8, 64, 64, 62)
    grad.addColorStop(0, 'rgba(255, 250, 224, 0.34)')
    grad.addColorStop(0.55, 'rgba(254, 243, 199, 0.14)')
    grad.addColorStop(1, 'rgba(254, 243, 199, 0.0)')
    ctx.fillStyle = grad
    ctx.fillRect(0, 0, 128, 128)
  }
  const tex = new THREE.CanvasTexture(canvas)
  tex.colorSpace = THREE.SRGBColorSpace
  tex.needsUpdate = true
  return tex
}

function createPedestalShadowTexture(): THREE.CanvasTexture {
  const canvas = document.createElement('canvas')
  canvas.width = 128
  canvas.height = 128
  const ctx = canvas.getContext('2d')
  if (ctx) {
    const grad = ctx.createRadialGradient(64, 64, 16, 64, 64, 62)
    grad.addColorStop(0, 'rgba(2, 6, 23, 0.52)')
    grad.addColorStop(0.65, 'rgba(2, 6, 23, 0.24)')
    grad.addColorStop(1, 'rgba(2, 6, 23, 0.0)')
    ctx.fillStyle = grad
    ctx.fillRect(0, 0, 128, 128)
  }
  const tex = new THREE.CanvasTexture(canvas)
  tex.colorSpace = THREE.SRGBColorSpace
  tex.needsUpdate = true
  return tex
}

interface FloorTextureSet {
  map: THREE.CanvasTexture
  roughnessMap: THREE.CanvasTexture
  bumpMap: THREE.CanvasTexture
  emissiveMap: THREE.CanvasTexture
}

function configureRepeatedTexture(
  canvas: HTMLCanvasElement,
  repeatX: number,
  repeatY: number,
  anisotropy: number,
  srgb = true,
): THREE.CanvasTexture {
  const tex = new THREE.CanvasTexture(canvas)
  tex.colorSpace = srgb ? THREE.SRGBColorSpace : THREE.NoColorSpace
  tex.wrapS = THREE.RepeatWrapping
  tex.wrapT = THREE.RepeatWrapping
  tex.repeat.set(repeatX, repeatY)
  tex.anisotropy = Math.max(1, Math.min(16, anisotropy))
  tex.needsUpdate = true
  return tex
}

/**
 * Builds an Architectural Glass-Tile PBR multi-map floor texture pack
 * (Albedo + Roughness contrast + Seam Bump + Emissive LED fiber channels).
 */
function createArchitecturalFloorTextureSet(
  mode: LightingThemeMode,
  maxAnisotropy: number,
): FloorTextureSet {
  const S = 512
  const half = S / 2
  const isDay = mode === 'day'

  const albedoCanvas = document.createElement('canvas')
  const roughCanvas = document.createElement('canvas')
  const bumpCanvas = document.createElement('canvas')
  const emissiveCanvas = document.createElement('canvas')
  for (const c of [albedoCanvas, roughCanvas, bumpCanvas, emissiveCanvas]) {
    c.width = S
    c.height = S
  }

  const aCtx = albedoCanvas.getContext('2d')
  const rCtx = roughCanvas.getContext('2d')
  const bCtx = bumpCanvas.getContext('2d')
  const eCtx = emissiveCanvas.getContext('2d')

  if (aCtx && rCtx && bCtx && eCtx) {
    // 1. Albedo base slate-glass panels (4 large 256x256 tiles per repeat, subtle sheen gradient, NO studs)
    aCtx.fillStyle = isDay ? '#35516E' : '#0B1828'
    aCtx.fillRect(0, 0, S, S)

    rCtx.fillStyle = isDay ? '#262626' : '#1E1E1E' // Smooth mirror-polished architectural glass (~0.13-0.15 roughness)
    rCtx.fillRect(0, 0, S, S)

    bCtx.fillStyle = '#CFCFCF'
    bCtx.fillRect(0, 0, S, S)

    eCtx.fillStyle = '#000000'
    eCtx.fillRect(0, 0, S, S)

    for (let gx = 0; gx < 2; gx += 1) {
      for (let gy = 0; gy < 2; gy += 1) {
        const px = gx * half
        const py = gy * half
        const panelGrad = aCtx.createLinearGradient(px, py, px + half, py + half)
        if ((gx + gy) % 2 === 0) {
          panelGrad.addColorStop(0, isDay ? '#3B5978' : '#0E2034')
          panelGrad.addColorStop(1, isDay ? '#334E6A' : '#0A1626')
        } else {
          panelGrad.addColorStop(0, isDay ? '#365270' : '#0C1B2D')
          panelGrad.addColorStop(1, isDay ? '#3F6081' : '#11253B')
        }
        aCtx.fillStyle = panelGrad
        aCtx.fillRect(px + 4, py + 4, half - 8, half - 8)

        // Hairline brushed inner architectural inlay bevel
        aCtx.strokeStyle = isDay ? 'rgba(148, 197, 233, 0.20)' : 'rgba(56, 189, 248, 0.14)'
        aCtx.lineWidth = 1.5
        aCtx.strokeRect(px + 14, py + 14, half - 28, half - 28)

        rCtx.strokeStyle = '#484848'
        rCtx.lineWidth = 3
        rCtx.strokeRect(px + 14, py + 14, half - 28, half - 28)

        bCtx.strokeStyle = '#9E9E9E'
        bCtx.lineWidth = 2
        bCtx.strokeRect(px + 14, py + 14, half - 28, half - 28)
      }
    }

    // 2. Grout expansion joints & chamfer borders across tile edges (0, 256, 512 for seamless wrap)
    const seams = [0, half, S]
    aCtx.strokeStyle = isDay ? '#16293D' : '#030811'
    aCtx.lineWidth = 7
    rCtx.strokeStyle = '#D8D8D8' // Matte grout seams (roughness ~0.85)
    rCtx.lineWidth = 8
    bCtx.strokeStyle = '#262626' // Deep recessed bevel joint
    bCtx.lineWidth = 8

    for (const pos of seams) {
      for (const ctx of [aCtx, rCtx, bCtx]) {
        ctx.beginPath()
        ctx.moveTo(pos, 0)
        ctx.lineTo(pos, S)
        ctx.stroke()
        ctx.beginPath()
        ctx.moveTo(0, pos)
        ctx.lineTo(S, pos)
        ctx.stroke()
      }
    }

    // Fine bright chamfer rim next to dark seams in albedo
    aCtx.strokeStyle = isDay ? 'rgba(125, 211, 252, 0.34)' : 'rgba(34, 211, 238, 0.24)'
    aCtx.lineWidth = 2
    for (const pos of seams) {
      aCtx.beginPath()
      aCtx.moveTo(pos + 4, 0)
      aCtx.lineTo(pos + 4, S)
      aCtx.moveTo(pos - 4, 0)
      aCtx.lineTo(pos - 4, S)
      aCtx.moveTo(0, pos + 4)
      aCtx.lineTo(S, pos + 4)
      aCtx.moveTo(0, pos - 4)
      aCtx.lineTo(S, pos - 4)
      aCtx.stroke()
    }

    // 3. Dedicated Emissive Fiber-Optic Cyan Grid Channels + Corner Nodes
    for (const pos of seams) {
      eCtx.strokeStyle = isDay ? 'rgba(56, 189, 248, 0.30)' : 'rgba(34, 211, 238, 0.42)'
      eCtx.lineWidth = 10
      eCtx.beginPath()
      eCtx.moveTo(pos, 0)
      eCtx.lineTo(pos, S)
      eCtx.moveTo(0, pos)
      eCtx.lineTo(S, pos)
      eCtx.stroke()

      eCtx.strokeStyle = isDay ? '#7DD3FC' : '#67E8F9'
      eCtx.lineWidth = 2.4
      eCtx.beginPath()
      eCtx.moveTo(pos, 0)
      eCtx.lineTo(pos, S)
      eCtx.moveTo(0, pos)
      eCtx.lineTo(S, pos)
      eCtx.stroke()
    }

    // Crisp luminous corner intersection pips
    eCtx.fillStyle = '#E0F2FE'
    for (const sx of seams) {
      for (const sy of seams) {
        eCtx.beginPath()
        eCtx.arc(sx, sy, 3.6, 0, Math.PI * 2)
        eCtx.fill()
      }
    }
  }

  return {
    map: configureRepeatedTexture(albedoCanvas, 4.5, 3.2, maxAnisotropy, true),
    roughnessMap: configureRepeatedTexture(roughCanvas, 4.5, 3.2, maxAnisotropy, false),
    bumpMap: configureRepeatedTexture(bumpCanvas, 4.5, 3.2, maxAnisotropy, false),
    emissiveMap: configureRepeatedTexture(emissiveCanvas, 4.5, 3.2, maxAnisotropy, true),
  }
}

function createCouncilPlazaDecalTexture(anisotropy: number): THREE.CanvasTexture {
  const S = 512
  const canvas = document.createElement('canvas')
  canvas.width = S
  canvas.height = S
  const ctx = canvas.getContext('2d')
  if (ctx) {
    const cx = S / 2
    const cy = S / 2
    ctx.clearRect(0, 0, S, S)

    // Soft glass refraction disc wash
    const rad = ctx.createRadialGradient(cx, cy, 40, cx, cy, 248)
    rad.addColorStop(0, 'rgba(34, 211, 238, 0.16)')
    rad.addColorStop(0.68, 'rgba(14, 116, 144, 0.10)')
    rad.addColorStop(1, 'rgba(14, 116, 144, 0.0)')
    ctx.fillStyle = rad
    ctx.beginPath()
    ctx.arc(cx, cy, 248, 0, Math.PI * 2)
    ctx.fill()

    // Precision architectural compass rings & degree ticks
    ctx.strokeStyle = 'rgba(103, 232, 249, 0.55)'
    ctx.lineWidth = 2
    ctx.beginPath()
    ctx.arc(cx, cy, 218, 0, Math.PI * 2)
    ctx.stroke()

    ctx.strokeStyle = 'rgba(125, 211, 252, 0.28)'
    ctx.lineWidth = 1.5
    ctx.setLineDash([12, 10])
    ctx.beginPath()
    ctx.arc(cx, cy, 178, 0, Math.PI * 2)
    ctx.stroke()
    ctx.setLineDash([])

    // 48 perimeter financial dial ticks
    for (let i = 0; i < 48; i += 1) {
      const a = (i / 48) * Math.PI * 2
      const major = i % 4 === 0
      const rIn = major ? 203 : 210
      const rOut = 218
      ctx.strokeStyle = major ? 'rgba(165, 243, 252, 0.80)' : 'rgba(103, 232, 249, 0.42)'
      ctx.lineWidth = major ? 2.5 : 1.3
      ctx.beginPath()
      ctx.moveTo(cx + Math.cos(a) * rIn, cy + Math.sin(a) * rIn)
      ctx.lineTo(cx + Math.cos(a) * rOut, cy + Math.sin(a) * rOut)
      ctx.stroke()
    }
  }
  const tex = new THREE.CanvasTexture(canvas)
  tex.colorSpace = THREE.SRGBColorSpace
  tex.anisotropy = Math.max(1, Math.min(16, anisotropy))
  tex.needsUpdate = true
  return tex
}

function createExecutiveFloorTexture(anisotropy: number): THREE.CanvasTexture {
  const S = 512
  const canvas = document.createElement('canvas')
  canvas.width = S
  canvas.height = S
  const ctx = canvas.getContext('2d')
  if (ctx) {
    ctx.fillStyle = '#1E2533'
    ctx.fillRect(0, 0, S, S)
    const rows = 8
    const rowH = S / rows
    for (let r = 0; r < rows; r += 1) {
      const offset = (r % 2) * 128
      for (let col = -1; col < 3; col += 1) {
        const px = col * 256 + offset
        const py = r * rowH
        const tone = (r * 3 + col) % 3 === 0 ? '#273142' : (r + col) % 2 === 0 ? '#222B3B' : '#1D2432'
        ctx.fillStyle = tone
        ctx.fillRect(px + 1, py + 1, 254, rowH - 2)
        // Subtle champagne inlay seam highlight
        ctx.strokeStyle = 'rgba(251, 191, 36, 0.12)'
        ctx.lineWidth = 1
        ctx.strokeRect(px + 2, py + 2, 252, rowH - 4)
      }
    }
  }
  return configureRepeatedTexture(canvas, 2.5, 2, anisotropy, true)
}

function createDuelArenaSurfaceTexture(anisotropy: number): THREE.CanvasTexture {
  const S = 256
  const canvas = document.createElement('canvas')
  canvas.width = S
  canvas.height = S
  const ctx = canvas.getContext('2d')
  if (ctx) {
    const cx = S / 2
    const cy = S / 2
    ctx.fillStyle = '#0D2136'
    ctx.fillRect(0, 0, S, S)

    ctx.strokeStyle = 'rgba(56, 189, 248, 0.42)'
    ctx.lineWidth = 3
    ctx.beginPath()
    ctx.arc(cx, cy, 98, 0, Math.PI * 2)
    ctx.stroke()

    ctx.strokeStyle = 'rgba(125, 211, 252, 0.25)'
    ctx.lineWidth = 2
    ctx.beginPath()
    ctx.moveTo(cx - 88, cy)
    ctx.lineTo(cx + 88, cy)
    ctx.moveTo(cx, cy - 88)
    ctx.lineTo(cx, cy + 88)
    ctx.stroke()

    ctx.fillStyle = 'rgba(56, 189, 248, 0.24)'
    ctx.beginPath()
    ctx.arc(cx, cy, 22, 0, Math.PI * 2)
    ctx.fill()
  }
  const tex = new THREE.CanvasTexture(canvas)
  tex.colorSpace = THREE.SRGBColorSpace
  tex.anisotropy = Math.max(1, Math.min(16, anisotropy))
  tex.needsUpdate = true
  return tex
}

function createLoungeRugTexture(anisotropy: number): THREE.CanvasTexture {
  const S = 256
  const canvas = document.createElement('canvas')
  canvas.width = S
  canvas.height = S
  const ctx = canvas.getContext('2d')
  if (ctx) {
    ctx.fillStyle = '#263447'
    ctx.fillRect(0, 0, S, S)
    // Woven architectural thread micro-texture
    ctx.fillStyle = 'rgba(148, 163, 184, 0.08)'
    for (let y = 0; y < S; y += 6) {
      ctx.fillRect(0, y, S, 2)
    }
    for (let x = 0; x < S; x += 8) {
      ctx.fillRect(x, 0, 2, S)
    }
    ctx.strokeStyle = 'rgba(148, 163, 184, 0.35)'
    ctx.lineWidth = 6
    ctx.strokeRect(10, 10, S - 20, S - 20)
    ctx.strokeStyle = 'rgba(56, 189, 248, 0.28)'
    ctx.lineWidth = 2
    ctx.strokeRect(18, 18, S - 36, S - 36)
  }
  const tex = new THREE.CanvasTexture(canvas)
  tex.colorSpace = THREE.SRGBColorSpace
  tex.anisotropy = Math.max(1, Math.min(16, anisotropy))
  tex.needsUpdate = true
  return tex
}

function createWindowWashTexture(): THREE.CanvasTexture {
  const canvas = document.createElement('canvas')
  canvas.width = 64
  canvas.height = 128
  const ctx = canvas.getContext('2d')
  if (ctx) {
    const grad = ctx.createLinearGradient(0, 0, 0, 128)
    grad.addColorStop(0, 'rgba(186, 230, 253, 0.30)')
    grad.addColorStop(0.45, 'rgba(125, 211, 252, 0.12)')
    grad.addColorStop(1, 'rgba(56, 189, 248, 0.0)')
    ctx.fillStyle = grad
    ctx.fillRect(0, 0, 64, 128)
  }
  const tex = new THREE.CanvasTexture(canvas)
  tex.colorSpace = THREE.SRGBColorSpace
  tex.needsUpdate = true
  return tex
}

function createSkyscraperFacadeTexture(isDay: boolean): THREE.CanvasTexture {
  const canvas = document.createElement('canvas')
  canvas.width = 128
  canvas.height = 256
  const ctx = canvas.getContext('2d')
  if (ctx) {
    ctx.fillStyle = isDay ? '#6B8CAE' : '#0F172A'
    ctx.fillRect(0, 0, 128, 256)

    const cols = 4
    const rows = 12
    const padX = 6
    const padY = 5
    const w = (128 - (cols + 1) * padX) / cols
    const h = (256 - (rows + 1) * padY) / rows

    for (let r = 0; r < rows; r += 1) {
      for (let c = 0; c < cols; c += 1) {
        const x = padX + c * (w + padX)
        const y = padY + r * (h + padY)
        if (isDay) {
          const sunReflect = (r + c) % 3 === 0 ? '#BAE6FD' : '#93C5FD'
          ctx.fillStyle = sunReflect
        } else {
          const lit = (r * 7 + c * 11) % 5
          ctx.fillStyle = lit === 0 ? '#FBBF24' : lit <= 2 ? '#38BDF8' : '#1E293B'
        }
        ctx.fillRect(x, y, w, h)
      }
    }
  }
  const tex = new THREE.CanvasTexture(canvas)
  tex.colorSpace = THREE.SRGBColorSpace
  tex.needsUpdate = true
  return tex
}

function createDualMonitorTexture(): THREE.CanvasTexture {
  const canvas = document.createElement('canvas')
  canvas.width = 256
  canvas.height = 128
  const ctx = canvas.getContext('2d')
  if (ctx) {
    ctx.fillStyle = '#071321'
    ctx.fillRect(0, 0, 256, 128)

    // Subtle grid lines
    ctx.strokeStyle = 'rgba(56, 189, 248, 0.16)'
    ctx.lineWidth = 1
    for (let y = 24; y < 120; y += 24) {
      ctx.beginPath()
      ctx.moveTo(8, y)
      ctx.lineTo(248, y)
      ctx.stroke()
    }

    // Mini candlestick cluster + emerald/crimson highlights
    const bars = [
      { h: 28, bull: true },
      { h: 18, bull: false },
      { h: 36, bull: true },
      { h: 48, bull: true },
      { h: 24, bull: false },
      { h: 52, bull: true },
      { h: 68, bull: true },
    ]
    bars.forEach((b, i) => {
      const bx = 22 + i * 32
      ctx.fillStyle = b.bull ? '#22C55E' : '#EF4444'
      ctx.fillRect(bx, 108 - b.h, 14, b.h * 0.72)
    })

    // Cyan financial chart trend line
    ctx.strokeStyle = '#38BDF8'
    ctx.lineWidth = 3.5
    ctx.beginPath()
    ctx.moveTo(14, 96)
    ctx.lineTo(58, 72)
    ctx.lineTo(96, 80)
    ctx.lineTo(148, 40)
    ctx.lineTo(192, 52)
    ctx.lineTo(240, 22)
    ctx.stroke()
  }
  const tex = new THREE.CanvasTexture(canvas)
  tex.colorSpace = THREE.SRGBColorSpace
  tex.needsUpdate = true
  return tex
}

function stanceColorsForCanvas(stance?: string | null): { bg: string; fg: string } {
  switch (stance) {
    case 'BUY':
      return { bg: '#065F46', fg: '#6EE7B7' }
    case 'OVERWEIGHT':
      return { bg: '#115E59', fg: '#5EEAD4' }
    case 'HOLD':
      return { bg: '#78350F', fg: '#FCD34D' }
    case 'UNDERWEIGHT':
      return { bg: '#7C2D12', fg: '#FDBA74' }
    case 'SELL':
      return { bg: '#7F1D1D', fg: '#FCA5A5' }
    default:
      return { bg: '#1E293B', fg: '#94A3B8' }
  }
}

function drawMainWallCanvas(ctx: CanvasRenderingContext2D, state: WallboardState): void {
  const W = 1024
  const H = 448
  const grad = ctx.createLinearGradient(0, 0, W, H)
  grad.addColorStop(0, '#071422')
  grad.addColorStop(1, '#0F2A44')
  ctx.fillStyle = grad
  ctx.fillRect(0, 0, W, H)

  ctx.strokeStyle = 'rgba(34, 211, 238, 0.48)'
  ctx.lineWidth = 4
  ctx.strokeRect(10, 10, W - 20, H - 20)

  ctx.fillStyle = '#22D3EE'
  ctx.font = 'bold 24px "Segoe UI", Tahoma, sans-serif'
  ctx.fillText('TRINITY METROPOLITAN HUB · GURU COUNCIL', 42, 62)

  const tickerText = state.primaryTicker ? `$${state.primaryTicker.toUpperCase()}` : 'PORTFOLIO WAR ROOM'
  ctx.fillStyle = '#FFFFFF'
  ctx.font = '900 60px "Segoe UI", Tahoma, sans-serif'
  ctx.fillText(tickerText, 42, 144)

  const stageText = state.activeStageLabel || 'พร้อมเปิดวาระประชุม'
  ctx.fillStyle = 'rgba(14, 165, 233, 0.22)'
  ctx.fillRect(580, 36, 404, 48)
  ctx.strokeStyle = 'rgba(56, 189, 248, 0.6)'
  ctx.lineWidth = 2
  ctx.strokeRect(580, 36, 404, 48)
  ctx.fillStyle = '#E0F2FE'
  ctx.font = 'bold 22px "Segoe UI", Tahoma, sans-serif'
  ctx.fillText(stageText.slice(0, 32), 600, 68)

  if (state.portfolioLabel) {
    ctx.fillStyle = '#BAE6FD'
    ctx.font = '600 22px "Segoe UI", Tahoma, sans-serif'
    ctx.fillText(state.portfolioLabel, 580, 128)
  }

  ctx.fillStyle = 'rgba(34, 211, 238, 0.25)'
  ctx.fillRect(42, 178, W - 84, 2)

  const panelMetas = state.metas.slice(0, 6)
  const colW = (W - 84 - 5 * 14) / Math.max(1, panelMetas.length)
  panelMetas.forEach((m, idx) => {
    const x = 42 + idx * (colW + 14)
    const y = 204
    const h = 202
    const stance = state.stancesBySpeaker[m.speakerId]
    const sc = stanceColorsForCanvas(stance)

    ctx.fillStyle = m.enabled ? 'rgba(15, 23, 42, 0.84)' : 'rgba(15, 23, 42, 0.35)'
    ctx.fillRect(x, y, colW, h)
    ctx.strokeStyle = m.enabled ? m.accentColor : '#334155'
    ctx.lineWidth = 3
    ctx.strokeRect(x, y, colW, h)

    ctx.fillStyle = m.enabled ? m.accentColor : '#475569'
    ctx.fillRect(x, y, colW, 8)

    ctx.fillStyle = m.enabled ? '#F8FAFC' : '#64748B'
    ctx.font = 'bold 24px "Segoe UI", Tahoma, sans-serif'
    ctx.fillText(m.shortName, x + 14, y + 48)

    ctx.fillStyle = '#94A3B8'
    ctx.font = '600 16px "Segoe UI", Tahoma, sans-serif'
    ctx.fillText(m.roleLensLabel.slice(0, 18), x + 14, y + 78)

    ctx.fillStyle = sc.bg
    ctx.fillRect(x + 14, y + 128, colW - 28, 46)
    ctx.fillStyle = sc.fg
    ctx.font = '900 22px "Segoe UI", Tahoma, sans-serif'
    ctx.fillText(m.enabled ? stance || 'WAITING' : 'MUTED', x + 26, y + 158)
  })
}

function drawBoardWallCanvas(ctx: CanvasRenderingContext2D, state: WallboardState): void {
  const W = 640
  const H = 384
  ctx.fillStyle = '#071422'
  ctx.fillRect(0, 0, W, H)

  ctx.strokeStyle = 'rgba(34, 211, 238, 0.55)'
  ctx.lineWidth = 4
  ctx.strokeRect(8, 8, W - 16, H - 16)

  ctx.fillStyle = '#22D3EE'
  ctx.font = 'bold 20px "Segoe UI", Tahoma, sans-serif'
  ctx.fillText('LIVE MARKET PULSE & 5-PILLAR COVERAGE', 26, 42)

  // Left half: Stylized Mountain / Trend Area Chart (inspired by right display in reference)
  const chartX = 26
  const chartY = 62
  const chartW = 250
  const chartH = 212
  ctx.fillStyle = 'rgba(15, 23, 42, 0.85)'
  ctx.fillRect(chartX, chartY, chartW, chartH)
  ctx.strokeStyle = 'rgba(56, 189, 248, 0.35)'
  ctx.lineWidth = 1.5
  ctx.strokeRect(chartX, chartY, chartW, chartH)

  // Green rising trend area
  const pts = [
    [chartX + 12, chartY + 178],
    [chartX + 52, chartY + 145],
    [chartX + 92, chartY + 158],
    [chartX + 135, chartY + 102],
    [chartX + 178, chartY + 114],
    [chartX + 215, chartY + 52],
    [chartX + 240, chartY + 34],
  ] as const
  const areaGrad = ctx.createLinearGradient(0, chartY + 30, 0, chartY + chartH)
  areaGrad.addColorStop(0, 'rgba(34, 197, 94, 0.55)')
  areaGrad.addColorStop(1, 'rgba(16, 185, 129, 0.05)')
  ctx.fillStyle = areaGrad
  ctx.beginPath()
  ctx.moveTo(pts[0][0], chartY + chartH - 8)
  for (const [px, py] of pts) ctx.lineTo(px, py)
  ctx.lineTo(pts[pts.length - 1]![0], chartY + chartH - 8)
  ctx.closePath()
  ctx.fill()

  ctx.strokeStyle = '#4ADE80'
  ctx.lineWidth = 3.5
  ctx.beginPath()
  pts.forEach(([px, py], i) => (i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py)))
  ctx.stroke()

  // Contrasting amber/crimson resistance line
  ctx.strokeStyle = '#F87171'
  ctx.lineWidth = 2.5
  ctx.beginPath()
  ctx.moveTo(chartX + 12, chartY + 136)
  ctx.lineTo(chartX + 75, chartY + 118)
  ctx.lineTo(chartX + 145, chartY + 142)
  ctx.lineTo(chartX + 238, chartY + 86)
  ctx.stroke()

  ctx.fillStyle = '#BAE6FD'
  ctx.font = 'bold 15px "Segoe UI", Tahoma, sans-serif'
  ctx.fillText(
    state.primaryTicker ? `$${state.primaryTicker.toUpperCase()} TREND` : 'HUB INDEX',
    chartX + 12,
    chartY + 24,
  )

  // Right half: 5-Pillar Coverage Bars
  const defaultPillars = [
    { pillar: 'Valuation & MoS', is_challenged: false, raised_by: [] as string[] },
    { pillar: 'Quality & Moat', is_challenged: false, raised_by: [] as string[] },
    { pillar: 'Growth & Guidance', is_challenged: false, raised_by: [] as string[] },
    { pillar: 'Macro & Flow', is_challenged: false, raised_by: [] as string[] },
    { pillar: 'Risk & Kill-Switch', is_challenged: false, raised_by: [] as string[] },
  ]
  const pillars = state.pillarCoverage?.pillars?.length
    ? state.pillarCoverage.pillars.slice(0, 5)
    : defaultPillars

  pillars.forEach((p, idx) => {
    const px = 294
    const pw = W - 26 - px
    const py = 62 + idx * 43
    const isCovered = p.raised_by.length > 0

    ctx.fillStyle = 'rgba(30, 58, 95, 0.55)'
    ctx.fillRect(px, py, pw, 34)

    ctx.fillStyle = p.is_challenged ? '#10B981' : isCovered ? '#0EA5E9' : '#334155'
    ctx.fillRect(px, py, p.is_challenged ? pw : isCovered ? pw * 0.65 : 22, 34)

    ctx.fillStyle = '#F8FAFC'
    ctx.font = 'bold 15px "Segoe UI", Tahoma, sans-serif'
    ctx.fillText(p.pillar.slice(0, 19), px + 10, py + 22)

    ctx.fillStyle = '#E2E8F0'
    ctx.font = 'bold 13px "Segoe UI", Tahoma, sans-serif'
    const tag = p.is_challenged ? 'OK' : isCovered ? 'ACTIVE' : 'WAIT'
    ctx.fillText(tag, px + pw - 58, py + 22)
  })

  // Bottom Verdict Card
  const verdict = state.finalRating || 'REVIEW'
  const vc = stanceColorsForCanvas(verdict)
  ctx.fillStyle = vc.bg
  ctx.fillRect(26, 294, W - 52, 64)
  ctx.strokeStyle = vc.fg
  ctx.lineWidth = 3
  ctx.strokeRect(26, 294, W - 52, 64)

  ctx.fillStyle = '#E2E8F0'
  ctx.font = 'bold 20px "Segoe UI", Tahoma, sans-serif'
  ctx.fillText('FINAL CIO RATING:', 48, 333)
  ctx.fillStyle = vc.fg
  ctx.font = '900 30px "Segoe UI", Tahoma, sans-serif'
  ctx.fillText(String(verdict), 280, 337)
}

function buildOfficeChair(seatColor: string, accentHex?: string): THREE.Group {
  const g = new THREE.Group()
  const seatMat = new THREE.MeshStandardMaterial({ color: seatColor, roughness: 0.45, metalness: 0.15 })
  const baseMat = new THREE.MeshStandardMaterial({ color: '#0F172A', roughness: 0.45, metalness: 0.35 })

  const cushion = new THREE.Mesh(new THREE.BoxGeometry(0.58, 0.09, 0.54), seatMat)
  cushion.position.set(0, 0.42, 0)
  cushion.castShadow = true

  const backrest = new THREE.Mesh(new THREE.BoxGeometry(0.54, 0.38, 0.09), seatMat)
  backrest.position.set(0, 0.62, -0.24)
  backrest.castShadow = true

  const stem = new THREE.Mesh(new THREE.CylinderGeometry(0.05, 0.05, 0.36, 10), baseMat)
  stem.position.set(0, 0.21, 0)
  const crossA = new THREE.Mesh(new THREE.BoxGeometry(0.50, 0.045, 0.065), baseMat)
  crossA.position.set(0, 0.04, 0)
  const crossB = new THREE.Mesh(new THREE.BoxGeometry(0.065, 0.045, 0.50), baseMat)
  crossB.position.set(0, 0.04, 0)

  g.add(cushion, backrest, stem, crossA, crossB)

  if (accentHex) {
    const strip = new THREE.Mesh(
      new THREE.BoxGeometry(0.42, 0.045, 0.025),
      new THREE.MeshStandardMaterial({
        color: accentHex,
        emissive: accentHex,
        emissiveIntensity: 0.45,
      }),
    )
    strip.position.set(0, 0.75, -0.29)
    g.add(strip)
  }

  return g
}

function buildDeskProp(
  propType: 'safe' | 'diamond' | 'rocket' | 'globe' | 'radar',
  accent: string,
  spinningTargets: THREE.Object3D[],
): THREE.Group {
  const g = new THREE.Group()
  const mat = new THREE.MeshStandardMaterial({
    color: accent,
    emissive: accent,
    emissiveIntensity: 0.3,
    roughness: 0.25,
    metalness: 0.35,
  })
  const dark = new THREE.MeshStandardMaterial({ color: '#1E293B', roughness: 0.45 })

  switch (propType) {
    case 'safe': {
      const box = new THREE.Mesh(new THREE.BoxGeometry(0.20, 0.20, 0.18), dark)
      box.position.y = 0.10
      const dial = new THREE.Mesh(new THREE.CylinderGeometry(0.05, 0.05, 0.04, 10), mat)
      dial.rotation.x = Math.PI / 2
      dial.position.set(0, 0.10, 0.095)
      g.add(box, dial)
      break
    }
    case 'diamond': {
      const ped = new THREE.Mesh(new THREE.CylinderGeometry(0.07, 0.09, 0.06, 8), dark)
      ped.position.y = 0.03
      const gem = new THREE.Mesh(new THREE.OctahedronGeometry(0.11, 0), mat)
      gem.position.y = 0.20
      g.add(ped, gem)
      spinningTargets.push(gem)
      break
    }
    case 'rocket': {
      const body = new THREE.Mesh(new THREE.CylinderGeometry(0.055, 0.065, 0.22, 10), mat)
      body.position.y = 0.13
      const nose = new THREE.Mesh(new THREE.ConeGeometry(0.06, 0.10, 10), dark)
      nose.position.y = 0.29
      g.add(body, nose)
      break
    }
    case 'globe': {
      const stand = new THREE.Mesh(new THREE.CylinderGeometry(0.04, 0.08, 0.06, 8), dark)
      stand.position.y = 0.03
      const orb = new THREE.Mesh(new THREE.IcosahedronGeometry(0.11, 1), mat)
      orb.position.y = 0.20
      g.add(stand, orb)
      spinningTargets.push(orb)
      break
    }
    case 'radar': {
      const base = new THREE.Mesh(new THREE.CylinderGeometry(0.08, 0.10, 0.06, 10), dark)
      base.position.y = 0.03
      const dish = new THREE.Mesh(new THREE.CylinderGeometry(0.08, 0.04, 0.12, 8), mat)
      dish.position.y = 0.16
      dish.rotation.z = 0.35
      g.add(base, dish)
      spinningTargets.push(dish)
      break
    }
  }
  return g
}

function buildRobloxPlant(scale = 1): THREE.Group {
  const g = new THREE.Group()
  const pot = new THREE.Mesh(
    new THREE.CylinderGeometry(0.30, 0.22, 0.42, 10),
    new THREE.MeshStandardMaterial({ color: '#E2E8F0', roughness: 0.4 }),
  )
  pot.position.y = 0.21
  pot.castShadow = true
  const leafMat = new THREE.MeshStandardMaterial({ color: '#10B981', roughness: 0.65 })
  const foliageA = new THREE.Mesh(new THREE.BoxGeometry(0.55, 0.48, 0.55), leafMat)
  foliageA.position.y = 0.62
  foliageA.rotation.y = 0.4
  foliageA.castShadow = true
  const foliageB = new THREE.Mesh(new THREE.BoxGeometry(0.38, 0.38, 0.38), leafMat)
  foliageB.position.y = 0.92
  foliageB.rotation.y = -0.35
  g.add(pot, foliageA, foliageB)
  g.scale.setScalar(scale)
  return g
}

/** Builds a sleeks circular beveled Holotable top deck using a Ring ExtrudeGeometry */
function buildCircularConsoleDeckMesh(mat: THREE.Material): THREE.Mesh {
  const outerR = 1.98
  const innerR = 1.12
  const shape = new THREE.Shape()
  shape.absarc(0, 0, outerR, 0, Math.PI * 2, false)
  const hole = new THREE.Path()
  hole.absarc(0, 0, innerR, 0, Math.PI * 2, true)
  shape.holes.push(hole)

  const geom = new THREE.ExtrudeGeometry(shape, {
    depth: 0.16,
    bevelEnabled: true,
    bevelThickness: 0.03,
    bevelSize: 0.035,
    bevelSegments: 2,
    curveSegments: 42,
  })
  // Rotate so the extruded ring lies flat on XZ plane
  geom.rotateX(-Math.PI / 2)
  const mesh = new THREE.Mesh(geom, mat)
  mesh.position.y = 0.58
  mesh.castShadow = true
  mesh.receiveShadow = true
  return mesh
}

/** Builds a strip of colorful glowing tactile arcade console buttons like the reference image */
function buildArcadeButtonConsole(
  length: number,
  buttonCount: number,
  frameMat: THREE.Material,
  arcadeButtonMeshes: THREE.Mesh[],
): THREE.Group {
  const g = new THREE.Group()
  const base = new THREE.Mesh(new THREE.BoxGeometry(length, 0.56, 0.58), frameMat)
  base.position.y = 0.28
  base.castShadow = true
  base.receiveShadow = true

  const tray = new THREE.Mesh(
    new THREE.BoxGeometry(length + 0.08, 0.10, 0.66),
    new THREE.MeshStandardMaterial({ color: '#64748B', roughness: 0.3, metalness: 0.45 }),
  )
  tray.position.y = 0.60
  g.add(base, tray)

  const palette = ['#FACC15', '#22D3EE', '#EF4444', '#22C55E', '#3B82F6']
  const spacing = (length - 0.35) / Math.max(1, buttonCount)
  const startX = -(length - 0.35) / 2 + spacing / 2

  for (let i = 0; i < buttonCount; i += 1) {
    const col = palette[i % palette.length]!
    const btn = new THREE.Mesh(
      new THREE.BoxGeometry(0.20, 0.075, 0.20),
      new THREE.MeshStandardMaterial({
        color: col,
        emissive: col,
        emissiveIntensity: 0.5,
        roughness: 0.25,
      }),
    )
    btn.position.set(startX + i * spacing, 0.675, 0.08)
    g.add(btn)
    arcadeButtonMeshes.push(btn)
  }

  return g
}

/** Builds the complete "Metropolitan Trading Hub" 3D room matching the reference design */
export function buildRobloxTycoonRoom(
  metasById: Record<string, SpeakerVisualMeta>,
  maxAnisotropy = 8,
): RobloxRoomHandle {
  const group = new THREE.Group()
  group.name = 'trinity_roblox_room'

  // Lighting setup (tuned dynamically by applyLightingTheme; morning sun enters via left/back glass)
  const ambientLight = new THREE.AmbientLight('#F0F9FF', 0.94)
  const hemiLight = new THREE.HemisphereLight('#E0F2FE', '#64748B', 0.86)
  const keyLight = new THREE.DirectionalLight('#FFFBEB', 1.48)
  keyLight.position.set(-16, 15, -13)
  keyLight.castShadow = true
  keyLight.shadow.mapSize.set(1024, 1024)
  keyLight.shadow.bias = -0.0004
  keyLight.shadow.normalBias = 0.02
  keyLight.shadow.camera.near = 2
  keyLight.shadow.camera.far = 48
  keyLight.shadow.camera.left = -19
  keyLight.shadow.camera.right = 19
  keyLight.shadow.camera.top = 16
  keyLight.shadow.camera.bottom = -16

  const rimLight = new THREE.PointLight('#22D3EE', 1.4, 24)
  rimLight.position.set(0, 3.6, -1.25)

  // Procedural Sky Dome Gradient Shader visible behind panoramic windows
  const skyDomeMat = createSkyDomeShaderMaterial()
  const skyDome = new THREE.Mesh(new THREE.SphereGeometry(52, 24, 16), skyDomeMat)
  group.add(ambientLight, hemiLight, keyLight, rimLight, skyDome)

  // 1. Floor (Sleek Architectural Glass-Tile Multi-Map PBR + Fiber-Optic Neon Grid)
  const dayFloorSet = createArchitecturalFloorTextureSet('day', maxAnisotropy)
  const twilightFloorSet = createArchitecturalFloorTextureSet('twilight', maxAnisotropy)

  const floorMesh = new THREE.Mesh(
    new THREE.BoxGeometry(28, 0.3, 20),
    new THREE.MeshStandardMaterial({
      map: dayFloorSet.map,
      roughnessMap: dayFloorSet.roughnessMap,
      bumpMap: dayFloorSet.bumpMap,
      bumpScale: 0.022,
      emissiveMap: dayFloorSet.emissiveMap,
      emissive: new THREE.Color('#38BDF8'),
      emissiveIntensity: 0.42,
      roughness: 1.0, // Driven per-pixel by roughnessMap (~0.14 glass panels, ~0.85 seams)
      metalness: 0.34,
    }),
  )
  floorMesh.position.set(0, -0.15, 0)
  floorMesh.receiveShadow = true
  group.add(floorMesh)

  // Subtle Window Sky-Wash Decals simulating glass curtain reflection across perimeter floor
  const windowWashTex = createWindowWashTexture()
  const windowWashMat = new THREE.MeshBasicMaterial({
    map: windowWashTex,
    color: '#BAE6FD',
    transparent: true,
    opacity: 0.38,
    blending: THREE.AdditiveBlending,
    depthWrite: false,
  })
  const backWindowWash = new THREE.Mesh(new THREE.PlaneGeometry(27.2, 5.8), windowWashMat)
  backWindowWash.rotation.x = -Math.PI / 2
  backWindowWash.position.set(0, 0.004, -6.75)

  const leftWindowWash = new THREE.Mesh(new THREE.PlaneGeometry(19.2, 5.4), windowWashMat)
  leftWindowWash.rotation.x = -Math.PI / 2
  leftWindowWash.rotation.z = Math.PI / 2
  leftWindowWash.position.set(-10.95, 0.004, 0)
  group.add(backWindowWash, leftWindowWash)

  // 2. Panoramic Floor-to-Ceiling Glass Curtain Walls + Ceiling LED Cove
  const frameMat = new THREE.MeshStandardMaterial({ color: '#475569', roughness: 0.35, metalness: 0.4 })
  const sillMat = new THREE.MeshStandardMaterial({ color: '#CBD5E1', roughness: 0.4, metalness: 0.25 })
  const wallMeshes: THREE.Mesh[] = []

  const glassWallMat = new THREE.MeshStandardMaterial({
    color: '#BAE6FD',
    transparent: true,
    opacity: 0.16,
    roughness: 0.08,
    metalness: 0.25,
  })
  const ledCoveMat = new THREE.MeshBasicMaterial({ color: '#67E8F9' })

  // Back wall sill + panoramic glass + top beam + cyan neon cove
  const backSill = new THREE.Mesh(new THREE.BoxGeometry(28, 0.30, 0.32), sillMat)
  backSill.position.set(0, 0.15, -9.8)
  const backGlass = new THREE.Mesh(new THREE.BoxGeometry(28, 5.7, 0.10), glassWallMat)
  backGlass.position.set(0, 3.15, -9.8)
  const backSoffit = new THREE.Mesh(new THREE.BoxGeometry(28.2, 0.32, 0.65), frameMat)
  backSoffit.position.set(0, 6.16, -9.68)
  const backLedStrip = new THREE.Mesh(new THREE.BoxGeometry(27.8, 0.09, 0.12), ledCoveMat)
  backLedStrip.position.set(0, 6.04, -9.40)

  // Vertical mullions across back glass curtain wall (cast real window shadows across floor)
  for (let mx = -13.5; mx <= 13.5; mx += 4.5) {
    const mullion = new THREE.Mesh(new THREE.BoxGeometry(0.18, 6.0, 0.24), frameMat)
    mullion.position.set(mx, 3.0, -9.8)
    mullion.castShadow = true
    group.add(mullion)
  }

  // Left wall sill + panoramic glass + ceiling LED cove
  const leftSill = new THREE.Mesh(new THREE.BoxGeometry(0.32, 0.30, 20), sillMat)
  leftSill.position.set(-13.8, 0.15, 0)
  const leftGlass = new THREE.Mesh(new THREE.BoxGeometry(0.10, 5.7, 20), glassWallMat)
  leftGlass.position.set(-13.8, 3.15, 0)
  const leftSoffit = new THREE.Mesh(new THREE.BoxGeometry(0.65, 0.32, 20), frameMat)
  leftSoffit.position.set(-13.68, 6.16, 0)
  const leftLedStrip = new THREE.Mesh(new THREE.BoxGeometry(0.12, 0.09, 19.6), ledCoveMat)
  leftLedStrip.position.set(-13.40, 6.04, 0)

  for (let mz = -9.0; mz <= 9.0; mz += 4.5) {
    const mullion = new THREE.Mesh(new THREE.BoxGeometry(0.24, 6.0, 0.18), frameMat)
    mullion.position.set(-13.8, 3.0, mz)
    mullion.castShadow = true
    group.add(mullion)
  }

  wallMeshes.push(backSill, leftSill)

  // Perimeter anodized steel floor skirting + recessed cyan cove light locking walls onto glass floor
  const perimeterTrimMat = new THREE.MeshStandardMaterial({
    color: '#1E293B',
    roughness: 0.3,
    metalness: 0.6,
  })
  const perimeterCoveMat = new THREE.MeshBasicMaterial({
    color: '#38BDF8',
    transparent: true,
    opacity: 0.35,
  })
  const backFloorTrim = new THREE.Mesh(new THREE.BoxGeometry(27.8, 0.06, 0.14), perimeterTrimMat)
  backFloorTrim.position.set(0, 0.03, -9.61)
  const backFloorCove = new THREE.Mesh(new THREE.BoxGeometry(27.6, 0.015, 0.06), perimeterCoveMat)
  backFloorCove.position.set(0, 0.008, -9.53)
  const leftFloorTrim = new THREE.Mesh(new THREE.BoxGeometry(0.14, 0.06, 19.6), perimeterTrimMat)
  leftFloorTrim.position.set(-13.61, 0.03, 0)
  const leftFloorCove = new THREE.Mesh(new THREE.BoxGeometry(0.06, 0.015, 19.4), perimeterCoveMat)
  leftFloorCove.position.set(-13.53, 0.008, 0)

  group.add(
    backSill,
    backGlass,
    backSoffit,
    backLedStrip,
    backFloorTrim,
    backFloorCove,
    leftSill,
    leftGlass,
    leftSoffit,
    leftLedStrip,
    leftFloorTrim,
    leftFloorCove,
  )

  // Exterior Terrace Balcony Deck & Railing outside Left & Back Glass
  const terraceBack = new THREE.Mesh(new THREE.BoxGeometry(31, 0.22, 2.2), sillMat)
  terraceBack.position.set(-1.5, -0.05, -11.05)
  const terraceLeft = new THREE.Mesh(new THREE.BoxGeometry(2.2, 0.22, 21), sillMat)
  terraceLeft.position.set(-15.05, -0.05, -0.5)
  const railBack = new THREE.Mesh(new THREE.BoxGeometry(30, 0.07, 0.08), frameMat)
  railBack.position.set(-1.5, 0.92, -12.05)
  const railLeft = new THREE.Mesh(new THREE.BoxGeometry(0.08, 0.07, 20.5), frameMat)
  railLeft.position.set(-16.05, 0.92, -0.5)
  group.add(terraceBack, terraceLeft, railBack, railLeft)

  // Terrace greenery pots visible out the windows
  for (const [bx, bz] of [
    [-11.2, -11.0],
    [-4.8, -11.0],
    [4.6, -11.0],
    [-15.0, -6.2],
    [-15.0, 0.4],
  ] as const) {
    const bp = buildRobloxPlant(0.95)
    bp.position.set(bx, 0.06, bz)
    group.add(bp)
  }

  // Animated 3D Metropolitan Skyscrapers & Drifting Voxel Clouds outside Glass
  const daySkyscraperTex = createSkyscraperFacadeTexture(true)
  const nightSkyscraperTex = createSkyscraperFacadeTexture(false)
  const skyscraperMat = new THREE.MeshStandardMaterial({
    map: daySkyscraperTex,
    roughness: 0.35,
    metalness: 0.25,
  })
  const rooftopMat = new THREE.MeshStandardMaterial({ color: '#334155', roughness: 0.6 })
  const skylineGroup = new THREE.Group()
  const beaconMeshes: THREE.Mesh[] = []

  const towerSpecs: Array<{ x: number; z: number; w: number; d: number; h: number }> = [
    // Back vista skyscrapers
    { x: -13.5, z: -17.5, w: 3.2, d: 2.8, h: 12.5 },
    { x: -9.2, z: -19.8, w: 3.8, d: 3.2, h: 16.2 },
    { x: -4.2, z: -16.8, w: 2.8, d: 2.6, h: 11.4 },
    { x: 1.5, z: -20.5, w: 4.0, d: 3.4, h: 17.0 },
    { x: 6.8, z: -17.5, w: 3.2, d: 2.8, h: 13.5 },
    { x: 11.8, z: -18.8, w: 3.5, d: 3.0, h: 14.8 },
    // Left vista skyscrapers (right behind council table from hero camera!)
    { x: -19.5, z: -12.0, w: 3.4, d: 3.4, h: 13.8 },
    { x: -21.2, z: -6.5, w: 3.6, d: 3.6, h: 16.5 },
    { x: -18.8, z: -1.2, w: 2.9, d: 3.1, h: 11.8 },
    { x: -20.5, z: 4.6, w: 3.4, d: 3.4, h: 14.4 },
  ]

  for (let i = 0; i < towerSpecs.length; i += 1) {
    const ts = towerSpecs[i]!
    const tower = new THREE.Mesh(
      new THREE.BoxGeometry(ts.w, ts.h, ts.d),
      [skyscraperMat, skyscraperMat, rooftopMat, rooftopMat, skyscraperMat, skyscraperMat],
    )
    tower.position.set(ts.x, ts.h / 2 - 2.0, ts.z)
    skylineGroup.add(tower)

    if (i % 2 === 0) {
      const mast = new THREE.Mesh(new THREE.CylinderGeometry(0.05, 0.05, 1.3, 6), frameMat)
      mast.position.set(ts.x, ts.h - 2.0 + 0.65, ts.z)
      const beacon = new THREE.Mesh(
        new THREE.SphereGeometry(0.14, 8, 8),
        new THREE.MeshBasicMaterial({ color: i % 4 === 0 ? '#38BDF8' : '#F43F5E' }),
      )
      beacon.position.set(ts.x, ts.h - 2.0 + 1.35, ts.z)
      skylineGroup.add(mast, beacon)
      beaconMeshes.push(beacon)
    }
  }
  group.add(skylineGroup)

  // 2B. Crisp Morning Sunbeams, Sunlit Floor Pools & Dust Motes + Eye-Level Multi-Cluster Clouds
  const sunGroup = new THREE.Group()
  sunGroup.name = 'morning_sunlight_fx'
  const sunbeamMat = createSunbeamShaderMaterial()
  const sunPoolTex = createSunFloorPoolTexture()
  const sunPoolMat = new THREE.MeshBasicMaterial({
    map: sunPoolTex,
    color: '#FFFBEB',
    transparent: true,
    opacity: 0.72,
    blending: THREE.AdditiveBlending,
    depthWrite: false,
  })

  // Align 4 soft volumetric sun shafts with morning sun direction (-16, 15, -13) -> interior floor
  const sunShaftSpecs: ReadonlyArray<{
    from: [number, number, number]
    to: [number, number, number]
    rTop: number
    rBottom: number
    poolScale: [number, number]
  }> = [
    { from: [-13.2, 5.2, -5.2], to: [-6.6, 0.02, -0.4], rTop: 1.35, rBottom: 2.35, poolScale: [4.8, 3.6] },
    { from: [-13.2, 5.2, 1.4], to: [-5.8, 0.02, 5.6], rTop: 1.45, rBottom: 2.55, poolScale: [5.4, 3.9] },
    { from: [-5.4, 5.2, -9.2], to: [-0.8, 0.02, -2.8], rTop: 1.40, rBottom: 2.45, poolScale: [5.0, 3.8] },
    { from: [3.2, 5.2, -9.2], to: [7.4, 0.02, -2.5], rTop: 1.30, rBottom: 2.30, poolScale: [4.6, 3.5] },
  ]
  const beamUpVec = new THREE.Vector3(0, 1, 0)
  for (const spec of sunShaftSpecs) {
    const startVec = new THREE.Vector3(...spec.from)
    const endVec = new THREE.Vector3(...spec.to)
    const shaftVec = new THREE.Vector3().subVectors(startVec, endVec)
    const length = shaftVec.length()
    const beamMesh = new THREE.Mesh(
      new THREE.CylinderGeometry(spec.rTop, spec.rBottom, length, 18, 1, true),
      sunbeamMat,
    )
    beamMesh.position.addVectors(startVec, endVec).multiplyScalar(0.5)
    beamMesh.quaternion.setFromUnitVectors(beamUpVec, shaftVec.normalize())
    sunGroup.add(beamMesh)

    const floorPool = new THREE.Mesh(
      new THREE.PlaneGeometry(spec.poolScale[0], spec.poolScale[1]),
      sunPoolMat,
    )
    floorPool.rotation.x = -Math.PI / 2
    floorPool.rotation.z = -0.38
    floorPool.position.set(endVec.x, 0.006, endVec.z)
    sunGroup.add(floorPool)
  }

  // 42 Floating Morning Dust Motes shimmering softly inside sunlit volume (1 GPU InstancedMesh draw call)
  const DUST_MOTE_COUNT = 42
  const dustDummy = new THREE.Object3D()
  const dustInstanced = new THREE.InstancedMesh(
    new THREE.BoxGeometry(0.038, 0.038, 0.038),
    new THREE.MeshBasicMaterial({
      color: '#FEF3C7',
      transparent: true,
      opacity: 0.62,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    }),
    DUST_MOTE_COUNT,
  )
  dustInstanced.instanceMatrix.setUsage(THREE.DynamicDrawUsage)
  const dustSeeds: Array<{
    x: number
    y: number
    z: number
    phase: number
    speed: number
    scale: number
  }> = []
  for (let i = 0; i < DUST_MOTE_COUNT; i += 1) {
    const sx = -9.5 + ((i * 53) % 190) / 10
    const sy = 0.75 + ((i * 29) % 34) / 10
    const sz = -6.8 + ((i * 67) % 125) / 10
    const sc = 0.65 + (i % 4) * 0.25
    dustSeeds.push({
      x: sx,
      y: sy,
      z: sz,
      phase: i * 0.62,
      speed: 0.45 + (i % 5) * 0.12,
      scale: sc,
    })
    dustDummy.position.set(sx, sy, sz)
    dustDummy.scale.setScalar(sc)
    dustDummy.updateMatrix()
    dustInstanced.setMatrixAt(i, dustDummy.matrix)
  }
  dustInstanced.instanceMatrix.needsUpdate = true
  sunGroup.add(dustInstanced)
  group.add(sunGroup)

  // Eye-Level Multi-Cluster Roblox Puffy Clouds drifting continuously across both window horizons
  const cloudGroup = new THREE.Group()
  cloudGroup.name = 'daytime_puffy_clouds'
  const cloudTopMat = new THREE.MeshStandardMaterial({
    color: '#FFFFFF',
    emissive: '#E0F2FE',
    emissiveIntensity: 0.34,
    roughness: 0.85,
    metalness: 0.02,
  })
  const cloudUnderMat = new THREE.MeshStandardMaterial({
    color: '#DCEEFB',
    emissive: '#BAE6FD',
    emissiveIntensity: 0.24,
    roughness: 0.92,
    metalness: 0.0,
  })

  const cloudSpecs: ReadonlyArray<{
    x: number
    y: number
    z: number
    s: number
    axis: 'x' | 'z'
    speed: number
    span: number
  }> = [
    // Back panoramic horizon clouds (visible right through back glass at y = 2.7..5.9)
    { x: -16, y: 3.6, z: -22.5, s: 1.15, axis: 'x', speed: 0.85, span: 28 },
    { x: -4, y: 4.8, z: -26.5, s: 1.45, axis: 'x', speed: 0.62, span: 28 },
    { x: 9, y: 3.1, z: -23.5, s: 1.1, axis: 'x', speed: 0.96, span: 28 },
    { x: 20, y: 5.4, z: -31.0, s: 1.65, axis: 'x', speed: 0.48, span: 30 },
    { x: -24, y: 4.2, z: -29.5, s: 1.35, axis: 'x', speed: 0.56, span: 30 },
    // Left panoramic horizon clouds (visible behind the council table through left glass!)
    { x: -23.5, y: 3.4, z: -10, s: 1.2, axis: 'z', speed: 0.78, span: 24 },
    { x: -27.5, y: 4.9, z: 0, s: 1.5, axis: 'z', speed: 0.55, span: 24 },
    { x: -24.5, y: 3.9, z: 11, s: 1.25, axis: 'z', speed: 0.72, span: 24 },
  ]
  const driftingClouds: Array<{
    mesh: THREE.Group
    baseX: number
    baseY: number
    baseZ: number
    axis: 'x' | 'z'
    speed: number
    span: number
    phase: number
  }> = []

  for (let i = 0; i < cloudSpecs.length; i += 1) {
    const cs = cloudSpecs[i]!
    const cg = new THREE.Group()
    // Build puffy multi-layered stylized Roblox cloud cluster (shaded base + sunlit top puffs)
    const baseSlab = new THREE.Mesh(
      new THREE.BoxGeometry(3.4 * cs.s, 0.72 * cs.s, 1.55 * cs.s),
      cloudUnderMat,
    )
    const puffLeft = new THREE.Mesh(
      new THREE.BoxGeometry(1.55 * cs.s, 0.82 * cs.s, 1.35 * cs.s),
      cloudTopMat,
    )
    puffLeft.position.set(-0.95 * cs.s, 0.34 * cs.s, 0.08 * cs.s)
    const puffCenter = new THREE.Mesh(
      new THREE.BoxGeometry(2.05 * cs.s, 1.05 * cs.s, 1.48 * cs.s),
      cloudTopMat,
    )
    puffCenter.position.set(0.1 * cs.s, 0.55 * cs.s, -0.05 * cs.s)
    const puffRight = new THREE.Mesh(
      new THREE.BoxGeometry(1.4 * cs.s, 0.75 * cs.s, 1.22 * cs.s),
      cloudTopMat,
    )
    puffRight.position.set(1.08 * cs.s, 0.28 * cs.s, 0.1 * cs.s)

    cg.add(baseSlab, puffLeft, puffCenter, puffRight)
    // Align left-window clouds lengthwise along Z so they show rich wide silhouette
    if (cs.axis === 'z') {
      cg.rotation.y = Math.PI / 2
    }
    cg.position.set(cs.x, cs.y, cs.z)
    cloudGroup.add(cg)
    driftingClouds.push({
      mesh: cg,
      baseX: cs.x,
      baseY: cs.y,
      baseZ: cs.z,
      axis: cs.axis,
      speed: cs.speed,
      span: cs.span,
      phase: i * 0.9,
    })
  }
  group.add(cloudGroup)

  // 3. ZONE A — Circular Holographic Trading Console & 3D Candlestick Emitter
  const cx = ROOM_LAYOUT.councilCenter.x
  const cz = ROOM_LAYOUT.councilCenter.z

  // Translucent Smoked Sapphire Glass Plinth (preserves glowing grid reflections beneath)
  const carpetMesh = new THREE.Mesh(
    new THREE.CylinderGeometry(3.42, 3.48, 0.014, 64),
    new THREE.MeshStandardMaterial({
      color: '#173452',
      transparent: true,
      opacity: 0.44,
      roughness: 0.14,
      metalness: 0.48,
    }),
  )
  carpetMesh.position.set(cx, 0.007, cz)
  carpetMesh.receiveShadow = true

  const plazaDecalTex = createCouncilPlazaDecalTexture(maxAnisotropy)
  const plazaDecalMesh = new THREE.Mesh(
    new THREE.PlaneGeometry(7.05, 7.05),
    new THREE.MeshBasicMaterial({
      map: plazaDecalTex,
      transparent: true,
      depthWrite: false,
    }),
  )
  plazaDecalMesh.rotation.x = -Math.PI / 2
  plazaDecalMesh.position.set(cx, 0.016, cz)

  const outerFlangeMesh = new THREE.Mesh(
    new THREE.RingGeometry(3.44, 3.70, 64),
    new THREE.MeshStandardMaterial({
      color: '#64748B',
      roughness: 0.26,
      metalness: 0.72,
      side: THREE.DoubleSide,
    }),
  )
  outerFlangeMesh.rotation.x = -Math.PI / 2
  outerFlangeMesh.position.set(cx, 0.012, cz)

  const carpetBorderMesh = new THREE.Mesh(
    new THREE.TorusGeometry(3.48, 0.032, 12, 64),
    new THREE.MeshBasicMaterial({
      color: '#38BDF8',
      transparent: true,
      opacity: 0.82,
    }),
  )
  carpetBorderMesh.rotation.x = -Math.PI / 2
  carpetBorderMesh.position.set(cx, 0.022, cz)
  group.add(carpetMesh, plazaDecalMesh, outerFlangeMesh, carpetBorderMesh)

  // Flush Floor Optical Conduit Traces (architectural leading lines linking Council to zones)
  const conduitMat = new THREE.MeshBasicMaterial({
    color: '#38BDF8',
    transparent: true,
    opacity: 0.42,
  })
  const conduitSegments: ReadonlyArray<{ x: number; z: number; w: number; d: number }> = [
    // Twin Hermes Dais approach rails
    { x: -0.58, z: -4.95, w: 0.06, d: 1.9 },
    { x: 0.58, z: -4.95, w: 0.06, d: 1.9 },
    // Center corridor link to Duel Arena
    { x: 0, z: 0.92, w: 0.07, d: 0.95 },
    // Right corridor bus to Quant Server Rack
    { x: 6.15, z: -1.25, w: 5.0, d: 0.06 },
    { x: 8.65, z: -0.58, w: 0.06, d: 1.4 },
    { x: 9.20, z: 0.10, w: 1.1, d: 0.06 },
    // Left VIP doorway approach bus
    { x: -5.85, z: -1.25, w: 4.5, d: 0.06 },
    { x: -8.05, z: -2.42, w: 0.06, d: 2.4 },
    // Rear Analyst Workstation bus bar
    { x: 0, z: 4.18, w: 17.8, d: 0.055 },
  ]
  for (const seg of conduitSegments) {
    const strip = new THREE.Mesh(new THREE.BoxGeometry(seg.w, 0.008, seg.d), conduitMat)
    strip.position.set(seg.x, 0.005, seg.z)
    group.add(strip)
  }

  // Circular Holo-Console Base, Grounded Contact Shadow & Emitter Well (clickable -> zoom council)
  const pedestalShadowTex = createPedestalShadowTexture()
  const pedestalShadow = new THREE.Mesh(
    new THREE.PlaneGeometry(4.4, 4.4),
    new THREE.MeshBasicMaterial({
      map: pedestalShadowTex,
      transparent: true,
      depthWrite: false,
    }),
  )
  pedestalShadow.rotation.x = -Math.PI / 2
  pedestalShadow.position.set(cx, 0.032, cz)
  group.add(pedestalShadow)

  const holoConsoleGroup = new THREE.Group()
  holoConsoleGroup.position.set(cx, 0, cz)
  holoConsoleGroup.userData = { holoFocus: true }

  const tableBodyMat = new THREE.MeshStandardMaterial({
    color: '#94A3B8',
    roughness: 0.28,
    metalness: 0.42,
  })
  const darkMetalMat = new THREE.MeshStandardMaterial({
    color: '#1E293B',
    roughness: 0.4,
    metalness: 0.45,
  })
  const consolePedestal = new THREE.Mesh(
    new THREE.CylinderGeometry(1.42, 1.62, 0.56, 44),
    darkMetalMat,
  )
  consolePedestal.position.y = 0.28
  consolePedestal.castShadow = true
  consolePedestal.receiveShadow = true

  const emitterWellFloor = new THREE.Mesh(
    new THREE.CylinderGeometry(1.13, 1.08, 0.16, 44),
    new THREE.MeshStandardMaterial({
      color: '#071626',
      emissive: '#083344',
      emissiveIntensity: 0.6,
      roughness: 0.2,
    }),
  )
  emitterWellFloor.position.y = 0.56

  const consoleRingDeck = buildCircularConsoleDeckMesh(tableBodyMat)

  // Signature Cyan Neon Concentric Rings (top rim + outer skirt)
  const neonRingMat = new THREE.MeshBasicMaterial({ color: '#22D3EE' })
  const innerHoloRim = new THREE.Mesh(new THREE.TorusGeometry(1.14, 0.042, 12, 48), neonRingMat)
  innerHoloRim.rotation.x = -Math.PI / 2
  innerHoloRim.position.y = 0.78

  const outerHoloRim = new THREE.Mesh(new THREE.TorusGeometry(2.01, 0.036, 12, 48), neonRingMat)
  outerHoloRim.rotation.x = -Math.PI / 2
  outerHoloRim.position.y = 0.67

  const pedestalGlowRing = new THREE.Mesh(new THREE.TorusGeometry(1.64, 0.03, 10, 48), neonRingMat)
  pedestalGlowRing.rotation.x = -Math.PI / 2
  pedestalGlowRing.position.y = 0.06

  // Twin luminous status keys on front apron of console (like reference holotable)
  const arcadeButtonMeshes: THREE.Mesh[] = []
  for (const kx of [-0.15, 0.15]) {
    const key = new THREE.Mesh(
      new THREE.BoxGeometry(0.14, 0.06, 0.12),
      new THREE.MeshStandardMaterial({
        color: '#E0F2FE',
        emissive: '#38BDF8',
        emissiveIntensity: 0.75,
      }),
    )
    key.position.set(kx, 0.78, 1.58)
    key.rotation.x = 0.25
    holoConsoleGroup.add(key)
    arcadeButtonMeshes.push(key)
  }

  holoConsoleGroup.add(
    consolePedestal,
    emitterWellFloor,
    consoleRingDeck,
    innerHoloRim,
    outerHoloRim,
    pedestalGlowRing,
  )

  // Angled Personal Trading Touchscreens around the Circular Holotable Rim
  const monitorTex = createDualMonitorTexture()
  const monitorGlowMeshes: THREE.Mesh[] = []
  const holoScreenMat = new THREE.MeshBasicMaterial({ map: monitorTex })

  ROOM_LAYOUT.councilAngles.forEach((angle) => {
    const sx = Math.cos(angle) * ROOM_LAYOUT.councilTableRadius
    const sz = Math.sin(angle) * ROOM_LAYOUT.councilTableRadius
    const pod = new THREE.Group()
    pod.position.set(sx, 0.80, sz)
    // Face outward radially toward the Guru seated/standing at this angle
    pod.rotation.y = Math.atan2(sx, sz)

    const bezel = new THREE.Mesh(new THREE.BoxGeometry(0.76, 0.06, 0.44), darkMetalMat)
    bezel.rotation.x = 0.34
    const glassScreen = new THREE.Mesh(new THREE.PlaneGeometry(0.66, 0.36), holoScreenMat)
    glassScreen.position.set(0, 0.036, 0)
    glassScreen.rotation.x = -Math.PI / 2 + 0.34
    bezel.add(glassScreen)
    pod.add(bezel)
    monitorGlowMeshes.push(glassScreen)
    holoConsoleGroup.add(pod)
  })

  group.add(holoConsoleGroup)

  // Mount Centerpiece 3D Volumetric Candlestick/Arrow Hologram inside the emitter well
  const hologram = buildCentralHologramProjector()
  hologram.group.position.set(cx, 0, cz)
  group.add(hologram.group)

  // 5 Swivel Council Chairs at exact choreography seat transforms
  for (let idx = 0; idx < 5; idx += 1) {
    const tf = getCouncilSeatTransform(idx)
    const chair = buildOfficeChair('#334155', '#38BDF8')
    chair.position.set(tf.pos.x, 0, tf.pos.z)
    chair.rotation.y = tf.rotY
    group.add(chair)
  }

  // Crossfire Octagonal Tactical Arena safely positioned in front of Circular Holotable
  const duelArenaTex = createDuelArenaSurfaceTexture(maxAnisotropy)
  const duelBase = new THREE.Mesh(
    new THREE.CylinderGeometry(1.36, 1.44, 0.028, 8),
    new THREE.MeshStandardMaterial({
      color: '#1E293B',
      roughness: 0.38,
      metalness: 0.55,
    }),
  )
  duelBase.rotation.y = Math.PI / 8
  duelBase.position.set(ROOM_LAYOUT.duelPadCenter.x, 0.014, ROOM_LAYOUT.duelPadCenter.z)
  duelBase.receiveShadow = true

  const duelSurface = new THREE.Mesh(
    new THREE.CircleGeometry(1.24, 8),
    new THREE.MeshStandardMaterial({
      map: duelArenaTex,
      roughness: 0.25,
      metalness: 0.35,
    }),
  )
  duelSurface.rotation.x = -Math.PI / 2
  duelSurface.rotation.z = Math.PI / 8
  duelSurface.position.set(ROOM_LAYOUT.duelPadCenter.x, 0.029, ROOM_LAYOUT.duelPadCenter.z)
  duelSurface.receiveShadow = true

  const duelRingMesh = new THREE.Mesh(
    new THREE.RingGeometry(1.14, 1.28, 8),
    new THREE.MeshBasicMaterial({
      color: '#22D3EE',
      side: THREE.DoubleSide,
      transparent: true,
      opacity: 0.62,
    }),
  )
  duelRingMesh.rotation.x = -Math.PI / 2
  duelRingMesh.rotation.z = Math.PI / 8
  duelRingMesh.position.set(
    ROOM_LAYOUT.duelPadCenter.x,
    0.034,
    ROOM_LAYOUT.duelPadCenter.z,
  )

  const duelInnerRingMesh = new THREE.Mesh(
    new THREE.RingGeometry(0.84, 0.90, 28, 1, 0, Math.PI * 1.45),
    new THREE.MeshBasicMaterial({
      color: '#67E8F9',
      side: THREE.DoubleSide,
      transparent: true,
      opacity: 0.50,
    }),
  )
  duelInnerRingMesh.rotation.x = -Math.PI / 2
  duelInnerRingMesh.position.set(
    ROOM_LAYOUT.duelPadCenter.x,
    0.035,
    ROOM_LAYOUT.duelPadCenter.z,
  )
  group.add(duelBase, duelSurface, duelRingMesh, duelInnerRingMesh)

  // 4. ZONE B — Hermes Dais, Podium & Dual Wall Displays
  const dais = new THREE.Mesh(
    new THREE.BoxGeometry(4.4, 0.14, 2.5),
    new THREE.MeshStandardMaterial({ color: '#334155', roughness: 0.4, metalness: 0.3 }),
  )
  dais.position.set(0, 0.07, -6.5)
  dais.receiveShadow = true

  const podium = new THREE.Mesh(new THREE.BoxGeometry(1.35, 0.92, 0.62), tableBodyMat)
  podium.position.set(
    ROOM_LAYOUT.hermesPodiumPos.x,
    0.46 + 0.14,
    ROOM_LAYOUT.hermesPodiumPos.z,
  )
  podium.castShadow = true
  const podiumTrim = new THREE.Mesh(
    new THREE.BoxGeometry(1.15, 0.14, 0.05),
    new THREE.MeshStandardMaterial({
      color: '#22D3EE',
      emissive: '#22D3EE',
      emissiveIntensity: 0.65,
    }),
  )
  podiumTrim.position.set(
    ROOM_LAYOUT.hermesPodiumPos.x,
    0.92,
    ROOM_LAYOUT.hermesPodiumPos.z + 0.32,
  )
  group.add(dais, podium, podiumTrim)

  // Main Screen Frame + Texture (floating slim bezel on back wall trusses)
  const mainScreenCanvas = document.createElement('canvas')
  mainScreenCanvas.width = 1024
  mainScreenCanvas.height = 448
  const mainScreenTex = new THREE.CanvasTexture(mainScreenCanvas)
  mainScreenTex.colorSpace = THREE.SRGBColorSpace
  const mainFrame = new THREE.Mesh(
    new THREE.BoxGeometry(8.8, 3.7, 0.18),
    darkMetalMat,
  )
  mainFrame.position.set(
    ROOM_LAYOUT.mainWallScreenPos.x,
    ROOM_LAYOUT.mainWallScreenPos.y,
    ROOM_LAYOUT.mainWallScreenPos.z,
  )
  const mainScreenFace = new THREE.Mesh(
    new THREE.PlaneGeometry(8.45, 3.4),
    new THREE.MeshBasicMaterial({ map: mainScreenTex }),
  )
  mainScreenFace.position.set(
    ROOM_LAYOUT.mainWallScreenPos.x,
    ROOM_LAYOUT.mainWallScreenPos.y,
    ROOM_LAYOUT.mainWallScreenPos.z + 0.105,
  )
  group.add(mainFrame, mainScreenFace)

  // Right Wallboard Screen (angled prominently toward council camera, featuring Mountain Chart + Pillars)
  const boardScreenCanvas = document.createElement('canvas')
  boardScreenCanvas.width = 640
  boardScreenCanvas.height = 384
  const boardScreenTex = new THREE.CanvasTexture(boardScreenCanvas)
  boardScreenTex.colorSpace = THREE.SRGBColorSpace
  const boardGroup = new THREE.Group()
  boardGroup.position.set(
    ROOM_LAYOUT.boardWallScreenPos.x,
    ROOM_LAYOUT.boardWallScreenPos.y,
    ROOM_LAYOUT.boardWallScreenPos.z,
  )
  boardGroup.rotation.y = -0.34
  const boardFrame = new THREE.Mesh(
    new THREE.BoxGeometry(5.2, 3.15, 0.18),
    darkMetalMat,
  )
  const boardFace = new THREE.Mesh(
    new THREE.PlaneGeometry(4.92, 2.88),
    new THREE.MeshBasicMaterial({ map: boardScreenTex }),
  )
  boardFace.position.z = 0.105
  boardGroup.add(boardFrame, boardFace)
  group.add(boardGroup)

  // Illuminated Tactile Arcade Control Consoles (Right counter under Chart Board & Left foreground)
  const rightArcadeConsole = buildArcadeButtonConsole(4.6, 8, darkMetalMat, arcadeButtonMeshes)
  rightArcadeConsole.position.set(7.2, 0, -6.9)
  rightArcadeConsole.rotation.y = -0.34

  const leftArcadeConsole = buildArcadeButtonConsole(2.9, 5, darkMetalMat, arcadeButtonMeshes)
  leftArcadeConsole.position.set(-4.9, 0, 2.55)
  leftArcadeConsole.rotation.y = 0.62
  group.add(rightArcadeConsole, leftArcadeConsole)

  // 5. ZONE C — 5 Analyst Dual-Monitor Workstations (using WORKSTATION_SLOTS exact transforms)
  const workstationScreenMat = new THREE.MeshBasicMaterial({ map: monitorTex })
  const deskMatColor = new THREE.MeshStandardMaterial({ color: '#1E293B', roughness: 0.78 })
  const spinningProps: THREE.Object3D[] = []

  for (const [sid, slot] of Object.entries(WORKSTATION_SLOTS)) {
    const meta = metasById[sid]
    const accent = meta?.accentColor ?? '#0EA5E9'

    const deskUnit = new THREE.Group()
    deskUnit.position.set(slot.deskPos.x, 0, slot.deskPos.z)
    deskUnit.rotation.y = slot.rotY
    deskUnit.userData = { workstationSpeakerId: sid }

    const matPad = new THREE.Mesh(new THREE.BoxGeometry(1.9, 0.014, 2.0), deskMatColor)
    matPad.position.set(0, 0.007, -0.35)
    matPad.receiveShadow = true
    const matTrim = new THREE.Mesh(
      new THREE.BoxGeometry(1.86, 0.008, 0.045),
      new THREE.MeshStandardMaterial({
        color: accent,
        emissive: accent,
        emissiveIntensity: 0.48,
      }),
    )
    matTrim.position.set(0, 0.015, 0.63)
    deskUnit.add(matPad, matTrim)

    const deskTop = new THREE.Mesh(new THREE.BoxGeometry(1.66, 0.08, 0.84), tableBodyMat)
    deskTop.position.set(0, 0.72, 0)
    deskTop.castShadow = true
    deskTop.receiveShadow = true
    deskTop.userData = { workstationSpeakerId: sid }

    for (const lx of [-0.72, 0.72]) {
      for (const lz of [-0.32, 0.32]) {
        const leg = new THREE.Mesh(new THREE.BoxGeometry(0.08, 0.70, 0.08), frameMat)
        leg.position.set(lx, 0.35, lz)
        deskUnit.add(leg)
      }
    }

    const deskEdgeLed = new THREE.Mesh(
      new THREE.BoxGeometry(1.38, 0.03, 0.04),
      new THREE.MeshStandardMaterial({
        color: accent,
        emissive: accent,
        emissiveIntensity: 0.55,
      }),
    )
    deskEdgeLed.position.set(0, 0.69, 0.43)

    for (const mx of [-0.34, 0.34]) {
      const stand = new THREE.Mesh(new THREE.BoxGeometry(0.10, 0.16, 0.08), frameMat)
      stand.position.set(mx, 0.84, 0.24)
      const panel = new THREE.Mesh(
        new THREE.BoxGeometry(0.58, 0.33, 0.05),
        darkMetalMat,
      )
      panel.position.set(mx, 0.99, 0.24)
      panel.rotation.y = mx < 0 ? 0.12 : -0.12
      const scr = new THREE.Mesh(new THREE.PlaneGeometry(0.52, 0.27), workstationScreenMat)
      scr.position.set(0, 0, -0.028)
      scr.rotation.y = Math.PI
      panel.add(scr)
      monitorGlowMeshes.push(scr)
      deskUnit.add(stand, panel)
    }

    const kb = new THREE.Mesh(new THREE.BoxGeometry(0.46, 0.025, 0.16), darkMetalMat)
    kb.position.set(0, 0.77, -0.14)
    const prop = buildDeskProp(slot.propType, accent, spinningProps)
    prop.position.set(0.65, 0.76, -0.12)

    const chair = buildOfficeChair('#334155', accent)
    chair.position.set(0, 0, -0.92)

    deskUnit.add(deskTop, deskEdgeLed, kb, prop, chair)
    group.add(deskUnit)
  }

  // Quant Sandbox Math Server Rack (right floor)
  const quantGroup = new THREE.Group()
  quantGroup.position.set(ROOM_LAYOUT.quantRackPos.x, 0, ROOM_LAYOUT.quantRackPos.z)
  const rackBody = new THREE.Mesh(new THREE.BoxGeometry(1.1, 2.2, 1.3), darkMetalMat)
  rackBody.position.y = 1.1
  rackBody.castShadow = true
  quantGroup.add(rackBody)

  const quantLedMeshes: THREE.Mesh[] = []
  for (let row = 0; row < 5; row += 1) {
    const led = new THREE.Mesh(
      new THREE.BoxGeometry(0.05, 0.10, 0.92),
      new THREE.MeshStandardMaterial({
        color: '#22D3EE',
        emissive: '#22D3EE',
        emissiveIntensity: 0.45,
      }),
    )
    led.position.set(-0.56, 0.55 + row * 0.34, 0)
    quantGroup.add(led)
    quantLedMeshes.push(led)
  }
  group.add(quantGroup)

  // 6. ZONE D — Portfolio Owner VIP Glass Box (Back-Left) & Bloxy Lounge (Back-Right)
  const ceoMat = new THREE.MeshStandardMaterial({
    color: '#BAE6FD',
    transparent: true,
    opacity: 0.22,
    roughness: 0.08,
  })
  const ceoFloorTex = createExecutiveFloorTexture(maxAnisotropy)
  const ceoFloor = new THREE.Mesh(
    new THREE.BoxGeometry(5.6, 0.022, 4.4),
    new THREE.MeshStandardMaterial({
      map: ceoFloorTex,
      roughness: 0.34,
      metalness: 0.14,
    }),
  )
  ceoFloor.position.set(ROOM_LAYOUT.ceoRoomCenter.x, 0.011, ROOM_LAYOUT.ceoRoomCenter.z)
  ceoFloor.receiveShadow = true

  const brassTrimMat = new THREE.MeshStandardMaterial({
    color: '#EAB308',
    roughness: 0.24,
    metalness: 0.78,
  })
  const cxCeo = ROOM_LAYOUT.ceoRoomCenter.x
  const czCeo = ROOM_LAYOUT.ceoRoomCenter.z
  const ceoTrimRight = new THREE.Mesh(new THREE.BoxGeometry(0.06, 0.024, 4.46), brassTrimMat)
  ceoTrimRight.position.set(cxCeo + 2.81, 0.012, czCeo)
  const ceoTrimLeft = new THREE.Mesh(new THREE.BoxGeometry(0.06, 0.024, 4.46), brassTrimMat)
  ceoTrimLeft.position.set(cxCeo - 2.81, 0.012, czCeo)
  const ceoTrimBack = new THREE.Mesh(new THREE.BoxGeometry(5.68, 0.024, 0.06), brassTrimMat)
  ceoTrimBack.position.set(cxCeo, 0.012, czCeo - 2.21)
  const ceoTrimFront = new THREE.Mesh(new THREE.BoxGeometry(5.68, 0.024, 0.06), brassTrimMat)
  ceoTrimFront.position.set(cxCeo, 0.012, czCeo + 2.21)
  group.add(ceoFloor, ceoTrimRight, ceoTrimLeft, ceoTrimBack, ceoTrimFront)

  const ceoGlassRight = new THREE.Mesh(new THREE.BoxGeometry(0.10, 2.35, 4.3), ceoMat)
  ceoGlassRight.position.set(-6.8, 1.18, -6.4)
  const ceoGlassFrontLeft = new THREE.Mesh(new THREE.BoxGeometry(3.2, 2.35, 0.10), ceoMat)
  ceoGlassFrontLeft.position.set(-10.8, 1.18, -4.2)
  const ceoTopBeam = new THREE.Mesh(new THREE.BoxGeometry(5.6, 0.10, 0.14), frameMat)
  ceoTopBeam.position.set(-9.6, 2.36, -4.2)
  group.add(ceoGlassRight, ceoGlassFrontLeft, ceoTopBeam)

  const ceoDesk = new THREE.Mesh(new THREE.BoxGeometry(2.1, 0.74, 0.95), darkMetalMat)
  ceoDesk.position.set(ROOM_LAYOUT.ceoRoomCenter.x, 0.37, ROOM_LAYOUT.ceoRoomCenter.z + 0.35)
  ceoDesk.castShadow = true
  const ceoChair = buildOfficeChair('#0C4A6E', '#FBBF24')
  ceoChair.position.set(ROOM_LAYOUT.userCeoSeatPos.x, 0, ROOM_LAYOUT.userCeoSeatPos.z)
  ceoChair.rotation.y = 0.28
  group.add(ceoDesk, ceoChair)

  // Bloxy Lounge corner with plush woven area rug
  const loungeRugTex = createLoungeRugTexture(maxAnisotropy)
  const loungeRug = new THREE.Mesh(
    new THREE.BoxGeometry(3.15, 0.016, 3.85),
    new THREE.MeshStandardMaterial({
      map: loungeRugTex,
      roughness: 0.90,
      metalness: 0.02,
    }),
  )
  loungeRug.position.set(9.35, 0.008, -4.8)
  loungeRug.receiveShadow = true

  const loungeSofa = new THREE.Mesh(
    new THREE.BoxGeometry(0.95, 0.48, 3.2),
    new THREE.MeshStandardMaterial({ color: '#334155', roughness: 0.65 }),
  )
  loungeSofa.position.set(10.1, 0.24, -4.8)
  loungeSofa.castShadow = true
  const loungeTable = new THREE.Mesh(new THREE.BoxGeometry(0.8, 0.42, 1.2), tableBodyMat)
  loungeTable.position.set(8.6, 0.21, -4.8)
  group.add(loungeRug, loungeSofa, loungeTable)

  // 7. Ambient Interior Greenery
  for (const [px, pz, ps] of [
    [-12.4, -4.8, 1.05],
    [-6.2, -8.5, 0.95],
    [12.2, 6.2, 1.05],
  ] as const) {
    const p = buildRobloxPlant(ps)
    p.position.set(px, 0, pz)
    group.add(p)
  }

  let lastScreenSignature = ''
  let activeLightingMode: LightingThemeMode = 'day'

  const applyLightingTheme = (mode: LightingThemeMode) => {
    activeLightingMode = mode
    const floorMat = floorMesh.material as THREE.MeshStandardMaterial
    const carpetMat = carpetMesh.material as THREE.MeshStandardMaterial
    if (mode === 'day') {
      ambientLight.color.set('#F0F9FF')
      ambientLight.intensity = 0.94
      hemiLight.color.set('#E0F2FE')
      hemiLight.groundColor.set('#64748B')
      hemiLight.intensity = 0.86
      keyLight.color.set('#FFFBEB')
      keyLight.position.set(-16, 15, -13)
      keyLight.intensity = 1.48
      rimLight.intensity = 1.1
      floorMat.map = dayFloorSet.map
      floorMat.roughnessMap = dayFloorSet.roughnessMap
      floorMat.bumpMap = dayFloorSet.bumpMap
      floorMat.emissiveMap = dayFloorSet.emissiveMap
      floorMat.emissiveIntensity = 0.42
      floorMat.metalness = 0.34
      floorMat.needsUpdate = true
      carpetMat.color.set('#1E4063')
      carpetMat.opacity = 0.42
      windowWashMat.color.set('#BAE6FD')
      windowWashMat.opacity = 0.38
      conduitMat.opacity = 0.38
      tableBodyMat.color.set('#CBD5E1')
      sillMat.color.set('#E2E8F0')
      skyscraperMat.map = daySkyscraperTex
      skyscraperMat.needsUpdate = true
      skylineGroup.visible = true
      cloudGroup.visible = true
      sunGroup.visible = true
      ;(skyDomeMat.uniforms.uZenithColor!.value as THREE.Color).set('#0EA5E9')
      ;(skyDomeMat.uniforms.uHorizonColor!.value as THREE.Color).set('#E0F2FE')
      ;(skyDomeMat.uniforms.uBottomColor!.value as THREE.Color).set('#64748B')
      if (skyDomeMat.uniforms.uSunIntensity) {
        skyDomeMat.uniforms.uSunIntensity.value = 1.0
      }
    } else {
      ambientLight.color.set('#D8ECFF')
      ambientLight.intensity = 0.66
      hemiLight.color.set('#E0F2FE')
      hemiLight.groundColor.set('#0F172A')
      hemiLight.intensity = 0.56
      keyLight.color.set('#E2E8F0')
      keyLight.position.set(11, 18, 12)
      keyLight.intensity = 1.15
      rimLight.intensity = 1.45
      floorMat.map = twilightFloorSet.map
      floorMat.roughnessMap = twilightFloorSet.roughnessMap
      floorMat.bumpMap = twilightFloorSet.bumpMap
      floorMat.emissiveMap = twilightFloorSet.emissiveMap
      floorMat.emissiveIntensity = 0.74
      floorMat.metalness = 0.40
      floorMat.needsUpdate = true
      carpetMat.color.set('#0C1E33')
      carpetMat.opacity = 0.52
      windowWashMat.color.set('#38BDF8')
      windowWashMat.opacity = 0.24
      conduitMat.opacity = 0.56
      tableBodyMat.color.set('#64748B')
      sillMat.color.set('#334155')
      skyscraperMat.map = nightSkyscraperTex
      skyscraperMat.needsUpdate = true
      skylineGroup.visible = true
      cloudGroup.visible = false
      sunGroup.visible = false
      ;(skyDomeMat.uniforms.uZenithColor!.value as THREE.Color).set('#020617')
      ;(skyDomeMat.uniforms.uHorizonColor!.value as THREE.Color).set('#1E3A5F')
      ;(skyDomeMat.uniforms.uBottomColor!.value as THREE.Color).set('#091322')
      if (skyDomeMat.uniforms.uSunIntensity) {
        skyDomeMat.uniforms.uSunIntensity.value = 0.0
      }
    }
  }

  const updateWallboards = (state: WallboardState) => {
    hologram.updateState({
      primaryTicker: state.primaryTicker,
      activeStageLabel: state.activeStageLabel,
      finalRating: state.finalRating,
      stancesBySpeaker: state.stancesBySpeaker,
      candles: state.candles,
    })

    const sig = JSON.stringify({
      t: state.primaryTicker,
      p: state.portfolioLabel,
      r: state.finalRating,
      s: state.activeStageLabel,
      st: state.stancesBySpeaker,
      en: state.metas.map((m) => `${m.speakerId}:${m.enabled}:${m.shortName}`),
      pc: state.pillarCoverage?.pillars?.map((x) => `${x.pillar}:${x.is_challenged}:${x.raised_by.length}`),
    })
    if (sig === lastScreenSignature) return
    lastScreenSignature = sig

    const mainCtx = mainScreenCanvas.getContext('2d')
    if (mainCtx) {
      drawMainWallCanvas(mainCtx, state)
      mainScreenTex.needsUpdate = true
    }
    const boardCtx = boardScreenCanvas.getContext('2d')
    if (boardCtx) {
      drawBoardWallCanvas(boardCtx, state)
      boardScreenTex.needsUpdate = true
    }
  }

  const animateRoomEffects = (
    elapsedSec: number,
    isCrossfire: boolean,
    isSandboxMathActive: boolean,
    reducedMotion: boolean,
    voiceEnergy = 0,
  ) => {
    // 1. Centerpiece 3D Candlesticks / Trendline / Bull-Bear Arrows Hologram
    hologram.animate(elapsedSec, isCrossfire, isSandboxMathActive, reducedMotion, voiceEnergy)
    const baseRim = activeLightingMode === 'day' ? 1.1 : 1.45
    rimLight.intensity = baseRim + (reducedMotion ? 0 : voiceEnergy * 0.65)

    // 2. Holotable Neon Rim subtle breathing
    neonRingMat.color.set(isCrossfire ? '#F59E0B' : '#22D3EE')

    // 3. Crossfire Octagonal Duel Arena Pulse & Counter-Rotating Inner Arc
    const duelMat = duelRingMesh.material as THREE.MeshBasicMaterial
    const duelInnerMat = duelInnerRingMesh.material as THREE.MeshBasicMaterial
    if (isCrossfire) {
      duelMat.color.set('#EF4444')
      duelMat.opacity = reducedMotion ? 0.85 : 0.55 + Math.sin(elapsedSec * 6) * 0.3
      const s = reducedMotion ? 1.05 : 1.0 + Math.sin(elapsedSec * 5) * 0.06
      duelRingMesh.scale.set(s, s, 1)
      duelInnerMat.color.set('#F59E0B')
      duelInnerMat.opacity = 0.78
      if (!reducedMotion) {
        duelInnerRingMesh.rotation.z = -elapsedSec * 1.15
      }
    } else {
      duelMat.color.set('#22D3EE')
      duelMat.opacity = 0.52
      duelRingMesh.scale.set(1, 1, 1)
      duelInnerMat.color.set('#67E8F9')
      duelInnerMat.opacity = 0.42
      if (!reducedMotion) {
        duelInnerRingMesh.rotation.z = elapsedSec * 0.35
      }
    }

    // Conduit traces glow slightly brighter during Sandbox Math or Crossfire
    const baseConduitAlpha = activeLightingMode === 'day' ? 0.38 : 0.56
    conduitMat.opacity = isSandboxMathActive
      ? Math.min(0.92, baseConduitAlpha + 0.25)
      : baseConduitAlpha
    conduitMat.color.set(isCrossfire ? '#F59E0B' : '#38BDF8')

    // 4. Quant Server Telemetry Chase
    quantLedMeshes.forEach((led, idx) => {
      const m = led.material as THREE.MeshStandardMaterial
      if (isSandboxMathActive && !reducedMotion) {
        const wave = Math.sin(elapsedSec * 12 + idx * 1.1) > 0
        m.color.set(wave ? '#10B981' : '#22D3EE')
        m.emissive.set(wave ? '#10B981' : '#22D3EE')
        m.emissiveIntensity = wave ? 0.95 : 0.35
      } else {
        m.color.set('#22D3EE')
        m.emissive.set('#22D3EE')
        m.emissiveIntensity = isSandboxMathActive ? 0.78 : 0.35
      }
    })

    // 5. Tactile Arcade Control Console Button Sequencer
    if (!reducedMotion) {
      arcadeButtonMeshes.forEach((btn, idx) => {
        const m = btn.material as THREE.MeshStandardMaterial
        const pulse = Math.sin(elapsedSec * (isCrossfire ? 5.2 : 2.6) + idx * 0.85)
        m.emissiveIntensity = 0.38 + Math.max(0, pulse) * 0.65
      })

      // 6. Spinning Guru Desk Props (Quality Crystal Gem, Macro Globe, Skeptic Radar)
      spinningProps.forEach((prop, idx) => {
        prop.rotation.y = elapsedSec * (0.75 + idx * 0.15)
      })

      // 7. Continuous Eye-Level Cloud Drift, Sunlit Floor/Beam Modulation, Dust Motes & Beacons
      if (cloudGroup.visible) {
        for (const dc of driftingClouds) {
          const totalSpan = dc.span * 2
          if (dc.axis === 'x') {
            const wrappedX = (((dc.baseX + dc.span + elapsedSec * dc.speed) % totalSpan) + totalSpan) % totalSpan - dc.span
            dc.mesh.position.x = wrappedX
          } else {
            const wrappedZ = (((dc.baseZ + dc.span + elapsedSec * dc.speed) % totalSpan) + totalSpan) % totalSpan - dc.span
            dc.mesh.position.z = wrappedZ
          }
          dc.mesh.position.y = dc.baseY + Math.sin(elapsedSec * 0.55 + dc.phase) * 0.14
        }
      }
      if (sunGroup.visible && activeLightingMode === 'day') {
        const cloudPassFactor = 0.90 + 0.10 * Math.sin(elapsedSec * 0.26)
        keyLight.intensity = 1.48 * cloudPassFactor
        if (sunbeamMat.uniforms.uTime) sunbeamMat.uniforms.uTime.value = elapsedSec
        if (sunbeamMat.uniforms.uCloudFactor) sunbeamMat.uniforms.uCloudFactor.value = cloudPassFactor
        sunPoolMat.opacity = 0.72 * cloudPassFactor
        windowWashMat.opacity = 0.38 * cloudPassFactor

        for (let i = 0; i < dustSeeds.length; i += 1) {
          const ds = dustSeeds[i]!
          const lift = ((elapsedSec * ds.speed * 0.18 + ds.phase) % 2.4)
          dustDummy.position.set(
            ds.x + Math.sin(elapsedSec * 0.5 + ds.phase) * 0.22,
            ds.y + lift * 0.35,
            ds.z + Math.cos(elapsedSec * 0.45 + ds.phase) * 0.22,
          )
          const pulse = ds.scale * (0.75 + 0.35 * Math.sin(elapsedSec * 1.4 + ds.phase))
          dustDummy.scale.setScalar(pulse)
          dustDummy.rotation.set(elapsedSec * 0.4, elapsedSec * 0.6, 0)
          dustDummy.updateMatrix()
          dustInstanced.setMatrixAt(i, dustDummy.matrix)
        }
        dustInstanced.instanceMatrix.needsUpdate = true
      }
      beaconMeshes.forEach((beacon, idx) => {
        const pulse = Math.sin(elapsedSec * 3.0 + idx * 1.4) > 0.2 ? 1.15 : 0.7
        beacon.scale.setScalar(pulse)
      })
    }
  }

  const dispose = () => {
    hologram.dispose()
    pedestalShadowTex.dispose()
    sunPoolTex.dispose()
    dayFloorSet.map.dispose()
    dayFloorSet.roughnessMap.dispose()
    dayFloorSet.bumpMap.dispose()
    dayFloorSet.emissiveMap.dispose()
    twilightFloorSet.map.dispose()
    twilightFloorSet.roughnessMap.dispose()
    twilightFloorSet.bumpMap.dispose()
    twilightFloorSet.emissiveMap.dispose()
    windowWashTex.dispose()
    plazaDecalTex.dispose()
    duelArenaTex.dispose()
    ceoFloorTex.dispose()
    loungeRugTex.dispose()
    daySkyscraperTex.dispose()
    nightSkyscraperTex.dispose()
    monitorTex.dispose()
    mainScreenTex.dispose()
    boardScreenTex.dispose()
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
    ambientLight,
    hemiLight,
    keyLight,
    rimLight,
    floorMesh,
    carpetMesh,
    carpetBorderMesh,
    duelRingMesh,
    wallMeshes,
    skylineGroup,
    quantLedMeshes,
    monitorGlowMeshes,
    hologram,
    applyLightingTheme,
    updateWallboards,
    animateRoomEffects,
    dispose,
  }
}
