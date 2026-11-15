import * as THREE from 'three'
import type { SpeakerVisualMeta } from './personas'
import type { RobloxAvatarPose } from './roundtableChoreography'

export type RobloxFaceMood = 'normal' | 'speaking' | 'thinking' | 'blink'

export interface RobloxR6AvatarHandle {
  speakerId: string
  root: THREE.Group
  bodyRoot: THREE.Group
  headPivot: THREE.Group
  leftArmPivot: THREE.Group
  rightArmPivot: THREE.Group
  leftLegPivot: THREE.Group
  rightLegPivot: THREE.Group
  headAnchor: THREE.Object3D
  statusRing: THREE.Mesh
  contactShadow: THREE.Mesh
  chestEmblem?: THREE.Mesh
  faceMaterial: THREE.MeshStandardMaterial
  faceTextures: Record<RobloxFaceMood, THREE.CanvasTexture>
  currentMood: RobloxFaceMood
}

const SKIN_HEX = '#F6CF58' // Classic warm Roblox Stud Yellow-Gold

/** Safely disposes an R6 avatar subtree along with all cached facial mood textures */
export function disposeRobloxR6Avatar(avatar: RobloxR6AvatarHandle): void {
  avatar.faceMaterial.map = null
  for (const tex of Object.values(avatar.faceTextures)) {
    tex.dispose()
  }
  disposeObjectSubtree(avatar.root)
}

/** Safely disposes every geometry, material, and texture under a Three.js subtree */
export function disposeObjectSubtree(root: THREE.Object3D): void {
  root.traverse((obj) => {
    const mesh = obj as THREE.Mesh
    if (mesh.geometry) {
      mesh.geometry.dispose()
    }
    const mat = mesh.material
    if (Array.isArray(mat)) {
      for (const m of mat) disposeMaterial(m)
    } else if (mat) {
      disposeMaterial(mat)
    }
  })
}

function disposeMaterial(mat: THREE.Material): void {
  const std = mat as THREE.MeshStandardMaterial & { map?: THREE.Texture | null }
  if (std.map) std.map.dispose()
  mat.dispose()
}

function createRobloxFaceTexture(mood: RobloxFaceMood, accentHex: string): THREE.CanvasTexture {
  const canvas = document.createElement('canvas')
  canvas.width = 256
  canvas.height = 128
  const ctx = canvas.getContext('2d')
  if (ctx) {
    // Base Roblox skin fill across entire wrap cylinder
    ctx.fillStyle = SKIN_HEX
    ctx.fillRect(0, 0, canvas.width, canvas.height)

    // Subtle rosy cheek warmth (gives crisp stylized look against camera lights)
    ctx.fillStyle = 'rgba(239, 108, 87, 0.16)'
    ctx.beginPath()
    ctx.ellipse(98, 76, 8, 4.5, 0, 0, Math.PI * 2)
    ctx.ellipse(158, 76, 8, 4.5, 0, 0, Math.PI * 2)
    ctx.fill()

    // Classic glossy Roblox oval eyes centered at x=128 (or natural closed blink lines)
    if (mood === 'blink') {
      ctx.strokeStyle = '#182230'
      ctx.lineWidth = 3.2
      ctx.lineCap = 'round'
      for (const cx of [111, 145]) {
        ctx.beginPath()
        ctx.moveTo(cx - 7, 57)
        ctx.quadraticCurveTo(cx, 62, cx + 7, 57)
        ctx.stroke()
      }
    } else {
      const drawEye = (cx: number, cy: number) => {
        ctx.fillStyle = '#182230'
        ctx.beginPath()
        ctx.ellipse(cx, cy, 5.5, 8.2, 0, 0, Math.PI * 2)
        ctx.fill()
        ctx.fillStyle = '#FFFFFF'
        ctx.beginPath()
        ctx.arc(cx - 1.8, cy - 3, 2.0, 0, Math.PI * 2)
        ctx.fill()
      }
      drawEye(111, 56)
      drawEye(145, 56)
    }

    // Eyebrows vary slightly when analyzing ("thinking")
    ctx.strokeStyle = '#33291A'
    ctx.lineWidth = 3.2
    ctx.lineCap = 'round'
    if (mood === 'thinking') {
      ctx.beginPath()
      ctx.moveTo(103, 42)
      ctx.lineTo(118, 45)
      ctx.moveTo(138, 44)
      ctx.lineTo(153, 40)
      ctx.stroke()
    } else {
      ctx.beginPath()
      ctx.moveTo(104, 43)
      ctx.lineTo(117, 42)
      ctx.moveTo(139, 42)
      ctx.lineTo(152, 43)
      ctx.stroke()
    }

    // Mouth expression
    if (mood === 'speaking') {
      ctx.fillStyle = '#7F1D1D'
      ctx.strokeStyle = '#182230'
      ctx.lineWidth = 2.6
      ctx.beginPath()
      ctx.ellipse(128, 82, 10, 7.5, 0, 0, Math.PI * 2)
      ctx.fill()
      ctx.stroke()
      ctx.fillStyle = '#F8FAFC'
      ctx.fillRect(122, 76, 12, 3)
    } else if (mood === 'thinking') {
      ctx.strokeStyle = '#182230'
      ctx.lineWidth = 3
      ctx.beginPath()
      ctx.moveTo(120, 82)
      ctx.quadraticCurveTo(128, 79, 136, 82)
      ctx.stroke()
    } else {
      // Classic upbeat Roblox smile
      ctx.strokeStyle = '#182230'
      ctx.lineWidth = 3.2
      ctx.beginPath()
      ctx.arc(128, 74, 11, 0.28 * Math.PI, 0.72 * Math.PI)
      ctx.stroke()
    }

    // Back-of-head subtle collar hairline tint so back angles look tidy
    ctx.fillStyle = accentHex
    ctx.globalAlpha = 0.08
    ctx.fillRect(0, 118, 256, 10)
    ctx.globalAlpha = 1
  }

  const tex = new THREE.CanvasTexture(canvas)
  tex.colorSpace = THREE.SRGBColorSpace
  tex.needsUpdate = true
  return tex
}

function createContactShadowTexture(): THREE.CanvasTexture {
  const canvas = document.createElement('canvas')
  canvas.width = 64
  canvas.height = 64
  const ctx = canvas.getContext('2d')
  if (ctx) {
    const grad = ctx.createRadialGradient(32, 32, 4, 32, 32, 30)
    grad.addColorStop(0, 'rgba(2, 6, 23, 0.45)')
    grad.addColorStop(0.55, 'rgba(2, 6, 23, 0.20)')
    grad.addColorStop(1, 'rgba(2, 6, 23, 0.0)')
    ctx.fillStyle = grad
    ctx.fillRect(0, 0, 64, 64)
  }
  const tex = new THREE.CanvasTexture(canvas)
  tex.colorSpace = THREE.SRGBColorSpace
  tex.needsUpdate = true
  return tex
}

function createSuitFrontTexture(accentHex: string, speakerId: string): THREE.CanvasTexture {
  const canvas = document.createElement('canvas')
  canvas.width = 128
  canvas.height = 160
  const ctx = canvas.getContext('2d')
  if (ctx) {
    const jacketColor =
      speakerId === 'skeptic_guru'
        ? '#1E293B'
        : speakerId === 'growth_guru'
          ? '#B45309'
          : speakerId === 'user'
            ? '#0C4A6E'
            : '#1E3A5F'

    ctx.fillStyle = jacketColor
    ctx.fillRect(0, 0, 128, 160)

    // Crisp white dress shirt V-neck
    ctx.fillStyle = '#F8FAFC'
    ctx.beginPath()
    ctx.moveTo(42, 0)
    ctx.lineTo(86, 0)
    ctx.lineTo(64, 92)
    ctx.closePath()
    ctx.fill()

    // Accent tie matching speaker persona
    ctx.fillStyle = accentHex
    ctx.beginPath()
    ctx.moveTo(59, 6)
    ctx.lineTo(69, 6)
    ctx.lineTo(72, 74)
    ctx.lineTo(64, 88)
    ctx.lineTo(56, 74)
    ctx.closePath()
    ctx.fill()

    // Suit lapel trim
    ctx.fillStyle = accentHex
    ctx.fillRect(16, 18, 16, 6) // pocket square / emblem strip
    ctx.fillStyle = '#0F172A'
    ctx.fillRect(0, 146, 128, 14) // Belt band
    ctx.fillStyle = '#FBBF24'
    ctx.fillRect(56, 148, 16, 10) // Buckle
  }
  const tex = new THREE.CanvasTexture(canvas)
  tex.colorSpace = THREE.SRGBColorSpace
  tex.needsUpdate = true
  return tex
}

function buildAccessoryProps(
  speakerId: string,
  accentColor: string,
  headPivot: THREE.Group,
  torsoGroup: THREE.Group,
  rightArmPivot: THREE.Group,
): { chestEmblem?: THREE.Mesh } {
  const accentMat = new THREE.MeshStandardMaterial({
    color: accentColor,
    roughness: 0.3,
    metalness: 0.25,
    emissive: accentColor,
    emissiveIntensity: 0.16,
  })
  const darkMat = new THREE.MeshStandardMaterial({ color: '#1E293B', roughness: 0.45 })
  const goldMat = new THREE.MeshStandardMaterial({
    color: '#FBBF24',
    roughness: 0.25,
    metalness: 0.65,
  })
  const hairBrownMat = new THREE.MeshStandardMaterial({ color: '#452C1E', roughness: 0.65 })
  const hairSilverMat = new THREE.MeshStandardMaterial({ color: '#CBD5E1', roughness: 0.55 })

  const addHeadCap = (mat: THREE.Material) => {
    const cap = new THREE.Mesh(new THREE.BoxGeometry(0.50, 0.13, 0.50), mat)
    cap.position.set(0, 0.43, 0)
    cap.castShadow = true
    headPivot.add(cap)
    const crest = new THREE.Mesh(new THREE.BoxGeometry(0.36, 0.08, 0.42), mat)
    crest.position.set(0, 0.51, -0.02)
    headPivot.add(crest)
  }

  const addGlasses = (frameMat: THREE.Material) => {
    const bridge = new THREE.Mesh(new THREE.BoxGeometry(0.38, 0.035, 0.04), frameMat)
    bridge.position.set(0, 0.24, 0.255)
    headPivot.add(bridge)
    const leftRim = new THREE.Mesh(new THREE.BoxGeometry(0.14, 0.10, 0.05), frameMat)
    leftRim.position.set(-0.10, 0.22, 0.25)
    const rightRim = leftRim.clone()
    rightRim.position.x = 0.10
    headPivot.add(leftRim, rightRim)
  }

  let chestEmblem: THREE.Mesh | undefined

  switch (speakerId) {
    case 'hermes_moderator': {
      // Sleek Roblox headphones with golden Hermes wings + handheld mic
      const band = new THREE.Mesh(new THREE.BoxGeometry(0.58, 0.08, 0.14), accentMat)
      band.position.set(0, 0.48, 0)
      headPivot.add(band)
      for (const side of [-1, 1]) {
        const ear = new THREE.Mesh(new THREE.CylinderGeometry(0.11, 0.11, 0.08, 12), accentMat)
        ear.rotation.z = Math.PI / 2
        ear.position.set(side * 0.29, 0.24, 0)
        const wing = new THREE.Mesh(new THREE.BoxGeometry(0.05, 0.20, 0.16), goldMat)
        wing.position.set(side * 0.34, 0.34, -0.04)
        wing.rotation.z = side * -0.25
        headPivot.add(ear, wing)
      }
      const micHandle = new THREE.Mesh(new THREE.CylinderGeometry(0.03, 0.025, 0.22, 8), darkMat)
      micHandle.position.set(0, -0.62, 0.12)
      micHandle.rotation.x = Math.PI / 3
      const micHead = new THREE.Mesh(new THREE.SphereGeometry(0.055, 10, 10), goldMat)
      micHead.position.set(0, -0.57, 0.21)
      rightArmPivot.add(micHandle, micHead)
      break
    }

    case 'value_guru': {
      addHeadCap(hairSilverMat)
      addGlasses(darkMat)
      // Ledger badge on left side
      const ledger = new THREE.Mesh(new THREE.BoxGeometry(0.10, 0.28, 0.22), accentMat)
      ledger.position.set(-0.33, -0.08, 0.04)
      torsoGroup.add(ledger)
      break
    }

    case 'quality_guru': {
      addHeadCap(hairBrownMat)
      addGlasses(goldMat)
      // Spinning 3D Moat Diamond floating slightly proud of chest
      const diamond = new THREE.Mesh(
        new THREE.OctahedronGeometry(0.12, 0),
        new THREE.MeshStandardMaterial({
          color: '#A5B4FC',
          emissive: '#6366F1',
          emissiveIntensity: 0.45,
          roughness: 0.15,
          metalness: 0.3,
        }),
      )
      diamond.position.set(0, 0.10, 0.22)
      torsoGroup.add(diamond)
      chestEmblem = diamond
      break
    }

    case 'growth_guru': {
      // Spiky blonde-gold top + cyberpunk visor + twin Roblox jetpack boosters
      addHeadCap(goldMat)
      const visor = new THREE.Mesh(
        new THREE.BoxGeometry(0.42, 0.11, 0.08),
        new THREE.MeshStandardMaterial({
          color: '#FBBF24',
          emissive: '#F59E0B',
          emissiveIntensity: 0.35,
          transparent: true,
          opacity: 0.78,
        }),
      )
      visor.position.set(0, 0.23, 0.24)
      headPivot.add(visor)
      for (const side of [-1, 1]) {
        const tank = new THREE.Mesh(new THREE.CylinderGeometry(0.08, 0.09, 0.42, 10), accentMat)
        tank.position.set(side * 0.13, 0.04, -0.22)
        const nozzle = new THREE.Mesh(new THREE.CylinderGeometry(0.05, 0.08, 0.08, 10), darkMat)
        nozzle.position.set(side * 0.13, -0.20, -0.22)
        torsoGroup.add(tank, nozzle)
      }
      break
    }

    case 'macro_tactical_guru': {
      // Modern cap + antenna beacon
      const brim = new THREE.Mesh(new THREE.BoxGeometry(0.52, 0.06, 0.62), accentMat)
      brim.position.set(0, 0.40, 0.06)
      const crown = new THREE.Mesh(new THREE.BoxGeometry(0.48, 0.16, 0.48), darkMat)
      crown.position.set(0, 0.49, 0)
      const antenna = new THREE.Mesh(new THREE.CylinderGeometry(0.02, 0.02, 0.22, 6), accentMat)
      antenna.position.set(0.22, 0.56, -0.10)
      headPivot.add(brim, crown, antenna)
      break
    }

    case 'skeptic_guru': {
      // Sleek detective fedora + Kill-Switch Shield on back
      const brim = new THREE.Mesh(new THREE.CylinderGeometry(0.36, 0.36, 0.045, 16), darkMat)
      brim.position.set(0, 0.41, 0)
      const hatTop = new THREE.Mesh(new THREE.CylinderGeometry(0.22, 0.25, 0.20, 16), darkMat)
      hatTop.position.set(0, 0.52, 0)
      const hatBand = new THREE.Mesh(new THREE.CylinderGeometry(0.255, 0.255, 0.05, 16), accentMat)
      hatBand.position.set(0, 0.45, 0)
      headPivot.add(brim, hatTop, hatBand)

      const shieldBase = new THREE.Mesh(new THREE.BoxGeometry(0.46, 0.54, 0.07), darkMat)
      shieldBase.position.set(0, 0.04, -0.20)
      const shieldCore = new THREE.Mesh(new THREE.BoxGeometry(0.32, 0.38, 0.085), accentMat)
      shieldCore.position.set(0, 0.04, -0.20)
      torsoGroup.add(shieldBase, shieldCore)
      chestEmblem = shieldCore
      break
    }

    case 'user':
    default: {
      // Golden Roblox VIP Owner Crown + Hair
      addHeadCap(darkMat)
      const crownBase = new THREE.Mesh(new THREE.CylinderGeometry(0.22, 0.20, 0.12, 8), goldMat)
      crownBase.position.set(0, 0.56, 0)
      headPivot.add(crownBase)
      for (let i = 0; i < 4; i += 1) {
        const ang = (i * Math.PI) / 2
        const spike = new THREE.Mesh(new THREE.ConeGeometry(0.055, 0.12, 4), goldMat)
        spike.position.set(Math.cos(ang) * 0.15, 0.65, Math.sin(ang) * 0.15)
        headPivot.add(spike)
      }
      // Gold VIP shoulder pads
      for (const side of [-1, 1]) {
        const pad = new THREE.Mesh(new THREE.BoxGeometry(0.24, 0.07, 0.30), goldMat)
        pad.position.set(side * 0.40, 0.34, 0)
        torsoGroup.add(pad)
      }
      break
    }
  }

  return { chestEmblem }
}

/** Builds a classic Roblox R6 blocky character rig with 6 articulated limb pivots */
export function createRobloxR6Avatar(meta: SpeakerVisualMeta): RobloxR6AvatarHandle {
  const root = new THREE.Group()
  root.name = `avatar_${meta.speakerId}`
  root.userData = { speakerId: meta.speakerId }

  const bodyRoot = new THREE.Group()
  root.add(bodyRoot)

  const pantsMat = new THREE.MeshStandardMaterial({
    color: meta.speakerId === 'user' ? '#1E3A8A' : '#1E293B',
    roughness: 0.6,
    metalness: 0.08,
  })
  const sleeveMat = new THREE.MeshStandardMaterial({
    color:
      meta.speakerId === 'skeptic_guru'
        ? '#1E293B'
        : meta.speakerId === 'growth_guru'
          ? '#D97706'
          : meta.speakerId === 'user'
            ? '#0369A1'
            : '#254266',
    roughness: 0.42,
    metalness: 0.1,
  })
  const skinMat = new THREE.MeshStandardMaterial({
    color: SKIN_HEX,
    roughness: 0.45,
    metalness: 0.05,
  })

  // 1. Legs (hips at y = 0.72 so legs rest cleanly at y = 0)
  const makeLegPivot = (xOffset: number): THREE.Group => {
    const pivot = new THREE.Group()
    pivot.position.set(xOffset, 0.72, 0)
    const legMesh = new THREE.Mesh(new THREE.BoxGeometry(0.24, 0.72, 0.28), pantsMat)
    legMesh.position.set(0, -0.36, 0)
    legMesh.castShadow = true
    legMesh.receiveShadow = true
    legMesh.userData = { speakerId: meta.speakerId }
    pivot.add(legMesh)
    // Crisp dark shoes cuff
    const shoe = new THREE.Mesh(
      new THREE.BoxGeometry(0.25, 0.11, 0.31),
      new THREE.MeshStandardMaterial({ color: '#0F172A', roughness: 0.55 }),
    )
    shoe.position.set(0, -0.665, 0.015)
    shoe.userData = { speakerId: meta.speakerId }
    pivot.add(shoe)
    return pivot
  }

  const leftLegPivot = makeLegPivot(-0.14)
  const rightLegPivot = makeLegPivot(0.14)
  bodyRoot.add(leftLegPivot, rightLegPivot)

  // 2. Torso (blocky R6 chest with suit front CanvasTexture + uniform accent back/sides)
  const torsoGroup = new THREE.Group()
  torsoGroup.position.set(0, 1.08, 0)
  const frontTex = createSuitFrontTexture(meta.accentColor, meta.speakerId)
  const frontMat = new THREE.MeshStandardMaterial({
    map: frontTex,
    roughness: 0.4,
    metalness: 0.1,
  })
  const torsoMaterials = [
    sleeveMat, // +x
    sleeveMat, // -x
    sleeveMat, // +y
    pantsMat, // -y
    frontMat, // +z (chest front)
    sleeveMat, // -z (back)
  ]
  const torsoMesh = new THREE.Mesh(new THREE.BoxGeometry(0.54, 0.72, 0.30), torsoMaterials)
  torsoMesh.castShadow = true
  torsoMesh.receiveShadow = true
  torsoMesh.userData = { speakerId: meta.speakerId }
  torsoGroup.add(torsoMesh)

  // Classic Roblox back stud emblem strip in speaker's accent color
  const badgeBack = new THREE.Mesh(
    new THREE.BoxGeometry(0.28, 0.18, 0.03),
    new THREE.MeshStandardMaterial({
      color: meta.accentColor,
      emissive: meta.accentColor,
      emissiveIntensity: 0.2,
      roughness: 0.3,
    }),
  )
  badgeBack.position.set(0, 0.10, -0.155)
  torsoGroup.add(badgeBack)
  bodyRoot.add(torsoGroup)

  // 3. Arms (shoulders at y = 1.40)
  const makeArmPivot = (xOffset: number): THREE.Group => {
    const pivot = new THREE.Group()
    pivot.position.set(xOffset, 1.40, 0)
    const upperArm = new THREE.Mesh(new THREE.BoxGeometry(0.22, 0.50, 0.24), sleeveMat)
    upperArm.position.set(0, -0.25, 0)
    upperArm.castShadow = true
    upperArm.userData = { speakerId: meta.speakerId }
    const hand = new THREE.Mesh(new THREE.BoxGeometry(0.21, 0.18, 0.22), skinMat)
    hand.position.set(0, -0.59, 0)
    hand.castShadow = true
    hand.userData = { speakerId: meta.speakerId }
    pivot.add(upperArm, hand)
    return pivot
  }

  const leftArmPivot = makeArmPivot(-0.40)
  const rightArmPivot = makeArmPivot(0.40)
  bodyRoot.add(leftArmPivot, rightArmPivot)

  // 4. Head (neck at y = 1.44; signature Roblox rounded-cap cylinder + bevels)
  const headPivot = new THREE.Group()
  headPivot.position.set(0, 1.44, 0)

  const faceTextures: Record<RobloxFaceMood, THREE.CanvasTexture> = {
    normal: createRobloxFaceTexture('normal', meta.accentColor),
    speaking: createRobloxFaceTexture('speaking', meta.accentColor),
    thinking: createRobloxFaceTexture('thinking', meta.accentColor),
    blink: createRobloxFaceTexture('blink', meta.accentColor),
  }
  const faceMaterial = new THREE.MeshStandardMaterial({
    map: faceTextures.normal,
    roughness: 0.42,
    metalness: 0.05,
  })

  const headCyl = new THREE.Mesh(new THREE.CylinderGeometry(0.245, 0.245, 0.34, 24), [
    faceMaterial,
    skinMat,
    skinMat,
  ])
  headCyl.position.set(0, 0.24, 0)
  // Rotate π so canvas center (u=0.5) faces avatar forward (+Z)
  headCyl.rotation.y = Math.PI
  headCyl.castShadow = true
  headCyl.userData = { speakerId: meta.speakerId }

  const headTopBevel = new THREE.Mesh(
    new THREE.CylinderGeometry(0.18, 0.245, 0.07, 24),
    skinMat,
  )
  headTopBevel.position.set(0, 0.445, 0)
  const headBottomBevel = new THREE.Mesh(
    new THREE.CylinderGeometry(0.245, 0.18, 0.06, 24),
    skinMat,
  )
  headBottomBevel.position.set(0, 0.04, 0)
  const legoStud = new THREE.Mesh(
    new THREE.CylinderGeometry(0.10, 0.10, 0.04, 14),
    skinMat,
  )
  legoStud.position.set(0, 0.495, 0)

  headPivot.add(headCyl, headTopBevel, headBottomBevel, legoStud)
  bodyRoot.add(headPivot)

  // 5. Character-specific hats, glasses & gadgets
  const { chestEmblem } = buildAccessoryProps(
    meta.speakerId,
    meta.accentColor,
    headPivot,
    torsoGroup,
    rightArmPivot,
  )

  // 6. Soft grounded contact shadow decal under feet + Speaking / Thinking ForceField Ring
  const contactShadow = new THREE.Mesh(
    new THREE.PlaneGeometry(0.95, 0.95),
    new THREE.MeshBasicMaterial({
      map: createContactShadowTexture(),
      transparent: true,
      depthWrite: false,
    }),
  )
  contactShadow.rotation.x = -Math.PI / 2
  contactShadow.position.set(0, 0.022, 0)
  root.add(contactShadow)

  const statusRing = new THREE.Mesh(
    new THREE.RingGeometry(0.46, 0.62, 32),
    new THREE.MeshBasicMaterial({
      color: meta.accentColor,
      side: THREE.DoubleSide,
      transparent: true,
      opacity: 0.85,
    }),
  )
  statusRing.rotation.x = -Math.PI / 2
  statusRing.position.set(0, 0.045, 0)
  statusRing.visible = false
  root.add(statusRing)

  // 7. 3D-to-2D Screen Projection anchor cleanly above crown/hat
  const headAnchor = new THREE.Object3D()
  headAnchor.position.set(0, 2.28, 0)
  root.add(headAnchor)

  return {
    speakerId: meta.speakerId,
    root,
    bodyRoot,
    headPivot,
    leftArmPivot,
    rightArmPivot,
    leftLegPivot,
    rightLegPivot,
    headAnchor,
    statusRing,
    contactShadow,
    chestEmblem,
    faceMaterial,
    faceTextures,
    currentMood: 'normal',
  }
}

/** Animates Roblox R6 joints smoothly at 60 FPS with pose blending, eye blinking & voice lip-sync */
export function animateRobloxR6Avatar(
  rig: RobloxR6AvatarHandle,
  pose: RobloxAvatarPose,
  elapsedSec: number,
  isSpeaking: boolean,
  isThinking: boolean,
  reducedMotion: boolean,
  headLookYaw = 0,
  dt = 0.016,
  voiceEnergy = 0,
): void {
  const speakerSeed = rig.speakerId.split('').reduce((acc, ch) => acc + ch.charCodeAt(0), 0) % 17
  const blinkCycle = (elapsedSec + speakerSeed * 0.43) % 4.1
  const isBlinking = !reducedMotion && blinkCycle > 3.94

  const speechGate =
    voiceEnergy > 0.02 ? voiceEnergy > 0.14 : Math.sin(elapsedSec * 12) > -0.15
  const desiredMood: RobloxFaceMood = isSpeaking
    ? !reducedMotion && speechGate
      ? 'speaking'
      : isBlinking
        ? 'blink'
        : 'normal'
    : isBlinking
      ? 'blink'
      : isThinking
        ? 'thinking'
        : 'normal'

  if (desiredMood !== rig.currentMood) {
    rig.currentMood = desiredMood
    rig.faceMaterial.map = rig.faceTextures[desiredMood]
    rig.faceMaterial.needsUpdate = true
  }

  // Toggle underfoot Neon status halo (active ONLY for speaking/thinking) and pulse to voice audio energy
  rig.statusRing.visible = isSpeaking || isThinking
  if (rig.statusRing.visible) {
    const audioBoost = isSpeaking && !reducedMotion ? voiceEnergy * 0.16 : 0
    const pulse = reducedMotion ? 1.0 : 0.94 + Math.sin(elapsedSec * 5.2) * 0.08 + audioBoost
    rig.statusRing.scale.set(pulse, pulse, 1)
  }

  if (rig.chestEmblem && rig.speakerId === 'quality_guru' && !reducedMotion) {
    rig.chestEmblem.rotation.y = elapsedSec * 1.4
  }

  const phase = reducedMotion ? 0 : elapsedSec
  const audioAmp = !reducedMotion && isSpeaking ? voiceEnergy : 0

  let targetBodyY = 0
  let targetLegL = 0
  let targetLegR = 0
  let targetArmLX = 0
  let targetArmLZ = 0.05
  let targetArmRX = 0
  let targetArmRZ = -0.05
  let targetHeadX = 0
  let targetHeadY = 0

  switch (pose) {
    case 'walk': {
      const swing = Math.sin(phase * 9.5) * 0.62
      targetBodyY = Math.abs(Math.cos(phase * 9.5)) * 0.055
      targetLegL = swing
      targetLegR = -swing
      targetArmLX = -swing * 0.85
      targetArmLZ = 0.04
      targetArmRX = swing * 0.85
      targetArmRZ = -0.04
      break
    }

    case 'sit_desk':
    case 'sit_table':
    case 'lounge': {
      const idleBreath = reducedMotion ? 0 : Math.sin(phase * 2.1 + speakerSeed) * 0.012
      targetBodyY = -0.25 + idleBreath
      targetLegL = -Math.PI / 2.05
      targetLegR = -Math.PI / 2.05
      if (pose === 'sit_desk') {
        const bob = Math.sin(phase * 2.2) * 0.05
        targetArmLX = -0.78 + bob
        targetArmLZ = 0.04
        targetArmRX = -0.78 - bob
        targetArmRZ = -0.04
        targetHeadX = 0.06
        targetHeadY = headLookYaw * 0.4
      } else if (pose === 'lounge') {
        targetArmLX = -0.25
        targetArmLZ = 0.06
        targetArmRX = -0.25
        targetArmRZ = -0.06
        targetHeadX = 0.08
      } else {
        const scanYaw = reducedMotion ? 0 : Math.sin(phase * 1.1 + speakerSeed) * 0.07
        targetArmLX = -0.62
        targetArmRX = -0.62
        targetHeadX = 0.04
        targetHeadY = headLookYaw + scanYaw
      }
      break
    }

    case 'tap_terminal': {
      targetBodyY = -0.24
      targetLegL = -Math.PI / 2.05
      targetLegR = -Math.PI / 2.05
      const tapL = Math.sin(phase * 10.5) * 0.10
      const tapR = Math.cos(phase * 12.0) * 0.11
      targetArmLX = -0.78 + tapL
      targetArmLZ = -0.08
      targetArmRX = -0.84 + tapR
      targetArmRZ = 0.08
      targetHeadX = 0.18 + Math.sin(phase * 1.8) * 0.06
      targetHeadY = Math.sin(phase * 1.4) * 0.1
      break
    }

    case 'typing': {
      targetBodyY = -0.25
      targetLegL = -Math.PI / 2.05
      targetLegR = -Math.PI / 2.05
      targetArmLX = -0.95 + Math.sin(phase * 14) * 0.14
      targetArmRX = -0.95 + Math.cos(phase * 14) * 0.14
      targetArmLZ = 0.04
      targetArmRZ = -0.04
      targetHeadX = 0.14
      break
    }

    case 'holo_point': {
      targetBodyY = isSpeaking ? Math.abs(Math.sin(phase * 3.8)) * 0.03 + audioAmp * 0.025 : 0
      const pointSweep = Math.sin(phase * 4.4) * (0.14 + audioAmp * 0.16)
      targetArmRX = -1.45 + pointSweep
      targetArmRZ = -0.22
      targetArmLX = -0.68 - pointSweep * 0.25
      targetArmLZ = 0.10
      targetHeadX = -0.10
      targetHeadY = Math.sin(phase * 2.2) * 0.12
      break
    }

    case 'stand_talk':
    case 'duel': {
      targetBodyY = isSpeaking ? Math.abs(Math.sin(phase * 4.2)) * 0.032 + audioAmp * 0.025 : 0
      if (isSpeaking) {
        const gesture = Math.sin(phase * 5.8) * (0.32 + audioAmp * 0.24)
        targetArmRX = -1.15 + gesture
        targetArmRZ = -0.18
        targetArmLX = -0.45 - gesture * 0.35
        targetArmLZ = 0.10
        targetHeadY = Math.sin(phase * 2.6) * 0.09
      } else {
        targetArmLX = pose === 'duel' ? -0.35 : -0.12
        targetArmLZ = 0.08
        targetArmRX = pose === 'duel' ? -0.35 : -0.12
        targetArmRZ = -0.08
        targetHeadY = headLookYaw * 0.5
      }
      break
    }

    case 'calc': {
      targetArmLX = -1.05 + Math.sin(phase * 12) * 0.18
      targetArmRX = -1.05 + Math.cos(phase * 12) * 0.18
      targetArmLZ = 0.04
      targetArmRZ = -0.04
      targetHeadX = -0.05
      break
    }

    case 'jump_interject': {
      const jump = reducedMotion ? 0 : Math.abs(Math.sin(phase * 3.8)) * 0.32
      targetBodyY = jump
      targetLegL = jump * 0.3
      targetLegR = -jump * 0.3
      targetArmRX = -2.55 + Math.sin(phase * 6) * 0.18
      targetArmRZ = -0.22
      targetArmLX = -0.25
      targetArmLZ = 0.08
      break
    }

    case 'celebrate': {
      targetBodyY = reducedMotion ? 0 : Math.abs(Math.sin(phase * 4.5)) * 0.16
      targetArmLX = -2.65
      targetArmRX = -2.65
      targetArmLZ = 0.2
      targetArmRZ = -0.2
      targetHeadX = -0.08
      break
    }
  }

  // Smoothly blend joint rotations across poses (snaps cleanly if reducedMotion is enabled)
  const blend = reducedMotion || pose === 'walk' ? 1 : Math.min(1, Math.max(0.12, dt * 13))
  rig.bodyRoot.position.y = THREE.MathUtils.lerp(rig.bodyRoot.position.y, targetBodyY, blend)
  rig.leftLegPivot.rotation.x = THREE.MathUtils.lerp(rig.leftLegPivot.rotation.x, targetLegL, blend)
  rig.rightLegPivot.rotation.x = THREE.MathUtils.lerp(rig.rightLegPivot.rotation.x, targetLegR, blend)
  rig.leftArmPivot.rotation.x = THREE.MathUtils.lerp(rig.leftArmPivot.rotation.x, targetArmLX, blend)
  rig.leftArmPivot.rotation.z = THREE.MathUtils.lerp(rig.leftArmPivot.rotation.z, targetArmLZ, blend)
  rig.rightArmPivot.rotation.x = THREE.MathUtils.lerp(rig.rightArmPivot.rotation.x, targetArmRX, blend)
  rig.rightArmPivot.rotation.z = THREE.MathUtils.lerp(rig.rightArmPivot.rotation.z, targetArmRZ, blend)
  rig.headPivot.rotation.x = THREE.MathUtils.lerp(rig.headPivot.rotation.x, targetHeadX, blend)
  rig.headPivot.rotation.y = THREE.MathUtils.lerp(rig.headPivot.rotation.y, targetHeadY, blend)
}
