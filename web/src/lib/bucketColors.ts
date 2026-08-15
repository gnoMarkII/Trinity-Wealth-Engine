/**
 * Curated Vibrant Bucket Color Palette
 * 16 modern, harmonious UI colors suited for financial portfolio charts and allocation UI.
 */
export const VIBRANT_BUCKET_PALETTE: string[] = [
  '#3B82F6', // Blue (sky/blue 500)
  '#8B5CF6', // Purple (violet 500)
  '#06B6D4', // Cyan (cyan 500)
  '#10B981', // Emerald (emerald 500)
  '#F59E0B', // Amber (amber 500)
  '#EC4899', // Pink (pink 500)
  '#6366F1', // Indigo (indigo 500)
  '#14B8A6', // Teal (teal 500)
  '#F97316', // Orange (orange 500)
  '#7C3AED', // Violet (violet 600)
  '#84CC16', // Lime (lime 500)
  '#E11D48', // Rose (rose 600)
  '#0284C7', // Sky (sky 600)
  '#D97706', // Amber (amber 600)
  '#059669', // Emerald (emerald 600)
  '#4F46E5', // Indigo (indigo 600)
]

/**
 * Normalizes hex string to standard uppercase hex format (e.g. #3B82F6).
 */
export function normalizeHex(color: string | null | undefined): string {
  if (!color || typeof color !== 'string') return '#3B82F6'
  const trimmed = color.trim().toUpperCase()
  if (/^#[0-9A-F]{6}$/.test(trimmed)) return trimmed
  if (/^#[0-9A-F]{3}$/.test(trimmed)) {
    return `#${trimmed[1]}${trimmed[1]}${trimmed[2]}${trimmed[2]}${trimmed[3]}${trimmed[3]}`
  }
  return '#3B82F6'
}

/**
 * Converts Hex string (#RRGGBB) to HSL values [hue (0-360), saturation (0-100), lightness (0-100)]
 */
export function hexToHsl(hex: string): [number, number, number] {
  const norm = normalizeHex(hex)
  const r = parseInt(norm.substring(1, 3), 16) / 255
  const g = parseInt(norm.substring(3, 5), 16) / 255
  const b = parseInt(norm.substring(5, 7), 16) / 255

  const max = Math.max(r, g, b)
  const min = Math.min(r, g, b)
  let h = 0
  let s = 0
  const l = (max + min) / 2

  if (max !== min) {
    const d = max - min
    s = l > 0.5 ? d / (2 - max - min) : d / (max + min)
    switch (max) {
      case r:
        h = (g - b) / d + (g < b ? 6 : 0)
        break
      case g:
        h = (b - r) / d + 2
        break
      case b:
        h = (r - g) / d + 4
        break
    }
    h /= 6
  }

  return [h * 360, s * 100, l * 100]
}

/**
 * Converts HSL values [hue (0-360), saturation (0-100), lightness (0-100)] to Hex string (#RRGGBB)
 */
export function hslToHex(h: number, s: number, l: number): string {
  const hFrac = ((h % 360) + 360) % 360 / 360
  const sFrac = Math.min(100, Math.max(0, s)) / 100
  const lFrac = Math.min(100, Math.max(0, l)) / 100

  if (sFrac === 0) {
    const val = Math.round(lFrac * 255).toString(16).padStart(2, '0').toUpperCase()
    return `#${val}${val}${val}`
  }

  const q = lFrac < 0.5 ? lFrac * (1 + sFrac) : lFrac + sFrac - lFrac * sFrac
  const p = 2 * lFrac - q

  const hue2rgb = (t: number) => {
    let tNorm = t
    if (tNorm < 0) tNorm += 1
    if (tNorm > 1) tNorm -= 1
    if (tNorm < 1 / 6) return p + (q - p) * 6 * tNorm
    if (tNorm < 1 / 2) return q
    if (tNorm < 2 / 3) return p + (q - p) * (2 / 3 - tNorm) * 6
    return p
  }

  const r = Math.round(hue2rgb(hFrac + 1 / 3) * 255)
  const g = Math.round(hue2rgb(hFrac) * 255)
  const b = Math.round(hue2rgb(hFrac - 1 / 3) * 255)

  return `#${r.toString(16).padStart(2, '0').toUpperCase()}${g.toString(16).padStart(2, '0').toUpperCase()}${b.toString(16).padStart(2, '0').toUpperCase()}`
}

/**
 * Calculates a unique color that is not present in `existingColors`.
 * If all 16 palette colors are used, calculates a color with maximal hue distance on HSL (75% Saturation, 50% Lightness).
 */
export function getUniqueBucketColor(existingColors: (string | null | undefined)[]): string {
  const normExisting = new Set(existingColors.map(normalizeHex))

  // 1. Try finding unused color from static palette
  const unusedFromPalette = VIBRANT_BUCKET_PALETTE.find((c) => !normExisting.has(normalizeHex(c)))
  if (unusedFromPalette) return unusedFromPalette

  // 2. Fallback: Find maximal hue distance
  const existingHues = Array.from(normExisting).map((hex) => hexToHsl(hex)[0])
  if (existingHues.length === 0) return VIBRANT_BUCKET_PALETTE[0] ?? '#3B82F6'

  // Test candidate hues every 5 degrees to find max minimal distance
  let bestHue = 0
  let maxMinDist = -1

  for (let candidate = 0; candidate < 360; candidate += 5) {
    let minDist = 360
    for (const existingHue of existingHues) {
      const diff = Math.abs(candidate - existingHue)
      const dist = Math.min(diff, 360 - diff)
      if (dist < minDist) minDist = dist
    }
    if (minDist > maxMinDist) {
      maxMinDist = minDist
      bestHue = candidate
    }
  }

  return hslToHex(bestHue, 75, 50)
}

/**
 * Returns a randomized color from the palette that is NOT currently used by any bucket.
 */
export function getRandomizedBucketColor(
  existingColors: (string | null | undefined)[],
  currentColor?: string | null
): string {
  const normCurrent = currentColor ? normalizeHex(currentColor) : null
  const normExisting = new Set(existingColors.map(normalizeHex).filter((c) => c !== normCurrent))

  const availableFromPalette = VIBRANT_BUCKET_PALETTE.filter(
    (c) => !normExisting.has(normalizeHex(c)) && normalizeHex(c) !== normCurrent
  )

  if (availableFromPalette.length > 0) {
    const randomIndex = Math.floor(Math.random() * availableFromPalette.length)
    return availableFromPalette[randomIndex] ?? '#3B82F6'
  }

  // Fallback to maximal hue distance
  return getUniqueBucketColor(Array.from(normExisting))
}

/**
 * Randomizes colors for an entire array of bucket targets in one pass,
 * ensuring all items get distinct colors from the curated palette (or HSL fallback).
 */
export function randomizeAllBucketColors<T extends { color?: string | null }>(items: T[]): T[] {
  const assignedColors: string[] = []

  // Create a shuffled copy of palette
  const shuffledPalette = VIBRANT_BUCKET_PALETTE.slice().sort(() => Math.random() - 0.5)

  return items.map((item) => {
    let chosenColor: string
    if (shuffledPalette.length > 0) {
      chosenColor = shuffledPalette.pop()!
    } else {
      chosenColor = getUniqueBucketColor(assignedColors)
    }
    assignedColors.push(chosenColor)
    return {
      ...item,
      color: chosenColor,
    }
  })
}
