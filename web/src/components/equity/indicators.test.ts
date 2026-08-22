import { describe, it, expect, vi } from 'vitest'
import {
  calculateWilderRSI,
  calculateWilderATR,
  registerRSI14WithLevels,
  registerATR14,
  registerCorporateActionOverlay,
  register52WLevelOverlay,
  registerValuationLevelOverlay,
  registerInsiderOverlay,
  registerVWAPIndicator,
  registerAnalystTargetOverlay,
} from './indicators'

import * as klinecharts from 'klinecharts'


vi.mock('klinecharts', () => {
  return {
    registerIndicator: vi.fn(),
    getSupportedIndicators: vi.fn(() => []),
    registerOverlay: vi.fn(),
    getSupportedOverlays: vi.fn(() => []),
  }
})

describe('Wilder RSI Pure Mathematical Contract', () => {
  // Standard Wilder 20-point benchmark dataset
  const standardPrices = [
    44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.1, 45.42, 45.84, 46.08,
    45.89, 46.03, 45.61, 46.28, 46.28, 46.0, 46.03, 46.41, 46.22, 45.64,
  ].map((c) => ({ close: c }))

  it('preserves the 15-closes invariant (first 14 entries t < 14 are null)', () => {
    const result = calculateWilderRSI(standardPrices, 14)
    expect(result).toHaveLength(20)

    for (let i = 0; i < 14; i++) {
      expect(result[i]).toBeNull()
    }
    expect(result[14]).not.toBeNull()
  })

  it('returns all nulls if dataset length <= period', () => {
    const shortList = standardPrices.slice(0, 14)
    const result = calculateWilderRSI(shortList, 14)
    expect(result).toHaveLength(14)
    expect(result.every((v) => v === null)).toBe(true)
  })

  it('calculates exact Wilder RSI values at t=14, t=15, t=16 matching RMA formula', () => {
    const result = calculateWilderRSI(standardPrices, 14)

    // Expected benchmark values
    // t=14: Seed SMA AvgGain=3.34/14, AvgLoss=1.40/14 -> RS=2.385714 -> RSI=70.46413502
    expect(result[14]).toBeCloseTo(70.464135, 4)

    // t=15: Wilder RMA AvgGain=(3.34*13/14+0)/14, AvgLoss=(1.40*13/14+0.28)/14 -> RSI=66.24961855
    expect(result[15]).toBeCloseTo(66.24961855, 4)

    // t=16: Wilder RMA -> RSI=66.480938
    expect(result[16]).toBeCloseTo(66.480938, 4)
  })

  it('handles flat price edge case (50.0)', () => {
    const flatPrices = new Array(20).fill({ close: 100.0 })
    const result = calculateWilderRSI(flatPrices, 14)

    expect(result[14]).toBe(50.0)
    expect(result[15]).toBe(50.0)
  })

  it('handles all-up edge case (100.0)', () => {
    const allUp = Array.from({ length: 20 }, (_, i) => ({ close: 100.0 + i }))
    const result = calculateWilderRSI(allUp, 14)

    expect(result[14]).toBe(100.0)
    expect(result[15]).toBe(100.0)
  })

  it('handles all-down edge case (0.0)', () => {
    const allDown = Array.from({ length: 20 }, (_, i) => ({ close: 100.0 - i }))
    const result = calculateWilderRSI(allDown, 14)

    expect(result[14]).toBe(0.0)
    expect(result[15]).toBe(0.0)
  })

  it('registers RSI14_WITH_LEVELS indicator idempotently', () => {
    registerRSI14WithLevels()
    registerRSI14WithLevels() // Second call should be no-op

    expect(klinecharts.registerIndicator).toHaveBeenCalledTimes(1)
  })
})

describe('Wilder ATR Pure Mathematical Contract', () => {
  const sampleBars = [
    { high: 48.70, low: 47.79, close: 48.16 },
    { high: 48.72, low: 48.14, close: 48.61 },
    { high: 48.90, low: 48.39, close: 48.75 },
    { high: 48.87, low: 48.37, close: 48.63 },
    { high: 48.82, low: 48.24, close: 48.74 },
    { high: 49.05, low: 48.64, close: 49.03 },
    { high: 49.20, low: 48.94, close: 49.07 },
    { high: 49.35, low: 48.86, close: 49.32 },
    { high: 49.92, low: 49.50, close: 49.91 },
    { high: 50.19, low: 49.87, close: 50.13 },
    { high: 50.12, low: 49.20, close: 49.53 },
    { high: 49.66, low: 48.90, close: 49.50 },
    { high: 49.88, low: 49.43, close: 49.75 },
    { high: 50.19, low: 49.73, close: 50.03 },
    { high: 50.36, low: 49.26, close: 50.31 }, // t = 14
    { high: 50.57, low: 50.09, close: 50.52 }, // t = 15
    { high: 50.65, low: 50.30, close: 50.41 }, // t = 16
  ]

  it('preserves the 15-bars invariant (first 14 entries t < 14 are null)', () => {
    const result = calculateWilderATR(sampleBars, 14)
    expect(result).toHaveLength(17)

    for (let i = 0; i < 14; i++) {
      expect(result[i]).toBeNull()
    }
    expect(result[14]).not.toBeNull()
  })

  it('returns all nulls if dataset length <= period', () => {
    const shortList = sampleBars.slice(0, 14)
    const result = calculateWilderATR(shortList, 14)
    expect(result).toHaveLength(14)
    expect(result.every((v) => v === null)).toBe(true)
  })

  it('handles large price gaps using |H - PrevClose| and |L - PrevClose| correctly', () => {
    const gapBars = [
      { high: 100, low: 90, close: 95 },
      // Upward Gap: High=130, Low=120, PrevClose=95 -> TR = max(10, |130-95|=35, |120-95|=25) = 35
      { high: 130, low: 120, close: 125 },
      // Downward Gap: High=80, Low=70, PrevClose=125 -> TR = max(10, |80-125|=45, |70-125|=55) = 55
      { high: 80, low: 70, close: 75 },
      ...Array.from({ length: 15 }, () => ({ high: 76, low: 74, close: 75 })),
    ]

    const result = calculateWilderATR(gapBars, 14)
    expect(result[14]).toBeGreaterThan(0)
  })

  it('calculates 0 ATR for constant prices (H=L=C)', () => {
    const flatBars = Array.from({ length: 20 }, () => ({ high: 100, low: 100, close: 100 }))
    const result = calculateWilderATR(flatBars, 14)
    expect(result[14]).toBe(0)
    expect(result[15]).toBe(0)
  })

  it('registers ATR14 and custom overlays idempotently', () => {
    registerATR14()
    registerATR14()
    expect(klinecharts.registerIndicator).toHaveBeenCalled()

    registerCorporateActionOverlay()
    registerCorporateActionOverlay()
    expect(klinecharts.registerOverlay).toHaveBeenCalled()

    register52WLevelOverlay()
    register52WLevelOverlay()

    registerValuationLevelOverlay()
    registerValuationLevelOverlay()

    registerInsiderOverlay()
    registerInsiderOverlay()

    registerVWAPIndicator()
    registerVWAPIndicator()

    registerAnalystTargetOverlay()
    registerAnalystTargetOverlay()
  })

  it('restores insider marker point coordinates when performEventPressedMove is triggered', () => {
    const calls = vi.mocked(klinecharts.registerOverlay).mock.calls
    const insiderCall = calls.find((c) => c[0]?.name === 'secInsiderMarker')
    const overlayConfig: any = insiderCall?.[0]

    expect(overlayConfig).toBeDefined()
    const points = [{ timestamp: 999999, value: 888.88 }]
    const overlay = {
      extendData: {
        anchorTimestamp: 1700000000000,
        anchorValue: 150.0,
      },
    }

    overlayConfig?.performEventPressedMove?.({
      points,
      performPointIndex: 0,
      overlay,
    })

    expect(points[0]?.timestamp).toBe(1700000000000)
    expect(points[0]?.value).toBe(150.0)
  })
})


