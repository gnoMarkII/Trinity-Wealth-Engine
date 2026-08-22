import {
  registerIndicator,
  getSupportedIndicators,
  registerOverlay,
  getSupportedOverlays,
  type KLineData,
} from 'klinecharts'
import type { CorporateActionEventDTO } from '../../api/types'

export interface RSIResultItem {
  rsi: number | null
  ob70: number
  os30: number
}

export interface ATRResultItem {
  atr: number | null
}

/**
 * Pure function: คำนวณ Wilder's RSI(14) ตามมาตรฐานคณิตศาสตร์
 * - 15 Closes Invariant: สำหรับ t < period (เช่น t < 14) คืนค่า null
 * - Seed ที่ t = 14 คำนวณ AvgGain / AvgLoss จาก 14 แท่งแรก
 * - Wilder Smoothing (RMA) สำหรับ t > 14: (prev * 13 + current) / 14
 * - Edge cases: Flat price = 50.0, All up = 100.0, All down = 0.0
 */
export function calculateWilderRSI(
  dataList: { close: number }[],
  period: number = 14
): (number | null)[] {
  const len = dataList ? dataList.length : 0
  const result: (number | null)[] = new Array(len).fill(null)

  if (len <= period || period <= 0) {
    return result
  }

  // 1. Calculate Initial Seed at index = period (15th price, t = 14)
  let sumGain = 0
  let sumLoss = 0

  for (let i = 1; i <= period; i++) {
    const curr = dataList[i]
    const prev = dataList[i - 1]
    if (!curr || !prev) continue
    const diff = curr.close - prev.close
    if (diff > 0) {
      sumGain += diff
    } else {
      sumLoss += Math.abs(diff)
    }
  }

  let avgGain = sumGain / period
  let avgLoss = sumLoss / period

  const computeRsi = (gain: number, loss: number): number => {
    if (gain === 0 && loss === 0) {
      return 50.0
    }
    if (loss === 0) {
      return 100.0
    }
    if (gain === 0) {
      return 0.0
    }
    const rs = gain / loss
    return 100.0 - 100.0 / (1.0 + rs)
  }

  result[period] = computeRsi(avgGain, avgLoss)

  // 2. Wilder Smoothing for subsequent bars (t > period)
  for (let i = period + 1; i < len; i++) {
    const curr = dataList[i]
    const prev = dataList[i - 1]
    if (!curr || !prev) continue
    const diff = curr.close - prev.close
    const gain = diff > 0 ? diff : 0
    const loss = diff < 0 ? Math.abs(diff) : 0

    avgGain = (avgGain * (period - 1) + gain) / period
    avgLoss = (avgLoss * (period - 1) + loss) / period

    result[i] = computeRsi(avgGain, avgLoss)
  }

  return result
}

/**
 * Pure function: คำนวณ Wilder's ATR(14) ตามมาตรฐานคณิตศาสตร์
 * - Invariant: indices 0..13 (t < 14) คืนค่า null
 * - Seed at index 14 (t = 14) จากค่าเฉลี่ย True Range 14 ตัวแรก (transitions 1..14)
 * - Wilder Smoothing (RMA) สำหรับ t > 14: ATR[t] = (ATR[t-1] * 13 + TR[t]) / 14
 * - True Range: max(High - Low, |High - PrevClose|, |Low - PrevClose|)
 */
export function calculateWilderATR(
  dataList: { high: number; low: number; close: number }[],
  period: number = 14
): (number | null)[] {
  const len = dataList ? dataList.length : 0
  const result: (number | null)[] = new Array(len).fill(null)

  if (len <= period || period <= 0) {
    return result
  }

  const trs: number[] = new Array(len).fill(0)
  if (dataList[0]) {
    trs[0] = dataList[0].high - dataList[0].low
  }

  for (let i = 1; i < len; i++) {
    const curr = dataList[i]
    const prev = dataList[i - 1]
    if (!curr || !prev) continue
    const hl = curr.high - curr.low
    const hc = Math.abs(curr.high - prev.close)
    const lc = Math.abs(curr.low - prev.close)
    trs[i] = Math.max(hl, hc, lc)
  }

  // Seed at index = period (15th price, t = 14) from transitions 1..period
  let sumTR = 0
  for (let i = 1; i <= period; i++) {
    sumTR += trs[i] || 0
  }
  let currentATR = sumTR / period
  result[period] = currentATR

  // Wilder Smoothing for t > period
  for (let i = period + 1; i < len; i++) {
    currentATR = (currentATR * (period - 1) + (trs[i] || 0)) / period
    result[i] = currentATR
  }

  return result
}

let isRSI14Registered = false
let isATR14Registered = false
let isCorporateOverlayRegistered = false
let is52WOverlayRegistered = false

/**
 * ลงทะเบียน Custom Indicator 'RSI14_WITH_LEVELS' บน KlineCharts แบบ Idempotent
 */
export function registerRSI14WithLevels(): void {
  if (isRSI14Registered) return

  try {
    const supported = getSupportedIndicators?.()
    if (supported && supported.includes('RSI14_WITH_LEVELS')) {
      isRSI14Registered = true
      return
    }

    registerIndicator<RSIResultItem, number>({
      name: 'RSI14_WITH_LEVELS',
      shortName: 'RSI',
      series: 'normal',
      precision: 2,
      calcParams: [14],
      figures: [
        {
          key: 'rsi',
          title: 'RSI(14): ',
          type: 'line',
          styles: () => ({
            color: '#0284c7', // Sky blue
            size: 1.5,
          }),
        },
        {
          key: 'ob70',
          title: 'OB(70): ',
          type: 'line',
          styles: () => ({
            style: 'dashed',
            color: '#ef4444', // Red dashed
            size: 1,
            dashedValue: [4, 4],
          }),
        },
        {
          key: 'os30',
          title: 'OS(30): ',
          type: 'line',
          styles: () => ({
            style: 'dashed',
            color: '#22c55e', // Green dashed
            size: 1,
            dashedValue: [4, 4],
          }),
        },
      ],
      minValue: 0,
      maxValue: 100,
      calc: (dataList: KLineData[], indicator) => {
        const period = (indicator.calcParams?.[0] as number) || 14
        const rsiValues = calculateWilderRSI(dataList, period)
        return rsiValues.map((val) => ({
          rsi: val,
          ob70: 70,
          os30: 30,
        }))
      },
    })
    isRSI14Registered = true
  } catch {
    isRSI14Registered = true
  }
}

/**
 * ลงทะเบียน Custom Indicator 'ATR14' บน KlineCharts แบบ Idempotent
 */
export function registerATR14(): void {
  if (isATR14Registered) return

  try {
    const supported = getSupportedIndicators?.()
    if (supported && supported.includes('ATR14')) {
      isATR14Registered = true
      return
    }

    registerIndicator<ATRResultItem, number>({
      name: 'ATR14',
      shortName: 'ATR',
      series: 'normal',
      precision: 2,
      calcParams: [14],
      figures: [
        {
          key: 'atr',
          title: 'ATR(14): ',
          type: 'line',
          styles: () => ({
            color: '#f59e0b', // Amber/Orange
            size: 1.5,
          }),
        },
      ],
      calc: (dataList: KLineData[], indicator) => {
        const period = (indicator.calcParams?.[0] as number) || 14
        const atrValues = calculateWilderATR(dataList, period)
        return atrValues.map((val) => ({ atr: val }))
      },
    })
    isATR14Registered = true
  } catch {
    isATR14Registered = true
  }
}

export interface CorporateMarkerExtendData {
  event: CorporateActionEventDTO
  stackIndex: number
  stackCount: number
}

/**
 * ลงทะเบียน Custom Overlay 'corporateEventMarker' บน KlineCharts แบบ Idempotent
 */
export function registerCorporateActionOverlay(): void {
  if (isCorporateOverlayRegistered) return

  try {
    const supported = getSupportedOverlays?.()
    if (supported && supported.includes('corporateEventMarker')) {
      isCorporateOverlayRegistered = true
      return
    }

    registerOverlay({
      name: 'corporateEventMarker',
      needDefaultPointFigure: false,
      needDefaultXAxisFigure: false,
      needDefaultYAxisFigure: false,
      lock: true,
      createPointFigures: ({ coordinates, overlay }) => {
        const extend = overlay.extendData as CorporateMarkerExtendData | undefined
        if (!coordinates || !coordinates[0] || !extend || !extend.event) return []

        const { event, stackIndex = 0, stackCount = 1 } = extend
        const coord = coordinates[0]
        const offsetX = (stackIndex - (stackCount - 1) / 2) * 20
        const anchorX = coord.x + offsetX
        const anchorY = Math.max(26, coord.y - 12)

        const colorMap: Record<string, string> = {
          green: '#16a34a',
          red: '#dc2626',
          blue: '#2563eb',
          purple: '#9333ea',
        }
        const bgColorMap: Record<string, string> = {
          green: '#f0fdf4',
          red: '#fef2f2',
          blue: '#eff6ff',
          purple: '#faf5ff',
        }
        const borderColorMap: Record<string, string> = {
          green: '#bbf7d0',
          red: '#fecaca',
          blue: '#bfdbfe',
          purple: '#e9d5ff',
        }
        const markerColor = colorMap[event.color] || '#2563eb'
        const bgColor = bgColorMap[event.color] || '#eff6ff'
        const borderColor = borderColorMap[event.color] || '#bfdbfe'

        return [
          {
            type: 'polygon',
            attrs: {
              coordinates: [
                { x: anchorX, y: anchorY },
                { x: anchorX - 5, y: anchorY - 8 },
                { x: anchorX + 5, y: anchorY - 8 },
              ],
            },
            styles: {
              style: 'fill',
              color: markerColor,
            },
          },
          {
            type: 'text',
            attrs: {
              x: anchorX,
              y: anchorY - 10,
              text: event.label,
              align: 'center',
              baseline: 'bottom',
            },
            styles: {
              color: markerColor,
              size: 9.5,
              weight: 'bold',
              backgroundColor: bgColor,
              borderColor: borderColor,
              borderSize: 1,
              borderRadius: 3,
              paddingLeft: 4,
              paddingRight: 4,
              paddingTop: 1,
              paddingBottom: 1,
            },
          },
        ]
      },
    })
    isCorporateOverlayRegistered = true
  } catch {
    isCorporateOverlayRegistered = true
  }
}

export interface Level52WExtendData {
  type: 'high' | 'low'
  label: string
  color: string
}

/**
 * ลงทะเบียน Custom Overlay 'system52WLevel' บน KlineCharts แบบ Idempotent
 */
export function register52WLevelOverlay(): void {
  if (is52WOverlayRegistered) return

  try {
    const supported = getSupportedOverlays?.()
    if (supported && supported.includes('system52WLevel')) {
      is52WOverlayRegistered = true
      return
    }

    registerOverlay({
      name: 'system52WLevel',
      needDefaultPointFigure: false,
      needDefaultXAxisFigure: false,
      needDefaultYAxisFigure: false,
      lock: true,
      createPointFigures: ({ coordinates, bounding, overlay }) => {
        const extend = overlay.extendData as Level52WExtendData | undefined
        if (!coordinates || !coordinates[0] || !extend) return []
        const y = coordinates[0].y
        const width = bounding.width
        const isHigh = extend.type === 'high'
        const textColor = isHigh ? '#dc2626' : '#16a34a'
        const bgColor = isHigh ? '#fef2f2' : '#f0fdf4'
        const borderColor = isHigh ? '#fecaca' : '#bbf7d0'

        return [
          {
            type: 'line',
            attrs: {
              coordinates: [
                { x: 0, y },
                { x: width, y },
              ],
            },
            styles: {
              style: 'dashed',
              dashedValue: [4, 4],
              color: extend.color,
              size: 1,
            },
          },
          {
            type: 'text',
            attrs: {
              x: width - 8,
              y: isHigh ? y - 5 : y + 5,
              text: extend.label,
              align: 'right',
              baseline: isHigh ? 'bottom' : 'top',
            },
            styles: {
              color: textColor,
              size: 10,
              weight: 'bold',
              backgroundColor: bgColor,
              borderColor: borderColor,
              borderSize: 1,
              borderRadius: 4,
              paddingLeft: 6,
              paddingRight: 6,
              paddingTop: 2,
              paddingBottom: 2,
            },
          },
        ]
      },
    })
    is52WOverlayRegistered = true
  } catch {
    is52WOverlayRegistered = true
  }
}

let isValuationOverlayRegistered = false

export interface ValuationLevelExtendData {
  scenario_name: string
  label: string
  color: string
}

/**
 * ลงทะเบียน Custom Overlay 'valuationLevelRay' บน KlineCharts แบบ Idempotent
 * วาดเป็น Forward-Looking Ray จาก evaluated_at ไปทางขวาสุดของ Canvas
 */
export function registerValuationLevelOverlay(): void {
  if (isValuationOverlayRegistered) return

  try {
    const supported = getSupportedOverlays?.()
    if (supported && supported.includes('valuationLevelRay')) {
      isValuationOverlayRegistered = true
      return
    }

    registerOverlay({
      name: 'valuationLevelRay',
      needDefaultPointFigure: false,
      needDefaultXAxisFigure: false,
      needDefaultYAxisFigure: false,
      lock: true,
      createPointFigures: ({ coordinates, bounding, overlay }) => {
        const extend = overlay.extendData as ValuationLevelExtendData | undefined
        if (!coordinates || !coordinates[0] || !extend) return []
        const startX = Math.max(0, coordinates[0].x)
        const y = coordinates[0].y
        const width = bounding.width

        return [
          {
            type: 'line',
            attrs: {
              coordinates: [
                { x: startX, y },
                { x: width, y },
              ],
            },
            styles: {
              style: 'dashed',
              dashedValue: [5, 5],
              color: extend.color,
              size: 1.5,
            },
          },
          {
            type: 'text',
            attrs: {
              x: width - 8,
              y: y - 5,
              text: extend.label,
              align: 'right',
              baseline: 'bottom',
            },
            styles: {
              color: extend.color,
              size: 10,
              weight: 'bold',
              backgroundColor: '#ffffff',
              borderColor: 'rgba(0, 0, 0, 0.15)',
              borderSize: 1,
              borderRadius: 4,
              paddingLeft: 6,
              paddingRight: 6,
              paddingTop: 2,
              paddingBottom: 2,
            },
          },
        ]
      },
    })
    isValuationOverlayRegistered = true
  } catch {
    isValuationOverlayRegistered = true
  }
}

let isInsiderOverlayRegistered = false

export interface InsiderMarkerExtendData {
  accession_number: string
  insider_name: string
  action_type: 'buy' | 'sell' | 'cluster_buy'
  label: string
  color: string
  shares: number
  price: number
  filing_url: string
  officer_title?: string | null
  anchorTimestamp?: number
  anchorValue?: number
}

/**
 * ลงทะเบียน Custom Overlay 'secInsiderMarker' บน KlineCharts แบบ Idempotent
 * ปลด lock: true เพื่อรับ mouse events (click/hover) และใช้ performEventPressedMove เพื่อ restore anchor points ไม่ให้ point ถูกลากขยับ
 */
export function registerInsiderOverlay(): void {
  if (isInsiderOverlayRegistered) return

  try {
    const supported = getSupportedOverlays?.()
    if (supported && supported.includes('secInsiderMarker')) {
      isInsiderOverlayRegistered = true
      return
    }

    registerOverlay({
      name: 'secInsiderMarker',
      needDefaultPointFigure: false,
      needDefaultXAxisFigure: false,
      needDefaultYAxisFigure: false,
      // Restore point coordinates immediately if user attempts to drag the marker point
      performEventPressedMove: ({ points, performPointIndex, overlay }: any) => {
        const extend = overlay?.extendData as InsiderMarkerExtendData | undefined
        if (extend?.anchorTimestamp != null && extend?.anchorValue != null && points[performPointIndex]) {
          points[performPointIndex].timestamp = extend.anchorTimestamp
          points[performPointIndex].value = extend.anchorValue
        }
      },
      createPointFigures: ({ coordinates, overlay }) => {
        const extend = overlay.extendData as InsiderMarkerExtendData | undefined
        if (!coordinates || !coordinates[0] || !extend) return []
        const anchorX = coordinates[0].x
        const anchorY = coordinates[0].y

        if (extend.action_type === 'cluster_buy') {
          return [
            {
              type: 'polygon',
              attrs: {
                coordinates: [
                  { x: anchorX, y: anchorY + 4 },
                  { x: anchorX + 5, y: anchorY + 10 },
                  { x: anchorX, y: anchorY + 16 },
                  { x: anchorX - 5, y: anchorY + 10 },
                ],
              },
              styles: {
                style: 'fill',
                color: '#8b5cf6',
              },
            },
            {
              type: 'text',
              attrs: {
                x: anchorX,
                y: anchorY + 19,
                text: extend.label,
                align: 'center',
                baseline: 'top',
              },
              styles: {
                color: '#6d28d9',
                size: 9.5,
                weight: 'bold',
                backgroundColor: '#f5f3ff',
                borderColor: '#ddd6fe',
                borderSize: 1,
                borderRadius: 4,
                paddingLeft: 5,
                paddingRight: 5,
                paddingTop: 1.5,
                paddingBottom: 1.5,
              },
            },
          ]
        }

        if (extend.action_type === 'buy') {
          return [
            {
              type: 'polygon',
              attrs: {
                coordinates: [
                  { x: anchorX, y: anchorY + 4 },
                  { x: anchorX - 5, y: anchorY + 12 },
                  { x: anchorX + 5, y: anchorY + 12 },
                ],
              },
              styles: {
                style: 'fill',
                color: '#10b981',
              },
            },
            {
              type: 'text',
              attrs: {
                x: anchorX,
                y: anchorY + 15,
                text: extend.label,
                align: 'center',
                baseline: 'top',
              },
              styles: {
                color: '#047857',
                size: 9,
                weight: 'bold',
                backgroundColor: '#ecfdf5',
                borderColor: '#a7f3d0',
                borderSize: 1,
                borderRadius: 3,
                paddingLeft: 4,
                paddingRight: 4,
                paddingTop: 1,
                paddingBottom: 1,
              },
            },
          ]
        }

        return [
          {
            type: 'polygon',
            attrs: {
              coordinates: [
                { x: anchorX, y: anchorY - 4 },
                { x: anchorX - 5, y: anchorY - 12 },
                { x: anchorX + 5, y: anchorY - 12 },
              ],
            },
            styles: {
              style: 'fill',
              color: '#ef4444',
            },
          },
          {
            type: 'text',
            attrs: {
              x: anchorX,
              y: anchorY - 15,
              text: extend.label,
              align: 'center',
              baseline: 'bottom',
            },
            styles: {
              color: '#b91c1c',
              size: 9,
              weight: 'bold',
              backgroundColor: '#fef2f2',
              borderColor: '#fecaca',
              borderSize: 1,
              borderRadius: 3,
              paddingLeft: 4,
              paddingRight: 4,
              paddingTop: 1,
              paddingBottom: 1,
            },
          },
        ]
      },
    })
    isInsiderOverlayRegistered = true
  } catch {
    isInsiderOverlayRegistered = true
  }
}

let isAnalystTargetOverlayRegistered = false

export interface AnalystTargetExtendData {
  label: string
  color: string
  isStale?: boolean
}

/**
 * ลงทะเบียน Custom Overlay 'analystTargetRay' บน KlineCharts แบบ Idempotent
 * วาดเส้นประแนวนอน Mean Consensus Target พร้อม Pill badge แสดง % upside
 */
export function registerAnalystTargetOverlay(): void {
  if (isAnalystTargetOverlayRegistered) return

  try {
    const supported = getSupportedOverlays?.()
    if (supported && supported.includes('analystTargetRay')) {
      isAnalystTargetOverlayRegistered = true
      return
    }

    registerOverlay({
      name: 'analystTargetRay',
      needDefaultPointFigure: false,
      needDefaultXAxisFigure: false,
      needDefaultYAxisFigure: false,
      lock: true,
      createPointFigures: ({ coordinates, bounding, overlay }) => {
        const extend = overlay.extendData as AnalystTargetExtendData | undefined
        if (!coordinates || !coordinates[0] || !extend) return []
        const startX = Math.max(0, coordinates[0].x)
        const y = coordinates[0].y
        const width = bounding.width

        return [
          {
            type: 'line',
            attrs: {
              coordinates: [
                { x: startX, y },
                { x: width, y },
              ],
            },
            styles: {
              style: 'dashed',
              dashedValue: extend.isStale ? [4, 4] : [6, 3],
              color: extend.color || '#f59e0b',
              size: 1.5,
            },
          },
          {
            type: 'text',
            attrs: {
              x: width - 8,
              y: y - 5,
              text: extend.label,
              align: 'right',
              baseline: 'bottom',
            },
            styles: {
              color: extend.color || '#f59e0b',
              size: 10,
              weight: 'bold',
              backgroundColor: extend.isStale ? '#fffbeb' : '#ffffff',
              borderColor: extend.isStale ? '#fde68a' : 'rgba(245, 158, 11, 0.3)',
              borderSize: 1,
              borderRadius: 4,
              paddingLeft: 6,
              paddingRight: 6,
              paddingTop: 2,
              paddingBottom: 2,
            },
          },
        ]
      },
    })
    isAnalystTargetOverlayRegistered = true
  } catch {
    isAnalystTargetOverlayRegistered = true
  }
}

let isVWAPRegistered = false

export interface VWAPResultItem {
  vwap: number | null
}

function formatDateToPartsYMD(date: Date, timeZone: string): string {
  try {
    const formatter = new Intl.DateTimeFormat('en-US', {
      timeZone,
      year: 'numeric',
      month: '2-digit',
      day: '2-digit',
    })
    const parts = formatter.formatToParts(date)
    let year = ''
    let month = ''
    let day = ''
    for (const p of parts) {
      if (p.type === 'year') year = p.value
      else if (p.type === 'month') month = p.value
      else if (p.type === 'day') day = p.value
    }
    return `${year}-${month}-${day}`
  } catch {
    return date.toISOString().slice(0, 10)
  }
}

/**
 * ลงทะเบียน Custom Indicator 'SESSION_VWAP' บน KlineCharts แบบ Idempotent
 * คำนวณ Session VWAP: Σ(TypicalPrice * Volume) / Σ(Volume) โดย TypicalPrice = (H+L+C)/3
 * Session reset คำนวณตาม exchange timezone โดยใช้ formatToParts
 */
export function registerVWAPIndicator(): void {
  if (isVWAPRegistered) return

  try {
    const supported = getSupportedIndicators?.()
    if (supported && supported.includes('SESSION_VWAP')) {
      isVWAPRegistered = true
      return
    }

    registerIndicator<VWAPResultItem, number>({
      name: 'SESSION_VWAP',
      shortName: 'VWAP',
      series: 'price',
      precision: 2,
      figures: [
        {
          key: 'vwap',
          title: 'VWAP: ',
          type: 'line',
          styles: () => ({
            color: '#6366f1', // Indigo line
            size: 1.5,
          }),
        },
      ],
      calc: (dataList: KLineData[], indicator) => {
        const tz = ((indicator as any).extendData?.exchangeTz as string) || 'America/New_York'
        let cumTpv = 0
        let cumVol = 0
        let prevDateStr = ''
        let prevVwap: number | null = null

        return dataList.map((bar) => {
          const dateStr = formatDateToPartsYMD(new Date(bar.timestamp), tz)
          if (dateStr !== prevDateStr) {
            // New Session: Reset accumulators
            cumTpv = 0
            cumVol = 0
            prevDateStr = dateStr
            prevVwap = null
          }

          const vol = bar.volume ?? 0
          if (vol > 0) {
            const tp = (bar.high + bar.low + bar.close) / 3
            cumTpv += tp * vol
            cumVol += vol
            prevVwap = cumVol > 0 ? cumTpv / cumVol : null
          }

          return { vwap: prevVwap }
        })
      },
    })
    isVWAPRegistered = true
  } catch {
    isVWAPRegistered = true
  }
}



