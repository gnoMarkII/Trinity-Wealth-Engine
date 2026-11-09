import React, { Component, useEffect, useMemo, useRef, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import type { Components } from 'react-markdown'
import TradingViewMiniWidget from '../TradingViewMiniWidget'
import { RadialGauge } from '../charts/RadialGauge'
import type { GaugeSubItem, GaugeZone } from '../charts/RadialGauge'

export type AdvisorActionHandler = (prompt: string, contextRefs?: string[]) => void

export interface RebalanceTargetItem {
  symbol: string
  targetWeightPct: number
  currentWeightPct?: number
}

export interface WidgetBlockData {
  widgetType: 'gauge' | 'rebalance_simulator'
  title?: string
  payload: Record<string, unknown>
}

export interface TableData {
  headers: string[]
  alignments: ('left' | 'center' | 'right')[]
  rows: string[][]
}

export type ContentSegment =
  | { type: 'markdown'; content: string }
  | { type: 'table'; data: TableData }
  | {
      type: 'widget'
      data: WidgetBlockData | null
      isComplete: boolean
      isValid: boolean
      error?: string
      unsupportedType?: string
      title?: string
      raw: string
    }

export function renderWithEvidenceBadges(node: React.ReactNode): React.ReactNode {
  if (typeof node !== 'string') return node
  const pattern = /\[(DOCUMENTED|FACT|TESTED|MATH|INFERRED|UNVERIFIED)(?::\s*([^\]]+))?\]/g
  if (!pattern.test(node)) return node

  const parts: React.ReactNode[] = []
  let lastIndex = 0
  pattern.lastIndex = 0
  let match: RegExpExecArray | null

  while ((match = pattern.exec(node)) !== null) {
    if (match.index > lastIndex) {
      parts.push(node.substring(lastIndex, match.index))
    }
    const kind = match[1]?.toUpperCase()
    const detail = match[2]?.trim()

    if (kind === 'DOCUMENTED' || kind === 'FACT') {
      parts.push(
        <span
          key={match.index}
          className="inline-flex items-center gap-1 mx-1 px-2 py-0.5 rounded-md border border-emerald-200 bg-emerald-50 text-xs font-semibold text-emerald-800 shadow-2xs"
          title={detail ? `Verified Primary Document: ${detail}` : 'Verified Primary Document'}
        >
          <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 shrink-0" />
          {detail ? `DOC: ${detail}` : 'DOCUMENTED'}
        </span>
      )
    } else if (kind === 'TESTED' || kind === 'MATH') {
      parts.push(
        <span
          key={match.index}
          className="inline-flex items-center gap-1 mx-1 px-2 py-0.5 rounded-md border border-sky-200 bg-sky-50 text-xs font-semibold text-sky-800 shadow-2xs"
          title={detail ? `Deterministic Python Code: ${detail}` : 'Deterministic Python Code'}
        >
          <span className="w-1.5 h-1.5 rounded-full bg-sky-500 shrink-0" />
          {detail ? `MATH: ${detail}` : 'TESTED'}
        </span>
      )
    } else if (kind === 'INFERRED') {
      parts.push(
        <span
          key={match.index}
          className="inline-flex items-center gap-1 mx-1 px-2 py-0.5 rounded-md border border-purple-200 bg-purple-50 text-xs font-semibold text-purple-800 shadow-2xs"
          title={detail ? `Analytical Deduction: ${detail}` : 'Analytical Deduction'}
        >
          <span className="w-1.5 h-1.5 rounded-full bg-purple-500 shrink-0" />
          {detail ? `INFERRED: ${detail}` : 'INFERRED'}
        </span>
      )
    } else {
      parts.push(
        <span
          key={match.index}
          className="inline-flex items-center gap-1 mx-1 px-2 py-0.5 rounded-md border border-amber-200 bg-amber-50 text-xs font-semibold text-amber-800 shadow-2xs"
          title={detail ? `Unverified / Rumor: ${detail}` : 'Unverified Report'}
        >
          <span className="w-1.5 h-1.5 rounded-full bg-amber-500 shrink-0" />
          {detail ? `UNVERIFIED: ${detail}` : 'UNVERIFIED'}
        </span>
      )
    }
    lastIndex = pattern.lastIndex
  }

  if (lastIndex < node.length) {
    parts.push(node.substring(lastIndex))
  }

  return parts
}

export function parseTableRow(line: string): string[] {
  const trimmed = line.trim().replace(/^\|/, '').replace(/\|$/, '')
  return trimmed.split('|').map((c) => c.trim())
}

export function parseAlignment(cell: string): 'left' | 'center' | 'right' {
  const c = cell.trim()
  if (c.startsWith(':') && c.endsWith(':')) return 'center'
  if (c.endsWith(':')) return 'right'
  return 'left'
}

const _NON_TICKER_WORDS = new Set([
  'BUY',
  'SELL',
  'HOLD',
  'TRIM',
  'ADD',
  'WAIT',
  'WATCH',
  'AVOID',
  'PASS',
  'FAIL',
  'HIGH',
  'LOW',
  'MEDIUM',
  'MODERATE',
  'STRONG',
  'WEAK',
  'CORE',
  'CASH',
  'THB',
  'USD',
  'EUR',
  'GBP',
  'JPY',
  'CNY',
  'BPS',
  'YES',
  'NO',
  'NA',
  'OK',
  'TBD',
  'NULL',
  'NONE',
])

const _MISSING_VALUE_RE = /^(?:-|—|–|n\/a|na|null|none|unavailable|tbd|n\.a\.|ไม่มีข้อมูล|\?)$/i

function isPlainRecord(val: unknown): val is Record<string, unknown> {
  return typeof val === 'object' && val !== null && !Array.isArray(val)
}

function parseValidatedWidgetBlock(raw: string): {
  data: WidgetBlockData | null
  isValid: boolean
  error?: string
  unsupportedType?: string
  title?: string
} {
  let parsed: unknown
  try {
    parsed = JSON.parse(raw)
  } catch {
    return { data: null, isValid: false, error: 'รูปแบบ JSON ของ Widget ไม่ถูกต้อง' }
  }

  if (!isPlainRecord(parsed)) {
    return { data: null, isValid: false, error: 'ข้อมูล Widget ต้องเป็น JSON Object' }
  }

  const rawType =
    typeof parsed.widgetType === 'string'
      ? parsed.widgetType.trim()
      : typeof parsed.type === 'string'
        ? parsed.type.trim()
        : ''

  if (!rawType) {
    return { data: null, isValid: false, error: 'ไม่ได้ระบุประเภท Widget (widgetType)' }
  }

  if (parsed.title !== undefined && parsed.title !== null && typeof parsed.title !== 'string') {
    return { data: null, isValid: false, error: 'ฟิลด์ title ของ Widget ต้องเป็นข้อความ' }
  }
  const topTitle = typeof parsed.title === 'string' ? parsed.title : undefined

  let payload: Record<string, unknown>
  if ('payload' in parsed && parsed.payload !== undefined) {
    if (!isPlainRecord(parsed.payload)) {
      return { data: null, isValid: false, error: 'ฟิลด์ payload ของ Widget ต้องเป็น Object', title: topTitle }
    }
    payload = parsed.payload
  } else {
    payload = { ...parsed }
  }

  const optStr = (val: unknown, field: string): string | undefined | { invalid: string } => {
    if (val === undefined || val === null) return undefined
    if (typeof val !== 'string') return { invalid: `ฟิลด์ ${field} ต้องเป็นข้อความ` }
    return val
  }

  const pTitle = optStr(payload.title, 'payload.title')
  if (isPlainRecord(pTitle)) return { data: null, isValid: false, error: pTitle.invalid, title: topTitle }

  if (rawType === 'gauge') {
    const pSubtitle = optStr(payload.subtitle, 'subtitle')
    if (isPlainRecord(pSubtitle)) return { data: null, isValid: false, error: pSubtitle.invalid, title: topTitle || pTitle }
    const pUnit = optStr(payload.unit, 'unit')
    if (isPlainRecord(pUnit)) return { data: null, isValid: false, error: pUnit.invalid, title: topTitle || pTitle }
    const pAsOf = optStr(payload.asOfDate, 'asOfDate')
    if (isPlainRecord(pAsOf)) return { data: null, isValid: false, error: pAsOf.invalid, title: topTitle || pTitle }
    const pLag = optStr(payload.dataLagNote, 'dataLagNote')
    if (isPlainRecord(pLag)) return { data: null, isValid: false, error: pLag.invalid, title: topTitle || pTitle }

    if (typeof payload.value !== 'number' || !Number.isFinite(payload.value)) {
      return { data: null, isValid: false, error: 'ค่ามาตรวัด (value) ต้องเป็นตัวเลขจำกัด', title: topTitle || pTitle }
    }

    const min = payload.min !== undefined ? payload.min : -3
    const max = payload.max !== undefined ? payload.max : 3
    if (typeof min !== 'number' || !Number.isFinite(min) || typeof max !== 'number' || !Number.isFinite(max) || min >= max) {
      return { data: null, isValid: false, error: 'ช่วงค่า min/max ของมาตรวัดไม่ถูกต้อง', title: topTitle || pTitle }
    }

    if (payload.zones !== undefined) {
      if (!Array.isArray(payload.zones)) {
        return { data: null, isValid: false, error: 'ฟิลด์ zones ของมาตรวัดไม่ถูกต้อง', title: topTitle || pTitle }
      }
      for (const z of payload.zones) {
        if (
          !isPlainRecord(z) ||
          typeof z.from !== 'number' ||
          !Number.isFinite(z.from) ||
          typeof z.to !== 'number' ||
          !Number.isFinite(z.to) ||
          z.from > z.to ||
          typeof z.color !== 'string' ||
          typeof z.label !== 'string'
        ) {
          return { data: null, isValid: false, error: 'รูปแบบโซนของมาตรวัดไม่ถูกต้อง', title: topTitle || pTitle }
        }
      }
    }

    if (payload.subItems !== undefined) {
      if (!Array.isArray(payload.subItems)) {
        return { data: null, isValid: false, error: 'ฟิลด์ subItems ของมาตรวัดไม่ถูกต้อง', title: topTitle || pTitle }
      }
      for (const s of payload.subItems) {
        if (
          !isPlainRecord(s) ||
          typeof s.label !== 'string' ||
          typeof s.value !== 'number' ||
          !Number.isFinite(s.value) ||
          (s.unit !== undefined && s.unit !== null && typeof s.unit !== 'string')
        ) {
          return { data: null, isValid: false, error: 'รายการย่อย subItems ของมาตรวัดไม่ถูกต้อง', title: topTitle || pTitle }
        }
      }
    }

    return {
      data: {
        widgetType: 'gauge',
        title: topTitle || pTitle,
        payload,
      },
      isValid: true,
      title: topTitle || pTitle,
    }
  }

  if (rawType === 'rebalance_simulator') {
    const rawList = Array.isArray(payload.targets)
      ? payload.targets
      : Array.isArray(payload.holdings)
        ? payload.holdings
        : Array.isArray(payload.positions)
          ? payload.positions
          : null

    if (!rawList || rawList.length === 0) {
      return {
        data: null,
        isValid: false,
        error: 'Widget จำลองสัดส่วนไม่มีรายการหุ้นเป้าหมายที่ถูกต้อง',
        title: topTitle || pTitle,
      }
    }

    const normalizedTargets: RebalanceTargetItem[] = []
    for (const item of rawList) {
      if (!isPlainRecord(item) || typeof item.symbol !== 'string' || !item.symbol.trim()) {
        return {
          data: null,
          isValid: false,
          error: 'ข้อมูลสัญลักษณ์หุ้นใน Widget จำลองสัดส่วนไม่ถูกต้อง',
          title: topTitle || pTitle,
        }
      }
      const rawW = item.targetWeightPct ?? item.target_weight_pct ?? item.weight
      if (typeof rawW !== 'number' || !Number.isFinite(rawW) || rawW < 0 || rawW > 100) {
        return {
          data: null,
          isValid: false,
          error: 'ค่าสัดส่วนเป้าหมายใน Widget ต้องอยู่ระหว่าง 0 ถึง 100%',
          title: topTitle || pTitle,
        }
      }
      const rawCur = item.currentWeightPct ?? item.current_weight_pct
      if (rawCur !== undefined && (typeof rawCur !== 'number' || !Number.isFinite(rawCur))) {
        return {
          data: null,
          isValid: false,
          error: 'ค่าสัดส่วนปัจจุบันใน Widget ต้องเป็นตัวเลขจำกัด',
          title: topTitle || pTitle,
        }
      }
      normalizedTargets.push({
        symbol: item.symbol.trim().replace(/^\$/, ''),
        targetWeightPct: rawW,
        currentWeightPct: typeof rawCur === 'number' ? rawCur : undefined,
      })
    }

    return {
      data: {
        widgetType: 'rebalance_simulator',
        title: topTitle || pTitle,
        payload: { ...payload, normalizedTargets },
      },
      isValid: true,
      title: topTitle || pTitle,
    }
  }

  return {
    data: null,
    isValid: false,
    unsupportedType: rawType,
    error: `ไม่รองรับ Interactive Widget ประเภท "${rawType}"`,
    title: topTitle || pTitle,
  }
}

export function splitMarkdownAndTables(text: string): ContentSegment[] {
  const lines = text.split('\n')
  const segments: ContentSegment[] = []
  let currentMarkdownLines: string[] = []
  let i = 0

  while (i < lines.length) {
    const line = lines[i] ?? ''
    const trimmed = line.trim()

    // 1. Detect Code Fence (``` or ~~~ of length >= 3) cleanly
    const openFenceMatch = line.match(/^\s*(`{3,}|~{3,})(.*)$/)
    if (openFenceMatch) {
      const marker = openFenceMatch[1] ?? '```'
      const openChar = marker[0] ?? '`'
      const openLen = marker.length
      const infoRaw = (openFenceMatch[2] ?? '').trim()
      const infoToken = (infoRaw.split(/\s+/)[0] ?? '').toLowerCase()
      const validBacktickInfo = openChar !== '`' || !infoRaw.includes('`')

      if (validBacktickInfo && (infoToken === 'widget' || infoToken === 'json:widget')) {
        if (currentMarkdownLines.length > 0) {
          segments.push({
            type: 'markdown',
            content: currentMarkdownLines.join('\n'),
          })
          currentMarkdownLines = []
        }
        i++
        const widgetLines: string[] = []
        let isClosed = false
        while (i < lines.length) {
          const wLine = lines[i] ?? ''
          const closeMatch = wLine.match(/^\s*(`{3,}|~{3,})\s*$/)
          if (closeMatch && (closeMatch[1]?.[0] ?? '') === openChar && (closeMatch[1]?.length ?? 0) >= openLen) {
            isClosed = true
            i++
            break
          }
          widgetLines.push(wLine)
          i++
        }

        const raw = widgetLines.join('\n')
        if (isClosed) {
          const parsedRes = parseValidatedWidgetBlock(raw)
          segments.push({
            type: 'widget',
            data: parsedRes.data,
            isComplete: true,
            isValid: parsedRes.isValid,
            error: parsedRes.error,
            unsupportedType: parsedRes.unsupportedType,
            title: parsedRes.title,
            raw,
          })
        } else {
          segments.push({
            type: 'widget',
            data: null,
            isComplete: false,
            isValid: false,
            raw,
          })
        }
        continue
      }

      if (validBacktickInfo) {
        // Ordinary non-widget code fence: consume intact until matching closing fence
        currentMarkdownLines.push(line)
        i++
        while (i < lines.length) {
          const codeLine = lines[i] ?? ''
          currentMarkdownLines.push(codeLine)
          const closeMatch = codeLine.match(/^\s*(`{3,}|~{3,})\s*$/)
          i++
          if (closeMatch && (closeMatch[1]?.[0] ?? '') === openChar && (closeMatch[1]?.length ?? 0) >= openLen) {
            break
          }
        }
        continue
      }
    }

    // 2. Detect Table Block
    const isTableLine = trimmed.startsWith('|') && trimmed.endsWith('|')
    if (isTableLine && i + 1 < lines.length) {
      const nextLine = (lines[i + 1] ?? '').trim()
      const isSeparator = /^\|(?:\s*:?-+:?\s*\|)+\s*$/.test(nextLine)

      if (isSeparator) {
        if (currentMarkdownLines.length > 0) {
          segments.push({
            type: 'markdown',
            content: currentMarkdownLines.join('\n'),
          })
          currentMarkdownLines = []
        }

        const headers = parseTableRow(line)
        const alignments = parseTableRow(nextLine).map(parseAlignment)
        const rows: string[][] = []
        i += 2

        while (i < lines.length) {
          const rowLine = (lines[i] ?? '').trim()
          if (!rowLine.startsWith('|') || !rowLine.endsWith('|')) {
            break
          }
          rows.push(parseTableRow(rowLine))
          i++
        }

        segments.push({
          type: 'table',
          data: { headers, alignments, rows },
        })
        continue
      }
    }

    currentMarkdownLines.push(line)
    i++
  }

  if (currentMarkdownLines.length > 0) {
    segments.push({
      type: 'markdown',
      content: currentMarkdownLines.join('\n'),
    })
  }

  return segments
}

export interface RowMetric {
  ticker?: string
  label: string
  numericValue?: number
  isPercentage: boolean
  unit?: string
  rawCells: string[]
}

interface ParsedCellNumeric {
  kind: 'numeric' | 'missing' | 'text'
  value?: number
  isPercentage: boolean
  unit: string
}

interface MetricColumnInfo {
  colIndex: number
  header: string
  unit: string
  hasMixedUnits: boolean
  mixedUnitLabels: string[]
  isPercentage: boolean
  isWeightColumn: boolean
  isReturnLike: boolean
  validCount: number
  values: (number | undefined)[]
}

function stripCellEmphasis(cell: string): string {
  return cell
    .trim()
    .replace(/^\*\*([\s\S]*)\*\*$/, '$1')
    .replace(/^\*([\s\S]*)\*$/, '$1')
    .replace(/^`([\s\S]*)`$/, '$1')
    .trim()
}

function inferUnitFromHeader(header: string): string {
  const h = header.trim()
  if (!h) return ''
  if (/\(%?\s*%|percentage|percent|สัดส่วน|กำไร\/ขาดทุน\s*\(%\)/i.test(h) || h.includes('%')) return '%'
  if (/\bTHB\b|บาท|฿/i.test(h)) return 'THB'
  if (/\bUSD\b|\$/i.test(h)) return 'USD'
  if (/\bEUR\b|€/i.test(h)) return 'EUR'
  if (/\bbps\b/i.test(h)) return 'bps'
  return ''
}

function normalizeCurrencyToken(raw?: string): string {
  if (!raw) return ''
  const upper = raw.trim().toUpperCase()
  if (upper === '฿' || upper === 'THB') return 'THB'
  if (upper === '$' || upper === 'USD') return 'USD'
  if (upper === '€' || upper === 'EUR') return 'EUR'
  if (upper === '£' || upper === 'GBP') return 'GBP'
  if (upper === 'BPS') return 'bps'
  if (upper === 'X') return 'x'
  return upper
}

function parseNumericCell(cell: string, headerUnit: string = ''): ParsedCellNumeric {
  const clean = stripCellEmphasis(cell)
  if (_MISSING_VALUE_RE.test(clean)) {
    return { kind: 'missing', isPercentage: false, unit: '' }
  }

  // Percentage e.g. +71.86%, -20.70%, 15%, 1,250.5%
  const pctMatch = clean.match(/^([+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)\s*%$/)
  if (pctMatch && pctMatch[1]) {
    const val = Number(pctMatch[1].replace(/,/g, ''))
    if (Number.isFinite(val)) {
      return { kind: 'numeric', value: val, isPercentage: true, unit: '%' }
    }
    return { kind: 'text', isPercentage: false, unit: '' }
  }

  // Plain or currency/unit numbers (with or without commas), rejecting conflicting double signs
  const numMatch = clean.match(
    /^([+-]?)\s*(?:(THB|USD|EUR|GBP|JPY|CNY|[฿$€£])\s*)?([+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)(?:\s*(THB|USD|EUR|GBP|JPY|CNY|bps|x|X))?$/i
  )
  if (numMatch && numMatch[3]) {
    const leadSign = numMatch[1] ?? ''
    const body = numMatch[3]
    if (leadSign && (body.startsWith('+') || body.startsWith('-'))) {
      return { kind: 'text', isPercentage: false, unit: '' }
    }
    const rawNum = `${leadSign}${body}`.replace(/,/g, '')
    const val = Number(rawNum)
    if (!Number.isFinite(val)) {
      return { kind: 'text', isPercentage: false, unit: '' }
    }
    const prefixCur = normalizeCurrencyToken(numMatch[2])
    const suffixUnit = normalizeCurrencyToken(numMatch[4])
    if (prefixCur && suffixUnit && prefixCur !== suffixUnit) {
      return { kind: 'text', isPercentage: false, unit: '' }
    }
    const explicitUnit = prefixCur || suffixUnit
    const finalUnit = explicitUnit || headerUnit
    const isPct = finalUnit === '%'
    return { kind: 'numeric', value: val, isPercentage: isPct, unit: finalUnit }
  }

  return { kind: 'text', isPercentage: false, unit: '' }
}

function isTickerHeader(header: string): boolean {
  const h = header.trim().toLowerCase()
  if (!h) return false
  if (/action|status|rating|recommendation|signal|notes?|comment|คำแนะนำ|สถานะ|หมายเหตุ/.test(h)) {
    return false
  }
  return /ticker|symbol|asset|stock|holding|equity|fund|หลักทรัพย์|สัญลักษณ์|หุ้น|สินทรัพย์|ตัวย่อ/.test(h)
}

function isWeightHeader(header: string): boolean {
  const h = header.trim().toLowerCase()
  if (!h) return false
  if (/diff|drift|delta|change|ส่วนต่าง/.test(h)) return false
  return /weight|allocation|target|สัดส่วน|น้ำหนัก/.test(h)
}

function isReturnLikeHeader(header: string): boolean {
  const h = header.trim().toLowerCase()
  if (!h) return false
  return /return|pnl|p\/l|gain|loss|upside|yield|กำไร|ขาดทุน|ผลตอบแทน/.test(h)
}

function parseTickerToken(cell: string, opts: { requireExplicitOrCol: boolean }): string | undefined {
  const clean = cell.trim()
  if (!clean) return undefined
  const boldMatch = clean.match(/^\*\*(\$?[A-Z0-9][A-Z0-9._\-]{0,14})\*\*$/)
  const dollarMatch = clean.match(/^\$([A-Z0-9][A-Z0-9._\-]{0,14})$/)
  const bareMatch = clean.match(/^([A-Z]{1,6}(?:\.[A-Z]{1,3})?)$/)

  let sym: string | undefined
  if (boldMatch && boldMatch[1]) {
    sym = boldMatch[1].replace(/^\$/, '')
  } else if (dollarMatch && dollarMatch[1]) {
    sym = dollarMatch[1]
  } else if (opts.requireExplicitOrCol && bareMatch && bareMatch[1]) {
    sym = bareMatch[1]
  }

  if (!sym) return undefined
  if (_NON_TICKER_WORDS.has(sym.toUpperCase())) return undefined
  if (/^\d+(?:\.\d+)?$/.test(sym)) return undefined
  return sym
}

function identifyTickerColIndex(headers: string[], rows: string[][]): number {
  const explicitIdx = headers.findIndex(isTickerHeader)
  if (explicitIdx >= 0) return explicitIdx

  if (headers.length > 0 && rows.length > 0) {
    const firstHeader = (headers[0] ?? '').trim().toLowerCase()
    const col0IsForbidden = /action|status|rating|signal|คำแนะนำ|สถานะ/.test(firstHeader)
    if (!col0IsForbidden) {
      const explicitDollarOrBoldHits = rows.filter((r) => {
        const c = (r[0] ?? '').trim()
        return (/^\$\$?[A-Z]/.test(c) || /^\*\*[A-Z0-9._\-]+\*\*$/.test(c)) && parseTickerToken(c, { requireExplicitOrCol: false })
      }).length
      if (explicitDollarOrBoldHits > 0 && explicitDollarOrBoldHits === rows.length) {
        return 0
      }
    }
  }
  return -1
}

function analyzeTableForChartsAndSimulator(data: TableData): {
  tickerColIndex: number
  weightColIndex: number
  rowTickers: (string | undefined)[]
  uniqueTickers: string[]
  explicitWeightsByTicker: Record<string, number>
  allTickersHaveExplicitWeight: boolean
  metricColumns: MetricColumnInfo[]
  defaultMetricColIndex: number | null
} {
  const tickerColIndex = identifyTickerColIndex(data.headers, data.rows)
  const weightColIndex = data.headers.findIndex((h, idx) => idx !== tickerColIndex && isWeightHeader(h))

  const rowTickers = data.rows.map((row) =>
    tickerColIndex >= 0 ? parseTickerToken(row[tickerColIndex] ?? '', { requireExplicitOrCol: true }) : undefined
  )

  const uniqueTickers: string[] = []
  const seenTickers = new Set<string>()
  const explicitWeightsByTicker: Record<string, number> = {}
  const weightCountByTicker: Record<string, number> = {}

  data.rows.forEach((row, rIdx) => {
    const sym = rowTickers[rIdx]
    if (!sym) return
    if (!seenTickers.has(sym)) {
      seenTickers.add(sym)
      uniqueTickers.push(sym)
    }
    if (weightColIndex >= 0) {
      const parsedW = parseNumericCell(row[weightColIndex] ?? '', inferUnitFromHeader(data.headers[weightColIndex] ?? ''))
      if (parsedW.kind === 'numeric' && parsedW.value !== undefined && parsedW.value >= 0) {
        explicitWeightsByTicker[sym] = (explicitWeightsByTicker[sym] ?? 0) + parsedW.value
        weightCountByTicker[sym] = (weightCountByTicker[sym] ?? 0) + 1
      }
    }
  })

  const allTickersHaveExplicitWeight =
    uniqueTickers.length > 0 &&
    weightColIndex >= 0 &&
    uniqueTickers.every((sym) => (weightCountByTicker[sym] ?? 0) > 0)

  const metricColumns: MetricColumnInfo[] = []
  for (let cIdx = 0; cIdx < data.headers.length; cIdx++) {
    if (cIdx === tickerColIndex) continue
    const header = data.headers[cIdx] ?? `Column ${cIdx + 1}`
    const headerUnit = inferUnitFromHeader(header)
    const values: (number | undefined)[] = []
    const seenUnits = new Set<string>()
    let validCount = 0
    let pctCount = 0

    for (const row of data.rows) {
      const parsed = parseNumericCell(row[cIdx] ?? '', headerUnit)
      if (parsed.kind === 'numeric' && parsed.value !== undefined) {
        values.push(parsed.value)
        validCount++
        if (parsed.isPercentage) pctCount++
        if (parsed.unit) seenUnits.add(parsed.unit)
      } else {
        values.push(undefined)
      }
    }

    if (validCount === 0) continue

    const hasMixedUnits = seenUnits.size > 1
    const unit = hasMixedUnits ? '' : Array.from(seenUnits)[0] || headerUnit || ''
    const isPercentage = !hasMixedUnits && (unit === '%' || pctCount === validCount)

    metricColumns.push({
      colIndex: cIdx,
      header,
      unit,
      hasMixedUnits,
      mixedUnitLabels: Array.from(seenUnits),
      isPercentage,
      isWeightColumn: isWeightHeader(header),
      isReturnLike: isReturnLikeHeader(header),
      validCount,
      values,
    })
  }

  let defaultMetricColIndex: number | null = null
  if (metricColumns.length > 0) {
    const homogeneous = metricColumns.filter((c) => !c.hasMixedUnits)
    const pool = homogeneous.length > 0 ? homogeneous : metricColumns
    const ranked = [...pool].sort((a, b) => {
      const score = (c: MetricColumnInfo) =>
        (c.isReturnLike ? 40 : 0) + (c.isPercentage && !c.isWeightColumn ? 20 : 0) + (c.validCount === data.rows.length ? 10 : 0) + c.validCount
      const diff = score(b) - score(a)
      return diff !== 0 ? diff : a.colIndex - b.colIndex
    })
    defaultMetricColIndex = ranked[0]?.colIndex ?? null
  }

  return {
    tickerColIndex,
    weightColIndex,
    rowTickers,
    uniqueTickers,
    explicitWeightsByTicker,
    allTickersHaveExplicitWeight,
    metricColumns,
    defaultMetricColIndex,
  }
}

export function allocateDeterministicIntegerWeights(
  uniqueTickers: string[],
  explicitWeightsByTicker: Record<string, number>,
  useExplicit: boolean
): Record<string, number> {
  const result: Record<string, number> = {}
  const n = uniqueTickers.length
  if (n === 0) return result

  if (useExplicit) {
    const rawList = uniqueTickers.map((t) => Math.max(0, explicitWeightsByTicker[t] ?? 0))
    const rawSum = rawList.reduce((acc, w) => acc + w, 0)
    if (rawSum > 0 && Math.abs(rawSum - 100) <= 0.5) {
      // Hamilton largest-remainder rounding to guarantee exact 100% sum
      const scaled = rawList.map((w) => (w / rawSum) * 100)
      const floored = scaled.map((w) => Math.floor(w))
      let remainder = 100 - floored.reduce((acc, w) => acc + w, 0)
      const order = scaled
        .map((w, idx) => ({ idx, frac: w - (floored[idx] ?? 0) }))
        .sort((a, b) => (b.frac !== a.frac ? b.frac - a.frac : a.idx - b.idx))
      for (const item of order) {
        if (remainder <= 0) break
        floored[item.idx] = (floored[item.idx] ?? 0) + 1
        remainder--
      }
      uniqueTickers.forEach((t, idx) => {
        result[t] = floored[idx] ?? 0
      })
      return result
    }
    uniqueTickers.forEach((t, idx) => {
      result[t] = Math.round(rawList[idx] ?? 0)
    })
    return result
  }

  const base = Math.floor(100 / n)
  const remainder = 100 - base * n
  uniqueTickers.forEach((t, idx) => {
    result[t] = base + (idx < remainder ? 1 : 0)
  })
  return result
}

function distributeRemainingWeights(
  tickers: string[],
  targetRemainder: number
): Record<string, number> {
  const out: Record<string, number> = {}
  const n = tickers.length
  if (n === 0) return out
  const safeRem = Math.max(0, Math.round(targetRemainder))
  const base = Math.floor(safeRem / n)
  const extra = safeRem - base * n
  tickers.forEach((t, idx) => {
    out[t] = base + (idx < extra ? 1 : 0)
  })
  return out
}

export function extractTableMetrics(data: TableData, selectedColumnIndex?: number): RowMetric[] {
  const analysis = analyzeTableForChartsAndSimulator(data)
  const chosenCol =
    analysis.metricColumns.find((c) => c.colIndex === selectedColumnIndex) ??
    analysis.metricColumns.find((c) => c.colIndex === analysis.defaultMetricColIndex)

  return data.rows.map((row, rIdx) => {
    const ticker = analysis.rowTickers[rIdx]
    const label = ticker ? `$${ticker}` : stripCellEmphasis(row[0] ?? '') || `Item ${rIdx + 1}`
    const numericValue = chosenCol && !chosenCol.hasMixedUnits ? chosenCol.values[rIdx] : undefined
    return {
      ticker,
      label,
      numericValue,
      isPercentage: Boolean(chosenCol?.isPercentage),
      unit: chosenCol?.unit || undefined,
      rawCells: row,
    }
  })
}

function renderCellContent(
  cell: string,
  opts: {
    isTickerCol: boolean
    isWeightCol: boolean
    onTickerClick?: (ticker: string) => void
  }
) {
  const clean = cell.trim()

  // 1. Positive / Neutral / Zero Percentage (e.g. +71.86%, 60%, 0%)
  if (/^[+-]?\d+(?:\.\d+)?%$/.test(clean)) {
    const numVal = Number(clean.slice(0, -1))
    const isNegative = numVal < 0 || clean.startsWith('-')
    const isUnsigned = !clean.startsWith('+') && !isNegative
    if (isNegative && numVal !== 0) {
      return (
        <span className="inline-flex items-center rounded-md border border-rose-200 bg-rose-50 px-2 py-0.5 font-mono text-xs font-bold text-rose-700 shadow-xs">
          {clean}
        </span>
      )
    }
    if (numVal === 0 && !clean.startsWith('+')) {
      return (
        <span className="inline-flex items-center rounded-md border border-zinc-200 bg-zinc-50 px-2 py-0.5 font-mono text-xs font-semibold text-zinc-700 shadow-2xs">
          {clean}
        </span>
      )
    }
    if (isUnsigned && opts.isWeightCol) {
      return (
        <span className="inline-flex items-center rounded-md border border-sky-200 bg-sky-50 px-2 py-0.5 font-mono text-xs font-semibold text-sky-900 shadow-2xs">
          {clean}
        </span>
      )
    }
    return (
      <span className="inline-flex items-center rounded-md border border-emerald-200 bg-emerald-50 px-2 py-0.5 font-mono text-xs font-bold text-emerald-700 shadow-xs">
        {clean.startsWith('+') ? clean : `+${clean}`}
      </span>
    )
  }

  // 2. Ticker pill (only in identified ticker column or explicit $TICKER notation)
  const sym = opts.isTickerCol
    ? parseTickerToken(clean, { requireExplicitOrCol: true })
    : clean.startsWith('$')
      ? parseTickerToken(clean, { requireExplicitOrCol: false })
      : undefined
  if (sym) {
    return (
      <button
        type="button"
        onClick={() => opts.onTickerClick?.(sym)}
        className="inline-flex items-center gap-1 rounded-lg border border-sky-200/90 bg-sky-50/80 px-2.5 py-0.5 font-mono text-xs font-bold text-sky-950 shadow-xs hover:border-sky-400 hover:bg-sky-100/90 hover:text-sky-900 transition-all cursor-pointer group"
        title={`คลิกเพื่อเปิดชาร์ตราคาและสั่งวิเคราะห์ $${sym}`}
      >
        <span>${sym}</span>
        <span className="text-[10px] text-sky-500 opacity-60 group-hover:opacity-100 transition-opacity">📈</span>
      </button>
    )
  }

  // 3. Currency or numeric values (e.g. 13,065.73, 100, -500.00 THB)
  const parsedNum = parseNumericCell(clean)
  if (parsedNum.kind === 'numeric') {
    const colorCls =
      parsedNum.value !== undefined && parsedNum.value < 0 ? 'text-rose-700' : 'text-zinc-900'
    return <span className={`font-mono text-xs font-semibold tabular-nums ${colorCls}`}>{clean}</span>
  }

  // 4. Default markdown rendering for complex text inside cell (e.g. **BUY**, **HOLD**, notes)
  return (
    <span className="text-sm leading-relaxed text-zinc-800">
      <ReactMarkdown>{clean}</ReactMarkdown>
    </span>
  )
}

function AdvisorTable({
  data,
  isStreaming,
  onActionClick,
}: {
  data: TableData
  isStreaming?: boolean
  onActionClick?: AdvisorActionHandler
}) {
  const { headers, alignments, rows } = data
  const [viewMode, setViewMode] = useState<'table' | 'chart' | 'simulator'>('table')
  const [selectedTicker, setSelectedTicker] = useState<string | null>(null)
  const [selectedMetricColIndex, setSelectedMetricColIndex] = useState<number | null>(null)

  const tableAnalysis = useMemo(() => analyzeTableForChartsAndSimulator(data), [data])
  const {
    tickerColIndex,
    weightColIndex,
    uniqueTickers,
    explicitWeightsByTicker,
    allTickersHaveExplicitWeight,
    metricColumns,
    defaultMetricColIndex,
  } = tableAnalysis

  const activeMetricColumn = useMemo(() => {
    if (selectedMetricColIndex !== null) {
      const explicit = metricColumns.find((c) => c.colIndex === selectedMetricColIndex)
      if (explicit) return explicit
    }
    return metricColumns.find((c) => c.colIndex === defaultMetricColIndex) ?? metricColumns[0] ?? null
  }, [metricColumns, selectedMetricColIndex, defaultMetricColIndex])

  const hasNumeric = metricColumns.length > 0

  const metricRows = useMemo(
    () => extractTableMetrics(data, activeMetricColumn?.colIndex),
    [data, activeMetricColumn]
  )

  const userEditedRef = useRef<Set<string>>(new Set())
  const prevTickersKeyRef = useRef<string>('')
  const prevWeightsSigRef = useRef<string>('')

  const tickersSignature = uniqueTickers.join('|')
  const explicitWeightSignature = uniqueTickers
    .map((t) => `${t}:${explicitWeightsByTicker[t] ?? ''}`)
    .join('|')

  const [weights, setWeights] = useState<Record<string, number>>(() =>
    allocateDeterministicIntegerWeights(uniqueTickers, explicitWeightsByTicker, allTickersHaveExplicitWeight)
  )

  useEffect(() => {
    const tickersChanged = tickersSignature !== prevTickersKeyRef.current
    const weightsSigChanged = explicitWeightSignature !== prevWeightsSigRef.current

    if (!tickersChanged && !weightsSigChanged) {
      return
    }

    prevTickersKeyRef.current = tickersSignature
    prevWeightsSigRef.current = explicitWeightSignature

    // Prune user-edited markers for tickers no longer present
    const validSet = new Set(uniqueTickers)
    for (const editedSym of Array.from(userEditedRef.current)) {
      if (!validSet.has(editedSym)) {
        userEditedRef.current.delete(editedSym)
      }
    }

    setWeights((prev) => {
      if (uniqueTickers.length === 0) return {}

      if (userEditedRef.current.size === 0) {
        return allocateDeterministicIntegerWeights(
          uniqueTickers,
          explicitWeightsByTicker,
          allTickersHaveExplicitWeight
        )
      }

      // Preserve user edits for remaining tickers and reconcile unedited/newly streamed tickers
      const next: Record<string, number> = {}
      const editedTickers: string[] = []
      const uneditedTickers: string[] = []

      for (const sym of uniqueTickers) {
        if (userEditedRef.current.has(sym) && typeof prev[sym] === 'number') {
          next[sym] = prev[sym]
          editedTickers.push(sym)
        } else {
          uneditedTickers.push(sym)
        }
      }

      const editedSum = editedTickers.reduce((acc, sym) => acc + (next[sym] ?? 0), 0)
      if (uneditedTickers.length > 0) {
        if (!tickersChanged && weightsSigChanged) {
          for (const sym of uneditedTickers) {
            if (typeof prev[sym] === 'number') {
              next[sym] = prev[sym]
            } else {
              next[sym] = 0
            }
          }
        } else {
          const remainder = Math.max(0, 100 - editedSum)
          const distributed = distributeRemainingWeights(uneditedTickers, remainder)
          for (const sym of uneditedTickers) {
            next[sym] = distributed[sym] ?? 0
          }
        }
      }

      return next
    })
  }, [
    tickersSignature,
    explicitWeightSignature,
    uniqueTickers,
    explicitWeightsByTicker,
    allTickersHaveExplicitWeight,
  ])

  const totalWeight = useMemo(
    () => uniqueTickers.reduce((acc, sym) => acc + (weights[sym] ?? 0), 0),
    [uniqueTickers, weights]
  )

  const maxAbsMetric = useMemo(() => {
    const vals = metricRows
      .filter((r) => r.numericValue !== undefined)
      .map((r) => Math.abs(r.numericValue as number))
    return vals.length > 0 ? Math.max(...vals, 1e-9) : 1
  }, [metricRows])

  const canSubmitSimulator =
    !isStreaming && uniqueTickers.length > 0 && totalWeight === 100 && typeof onActionClick === 'function'

  const handleApplySimulator = () => {
    if (!canSubmitSimulator || !onActionClick) return
    const summary = uniqueTickers.map((t) => `${t}: ${weights[t] ?? 0}%`).join(', ')
    onActionClick(
      `ช่วยคำนวณแผนการปรับสัดส่วนพอร์ต (Rebalance) ตามสัดส่วนเป้าหมายใหม่ดังนี้:\n${summary}\nโดยใช้ run_sandbox_calculation คิดคำนวณจำนวนหุ้นและมูลค่าเงินที่ต้องซื้อ/ขายด้วย Decimal`,
      ['@Portfolio']
    )
  }

  return (
    <div className="my-4 overflow-hidden rounded-2xl border border-sky-200/90 bg-white/80 shadow-[0_4px_20px_rgba(14,165,233,0.06)] backdrop-blur-md">
      {/* Table Header Controls / View Switcher */}
      {hasNumeric && (
        <div className="flex flex-wrap items-center justify-between gap-2 border-b border-sky-100 bg-gradient-to-r from-sky-50/70 via-cyan-50/40 to-sky-50/70 px-4 py-2 text-xs">
          <div className="flex items-center gap-1.5 font-semibold text-sky-900">
            <span>📊 ตารางข้อมูลและการวิเคราะห์</span>
          </div>

          <div className="flex items-center rounded-lg bg-sky-100/70 p-0.5">
            <button
              type="button"
              onClick={() => setViewMode('table')}
              className={`rounded-md px-2.5 py-1 text-xs font-medium transition-all ${
                viewMode === 'table'
                  ? 'bg-white text-sky-950 shadow-2xs font-semibold'
                  : 'text-sky-700 hover:text-sky-900'
              }`}
            >
              📋 ตาราง
            </button>
            <button
              type="button"
              onClick={() => setViewMode('chart')}
              className={`rounded-md px-2.5 py-1 text-xs font-medium transition-all ${
                viewMode === 'chart'
                  ? 'bg-white text-sky-950 shadow-2xs font-semibold'
                  : 'text-sky-700 hover:text-sky-900'
              }`}
            >
              📊 กราฟเปรียบเทียบ
            </button>
            {uniqueTickers.length > 0 && (
              <button
                type="button"
                onClick={() => setViewMode('simulator')}
                className={`rounded-md px-2.5 py-1 text-xs font-medium transition-all ${
                  viewMode === 'simulator'
                    ? 'bg-white text-sky-950 shadow-2xs font-semibold'
                    : 'text-sky-700 hover:text-sky-900'
                }`}
              >
                🎛️ จำลองสัดส่วน
              </button>
            )}
          </div>
        </div>
      )}

      {/* Mode 1: Table View */}
      {viewMode === 'table' && (
        <div className="overflow-x-auto custom-scrollbar">
          <table className="w-full text-left text-sm border-collapse">
            <thead>
              <tr className="border-b border-sky-100 bg-gradient-to-r from-sky-50/90 via-cyan-50/50 to-sky-50/90 text-sky-950">
                {headers.map((h, i) => (
                  <th
                    key={i}
                    className={`px-4 py-3 font-bold tracking-tight text-sky-950 text-sm ${
                      alignments[i] === 'right'
                        ? 'text-right'
                        : alignments[i] === 'center'
                          ? 'text-center'
                          : 'text-left'
                    }`}
                  >
                    <ReactMarkdown>{h}</ReactMarkdown>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-sky-100/60 bg-white/40">
              {rows.map((row, rIdx) => (
                <tr key={rIdx} className="hover:bg-sky-50/40 transition-colors">
                  {row.map((cell, cIdx) => (
                    <td
                      key={cIdx}
                      className={`px-4 py-3 text-sm ${
                        alignments[cIdx] === 'right'
                          ? 'text-right'
                          : alignments[cIdx] === 'center'
                            ? 'text-center'
                            : 'text-left'
                      }`}
                    >
                      {renderCellContent(cell, {
                        isTickerCol: cIdx === tickerColIndex,
                        isWeightCol: cIdx === weightColIndex,
                        onTickerClick: (sym) => setSelectedTicker((prev) => (prev === sym ? null : sym)),
                      })}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Mode 2: Interactive Comparison Chart View */}
      {viewMode === 'chart' && activeMetricColumn && (
        <div className="p-4 space-y-3 bg-white/50">
          <div className="flex flex-wrap items-center justify-between gap-2 mb-2">
            <p className="text-xs text-zinc-500">
              เปรียบเทียบสัดส่วนและผลลัพธ์ของแต่ละรายการ:{' '}
              <span className="font-semibold text-sky-900">
                {activeMetricColumn.header}
                {activeMetricColumn.unit ? ` (${activeMetricColumn.unit})` : ''}
              </span>
            </p>
            {metricColumns.length > 1 && (
              <label className="flex items-center gap-1.5 text-xs text-zinc-600">
                <span>ตัวชี้วัด:</span>
                <select
                  aria-label="เลือกคอลัมน์ตัวชี้วัด"
                  value={activeMetricColumn.colIndex}
                  onChange={(e) => setSelectedMetricColIndex(Number(e.target.value))}
                  className="rounded-lg border border-sky-200 bg-white px-2 py-1 text-xs font-medium text-sky-950"
                >
                  {metricColumns.map((col) => (
                    <option key={col.colIndex} value={col.colIndex}>
                      {col.header}
                      {col.hasMixedUnits ? ` (${col.mixedUnitLabels.join('/')})` : col.unit ? ` (${col.unit})` : ''}
                    </option>
                  ))}
                </select>
              </label>
            )}
          </div>

          {activeMetricColumn.hasMixedUnits ? (
            <div
              role="status"
              className="rounded-xl border border-amber-200 bg-amber-50/80 p-3 text-xs text-amber-900"
            >
              คอลัมน์ <strong>{activeMetricColumn.header}</strong> มีหน่วย/สกุลเงินผสมกัน (
              {activeMetricColumn.mixedUnitLabels.join(', ')}) จึงงดแสดงแถบเปรียบเทียบบนสเกลเดียวกัน
            </div>
          ) : (
            <div className="space-y-2.5">
              {metricRows.map((metric, idx) => {
                const val = metric.numericValue
                const isMissing = val === undefined
                const isPos = (val ?? 0) >= 0
                const widthPct =
                  isMissing || val === 0
                    ? 0
                    : Math.min(100, Math.max(2, Math.round((Math.abs(val) / maxAbsMetric) * 100)))

                const formattedValue = (() => {
                  if (isMissing) return 'ไม่มีข้อมูล'
                  if (activeMetricColumn.isPercentage) {
                    return val > 0 && !activeMetricColumn.isWeightColumn ? `+${val}%` : `${val}%`
                  }
                  const numStr = val.toLocaleString()
                  return activeMetricColumn.unit ? `${numStr} ${activeMetricColumn.unit}` : numStr
                })()

                return (
                  <div
                    key={idx}
                    role={metric.ticker ? 'button' : undefined}
                    tabIndex={metric.ticker ? 0 : undefined}
                    onClick={() => {
                      if (metric.ticker) {
                        setSelectedTicker((prev) => (prev === metric.ticker ? null : metric.ticker!))
                      }
                    }}
                    onKeyDown={(e) => {
                      if (metric.ticker && (e.key === 'Enter' || e.key === ' ')) {
                        e.preventDefault()
                        setSelectedTicker((prev) => (prev === metric.ticker ? null : metric.ticker!))
                      }
                    }}
                    className={`flex items-center gap-3 p-2 rounded-xl transition-all ${
                      metric.ticker ? 'hover:bg-sky-50/60 cursor-pointer focus:outline-none focus:ring-2 focus:ring-sky-300' : ''
                    }`}
                  >
                    <div className="w-24 shrink-0 font-mono text-xs font-bold text-sky-950 truncate">
                      {metric.label}
                    </div>
                    <div className="flex-1 bg-sky-50 h-6 rounded-lg overflow-hidden flex items-center p-0.5">
                      {isMissing ? (
                        <span className="px-2 text-[11px] italic text-zinc-400">ไม่มีข้อมูล</span>
                      ) : (
                        <div
                          data-bar-width={widthPct}
                          style={{ width: `${widthPct}%` }}
                          className={`h-full rounded-md transition-all duration-500 ${
                            val === 0
                              ? 'bg-transparent'
                              : isPos
                                ? 'bg-emerald-500/80 shadow-xs'
                                : 'bg-rose-500/80 shadow-xs'
                          }`}
                        />
                      )}
                    </div>
                    <div className="w-28 shrink-0 text-right font-mono text-xs font-bold">
                      <span
                        className={
                          isMissing
                            ? 'text-zinc-400 font-normal'
                            : val === 0
                              ? 'text-zinc-600'
                              : isPos
                                ? 'text-emerald-700'
                                : 'text-rose-700'
                        }
                      >
                        {formattedValue}
                      </span>
                    </div>
                  </div>
                )
              })}
            </div>
          )}
        </div>
      )}

      {/* Mode 3: Interactive Rebalance Simulator */}
      {viewMode === 'simulator' && (
        <div className="p-4 space-y-4 bg-white/60">
          <div className="flex items-center justify-between">
            <h4 className="text-xs font-bold text-sky-950 uppercase tracking-wider">
              🎛️ จำลองปรับสัดส่วนเป้าหมาย (Weight Simulator)
            </h4>
            <span
              className={`rounded-full px-2.5 py-0.5 text-xs font-mono font-bold ${
                totalWeight === 100
                  ? 'bg-emerald-100 text-emerald-800'
                  : 'bg-amber-100 text-amber-800'
              }`}
            >
              รวม: {totalWeight}%
            </span>
          </div>

          <div className="space-y-3.5">
            {uniqueTickers.map((sym) => {
              const currentW = weights[sym] ?? 0
              return (
                <div key={sym} className="space-y-1">
                  <div className="flex items-center justify-between text-xs">
                    <span className="font-mono font-bold text-sky-950">${sym}</span>
                    <span className="font-mono font-bold text-sky-700">{currentW}%</span>
                  </div>
                  <input
                    type="range"
                    aria-label={`สัดส่วนเป้าหมาย ${sym}`}
                    min="0"
                    max="100"
                    step="1"
                    disabled={Boolean(isStreaming)}
                    value={currentW}
                    onChange={(e) => {
                      const parsed = parseInt(e.target.value, 10)
                      const clamped = Number.isFinite(parsed) ? Math.max(0, Math.min(100, parsed)) : 0
                      userEditedRef.current.add(sym)
                      setWeights((prev) => ({ ...prev, [sym]: clamped }))
                    }}
                    className="w-full accent-sky-500 h-1.5 rounded-lg bg-sky-100 cursor-pointer disabled:opacity-50"
                  />
                </div>
              )
            })}
          </div>

          <div className="flex items-center justify-between gap-3 pt-2 border-t border-sky-100">
            <p className="text-xs text-zinc-500">
              {isStreaming
                ? '⏳ รอข้อมูลสตรีมครบก่อนส่งคำนวณ'
                : !onActionClick
                  ? 'ℹ️ โหมดอ่านอย่างเดียว (ไม่สามารถส่งคำสั่งแชทได้)'
                  : totalWeight !== 100
                    ? '⚠️ สัดส่วนต้องรวมกันได้ 100%'
                    : '✅ สัดส่วนครบ 100% พร้อมส่งคำนวณ'}
            </p>
            <button
              type="button"
              disabled={!canSubmitSimulator}
              onClick={handleApplySimulator}
              className="rounded-xl border border-sky-300 bg-sky-600 hover:bg-sky-700 disabled:cursor-not-allowed disabled:opacity-50 text-white px-3.5 py-1.5 text-xs font-semibold shadow-xs transition-colors"
            >
              🚀 ส่งแผนสัดส่วนให้ Hermes คำนวณ
            </button>
          </div>
        </div>
      )}

      {/* Inline Stock Inspector Drawer (เมื่อคลิกที่ $TICKER) */}
      {selectedTicker && (
        <div className="border-t border-sky-100 bg-gradient-to-b from-sky-50/40 via-white to-sky-50/20 p-4 transition-all">
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center gap-2">
              <span className="font-bold text-sky-950 text-base">${selectedTicker}</span>
              <span className="rounded-full bg-sky-100 px-2.5 py-0.5 text-[11px] font-medium text-sky-700">
                Interactive Chart Drawer
              </span>
            </div>
            <button
              type="button"
              onClick={() => setSelectedTicker(null)}
              className="text-zinc-400 hover:text-zinc-600 text-xs px-2 py-1 rounded-md hover:bg-sky-100/50"
            >
              ✕ ปิด
            </button>
          </div>

          <div className="mb-3">
            <TradingViewMiniWidget symbol={selectedTicker} title={`กราฟราคา ${selectedTicker}`} />
          </div>

          <div className="flex flex-wrap items-center gap-2">
            <button
              type="button"
              disabled={!onActionClick || Boolean(isStreaming)}
              onClick={() =>
                onActionClick?.(
                  `ช่วยวิเคราะห์เจาะลึก Valuation, DCF และความเสี่ยงของหุ้น ${selectedTicker}`,
                  ['@Portfolio', `$${selectedTicker}`]
                )
              }
              className="rounded-xl border border-sky-200 bg-sky-50 px-3 py-1.5 text-xs font-semibold text-sky-800 hover:bg-sky-100 disabled:opacity-50 disabled:cursor-not-allowed transition-colors shadow-2xs"
            >
              🔬 วิเคราะห์ Valuation & DCF (${selectedTicker})
            </button>
            <button
              type="button"
              disabled={!onActionClick || Boolean(isStreaming)}
              onClick={() =>
                onActionClick?.(
                  `ช่วยตรวจสุขภาพ Thesis และติดตามสัญญาผู้บริหารของหุ้น ${selectedTicker}`,
                  ['@Portfolio', `$${selectedTicker}`]
                )
              }
              className="rounded-xl border border-sky-200 bg-white px-3 py-1.5 text-xs font-semibold text-sky-700 hover:bg-sky-50 disabled:opacity-50 disabled:cursor-not-allowed transition-colors shadow-2xs"
            >
              🛡️ ตรวจสุขภาพ Thesis (${selectedTicker})
            </button>
          </div>
        </div>
      )}
    </div>
  )
}

interface WidgetBoundaryProps {
  children: React.ReactNode
}

interface WidgetBoundaryState {
  hasError: boolean
}

class WidgetErrorBoundary extends Component<WidgetBoundaryProps, WidgetBoundaryState> {
  constructor(props: WidgetBoundaryProps) {
    super(props)
    this.state = { hasError: false }
  }

  static getDerivedStateFromError(): WidgetBoundaryState {
    return { hasError: true }
  }

  render() {
    if (this.state.hasError) {
      return (
        <div
          role="alert"
          className="my-3 rounded-xl border border-amber-200 bg-amber-50/70 p-3 text-xs text-amber-900"
        >
          ไม่สามารถแสดงผล Interactive Widget นี้ได้ (ข้อผิดพลาดภายใน Widget)
        </div>
      )
    }
    return this.props.children
  }
}

function RebalanceSimulatorWidget({
  title,
  targets,
  isStreaming,
  onActionClick,
}: {
  title?: string
  targets: RebalanceTargetItem[]
  isStreaming?: boolean
  onActionClick?: AdvisorActionHandler
}) {
  const initialMap = useMemo(() => {
    const raw: Record<string, number> = {}
    targets.forEach((t) => {
      raw[t.symbol] = t.targetWeightPct
    })
    return allocateDeterministicIntegerWeights(
      targets.map((t) => t.symbol),
      raw,
      true
    )
  }, [targets])

  const [weights, setWeights] = useState<Record<string, number>>(initialMap)
  useEffect(() => {
    setWeights(initialMap)
  }, [initialMap])

  const totalWeight = useMemo(
    () => targets.reduce((acc, t) => acc + (weights[t.symbol] ?? 0), 0),
    [targets, weights]
  )
  const canSubmit = !isStreaming && totalWeight === 100 && typeof onActionClick === 'function'

  return (
    <div className="my-4 rounded-2xl border border-sky-200/80 bg-white/85 p-4 space-y-3 shadow-xs">
      <div className="flex items-center justify-between">
        <h4 className="text-xs font-bold text-sky-950">{title || '🎛️ จำลองปรับสัดส่วนพอร์ต'}</h4>
        <span
          className={`rounded-full px-2.5 py-0.5 text-xs font-mono font-bold ${
            totalWeight === 100 ? 'bg-emerald-100 text-emerald-800' : 'bg-amber-100 text-amber-800'
          }`}
        >
          รวม: {totalWeight}%
        </span>
      </div>
      <div className="space-y-2.5">
        {targets.map((item) => {
          const w = weights[item.symbol] ?? 0
          return (
            <div key={item.symbol} className="space-y-1">
              <div className="flex items-center justify-between text-xs">
                <span className="font-mono font-bold text-sky-950">${item.symbol}</span>
                <span className="font-mono font-bold text-sky-700">{w}%</span>
              </div>
              <input
                type="range"
                aria-label={`Widget สัดส่วน ${item.symbol}`}
                min="0"
                max="100"
                step="1"
                disabled={Boolean(isStreaming)}
                value={w}
                onChange={(e) => {
                  const val = parseInt(e.target.value, 10)
                  setWeights((prev) => ({
                    ...prev,
                    [item.symbol]: Number.isFinite(val) ? Math.max(0, Math.min(100, val)) : 0,
                  }))
                }}
                className="w-full accent-sky-500 h-1.5 rounded-lg bg-sky-100 cursor-pointer"
              />
            </div>
          )
        })}
      </div>
      <div className="flex items-center justify-between pt-2 border-t border-sky-100">
        <span className="text-xs text-zinc-500">
          {totalWeight !== 100 ? '⚠️ สัดส่วนต้องรวมกันได้ 100%' : '✅ พร้อมส่งคำนวณ'}
        </span>
        <button
          type="button"
          disabled={!canSubmit}
          onClick={() => {
            if (!canSubmit || !onActionClick) return
            const summary = targets.map((t) => `${t.symbol}: ${weights[t.symbol] ?? 0}%`).join(', ')
            onActionClick(
              `ช่วยคำนวณแผนการปรับสัดส่วนพอร์ต (Rebalance) ตามสัดส่วนเป้าหมายใหม่ดังนี้:\n${summary}\nโดยใช้ run_sandbox_calculation คิดคำนวณจำนวนหุ้นและมูลค่าเงินที่ต้องซื้อ/ขายด้วย Decimal`,
              ['@Portfolio']
            )
          }}
          className="rounded-xl border border-sky-300 bg-sky-600 hover:bg-sky-700 disabled:opacity-50 disabled:cursor-not-allowed text-white px-3 py-1.5 text-xs font-semibold"
        >
          🚀 ส่งแผนสัดส่วนให้ Hermes คำนวณ
        </button>
      </div>
    </div>
  )
}

function AdvisorWidgetBlock({
  segment,
  isStreaming,
  onActionClick,
}: {
  segment: Extract<ContentSegment, { type: 'widget' }>
  isStreaming?: boolean
  onActionClick?: AdvisorActionHandler
}) {
  if (!segment.isComplete) {
    if (isStreaming) {
      return (
        <div className="my-4 rounded-2xl border border-sky-200/80 bg-sky-50/50 p-4 animate-pulse">
          <div className="flex items-center gap-2 text-sky-800 text-xs font-medium">
            <span className="w-2 h-2 rounded-full bg-flow-blue animate-ping" />
            <span>กำลังสตรีม Interactive Widget...</span>
          </div>
          <div className="mt-3 h-28 bg-white/70 rounded-xl" />
        </div>
      )
    }
    return (
      <div
        role="alert"
        className="my-3 rounded-xl border border-amber-200 bg-amber-50/70 p-3.5 text-xs text-amber-900"
      >
        <p className="font-semibold">⚠️ ไม่สามารถแสดง Interactive Widget ได้ (ข้อมูลสตรีมไม่สมบูรณ์)</p>
      </div>
    )
  }

  if (!segment.isValid || !segment.data) {
    return (
      <div
        role="alert"
        className="my-3 rounded-xl border border-amber-200 bg-amber-50/70 p-3.5 text-xs text-amber-900"
      >
        <p className="font-semibold">
          {segment.title ? `${segment.title}: ` : ''}
          {segment.error || 'ไม่สามารถแสดง Interactive Widget ได้ (รูปแบบข้อมูลไม่ถูกต้อง)'}
        </p>
      </div>
    )
  }

  const { widgetType, title, payload } = segment.data

  if (widgetType === 'gauge') {
    const val = payload.value as number
    const min = typeof payload.min === 'number' ? payload.min : -3
    const max = typeof payload.max === 'number' ? payload.max : 3
    return (
      <WidgetErrorBoundary>
        <div className="my-4">
          <RadialGauge
            value={val}
            min={min}
            max={max}
            title={title || 'มาตรวัดดัชนี'}
            subtitle={typeof payload.subtitle === 'string' ? payload.subtitle : undefined}
            unit={typeof payload.unit === 'string' ? payload.unit : ''}
            zones={Array.isArray(payload.zones) ? (payload.zones as GaugeZone[]) : undefined}
            subItems={Array.isArray(payload.subItems) ? (payload.subItems as GaugeSubItem[]) : undefined}
            asOfDate={typeof payload.asOfDate === 'string' ? payload.asOfDate : undefined}
            dataLagNote={typeof payload.dataLagNote === 'string' ? payload.dataLagNote : undefined}
          />
        </div>
      </WidgetErrorBoundary>
    )
  }

  if (widgetType === 'rebalance_simulator') {
    const normalizedTargets = Array.isArray(payload.normalizedTargets)
      ? (payload.normalizedTargets as RebalanceTargetItem[])
      : []
    return (
      <WidgetErrorBoundary>
        <RebalanceSimulatorWidget
          title={title}
          targets={normalizedTargets}
          isStreaming={isStreaming}
          onActionClick={onActionClick}
        />
      </WidgetErrorBoundary>
    )
  }

  return null
}

const markdownComponents: Components = {
  h1: ({ children }) => (
    <h3 className="mt-5 mb-2.5 text-lg sm:text-xl font-bold text-sky-950 first:mt-0">
      {children}
    </h3>
  ),
  h2: ({ children }) => (
    <h4 className="mt-4 mb-2 text-base sm:text-lg font-bold text-sky-950 first:mt-0">
      {children}
    </h4>
  ),
  h3: ({ children }) => (
    <h5 className="mt-3.5 mb-1.5 text-sm sm:text-base font-bold text-sky-900 tracking-wide first:mt-0">
      {children}
    </h5>
  ),
  p: ({ children }) => (
    <p className="mt-2.5 text-sm sm:text-[15px] leading-relaxed text-zinc-800 first:mt-0">
      {React.Children.map(children, renderWithEvidenceBadges)}
    </p>
  ),
  strong: ({ children }) => (
    <strong className="font-bold text-sky-950">
      {children}
    </strong>
  ),
  em: ({ children }) => (
    <em className="italic text-zinc-600">
      {children}
    </em>
  ),
  ul: ({ children }) => (
    <ul className="my-2.5 space-y-2 pl-1">
      {children}
    </ul>
  ),
  ol: ({ children }) => (
    <ol className="my-2.5 space-y-2 pl-4 list-decimal text-zinc-800 text-sm sm:text-[15px]">
      {children}
    </ol>
  ),
  li: ({ children }) => (
    <li className="ml-4 list-disc pl-1 text-sm sm:text-[15px] leading-relaxed text-zinc-800 marker:text-sky-400">
      {React.Children.map(children, renderWithEvidenceBadges)}
    </li>
  ),
  hr: () => (
    <hr className="my-4 border-t border-sky-100" />
  ),
  code: ({ children }) => {
    const isMultiLine = typeof children === 'string' && children.includes('\n')
    if (isMultiLine) {
      return (
        <pre className="my-3 overflow-x-auto rounded-xl border border-sky-100 bg-sky-50/50 p-3.5 font-mono text-xs sm:text-sm text-sky-950 custom-scrollbar">
          <code>{children}</code>
        </pre>
      )
    }
    return (
      <code className="rounded-md border border-sky-100 bg-sky-50/80 px-1.5 py-0.5 font-mono text-[0.88em] font-semibold text-sky-900">
        {children}
      </code>
    )
  },
  blockquote: ({ children }) => (
    <blockquote className="my-3 rounded-r-xl border-l-3 border-sky-400 bg-sky-50/40 py-2 pl-3.5 pr-2.5 text-sm italic leading-relaxed text-zinc-700">
      {children}
    </blockquote>
  ),
  a: ({ href, children }) => (
    <a
      href={href}
      target="_blank"
      rel="noreferrer"
      className="font-medium text-sky-700 underline underline-offset-2 hover:text-sky-900 transition-colors"
    >
      {children}
    </a>
  ),
}

export default function AdvisorMarkdown({
  content,
  isStreaming,
  onActionClick,
}: {
  content: string
  isStreaming?: boolean
  onActionClick?: AdvisorActionHandler
}) {
  const segments = splitMarkdownAndTables(content)

  return (
    <div className="advisor-markdown break-words">
      {segments.map((seg, idx) => {
        if (seg.type === 'table') {
          return (
            <AdvisorTable
              key={idx}
              data={seg.data}
              isStreaming={isStreaming}
              onActionClick={onActionClick}
            />
          )
        }
        if (seg.type === 'widget') {
          return (
            <AdvisorWidgetBlock
              key={idx}
              segment={seg}
              isStreaming={isStreaming}
              onActionClick={onActionClick}
            />
          )
        }
        return (
          <ReactMarkdown key={idx} components={markdownComponents}>
            {seg.content}
          </ReactMarkdown>
        )
      })}
      {isStreaming && (
        <span className="inline-block h-4 w-1.5 translate-y-0.5 rounded-sm bg-flow-blue animate-stream-caret ml-1" />
      )}
    </div>
  )
}
