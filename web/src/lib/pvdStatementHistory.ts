import type { TransactionItemDTO } from '../api/types'

export type TransactionView = 'assets' | 'pvd'
export type PvdIssue = { code: string; message: string; transactionIds: string[] }
export type PvdFundRow = {
  symbol: string
  code: string
  units: number
  nav: number
  rawValueCents: bigint
  adjustmentCents: bigint
  valueCents: bigint
}
export type PvdFundComparison = {
  code: string
  previous: PvdFundRow | null
  current: PvdFundRow | null
  deltaCents: bigint | null
  status: 'baseline' | 'new' | 'zero' | 'zero_both' | 'increase' | 'decrease' | 'unchanged' | 'missing' | 'unavailable'
}
export type PvdStatementGroup = {
  id: string
  scopeKey: string | null
  asOfDate: string | null
  archiveId: string | null
  totalNavCents: bigint | null
  isValid: boolean
  funds: PvdFundRow[]
  issues: PvdIssue[]
  previousId: string | null
  deltaNavCents: bigint | null
  comparisons: PvdFundComparison[]
}
export type PvdHistory = {
  statements: PvdStatementGroup[]
  unresolvedIssues: PvdIssue[]
  latestStatementId: string | null
  summaryUnavailableReason: string | null
}

type Metadata = Record<string, unknown>
type ParsedRow = { tx: TransactionItemDTO; metadata: Metadata; groupId: string; scopeKey: string | null; date: string | null }

const sha256 = /^[a-f0-9]{64}$/i
const dateOnly = /^\d{4}-\d{2}-\d{2}$/
const normalize = (value: unknown) => String(value ?? '').trim().toUpperCase()
const safeId = (tx: TransactionItemDTO) => String(tx.transaction_id ?? '').trim() || '(missing transaction id)'

export function isPvdTransaction(tx: TransactionItemDTO): boolean {
  return normalize(tx.action) === 'PVD_SNAPSHOT'
    || normalize(tx.source) === 'BBLAM_PVD'
    || normalize(tx.asset_type) === 'PVD'
    || normalize(tx.symbol).startsWith('PVD_')
}

export function partitionTransactions(rows: readonly TransactionItemDTO[]) {
  const assets: TransactionItemDTO[] = []
  const pvd: TransactionItemDTO[] = []
  for (const tx of rows) (isPvdTransaction(tx) ? pvd : assets).push(tx)
  return { assets, pvd }
}

export function parseMoneyCents(value: string): bigint {
  if (typeof value !== 'string' || !/^-?\d+(?:\.\d{1,2})?$/.test(value.trim())) {
    throw new Error('Money must be a decimal string with at most two fractional digits')
  }
  const text = value.trim()
  const negative = text.startsWith('-')
  const unsigned = negative ? text.slice(1) : text
  const [whole = '0', fraction = ''] = unsigned.split('.')
  const cents = BigInt(whole) * 100n + BigInt((fraction + '00').slice(0, 2))
  return negative ? -cents : cents
}

export function formatMoneyCents(value: bigint, showPositiveSign = false): string {
  const negative = value < 0n
  const abs = negative ? -value : value
  const amount = `${abs / 100n}.${String(abs % 100n).padStart(2, '0')}`
  return `${negative ? '-' : showPositiveSign && value > 0n ? '+' : ''}${amount}`
}

function addIssue(target: PvdIssue[], code: string, message: string, ids: string[]) {
  const issue = target.find((item) => item.code === code && item.message === message)
  const uniqueIds = [...new Set(ids.filter(Boolean))].sort()
  if (issue) issue.transactionIds = [...new Set([...issue.transactionIds, ...uniqueIds])].sort()
  else target.push({ code, message, transactionIds: uniqueIds })
}

function validDate(value: unknown): value is string {
  if (typeof value !== 'string' || !dateOnly.test(value)) return false
  const parsed = new Date(`${value}T00:00:00.000Z`)
  return Number.isFinite(parsed.getTime()) && parsed.toISOString().slice(0, 10) === value
}

function metadataFor(tx: TransactionItemDTO): Metadata | null {
  if (!tx.pvd_metadata_json) return null
  try {
    const parsed: unknown = JSON.parse(tx.pvd_metadata_json)
    return parsed && typeof parsed === 'object' && !Array.isArray(parsed) ? parsed as Metadata : null
  } catch {
    return null
  }
}

function centsField(value: unknown): bigint {
  if (typeof value !== 'string') throw new Error('Missing or non-string monetary value')
  return parseMoneyCents(value)
}

function canonicalMetadata(metadata: Metadata): string {
  const keys = [
    'payload_version', 'fund_code', 'member_hash', 'as_of_date', 'period_start', 'period_end',
    'masked_member_id',
    'total_contribution', 'total_nav', 'member_principal', 'member_benefit', 'employer_principal',
    'employer_benefit', 'employer_vesting_percent', 'member_vested_amount', 'employer_vested_amount',
    'source_pdf_sha256', 'semantic_fingerprint', 'archive_id', 'reconciliation_adjustment',
  ]
  return JSON.stringify(keys.map((key) => [key, metadata[key] ?? null]))
}

function rowCanonical(tx: TransactionItemDTO) {
  const metadata = metadataFor(tx)
  const money = (value: string | null | undefined) => {
    if (value === null || value === undefined) return null
    if (typeof value !== 'string') return `invalid:${String(value)}`
    if (value.trim() === '') return null
    try { return parseMoneyCents(value).toString() } catch { return value.trim() }
  }
  return JSON.stringify([
    String(tx.timestamp ?? '').trim().slice(0, 10), normalize(tx.symbol), normalize(tx.action), tx.units, tx.price, normalize(tx.currency),
    money(tx.gross_amount), money(tx.net_amount), String(tx.confirmation_no ?? '').trim().toLowerCase(), normalize(tx.source),
    String(tx.fingerprint ?? '').trim().toLowerCase(), normalize(tx.asset_type), String(tx.bucket_id ?? '').trim(), money(tx.reconciliation_adjustment),
    metadata ? canonicalMetadata(metadata) : String(tx.pvd_metadata_json ?? '').trim(),
  ])
}

export function buildPvdStatementHistory(portfolioId: string, rows: readonly TransactionItemDTO[]): PvdHistory {
  const pvdRows = partitionTransactions(rows).pvd
  const unresolvedIssues: PvdIssue[] = []
  const grouped = new Map<string, ParsedRow[]>()
  const canonicalById = new Map<string, string>()

  for (const [rowIndex, tx] of pvdRows.entries()) {
    const id = safeId(tx)
    const actualId = String(tx.transaction_id ?? '').trim()
    const canonical = rowCanonical(tx)
    if (!actualId) {
      addIssue(unresolvedIssues, 'missing_transaction_id', 'รายการ PVD ไม่มี transaction ID จึงตรวจสอบประวัติซ้ำไม่ได้', [`${id} #${rowIndex + 1}`])
    } else {
      const prior = canonicalById.get(actualId)
      if (prior !== undefined) {
        if (prior !== canonical) addIssue(unresolvedIssues, 'duplicate_transaction_id', 'พบ transaction ID ซ้ำแต่ข้อมูลไม่ตรงกัน', [id])
        continue
      }
      canonicalById.set(actualId, canonical)
    }

    const metadata = metadataFor(tx)
    const confirmation = String(tx.confirmation_no ?? '').trim().toLowerCase()
    const metadataFingerprint = String(metadata?.semantic_fingerprint ?? '').trim().toLowerCase()
    const groupFingerprint = metadataFingerprint || confirmation
    if (!metadata) {
      addIssue(unresolvedIssues, 'invalid_metadata', 'อ่านข้อมูลกำกับ Statement ไม่ได้ จึงยืนยันงวดและยอดล่าสุดไม่ได้', [id])
      continue
    }
    if (!sha256.test(groupFingerprint)) {
      addIssue(unresolvedIssues, 'invalid_fingerprint', 'รหัสอ้างอิง Statement ไม่ใช่ SHA-256 ที่ถูกต้อง', [id])
      continue
    }
    const fundCode = String(metadata.fund_code ?? '').trim().toUpperCase()
    const memberHash = String(metadata.member_hash ?? '').trim().toLowerCase()
    const date = metadata.as_of_date
    if (!fundCode || !sha256.test(memberHash) || !validDate(date)) {
      addIssue(unresolvedIssues, 'unresolved_identity', 'ข้อมูลกองทุน สมาชิก หรือวันที่ของ Statement ไม่ครบหรือไม่ถูกต้อง', [id])
      continue
    }
    const key = `${portfolioId}\u0000${groupFingerprint}`
    const list = grouped.get(key) ?? []
    list.push({ tx, metadata, groupId: groupFingerprint, scopeKey: `${portfolioId}\u0000${fundCode}\u0000${memberHash}`, date })
    grouped.set(key, list)
  }

  const statementGroups: PvdStatementGroup[] = []
  for (const entries of grouped.values()) {
    const fingerprint = entries[0]!.groupId
    const ids = entries.map(({ tx }) => safeId(tx))
    const issues: PvdIssue[] = []
    let isValid = true
    const invalidate = (code: string, message: string, transactionIds = ids) => {
      isValid = false
      addIssue(issues, code, message, transactionIds)
    }
    const first = entries[0]!
    const metadataCanonical = canonicalMetadata(first.metadata)
    for (const entry of entries) {
      const { tx, metadata } = entry
      if (canonicalMetadata(metadata) !== metadataCanonical) invalidate('metadata_conflict', 'ข้อมูลกำกับของแถวใน Statement เดียวกันไม่ตรงกัน', [safeId(tx)])
      if (entry.scopeKey !== first.scopeKey || entry.date !== first.date) invalidate('identity_conflict', 'แถวใน Statement มีขอบเขตสมาชิกหรือวันที่ไม่ตรงกัน', [safeId(tx)])
      if (String(metadata.semantic_fingerprint ?? '').trim().toLowerCase() !== fingerprint) invalidate('fingerprint_conflict', 'fingerprint ใน metadata ไม่ตรงกับตัวตนของ Statement', [safeId(tx)])
      if (String(tx.confirmation_no ?? '').trim().toLowerCase() !== fingerprint) invalidate('confirmation_conflict', 'confirmation number ของแถวไม่ตรงกับ Statement', [safeId(tx)])
      if (normalize(tx.action) !== 'PVD_SNAPSHOT' || normalize(tx.source) !== 'BBLAM_PVD' || normalize(tx.asset_type) !== 'PVD' || normalize(tx.currency) !== 'THB') {
        invalidate('row_type_conflict', 'ประเภทธุรกรรม PVD ไม่ตรงกับสัญญาข้อมูล', [safeId(tx)])
      }
      if (!sha256.test(String(tx.fingerprint ?? ''))) invalidate('invalid_row_fingerprint', 'fingerprint ของแถวไม่ใช่ SHA-256 ที่ถูกต้อง', [safeId(tx)])
      const timestampDate = String(tx.timestamp ?? '').slice(0, 10)
      if (timestampDate !== entry.date) invalidate('row_date_conflict', 'วันที่ของแถวไม่ตรงกับวันที่ Statement', [safeId(tx)])
      if (String(metadata.payload_version) !== '1') invalidate('unsupported_metadata', 'ไม่รองรับรุ่น metadata ของ Statement นี้', [safeId(tx)])
      for (const periodKey of ['period_start', 'period_end']) {
        const periodDate = metadata[periodKey]
        if (periodDate !== undefined && periodDate !== null && !validDate(periodDate)) invalidate('invalid_period_date', 'ช่วงวันที่ใน metadata ไม่ถูกต้อง', [safeId(tx)])
      }
      if (validDate(metadata.period_start) && validDate(metadata.period_end) && metadata.period_start > metadata.period_end) invalidate('invalid_period_range', 'วันเริ่มต้นของ Statement อยู่หลังวันสิ้นสุด', [safeId(tx)])
      if (!sha256.test(String(metadata.source_pdf_sha256 ?? '')) || !sha256.test(String(metadata.archive_id ?? ''))) {
        invalidate('invalid_archive_reference', 'hash ของ PDF หรือ archive reference ไม่ถูกต้อง จึงเปิดเอกสารไม่ได้', [safeId(tx)])
      }
      if (String(metadata.archive_id ?? '').trim().toLowerCase() !== String(metadata.source_pdf_sha256 ?? '').trim().toLowerCase()) {
        invalidate('archive_hash_mismatch', 'archive reference ไม่ตรงกับ hash ของ PDF ต้นฉบับ', [safeId(tx)])
      }
    }

    const funds: PvdFundRow[] = []
    const seenCodes = new Map<string, string>()
    const missingAdjustmentIds: string[] = []
    let rawTotal = 0n
    let adjustmentTotal = 0n
    const metadataAdjustment = (() => {
      if (first.metadata.reconciliation_adjustment === undefined || first.metadata.reconciliation_adjustment === null || first.metadata.reconciliation_adjustment === '') return 0n
      try { return centsField(first.metadata.reconciliation_adjustment) } catch { return null }
    })()
    let totalNavCents: bigint | null = null
    try { totalNavCents = centsField(first.metadata.total_nav) } catch { invalidate('invalid_total_nav', 'ยอด NAV รวมใน metadata ไม่ใช่จำนวนเงินที่ถูกต้อง') }
    if (totalNavCents !== null && totalNavCents < 0n) invalidate('negative_total_nav', 'ยอด NAV รวมต้องไม่ติดลบ')
    try {
      const contribution = centsField(first.metadata.total_contribution)
      if (contribution < 0n) invalidate('negative_contribution', 'ยอดเงินสมทบต้องไม่ติดลบ')
    } catch { invalidate('invalid_contribution', 'ยอดเงินสมทบใน metadata ไม่ใช่จำนวนเงินที่ถูกต้อง') }

    for (const { tx } of entries) {
      const id = safeId(tx)
      const code = normalize(tx.symbol).startsWith('PVD_') ? normalize(tx.symbol).slice(4) : ''
      if (!code) invalidate('invalid_fund_symbol', 'symbol ของรายการ PVD ไม่มีรหัสกองทุนย่อย', [id])
      const oldId = seenCodes.get(code)
      if (oldId) invalidate('duplicate_fund_code', 'มีรหัสกองทุนซ้ำใน Statement เดียวกัน', [oldId, id])
      else seenCodes.set(code, id)

      let rawValue: bigint | null = null
      try {
        const gross = tx.gross_amount
        const net = tx.net_amount
        const hasGross = gross !== null && gross !== undefined && gross !== ''
        const hasNet = net !== null && net !== undefined && net !== ''
        if (hasGross) rawValue = parseMoneyCents(gross)
        else if (normalize(tx.source) === 'BBLAM_PVD' && normalize(tx.currency) === 'THB') rawValue = parseMoneyCents(net ?? '')
        else throw new Error('No validated legacy amount')
        if (hasNet) {
          const netCents = parseMoneyCents(net)
          if (hasGross && netCents !== rawValue) invalidate('gross_net_mismatch', 'gross amount and net amount differ', [id])
        }
        if (rawValue < 0n) invalidate('negative_fund_value', 'Fund value before adjustment cannot be negative', [id])
      } catch {
        invalidate('invalid_fund_value', 'มูลค่ากองทุนในแถวไม่ใช่จำนวนเงินสองตำแหน่ง', [id])
      }
      let adjustment: bigint | null = null
      try {
        if (tx.reconciliation_adjustment === null || tx.reconciliation_adjustment === undefined || tx.reconciliation_adjustment === '') {
          missingAdjustmentIds.push(id)
          adjustment = 0n
        } else adjustment = parseMoneyCents(tx.reconciliation_adjustment)
      } catch {
        invalidate('invalid_adjustment', 'ยอดปรับกระทบยอดในแถวไม่ใช่จำนวนเงินสองตำแหน่ง', [id])
      }
      const units = Number(tx.units)
      const nav = Number(tx.price)
      if (!Number.isFinite(units) || units < 0 || !Number.isFinite(nav) || nav < 0) invalidate('invalid_fund_measure', 'จำนวนหน่วยหรือ NAV ของกองทุนไม่ถูกต้อง', [id])
      const symbolCode = normalize(tx.symbol).slice(4)
      if (metadataAdjustment === null) invalidate('invalid_metadata_adjustment', 'ยอดปรับกระทบยอดรวมใน metadata ไม่ถูกต้อง')
      if (rawValue !== null && adjustment !== null) {
        rawTotal += rawValue
        adjustmentTotal += adjustment
        const valueCents = rawValue + adjustment
        if (valueCents < 0n) invalidate('negative_adjusted_value', 'มูลค่ากองทุนหลังปรับต้องไม่ติดลบ', [id])
        funds.push({ symbol: tx.symbol, code: symbolCode, units, nav, rawValueCents: rawValue, adjustmentCents: adjustment, valueCents })
      }
    }
    if (missingAdjustmentIds.length > 0 && missingAdjustmentIds.length < entries.length) invalidate('partial_adjustments', 'Some rows are missing reconciliation adjustments', missingAdjustmentIds)
    if (missingAdjustmentIds.length === entries.length && metadataAdjustment !== 0n) invalidate('missing_adjustments', 'Statement adjustment is nonzero but row adjustments are missing')
    if (metadataAdjustment !== null && metadataAdjustment !== adjustmentTotal) invalidate('adjustment_total_mismatch', 'ผลรวมยอดปรับรายกองทุนไม่ตรงกับยอดปรับใน metadata')
    if (totalNavCents !== null && rawTotal + adjustmentTotal !== totalNavCents) invalidate('nav_total_mismatch', 'ผลรวมมูลค่ารายกองทุนหลังปรับไม่ตรงกับ NAV รวมของ Statement')
    funds.sort((a, b) => a.code.localeCompare(b.code))

    const scopeKey = entries.every((entry) => entry.scopeKey === first.scopeKey) ? first.scopeKey : null
    const asOfDate = entries.every((entry) => entry.date === first.date) ? first.date : null
    if (!scopeKey || !asOfDate) {
      invalidate('unresolved_statement_identity', 'แถวใน Statement เดียวกันมีขอบเขตสมาชิกหรือวันที่ขัดแย้งกัน')
      addIssue(unresolvedIssues, 'unresolved_statement_identity', 'แถวใน Statement เดียวกันมีขอบเขตสมาชิกหรือวันที่ขัดแย้งกัน จึงระบุงวดล่าสุดไม่ได้', ids)
    }
    statementGroups.push({
      id: fingerprint,
      scopeKey,
      asOfDate,
      archiveId: sha256.test(String(first.metadata.archive_id ?? ''))
        && sha256.test(String(first.metadata.source_pdf_sha256 ?? ''))
        && String(first.metadata.archive_id).toLowerCase() === String(first.metadata.source_pdf_sha256).toLowerCase()
        ? String(first.metadata.archive_id).toLowerCase() : null,
      totalNavCents,
      isValid,
      funds,
      issues,
      previousId: null,
      deltaNavCents: null,
      comparisons: [],
    })
  }

  const scopes = new Set(statementGroups.map((statement) => statement.scopeKey).filter(Boolean))
  if (scopes.size > 1) addIssue(unresolvedIssues, 'multiple_scopes', 'พบ Statement หลายขอบเขตสมาชิกในพอร์ตเดียวกัน จึงไม่สามารถเลือกยอดล่าสุดได้', [])
  statementGroups.sort((a, b) => (b.asOfDate ?? '').localeCompare(a.asOfDate ?? '') || a.id.localeCompare(b.id))

  const idsByStatement = new Map<string, string[]>()
  for (const entries of grouped.values()) {
    if (entries.length > 0) idsByStatement.set(entries[0]!.groupId, entries.map(({ tx }) => safeId(tx)))
  }
  const byScope = new Map<string, PvdStatementGroup[]>()
  for (const statement of statementGroups) {
    if (!statement.scopeKey || !statement.asOfDate) continue
    const scope = byScope.get(statement.scopeKey) ?? []
    scope.push(statement)
    byScope.set(statement.scopeKey, scope)
  }
  for (const scope of byScope.values()) {
    const byDate = new Map<string, PvdStatementGroup[]>()
    for (const statement of scope) {
      const sameDate = byDate.get(statement.asOfDate!) ?? []
      sameDate.push(statement)
      byDate.set(statement.asOfDate!, sameDate)
    }
    for (const sameDate of byDate.values()) {
      if (sameDate.length < 2) continue
      for (const statement of sameDate) {
        statement.isValid = false
        addIssue(statement.issues, 'same_day_conflict', 'Several statement fingerprints have the same member and date', idsByStatement.get(statement.id) ?? [])
      }
    }
    scope.sort((a, b) => (a.asOfDate ?? '').localeCompare(b.asOfDate ?? '') || a.id.localeCompare(b.id))
    for (let i = 0; i < scope.length; i += 1) {
      const current = scope[i]!
      const previous = scope[i - 1]
      if (!previous) {
        current.comparisons = current.funds.map((fund) => ({ code: fund.code, previous: null, current: fund, deltaCents: null, status: 'baseline' }))
        continue
      }
      current.previousId = previous.id
      if (!current.isValid || !previous.isValid) {
        current.comparisons = []
        continue
      }
      current.deltaNavCents = current.totalNavCents !== null && previous.totalNavCents !== null
        ? current.totalNavCents - previous.totalNavCents
        : null
      const oldFunds = new Map(previous.funds.map((fund) => [fund.code, fund]))
      const newFunds = new Map(current.funds.map((fund) => [fund.code, fund]))
      current.comparisons = [...new Set([...oldFunds.keys(), ...newFunds.keys()])].sort().map((code) => {
        const oldRow = oldFunds.get(code) ?? null
        const newRow = newFunds.get(code) ?? null
        if (!oldRow) {
          const isZero = Boolean(newRow && newRow.units === 0 && newRow.valueCents === 0n)
          return { code, previous: null, current: newRow, deltaCents: newRow?.valueCents ?? null, status: isZero ? 'zero' : 'new' }
        }
        if (!newRow) return { code, previous: oldRow, current: null, deltaCents: null, status: 'missing' }
        const deltaCents = newRow.valueCents - oldRow.valueCents
        const bothZero = oldRow.units === 0 && newRow.units === 0 && oldRow.valueCents === 0n && newRow.valueCents === 0n
        const oneZero = (oldRow.units === 0 && oldRow.valueCents === 0n) || (newRow.units === 0 && newRow.valueCents === 0n)
        const status = bothZero ? 'zero_both'
          : oneZero ? 'zero'
            : deltaCents > 0n ? 'increase'
              : deltaCents < 0n ? 'decrease' : 'unchanged'
        return { code, previous: oldRow, current: newRow, deltaCents, status }
      })
      if (current.comparisons.some((comparison) => comparison.status === 'missing')) current.deltaNavCents = null
    }
  }

  const latest = statementGroups[0] ?? null
  let summaryUnavailableReason: string | null = null
  if (unresolvedIssues.length) summaryUnavailableReason = 'มีข้อมูล PVD ที่ระบุตัวตนหรือจัดลำดับงวดไม่ได้ จึงพักการแสดงยอดล่าสุด'
  else if (scopes.size > 1) summaryUnavailableReason = 'มี PVD มากกว่าหนึ่งขอบเขตสมาชิกในพอร์ตนี้'
  else if (latest && !latest.isValid) summaryUnavailableReason = 'Statement ล่าสุดมีข้อมูลขัดแย้งหรือยอดรวมไม่ตรงกัน'
  else if (latest && latest.totalNavCents === null) summaryUnavailableReason = 'ไม่มี NAV รวมที่ตรวจสอบได้ใน Statement ล่าสุด'

  return {
    statements: statementGroups,
    unresolvedIssues,
    latestStatementId: latest?.id ?? null,
    summaryUnavailableReason,
  }
}
