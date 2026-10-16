import { describe, expect, it } from 'vitest'
import type { TransactionItemDTO } from '../api/types'
import { buildPvdStatementHistory, formatMoneyCents, isPvdTransaction, parseMoneyCents, partitionTransactions } from './pvdStatementHistory'

const sha = (letter: string) => letter.repeat(64)

function row(options: {
  date: string
  statement: string
  code?: string
  value: string
  adjustment?: string
  totalNav: string
  statementAdjustment?: string
  units?: number
  nav?: number
  transactionId?: string
  source?: string
  symbol?: string
  metadataPatch?: Record<string, unknown>
}): TransactionItemDTO {
  const code = options.code || 'SINSA_GTO'
  const semanticFingerprint = options.statement
  const metadata = {
    payload_version: 1,
    fund_code: 'BBLAM',
    member_hash: sha('c'),
    masked_member_id: '***1234',
    as_of_date: options.date,
    period_start: null,
    period_end: null,
    total_contribution: '250.00',
    total_nav: options.totalNav,
    member_principal: '100.00',
    member_benefit: '50.00',
    employer_principal: '50.00',
    employer_benefit: '50.00',
    employer_vesting_percent: '100',
    member_vested_amount: '150.00',
    employer_vested_amount: '100.00',
    source_pdf_sha256: sha('d'),
    semantic_fingerprint: semanticFingerprint,
    archive_id: sha('d'),
    reconciliation_adjustment: options.statementAdjustment || '0.00',
    ...options.metadataPatch,
  }
  return {
    transaction_id: options.transactionId || `pvd-${semanticFingerprint.slice(0, 4)}-${code}`,
    timestamp: `${options.date} 00:00:00`,
    symbol: options.symbol || `PVD_${code}`,
    action: 'PVD_SNAPSHOT',
    units: options.units ?? 10,
    price: options.nav ?? 100,
    currency: 'THB',
    cost_thb: 0,
    realized_pnl_thb: null,
    notes: '',
    gross_amount: options.value,
    net_amount: options.value,
    commission: '0.00',
    vat: '0.00',
    other_fees: '0.00',
    fee_currency: 'THB',
    confirmation_no: semanticFingerprint,
    source: options.source ?? 'BBLAM_PVD',
    fingerprint: sha(options.transactionId?.startsWith('other') ? 'e' : 'f'),
    asset_type: 'PVD',
    bucket_id: 'retirement_pvd',
    reconciliation_adjustment: options.adjustment ?? '0.00',
    pvd_metadata_json: JSON.stringify(metadata),
  }
}

describe('PVD statement history', () => {
  it('parses and formats cents without floating point conversion', () => {
    expect(parseMoneyCents('0.01')).toBe(1n)
    expect(parseMoneyCents('-0.01')).toBe(-1n)
    expect(formatMoneyCents(900719925474099301n)).toBe('9007199254740993.01')
    expect(() => parseMoneyCents('1.001')).toThrow()
    expect(() => parseMoneyCents('1e3')).toThrow()
  })

  it('partitions each PVD signal into PVD and keeps ordinary transactions in assets', () => {
    const bySource = row({ date: '2026-01-31', statement: sha('a'), value: '1.00', totalNav: '1.00', source: ' bblam_pvd ' })
    const bySymbol = { ...bySource, transaction_id: 'symbol', source: null, action: 'OTHER', asset_type: null }
    const byType = { ...bySource, transaction_id: 'type', symbol: 'FUND', source: null, action: 'OTHER' }
    const asset = { ...bySource, transaction_id: 'asset', action: 'BUY', symbol: 'AAPL', asset_type: 'STOCK', source: 'DIME' }
    expect(isPvdTransaction(bySource)).toBe(true)
    expect(isPvdTransaction(bySymbol)).toBe(true)
    expect(isPvdTransaction(byType)).toBe(true)
    const result = partitionTransactions([bySource, bySymbol, byType, asset])
    expect(result.pvd).toHaveLength(3)
    expect(result.assets).toEqual([asset])
    expect(result.assets.length + result.pvd.length).toBe(4)
  })

  it('groups rows into statement snapshots and computes exact adjusted value changes', () => {
    const previous = row({ date: '2026-01-31', statement: sha('a'), value: '1000.00', totalNav: '1000.00' })
    const currentGto = row({ date: '2026-06-30', statement: sha('b'), value: '800.01', adjustment: '0.01', totalNav: '1250.02', statementAdjustment: '0.01' })
    const currentBtp = row({ date: '2026-06-30', statement: sha('b'), code: 'SINSA_BTP', value: '450.00', totalNav: '1250.02', statementAdjustment: '0.01', transactionId: 'other-btp' })
    const history = buildPvdStatementHistory('portfolio-a', [currentGto, previous, currentBtp, { ...currentGto }])
    expect(history.statements.map((statement) => statement.asOfDate)).toEqual(['2026-06-30', '2026-01-31'])
    expect(history.latestStatementId).toBe(sha('b'))
    expect(history.statements[0]!.totalNavCents).toBe(125002n)
    expect(history.statements[0]!.deltaNavCents).toBe(25002n)
    expect(history.statements[0]!.funds.find((fund) => fund.code === 'SINSA_GTO')?.valueCents).toBe(80002n)
    expect(history.statements[0]!.comparisons.find((item) => item.code === 'SINSA_GTO')?.deltaCents).toBe(-19998n)
    expect(history.statements[0]!.comparisons.find((item) => item.code === 'SINSA_BTP')?.status).toBe('new')
    expect(history.unresolvedIssues).toEqual([])
    expect(history.summaryUnavailableReason).toBeNull()
  })

  it('treats a fund missing from the current statement as missing data rather than zero', () => {
    const priorGto = row({ date: '2026-01-31', statement: sha('a'), value: '700.00', totalNav: '1000.00' })
    const priorBtp = row({ date: '2026-01-31', statement: sha('a'), code: 'SINSA_BTP', value: '300.00', totalNav: '1000.00', transactionId: 'other-old-btp' })
    const current = row({ date: '2026-06-30', statement: sha('b'), value: '800.00', totalNav: '800.00' })
    const history = buildPvdStatementHistory('portfolio-a', [priorGto, priorBtp, current])
    const missing = history.statements[0]!.comparisons.find((item) => item.code === 'SINSA_BTP')
    expect(missing?.status).toBe('missing')
    expect(missing?.deltaCents).toBeNull()
    expect(missing?.current).toBeNull()
  })

  it('compares the planned seven and eight subfund snapshots with signed cent adjustments and zero rows', () => {
    const oldCodes = ['FUND_A', 'FUND_B', 'FUND_C', 'FUND_D', 'FUND_E', 'FUND_F', 'FUND_G']
    const previousRows = oldCodes.map((code) => row({
      date: '2026-01-31', statement: sha('a'), code, value: code === 'FUND_C' ? '0.00' : '100.00', totalNav: '600.00',
      units: code === 'FUND_C' ? 0 : 100, nav: 1.01,
    }))
    const newCodes = [...oldCodes, 'FUND_H']
    const currentRows = newCodes.map((code) => row({
      date: '2026-06-30', statement: sha('b'), code,
      value: code === 'FUND_A' ? '100.01' : code === 'FUND_B' ? '99.99' : code === 'FUND_C' ? '0.00' : code === 'FUND_H' ? '50.00' : '100.00',
      adjustment: code === 'FUND_A' ? '0.01' : code === 'FUND_B' ? '-0.01' : '0.00',
      totalNav: '650.00', statementAdjustment: '0.00', units: code === 'FUND_C' ? 0 : 100, nav: 1.01,
    }))
    const history = buildPvdStatementHistory('portfolio-a', [...previousRows, ...currentRows])
    const latest = history.statements[0]!
    expect(previousRows).toHaveLength(7)
    expect(currentRows).toHaveLength(8)
    expect(latest.isValid).toBe(true)
    expect(latest.totalNavCents).toBe(65000n)
    expect(latest.deltaNavCents).toBe(5000n)
    expect(latest.comparisons.find((item) => item.code === 'FUND_A')?.deltaCents).toBe(2n)
    expect(latest.comparisons.find((item) => item.code === 'FUND_B')?.deltaCents).toBe(-2n)
    expect(latest.comparisons.find((item) => item.code === 'FUND_C')?.status).toBe('zero_both')
    expect(latest.comparisons.find((item) => item.code === 'FUND_H')?.status).toBe('new')
    expect(latest.funds.find((fund) => fund.code === 'FUND_A')?.rawValueCents).toBe(10001n)
    expect(latest.funds.find((fund) => fund.code === 'FUND_A')?.units! * latest.funds.find((fund) => fund.code === 'FUND_A')!.nav).not.toBe(Number(latest.funds.find((fund) => fund.code === 'FUND_A')!.rawValueCents) / 100)
  })

  it('flags conflicting duplicate IDs, duplicate fund codes, partial adjustments, and same-day statements', () => {
    const base = row({ date: '2026-01-31', statement: sha('a'), value: '100.00', totalNav: '100.00' })
    const conflictingDuplicate = { ...base, gross_amount: '101.00', net_amount: '101.00' }
    const duplicateIdHistory = buildPvdStatementHistory('portfolio-a', [base, conflictingDuplicate])
    expect(duplicateIdHistory.unresolvedIssues.some((issue) => issue.code === 'duplicate_transaction_id')).toBe(true)

    const duplicateCode = row({ date: '2026-06-30', statement: sha('b'), value: '50.00', totalNav: '100.00', transactionId: 'other-duplicate-code' })
    const duplicateFundHistory = buildPvdStatementHistory('portfolio-a', [base, row({ date: '2026-06-30', statement: sha('b'), value: '50.00', totalNav: '100.00' }), duplicateCode])
    expect(duplicateFundHistory.statements[0]!.issues.some((issue) => issue.code === 'duplicate_fund_code')).toBe(true)

    const partialRows = [
      row({ date: '2026-06-30', statement: sha('b'), value: '60.00', totalNav: '100.01', adjustment: '0.01', statementAdjustment: '0.01' }),
      { ...row({ date: '2026-06-30', statement: sha('b'), code: 'SINSA_BTP', value: '40.00', totalNav: '100.01', adjustment: '0.00', statementAdjustment: '0.01', transactionId: 'other-partial' }), reconciliation_adjustment: null },
    ]
    const partialHistory = buildPvdStatementHistory('portfolio-a', partialRows)
    expect(partialHistory.statements[0]!.issues.some((issue) => issue.code === 'partial_adjustments')).toBe(true)

    const sameDay = row({ date: '2026-01-31', statement: sha('b'), value: '100.00', totalNav: '100.00', transactionId: 'other-same-day' })
    const afterSameDay = row({ date: '2026-06-30', statement: sha('e'), value: '120.00', totalNav: '120.00', transactionId: 'other-after-same-day' })
    const sameDayHistory = buildPvdStatementHistory('portfolio-a', [base, sameDay, afterSameDay])
    expect(sameDayHistory.statements.find((statement) => statement.id === sha('a'))!.isValid).toBe(false)
    expect(sameDayHistory.statements.find((statement) => statement.id === sha('b'))!.isValid).toBe(false)
    expect(sameDayHistory.statements[0]!.deltaNavCents).toBeNull()
    expect(sameDayHistory.statements[0]!.comparisons).toEqual([])
  })

  it('suppresses latest totals for an invalid newest statement and unresolved rows', () => {
    const old = row({ date: '2026-01-31', statement: sha('a'), value: '100.00', totalNav: '100.00' })
    const invalidLatest = row({ date: '2026-06-30', statement: sha('b'), value: '120.00', totalNav: '121.00' })
    const invalidHistory = buildPvdStatementHistory('portfolio-a', [old, invalidLatest])
    expect(invalidHistory.latestStatementId).toBe(sha('b'))
    expect(invalidHistory.statements[0]!.isValid).toBe(false)
    expect(invalidHistory.summaryUnavailableReason).toBeTruthy()
    expect(invalidHistory.statements[0]!.deltaNavCents).toBeNull()

    const unresolved = { ...old, transaction_id: 'broken', pvd_metadata_json: '{' }
    const unresolvedHistory = buildPvdStatementHistory('portfolio-a', [old, unresolved])
    expect(unresolvedHistory.unresolvedIssues[0]!.code).toBe('invalid_metadata')
    expect(unresolvedHistory.summaryUnavailableReason).toBeTruthy()
  })
})
