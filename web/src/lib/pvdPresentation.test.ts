import { describe, expect, it } from 'vitest'
import type { ActualHoldingDTO, PvdStatementMetadataDTO } from '../api/types'
import {
  getPvdValuationContext,
  isPvdHolding,
  isPvdSymbol,
  isValidIsoDate,
  partitionPriceRefreshInfo,
} from './pvdPresentation'

function holding(overrides: Partial<ActualHoldingDTO> = {}): ActualHoldingDTO {
  return {
    symbol: 'ABC', asset_type: 'Stock', units: 10, status: 'active', archived_at: null,
    bucket_id: null, avg_cost_usd: null, avg_cost_thb: 10, current_price_usd: null,
    current_price_thb: 12, market_value_thb: 120, unrealized_pnl_percent: 20,
    unrealized_pnl_value: 20, market_cap_tier: null, yield_on_cost: null, company_name: null,
    pe_ratio: null, eps: null, payout_ratio: null, market_cap_value: null, dividend_per_share: null,
    dividend_yield: null, accumulated_dividend_thb: null, fundamentals_updated_at: null,
    pvd_valuation: null, ...overrides,
  }
}

function statement(as_of_date: string): Pick<PvdStatementMetadataDTO, 'as_of_date'> {
  return { as_of_date }
}

describe('PVD presentation helpers', () => {
  it('identifies PVD by normalized type, reserved symbol, or valuation metadata', () => {
    expect(isPvdHolding(holding({ asset_type: ' pvd ' }))).toBe(true)
    expect(isPvdHolding(holding({ symbol: 'pvd_sinsa_fi', asset_type: 'Fund' }))).toBe(true)
    expect(isPvdHolding(holding({ pvd_valuation: { as_of_date: '2026-06-30' } as ActualHoldingDTO['pvd_valuation'] }))).toBe(true)
    expect(isPvdHolding(holding())).toBe(false)
    expect(isPvdSymbol('  PVD_SINSA_FI ')).toBe(true)
    expect(isPvdSymbol('SINSA_FI')).toBe(false)
  })

  it('validates calendar dates without converting date-only values through a timezone', () => {
    expect(isValidIsoDate('2024-02-29')).toBe(true)
    expect(isValidIsoDate('2026-02-29')).toBe(false)
    expect(isValidIsoDate('2026-13-01')).toBe(false)
    expect(isValidIsoDate('2026-06-30T00:00:00Z')).toBe(false)
  })

  it('summarizes active PVD dates, ignores archived zero rows, and distinguishes cash from market holdings', () => {
    const result = getPvdValuationContext([
      holding({
        symbol: 'PVD_SINSA_FI', asset_type: 'PVD', units: 20, market_value_thb: 200,
        pvd_valuation: { as_of_date: '2026-06-30' } as ActualHoldingDTO['pvd_valuation'],
      }),
      holding({
        symbol: 'PVD_OLD', asset_type: 'PVD', units: 0, market_value_thb: 0, status: 'archived',
        pvd_valuation: { as_of_date: '2025-01-01' } as ActualHoldingDTO['pvd_valuation'],
      }),
      holding({ symbol: 'CASH_THB', asset_type: 'Cash', units: 100, market_value_thb: 100 }),
    ], [statement('2026-07-01'), statement('2026-02-30')])

    expect(result).toEqual({
      hasPvd: true,
      hasMarketHoldings: false,
      hasNonPvdValue: true,
      dates: ['2026-06-30'],
      hasUnknownDate: false,
      hasUnknownValuation: false,
      latestRecordedStatementDate: '2026-07-01',
    })
  })

  it('reports mixed and incomplete valuation dates without guessing from statement history', () => {
    const result = getPvdValuationContext([
      holding({
        symbol: 'PVD_A', asset_type: 'PVD', units: 5, market_value_thb: 50,
        pvd_valuation: { as_of_date: '2026-03-31' } as ActualHoldingDTO['pvd_valuation'],
      }),
      holding({ symbol: 'PVD_B', asset_type: 'PVD', units: 3, market_value_thb: 30 }),
      holding({ symbol: 'XYZ', asset_type: 'ETF', units: 1, market_value_thb: 20 }),
    ], [statement('2026-06-30')])

    expect(result.dates).toEqual(['2026-03-31'])
    expect(result.hasUnknownDate).toBe(true)
    expect(result.hasUnknownValuation).toBe(true)
    expect(result.latestRecordedStatementDate).toBe('2026-06-30')
    expect(result.hasMarketHoldings).toBe(true)
  })

  it('partitions PVD statuses before numeric price statuses are counted', () => {
    expect(partitionPriceRefreshInfo({
      PVD_SINSA_FI: 'Statement NAV 12.3456 THB (2026-06-30)',
      PVD_SINSA_GOLD: 'Statement NAV unavailable',
      ABC: '12.50 THB',
      USDTHB: '36.10 (live)',
    }, [holding({ symbol: 'PVD_SINSA_FI', asset_type: 'PVD' })])).toEqual({
      marketEntries: [['ABC', '12.50 THB'], ['USDTHB', '36.10 (live)']],
      pvdEntries: [
        ['PVD_SINSA_FI', 'Statement NAV 12.3456 THB (2026-06-30)'],
        ['PVD_SINSA_GOLD', 'Statement NAV unavailable'],
      ],
    })
  })
})
