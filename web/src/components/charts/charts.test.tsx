import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { RadialGauge } from './RadialGauge'
import { CalendarHeatmap } from './CalendarHeatmap'
import { OptionsOiStrikeLadder } from './OptionsOiStrikeLadder'
import { PolicyRateComparisonBar } from './PolicyRateComparisonBar'
import { TreemapChart } from './TreemapChart'
import { StackedAreaChart } from './StackedAreaChart'
import { BubbleChart } from './BubbleChart'
import { OptionsVolSmile } from './OptionsVolSmile'
import { NasdaqConsensusCard } from '../equity/NasdaqConsensusCard'
import { MetalsCotCard } from '../macro/MetalsCotCard'

describe('Terminal V2 Phase 3 Institutional Charts', () => {
  it('RadialGauge renders stress index, categories, and T-2 lag note', () => {
    render(
      <RadialGauge
        title="US Financial Stress Index"
        subtitle="Office of Financial Research (OFR)"
        value={-0.25}
        min={-3.0}
        max={3.0}
        unit="σ"
        asOfDate="2026-09-24"
        dataLagNote="T-2 Business Days Lag"
        subItems={[
          { label: 'Credit', value: -0.09 },
          { label: 'Equity valuation', value: -0.05 },
          { label: 'Safe assets', value: -0.04 },
          { label: 'Funding', value: -0.04 },
          { label: 'Volatility', value: -0.03 },
        ]}
      />
    )

    expect(screen.getByText('US Financial Stress Index')).toBeInTheDocument()
    expect(screen.getByText('2026-09-24')).toBeInTheDocument()
    expect(screen.getByText('T-2 Business Days Lag')).toBeInTheDocument()
    expect(screen.getByText('-0.25')).toBeInTheDocument()
    expect(screen.getByText('Credit')).toBeInTheDocument()
    expect(screen.getByText('Equity valuation')).toBeInTheDocument()
  })

  it('CalendarHeatmap renders daily PnL and summary statistics', () => {
    const data = [
      { date: '2026-09-01', value: 1200, count: 2 },
      { date: '2026-09-02', value: -450, count: 1 },
      { date: '2026-09-03', value: 2500, count: 3 },
    ]

    render(
      <CalendarHeatmap
        data={data}
        title="Trading Performance Calendar"
        subtitle="Daily realized PnL heatmap"
        unit="$"
      />
    )

    expect(screen.getByText('Trading Performance Calendar')).toBeInTheDocument()
    expect(screen.getByText('Total PnL')).toBeInTheDocument()
    expect(screen.getByText('Win Rate')).toBeInTheDocument()
    // Win rate = 2 / 3 = 66.7%
    expect(screen.getByText('66.7%')).toBeInTheDocument()
  })

  it('OptionsOiStrikeLadder renders call/put open interest and invariant note', () => {
    const strikes = [
      { strike: 120, callOi: 5000, putOi: 2000 },
      { strike: 125, callOi: 15000, putOi: 8000 },
      { strike: 130, callOi: 12000, putOi: 18000 },
    ]

    render(
      <OptionsOiStrikeLadder
        symbol="NVDA"
        strikes={strikes}
        currentPrice={124.5}
        maxPainStrike={125.0}
        expirationDate="2026-10-16"
        putCallRatio={0.87}
      />
    )

    expect(screen.getByText(/NVDA Options Open Interest by Strike/i)).toBeInTheDocument()
    expect(screen.getByText('Exp: 2026-10-16')).toBeInTheDocument()
    expect(screen.getByText('0.87')).toBeInTheDocument()
    expect(screen.getByText(/Max Pain is calculated from cumulative open interest/i)).toBeInTheDocument()
    expect(screen.getAllByText('$125.0').length).toBeGreaterThanOrEqual(1)
  })

  it('PolicyRateComparisonBar renders 12 countries with spreads vs BOT', () => {
    const rates = [
      { country: 'TH', rateValue: 2.5, rateType: '1-Day Bilateral Repo', effectiveDate: '2026-09-18' },
      { country: 'US', rateValue: 5.0, rateType: 'Federal Funds Target', effectiveDate: '2026-09-18' },
      { country: 'XM', rateValue: 3.5, rateType: 'Deposit Facility Rate', effectiveDate: '2026-09-18' },
    ]
    const spreads = { US: 250, XM: 100, TH: 0 }

    render(
      <PolicyRateComparisonBar
        rates={rates}
        spreadsVsBotRepo={spreads}
        botRepoRate={2.5}
      />
    )

    expect(screen.getByText('Global Central Bank Policy Rates')).toBeInTheDocument()
    expect(screen.getByText('2.50% (1D Repo)')).toBeInTheDocument()
    expect(screen.getByText('+250 bps vs BOT')).toBeInTheDocument()
    expect(screen.getByText('Federal Funds Target')).toBeInTheDocument()
  })

  it('MetalsCotCard separates Tuesday as-of-date from Friday published date', () => {
    render(
      <MetalsCotCard
        commodity="gold"
        commodityCode="088691"
        asOfDate="2026-09-22"
        publishedAt="2026-09-25"
        openInterest={480000}
        netManagedMoney={202800}
        percentile52w={85.5}
        managedMoney={{
          class_name: 'Managed Money',
          long_contracts: 245100,
          short_contracts: 42300,
          net_contracts: 202800,
          change_long: 5200,
          change_short: -1800,
        }}
        swapDealers={{
          class_name: 'Swap Dealers',
          long_contracts: 120000,
          short_contracts: 145000,
          net_contracts: -25000,
        }}
        producerMerchant={{
          class_name: 'Producer/Merchant',
          long_contracts: 78500,
          short_contracts: 298000,
          net_contracts: -219500,
        }}
      />
    )

    expect(screen.getByText(/Futures Positioning \(CFTC COT\)/i)).toBeInTheDocument()
    expect(screen.getByText('As of: 2026-09-22 (Tue)')).toBeInTheDocument()
    expect(screen.getByText('Pub: 2026-09-25 (Fri)')).toBeInTheDocument()
    expect(screen.getAllByText('+202,800').length).toBeGreaterThanOrEqual(1)
    expect(screen.getByText('85.5%')).toBeInTheDocument()
  })

  it('NasdaqConsensusCard handles full coverage and graceful no_coverage', () => {
    const { rerender } = render(
      <NasdaqConsensusCard
        symbol="NVDA"
        coverageStatus="full"
        hasEarningsSurprise={true}
        hasAnalystRatings={true}
        upcomingEarnings={{
          earnings_date: '2026-11-18',
          date_status: 'confirmed',
          fiscal_quarter: 'Q3',
        }}
        ratings={{
          consensus: 'Strong Buy',
          analyst_count: 42,
          target_price_mean: 155.0,
          broker_names: ['Goldman Sachs', 'Morgan Stanley'],
        }}
        surpriseHistory={[
          {
            fiscal_quarter_end: 'Jul 2026',
            date_reported: '2026-08-26',
            eps: 0.68,
            consensus_eps: 0.64,
            surprise_pct: 6.25,
          },
        ]}
      />
    )

    expect(screen.getByText(/NVDA Sell-Side Consensus & Earnings/i)).toBeInTheDocument()
    expect(screen.getByText('CONFIRMED')).toBeInTheDocument()
    expect(screen.getByText('2026-11-18')).toBeInTheDocument()
    expect(screen.getByText('Strong Buy')).toBeInTheDocument()
    expect(screen.getByText('$155.00')).toBeInTheDocument()
    expect(screen.getByText('+6.3%')).toBeInTheDocument()

    // Test graceful no_coverage fallback
    rerender(
      <NasdaqConsensusCard
        symbol="MICROCAP"
        coverageStatus="no_coverage"
        hasEarningsSurprise={false}
        hasAnalystRatings={false}
      />
    )

    expect(screen.getByText('No Coverage')).toBeInTheDocument()
    expect(screen.getByText(/No sell-side analyst ratings/i)).toBeInTheDocument()
  })

  // ==========================================================================
  // Terminal V2 Phase 4 Chart Primitives Tests
  // ==========================================================================

  it('TreemapChart renders items, total value, and empty state', () => {
    const items = [
      { id: '1', label: 'Cash THB', value: 250000, group: 'Cash' },
      { id: '2', label: 'US Equities', value: 500000, group: 'Equity', return_pct: 0.12 },
      { id: '3', label: 'Thai Equities', value: 250000, group: 'Equity', return_pct: -0.04 },
    ]

    const { rerender } = render(
      <TreemapChart
        items={items}
        title="Asset Allocation Treemap"
        subtitle="Current portfolio holdings by asset type"
        unit="THB"
      />
    )

    expect(screen.getByText('Asset Allocation Treemap')).toBeInTheDocument()
    expect(screen.getByText('Current portfolio holdings by asset type')).toBeInTheDocument()
    expect(screen.getByText(/1,000,000.00 THB/i)).toBeInTheDocument()
    expect(screen.getByText('US Equities')).toBeInTheDocument()
    expect(screen.getByText('50.0%')).toBeInTheDocument()

    // Empty state
    rerender(<TreemapChart items={[]} title="Asset Allocation Treemap" />)
    expect(screen.getByText(/No holdings or asset-class breakdown available/i)).toBeInTheDocument()
  })

  it('StackedAreaChart renders area layers, mode toggle, and empty state', () => {
    const data = [
      { date: '2026-09-01', values_by_category: { public: 28000, intragov: 7000 } },
      { date: '2026-09-02', values_by_category: { public: 28100, intragov: 7020 } },
    ]
    const categories = [
      { key: 'public', label: 'Debt Held by Public', color: '#38bdf8' },
      { key: 'intragov', label: 'Intragovernmental Holdings', color: '#fbbf24' },
    ]

    const { rerender } = render(
      <StackedAreaChart
        data={data}
        categories={categories}
        title="US Debt Composition"
        unit="B USD"
      />
    )

    expect(screen.getByText('US Debt Composition')).toBeInTheDocument()
    expect(screen.getByText('Debt Held by Public')).toBeInTheDocument()
    expect(screen.getByText('Intragovernmental Holdings')).toBeInTheDocument()
    expect(screen.getByText('100% Share')).toBeInTheDocument()

    // Empty state
    rerender(<StackedAreaChart data={[]} categories={categories} title="US Debt Composition" />)
    expect(screen.getByText(/No historical series data available/i)).toBeInTheDocument()
  })

  it('BubbleChart renders bubbles with axes labels, scaling, and empty state', () => {
    const data = [
      { id: 'nvda', label: 'NVDA', x: 25.5, y: 55.0, size: 3100 },
      { id: 'aapl', label: 'AAPL', x: 12.0, y: 32.0, size: 3400 },
    ]

    const { rerender } = render(
      <BubbleChart
        data={data}
        title="Valuation vs Growth"
        xLabel="Sales Growth %"
        yLabel="Forward P/E"
        sizeLabel="Market Cap ($B)"
        xUnit="%"
        sizeUnit="B"
      />
    )

    expect(screen.getByText('Valuation vs Growth')).toBeInTheDocument()
    expect(screen.getByText(/Sales Growth %/i)).toBeInTheDocument()
    expect(screen.getByText(/Forward P\/E/i)).toBeInTheDocument()
    expect(screen.getByText('NVDA')).toBeInTheDocument()
    expect(screen.getByText('AAPL')).toBeInTheDocument()

    // Empty state
    rerender(
      <BubbleChart
        data={[]}
        title="Valuation vs Growth"
        xLabel="Sales Growth"
        yLabel="P/E"
        sizeLabel="Cap"
      />
    )
    expect(screen.getByText(/No cross-sectional multi-dimensional data available/i)).toBeInTheDocument()
  })

  it('OptionsVolSmile renders call/put IV smile and handles insufficient quotes gracefully', () => {
    const contracts = [
      { strike: 120, option_type: 'call' as const, implied_volatility: 0.28 },
      { strike: 120, option_type: 'put' as const, implied_volatility: 0.32 },
      { strike: 125, option_type: 'call' as const, implied_volatility: 0.24 },
      { strike: 125, option_type: 'put' as const, implied_volatility: 0.26 },
      { strike: 130, option_type: 'call' as const, implied_volatility: 0.22 },
      { strike: 130, option_type: 'put' as const, implied_volatility: 0.23 },
    ]

    const { rerender } = render(
      <OptionsVolSmile
        symbol="NVDA"
        expiry="2026-10-16"
        currentPrice={124.5}
        contracts={contracts}
        delayMinutes={15}
      />
    )

    expect(screen.getByText(/NVDA Implied Volatility Smile/i)).toBeInTheDocument()
    expect(screen.getByText('2026-10-16')).toBeInTheDocument()
    expect(screen.getByText(/Call Implied Volatility/i)).toBeInTheDocument()
    expect(screen.getByText(/Put Implied Volatility/i)).toBeInTheDocument()
    expect(screen.getByText(/Spot: \$124.50/i)).toBeInTheDocument()
    expect(screen.getByText(/Delayed by 15m/i)).toBeInTheDocument()

    // Graceful insufficient quotes (< 3 strikes)
    rerender(
      <OptionsVolSmile
        symbol="NVDA"
        expiry="2026-10-16"
        contracts={[{ strike: 120, option_type: 'call', implied_volatility: 0.25 }]}
      />
    )

    expect(screen.getByText(/Insufficient implied volatility quotes to render smile/i)).toBeInTheDocument()
  })
})
