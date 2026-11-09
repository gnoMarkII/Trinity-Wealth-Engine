import { render, screen, fireEvent } from '@testing-library/react'
import { describe, it, expect } from 'vitest'
import ValuationWorkbenchCard from './ValuationWorkbenchCard'

describe('ValuationWorkbenchCard', () => {
  const mockReverseDcf = {
    target_price_12m: 117.24,
    upside_12m_pct: -29.4,
    intrinsic_value_today: 108.50,
    market_implied_growth_pct: 24.05,
    market_implied_margin_pct: 36.06,
    enterprise_value: 48000000000,
    equity_value: 45000000000,
    sum_pv_5y_fcf: 12000000000,
    terminal_value_pv: 36000000000,
    solver_status: 'converged' as const,
  }

  const mockDcf = {
    valuation_verdict: 'overvalued' as const,
    wacc_pct: 9.5,
    cost_of_equity_pct: 10.2,
    cost_of_debt_pct: 5.0,
    risk_free_rate_pct: 4.25,
    erp_pct: 5.5,
    observable_refs: [],
    scenarios: {
      bear: { target_price: 85.0, upside_pct: -45.0, probability_weight: 0.25, margin_of_safety_pct: 0 },
      base: { target_price: 117.24, upside_pct: -29.4, probability_weight: 0.50, margin_of_safety_pct: 0 },
      bull: { target_price: 165.0, upside_pct: 15.0, probability_weight: 0.25, margin_of_safety_pct: 0 },
    },
    sensitivity_matrix: [],
  }

  const mockPiotroski = {
    f_score: 6,
    status: 'AVAILABLE',
    profitability_points: 3,
    leverage_liquidity_points: 2,
    operating_efficiency_points: 1,
  }

  const mockQuality = {
    roic_pct: 28.5,
    fcf_margin_pct: 38.2,
    fcf_yield_pct: 5.4,
    ocf_to_net_income: 1.25,
    solvency_score: 76.1,
  }

  it('renders Reverse DCF Expectation Gap metrics correctly', () => {
    render(
      <ValuationWorkbenchCard
        reverseDcf={mockReverseDcf}
        dcf={mockDcf}
        piotroski={mockPiotroski}
        qualityMetrics={mockQuality}
      />
    )

    expect(screen.getByText('24.1%')).toBeInTheDocument()
    expect(screen.getByText('36.1%')).toBeInTheDocument()
    expect(screen.getAllByText('$117.24').length).toBeGreaterThanOrEqual(1)
    expect(screen.getByText('overvalued')).toBeInTheDocument()
    expect(screen.getByText('6 / 9')).toBeInTheDocument()
    expect(screen.getByText('28.5%')).toBeInTheDocument()
    expect(screen.getByText('Reverse DCF Verdict (12M)')).toBeInTheDocument()
  })

  it('keeps Reverse DCF verdict badge overvalued even if Generic DCF scenarios are highly bullish', () => {
    const bullishGenericDcf = {
      ...mockDcf,
      scenarios: {
        bear: { target_price: 261.47, upside_pct: 57.5, probability_weight: 0.25, margin_of_safety_pct: 0 },
        base: { target_price: 437.26, upside_pct: 163.4, probability_weight: 0.50, margin_of_safety_pct: 0 },
        bull: { target_price: 555.45, upside_pct: 234.6, probability_weight: 0.25, margin_of_safety_pct: 0 },
      },
    }

    render(
      <ValuationWorkbenchCard
        reverseDcf={mockReverseDcf}
        dcf={bullishGenericDcf}
        piotroski={mockPiotroski}
        qualityMetrics={mockQuality}
      />
    )

    expect(screen.getByText('overvalued')).toBeInTheDocument()
    expect(screen.getByText('Reverse DCF Verdict (12M)')).toBeInTheDocument()
  })

  it('returns null when no data provided', () => {
    const { container } = render(<ValuationWorkbenchCard />)
    expect(container.firstChild).toBeNull()
  })

  it('renders sector_name and health score in Sector Operating KPIs', () => {
    render(
      <ValuationWorkbenchCard
        reverseDcf={mockReverseDcf}
        sectorKpis={{
          status: 'available',
          sector_name: 'Communication Services',
          sector_health_score: 90,
          kpis: [
            {
              name: 'Operating (EBIT) Margin',
              value: 29.9,
              unit: '%',
              benchmark: '>= 15% healthy',
            },
          ],
        }}
      />
    )

    expect(screen.getByText(/Sector Operating KPIs \(Communication Services\)/i)).toBeInTheDocument()
    expect(screen.getByText(/Health Score: 90\/100/i)).toBeInTheDocument()
    expect(screen.getByText('Operating (EBIT) Margin')).toBeInTheDocument()
    expect(screen.getByText('29.9%')).toBeInTheDocument()
  })

  it('renders alternative valuation model and intrinsic value', () => {
    render(
      <ValuationWorkbenchCard
        reverseDcf={{
          ...mockReverseDcf,
          alternative_valuation_model: 'justified_pbv',
          alternative_intrinsic_value: 45.5,
          alternative_upside_12m_pct: 12.3,
        }}
      />
    )

    expect(screen.getByText(/Justified P\/BV \(Financials\):/i)).toBeInTheDocument()
    expect(screen.getByText(/\$45.50/i)).toBeInTheDocument()
    expect(screen.getByText(/\(\+12.3%\)/i)).toBeInTheDocument()
  })

  it('renders Valuation Reliability Badges, Mid-Year Convention, and Implied Exit Multiple', () => {
    render(
      <ValuationWorkbenchCard
        reverseDcf={{
          ...mockReverseDcf,
          discounting_convention: 'mid_year',
          valuation_sanity_audit: {
            model_reliability_tier: 'high',
            reliability_score: 95,
            terminal_value_pct_of_ev: 62.5,
            implied_exit_fcf_multiple: 18.2,
            benchmark_sector_multiple: 22.0,
            sanity_warnings: ['Terminal Growth ถูกตรึงไว้ที่ 2.50% เพื่อไม่ให้เกิน Risk-Free Rate'],
          },
        }}
      />
    )

    expect(screen.getByText('⏱️ Mid-Year Convention')).toBeInTheDocument()
    expect(screen.getByText(/High Model Reliability/i)).toBeInTheDocument()
    expect(screen.getByText(/(TV 62.5%)/i)).toBeInTheDocument()
    expect(screen.getByText(/Score: 95\/100/i)).toBeInTheDocument()
    expect(screen.getByText('18.2x FCF')).toBeInTheDocument()
    expect(screen.getByText(/Sector Median:/i)).toBeInTheDocument()
    expect(screen.getByText('22x')).toBeInTheDocument()
    expect(screen.getByText(/Terminal Growth ถูกตรึงไว้ที่/i)).toBeInTheDocument()
  })

  it('renders and toggles 5x5 Valuation Sensitivity Matrix', () => {
    render(
      <ValuationWorkbenchCard
        reverseDcf={{
          ...mockReverseDcf,
          sensitivity_matrix: {
            wacc_values: [8.0, 8.5, 9.0, 9.5, 10.0],
            terminal_growth_values: [1.5, 1.8, 2.1, 2.4, 2.5],
            grid: [
              [
                { wacc_pct: 8.0, terminal_growth_pct: 1.5, target_price: 135.0, upside_pct: 25.0, verdict: 'undervalued' },
                { wacc_pct: 8.0, terminal_growth_pct: 1.8, target_price: 142.0, upside_pct: 32.0, verdict: 'undervalued' },
                { wacc_pct: 8.0, terminal_growth_pct: 2.1, target_price: 150.0, upside_pct: 40.0, verdict: 'undervalued' },
                { wacc_pct: 8.0, terminal_growth_pct: 2.4, target_price: 160.0, upside_pct: 50.0, verdict: 'undervalued' },
                { wacc_pct: 8.0, terminal_growth_pct: 2.5, target_price: 165.0, upside_pct: 55.0, verdict: 'undervalued' },
              ],
              [
                { wacc_pct: 8.5, terminal_growth_pct: 1.5, target_price: 120.0, upside_pct: 10.0, verdict: 'fairly_valued' },
                { wacc_pct: 8.5, terminal_growth_pct: 1.8, target_price: 128.0, upside_pct: 18.0, verdict: 'undervalued' },
                { wacc_pct: 8.5, terminal_growth_pct: 2.1, target_price: 135.0, upside_pct: 25.0, verdict: 'undervalued' },
                { wacc_pct: 8.5, terminal_growth_pct: 2.4, target_price: 142.0, upside_pct: 32.0, verdict: 'undervalued' },
                { wacc_pct: 8.5, terminal_growth_pct: 2.5, target_price: 145.0, upside_pct: 35.0, verdict: 'undervalued' },
              ],
              [
                { wacc_pct: 9.0, terminal_growth_pct: 1.5, target_price: 105.0, upside_pct: -5.0, verdict: 'fairly_valued' },
                { wacc_pct: 9.0, terminal_growth_pct: 1.8, target_price: 112.0, upside_pct: 2.0, verdict: 'fairly_valued' },
                { wacc_pct: 9.0, terminal_growth_pct: 2.1, target_price: 117.2, upside_pct: 7.0, verdict: 'fairly_valued', is_base: true },
                { wacc_pct: 9.0, terminal_growth_pct: 2.4, target_price: 125.0, upside_pct: 15.0, verdict: 'fairly_valued' },
                { wacc_pct: 9.0, terminal_growth_pct: 2.5, target_price: 128.0, upside_pct: 18.0, verdict: 'undervalued' },
              ],
              [
                { wacc_pct: 9.5, terminal_growth_pct: 1.5, target_price: 92.0, upside_pct: -18.0, verdict: 'overvalued' },
                { wacc_pct: 9.5, terminal_growth_pct: 1.8, target_price: 98.0, upside_pct: -12.0, verdict: 'fairly_valued' },
                { wacc_pct: 9.5, terminal_growth_pct: 2.1, target_price: 104.0, upside_pct: -6.0, verdict: 'fairly_valued' },
                { wacc_pct: 9.5, terminal_growth_pct: 2.4, target_price: 110.0, upside_pct: 0.0, verdict: 'fairly_valued' },
                { wacc_pct: 9.5, terminal_growth_pct: 2.5, target_price: 112.0, upside_pct: 2.0, verdict: 'fairly_valued' },
              ],
              [
                { wacc_pct: 10.0, terminal_growth_pct: 1.5, target_price: 80.0, upside_pct: -28.0, verdict: 'overvalued' },
                { wacc_pct: 10.0, terminal_growth_pct: 1.8, target_price: 85.0, upside_pct: -23.0, verdict: 'overvalued' },
                { wacc_pct: 10.0, terminal_growth_pct: 2.1, target_price: 90.0, upside_pct: -18.0, verdict: 'overvalued' },
                { wacc_pct: 10.0, terminal_growth_pct: 2.4, target_price: 96.0, upside_pct: -13.0, verdict: 'fairly_valued' },
                { wacc_pct: 10.0, terminal_growth_pct: 2.5, target_price: 98.0, upside_pct: -11.0, verdict: 'fairly_valued' },
              ],
            ],
            current_price: 110.0,
          },
          sensitivity_summary: {
            min_target_price: 80.0,
            max_target_price: 165.0,
            min_upside_pct: -28.0,
            max_upside_pct: 55.0,
            wacc_spread_pct: [8.0, 10.0],
            growth_spread_pct: [1.5, 2.5],
          },
        }}
      />
    )

    expect(screen.getByText(/5×5 Valuation Sensitivity Matrix/i)).toBeInTheDocument()
    const toggleButton = screen.getByRole('button', { name: /แสดงตาราง Sensitivity \(5×5\)/i })
    expect(toggleButton).toBeInTheDocument()

    // Click to open table
    fireEvent.click(toggleButton)
    expect(screen.getByText('WACC \\ g')).toBeInTheDocument()
    expect(screen.getAllByText('Base').length).toBeGreaterThanOrEqual(1)
    expect(screen.getByText(/ช่วงราคาประเมิน:/i)).toBeInTheDocument()
    expect(screen.getAllByText(/\$80\.0/i).length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText(/\$165\.0/i).length).toBeGreaterThanOrEqual(1)
  })

  it('renders Dual Valuation Method Comparison Bar (Gordon vs Exit Multiple)', () => {
    render(
      <ValuationWorkbenchCard
        reverseDcf={{
          ...mockReverseDcf,
          target_price_12m: 152.40,
          upside_12m_pct: 17.2,
          intrinsic_value_today: 140.0,
          target_price_exit_multiple_12m: 148.50,
          intrinsic_value_exit_multiple: 137.50,
          exit_multiple_used: 22.0,
          current_stock_price: 130.0,
        }}
      />
    )

    expect(screen.getByText(/Dual Valuation Method Comparison/i)).toBeInTheDocument()
    expect(screen.getByText(/1\. Perpetuity Growth DCF \(Gordon Model\)/i)).toBeInTheDocument()
    expect(screen.getAllByText(/\$152\.40/i).length).toBeGreaterThanOrEqual(1)
    expect(screen.getByText(/2\. Exit Multiple Method \(22x EV\/EBITDA\)/i)).toBeInTheDocument()
    expect(screen.getAllByText(/\$148\.50/i).length).toBeGreaterThanOrEqual(1)
  })

  it('renders Enterprise to Equity Value Bridge (Visual Waterfall)', () => {
    render(
      <ValuationWorkbenchCard
        reverseDcf={{
          ...mockReverseDcf,
          enterprise_value: 178000000000,
          sum_pv_5y_fcf: 35200000000,
          terminal_value_pv: 142800000000,
          net_cash_debt: 12500000000,
          equity_value: 190500000000,
          shares_outstanding: 1250000000,
          target_price_12m: 152.40,
        }}
      />
    )

    expect(screen.getByText(/Enterprise to Equity Value Bridge \(Visual Waterfall\)/i)).toBeInTheDocument()
    expect(screen.getByText(/1\. PV 5Y FCF/i)).toBeInTheDocument()
    expect(screen.getAllByText(/\$35\.20B/i).length).toBeGreaterThanOrEqual(1)
    expect(screen.getByText(/\(19\.8% of EV\)/i)).toBeInTheDocument()
    expect(screen.getByText(/\(\+\) 2\. PV Terminal Value/i)).toBeInTheDocument()
    expect(screen.getAllByText(/\$142\.80B/i).length).toBeGreaterThanOrEqual(1)
    expect(screen.getByText(/\(80\.2% of EV\)/i)).toBeInTheDocument()
    expect(screen.getByText(/\(=\) 3\. Enterprise Value/i)).toBeInTheDocument()
    expect(screen.getAllByText(/\$178\.00B/i).length).toBeGreaterThanOrEqual(1)
    expect(screen.getByText(/\(\+\) 4\. Net Cash/i)).toBeInTheDocument()
    expect(screen.getAllByText(/\+\$12\.50B/i).length).toBeGreaterThanOrEqual(1)
    expect(screen.getByText(/\(=\) 5\. Equity Value/i)).toBeInTheDocument()
    expect(screen.getAllByText(/\$190\.50B/i).length).toBeGreaterThanOrEqual(1)
  })

  it('renders and toggles 8-Column Cash Flow & Reinvestment Schedule Table', () => {
    render(
      <ValuationWorkbenchCard
        reverseDcf={{
          ...mockReverseDcf,
          historical_financials_3y: [
            { fiscal_year: '2021A', revenue: 100000000, revenue_growth_pct: null, ebit: 20000000, ebit_margin_pct: 20.0, capex: 5000000, depreciation: 4000000, fcf: 12000000 },
            { fiscal_year: '2022A', revenue: 115000000, revenue_growth_pct: 15.0, ebit: 24000000, ebit_margin_pct: 20.9, capex: 6000000, depreciation: 4500000, fcf: 14000000 },
            { fiscal_year: '2023A', revenue: 130000000, revenue_growth_pct: 13.0, ebit: 28000000, ebit_margin_pct: 21.5, capex: 7000000, depreciation: 5000000, fcf: 16000000 },
          ],
          explicit_forecast_5y: [
            { year_index: 1, projected_revenue: 145000000, projected_revenue_growth_pct: 11.5, projected_ebit: 32000000, projected_ebit_margin_pct: 22.0, projected_nopat: 25600000, projected_depreciation: 6000000, projected_capex: 8000000, projected_net_capex: 2000000, projected_delta_nwc: 300000, projected_reinvestment_currency: 2300000, projected_fcf: 23300000, discount_factor: 0.956, pv_fcf: 22274800 },
            { year_index: 2, projected_revenue: 160000000, projected_revenue_growth_pct: 10.3, projected_ebit: 36000000, projected_ebit_margin_pct: 22.5, projected_nopat: 28800000, projected_depreciation: 6500000, projected_capex: 8500000, projected_net_capex: 2000000, projected_delta_nwc: 300000, projected_reinvestment_currency: 2300000, projected_fcf: 26500000, discount_factor: 0.873, pv_fcf: 23134500 },
          ],
        }}
      />
    )

    expect(screen.getByText(/3Y History Actuals \| 5Y Staged Forecast FCF Schedule Table/i)).toBeInTheDocument()
    const toggleScheduleBtn = screen.getByRole('button', { name: /แสดงตารางงบกระแสเงินสด \(8 คอลัมน์\)/i })
    expect(toggleScheduleBtn).toBeInTheDocument()

    fireEvent.click(toggleScheduleBtn)
    expect(screen.getByText(/Financial Line Item/i)).toBeInTheDocument()
    expect(screen.getByText(/2021A/i)).toBeInTheDocument()
    expect(screen.getByText(/Year 1E/i)).toBeInTheDocument()
    expect(screen.getByText(/\(-\) Net CapEx \(CapEx - D&A\)/i)).toBeInTheDocument()
    expect(screen.getByText(/\(-\) ΔNWC \(Change in Working Capital\)/i)).toBeInTheDocument()
    expect(screen.getByText(/\(=\) Total Reinvestment \(Net CapEx \+ ΔNWC\)/i)).toBeInTheDocument()
    expect(screen.getByText(/\(=\) Unlevered Free Cash Flow \(UFCF\)/i)).toBeInTheDocument()
  })

  it('switches between 3 Sensitivity Suite tabs (Valuation, Operating, Macro Shock) and selects a cell', () => {
    render(
      <ValuationWorkbenchCard
        reverseDcf={{
          ...mockReverseDcf,
          sensitivity_matrices: {
            valuation: {
              wacc_values: [9.0, 9.5],
              terminal_growth_values: [2.0, 2.5],
              grid: [
                [
                  { wacc_pct: 9.0, terminal_growth_pct: 2.0, target_price: 120.0, upside_pct: 10.0, verdict: 'undervalued', is_base: true },
                  { wacc_pct: 9.0, terminal_growth_pct: 2.5, target_price: 130.0, upside_pct: 20.0, verdict: 'undervalued' },
                ],
                [
                  { wacc_pct: 9.5, terminal_growth_pct: 2.0, target_price: 105.0, upside_pct: -5.0, verdict: 'fairly_valued' },
                  { wacc_pct: 9.5, terminal_growth_pct: 2.5, target_price: 112.0, upside_pct: 2.0, verdict: 'fairly_valued' },
                ],
              ],
              current_price: 110.0,
            },
            operating: {
              growth_values: [8.0, 10.0],
              margin_values: [25.0, 30.0],
              grid: [
                [
                  { revenue_growth_pct: 8.0, ebit_margin_pct: 25.0, target_price: 115.0, upside_pct: 5.0, verdict: 'fairly_valued', is_base: true },
                  { revenue_growth_pct: 8.0, ebit_margin_pct: 30.0, target_price: 135.0, upside_pct: 25.0, verdict: 'undervalued' },
                ],
                [
                  { revenue_growth_pct: 10.0, ebit_margin_pct: 25.0, target_price: 125.0, upside_pct: 15.0, verdict: 'undervalued' },
                  { revenue_growth_pct: 10.0, ebit_margin_pct: 30.0, target_price: 145.0, upside_pct: 35.0, verdict: 'undervalued' },
                ],
              ],
              current_price: 110.0,
            },
            macro: {
              beta_values: [1.0, 1.2],
              yield_values: [4.0, 4.5],
              grid: [
                [
                  { beta: 1.0, risk_free_rate_pct: 4.0, effective_wacc_pct: 9.0, target_price: 125.0, upside_pct: 15.0, verdict: 'undervalued', is_base: true },
                  { beta: 1.0, risk_free_rate_pct: 4.5, effective_wacc_pct: 9.5, target_price: 112.0, upside_pct: 2.0, verdict: 'fairly_valued' },
                ],
                [
                  { beta: 1.2, risk_free_rate_pct: 4.0, effective_wacc_pct: 9.6, target_price: 110.0, upside_pct: 0.0, verdict: 'fairly_valued' },
                  { beta: 1.2, risk_free_rate_pct: 4.5, effective_wacc_pct: 10.1, target_price: 98.0, upside_pct: -11.0, verdict: 'overvalued' },
                ],
              ],
              current_price: 110.0,
            },
          },
        }}
      />
    )

    // Open sensitivity matrix
    const toggleBtn = screen.getByRole('button', { name: /แสดงตาราง Sensitivity \(5×5\)/i })
    fireEvent.click(toggleBtn)

    // Tab 1: Valuation
    expect(screen.getByRole('button', { name: /1\. Valuation \(WACC vs g\)/i })).toBeInTheDocument()
    expect(screen.getByText('WACC \\ g')).toBeInTheDocument()

    // Switch to Tab 2: Operating
    const operatingTabBtn = screen.getByRole('button', { name: /2\. Operating \(Growth vs Margin\)/i })
    fireEvent.click(operatingTabBtn)
    expect(screen.getByText('Growth \\ Margin')).toBeInTheDocument()
    expect(screen.getByText('25.0%')).toBeInTheDocument()

    // Click cell in operating table
    const cellEl = screen.getByText('$135.0')
    fireEvent.click(cellEl)
    expect(screen.getByText(/Selected Scenario:/i)).toBeInTheDocument()
    expect(screen.getByText(/Revenue Growth:/i)).toBeInTheDocument()
    expect(screen.getByText(/Target Price: \$135\.00/i)).toBeInTheDocument()

    // Switch to Tab 3: Macro Shock
    const macroTabBtn = screen.getByRole('button', { name: /3\. Macro Shock \(Beta vs 10Y Yield\)/i })
    fireEvent.click(macroTabBtn)
    expect(screen.getByText('Beta \\ 10Y Yield')).toBeInTheDocument()
    expect(screen.getByText('1.00β')).toBeInTheDocument()
  })

  it('switches scenarios using toggle and opens WACC inspector modal', () => {
    render(
      <ValuationWorkbenchCard
        reverseDcf={mockReverseDcf}
        dcf={mockDcf}
      />
    )

    // Scenario toggle
    const bearBtn = screen.getByRole('button', { name: 'Bear' })
    const bullBtn = screen.getByRole('button', { name: 'Bull' })
    expect(bearBtn).toBeInTheDocument()

    fireEvent.click(bearBtn)
    expect(screen.getByText('DCF Fair Value (BEAR)')).toBeInTheDocument()

    fireEvent.click(bullBtn)
    expect(screen.getByText('DCF Fair Value (BULL)')).toBeInTheDocument()

    // WACC Inspector button and modal
    const waccBtn = screen.getByTitle(/คลิกเพื่อดูการแจกแจงส่วนประกอบ WACC/i)
    expect(waccBtn).toBeInTheDocument()

    fireEvent.click(waccBtn)
    expect(screen.getByText(/WACC & Capital Structure Decomposition/i)).toBeInTheDocument()
    expect(screen.getByText(/WACC = \(E \/ V × K_e\) \+ \(D \/ V × K_d × \(1 - t\)\)/i)).toBeInTheDocument()
    expect(screen.getByText(/Cost of Equity \(K_e\)/i)).toBeInTheDocument()
    expect(screen.getByText('10.20%')).toBeInTheDocument()

    // Close modal
    const closeBtn = screen.getByRole('button', { name: 'ปิดหน้าต่าง' })
    fireEvent.click(closeBtn)
    expect(screen.queryByText(/WACC & Capital Structure Decomposition/i)).not.toBeInTheDocument()
  })

  it('renders Damodaran Synthetic Credit Rating Benchmark in WACC inspector modal', () => {
    const dcfWithCredit = {
      ...mockReverseDcf,
      synthetic_credit_rating: 'AAA',
      interest_coverage_ratio: 18.4,
      credit_spread_bps: 60,
    }

    render(
      <ValuationWorkbenchCard
        reverseDcf={dcfWithCredit}
        dcf={mockDcf}
      />
    )

    const waccBtn = screen.getByTitle(/คลิกเพื่อดูการแจกแจงส่วนประกอบ WACC/i)
    fireEvent.click(waccBtn)

    expect(screen.getByText(/Damodaran Credit Rating Benchmark/i)).toBeInTheDocument()
    expect(screen.getByText('AAA')).toBeInTheDocument()
    expect(screen.getByText('18.4x')).toBeInTheDocument()
    expect(screen.getByText('+60 bps')).toBeInTheDocument()
  })

  it('renders Functional OpEx rows in 8-Column schedule table when available', () => {
    const dcfWithOpex = {
      ...mockReverseDcf,
      historical_financials_3y: [
        {
          fiscal_year: '2023A',
          revenue: 100000000,
          ebit: 25000000,
          ebit_margin_pct: 25.0,
          gross_profit: 50000000,
          gross_margin_pct: 50.0,
          research_and_development: 12000000,
          rd_pct_of_rev: 12.0,
          selling_general_admin: 13000000,
          sga_pct_of_rev: 13.0,
        },
      ],
      explicit_forecast_5y: [
        {
          year_index: 1,
          projected_revenue: 110000000,
          projected_revenue_growth_pct: 10.0,
          projected_gross_profit: 56000000,
          projected_gross_margin_pct: 50.9,
          projected_rd: 13000000,
          projected_rd_pct_of_rev: 11.8,
          projected_sga: 14000000,
          projected_sga_pct_of_rev: 12.7,
          projected_ebit: 29000000,
          projected_ebit_margin_pct: 26.4,
          projected_nopat: 22910000,
          projected_depreciation: 4400000,
          projected_capex: 6000000,
          projected_net_capex: 1600000,
          projected_delta_nwc: 200000,
          projected_reinvestment_currency: 1800000,
          projected_fcf: 21110000,
          discount_factor: 0.95,
          pv_fcf: 20054500,
        },
      ],
    }

    render(
      <ValuationWorkbenchCard
        reverseDcf={dcfWithOpex}
        dcf={mockDcf}
      />
    )

    const schedBtn = screen.getByRole('button', { name: /แสดงตารางงบกระแสเงินสด/i })
    fireEvent.click(schedBtn)

    expect(screen.getByText('Gross Profit')).toBeInTheDocument()
    expect(screen.getByText('(-) Research & Development (R&D)')).toBeInTheDocument()
    expect(screen.getByText('(-) SG&A Expenses')).toBeInTheDocument()
    expect(screen.getByText('└ R&D % of Revenue')).toBeInTheDocument()
    expect(screen.getByText('└ SG&A % of Revenue')).toBeInTheDocument()
  })

  it('renders Implied Exit EV/EBITDA badges and EBITDA schedule rows', () => {
    const dcfWithExitEbitda = {
      ...mockReverseDcf,
      implied_exit_ebitda_multiple: 12.4,
      implied_perpetual_growth_from_exit_pct: 2.3,
      terminal_ebitda: 14500000000,
      exit_multiple_used: 16.0,
      target_price_exit_multiple_12m: 135.50,
      intrinsic_value_exit_multiple: 125.00,
      valuation_sanity_audit: {
        model_reliability_tier: 'high' as const,
        reliability_score: 90,
        terminal_value_pct_of_ev: 65.0,
        implied_exit_ebitda_multiple: 12.4,
      },
      explicit_forecast_5y: [
        {
          year_index: 1,
          projected_revenue: 100000000,
          projected_revenue_growth_pct: 10.0,
          projected_ebit: 25000000,
          projected_ebit_margin_pct: 25.0,
          projected_nopat: 19750000,
          projected_depreciation: 4000000,
          projected_capex: 5000000,
          projected_net_capex: 1000000,
          projected_delta_nwc: 200000,
          projected_reinvestment_currency: 1200000,
          projected_fcf: 18550000,
          discount_factor: 0.95,
          pv_fcf: 17622500,
        },
      ],
    }

    render(
      <ValuationWorkbenchCard
        reverseDcf={dcfWithExitEbitda}
        dcf={mockDcf}
      />
    )

    // Check Implied Exit EBITDA badge in header
    expect(screen.getByText(/📊 Exit: 12\.4x EBITDA/i)).toBeInTheDocument()

    // Check Gordon Model card and header have Implied Exit badges
    expect(screen.getAllByText(/Exit: 12\.4x EBITDA/i).length).toBeGreaterThanOrEqual(2)

    // Check Exit Multiple card has Implied Perpetual g badge
    expect(screen.getByText(/Implied g: 2\.3%/i)).toBeInTheDocument()

    // Open schedule table and check EBITDA rows
    const schedBtn = screen.getByRole('button', { name: /แสดงตารางงบกระแสเงินสด/i })
    fireEvent.click(schedBtn)

    expect(screen.getByText('EBITDA (EBIT + D&A)')).toBeInTheDocument()
    expect(screen.getByText('└ EBITDA Margin (%)')).toBeInTheDocument()
  })

  it('renders CapEx / D&A lifecycle indicator badge, schedule row, and updates 5Y schedule when scenario is toggled', () => {
    const dcfWithScenarios = {
      ...mockReverseDcf,
      historical_financials_3y: [
        {
          fiscal_year: '2023A',
          revenue: 100000000,
          ebit: 20000000,
          ebit_margin_pct: 20.0,
          capex: 5000000,
          depreciation: 4000000,
          fcf: 12000000,
          capex_to_depreciation_ratio: 1.25,
          reinvestment_lifecycle_phase: 'sustainable_maintenance' as const,
        },
      ],
      explicit_forecast_5y: [
        {
          year_index: 1,
          projected_revenue: 150000000,
          projected_revenue_growth_pct: 12.0,
          projected_ebit: 35000000,
          projected_ebit_margin_pct: 23.3,
          projected_nopat: 28000000,
          projected_depreciation: 6000000,
          projected_capex: 8000000,
          projected_net_capex: 2000000,
          projected_delta_nwc: 300000,
          projected_reinvestment_currency: 2300000,
          projected_fcf: 25700000,
          discount_factor: 0.95,
          pv_fcf: 24415000,
          capex_to_depreciation_ratio: 1.33,
          reinvestment_lifecycle_phase: 'sustainable_maintenance' as const,
        },
      ],
      explicit_forecast_5y_scenarios: {
        bear: [
          {
            year_index: 1,
            projected_revenue: 120000000,
            projected_revenue_growth_pct: 5.0,
            projected_ebit: 24000000,
            projected_ebit_margin_pct: 20.0,
            projected_nopat: 19200000,
            projected_depreciation: 6000000,
            projected_capex: 4000000,
            projected_net_capex: -2000000,
            projected_delta_nwc: 100000,
            projected_reinvestment_currency: -1900000,
            projected_fcf: 21100000,
            discount_factor: 0.95,
            pv_fcf: 20045000,
            capex_to_depreciation_ratio: 0.67,
            reinvestment_lifecycle_phase: 'underinvestment_warning' as const,
          },
        ],
        base: [
          {
            year_index: 1,
            projected_revenue: 150000000,
            projected_revenue_growth_pct: 12.0,
            projected_ebit: 35000000,
            projected_ebit_margin_pct: 23.3,
            projected_nopat: 28000000,
            projected_depreciation: 6000000,
            projected_capex: 8000000,
            projected_net_capex: 2000000,
            projected_delta_nwc: 300000,
            projected_reinvestment_currency: 2300000,
            projected_fcf: 25700000,
            discount_factor: 0.95,
            pv_fcf: 24415000,
            capex_to_depreciation_ratio: 1.33,
            reinvestment_lifecycle_phase: 'sustainable_maintenance' as const,
          },
        ],
        bull: [
          {
            year_index: 1,
            projected_revenue: 180000000,
            projected_revenue_growth_pct: 18.0,
            projected_ebit: 45000000,
            projected_ebit_margin_pct: 25.0,
            projected_nopat: 36000000,
            projected_depreciation: 6000000,
            projected_capex: 12000000,
            projected_net_capex: 6000000,
            projected_delta_nwc: 500000,
            projected_reinvestment_currency: 6500000,
            projected_fcf: 29500000,
            discount_factor: 0.95,
            pv_fcf: 28025000,
            capex_to_depreciation_ratio: 2.00,
            reinvestment_lifecycle_phase: 'aggressive_expansion' as const,
          },
        ],
      },
    }

    render(
      <ValuationWorkbenchCard
        reverseDcf={dcfWithScenarios}
        dcf={mockDcf}
      />
    )

    // Check Header Badge for CapEx / D&A
    expect(screen.getByText(/🏗️ CapEx\/D&A: 1\.33x \(Sustainable\)/i)).toBeInTheDocument()

    // Open schedule table
    const schedBtn = screen.getByRole('button', { name: /แสดงตารางงบกระแสเงินสด/i })
    fireEvent.click(schedBtn)

    // Check scenario tag in schedule table header
    expect(screen.getByText('Scenario: BASE')).toBeInTheDocument()

    // Check CapEx / D&A Coverage row
    expect(screen.getByText('└ CapEx / D&A Coverage')).toBeInTheDocument()
    expect(screen.getByText('1.25x')).toBeInTheDocument()
    expect(screen.getByText('1.33x')).toBeInTheDocument()
    expect(screen.getAllByText('Maint').length).toBeGreaterThanOrEqual(1)
    expect(screen.getByText('$150.00M')).toBeInTheDocument()

    // Switch to BULL scenario
    const bullBtn = screen.getByRole('button', { name: 'Bull' })
    fireEvent.click(bullBtn)

    expect(screen.getByText('Scenario: BULL')).toBeInTheDocument()
    expect(screen.getByText('$180.00M')).toBeInTheDocument()
    expect(screen.getByText('2.00x')).toBeInTheDocument()
    expect(screen.getByText('Exp')).toBeInTheDocument()

    // Switch to BEAR scenario
    const bearBtn = screen.getByRole('button', { name: 'Bear' })
    fireEvent.click(bearBtn)

    expect(screen.getByText('Scenario: BEAR')).toBeInTheDocument()
    expect(screen.getByText('$120.00M')).toBeInTheDocument()
    expect(screen.getByText('0.67x')).toBeInTheDocument()
    expect(screen.getByText('Under')).toBeInTheDocument()
  })

  it('renders Company Archetype badge and Mid-Cycle Normalization tag on EBIT margin row', () => {
    const dcfWithArchetype = {
      ...mockReverseDcf,
      company_archetype: 'cyclical' as const,
      is_mid_cycle_normalized: true,
      normalized_mid_cycle_ebit_margin_pct: 15.0,
      archetype_rationale: 'Cyclical profile with peak earnings normalized to 15.0% mid-cycle',
      historical_financials_3y: [
        {
          fiscal_year: '2023A',
          revenue: 100000000,
          ebit: 26000000,
          ebit_margin_pct: 26.0,
          capex: 6000000,
          depreciation: 5000000,
          fcf: 15000000,
        },
      ],
      explicit_forecast_5y: [
        {
          year_index: 1,
          projected_revenue: 105000000,
          projected_revenue_growth_pct: 5.0,
          projected_ebit: 27000000,
          projected_ebit_margin_pct: 25.7,
          projected_nopat: 21330000,
          projected_depreciation: 5200000,
          projected_capex: 6000000,
          projected_net_capex: 800000,
          projected_delta_nwc: 100000,
          projected_reinvestment_currency: 900000,
          projected_fcf: 20430000,
          discount_factor: 0.95,
          pv_fcf: 19408500,
        },
      ],
    }

    render(
      <ValuationWorkbenchCard
        reverseDcf={dcfWithArchetype}
        dcf={mockDcf}
      />
    )

    // Check Archetype Badge in Header
    expect(screen.getByText('🔄 Cyclical (Mid-Cycle: 15.0%)')).toBeInTheDocument()

    // Open schedule table
    const schedBtn = screen.getByRole('button', { name: /แสดงตารางงบกระแสเงินสด/i })
    fireEvent.click(schedBtn)

    // Check Mid-Cycle tag on EBIT margin row
    expect(screen.getByText('Mid-Cycle (15.0%)')).toBeInTheDocument()
  })

  it('renders Margin of Safety (MoS) Buy Target correctly when out of buy zone', () => {
    const dcfWithMos = {
      ...mockReverseDcf,
      mos_entry_price_today: 92.23,
      mos_target_price_12m: 99.65,
      margin_of_safety_pct: 15.0,
      current_margin_of_safety_pct: -5.0,
      is_in_buy_zone: false,
      required_pullback_pct: 18.2,
    }

    render(
      <ValuationWorkbenchCard
        reverseDcf={dcfWithMos}
        dcf={mockDcf}
      />
    )

    // Header badge
    expect(screen.getByText('Buy Below (15% MoS)')).toBeInTheDocument()
    expect(screen.getByText('-18%')).toBeInTheDocument()

    // Gordon card
    expect(screen.getByText(/🎯 Buy Below \(15% MoS\):/i)).toBeInTheDocument()
    expect(screen.getByText(/Pullback -18%/i)).toBeInTheDocument()
    expect(screen.getAllByText('$92.23').length).toBeGreaterThanOrEqual(1)
  })

  it('renders Margin of Safety (MoS) in buy zone badge when price is in buy zone', () => {
    const dcfInBuyZone = {
      ...mockReverseDcf,
      mos_entry_price_today: 120.0,
      mos_target_price_12m: 130.0,
      margin_of_safety_pct: 15.0,
      current_margin_of_safety_pct: 22.0,
      is_in_buy_zone: true,
      required_pullback_pct: null,
    }

    render(
      <ValuationWorkbenchCard
        reverseDcf={dcfInBuyZone}
        dcf={mockDcf}
      />
    )

    // Header and Gordon card badges
    expect(screen.getByText('🟢 Buy Zone')).toBeInTheDocument()
    expect(screen.getByText('In Buy Zone')).toBeInTheDocument()
    expect(screen.getAllByText('$120.00').length).toBeGreaterThanOrEqual(1)
  })

  it('renders embedded DCFScenariosChart breakdown inside Right Pane when dcf scenarios exist', () => {
    render(
      <ValuationWorkbenchCard
        reverseDcf={mockReverseDcf}
        dcf={mockDcf}
      />
    )

    expect(screen.getByText('Scenarios Target Price')).toBeInTheDocument()
    expect(screen.getByText(/Ke: 10\.2%, Kd: 5%/i)).toBeInTheDocument()
    expect(screen.getByText('-45%')).toBeInTheDocument()
    expect(screen.getByText('+15%')).toBeInTheDocument()
  })
})



