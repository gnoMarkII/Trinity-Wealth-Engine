import { describe, it, expect, vi, beforeEach } from 'vitest'
import { initKlineChart } from './klineAdapter'
import * as klinecharts from 'klinecharts'
import type { OHLCVCandleDTO } from '../../api/types'

vi.mock('klinecharts', () => {
  return {
    init: vi.fn(),
    dispose: vi.fn(),
    registerIndicator: vi.fn(),
    getSupportedIndicators: vi.fn(() => []),
  }
})

describe('klineAdapter', () => {
  let mockChart: any
  let container: HTMLDivElement

  const sampleBars: OHLCVCandleDTO[] = [
    { timestamp: 1700000000000, open: 150, high: 155, low: 149, close: 154, volume: 10000 },
    { timestamp: 1700086400000, open: 154, high: 158, low: 153, close: 157, volume: 12000 },
    { timestamp: 1700172800000, open: 157, high: 160, low: 156, close: 159, volume: 15000 },
  ]

  beforeEach(() => {
    vi.clearAllMocks()
    container = document.createElement('div')
    Object.defineProperty(container, 'clientWidth', { value: 600, configurable: true })

    mockChart = {
      createIndicator: vi.fn().mockImplementation(({ name }) => `ind_${name}_123`),
      removeIndicator: vi.fn().mockReturnValue(true),
      createOverlay: vi.fn().mockImplementation(({ name }) => `overlay_${name}_123`),
      removeOverlay: vi.fn().mockReturnValue(true),
      setPaneOptions: vi.fn(),
      setSymbol: vi.fn(),
      setPeriod: vi.fn(),
      setDataLoader: vi.fn(),
      resetData: vi.fn(),
      setBarSpace: vi.fn(),
      setOffsetRightDistance: vi.fn(),
      scrollToRealTime: vi.fn(),
      getVisibleRange: vi.fn(() => ({ from: 0, to: 10, realFrom: 0, realTo: 10 })),
      resize: vi.fn(),
    }
    vi.mocked(klinecharts.init).mockReturnValue(mockChart)
  })

  it('initializes chart with config, symbol, period, and dataLoader', () => {
    const instance = initKlineChart(
      container,
      sampleBars,
      'AAPL',
      { span: 1, type: 'day' },
      { overlay: 'BOLL', volume: false, momentum: 'NONE', atr: false, events: true, week52: true }
    )

    expect(klinecharts.init).toHaveBeenCalledWith(container, expect.any(Object))
    expect(mockChart.createIndicator).toHaveBeenCalledWith(
      { name: 'BOLL', paneId: 'candle_pane', calcParams: [20, 2] },
      true
    )
    expect(mockChart.setSymbol).toHaveBeenCalledWith({
      ticker: 'AAPL',
      pricePrecision: 2,
      volumePrecision: 0,
    })
    expect(mockChart.setPeriod).toHaveBeenCalledWith({ span: 1, type: 'day' })
    expect(mockChart.setDataLoader).toHaveBeenCalled()
    expect(mockChart.resetData).toHaveBeenCalled()
    expect(instance.getChart()).toBe(mockChart)
  })

  it('performs symmetric atomic switching for overlay indicators (BOLL -> EMA -> NONE)', () => {
    const instance = initKlineChart(container, sampleBars, 'AAPL', { span: 1, type: 'day' }, {
      overlay: 'BOLL',
      volume: false,
      momentum: 'NONE',
      atr: false,
      events: true,
      week52: true,
    })

    expect(instance.getEffectiveConfig().overlay).toBe('BOLL')

    // Switch BOLL -> EMA
    const switchSuccess = instance.setOverlay('EMA')
    expect(switchSuccess).toBe(true)
    expect(mockChart.createIndicator).toHaveBeenCalledWith(
      { name: 'EMA', paneId: 'candle_pane', calcParams: [20, 50, 200] },
      true
    )
    // Previous BOLL should be removed after new indicator creation
    expect(mockChart.removeIndicator).toHaveBeenCalledWith({
      paneId: 'candle_pane',
      id: 'ind_BOLL_123',
    })
    expect(instance.getEffectiveConfig().overlay).toBe('EMA')

    // Switch EMA -> NONE
    instance.setOverlay('NONE')
    expect(mockChart.removeIndicator).toHaveBeenCalledWith({
      paneId: 'candle_pane',
      id: 'ind_EMA_123',
    })
    expect(instance.getEffectiveConfig().overlay).toBe('NONE')
  })

  it('performs symmetric atomic switching for momentum indicators (RSI -> MACD -> NONE)', () => {
    const instance = initKlineChart(container, sampleBars, 'AAPL', { span: 1, type: 'day' }, {
      overlay: 'NONE',
      volume: false,
      momentum: 'NONE',
      atr: false,
      events: true,
      week52: true,
    })

    // Enable RSI
    instance.setMomentum('RSI')
    expect(mockChart.createIndicator).toHaveBeenCalledWith(
      { name: 'RSI14_WITH_LEVELS', paneId: 'pane_momentum' },
      false
    )
    expect(instance.getEffectiveConfig().momentum).toBe('RSI')

    // Switch RSI -> MACD
    instance.setMomentum('MACD')
    expect(mockChart.createIndicator).toHaveBeenCalledWith(
      { name: 'MACD', paneId: 'pane_momentum' },
      false
    )
    expect(mockChart.removeIndicator).toHaveBeenCalledWith({
      paneId: 'pane_momentum',
    })
    expect(instance.getEffectiveConfig().momentum).toBe('MACD')

    // Switch MACD -> NONE
    instance.setMomentum('NONE')
    expect(mockChart.removeIndicator).toHaveBeenCalledWith({
      paneId: 'pane_momentum',
    })
    expect(instance.getEffectiveConfig().momentum).toBe('NONE')
  })

  it('reconciles subpanes with strict order and creates native VOL with calcParams [20]', () => {
    const instance = initKlineChart(container, sampleBars, 'AAPL', { span: 1, type: 'day' }, {
      overlay: 'NONE',
      volume: false,
      momentum: 'NONE',
      atr: false,
      events: true,
      week52: true,
    })

    instance.setVolume(true)
    expect(mockChart.createIndicator).toHaveBeenCalledWith(
      { name: 'VOL', paneId: 'pane_vol', calcParams: [20] },
      false
    )
    expect(mockChart.setPaneOptions).toHaveBeenCalledWith({ id: 'pane_vol', order: 1 })

    instance.setMomentum('RSI')
    expect(mockChart.setPaneOptions).toHaveBeenCalledWith({ id: 'pane_momentum', order: 2 })

    instance.setATR(true)
    expect(mockChart.createIndicator).toHaveBeenCalledWith(
      { name: 'ATR14', paneId: 'pane_atr' },
      false
    )
    expect(mockChart.setPaneOptions).toHaveBeenCalledWith({ id: 'pane_atr', order: 3 })
  })

  it('creates and manages Corporate Action Overlays with stacking metadata', () => {
    const instance = initKlineChart(container, sampleBars, 'AAPL')

    const events = [
      {
        event_type: 'earnings' as const,
        timestamp: sampleBars[0]!.timestamp,
        date_str: '2023-11-15',
        label: 'E: +5%',
        color: 'green' as const,
        tooltip: 'Earnings Beat',
        mapping_method: 'reported_date' as const,
      },
      {
        event_type: 'ex_dividend' as const,
        timestamp: sampleBars[0]!.timestamp,
        date_str: '2023-11-15',
        label: 'XD: $0.24',
        color: 'blue' as const,
        tooltip: 'Dividend $0.24',
        mapping_method: 'reported_date' as const,
      },
    ]

    instance.setEvents(events, true)

    expect(mockChart.removeOverlay).toHaveBeenCalledWith({ groupId: 'corporate-events' })
    expect(mockChart.createOverlay).toHaveBeenCalledTimes(2)
    expect(mockChart.createOverlay).toHaveBeenCalledWith(
      expect.objectContaining({
        name: 'corporateEventMarker',
        groupId: 'corporate-events',
        paneId: 'candle_pane',
        lock: true,
        points: [{ timestamp: sampleBars[0]!.timestamp, value: sampleBars[0]!.high }],
        extendData: expect.objectContaining({
          stackIndex: 0,
          stackCount: 2,
        }),
      })
    )
  })

  it('creates 52W High and Low Level overlays and handles guard <= 0', () => {
    const instance = initKlineChart(container, sampleBars, 'AAPL')

    // Valid 52W Levels
    instance.set52WLevels({
      high: 180.5,
      low: 120.0,
      latestClose: 154.0,
      currency: 'USD',
      coverageCalendarDays: 365,
      visible: true,
    })

    expect(mockChart.removeOverlay).toHaveBeenCalledWith({ groupId: 'week52-levels' })
    expect(mockChart.createOverlay).toHaveBeenCalledWith(
      expect.objectContaining({
        name: 'system52WLevel',
        groupId: 'week52-levels',
        id: '52w-high',
        points: [{ timestamp: sampleBars[0]!.timestamp, value: 180.5 }],
      })
    )
    expect(mockChart.createOverlay).toHaveBeenCalledWith(
      expect.objectContaining({
        name: 'system52WLevel',
        groupId: 'week52-levels',
        id: '52w-low',
        points: [{ timestamp: sampleBars[0]!.timestamp, value: 120.0 }],
      })
    )

    // Guard <= 0 or null
    mockChart.createOverlay.mockClear()
    instance.set52WLevels({
      high: 0,
      low: -10,
      latestClose: 154.0,
      currency: 'USD',
      coverageCalendarDays: 365,
      visible: true,
    })
    expect(mockChart.createOverlay).not.toHaveBeenCalled()
  })

  it('manages drawing tools lifecycle and clears drawings on ticker change', () => {
    const instance = initKlineChart(container, sampleBars, 'AAPL')

    instance.enableDrawing('straightLine')
    expect(mockChart.createOverlay).toHaveBeenCalledWith(
      expect.objectContaining({
        name: 'straightLine',
        groupId: 'user-drawings',
      })
    )

    instance.clearDrawings()
    expect(mockChart.removeOverlay).toHaveBeenCalledWith({ groupId: 'user-drawings' })

    // Changing ticker should clear user drawings
    mockChart.removeOverlay.mockClear()
    instance.updateChart({ symbol: 'NVDA' })
    expect(mockChart.removeOverlay).toHaveBeenCalledWith({ groupId: 'user-drawings' })
  })

  it('calculates readability-first bar spacing on narrow mobile (360px) viewports with displayStartTimestamp', () => {
    Object.defineProperty(container, 'clientWidth', { value: 360, configurable: true })

    const longBars: OHLCVCandleDTO[] = Array.from({ length: 300 }, (_, i) => ({
      timestamp: 1700000000000 + i * 86400000,
      open: 100,
      high: 105,
      low: 95,
      close: 102,
      volume: 1000,
    }))

    // 60 visible bars starting from index 240
    const displayStartTs = longBars[240]!.timestamp

    initKlineChart(
      container,
      longBars,
      'AAPL',
      { span: 1, type: 'day' },
      { overlay: 'NONE', volume: false, momentum: 'NONE', atr: false, events: true, week52: true },
      displayStartTs
    )

    expect(mockChart.setBarSpace).toHaveBeenCalledWith(expect.any(Number))
    const barSpaceCall = mockChart.setBarSpace.mock.calls[0][0]
    expect(barSpaceCall).toBeGreaterThanOrEqual(3)
    expect(mockChart.scrollToRealTime).toHaveBeenCalled()
  })

  it('renders DCF valuation levels when targets are comparable', () => {
    const instance = initKlineChart(container, sampleBars, 'AAPL')

    instance.setValuationTargets({
      targets: {
        evaluation_id: 'eval_123',
        ticker: 'AAPL',
        market: 'US',
        currency: 'USD',
        chart_price_basis: 'provider_proportional_adj_close_ratio',
        valuation_price_basis: 'split_adjusted_only',
        comparability_status: 'comparable',
        comparability_reasons: [],
        corporate_action_factors: [],
        evaluated_at: '2026-08-20T10:00:00Z',
        as_of_label: 'as of 2026-08-20',
        model_version: 'dcf_v1.0',
        valuation_verdict: 'undervalued',
        macro_observable_refs: [],
        data_quality_flags: [],
        status: 'available',
        scenario_order_valid: true,
        scenarios: [
          { scenario_name: 'base', label: 'DCF Base', target_price: 180.0, upside_pct: 15.0, color: 'emerald' },
          { scenario_name: 'bull', label: 'DCF Bull', target_price: 210.0, upside_pct: 35.0, color: 'green' },
          { scenario_name: 'bear', label: 'DCF Bear', target_price: 140.0, upside_pct: -10.0, color: 'rose' },
        ],
      },
      visible: true,
    })

    expect(mockChart.removeOverlay).toHaveBeenCalledWith({ groupId: 'dcf-levels' })
    expect(mockChart.createOverlay).toHaveBeenCalledWith(
      expect.objectContaining({
        name: 'valuationLevelRay',
        groupId: 'dcf-levels',
        id: 'dcf-base',
      })
    )
  })

  it('suppresses DCF valuation levels when comparability_status is not_comparable', () => {
    const instance = initKlineChart(container, sampleBars, 'AAPL')

    instance.setValuationTargets({
      targets: {
        evaluation_id: 'eval_123',
        ticker: 'AAPL',
        market: 'US',
        currency: 'USD',
        chart_price_basis: 'provider_proportional_adj_close_ratio',
        valuation_price_basis: 'split_adjusted_only',
        comparability_status: 'not_comparable',
        comparability_reasons: ['Stock split occurred'],
        corporate_action_factors: [],
        evaluated_at: '2026-08-20T10:00:00Z',
        as_of_label: 'as of 2026-08-20',
        model_version: 'dcf_v1.0',
        valuation_verdict: 'undervalued',
        macro_observable_refs: [],
        data_quality_flags: [],
        status: 'available',
        scenario_order_valid: true,
        scenarios: [
          { scenario_name: 'base', label: 'DCF Base', target_price: 180.0, upside_pct: 15.0, color: 'emerald' },
        ],
      },
      visible: true,
    })

    expect(mockChart.removeOverlay).toHaveBeenCalledWith({ groupId: 'dcf-levels' })
    expect(mockChart.createOverlay).not.toHaveBeenCalledWith(
      expect.objectContaining({
        groupId: 'dcf-levels',
      })
    )
  })

  it('renders SEC Insider overlays with isolated groupId: insider-events', () => {
    const instance = initKlineChart(container, sampleBars, 'AAPL')

    instance.setInsiderFilings(
      [
        {
          accession_number: '0000320193-26-000100',
          issuer_cik: '0000320193',
          ticker: 'AAPL',
          filing_url: 'https://sec.gov/...',
          filed_at: '2026-08-15',
          timestamp: 1700000000000,
          reporting_owner_name: 'Cook Tim',
          is_officer: true,
          is_director: true,
          is_ten_percent_owner: false,
          is_amendment: false,
          is_cluster_buy: true,
          transactions: [
            {
              transaction_id: 'tx_1',
              transaction_date: '2026-08-15',
              transaction_code: 'P',
              shares: 50000,
              price_per_share: 220.5,
              acquired_or_disposed: 'A',
              is_derivative: false,
              normalized_weight: 1.0,
            },
          ],
        },
      ],
      true
    )

    expect(mockChart.removeOverlay).toHaveBeenCalledWith({ groupId: 'insider-events' })
    expect(mockChart.createOverlay).toHaveBeenCalledWith(
      expect.objectContaining({
        name: 'secInsiderMarker',
        groupId: 'insider-events',
        id: 'insider-0000320193-26-000100',
        onClick: expect.any(Function),
        onMouseEnter: expect.any(Function),
        onMouseLeave: expect.any(Function),
        onPressedMoving: expect.any(Function),
        onPressedMoveEnd: expect.any(Function),
      })
    )
  })

  it('toggles VWAP indicator on candle_pane with isStack: true and custom exchange timezone', () => {
    const instance = initKlineChart(container, sampleBars, 'AAPL')

    // Enable VWAP
    const enabled = instance.setVWAP?.(true, 'America/New_York')
    expect(enabled).toBe(true)
    expect(mockChart.createIndicator).toHaveBeenCalledWith(
      {
        name: 'SESSION_VWAP',
        paneId: 'candle_pane',
        extendData: { exchangeTz: 'America/New_York' },
      },
      true
    )

    // Disable VWAP
    instance.setVWAP?.(false)
    expect(mockChart.removeIndicator).toHaveBeenCalledWith({
      paneId: 'candle_pane',
      id: expect.stringContaining('SESSION_VWAP'),
    })
  })

  it('renders and cleans up analyst target ray overlay with proper formatting and currency check', () => {
    const instance = initKlineChart(container, sampleBars, 'AAPL')

    // Valid analyst context
    instance.setAnalystContext?.({
      ctx: {
        ticker: 'AAPL',
        provider_symbol: 'AAPL',
        provider_tier: 'best_effort',
        market: 'US',
        currency: 'USD',
        target_mean: 230.0,
        target_high: 250.0,
        target_low: 180.0,
        num_analysts: 38,
        next_earnings_date: '2026-10-15',
        days_to_earnings: 54,
        earnings_history: [],
        data_status: 'ok',
        source_as_of: '2026-08-22T00:00:00',
        synced_at: '2026-08-22T00:00:00Z',
        exchange_tz: 'America/New_York',
      },
      visible: true,
      currentPrice: 200.0,
      currency: 'USD',
    })

    expect(mockChart.removeOverlay).toHaveBeenCalledWith({ groupId: 'analyst-target' })
    expect(mockChart.createOverlay).toHaveBeenCalledWith(
      expect.objectContaining({
        name: 'analystTargetRay',
        groupId: 'analyst-target',
        id: 'analyst-mean-target',
        extendData: expect.objectContaining({
          label: 'Street: $230.00 (+15.0%)',
          color: '#f59e0b',
        }),
      })
    )

    // Mismatched currency suppresses overlay
    mockChart.createOverlay.mockClear()
    instance.setAnalystContext?.({
      ctx: {
        ticker: 'AAPL',
        provider_symbol: 'AAPL',
        provider_tier: 'best_effort',
        market: 'US',
        currency: 'USD',
        target_mean: 230.0,
        target_high: 250.0,
        target_low: 180.0,
        num_analysts: 38,
        next_earnings_date: '2026-10-15',
        days_to_earnings: 54,
        earnings_history: [],
        data_status: 'ok',
        source_as_of: '2026-08-22T00:00:00',
        synced_at: '2026-08-22T00:00:00Z',
        exchange_tz: 'America/New_York',
      },
      visible: true,
      currentPrice: 200.0,
      currency: 'THB',
    })
    expect(mockChart.createOverlay).not.toHaveBeenCalledWith(
      expect.objectContaining({ groupId: 'analyst-target' })
    )
  })

  it('delegates resize and destroy correctly', () => {
    const instance = initKlineChart(container, sampleBars, 'AAPL')

    instance.resize()
    expect(mockChart.resize).toHaveBeenCalled()

    instance.destroy()
    expect(klinecharts.dispose).toHaveBeenCalledWith(container)
  })
})


