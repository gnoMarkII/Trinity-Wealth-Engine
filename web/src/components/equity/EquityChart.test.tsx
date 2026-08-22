import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { EquityChart } from './EquityChart'
import type { OHLCVCandleDTO } from '../../api/types'

vi.mock('./klineAdapter', () => {
  return {
    initKlineChart: vi.fn(() => ({
      updateChart: vi.fn(),
      setOverlay: vi.fn(),
      setVolume: vi.fn(),
      setMomentum: vi.fn(),
      setATR: vi.fn(),
      setEvents: vi.fn(),
      set52WLevels: vi.fn(),
      setValuationTargets: vi.fn(),
      setInsiderFilings: vi.fn(),
      enableDrawing: vi.fn(),
      cancelActiveDrawing: vi.fn(),
      clearDrawings: vi.fn(),
      getEffectiveConfig: vi.fn(() => ({
        overlay: 'BOLL',
        volume: false,
        momentum: 'NONE',
        atr: false,
        events: true,
        week52: true,
        dcf: true,
        insider: true,
      })),
      resize: vi.fn(),
      destroy: vi.fn(),
      getChart: vi.fn(),
    })),
  }
})

describe('EquityChart Component', () => {
  const sampleCandles: OHLCVCandleDTO[] = [
    { timestamp: 1700000000000, open: 150, high: 155, low: 149, close: 154, volume: 10000 },
  ]

  beforeEach(() => {
    vi.clearAllMocks()
    localStorage.clear()
  })

  it('renders header, execution badges, range selectors, and static hint', () => {
    render(
      <EquityChart
        ticker="AAPL"
        companyName="Apple Inc."
        currentPrice={180.5}
        priceChange={2.5}
        priceChangePct={1.4}
        priceAsOf="2026-08-19T16:00:00Z"
        currency="USD"
        candles={sampleCandles}
        selectedRange="6mo"
        onRangeChange={vi.fn()}
        isLoading={false}
      />
    )

    expect(screen.getByText('AAPL')).toBeInTheDocument()
    expect(screen.getByText('Apple Inc.')).toBeInTheDocument()
    expect(screen.getByText('$180.50')).toBeInTheDocument()
    expect(screen.getByText('BOLL (20, 2)')).toBeInTheDocument()
    expect(
      screen.getByText(/กราฟแสดงราคาปรับสิทธิ \(Adjusted Basis\)/)
    ).toBeInTheDocument()
  })

  it('handles accessible toolbar keyboard navigation (ArrowRight / ArrowLeft)', async () => {
    const user = userEvent.setup()

    render(
      <EquityChart
        ticker="AAPL"
        currentPrice={180.5}
        priceChange={2.5}
        priceChangePct={1.4}
        currency="USD"
        candles={sampleCandles}
        selectedRange="6mo"
        onRangeChange={vi.fn()}
        isLoading={false}
      />
    )

    const toolbar = screen.getByRole('toolbar', { name: /technical indicators/i })
    expect(toolbar).toBeInTheDocument()

    const bollBtn = screen.getByRole('button', { name: 'BOLL' })
    bollBtn.focus()
    expect(document.activeElement).toBe(bollBtn)

    // Press ArrowRight to move focus to next button (EMA)
    fireEvent.keyDown(toolbar, { key: 'ArrowRight' })
    const emaBtn = screen.getByRole('button', { name: 'EMA' })
    expect(document.activeElement).toBe(emaBtn)

    // Press Enter to select EMA
    await user.keyboard('{Enter}')
    expect(screen.getByText('EMA (20, 50, 200)')).toBeInTheDocument()
  })

  it('safely falls back to default config if localStorage contains corrupted or invalid schema', () => {
    localStorage.setItem('equity_chart_indicator_config:v1', 'INVALID_JSON{')

    render(
      <EquityChart
        ticker="AAPL"
        currentPrice={180.5}
        priceChange={2.5}
        priceChangePct={1.4}
        currency="USD"
        candles={sampleCandles}
        selectedRange="6mo"
        onRangeChange={vi.fn()}
        isLoading={false}
      />
    )

    // Should fall back to default BOLL without throwing
    expect(screen.getByText('BOLL (20, 2)')).toBeInTheDocument()
  })

  it('displays Quality Warning Banner only when warmupStatus is insufficient and EMA is active', async () => {
    const user = userEvent.setup()

    const { rerender } = render(
      <EquityChart
        ticker="AAPL"
        currentPrice={180.5}
        priceChange={2.5}
        priceChangePct={1.4}
        currency="USD"
        candles={sampleCandles}
        selectedRange="6mo"
        onRangeChange={vi.fn()}
        isLoading={false}
        availableWarmupBars={60}
        warmupStatus="insufficient"
      />
    )

    // Default overlay is BOLL -> warning banner is NOT shown
    expect(screen.queryByTestId('quality-warning-banner')).not.toBeInTheDocument()

    // Switch overlay to EMA
    const emaBtn = screen.getByRole('button', { name: 'EMA' })
    await user.click(emaBtn)

    // Now warning banner should appear
    const banner = screen.getByTestId('quality-warning-banner')
    expect(banner).toBeInTheDocument()
    expect(banner).toHaveTextContent('ประวัติก่อนแสดงผลมี 60 แท่ง (EMA 200 อาจยังไม่เสถียร)')

    // If warmupStatus becomes "not_applicable" (e.g. max range) or "sufficient" -> warning is hidden
    rerender(
      <EquityChart
        ticker="AAPL"
        currentPrice={180.5}
        priceChange={2.5}
        priceChangePct={1.4}
        currency="USD"
        candles={sampleCandles}
        selectedRange="max"
        onRangeChange={vi.fn()}
        isLoading={false}
        availableWarmupBars={0}
        warmupStatus="not_applicable"
      />
    )
    expect(screen.queryByTestId('quality-warning-banner')).not.toBeInTheDocument()
  })

  it('renders context indicators (ATR, 52W, Events) and drawing tool interactions', async () => {
    const user = userEvent.setup()

    render(
      <EquityChart
        ticker="AAPL"
        currentPrice={180.5}
        priceChange={2.5}
        priceChangePct={1.4}
        currency="USD"
        candles={sampleCandles}
        selectedRange="6mo"
        onRangeChange={vi.fn()}
        isLoading={false}
        week52High={190.0}
        week52Low={130.0}
      />
    )

    // Toggle ATR button
    const atrBtn = screen.getByRole('button', { name: 'ATR(14)' })
    await user.click(atrBtn)
    expect(screen.getByText('ATR (14)')).toBeInTheDocument()

    // Drawing tool button click
    const lineBtn = screen.getByRole('button', { name: /Line/i })
    await user.click(lineBtn)
    expect(screen.getByText(/โหมดวาด: คลิก 2 จุดเพื่อวางเส้น/)).toBeInTheDocument()

    // Press Escape to cancel active drawing
    fireEvent.keyDown(window, { key: 'Escape' })
    expect(screen.queryByText(/โหมดวาด: คลิก 2 จุดเพื่อวางเส้น/)).not.toBeInTheDocument()
  })

  it('displays Corporate Actions metadata status badges correctly', () => {
    const { rerender } = render(
      <EquityChart
        ticker="AAPL"
        currentPrice={180.5}
        priceChange={2.5}
        priceChangePct={1.4}
        currency="USD"
        candles={sampleCandles}
        selectedRange="6mo"
        onRangeChange={vi.fn()}
        isLoading={false}
        eventsMetadata={{
          status: 'partial',
          earnings_status: 'ok',
          dividends_status: 'failed',
          splits_status: 'empty',
          missing_sources: ['dividends'],
          data_provenance: 'yfinance',
        }}
      />
    )

    expect(screen.getByText(/⚠️ Partial Events/)).toBeInTheDocument()

    rerender(
      <EquityChart
        ticker="AAPL"
        currentPrice={180.5}
        priceChange={2.5}
        priceChangePct={1.4}
        currency="USD"
        candles={sampleCandles}
        selectedRange="6mo"
        onRangeChange={vi.fn()}
        isLoading={false}
        eventsMetadata={{
          status: 'unavailable',
          earnings_status: 'failed',
          dividends_status: 'failed',
          splits_status: 'failed',
          missing_sources: ['earnings', 'dividends', 'splits'],
          data_provenance: 'yfinance',
        }}
      />
    )

    expect(screen.getByText(/Events unavailable/)).toBeInTheDocument()
  })
})
