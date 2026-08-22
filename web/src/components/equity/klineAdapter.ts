import { init, dispose, type Chart, type KLineData } from 'klinecharts'
import type {
  CorporateActionEventDTO,
  InsiderFilingDTO,
  InsiderMarkerHoverDTO,
  OHLCVCandleDTO,
  ValuationTargetsDTO,
  AnalystContextDTO,
} from '../../api/types'
import {
  registerRSI14WithLevels,
  registerATR14,
  registerCorporateActionOverlay,
  register52WLevelOverlay,
  registerValuationLevelOverlay,
  registerInsiderOverlay,
  registerAnalystTargetOverlay,
  registerVWAPIndicator,
} from './indicators'

export type OverlayIndicatorType = 'BOLL' | 'EMA' | 'SMA' | 'NONE'
export type MomentumIndicatorType = 'RSI' | 'MACD' | 'NONE'
export type DrawingToolType = 'straightLine' | 'horizontalStraightLine' | 'fibonacciLine' | null

export interface IndicatorConfig {
  overlay: OverlayIndicatorType
  volume: boolean
  momentum: MomentumIndicatorType
  atr: boolean
  events: boolean
  week52: boolean
  dcf: boolean
  insider: boolean
  vwap: boolean
  consensus: boolean
}

export interface ChartPeriod {
  span: number
  type: 'day' | 'week' | 'month' | 'minute' | 'hour'
}

export interface KlineChartInstance {
  updateChart(params: {
    symbol?: string
    period?: ChartPeriod
    bars?: OHLCVCandleDTO[]
    displayStartTimestamp?: number | null
    events?: CorporateActionEventDTO[]
    week52High?: number | null
    week52Low?: number | null
    latestClose?: number | null
    currency?: 'USD' | 'THB'
    coverageCalendarDays?: number
    valuationTargets?: ValuationTargetsDTO | null
    insiderFilings?: InsiderFilingDTO[]
    analystContext?: AnalystContextDTO | null
  }): void
  setOverlay(overlay: OverlayIndicatorType): boolean
  setVolume(enabled: boolean): boolean
  setMomentum(momentum: MomentumIndicatorType): boolean
  setATR(enabled: boolean): boolean
  setVWAP(enabled: boolean, exchangeTz?: string): boolean
  setEvents(
    events: CorporateActionEventDTO[],
    visible: boolean,
    onHover?: (event: CorporateActionEventDTO | null, pos: { x: number; y: number } | null) => void
  ): void
  set52WLevels(params: {
    high: number | null
    low: number | null
    latestClose: number | null
    currency: 'USD' | 'THB'
    coverageCalendarDays: number
    visible: boolean
  }): void
  setValuationTargets(params: {
    targets: ValuationTargetsDTO | null
    visible: boolean
  }): void
  setAnalystContext(params: {
    ctx: AnalystContextDTO | null
    visible: boolean
    currentPrice: number | null
    currency: 'USD' | 'THB'
  }): void
  setInsiderFilings(
    filings: InsiderFilingDTO[],
    visible: boolean,
    onHover?: (payload: { filing: InsiderMarkerHoverDTO; pos: { x: number; y: number } } | null, isHover: boolean) => void
  ): void
  enableDrawing(tool: DrawingToolType): void
  cancelActiveDrawing(): void
  clearDrawings(): void
  getEffectiveConfig(): IndicatorConfig
  resize(): void
  destroy(): void
  getChart(): Chart | null
}



export function initKlineChart(
  container: HTMLElement,
  initialBars: OHLCVCandleDTO[],
  symbol: string,
  period: ChartPeriod = { span: 1, type: 'day' },
  initialConfig: Partial<IndicatorConfig> = {
    overlay: 'BOLL',
    volume: false,
    momentum: 'NONE',
    atr: false,
    events: true,
    week52: true,
    dcf: true,
    insider: true,
  },
  initialDisplayStartTimestamp?: number | null
): KlineChartInstance {
  // 1. Ensure custom indicators and overlays are registered
  registerRSI14WithLevels()
  registerATR14()
  registerCorporateActionOverlay()
  register52WLevelOverlay()
  registerValuationLevelOverlay()
  registerInsiderOverlay()
  registerAnalystTargetOverlay()
  registerVWAPIndicator()

  const chart = init(container, {
    styles: {
      grid: {
        show: true,
        horizontal: {
          show: true,
          size: 1,
          color: '#f1f5f9',
          style: 'dashed',
          dashedValue: [4, 4],
        },
        vertical: {
          show: true,
          size: 1,
          color: '#f1f5f9',
          style: 'dashed',
          dashedValue: [4, 4],
        },
      },
      candle: {
        bar: {
          upColor: '#16a34a',
          downColor: '#dc2626',
          noChangeColor: '#888888',
          upBorderColor: '#16a34a',
          downBorderColor: '#dc2626',
          noChangeBorderColor: '#888888',
          upWickColor: '#16a34a',
          downWickColor: '#dc2626',
          noChangeWickColor: '#888888',
        },
        tooltip: {
          showRule: 'always',
          showType: 'standard',
        },
      },
      overlay: {
        text: {
          color: '#334155',
          size: 10,
          family: 'ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
          weight: 'normal',
          paddingLeft: 4,
          paddingRight: 4,
          paddingTop: 2,
          paddingBottom: 2,
          borderRadius: 3,
          borderSize: 0,
          borderColor: 'transparent',
          backgroundColor: 'transparent',
        },
      },
    },
  })

  const effectiveInitialConfig: IndicatorConfig = {
    overlay: initialConfig.overlay ?? 'BOLL',
    volume: initialConfig.volume ?? false,
    momentum: initialConfig.momentum ?? 'NONE',
    atr: initialConfig.atr ?? false,
    events: initialConfig.events ?? true,
    week52: initialConfig.week52 ?? false,
    dcf: initialConfig.dcf ?? false,
    insider: initialConfig.insider ?? false,
    vwap: initialConfig.vwap ?? false,
    consensus: initialConfig.consensus ?? true,
  }

  let currentBars: KLineData[] = (initialBars || []).map((b) => ({
    timestamp: b.timestamp,
    open: b.open,
    high: b.high,
    low: b.low,
    close: b.close,
    volume: b.volume,
  }))

  let currentDisplayStartTimestamp = initialDisplayStartTimestamp

  // Track active overlay on main candle pane
  let activeOverlayId: string | null = null
  let activeOverlayType: OverlayIndicatorType = 'NONE'

  // Subpane reconciliation state
  let currentSubpaneState = {
    volume: false,
    momentum: 'NONE' as MomentumIndicatorType,
    atr: false,
  }
  let activeVolId: string | null = null
  let activeMomentumId: string | null = null
  let activeMomentumType: MomentumIndicatorType = 'NONE'
  let activeAtrId: string | null = null

  // VWAP state
  let isVwapEnabled = effectiveInitialConfig.vwap
  let activeVwapId: string | null = null
  let activeVwapExchangeTz: string | null = null

  // System events state
  let isEventsVisible = effectiveInitialConfig.events
  let currentEvents: CorporateActionEventDTO[] = []
  let onCorporateHoverCb:
    | ((event: CorporateActionEventDTO | null, pos: { x: number; y: number } | null) => void)
    | undefined

  // 52W level lines state
  let is52WVisible = effectiveInitialConfig.week52
  let current52WHigh: number | null = null
  let current52WLow: number | null = null
  let currentLatestClose: number | null = null
  let currentCurrency: 'USD' | 'THB' = 'USD'
  let currentCoverageDays = 0

  // DCF Valuation state
  let isValuationVisible = effectiveInitialConfig.dcf

  // Analyst Target state
  let isConsensusVisible = effectiveInitialConfig.consensus
  let currentAnalystContext: AnalystContextDTO | null = null

  // SEC Insider state
  let isInsiderVisible = effectiveInitialConfig.insider
  let onInsiderHoverCb:
    | ((payload: { filing: InsiderMarkerHoverDTO; pos: { x: number; y: number } } | null, isHover: boolean) => void)
    | undefined

  // Drawing tools state
  let activeDrawingId: string | null = null

  const applyBarSpacingAndScroll = () => {
    if (!chart || !container) return
    const containerWidth = container.clientWidth || 600
    const totalBars = currentBars.length
    if (totalBars === 0) return

    const displayStartIndex = currentDisplayStartTimestamp
      ? currentBars.findIndex((b) => b.timestamp >= currentDisplayStartTimestamp!)
      : 0
    const targetVisibleCount = displayStartIndex >= 0 ? totalBars - displayStartIndex : totalBars

    const availableWidth = Math.max(150, containerWidth - 65)
    // Readability-First: minimum bar space of 3px
    let optimalBarSpace = Math.max(3, Math.min(28, availableWidth / (targetVisibleCount + 2)))
    chart.setBarSpace(optimalBarSpace)
    chart.setOffsetRightDistance(20)
    chart.scrollToRealTime()

    // Inspect actual viewport via getVisibleRange API & re-check loop
    if (typeof chart.getVisibleRange === 'function') {
      let visibleRange = chart.getVisibleRange()
      let attempts = 0
      while (visibleRange && displayStartIndex > 0 && visibleRange.realFrom < displayStartIndex && attempts < 4) {
        optimalBarSpace = Math.min(50, optimalBarSpace * 1.25)
        chart.setBarSpace(optimalBarSpace)
        chart.scrollToRealTime()
        visibleRange = chart.getVisibleRange()
        attempts++
      }
    }
  }

  const setOverlay = (overlay: OverlayIndicatorType): boolean => {
    if (!chart) return false
    if (overlay === activeOverlayType) return true

    if (overlay === 'NONE') {
      if (activeOverlayId) {
        chart.removeIndicator({ paneId: 'candle_pane', id: activeOverlayId })
      } else if (activeOverlayType !== 'NONE') {
        chart.removeIndicator({ paneId: 'candle_pane', name: activeOverlayType })
      }
      activeOverlayId = null
      activeOverlayType = 'NONE'
      return true
    }

    try {
      let newId: string | null = null
      if (overlay === 'EMA') {
        newId = chart.createIndicator(
          { name: 'EMA', paneId: 'candle_pane', calcParams: [20, 50, 200] },
          true
        )
      } else if (overlay === 'SMA') {
        newId = chart.createIndicator(
          { name: 'SMA', paneId: 'candle_pane', calcParams: [20, 50] },
          true
        )
      } else if (overlay === 'BOLL') {
        newId = chart.createIndicator(
          { name: 'BOLL', paneId: 'candle_pane', calcParams: [20, 2] },
          true
        )
      }

      if (newId || newId === null) {
        if (activeOverlayId) {
          chart.removeIndicator({ paneId: 'candle_pane', id: activeOverlayId })
        } else if (activeOverlayType !== 'NONE') {
          chart.removeIndicator({ paneId: 'candle_pane', name: activeOverlayType })
        }
        activeOverlayId = newId
        activeOverlayType = overlay
        return true
      }
      return false
    } catch {
      return false
    }
  }

  /**
   * Idempotent Subpane Reconciliation Engine with Strict Order Enforcement
   */
  const syncSubpanes = (desired: {
    volume: boolean
    momentum: MomentumIndicatorType
    atr: boolean
  }): void => {
    if (!chart) return

    // 1. Reconcile Volume Pane ('pane_vol') with native VOL + calcParams [20]
    if (desired.volume) {
      if (!activeVolId) {
        activeVolId = chart.createIndicator(
          { name: 'VOL', paneId: 'pane_vol', calcParams: [20] },
          false
        )
      }
      chart.setPaneOptions?.({ id: 'pane_vol', order: 1 })
    } else {
      if (activeVolId) {
        chart.removeIndicator({ paneId: 'pane_vol' })
        activeVolId = null
      }
    }

    // 2. Reconcile Momentum Pane ('pane_momentum')
    if (desired.momentum !== 'NONE') {
      const targetName = desired.momentum === 'RSI' ? 'RSI14_WITH_LEVELS' : 'MACD'
      if (desired.momentum !== activeMomentumType || !activeMomentumId) {
        if (activeMomentumId) {
          chart.removeIndicator({ paneId: 'pane_momentum' })
        }
        activeMomentumId = chart.createIndicator(
          { name: targetName, paneId: 'pane_momentum' },
          false
        )
        activeMomentumType = desired.momentum
      }
      chart.setPaneOptions?.({ id: 'pane_momentum', order: 2 })
    } else {
      if (activeMomentumId) {
        chart.removeIndicator({ paneId: 'pane_momentum' })
        activeMomentumId = null
        activeMomentumType = 'NONE'
      }
    }

    // 3. Reconcile ATR Pane ('pane_atr')
    if (desired.atr) {
      if (!activeAtrId) {
        activeAtrId = chart.createIndicator(
          { name: 'ATR14', paneId: 'pane_atr' },
          false
        )
      }
      chart.setPaneOptions?.({ id: 'pane_atr', order: 3 })
    } else {
      if (activeAtrId) {
        chart.removeIndicator({ paneId: 'pane_atr' })
        activeAtrId = null
      }
    }

    currentSubpaneState = { ...desired }
  }

  const setEvents = (
    events: CorporateActionEventDTO[],
    visible: boolean,
    onHover?: (event: CorporateActionEventDTO | null, pos: { x: number; y: number } | null) => void
  ): void => {
    if (!chart) return
    currentEvents = events || []
    isEventsVisible = visible
    if (onHover !== undefined) {
      onCorporateHoverCb = onHover
    }

    try {
      chart.removeOverlay({ groupId: 'corporate-events' })
    } catch {
      // Ignore
    }

    if (!visible || currentEvents.length === 0 || currentBars.length === 0) {
      return
    }

    // Group events by timestamp for deterministic stacking offset
    const grouped = new Map<number, CorporateActionEventDTO[]>()
    for (const ev of currentEvents) {
      const list = grouped.get(ev.timestamp) || []
      list.push(ev)
      grouped.set(ev.timestamp, list)
    }

    for (const [ts, group] of grouped.entries()) {
      const matchingCandle = currentBars.find((b) => b.timestamp === ts)
      if (!matchingCandle) {
        continue // Omit if matching candle is missing from currentBars
      }

      group.forEach((ev, idx) => {
        try {
          chart.createOverlay({
            name: 'corporateEventMarker',
            groupId: 'corporate-events',
            paneId: 'candle_pane',
            lock: true,
            id: `event-${ev.timestamp}-${ev.event_type}-${idx}`,
            points: [{ timestamp: ev.timestamp, value: matchingCandle.high }],
            extendData: {
              event: ev,
              stackIndex: idx,
              stackCount: group.length,
            },
            onMouseEnter: (data: unknown) => {
              const coords = (data as { coordinates?: { x: number; y: number }[] })?.coordinates
              if (coords && coords[0]) {
                onCorporateHoverCb?.(ev, { x: coords[0].x, y: coords[0].y })
              } else {
                onCorporateHoverCb?.(ev, null)
              }
            },
            onMouseLeave: () => {
              onCorporateHoverCb?.(null, null)
            },
          })
        } catch {
          // Safe fallback
        }
      })
    }
  }

  const set52WLevels = (params: {
    high: number | null
    low: number | null
    latestClose: number | null
    currency: 'USD' | 'THB'
    coverageCalendarDays: number
    visible: boolean
  }): void => {
    if (!chart) return
    const { high, low, latestClose, currency, coverageCalendarDays, visible } = params
    is52WVisible = visible
    current52WHigh = high
    current52WLow = low
    currentLatestClose = latestClose
    currentCurrency = currency
    currentCoverageDays = coverageCalendarDays

    try {
      chart.removeOverlay({ groupId: 'week52-levels' })
    } catch {
      // Ignore
    }

    if (!visible || high === null || low === null || high <= 0 || low <= 0) {
      return
    }

    const currSym = currency === 'THB' ? '฿' : '$'
    const prefix = coverageCalendarDays < 365 ? `Available (${coverageCalendarDays}d)` : '52W'

    let pctFromHighStr = ''
    if (latestClose !== null && high > 0) {
      const pct = ((latestClose - high) / high) * 100.0
      pctFromHighStr = ` (${pct >= 0 ? '+' : ''}${pct.toFixed(1)}% from High)`
    }

    const firstTs = currentBars[0]?.timestamp || 0

    try {
      // Instance 1: 52W High
      chart.createOverlay({
        name: 'system52WLevel',
        groupId: 'week52-levels',
        paneId: 'candle_pane',
        lock: true,
        id: '52w-high',
        points: [{ timestamp: firstTs, value: high }],
        extendData: {
          type: 'high',
          label: `${prefix} H: ${currSym}${high.toFixed(2)}`,
          color: '#ef4444',
        },
      })

      // Instance 2: 52W Low
      chart.createOverlay({
        name: 'system52WLevel',
        groupId: 'week52-levels',
        paneId: 'candle_pane',
        lock: true,
        id: '52w-low',
        points: [{ timestamp: firstTs, value: low }],
        extendData: {
          type: 'low',
          label: `${prefix} L: ${currSym}${low.toFixed(2)}${pctFromHighStr}`,
          color: '#22c55e',
        },
      })
    } catch {

      // Safe fallback
    }
  }

  const setValuationTargets = (params: {
    targets: ValuationTargetsDTO | null
    visible: boolean
  }): void => {
    if (!chart) return
    const { targets, visible } = params
    isValuationVisible = visible

    try {
      chart.removeOverlay({ groupId: 'dcf-levels' })
    } catch {
      // Ignore
    }

    if (!visible || !targets || targets.comparability_status !== 'comparable' || !targets.scenarios || targets.scenarios.length === 0 || currentBars.length === 0) {
      return
    }

    let evalTs = 0
    try {
      evalTs = new Date(targets.evaluated_at).getTime()
    } catch {
      evalTs = currentBars[0]?.timestamp || 0
    }

    let anchorCandle = currentBars.find((b) => b.timestamp >= evalTs)
    if (!anchorCandle) {
      anchorCandle = currentBars[0]
    }
    const anchorTs = anchorCandle?.timestamp || 0

    targets.scenarios.forEach((sc) => {
      const colorMap: Record<string, string> = {
        emerald: '#10b981',
        green: '#22c55e',
        rose: '#f43f5e',
        zinc: '#71717a',
      }
      const lineColor = colorMap[sc.color] || '#10b981'
      const upsideStr = sc.upside_pct !== null && sc.upside_pct !== undefined ? ` (${sc.upside_pct >= 0 ? '+' : ''}${sc.upside_pct.toFixed(1)}%)` : ''

      try {
        chart.createOverlay({
          name: 'valuationLevelRay',
          groupId: 'dcf-levels',
          paneId: 'candle_pane',
          lock: true,
          id: `dcf-${sc.scenario_name}`,
          points: [{ timestamp: anchorTs, value: sc.target_price }],
          extendData: {
            scenario_name: sc.scenario_name,
            label: `${sc.label}: ${sc.target_price.toFixed(2)}${upsideStr}`,
            color: lineColor,
          },
        })
      } catch {
        // Safe fallback
      }
    })
  }

  const setVWAP = (enabled: boolean, exchangeTz: string = 'America/New_York'): boolean => {
    if (!chart) return false
    isVwapEnabled = enabled

    if (enabled) {
      if (activeVwapId && activeVwapExchangeTz !== exchangeTz) {
        try {
          chart.removeIndicator({ paneId: 'candle_pane', id: activeVwapId })
        } catch {
          // Ignore
        }
        activeVwapId = null
      }

      if (!activeVwapId) {
        try {
          const res = chart.createIndicator(
            {
              name: 'SESSION_VWAP',
              paneId: 'candle_pane',
              extendData: { exchangeTz },
            },
            true // isStack: true on candle_pane
          )
          activeVwapId = typeof res === 'string' ? res : (Array.isArray(res) && typeof res[0] === 'string' ? res[0] : null)
          activeVwapExchangeTz = exchangeTz
        } catch {
          activeVwapId = null
          return false
        }
      }
      return !!activeVwapId
    } else {
      if (activeVwapId) {
        try {
          chart.removeIndicator({ paneId: 'candle_pane', id: activeVwapId })
        } catch {
          // Ignore
        }
        activeVwapId = null
        activeVwapExchangeTz = null
      }
      return true
    }
  }

  const setAnalystContext = (params: {
    ctx: AnalystContextDTO | null
    visible: boolean
    currentPrice: number | null
    currency: 'USD' | 'THB'
  }): void => {
    if (!chart) return
    const { ctx, visible, currentPrice, currency } = params
    currentAnalystContext = ctx
    isConsensusVisible = visible

    try {
      chart.removeOverlay({ groupId: 'analyst-target' })
    } catch {
      // Ignore
    }

    if (!visible || !ctx || ctx.data_status === 'unavailable' || !ctx.target_mean || ctx.target_mean <= 0 || currentBars.length === 0) {
      return
    }

    if (ctx.currency !== currency) {
      return
    }

    const anchorTs = currentBars[0]?.timestamp || 0
    const isStale = ctx.data_status === 'stale'
    const currSym = currency === 'THB' ? '฿' : '$'
    let upsideStr = ''
    if (currentPrice && currentPrice > 0) {
      const upPct = ((ctx.target_mean - currentPrice) / currentPrice) * 100
      upsideStr = ` (${upPct >= 0 ? '+' : ''}${upPct.toFixed(1)}%)`
    }
    const stalePrefix = isStale ? 'Street [Stale]: ' : 'Street: '
    const label = `${stalePrefix}${currSym}${ctx.target_mean.toFixed(2)}${upsideStr}`
    const color = isStale ? '#d97706' : '#f59e0b'

    try {
      chart.createOverlay({
        name: 'analystTargetRay',
        groupId: 'analyst-target',
        paneId: 'candle_pane',
        lock: true,
        id: `analyst-mean-target`,
        points: [{ timestamp: anchorTs, value: ctx.target_mean }],
        extendData: {
          label,
          color,
          isStale,
        },
      })
    } catch {
      // Safe fallback
    }
  }

  const setInsiderFilings = (
    filings: InsiderFilingDTO[],
    visible: boolean,
    onHover?: (payload: { filing: InsiderMarkerHoverDTO; pos: { x: number; y: number } } | null, isHover: boolean) => void
  ): void => {
    if (!chart) return
    isInsiderVisible = visible
    onInsiderHoverCb = onHover

    try {
      chart.removeOverlay({ groupId: 'insider-events' })
    } catch {
      // Ignore
    }

    if (!visible || !filings || filings.length === 0 || currentBars.length === 0) return

    // Group filings by mapped candle to avoid 20+ overlapping markers on the same date/bar
    const candleMap = new Map<
      number,
      {
        candle: KLineData
        filings: InsiderFilingDTO[]
        totalBuyShares: number
        totalSellShares: number
        hasClusterBuy: boolean
        latestFiling: InsiderFilingDTO
      }
    >()

    filings.forEach((filing) => {
      const candle =
        currentBars.find((b) => b.timestamp >= filing.timestamp) ||
        currentBars[currentBars.length - 1]
      if (!candle) return

      const totalBuyShares = filing.transactions
        .filter((t) => t.acquired_or_disposed === 'A')
        .reduce((sum, t) => sum + t.shares, 0)
      const totalSellShares = filing.transactions
        .filter((t) => t.acquired_or_disposed === 'D')
        .reduce((sum, t) => sum + t.shares, 0)

      const existing = candleMap.get(candle.timestamp)
      if (existing) {
        existing.filings.push(filing)
        existing.totalBuyShares += totalBuyShares
        existing.totalSellShares += totalSellShares
        if (filing.is_cluster_buy) existing.hasClusterBuy = true
        if (filing.timestamp > existing.latestFiling.timestamp) {
          existing.latestFiling = filing
        }
      } else {
        candleMap.set(candle.timestamp, {
          candle,
          filings: [filing],
          totalBuyShares,
          totalSellShares,
          hasClusterBuy: Boolean(filing.is_cluster_buy),
          latestFiling: filing,
        })
      }
    })

    candleMap.forEach((group) => {
      const { candle, filings: groupFilings, totalBuyShares, totalSellShares, hasClusterBuy, latestFiling } =
        group
      const count = groupFilings.length

      let actionType: 'buy' | 'sell' | 'cluster_buy' =
        totalBuyShares >= totalSellShares ? 'buy' : 'sell'
      if (hasClusterBuy) {
        actionType = 'cluster_buy'
      }

      const markerColor =
        actionType === 'cluster_buy' ? '#8b5cf6' : actionType === 'buy' ? '#10b981' : '#ef4444'

      let label = ''
      if (actionType === 'cluster_buy') {
        label = count > 1 ? `💎 Cluster (${count})` : '💎 Cluster'
      } else if (actionType === 'buy') {
        const kShares = Math.round(totalBuyShares / 1000)
        label = kShares > 0 ? (count > 1 ? `▲ +${kShares}k (${count})` : `▲ +${kShares}k`) : '▲ Buy'
      } else {
        const kShares = Math.round(totalSellShares / 1000)
        label = kShares > 0 ? (count > 1 ? `▼ -${kShares}k (${count})` : `▼ -${kShares}k`) : '▼ Sell'
      }

      const anchorY = actionType === 'sell' ? candle.high : candle.low

      const allFilersList =
        count > 1
          ? groupFilings.slice(0, 3).map((f) => ({
              name: f.reporting_owner_name || 'Insider',
              officer_title: f.officer_title || null,
              shares: f.transactions.reduce((s, t) => s + t.shares, 0),
            }))
          : null

      const hoverDTO: InsiderMarkerHoverDTO = {
        action_type: actionType,
        label,
        accession_number: latestFiling.accession_number,
        insider_name:
          count > 1
            ? `${latestFiling.reporting_owner_name || 'Insider'} (+${count - 1} filings)`
            : latestFiling.reporting_owner_name || 'Insider',
        officer_title: latestFiling.officer_title || null,
        shares: totalBuyShares || totalSellShares,
        price: latestFiling.transactions[0]?.price_per_share || 0,
        filing_url: latestFiling.filing_url,
        all_filers: allFilersList,
      }

      try {
        chart.createOverlay({
          name: 'secInsiderMarker',
          groupId: 'insider-events',
          paneId: 'candle_pane',
          id: `insider-${latestFiling.accession_number}`,
          points: [{ timestamp: candle.timestamp, value: anchorY }],
          extendData: {
            accession_number: latestFiling.accession_number,
            insider_name: hoverDTO.insider_name,
            action_type: actionType,
            label,
            color: markerColor,
            shares: totalBuyShares || totalSellShares,
            price: latestFiling.transactions[0]?.price_per_share || 0,
            filing_url: latestFiling.filing_url,
            officer_title: latestFiling.officer_title,
            anchorTimestamp: candle.timestamp,
            anchorValue: anchorY,
          },
          onClick: (event: any) => {
            const coords = event?.coordinates?.[0]
            const x = coords ? coords.x : (event?.x ?? 0)
            const y = coords ? coords.y : (event?.y ?? 0)
            onInsiderHoverCb?.({ filing: hoverDTO, pos: { x, y } }, false) // click = pin
          },
          onMouseEnter: (event: any) => {
            const coords = event?.coordinates?.[0]
            const x = coords ? coords.x : (event?.x ?? 0)
            const y = coords ? coords.y : (event?.y ?? 0)
            onInsiderHoverCb?.({ filing: hoverDTO, pos: { x, y } }, true) // hover = preview
          },
          onMouseLeave: () => {
            onInsiderHoverCb?.(null, true) // clear preview
          },
          onPressedMoving: (event: any) => {
            if (event?.overlay) {
              event.overlay.points = [{ timestamp: candle.timestamp, value: anchorY }]
            }
          },
          onPressedMoveEnd: (event: any) => {
            if (event?.overlay) {
              event.overlay.points = [{ timestamp: candle.timestamp, value: anchorY }]
            }
          },
        })
      } catch {
        // Safe fallback
      }
    })
  }

  const cancelActiveDrawing = (): void => {
    if (activeDrawingId && chart) {
      try {
        chart.removeOverlay({ id: activeDrawingId })
      } catch {
        // Ignore
      }
      activeDrawingId = null
    }
  }

  const enableDrawing = (tool: DrawingToolType): void => {
    if (!chart) return
    cancelActiveDrawing()
    if (!tool) {
      return
    }

    try {
      const res = chart.createOverlay({
        name: tool,
        groupId: 'user-drawings',
        onDrawEnd: () => {
          activeDrawingId = null
        },
      })
      activeDrawingId = typeof res === 'string' ? res : (Array.isArray(res) && typeof res[0] === 'string' ? res[0] : null)
    } catch {
      activeDrawingId = null
    }
  }

  const clearDrawings = (): void => {
    cancelActiveDrawing()
    if (chart) {
      try {
        chart.removeOverlay({ groupId: 'user-drawings' })
      } catch {
        // Ignore
      }
    }
  }

  if (chart) {
    // Setup Symbol and Period
    chart.setSymbol({
      ticker: symbol,
      pricePrecision: 2,
      volumePrecision: 0,
    })
    chart.setPeriod(period)

    // Setup initial indicator configuration
    if (effectiveInitialConfig.overlay !== 'NONE') {
      setOverlay(effectiveInitialConfig.overlay)
    }

    syncSubpanes({
      volume: effectiveInitialConfig.volume,
      momentum: effectiveInitialConfig.momentum,
      atr: effectiveInitialConfig.atr,
    })

    if (effectiveInitialConfig.vwap) {
      setVWAP(true)
    }

    // Setup Data Loader & Reset
    chart.setDataLoader({
      getBars: ({ callback }) => {
        callback(currentBars, { forward: false, backward: false })
      },
    })
    chart.resetData()

    if (effectiveInitialConfig.events && currentEvents.length > 0) {
      setEvents(currentEvents, true)
    }

    // Apply initial bar spacing & scroll
    applyBarSpacingAndScroll()
  }

  return {
    updateChart: (params) => {
      const {
        symbol: newSymbol,
        period: newPeriod,
        bars: newBars,
        displayStartTimestamp: newDisplayStart,
        events: newEvents,
        week52High: new52WHigh,
        week52Low: new52WLow,
        latestClose: newLatestClose,
        currency: newCurrency,
        coverageCalendarDays: newCoverageDays,
        valuationTargets: newValuationTargets,
        insiderFilings: newInsiderFilings,
        analystContext: newAnalystContext,
      } = params

      if (newSymbol && newSymbol !== symbol) {
        symbol = newSymbol
        chart?.setSymbol({ ticker: newSymbol })
        clearDrawings()
      }

      if (newPeriod) {
        period = newPeriod
        chart?.setPeriod(newPeriod)
      }

      if (newDisplayStart !== undefined) {
        currentDisplayStartTimestamp = newDisplayStart
      }

      if (newBars) {
        currentBars = newBars.map((b) => ({
          timestamp: b.timestamp,
          open: b.open,
          high: b.high,
          low: b.low,
          close: b.close,
          volume: b.volume,
        }))
        chart?.resetData()
        applyBarSpacingAndScroll()
      }

      if (newEvents !== undefined) {
        currentEvents = newEvents
        setEvents(currentEvents, isEventsVisible, onCorporateHoverCb)
      }

      if (
        new52WHigh !== undefined ||
        new52WLow !== undefined ||
        newLatestClose !== undefined ||
        newCurrency !== undefined ||
        newCoverageDays !== undefined
      ) {
        if (newLatestClose !== undefined) currentLatestClose = newLatestClose
        if (newCurrency !== undefined) currentCurrency = newCurrency
        set52WLevels({
          high: new52WHigh !== undefined ? new52WHigh : current52WHigh,
          low: new52WLow !== undefined ? new52WLow : current52WLow,
          latestClose: currentLatestClose,
          currency: currentCurrency,
          coverageCalendarDays:
            newCoverageDays !== undefined ? newCoverageDays : currentCoverageDays,
          visible: is52WVisible,
        })
        if (currentAnalystContext) {
          setAnalystContext({
            ctx: currentAnalystContext,
            visible: isConsensusVisible,
            currentPrice: currentLatestClose,
            currency: currentCurrency,
          })
        }
      }

      if (newValuationTargets !== undefined) {
        setValuationTargets({
          targets: newValuationTargets,
          visible: isValuationVisible,
        })
      }

      if (newAnalystContext !== undefined) {
        setAnalystContext({
          ctx: newAnalystContext,
          visible: isConsensusVisible,
          currentPrice: currentLatestClose,
          currency: currentCurrency,
        })
      }

      if (newInsiderFilings !== undefined) {
        setInsiderFilings(newInsiderFilings, isInsiderVisible, onInsiderHoverCb)
      }
    },

    setOverlay,
    setVolume: (enabled: boolean) => {
      syncSubpanes({ ...currentSubpaneState, volume: enabled })
      return true
    },
    setMomentum: (momentum: MomentumIndicatorType) => {
      syncSubpanes({ ...currentSubpaneState, momentum })
      return true
    },
    setATR: (enabled: boolean) => {
      syncSubpanes({ ...currentSubpaneState, atr: enabled })
      return true
    },
    setVWAP,

    setEvents,
    set52WLevels,
    setValuationTargets,
    setAnalystContext,
    setInsiderFilings,
    enableDrawing,
    cancelActiveDrawing,
    clearDrawings,

    getEffectiveConfig: () => ({
      overlay: activeOverlayType,
      volume: currentSubpaneState.volume,
      momentum: currentSubpaneState.momentum,
      atr: currentSubpaneState.atr,
      events: isEventsVisible,
      week52: is52WVisible,
      dcf: isValuationVisible,
      insider: isInsiderVisible,
      vwap: isVwapEnabled,
      consensus: isConsensusVisible,
    }),

    resize: () => {
      chart?.resize()
      applyBarSpacingAndScroll()
    },

    destroy: () => {
      dispose(container)
    },

    getChart: () => chart,
  }
}

