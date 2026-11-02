import { useEffect, useState, useCallback } from 'react'
import { useNavigate, useSearchParams } from 'react-router-dom'
import { api, ApiError } from '../api/client'
import type {
  MacroDashboardDTO,
  TreasuryYieldCurveDTO,
  CommodityVolSnapshotDTO,
  AuctionDemandSnapshotDTO,
  UsNationalDebtDTO,
  ThaiFundFlowDTO,
  ThaiRetailGoldDTO,
  MarketValuationDTO,
  MarketBreadthDTO,
  ThaiYieldCurveDTO,
  ThaiPublicDebtDTO,
  CryptoMacroLiquidityDTO,
} from '../api/types'
import MacroReferenceDrawer from '../components/MacroReferenceDrawer'
import { MacroGlossaryDrawer } from '../components/macro/explainer/MacroGlossaryDrawer'
import Toast from '../components/common/Toast'
import { MacroRegionTabs, type MacroRegionTab } from '../components/macro/cockpit/MacroRegionTabs'
import { AiBriefingCard } from '../components/macro/cockpit/AiBriefingCard'
import { AiAnalysisSection } from '../components/macro/cockpit/AiAnalysisSection'
import { UsMacroSection } from '../components/macro/cockpit/UsMacroSection'
import { ThailandMacroSection } from '../components/macro/cockpit/ThailandMacroSection'
import { CrossBorderSection } from '../components/macro/cockpit/CrossBorderSection'
import { SourceProvenanceBadge } from '../components/macro/cockpit/SourceProvenanceBadge'
import { MacroNotebookLMExport } from '../components/macro/cockpit/MacroNotebookLMExport'

export default function Macro() {
  const navigate = useNavigate()
  const [searchParams, setSearchParams] = useSearchParams()

  // 1. Tab state synced with URL query (?tab=ai | us | th | cross-border)
  const tabParam = searchParams.get('tab') as MacroRegionTab | null
  const activeTab: MacroRegionTab =
    tabParam && ['ai', 'us', 'th', 'cross-border'].includes(tabParam) ? tabParam : 'ai'

  const handleTabChange = (nextTab: MacroRegionTab) => {
    setSearchParams({ tab: nextTab }, { replace: true })
  }

  // 2. Independent AI Dashboard state (isolated from market provider failures)
  const [aiData, setAiData] = useState<MacroDashboardDTO | null>(null)
  const [aiLoading, setAiLoading] = useState(true)
  const [aiError, setAiError] = useState<string | null>(null)

  // 3. Independent Market Provider states
  const [yieldCurve, setYieldCurve] = useState<TreasuryYieldCurveDTO | null>(null)
  const [loadingYield, setLoadingYield] = useState(false)
  const [errorYield, setErrorYield] = useState<string | null>(null)

  const [financialStress, setFinancialStress] = useState<any | null>(null)
  const [loadingStress, setLoadingStress] = useState(false)
  const [errorStress, setErrorStress] = useState<string | null>(null)

  const [metalsCot, setMetalsCot] = useState<any | null>(null)
  const [globalPolicyRates, setGlobalPolicyRates] = useState<any | null>(null)
  const [loadingPolicy, setLoadingPolicy] = useState(false)
  const [errorPolicy, setErrorPolicy] = useState<string | null>(null)

  const [commodityVol, setCommodityVol] = useState<CommodityVolSnapshotDTO[] | null>(null)
  const [auctionDemandNote, setAuctionDemandNote] = useState<AuctionDemandSnapshotDTO | null>(null)
  const [auctionDemandBill, setAuctionDemandBill] = useState<AuctionDemandSnapshotDTO | null>(null)
  const [nationalDebt, setNationalDebt] = useState<UsNationalDebtDTO[] | null>(null)
  const [cryptoLiquidity, setCryptoLiquidity] = useState<CryptoMacroLiquidityDTO | null>(null)
  const [loadingCrypto, setLoadingCrypto] = useState(false)
  const [marketIssues, setMarketIssues] = useState<string[]>([])

  // Thai Market Observables
  const [flow, setFlow] = useState<ThaiFundFlowDTO | null>(null)
  const [gold, setGold] = useState<ThaiRetailGoldDTO | null>(null)
  const [valuation, setValuation] = useState<MarketValuationDTO | null>(null)
  const [breadth, setBreadth] = useState<MarketBreadthDTO | null>(null)
  const [thaiYieldCurve, setThaiYieldCurve] = useState<ThaiYieldCurveDTO | null>(null)
  const [thaiPublicDebt, setThaiPublicDebt] = useState<ThaiPublicDebtDTO | null>(null)
  const [loadingThai, setLoadingThai] = useState(false)
  const [errorThai, setErrorThai] = useState<string | null>(null)

  // Metadata & UI states
  const [lastMarketFetched, setLastMarketFetched] = useState<string | null>(null)
  const [isRefreshingMarket, setIsRefreshingMarket] = useState(false)
  const [isRefDrawerOpen, setIsRefDrawerOpen] = useState(false)
  const [isGlossaryOpen, setIsGlossaryOpen] = useState(false)
  const [updating, setUpdating] = useState(false)
  const [toastState, setToastState] = useState<{
    message: string
    actionLabel?: string
    onAction?: () => void
    type?: 'info' | 'success' | 'error'
  } | null>(null)

  // Fetch AI Report independently
  const fetchAiDashboard = useCallback(async () => {
    setAiLoading(true)
    setAiError(null)
    try {
      const res = await api.getMacroDashboard()
      setAiData(res)
    } catch (e) {
      setAiError(e instanceof ApiError ? e.message : 'ยังไม่มีบทวิเคราะห์ภาพรวมเศรษฐกิจในคลัง')
    } finally {
      setAiLoading(false)
    }
  }, [])

  // Quick Refresh function for market observables without modifying AI evaluated_at
  const fetchMarketObservables = useCallback(async () => {
    setIsRefreshingMarket(true)
    setLoadingYield(true)
    setLoadingStress(true)
    setLoadingPolicy(true)
    setLoadingThai(true)
    setLoadingCrypto(true)

    const pYc = Promise.resolve(api.getTreasuryYieldCurve?.())
      .then((val) => {
        setYieldCurve(val)
        setErrorYield(null)
        return { status: 'fulfilled' as const, value: val }
      })
      .catch((err) => {
        setErrorYield('ไม่สามารถเชื่อมต่อ US Treasury Yield Curve ได้')
        return { status: 'rejected' as const, reason: err }
      })
      .finally(() => setLoadingYield(false))

    const pFsi = Promise.resolve(api.getFinancialStress?.())
      .then((val) => {
        setFinancialStress(val)
        setErrorStress(null)
        return { status: 'fulfilled' as const, value: val }
      })
      .catch((err) => {
        setErrorStress('ไม่สามารถเชื่อมต่อ OFR Financial Stress Index ได้')
        return { status: 'rejected' as const, reason: err }
      })
      .finally(() => setLoadingStress(false))

    const pCot = Promise.resolve(api.getMetalsCot?.('gold'))
      .then((val) => {
        setMetalsCot(val)
        return { status: 'fulfilled' as const, value: val }
      })
      .catch((err) => {
        setMetalsCot(null)
        return { status: 'rejected' as const, reason: err }
      })

    const pBis = Promise.resolve(api.getGlobalPolicyRates?.())
      .then((val) => {
        setGlobalPolicyRates(val)
        setErrorPolicy(null)
        return { status: 'fulfilled' as const, value: val }
      })
      .catch((err) => {
        setErrorPolicy('ไม่สามารถเชื่อมต่ออัตราดอกเบี้ยนโยบาย BIS ได้')
        return { status: 'rejected' as const, reason: err }
      })
      .finally(() => setLoadingPolicy(false))

    const pVol = Promise.resolve(api.getCommodityVolatility?.())
      .then((val) => {
        setCommodityVol(val)
        return { status: 'fulfilled' as const, value: val }
      })
      .catch((err) => {
        setCommodityVol(null)
        return { status: 'rejected' as const, reason: err }
      })

    const pAucNote = Promise.resolve(api.getTreasuryAuctionDemand?.('Note', '10-Year'))
      .then((val) => {
        setAuctionDemandNote(val)
        return { status: 'fulfilled' as const, value: val }
      })
      .catch((err) => {
        setAuctionDemandNote(null)
        return { status: 'rejected' as const, reason: err }
      })

    const pAucBill = Promise.resolve(api.getTreasuryAuctionDemand?.('Bill', '13-Week'))
      .then((val) => {
        setAuctionDemandBill(val)
        return { status: 'fulfilled' as const, value: val }
      })
      .catch((err) => {
        setAuctionDemandBill(null)
        return { status: 'rejected' as const, reason: err }
      })

    const pDebt = Promise.resolve(api.getNationalDebt?.(30))
      .then((val) => {
        setNationalDebt(val)
        return { status: 'fulfilled' as const, value: val }
      })
      .catch((err) => {
        setNationalDebt(null)
        return { status: 'rejected' as const, reason: err }
      })

    const pCryptoLiq = Promise.resolve(api.getCryptoMacroLiquidity?.())
      .then((val) => {
        setCryptoLiquidity(val)
        return { status: 'fulfilled' as const, value: val }
      })
      .catch((err) => {
        setCryptoLiquidity(null)
        return { status: 'rejected' as const, reason: err }
      })
      .finally(() => setLoadingCrypto(false))

    // Thai Observables
    const pThFlow = Promise.resolve(api.getThaiInvestorFlow?.('SET'))
      .then((val) => {
        setFlow(val)
        return { status: 'fulfilled' as const, value: val }
      })
      .catch((err) => {
        setFlow(null)
        return { status: 'rejected' as const, reason: err }
      })

    const pThGold = Promise.resolve(api.getThaiRetailGold?.())
      .then((val) => {
        setGold(val)
        return { status: 'fulfilled' as const, value: val }
      })
      .catch((err) => {
        setGold(null)
        return { status: 'rejected' as const, reason: err }
      })

    const pThVal = Promise.resolve(api.getThaiMarketValuation?.('SET'))
      .then((val) => {
        setValuation(val)
        return { status: 'fulfilled' as const, value: val }
      })
      .catch((err) => {
        setValuation(null)
        return { status: 'rejected' as const, reason: err }
      })

    const pThBreadth = Promise.resolve(api.getThaiMarketBreadth?.('SET'))
      .then((val) => {
        setBreadth(val)
        return { status: 'fulfilled' as const, value: val }
      })
      .catch((err) => {
        setBreadth(null)
        return { status: 'rejected' as const, reason: err }
      })

    const pThYield = Promise.resolve(api.getThaiYieldCurve?.())
      .then((val) => {
        setThaiYieldCurve(val)
        return { status: 'fulfilled' as const, value: val }
      })
      .catch((err) => {
        setThaiYieldCurve(null)
        return { status: 'rejected' as const, reason: err }
      })

    const pThDebt = Promise.resolve(api.getThaiPublicDebt?.())
      .then((val) => {
        setThaiPublicDebt(val)
        return { status: 'fulfilled' as const, value: val }
      })
      .catch((err) => {
        setThaiPublicDebt(null)
        return { status: 'rejected' as const, reason: err }
      })

    Promise.allSettled([pThFlow, pThGold, pThVal, pThBreadth, pThYield, pThDebt]).then(([f, g, v, b, y, d]) => {
      let thaiErrCount = 0
      if (f.status === 'fulfilled' && f.value.status === 'rejected') thaiErrCount++
      if (g.status === 'fulfilled' && g.value.status === 'rejected') thaiErrCount++
      if (v.status === 'fulfilled' && v.value.status === 'rejected') thaiErrCount++
      if (b.status === 'fulfilled' && b.value.status === 'rejected') thaiErrCount++
      if (y.status === 'fulfilled' && y.value.status === 'rejected') thaiErrCount++
      if (d.status === 'fulfilled' && d.value.status === 'rejected') thaiErrCount++
      if (thaiErrCount > 0) {
        setErrorThai(`มีข้อมูลตลาดไทย ${thaiErrCount} รายการที่ไม่สามารถดึงได้`)
      } else {
        setErrorThai(null)
      }
      setLoadingThai(false)
    })

    const [yc, fsi, cot, bis, vol, aucNote, aucBill, debt, thFlow, thGold, thVal, thBreadth, thYield, thDebt, cryptoLiq] =
      await Promise.all([pYc, pFsi, pCot, pBis, pVol, pAucNote, pAucBill, pDebt, pThFlow, pThGold, pThVal, pThBreadth, pThYield, pThDebt, pCryptoLiq])

    const providerResults = [yc, fsi, cot, bis, vol, aucNote, aucBill, debt, thFlow, thGold, thVal, thBreadth, thYield, thDebt, cryptoLiq]
    const providerNames = ['US Yield Curve', 'OFR Stress', 'Gold COT', 'Policy Rates', 'Commodity Volatility', '10Y Auction', '13W Auction', 'US Debt', 'SET Flow', 'GTA Gold', 'SET Valuation', 'SET Breadth', 'ThaiBMA Yield Curve', 'Thai Public Debt', 'Crypto Liquidity']
    setMarketIssues(providerResults.flatMap((result, index) => {
      if (result.status === 'rejected') return [`${providerNames[index]}: ดึงข้อมูลไม่สำเร็จ`]
      const values = Array.isArray(result.value) ? result.value : [result.value]
      return values.some((value) => value?.is_stale)
        ? [`${providerNames[index]}: ข้อมูลบางส่วนล้าสมัย`]
        : []
    }))

    setLastMarketFetched(
      new Date().toLocaleTimeString('th-TH', { hour: '2-digit', minute: '2-digit', second: '2-digit' })
    )
    setIsRefreshingMarket(false)
  }, [])

  // Initial load
  useEffect(() => {
    fetchAiDashboard()
    fetchMarketObservables()
  }, [fetchAiDashboard, fetchMarketObservables])

  // Fallback to 'us' tab if initial AI load completed without report and user did not explicitly choose ?tab
  useEffect(() => {
    if (!aiLoading && !aiData && !tabParam) {
      setSearchParams({ tab: 'us' }, { replace: true })
    }
  }, [aiLoading, aiData, tabParam, setSearchParams])

  // Trigger AI Job via Kanban
  const handleUpdateMacro = async () => {
    if (updating) return
    setUpdating(true)
    const cardTitle = 'วิเคราะห์ภาวะเศรษฐกิจมหภาค (Macro Analysis)'
    const instruction =
      'วิเคราะห์ภาวะเศรษฐกิจมหภาค (Macro Intelligence & Regime Analysis) ล่าสุดพร้อมประเมิน Asset Allocation'

    let cardId: string | undefined
    try {
      const { card } = await api.createKanbanCard(cardTitle, 'manager', instruction, 'both')
      cardId = card.card_id
    } catch (err) {
      console.error('Failed to create macro kanban card:', err)
      setToastState({
        message: err instanceof ApiError ? err.message : 'เกิดข้อผิดพลาดในการสร้างการ์ดวิเคราะห์เศรษฐกิจ',
        type: 'error',
      })
      setUpdating(false)
      return
    }

    try {
      await api.dispatchJob(instruction, cardId, 'manager', 'both')
      setToastState({
        message: 'สั่งงานวิเคราะห์ภาวะเศรษฐกิจมหภาคเรียบร้อย',
        actionLabel: 'ดูสถานะใน Kanban',
        onAction: () => navigate('/kanban'),
        type: 'success',
      })
    } catch (err) {
      console.error('Failed to dispatch macro job:', err)
      setToastState({
        message:
          (err instanceof ApiError ? err.message : 'เกิดข้อผิดพลาดในการเริ่มงานวิเคราะห์') +
          ' — สร้างการ์ดไว้ที่ Backlog แล้ว กด dispatch เองในการ์ดนั้นได้',
        type: 'error',
      })
    } finally {
      setUpdating(false)
    }
  }

  return (
    <div className="animate-page-in space-y-6 pb-12">
      {/* 1. Topmost Anchor: AI Executive Briefing (ส่วนที่ AI วิเคราะห์อยู่บนสุดทั้งหมด) */}
      <AiBriefingCard
        aiData={aiData}
        aiLoading={aiLoading}
        aiError={aiError}
        onUpdateMacro={handleUpdateMacro}
        updating={updating}
        onNavigateToAiTab={() => handleTabChange('ai')}
      />

      {/* 2. Cockpit Slim Toolbar & Controls (แถบควบคุมกะทัดรัด แนวนอนบรรทัดเดียว ไม่กินพื้นที่หน้าจอ) */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2.5 rounded-xl border border-sky-100/80 bg-white/95 px-3.5 py-2 shadow-2xs">
        <div className="flex items-center gap-2">
          <span className="text-base">🧭</span>
          <h1 className="text-xs sm:text-sm font-bold tracking-tight text-zinc-900">
            Macro Cockpit (ห้องควบคุมภาวะเศรษฐกิจมหภาค)
          </h1>
        </div>

        {/* Hidden from visual toolbar per user request, preserved for accessibility & screen readers */}
        <div className="sr-only">
          <SourceProvenanceBadge origin="provider" compact />
          <SourceProvenanceBadge origin="deterministic" compact />
          <SourceProvenanceBadge origin="ai" compact />
          <SourceProvenanceBadge origin="external" compact />
        </div>

        {/* Action Buttons Toolbar */}
        <div className="flex flex-wrap items-center gap-1.5 shrink-0">
          <MacroNotebookLMExport
            onNotify={(msg, type) =>
              setToastState({
                message: msg,
                type,
              })
            }
          />

          <button
            type="button"
            onClick={fetchMarketObservables}
            disabled={isRefreshingMarket}
            className="flex items-center gap-1.5 rounded-lg border border-sky-200 bg-white px-2.5 py-1.5 text-xs font-semibold text-sky-700 shadow-2xs transition-all hover:bg-sky-50 disabled:opacity-50"
            title="ดึงข้อมูลตลาดสดล่าสุด (ไม่รัน AI)"
          >
            <span>{isRefreshingMarket ? '⏳' : '🔄'}</span>
            <span>{isRefreshingMarket ? 'กำลังดึง...' : 'รีเฟรชข้อมูลตลาด'}</span>
          </button>

          <button
            type="button"
            onClick={() => setIsRefDrawerOpen(true)}
            className="flex items-center gap-1.5 rounded-lg border border-edge bg-panel px-2.5 py-1.5 text-xs font-semibold text-zinc-800 shadow-2xs transition-all hover:bg-surface-strong hover:shadow"
            title="เปิดเอกสารอ้างอิงและตัวชี้วัด"
          >
            <span>📚</span>
            <span>References</span>
          </button>

          <button
            type="button"
            onClick={() => setIsGlossaryOpen(true)}
            className="flex items-center gap-1.5 rounded-lg border border-indigo-200 bg-indigo-50/70 px-2.5 py-1.5 text-xs font-semibold text-indigo-900 shadow-2xs transition-all hover:bg-indigo-100 hover:shadow"
            title="เปิดคู่มือคำศัพท์เศรษฐกิจและการลงทุนฉบับเข้าใจง่าย"
          >
            <span>📖</span>
            <span>คู่มือศัพท์ Macro</span>
          </button>

          {lastMarketFetched && (
            <span
              className="text-[10px] text-zinc-400 font-mono hidden md:inline ml-1"
              title={`ดึงตลาดล่าสุด: ${lastMarketFetched}`}
            >
              {lastMarketFetched}
            </span>
          )}
        </div>
      </div>

      {marketIssues.length > 0 && (
        <div role="status" className="rounded-xl border border-amber-200 bg-amber-50 p-4 text-xs text-amber-900">
          ข้อมูลตลาดที่ต้องตรวจสอบ: {marketIssues.join(' • ')}
        </div>
      )}

      {/* 3. Region & Analysis Navigation (4 Tabs) */}
      <MacroRegionTabs activeTab={activeTab} onChange={handleTabChange} />

      {/* 4. Tab Contents with Error Isolation */}
      {activeTab === 'ai' && (
        <AiAnalysisSection
          aiData={aiData}
          aiLoading={aiLoading}
          aiError={aiError}
        />
      )}

      {activeTab === 'us' && (
        <UsMacroSection
          yieldCurve={yieldCurve}
          financialStress={financialStress}
          metalsCot={metalsCot}
          commodityVol={commodityVol}
          auctionDemandNote={auctionDemandNote}
          auctionDemandBill={auctionDemandBill}
          nationalDebt={nationalDebt}
          loadingYield={loadingYield}
          loadingStress={loadingStress}
          errorYield={errorYield}
          errorStress={errorStress}
          cryptoLiquidity={cryptoLiquidity}
          loadingCrypto={loadingCrypto}
        />
      )}

      {activeTab === 'th' && (
        <ThailandMacroSection
          flow={flow}
          gold={gold}
          valuation={valuation}
          breadth={breadth}
          thaiYieldCurve={thaiYieldCurve}
          thaiPublicDebt={thaiPublicDebt}
          aiData={aiData}
          globalPolicyRates={globalPolicyRates}
          loading={loadingThai}
          error={errorThai}
        />
      )}

      {activeTab === 'cross-border' && (
        <CrossBorderSection
          globalPolicyRates={globalPolicyRates}
          yieldCurve={yieldCurve}
          flow={flow}
          loadingPolicy={loadingPolicy}
          errorPolicy={errorPolicy}
        />
      )}

      {/* Floating Right Side Tab Button */}
      {aiData && (
        <button
          type="button"
          onClick={() => setIsRefDrawerOpen(true)}
          className="fixed right-0 top-1/3 z-40 flex items-center gap-2 rounded-l-xl border-y border-l border-edge bg-zinc-900 px-3 py-3.5 text-xs font-semibold text-white shadow-xl transition-all hover:bg-zinc-800 hover:pl-4"
          title="เปิดแท็บแหล่งอ้างอิงและตัวชี้วัด (Reference Drawer)"
        >
          <span className="text-sm">📚</span>
          <span className="writing-vertical tracking-wide">References</span>
        </button>
      )}

      {/* Slide-over Reference Drawer Panel */}
      {aiData && (
        <MacroReferenceDrawer
          data={aiData}
          isOpen={isRefDrawerOpen}
          onClose={() => setIsRefDrawerOpen(false)}
        />
      )}

      {/* Slide-over Macro Glossary Drawer Panel */}
      <MacroGlossaryDrawer
        isOpen={isGlossaryOpen}
        onClose={() => setIsGlossaryOpen(false)}
      />

      {toastState && (
        <Toast
          message={toastState.message}
          type={toastState.type}
          actionLabel={toastState.actionLabel}
          onAction={toastState.onAction}
          onClose={() => setToastState(null)}
        />
      )}
    </div>
  )
}
