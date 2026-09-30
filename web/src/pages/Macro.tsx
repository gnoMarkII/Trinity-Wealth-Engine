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
} from '../api/types'
import MacroReferenceDrawer from '../components/MacroReferenceDrawer'
import Toast from '../components/common/Toast'
import { MacroRegionTabs, type MacroRegionTab } from '../components/macro/cockpit/MacroRegionTabs'
import { UsMacroSection } from '../components/macro/cockpit/UsMacroSection'
import { ThailandMacroSection } from '../components/macro/cockpit/ThailandMacroSection'
import { CrossBorderSection } from '../components/macro/cockpit/CrossBorderSection'
import { SourceProvenanceBadge } from '../components/macro/cockpit/SourceProvenanceBadge'

export default function Macro() {
  const navigate = useNavigate()
  const [searchParams, setSearchParams] = useSearchParams()

  // 1. Tab state synced with URL query (?tab=us | th | cross-border)
  const tabParam = searchParams.get('tab') as MacroRegionTab | null
  const activeTab: MacroRegionTab =
    tabParam && ['us', 'th', 'cross-border'].includes(tabParam) ? tabParam : 'us'

  const handleTabChange = (nextTab: MacroRegionTab) => {
    setSearchParams({ tab: nextTab }, { replace: true })
  }

  // 2. Independent AI Dashboard state (isolated from market provider failures)
  const [aiData, setAiData] = useState<MacroDashboardDTO | null>(null)
  const [aiLoading, setAiLoading] = useState(false)
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

  // Thai Market Observables
  const [flow, setFlow] = useState<ThaiFundFlowDTO | null>(null)
  const [gold, setGold] = useState<ThaiRetailGoldDTO | null>(null)
  const [valuation, setValuation] = useState<MarketValuationDTO | null>(null)
  const [breadth, setBreadth] = useState<MarketBreadthDTO | null>(null)
  const [loadingThai, setLoadingThai] = useState(false)
  const [errorThai, setErrorThai] = useState<string | null>(null)

  // Metadata & UI states
  const [lastMarketFetched, setLastMarketFetched] = useState<string | null>(null)
  const [isRefreshingMarket, setIsRefreshingMarket] = useState(false)
  const [isRefDrawerOpen, setIsRefDrawerOpen] = useState(false)
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

    // Parallel calls with individual settled status
    const [yc, fsi, cot, bis, vol, aucNote, aucBill, debt, thFlow, thGold, thVal, thBreadth] =
      await Promise.allSettled([
        api.getTreasuryYieldCurve?.(),
        api.getFinancialStress?.(),
        api.getMetalsCot?.('gold'),
        api.getGlobalPolicyRates?.(),
        api.getCommodityVolatility?.(),
        api.getTreasuryAuctionDemand?.('Note', '10-Year'),
        api.getTreasuryAuctionDemand?.('Bill', '13-Week'),
        api.getNationalDebt?.(30),
        api.getThaiInvestorFlow?.('SET'),
        api.getThaiRetailGold?.(),
        api.getThaiMarketValuation?.('SET'),
        api.getThaiMarketBreadth?.('SET'),
      ])

    if (yc.status === 'fulfilled') {
      setYieldCurve(yc.value)
      setErrorYield(null)
    } else {
      setErrorYield('ไม่สามารถเชื่อมต่อ US Treasury Yield Curve ได้')
    }
    setLoadingYield(false)

    if (fsi.status === 'fulfilled') {
      setFinancialStress(fsi.value)
      setErrorStress(null)
    } else {
      setErrorStress('ไม่สามารถเชื่อมต่อ OFR Financial Stress Index ได้')
    }
    setLoadingStress(false)

    if (cot.status === 'fulfilled') setMetalsCot(cot.value)
    if (vol.status === 'fulfilled') setCommodityVol(vol.value)
    if (aucNote.status === 'fulfilled') setAuctionDemandNote(aucNote.value)
    if (aucBill.status === 'fulfilled') setAuctionDemandBill(aucBill.value)
    if (debt.status === 'fulfilled') setNationalDebt(debt.value)

    if (bis.status === 'fulfilled') {
      setGlobalPolicyRates(bis.value)
      setErrorPolicy(null)
    } else {
      setErrorPolicy('ไม่สามารถเชื่อมต่ออัตราดอกเบี้ยนโยบาย BIS ได้')
    }
    setLoadingPolicy(false)

    // Thai Observables
    let thaiErrCount = 0
    if (thFlow.status === 'fulfilled') setFlow(thFlow.value)
    else thaiErrCount++
    if (thGold.status === 'fulfilled') setGold(thGold.value)
    else thaiErrCount++
    if (thVal.status === 'fulfilled') setValuation(thVal.value)
    else thaiErrCount++
    if (thBreadth.status === 'fulfilled') setBreadth(thBreadth.value)
    else thaiErrCount++

    if (thaiErrCount > 0) {
      setErrorThai(`มีข้อมูลตลาดไทย ${thaiErrCount} รายการที่ไม่สามารถดึงได้`)
    } else {
      setErrorThai(null)
    }
    setLoadingThai(false)

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
      {/* 1. Cockpit Header: Title, Controls & Provenance Legend */}
      <div className="rounded-2xl border border-sky-100 bg-gradient-to-br from-white/95 via-sky-50/40 to-slate-50/60 p-6 shadow-sm">
        <div className="flex flex-col gap-4 md:flex-row md:items-start md:justify-between">
          <div className="space-y-1.5">
            <div className="flex items-center gap-2">
              <span className="text-2xl">🧭</span>
              <h1 className="text-xl font-bold tracking-tight text-zinc-900">
                Macro Cockpit (ห้องควบคุมภาวะเศรษฐกิจมหภาค)
              </h1>
            </div>
            <p className="text-xs text-zinc-500 max-w-2xl leading-relaxed">
              แยกแยะบริบทอย่างชัดเจนระหว่างสหรัฐฯ ไทย และความเชื่อมโยงข้ามพรมแดน พร้อมระบุแหล่งกำเนิดและวันสังเกตการณ์จริงในทุกข้อมูล
            </p>

            {/* Provenance Meaning Bar */}
            <div className="pt-2 flex flex-wrap items-center gap-1.5 text-[11px]">
              <span className="text-zinc-400 font-semibold mr-1">สัญลักษณ์แหล่งที่มา:</span>
              <SourceProvenanceBadge origin="provider" compact />
              <SourceProvenanceBadge origin="deterministic" compact />
              <SourceProvenanceBadge origin="ai" compact />
              <SourceProvenanceBadge origin="external" compact />
            </div>
          </div>

          {/* Action Buttons */}
          <div className="flex flex-col items-end gap-2 shrink-0">
            <div className="flex flex-wrap items-center gap-2">
              <button
                type="button"
                onClick={fetchMarketObservables}
                disabled={isRefreshingMarket}
                className="flex items-center gap-1.5 rounded-xl border border-sky-200 bg-white px-3.5 py-2 text-xs font-semibold text-sky-700 shadow-xs transition-all hover:bg-sky-50 disabled:opacity-50"
                title="ดึงข้อมูลตลาดสดล่าสุด (ไม่รัน AI)"
              >
                <span>{isRefreshingMarket ? '⏳' : '🔄'}</span>
                <span>{isRefreshingMarket ? 'กำลังดึง...' : 'รีเฟรชข้อมูลตลาด'}</span>
              </button>

              <button
                type="button"
                onClick={handleUpdateMacro}
                disabled={updating}
                className="flex items-center gap-1.5 rounded-xl border border-sky-200 bg-sky-50 px-3.5 py-2 text-xs font-semibold text-sky-700 shadow-xs transition-all hover:bg-sky-100 disabled:opacity-50"
                title="สร้างการ์ดใหม่ใน Kanban และเริ่มวิเคราะห์ภาวะเศรษฐกิจมหภาคด้วย AI"
              >
                <span>{updating ? '⏳' : '🧠'}</span>
                <span>{updating ? 'กำลังสั่งงาน...' : 'อัปเดตบทวิเคราะห์'}</span>
              </button>

              <button
                type="button"
                onClick={() => setIsRefDrawerOpen(true)}
                className="flex items-center gap-1.5 rounded-xl border border-edge bg-panel px-3.5 py-2 text-xs font-semibold text-zinc-800 shadow-xs transition-all hover:bg-surface-strong hover:shadow"
                title="เปิดเอกสารอ้างอิงและตัวชี้วัด"
              >
                <span>📚</span>
                <span>References</span>
              </button>
            </div>

            {lastMarketFetched && (
              <span className="text-[10px] text-zinc-400 font-mono">
                ดึงตลาดล่าสุด: {lastMarketFetched}
              </span>
            )}
          </div>
        </div>
      </div>

      {/* 2. Region Navigation (3 Tabs) */}
      <MacroRegionTabs activeTab={activeTab} onChange={handleTabChange} />

      {/* 3. Tab Contents with Error Isolation */}
      {activeTab === 'us' && (
        <UsMacroSection
          yieldCurve={yieldCurve}
          financialStress={financialStress}
          metalsCot={metalsCot}
          commodityVol={commodityVol}
          auctionDemandNote={auctionDemandNote}
          auctionDemandBill={auctionDemandBill}
          nationalDebt={nationalDebt}
          aiData={aiData}
          aiLoading={aiLoading}
          aiError={aiError}
          loadingYield={loadingYield}
          loadingStress={loadingStress}
          errorYield={errorYield}
          errorStress={errorStress}
        />
      )}

      {activeTab === 'th' && (
        <ThailandMacroSection
          flow={flow}
          gold={gold}
          valuation={valuation}
          breadth={breadth}
          aiData={aiData}
          aiLoading={aiLoading}
          aiError={aiError}
          loading={loadingThai}
          error={errorThai}
        />
      )}

      {activeTab === 'cross-border' && (
        <CrossBorderSection
          globalPolicyRates={globalPolicyRates}
          yieldCurve={yieldCurve}
          flow={flow}
          aiData={aiData}
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
