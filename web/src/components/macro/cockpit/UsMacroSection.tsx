import React, { useState } from 'react'
import type {
  TreasuryYieldCurveDTO,
  CommodityVolSnapshotDTO,
  AuctionDemandSnapshotDTO,
  UsNationalDebtDTO,
  MacroDashboardDTO,
  CryptoMacroLiquidityDTO,
} from '../../../api/types'
import { CryptoLiquidityCard } from '../CryptoLiquidityCard'
import { YieldCurveChart } from './YieldCurveChart'
import { OfrStressBar } from './OfrStressBar'
import { MetalsCotCard } from '../MetalsCotCard'
import { CommodityVolCard } from '../CommodityVolCard'
import { TreasuryAuctionDemandCard } from '../TreasuryAuctionDemandCard'
import { StackedAreaChart } from '../../charts/StackedAreaChart'
import { SectorRotationDashboard } from './SectorRotationDashboard'

interface UsMacroSectionProps {
  yieldCurve: TreasuryYieldCurveDTO | null
  financialStress: any | null
  metalsCot: any | null
  commodityVol: CommodityVolSnapshotDTO[] | null
  auctionDemandNote: AuctionDemandSnapshotDTO | null
  auctionDemandBill: AuctionDemandSnapshotDTO | null
  nationalDebt: UsNationalDebtDTO[] | null
  cryptoLiquidity?: CryptoMacroLiquidityDTO | null
  loadingCrypto?: boolean
  loadingYield?: boolean
  loadingStress?: boolean
  errorYield?: string | null
  errorStress?: string | null
  aiData?: MacroDashboardDTO | null
  aiLoading?: boolean
  aiError?: string | null
}

export const UsMacroSection: React.FC<UsMacroSectionProps> = ({
  yieldCurve,
  financialStress,
  metalsCot,
  commodityVol,
  auctionDemandNote,
  auctionDemandBill,
  nationalDebt,
  cryptoLiquidity = null,
  loadingCrypto = false,
  loadingYield = false,
  loadingStress = false,
  errorYield = null,
  errorStress = null,
}) => {
  const [showGlobalContext, setShowGlobalContext] = useState(false)

  return (
    <div className="space-y-6">
      {/* 1. S&P 500 Sector Rotation Analysis (RRG & Relative Momentum) */}
      <SectorRotationDashboard />

      {/* Crypto Macro Liquidity Radar (Level 1 Proxy) */}
      <CryptoLiquidityCard
        data={cryptoLiquidity}
        loading={loadingCrypto}
      />

      {/* 2. US Primary Market & Yield Observables (Verified Data) */}
      <div className="space-y-3">
        <div className="flex items-center justify-between border-b border-sky-100/70 pb-2">
          <div>
            <h2 className="text-base font-bold text-zinc-900 tracking-tight flex items-center gap-2">
              <span>ข้อมูลตลาดและอัตราผลตอบแทนสหรัฐฯ (US Observables)</span>
            </h2>
            <p className="text-xs text-zinc-500">
              ข้อมูลปฐมภูมิทางการจาก US Treasury และ OFR ใช้ประเมินเส้นอัตราผลตอบแทนและความตึงตัวของระบบการเงิน
            </p>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-5 items-start">
          <YieldCurveChart data={yieldCurve} loading={loadingYield} error={errorYield} />
          <OfrStressBar
            fsiValue={financialStress?.fsi_value}
            asOfDate={financialStress?.as_of_date}
            regime={financialStress?.regime}
            categories={financialStress?.categories}
            loading={loadingStress}
            error={errorStress}
          />
        </div>
      </div>

      {/* 3. Global Context with US Impact (Collapsible Accordion) */}
      <div className="rounded-2xl border border-slate-200/90 bg-white/80 p-4 shadow-sm">
        <button
          type="button"
          onClick={() => setShowGlobalContext(!showGlobalContext)}
          className="flex items-center justify-between w-full text-left font-semibold text-sm text-zinc-800 hover:text-sky-700 transition-colors"
        >
          <div className="flex items-center gap-2">
            <span>🌐</span>
            <span>บริบทโลกที่มีผลต่อสหรัฐฯ (Commodity Vol, Gold COT, Debt & Auctions)</span>
            <span className="text-[11px] font-normal text-zinc-400">
              {showGlobalContext ? '(คลิกเพื่อย่อ)' : '(คลิกเพื่อขยายดูรายละเอียด 4 แผง)'}
            </span>
          </div>
          <span className="font-mono text-xs text-zinc-400">{showGlobalContext ? '▲' : '▼'}</span>
        </button>

        {showGlobalContext && (
          <div className="mt-4 space-y-5 border-t border-slate-100 pt-4">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-5 items-start">
              {/* CFTC Gold COT */}
              {metalsCot ? (
                <MetalsCotCard
                  commodity={metalsCot.commodity}
                  commodityCode={metalsCot.commodity_code}
                  asOfDate={metalsCot.as_of_date}
                  publishedAt={metalsCot.published_at}
                  openInterest={metalsCot.open_interest}
                  netManagedMoney={metalsCot.net_managed_money}
                  percentile52w={metalsCot.percentile_52w}
                  managedMoney={metalsCot.managed_money}
                  swapDealers={metalsCot.swap_dealers}
                  producerMerchant={metalsCot.producer_merchant}
                />
              ) : (
                <div className="rounded-xl border border-slate-100 bg-slate-50 p-4 text-xs text-zinc-500">
                  ไม่มีข้อมูล CFTC Gold COT
                </div>
              )}

              {/* Commodity Volatility */}
              {Array.isArray(commodityVol) && commodityVol.length > 0 && (
                <CommodityVolCard indices={commodityVol} />
              )}
            </div>

            {/* Treasury Auction Demand */}
            {(auctionDemandNote || auctionDemandBill) && (
              <TreasuryAuctionDemandCard
                demandNote={auctionDemandNote}
                demandBill={auctionDemandBill}
              />
            )}

            {/* US National Debt to the Penny */}
            {Array.isArray(nationalDebt) && nationalDebt.length > 0 && (
              <StackedAreaChart
                data={[...nationalDebt]
                  .filter((d) => d && d.record_date)
                  .sort((a, b) => (a.record_date || '').localeCompare(b.record_date || ''))
                  .map((d) => ({
                    date: d.record_date,
                    values_by_category: {
                      public: d.debt_held_by_public_usd ? d.debt_held_by_public_usd / 1e12 : 0,
                      intragov: d.intragovernmental_holdings_usd ? d.intragovernmental_holdings_usd / 1e12 : 0,
                    },
                  }))}
                categories={[
                  { key: 'public', label: 'Debt Held by Public', color: '#0284c7' },
                  { key: 'intragov', label: 'Intragovernmental Holdings', color: '#8b5cf6' },
                ]}
                title="US National Debt: Debt to the Penny (Held by Public vs Intragovernmental)"
                subtitle="ข้อมูลหนี้สาธารณะสหรัฐฯ รายวันจาก US Treasury Fiscal Data ไม่นับรวม Maturity Composition ตามนิยามข้อมูล"
                unit="$T"
                height={260}
                defaultMode="absolute"
              />
            )}
          </div>
        )}
      </div>
    </div>
  )
}
