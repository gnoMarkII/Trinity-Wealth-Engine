import React, { useEffect, useState } from 'react'
import { api } from '../../api/client'
import type { PositionReviewResponseDTO } from '../../api/types'
import { PriceReturnDecompositionBar } from './PriceReturnDecompositionBar'
import { ContextTooltip } from './explainer'

export interface PositionReviewCardProps {
  ticker: string
  market?: string
  portfolioId?: string
  initialData?: PositionReviewResponseDTO | null
  className?: string
}

export const PositionReviewCard: React.FC<PositionReviewCardProps> = ({
  ticker,
  market = 'US',
  portfolioId = 'default',
  initialData = null,
  className = '',
}) => {
  const [data, setData] = useState<PositionReviewResponseDTO | null>(initialData)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const currencySymbol = market === 'TH' ? '฿' : '$'

  useEffect(() => {
    if (initialData) {
      setData(initialData)
      return
    }

    if (!ticker) {
      setData(null)
      return
    }

    let isMounted = true
    setLoading(true)
    setError(null)

    if (typeof api.getPositionReview !== 'function') {
      setLoading(false)
      return
    }

    api.getPositionReview(ticker, market, portfolioId)
      .then((res) => {
        if (isMounted) {
          setData(res)
          setLoading(false)
        }
      })
      .catch((err) => {
        if (isMounted) {
          setError(err?.message || 'Failed to load position review')
          setLoading(false)
        }
      })

    return () => {
      isMounted = false
    }
  }, [ticker, market, portfolioId, initialData])

  if (loading && !data) {
    return (
      <section
        id="position-review-section"
        data-testid="position-review-card-loading"
        className={`flow-panel rounded-2xl border border-edge/80 p-6 shadow-sm animate-pulse space-y-4 ${className}`}
      >
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-xl bg-zinc-200" />
          <div className="space-y-1.5 flex-1">
            <div className="h-4 w-48 bg-zinc-200 rounded" />
            <div className="h-3 w-80 bg-zinc-100 rounded" />
          </div>
        </div>
        <div className="h-28 bg-zinc-100/60 rounded-xl" />
      </section>
    )
  }

  if (error && !data) {
    return (
      <section
        id="position-review-section"
        data-testid="position-review-card-error"
        className={`flow-panel rounded-2xl border border-rose-200 bg-rose-50/30 p-6 text-xs text-rose-700 space-y-2 ${className}`}
      >
        <div className="flex items-center gap-2 font-bold text-sm text-rose-800">
          <span>⚠️</span>
          <span>ไม่สามารถโหลดข้อมูล Position Review ได้</span>
        </div>
        <p>{error}</p>
      </section>
    )
  }

  if (!data) return null

  const isHeld = data.holding_summary?.is_held ?? false

  return (
    <section
      id="position-review-section"
      data-testid="position-review-card"
      className={`flow-panel rounded-2xl border border-edge/80 p-6 shadow-sm space-y-6 animate-card-in ${className}`}
    >
      {/* 1. Header with Plain-Language Badge & ContextTooltip */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-edge/60 pb-4">
        <div>
          <div className="text-[11px] font-bold uppercase tracking-wider text-sky-700/80">
            Portfolio Context & Thesis Evaluation
          </div>
          <h3 className="text-lg font-bold text-zinc-900 flex items-center gap-2 mt-0.5">
            <span>💼 ทบทวนสถานะพอร์ตและสมมติฐานการลงทุน (Position Review)</span>
            <ContextTooltip
              title="Position Review & Decision Framework"
              whatIsIt="ระบบประเมินสถานะหุ้นโดยผสานข้อมูลพอร์ตจริง (ต้นทุน, กำไร/ขาดทุน, น้ำหนัก %) เข้ากับการแยกตัวประกอบราคา (Price Move Decomposition) และการเปรียบเทียบสมมติฐานวันซื้อ vs วันนี้ (Then vs Now)"
              howToRead="ตรวจดูว่าราคาที่ปรับตัวขึ้นมาเกิดจากผลประกอบการโตจริง (EPS Growth) หรือเกิดจากการเก็งกำไรดัน P/E (Multiple Expansion) และตอบคำถามเตือนสติเพื่อไม่ให้หลงรักหุ้น"
              impactOnMoney="ช่วยตัดสินใจว่าควรถือต่อ ทยอยขายทำกำไร หรือตัดขาดทุนอย่างมีวินัย ป้องกันการติดดอยยาวข้ามปี"
            />
          </h3>
          <p className="text-xs text-zinc-500 mt-0.5">
            ตรวจสุขภาพหุ้นในมือเราเอง — แยกแยะกำไรจริงออกจากกระแสเก็งกำไร พร้อมคำถามสะกิดสติไม่ให้ยึดติดราคาต้นทุน
          </p>
        </div>

        {/* Action Stance Badge */}
        <div className="flex items-center gap-2">
          {data.action_stance && (
            <span
              className={`px-3 py-1 rounded-full text-xs font-bold uppercase tracking-wider border ${
                data.action_stance.includes('ACCUMULATE') || data.action_stance.includes('BUY')
                  ? 'bg-emerald-50 text-emerald-700 border-emerald-200 shadow-2xs'
                  : data.action_stance.includes('TRIM') || data.action_stance.includes('EXIT')
                  ? 'bg-rose-50 text-rose-700 border-rose-200 shadow-2xs'
                  : 'bg-sky-50 text-sky-700 border-sky-200 shadow-2xs'
              }`}
            >
              {data.action_stance.replace(/_/g, ' ')}
            </span>
          )}
        </div>
      </div>

      {/* 2. Holding Context Banner: Held in Portfolio vs Watchlist Simulation */}
      {isHeld ? (
        <div className="p-4 rounded-xl bg-surface border border-edge/80 shadow-2xs">
          <div className="text-[11px] font-bold text-zinc-500 uppercase tracking-wider mb-2 flex items-center gap-1.5">
            <span className="flex h-2 w-2 relative">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75" />
              <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500" />
            </span>
            <span>สถานะการถือครองจริงในพอร์ตของคุณ (Active Holding)</span>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 text-xs">
            <div className="p-2.5 rounded-lg bg-surface-strong/40 border border-edge/40">
              <span className="text-[10px] font-medium text-zinc-400 block uppercase">จำนวนที่ถือครอง</span>
              <span className="font-mono text-base font-bold text-zinc-900 mt-0.5 block">
                {data.holding_summary.units.toLocaleString()} หุ้น
              </span>
            </div>

            <div className="p-2.5 rounded-lg bg-surface-strong/40 border border-edge/40">
              <span className="text-[10px] font-medium text-zinc-400 block uppercase">ต้นทุนเฉลี่ย (Avg Cost)</span>
              <span className="font-mono text-base font-bold text-zinc-900 mt-0.5 block">
                {data.holding_summary.avg_cost != null
                  ? `${currencySymbol}${data.holding_summary.avg_cost.toFixed(2)}`
                  : 'N/A'}
              </span>
            </div>

            <div className="p-2.5 rounded-lg bg-surface-strong/40 border border-edge/40">
              <span className="text-[10px] font-medium text-zinc-400 block uppercase">กำไร/ขาดทุนทางบัญชี (P&amp;L)</span>
              <span
                className={`font-mono text-base font-bold mt-0.5 block ${
                  (data.holding_summary.unrealized_pnl_pct ?? 0) >= 0 ? 'text-emerald-600' : 'text-rose-600'
                }`}
              >
                {data.holding_summary.unrealized_pnl_pct != null
                  ? `${data.holding_summary.unrealized_pnl_pct >= 0 ? '+' : ''}${data.holding_summary.unrealized_pnl_pct.toFixed(2)}%`
                  : 'N/A'}
              </span>
            </div>

            <div className="p-2.5 rounded-lg bg-surface-strong/40 border border-edge/40">
              <span className="text-[10px] font-medium text-zinc-400 block uppercase">สัดส่วนในพอร์ต (Weight)</span>
              <span className="font-mono text-base font-bold text-sky-700 mt-0.5 block">
                {data.holding_summary.current_weight_pct.toFixed(2)}%
              </span>
            </div>
          </div>
        </div>
      ) : (
        <div className="p-4 rounded-xl bg-sky-50/60 border border-sky-200/80 text-xs text-sky-950 flex items-start gap-3">
          <span className="text-base shrink-0 mt-0.5">💡</span>
          <div className="space-y-1">
            <span className="font-bold block text-sky-900">
              โหมดวิเคราะห์จำลอง (Simulation &amp; Watchlist Mode) — คุณยังไม่มีหุ้น {data.ticker} ในพอร์ต
            </span>
            <p className="text-sky-800 text-[11px] leading-relaxed">
              การวิเคราะห์ด้านล่างจะจำลองเปรียบเทียบจากราคาในอดีต (1 ปี และ 3 ปี) เพื่อให้คุณเห็นว่าราคาปัจจุบันมีแต้มต่อเพียงพอในการเปิดสถานะใหม่หรือไม่ หากจะเข้าซื้อ แนะนำควบคุมขนาดไม้ (Position Sizing) ไม่เกิน 5-10% ของพอร์ตโดยรวม
            </p>
          </div>
        </div>
      )}

      {/* 3. Price Move Decomposition (1Y & 3Y) */}
      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span className="text-sm">📊</span>
            <h4 className="text-sm font-bold text-zinc-900">
              1. ถอดรหัสการเคลื่อนที่ของราคา (Price Move Decomposition)
            </h4>
            <ContextTooltip
              title="การถอดรหัสผลตอบแทนราคา (Price Move Decomposition)"
              whatIsIt="การแยกย่อยผลตอบแทนราคาหุ้นออกเป็น 2 ปัจจัยหลัก: 1) กำไรบริษัทโตจริง (EPS Growth) และ 2) ตลาดให้มูลค่าเพิ่มขึ้นจากกระแสเก็งกำไร (P/E Expansion)"
              howToRead="หุ้นที่แข็งแกร่งควรขึ้นจาก EPS Growth เป็นหลัก (Fundamental Compounder) ส่วนหุ้นที่ขึ้นจาก P/E Expansion สูงแต่กำไรไม่โต ถือเป็นสัญญาณเก็งกำไรที่มีความเสี่ยงสูง"
              impactOnMoney="ช่วยให้เราไม่หลงดีใจกับราคาหุ้นที่วิ่งขึ้นมา แต่สามารถแยกแยะได้ว่านี่คือของจริงหรือเพียงฟองสบู่ชั่วคราว"
            />
          </div>
          <span className="text-[11px] text-zinc-400">Horizon: 1Y &amp; 3Y Historical</span>
        </div>

        {data.decompositions && data.decompositions.length > 0 ? (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {data.decompositions.map((decomp, idx) => (
              <PriceReturnDecompositionBar key={idx} decomposition={decomp} />
            ))}
          </div>
        ) : (
          <p className="text-xs text-zinc-400 p-4 rounded-xl bg-surface border border-edge/60 text-center">
            ไม่มีข้อมูล Price Move Decomposition สำหรับหุ้นตัวนี้
          </p>
        )}
      </div>

      {/* 4. Historical Thesis Reconstruction (Then vs Now) & Anti-Anchoring */}
      {data.then_vs_now && (
        <div className="space-y-3 pt-2 border-t border-edge/60">
          <div className="flex items-center gap-2">
            <span className="text-sm">🔄</span>
            <h4 className="text-sm font-bold text-zinc-900">
              2. ทบทวนสมมติฐานการลงทุน (Then vs Now Evolution)
            </h4>
            <ContextTooltip
              title="การเปรียบเทียบสมมติฐานวันซื้อ vs วันนี้ (Then vs Now)"
              whatIsIt="การทบทวนว่าเหตุผลและตัวเลขที่เราเคยใช้ตัดสินใจในอดีต (Then) ยังคงเป็นจริงในปัจจุบัน (Now) หรือไม่"
              howToRead="หากเหตุผลหลักที่เคยซื้อเปลี่ยนไปแล้ว เช่น อัตราการเติบโตชะลอตัวลง หรือ P/E สูงเกินกว่าสมมติฐานเดิมมาก แปลว่าเราต้องทบทวนการถือครองใหม่"
              impactOnMoney="ช่วยแก้ปัญหาอคติไม่กล้าขายขาดทุน (Loss Aversion) และช่วยให้เราตัดสินใจบนความเป็นจริงในปัจจุบัน"
            />
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* Then Card */}
            <div className="p-4 rounded-xl bg-surface border border-edge/70 space-y-2">
              <div className="flex items-center justify-between text-xs">
                <span className="font-bold text-zinc-600 uppercase tracking-wider text-[10px]">
                  สมมติฐานเดิมในอดีต (Then)
                </span>
                <span className="font-mono text-zinc-400 text-[11px]">{data.then_vs_now.then_date}</span>
              </div>
              <div className="flex items-baseline gap-3 text-xs">
                <span className="text-zinc-500">ราคา: <strong>{currencySymbol}{data.then_vs_now.then_price.toFixed(2)}</strong></span>
                <span className="text-zinc-500">P/E: <strong>{data.then_vs_now.then_pe != null ? `${data.then_vs_now.then_pe.toFixed(1)}x` : 'N/A'}</strong></span>
              </div>
              <p className="text-xs text-zinc-700 leading-relaxed pt-1 border-t border-edge/40">
                {data.then_vs_now.what_was_true_then}
              </p>
            </div>

            {/* Now Card */}
            <div className="p-4 rounded-xl bg-surface border border-sky-200/80 space-y-2">
              <div className="flex items-center justify-between text-xs">
                <span className="font-bold text-sky-800 uppercase tracking-wider text-[10px]">
                  ความเป็นจริงในปัจจุบัน (Now)
                </span>
                <span className="font-mono text-zinc-400 text-[11px]">{data.then_vs_now.now_date}</span>
              </div>
              <div className="flex items-baseline gap-3 text-xs">
                <span className="text-zinc-500">ราคา: <strong>{currencySymbol}{data.then_vs_now.now_price.toFixed(2)}</strong></span>
                <span className="text-zinc-500">P/E: <strong>{data.then_vs_now.now_pe != null ? `${data.then_vs_now.now_pe.toFixed(1)}x` : 'N/A'}</strong></span>
              </div>
              <p className="text-xs text-zinc-700 leading-relaxed pt-1 border-t border-edge/40">
                {data.then_vs_now.what_is_true_now}
              </p>
            </div>
          </div>

          {/* Anti-Anchoring Reflection Question */}
          <div className="p-3.5 rounded-xl bg-amber-50/80 border border-amber-200/90 text-xs text-amber-950 flex items-start gap-2.5 shadow-2xs">
            <span className="text-base shrink-0 mt-0.5">💡</span>
            <div className="space-y-0.5">
              <span className="font-bold block text-amber-900">
                คำถามสะกิดสติป้องกันการหลงรักหุ้น (Anti-Anchoring Reflection):
              </span>
              <p className="text-amber-800 text-[11px] leading-relaxed">
                {data.then_vs_now.thesis_reflection_question ||
                  'หากวันนี้คุณไม่มีหุ้นตัวนี้อยู่ในพอร์ตเลย คุณจะยอมควักเงินสดซื้อที่ราคานี้ ณ ตอนนี้ไหม? หากคำตอบคือไม่ แปลว่าคุณอาจกำลังถือหุ้นอยู่ด้วยความเสียดาย ไม่ใช่ด้วยเหตุผลทางธุรกิจ'}
              </p>
            </div>
          </div>
        </div>
      )}

      {/* 5. Portfolio-Aware Decision Triggers */}
      {data.triggers && data.triggers.length > 0 && (
        <div className="space-y-3 pt-2 border-t border-edge/60">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <span className="text-sm">🎯</span>
              <h4 className="text-sm font-bold text-zinc-900">
                3. ปัจจัยชี้ขาดเพื่อตัดสินใจ (Portfolio-Aware Decision Triggers)
              </h4>
              <ContextTooltip
                title="ปัจจัยชี้ขาดเพื่อการตัดสินใจ (Decision Triggers)"
                whatIsIt="เกณฑ์เตือนล่วงหน้าที่อิงกับพอร์ตของเรา เช่น หากราคาหรือผลประกอบการแตะเงื่อนไขนี้ ควร Take Profit, Trim หรือ Rebalance"
                howToRead="ดูว่ามี Trigger ใดที่แตะสถานะ Active (สีแดง/ส้ม) ซึ่งเตือนว่าต้องลงมือปรับพอร์ตตามแผน"
                impactOnMoney="ช่วยลดการใช้อารมณ์ในการเทรด ทำให้การซื้อขายเป็นไปตามวินัยที่วางไว้ล่วงหน้า"
              />
            </div>
            <span className="text-[11px] text-zinc-400">{data.triggers.length} เงื่อนไขเฝ้าระวัง</span>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            {data.triggers.map((trig) => (
              <div
                key={trig.trigger_id}
                className="p-3 rounded-xl bg-surface border border-edge/70 flex items-start justify-between gap-3 text-xs"
              >
                <div className="space-y-1">
                  <div className="font-bold text-zinc-900">{trig.title}</div>
                  <div className="text-[11px] text-zinc-500 leading-snug">{trig.condition_text}</div>
                </div>
                <span className="shrink-0 px-2 py-0.5 rounded text-[10px] font-bold uppercase bg-zinc-100 text-zinc-700 border border-zinc-200">
                  {trig.case_impact.replace(/_/g, ' ')}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </section>
  )
}

export default PositionReviewCard

