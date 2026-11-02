import type { EquitySentimentContextDTO, SmartMoneyFlagsDTO, TacticalSetupDTO } from '../../api/types'
import TacticalStageCycleBar from './TacticalStageCycleBar'
import PriceActionDepthLadder from './PriceActionDepthLadder'
import SmartMoneyFlowGauge from './SmartMoneyFlowGauge'
import { ContextTooltip } from './explainer'

export interface TacticalSetupProps extends Partial<TacticalSetupDTO> {
  current_price?: number | null
}

export interface InsiderConvictionProps {
  status?: string
  data_status?: string
  open_market_p_count_90d?: number
  open_market_p_value_usd?: number
  open_market_s_count_90d?: number
  open_market_s_value_usd?: number
  c_suite_p_count?: number
  insider_buy_range_min?: number
  insider_buy_range_max?: number
}

interface Props {
  tactical?: TacticalSetupProps | null
  insider?: InsiderConvictionProps | null
  smartMoney?: SmartMoneyFlagsDTO | null
  sentiment?: EquitySentimentContextDTO | null
  currency?: string
}

export default function TacticalFlowMatrixCard({
  tactical,
  insider,
  smartMoney,
  sentiment,
  currency = '$',
}: Props) {
  // If neither tactical nor flow data exists
  if (!tactical && !insider && !smartMoney && !sentiment) return null

  const stage = tactical?.price_stage || 'UNKNOWN'
  const isStage2 = stage.includes('STAGE_2') || stage.includes('MARKUP')
  const isStage4 = stage.includes('STAGE_4') || stage.includes('MARKDOWN')

  return (
    <div className="flow-panel rounded-2xl border border-edge/80 p-6 shadow-sm transition-all animate-card-in space-y-6">
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-edge/60 pb-4">
        <div>
          <div className="text-[11px] font-bold uppercase tracking-wider text-sky-700/80">
            Execution & Capital Flow Matrix
          </div>
          <h3 className="text-lg font-bold text-zinc-900 flex items-center gap-2 mt-0.5">
            <span>📊 Tactical Blueprint & Flow</span>
            <ContextTooltip
              title="แผนภูมิกลยุทธ์และกระแสเงินทุน (Tactical Blueprint & Flow)"
              whatIsIt="กรอบการตัดสินใจเข้าซื้อขายหุ้นโดยผสานวัฏจักรราคา (Cycle), โซนราคาที่เหมาะสม (Buy Zone), จุดยอมแพ้ (Stop Loss) และกระแสเงินทุนของสถาบัน"
              howToRead="ควรเลือกหุ้นที่อยู่ใน Stage 2 (ขาขึ้น), รอให้ราคาย่อตัวเข้า Buy Zone ที่มี Risk/Reward คุ้มค่า (> 2:1) และตรวจสอบว่าไม่มีแรงขายผิดปกติจากผู้บริหาร"
              impactOnMoney="ช่วยลดความเสี่ยงจากการซื้อไล่ราคาบนยอดดอย และสร้างวินัยในการเทรดที่มีแบบแผนชัดเจน"
            />
          </h3>
        </div>

        {/* Stage Badge */}
        <div className="flex items-center gap-2">
          <span
            className={`px-3 py-1 rounded-full text-xs font-bold uppercase tracking-wider border ${
              isStage2
                ? 'bg-emerald-50 text-emerald-700 border-emerald-200 shadow-xs'
                : isStage4
                ? 'bg-rose-50 text-rose-700 border-rose-200 shadow-xs'
                : 'bg-sky-50 text-sky-700 border-sky-200 shadow-xs'
            }`}
          >
            {stage.replace(/_/g, ' ')}
          </span>
        </div>
      </div>

      {/* Section 1: Tactical Trading Blueprint */}
      {tactical && (
        <div className="space-y-4">
          {/* Infographic: 4-Stage Market Cycle Bar */}
          <TacticalStageCycleBar currentStage={tactical.price_stage} />

          {/* Beginner Trading Discipline Tip */}
          <div className="flex items-start gap-2.5 p-3 rounded-xl bg-amber-50/70 border border-amber-200/70 text-xs text-amber-900">
            <span className="text-sm shrink-0 mt-0.5">🛡️</span>
            <div>
              <span className="font-bold block text-amber-950">วินัยการเข้าเทรดสำหรับมือใหม่ (Tactical Discipline):</span>
              <span className="text-amber-800 text-[11px] leading-relaxed">
                เข้าซื้อเมื่อราคาอยู่ใน <strong>Optimal Buy Zone</strong> เท่านั้น และต้องกำหนด <strong>Stop Loss</strong> ล่วงหน้าเสมอ เพื่อจำกัดการขาดทุนไม่ให้กระทบพอร์ตหลัก
              </span>
            </div>
          </div>

          <div className="flex items-center justify-between text-xs font-semibold text-zinc-500 uppercase tracking-wider pt-1">
            <span>🎯 Tactical Setup ({tactical.horizon_timeframe || '1-3M'})</span>
            {tactical.atr_14 != null && (
              <span className="text-zinc-500 font-normal">ATR(14): {currency}{tactical.atr_14.toFixed(2)}</span>
            )}
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
            {/* Buy Zone */}
            <div className="rounded-xl border border-emerald-200/80 bg-emerald-50/50 p-3">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-1">
                  <span className="text-emerald-800 font-semibold block text-[10px] uppercase">
                    Optimal Buy Zone
                  </span>
                  <ContextTooltip
                    title="โซนราคาที่เหมาะสมในการซื้อ (Optimal Buy Zone)"
                    whatIsIt="กรอบราคาที่เกิดจากการพักตัวหรือทดสอบแนวรับสำคัญ ซึ่งให้แต้มต่อในการเข้าซื้อสูง"
                    howToRead="หากราคาอยู่ในกรอบนี้ถือเป็นจังหวะสะสมที่ดี หากราคาเกินกรอบไปแล้วไม่ควรไล่ซื้อ ควรรอย่อตัวรอบถัดไป"
                    impactOnMoney="ช่วยป้องกันการติดดอยจากการซื้อหุ้นตอนที่ราคาวิ่งไปไกลแล้ว"
                  />
                </div>
                {tactical.is_in_buy_zone && (
                  <span className="text-[9px] px-1 py-0.2 rounded bg-emerald-200/80 text-emerald-900 font-bold">
                    IN ZONE
                  </span>
                )}
              </div>
              <span className="font-mono font-bold text-emerald-950 text-base sm:text-lg block mt-0.5">
                {tactical.buy_zone_min != null && tactical.buy_zone_max != null
                  ? `${currency}${tactical.buy_zone_min.toFixed(2)} - ${currency}${tactical.buy_zone_max.toFixed(2)}`
                  : 'N/A'}
              </span>
            </div>

            {/* Invalidation Stop Loss */}
            <div className="rounded-xl border border-rose-200/80 bg-rose-50/50 p-3">
              <div className="flex items-center gap-1">
                <span className="text-rose-800 font-semibold block text-[10px] uppercase">
                  Stop Loss (จุดยอมแพ้)
                </span>
                <ContextTooltip
                  title="จุดตัดขาดทุน (Invalidation Stop Loss)"
                  whatIsIt="ระดับราคาที่หากหลุดลงไป จะทำให้สมมติฐานทางเทคนิคหรือแนวรับหลักเสียหาย"
                  howToRead="หากราคาปิดต่ำกว่าระดับนี้ ต้องขายตัดขาดทุนตามแผนทันที ห้ามถัวเฉลี่ยขาลงเด็ดขาด"
                  impactOnMoney="ช่วยจำกัดความเสียหายไม่ให้ลุกลามจนกลายเป็นผู้ถือหุ้นติดดอยระยะยาว"
                />
              </div>
              <span className="font-mono font-bold text-rose-950 text-base sm:text-lg block mt-0.5">
                {tactical.invalidation_stop_loss != null
                  ? `${currency}${tactical.invalidation_stop_loss.toFixed(2)}`
                  : 'N/A'}
              </span>
            </div>

            {/* Tactical Target */}
            <div className="rounded-xl border border-sky-200/80 bg-sky-50/50 p-3">
              <div className="flex items-center gap-1">
                <span className="text-sky-800 font-semibold block text-[10px] uppercase">
                  Tactical Target
                </span>
                <ContextTooltip
                  title="ราคาเป้าหมายเชิงกลยุทธ์ (Tactical Target Price)"
                  whatIsIt="เป้าหมายทำกำไรตามแนวต้านสำคัญหรือสัดส่วนผลตอบแทนที่คำนวณไว้"
                  howToRead="เมื่อราคาขึ้นมาถึงระดับนี้ แนะนำให้พิจารณาแบ่งขายทำกำไร (Take Profit) หรือเลื่อนจุด Stop Loss ขึ้นมาเพื่อล็อคกำไร"
                  impactOnMoney="ช่วยให้สามารถเก็บกำไรเข้าพอร์ตได้อย่างเป็นรูปธรรม ไม่ปล่อยให้กำไรกลายเป็นขาดทุน"
                />
              </div>
              <span className="font-mono font-bold text-sky-950 text-base sm:text-lg block mt-0.5">
                {tactical.tactical_target_price != null
                  ? `${currency}${tactical.tactical_target_price.toFixed(2)}`
                  : 'N/A'}
              </span>
            </div>

            {/* Risk / Reward Ratio */}
            <div className="rounded-xl border border-edge/80 bg-surface p-3">
              <div className="flex items-center gap-1">
                <span className="text-zinc-500 font-semibold block text-[10px] uppercase">
                  R:R ณ Analysis Price
                </span>
                <ContextTooltip
                  title="อัตราส่วนผลตอบแทนต่อความเสี่ยง (Risk / Reward Ratio)"
                  whatIsIt="เปรียบเทียบกำไรที่มีโอกาสได้ กับเงินต้นที่มีโอกาสขาดทุนเมื่อแตะ Stop Loss"
                  howToRead="อัตราส่วนควรมากกว่า 2.0 : 1 ขึ้นไป (กำไรมีโอกาสได้มากกว่าขาดทุน 2 เท่าขึ้นไป) หากต่ำกว่า 1.0 : 1 ถือว่าไม่คุ้มเสี่ยง"
                  impactOnMoney="ช่วยให้พอร์ตเติบโตได้ในระยะยาว แม้จะทายถูกเพียง 50% ของจำนวนครั้งก็ตาม"
                />
              </div>
              <span className="font-mono font-bold text-zinc-900 text-base sm:text-lg block mt-0.5">
                {tactical.current_rr_ratio != null
                  ? `${tactical.current_rr_ratio.toFixed(2)} : 1`
                  : (tactical.pullback_entry_status === 'at_or_above_target'
                      ? 'N/A (เหนือ Target)'
                      : (tactical.tactical_risk_reward_ratio != null ? `${tactical.tactical_risk_reward_ratio.toFixed(2)} : 1` : 'N/A'))}
              </span>
              {tactical.buy_zone_rr_min != null && tactical.buy_zone_rr_max != null && (
                <span className="text-[10px] text-zinc-400 block mt-0.5 truncate">
                  ในโซน: {tactical.buy_zone_rr_min.toFixed(1)}-{tactical.buy_zone_rr_max.toFixed(1)}:1
                </span>
              )}
            </div>
          </div>

          {/* Infographic: Support / Resistance Depth Ladder & Execution Channels */}
          <PriceActionDepthLadder
            currentPrice={tactical.current_price ?? 0}
            keyResistance={tactical.key_resistance_level}
            keySupport={tactical.key_support_level}
            buyZoneMin={tactical.buy_zone_min}
            buyZoneMax={tactical.buy_zone_max}
            stopLoss={tactical.invalidation_stop_loss}
            tacticalTarget={tactical.tactical_target_price}
            currency={currency}
          />

          {/* Breakout Blueprint Strip */}
          {tactical.breakout_trigger_price != null && (
            <div className="rounded-xl border border-indigo-200/80 bg-indigo-50/40 p-3.5 text-xs space-y-2">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <div className="flex items-center gap-2">
                  <span className="font-bold text-indigo-900 uppercase text-[10px]">🚀 Breakout Setup:</span>
                  <span className="text-indigo-800">
                    Trigger <strong>{currency}{tactical.breakout_trigger_price.toFixed(2)}</strong> | Target <strong>{currency}{tactical.breakout_target_price?.toFixed(2)}</strong> | Stop <strong>{currency}{tactical.breakout_stop_loss?.toFixed(2)}</strong>
                  </span>
                </div>
                {tactical.breakout_entry_status && (
                  <span
                    className={`px-2 py-0.5 rounded text-[10px] font-bold uppercase tracking-wider border ${
                      tactical.breakout_entry_status === 'eligible'
                        ? 'bg-emerald-100 text-emerald-800 border-emerald-300'
                        : tactical.breakout_entry_status === 'chased'
                        ? 'bg-amber-100 text-amber-800 border-amber-300'
                        : tactical.breakout_entry_status === 'pre_trigger'
                        ? 'bg-sky-100 text-sky-800 border-sky-300'
                        : 'bg-zinc-100 text-zinc-700 border-zinc-300'
                    }`}
                  >
                    {tactical.breakout_entry_status === 'pre_trigger'
                      ? 'Pre-trigger (ยังเข้า Breakout ไม่ได้)'
                      : tactical.breakout_entry_status === 'eligible'
                      ? 'Breakout Eligible'
                      : tactical.breakout_entry_status === 'chased'
                      ? 'Chased / Low R:R'
                      : 'Expired'}
                  </span>
                )}
              </div>
              <div className="flex flex-wrap items-center justify-between gap-2 pt-1 border-t border-indigo-100 text-[11px] text-indigo-900">
                <span>
                  Planned R:R (ณ Trigger): <strong>{tactical.breakout_planned_rr != null ? `${tactical.breakout_planned_rr.toFixed(2)} : 1` : 'N/A'}</strong>
                </span>
                <span>
                  Current R:R (ราคาปัจจุบัน): <strong>{tactical.breakout_current_rr != null ? `${tactical.breakout_current_rr.toFixed(2)} : 1` : '— (Pre-trigger)'}</strong>
                </span>
                {tactical.breakout_volume_ratio != null && (
                  <span className="text-zinc-600">
                    Volume: <strong>{tactical.breakout_volume_ratio.toFixed(1)}x 20D</strong> {tactical.breakout_volume_confirmed ? '✅' : '⚠️'}
                  </span>
                )}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Section 2: SEC Form 4 & Smart Money Flow (Infographic 3) */}
      {(insider || smartMoney) && (
        <SmartMoneyFlowGauge
          smartMoney={smartMoney}
          insider={insider}
          currency={currency}
        />
      )}

      {/* Section 3: Sentiment & Tail Risks Warning */}
      {sentiment && (
        <div className="space-y-3 pt-1 border-t border-edge/60 text-xs">
          {sentiment.key_themes && sentiment.key_themes.length > 0 && (
            <div>
              <span className="text-[11px] font-bold text-zinc-500 uppercase tracking-wider block mb-1.5">
                Key Themes
              </span>
              <div className="flex flex-wrap gap-1.5">
                {sentiment.key_themes.map((t, i) => (
                  <span key={i} className="rounded-full border border-sky-100 bg-surface-strong px-2.5 py-0.5 text-xs font-medium text-zinc-700">
                    {t}
                  </span>
                ))}
              </div>
            </div>
          )}

          {sentiment.tail_risks && sentiment.tail_risks.length > 0 && (
            <div className="rounded-xl border-l-4 border-rose-400 bg-rose-50/60 p-3 space-y-1">
              <span className="text-xs font-bold text-rose-800 uppercase tracking-wider block">
                ⚠️ Tail Risks (ความเสี่ยงแฝง)
              </span>
              <ul className="list-disc pl-4 text-xs text-rose-700 space-y-0.5">
                {sentiment.tail_risks.map((r, i) => (
                  <li key={i}>{r}</li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
