import type { EquitySentimentContextDTO, SmartMoneyFlagsDTO, TacticalSetupDTO } from '../../api/types'

export interface TacticalSetupProps extends Partial<TacticalSetupDTO> {}

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

function formatLargeNum(val?: number | null, prefix = '$'): string {
  if (val == null || isNaN(val)) return 'N/A'
  const abs = Math.abs(val)
  if (abs >= 1e9) return `${prefix}${(val / 1e9).toFixed(2)}B`
  if (abs >= 1e6) return `${prefix}${(val / 1e6).toFixed(2)}M`
  if (abs >= 1e3) return `${prefix}${(val / 1e3).toFixed(1)}K`
  return `${prefix}${val.toLocaleString()}`
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
        <div className="space-y-3.5">
          <div className="flex items-center justify-between text-xs font-semibold text-zinc-500 uppercase tracking-wider">
            <span>🎯 Tactical Setup ({tactical.horizon_timeframe || '1-3M'})</span>
            {tactical.atr_14 != null && (
              <span className="text-zinc-500 font-normal">ATR(14): {currency}{tactical.atr_14.toFixed(2)}</span>
            )}
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
            {/* Buy Zone */}
            <div className="rounded-xl border border-emerald-200/80 bg-emerald-50/50 p-3">
              <div className="flex items-center justify-between">
                <span className="text-emerald-800 font-semibold block text-[10px] uppercase">
                  Optimal Buy Zone
                </span>
                {tactical.is_in_buy_zone && (
                  <span className="text-[9px] px-1 py-0.2 rounded bg-emerald-200/80 text-emerald-900 font-bold">
                    IN ZONE
                  </span>
                )}
              </div>
              <span className="font-bold text-emerald-950 text-sm block mt-0.5">
                {tactical.buy_zone_min != null && tactical.buy_zone_max != null
                  ? `${currency}${tactical.buy_zone_min.toFixed(2)} - ${currency}${tactical.buy_zone_max.toFixed(2)}`
                  : 'N/A'}
              </span>
            </div>

            {/* Invalidation Stop Loss */}
            <div className="rounded-xl border border-rose-200/80 bg-rose-50/50 p-3">
              <span className="text-rose-800 font-semibold block text-[10px] uppercase">
                Stop Loss (จุดยอมแพ้)
              </span>
              <span className="font-bold text-rose-950 text-sm block mt-0.5">
                {tactical.invalidation_stop_loss != null
                  ? `${currency}${tactical.invalidation_stop_loss.toFixed(2)}`
                  : 'N/A'}
              </span>
            </div>

            {/* Tactical Target */}
            <div className="rounded-xl border border-sky-200/80 bg-sky-50/50 p-3">
              <span className="text-sky-800 font-semibold block text-[10px] uppercase">
                Tactical Target
              </span>
              <span className="font-bold text-sky-950 text-sm block mt-0.5">
                {tactical.tactical_target_price != null
                  ? `${currency}${tactical.tactical_target_price.toFixed(2)}`
                  : 'N/A'}
              </span>
            </div>

            {/* Risk / Reward Ratio */}
            <div className="rounded-xl border border-edge/80 bg-surface p-3">
              <span className="text-zinc-500 font-semibold block text-[10px] uppercase">
                R:R ณ Analysis Price
              </span>
              <span className="font-bold text-zinc-900 text-sm block mt-0.5">
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

          {/* Support / Resistance Levels */}
          <div className="flex flex-wrap items-center justify-between text-xs rounded-xl bg-surface-strong/60 border border-sky-100 p-3">
            <div className="flex items-center gap-2">
              <span className="text-zinc-500 font-medium">Key Support:</span>
              <span className="font-bold text-zinc-900">
                {tactical.key_support_level != null ? `${currency}${tactical.key_support_level.toFixed(2)}` : 'N/A'}
              </span>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-zinc-500 font-medium">Key Resistance:</span>
              <span className="font-bold text-zinc-900">
                {tactical.key_resistance_level != null ? `${currency}${tactical.key_resistance_level.toFixed(2)}` : 'N/A'}
              </span>
            </div>
          </div>

          {/* Breakout Blueprint Strip */}
          {tactical.breakout_trigger_price != null && (
            <div className="rounded-xl border border-indigo-200/80 bg-indigo-50/40 p-3.5 text-xs space-y-2">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <div className="flex items-center gap-2">
                  <span className="font-bold text-indigo-900 uppercase text-[10px]">🚀 Breakout Setup:</span>
                  <span className="text-indigo-800">
                    Trigger <strong>${tactical.breakout_trigger_price.toFixed(2)}</strong> | Target <strong>${tactical.breakout_target_price?.toFixed(2)}</strong> | Stop <strong>${tactical.breakout_stop_loss?.toFixed(2)}</strong>
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

      {/* Section 2: SEC Form 4 & Smart Money Flow */}
      {(insider || smartMoney) && (
        <div className="space-y-3 pt-1 border-t border-edge/60">
          <div className="flex items-center justify-between text-xs font-semibold text-zinc-500 uppercase tracking-wider">
            <span>🕵️ Smart Money & SEC Form 4 Flow (90D)</span>
            {insider?.status && (
              <span className="px-2 py-0.5 rounded text-[10px] font-bold uppercase bg-surface border border-edge text-zinc-700">
                {insider.status.replace(/_/g, ' ')}
              </span>
            )}
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
            {/* Form 4 Purchases */}
            <div className="bg-surface/80 rounded-lg p-2.5 border border-edge/40">
              <span className="text-zinc-400 block text-[10px] uppercase">Open-Market Purchases (P)</span>
              <span className="font-bold text-zinc-900 text-sm block mt-0.5">
                {insider ? `${insider.open_market_p_count_90d || 0} (${formatLargeNum(insider.open_market_p_value_usd, currency)})` : 'N/A'}
              </span>
            </div>

            {/* C-Suite Purchases */}
            <div className="bg-surface/80 rounded-lg p-2.5 border border-edge/40">
              <span className="text-zinc-400 block text-[10px] uppercase">C-Suite Purchases (CEO/CFO)</span>
              <span className="font-bold text-zinc-900 text-sm block mt-0.5">
                {insider ? `${insider.c_suite_p_count || 0} รายการ` : 'N/A'}
              </span>
            </div>

            {/* Institutional Ownership */}
            <div className="bg-surface/80 rounded-lg p-2.5 border border-edge/40">
              <span className="text-zinc-400 block text-[10px] uppercase">Institutional Ownership</span>
              <span className="font-bold text-zinc-900 text-sm block mt-0.5">
                {smartMoney?.institutional_ownership_pct != null ? `${smartMoney.institutional_ownership_pct}%` : 'N/A'}
              </span>
            </div>

            {/* Short Interest */}
            <div className="bg-surface/80 rounded-lg p-2.5 border border-edge/40">
              <span className="text-zinc-400 block text-[10px] uppercase">Short Interest</span>
              <span className={`font-bold text-sm block mt-0.5 ${smartMoney?.short_squeeze_risk ? 'text-amber-600' : 'text-zinc-900'}`}>
                {smartMoney?.short_interest_pct != null ? `${smartMoney.short_interest_pct}%` : 'N/A'}
                {smartMoney?.short_squeeze_risk && ' ⚡'}
              </span>
            </div>
          </div>
        </div>
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
