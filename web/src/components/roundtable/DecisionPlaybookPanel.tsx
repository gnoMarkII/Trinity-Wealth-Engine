import { useMemo, useState } from 'react'
import { motion } from 'motion/react'
import type {
  ActionPlaybookOptionDTO,
  DebatePillarCoverageDTO,
  DistilledDecisionPlaybookDTO,
  GuruPersonaDTO,
  PastScorecardDTO,
} from '../../api/types'
import { getSpeakerMeta, portfolioAttachmentLabel, stanceBadgeClass } from './personas'
import {
  ChangeSummary,
  DecisionInbox,
  NumberTicker,
  SegmentedControl,
  SmartRecommendation,
  type Decision,
  type MetricChange,
  type Recommendation,
} from './ui'

interface DecisionPlaybookPanelProps {
  personas?: GuruPersonaDTO[]
  primaryTicker?: string | null
  portfolioAttachmentState: string
  missingPrerequisiteTickers: string[]
  contextDriftDetected: boolean
  pastScorecard?: PastScorecardDTO | null
  decisionPlaybook?: DistilledDecisionPlaybookDTO | null
  pillarCoverage?: DebatePillarCoverageDTO | null
  onDispatchTicker: (ticker: string) => void
  onDismissMissingPrerequisites?: () => void
  onRefreshContextOnDrift: () => void
  onDismissDrift?: () => void
  onSaveToVault: () => void
  onUpdateThesis: () => void
  onExportYoutubePack?: () => void
  onFocusSpeakerTurn?: (speakerId: string) => void
  hasTurns?: boolean
  isSavingVault?: boolean
  isUpdatingThesis?: boolean
  isExportingYoutube?: boolean
  onCollapseHud?: () => void
}

const PILLAR_SHORT_LABELS: Record<string, string> = {
  'Valuation & MOS': 'มูลค่า & MOS',
  'Quality, Moat & Capital': 'คุณภาพ & ROIC',
  'Growth & Catalysts': 'เติบโต & ตัวเร่ง',
  'Macro, Cycle & Portfolio': 'มหภาค & จังหวะ',
  'Downside, Credit & Falsifiers': 'เสี่ยง & Kill-Switch',
}

type PlaybookTab = 'playbook' | 'tensions' | 'alignment' | 'risks'

export function DecisionPlaybookPanel({
  personas,
  primaryTicker,
  portfolioAttachmentState,
  missingPrerequisiteTickers,
  contextDriftDetected,
  pastScorecard,
  decisionPlaybook,
  pillarCoverage,
  onDispatchTicker,
  onDismissMissingPrerequisites,
  onRefreshContextOnDrift,
  onDismissDrift,
  onSaveToVault,
  onUpdateThesis,
  onExportYoutubePack,
  onFocusSpeakerTurn,
  hasTurns = false,
  isSavingVault = false,
  isUpdatingThesis = false,
  isExportingYoutube = false,
  onCollapseHud,
}: DecisionPlaybookPanelProps) {
  const [activeTab, setActiveTab] = useState<PlaybookTab>('playbook')
  const [selectedAction, setSelectedAction] = useState<ActionPlaybookOptionDTO | null>(null)

  const portfolioMeta = portfolioAttachmentLabel(portfolioAttachmentState)

  const guardDecisions: Decision[] = useMemo(() => {
    const items: Decision[] = []
    for (const ticker of missingPrerequisiteTickers) {
      items.push({
        id: `missing-${ticker}`,
        title: `หุ้น $${ticker} ยังไม่มีบทวิเคราะห์เจาะลึกในระบบ`,
        detail:
          'แนะนำให้ส่งคำสั่งวิเคราะห์เจาะลึกก่อน เพื่อดึง DCF, คุณภาพกำไร และความเสี่ยงครบชุดมาให้สภากูรู',
        primary: `ส่งวิเคราะห์ $${ticker}`,
        secondary: 'ดำเนินการต่อด้วยข้อมูลที่มี',
      })
    }
    if (contextDriftDetected) {
      items.push({
        id: 'context-drift',
        title: 'พบข้อมูลพอร์ต / มหภาค / หุ้นมีการเปลี่ยนแปลงจากตอนเริ่มประชุม',
        detail:
          'ข้อมูลต้นฉบับถูกแช่แข็งไว้เพื่อความเที่ยงตรง ต้องการอัปเดต Context Packet ล่าสุดหรือไม่?',
        primary: 'รีเฟรชบริบทล่าสุด',
        secondary: 'คงบริบทเดิมต่อ',
      })
    }
    return items
  }, [missingPrerequisiteTickers, contextDriftDetected])

  const scorecardChanges: MetricChange[] = useMemo(() => {
    if (!pastScorecard) return []
    const rawNum = Number(pastScorecard.raw_return_pct)
    const alphaNum = Number(pastScorecard.alpha_pct)
    const benchNum = Number(pastScorecard.benchmark_return_pct)
    return [
      {
        label: 'คำแนะนำครั้งก่อน',
        change: `${pastScorecard.previous_final_rating} · ${pastScorecard.previous_date}`,
        tone: 'neutral',
      },
      {
        label: `ผลตอบแทน ${pastScorecard.primary_ticker}`,
        change: Number.isFinite(rawNum)
          ? `${rawNum >= 0 ? '+' : ''}${rawNum.toFixed(2)}%`
          : pastScorecard.settlement_status,
        tone: Number.isFinite(rawNum) ? (rawNum >= 0 ? 'good' : 'bad') : 'neutral',
      },
      {
        label: `เทียบ ${pastScorecard.benchmark_ticker}`,
        change: Number.isFinite(benchNum)
          ? `${benchNum >= 0 ? '+' : ''}${benchNum.toFixed(2)}% (Alpha ${
              Number.isFinite(alphaNum) ? `${alphaNum >= 0 ? '+' : ''}${alphaNum.toFixed(2)}%` : 'N/A'
            })`
          : 'รอปิดผลรอบ',
        tone: Number.isFinite(alphaNum) ? (alphaNum >= 0 ? 'good' : 'bad') : 'neutral',
      },
    ]
  }, [pastScorecard])

  const recommendationOptions: Recommendation[] = useMemo(() => {
    if (!decisionPlaybook?.action_playbook) return []
    const all = [
      decisionPlaybook.action_playbook.primary_action,
      ...decisionPlaybook.action_playbook.alternatives,
    ]
    return all.map((opt) => {
      const srcMeta = getSpeakerMeta(opt.speaker_source, personas)
      return {
        id: opt.option_id,
        label: opt.label,
        signal: Math.min(3, Math.max(0, Math.round(opt.conviction_level))) as 0 | 1 | 2 | 3,
        reason:
          [
            `เสนอโดย ${srcMeta.emoji} ${srcMeta.shortName}`,
            opt.entry_zone ? `โซนราคา: ${opt.entry_zone}` : null,
            opt.sizing_note ? `สัดส่วน: ${opt.sizing_note}` : null,
            opt.action_steps.length ? opt.action_steps.map((s) => `• ${s}`).join('\n') : null,
          ]
            .filter(Boolean)
            .join('\n') || `เสนอโดย ${srcMeta.shortName}`,
      }
    })
  }, [decisionPlaybook, personas])

  const activePlaybookOption = useMemo(() => {
    if (!decisionPlaybook?.action_playbook) return null
    if (selectedAction) return selectedAction
    return decisionPlaybook.action_playbook.primary_action
  }, [decisionPlaybook, selectedAction])

  return (
    <aside className="flex min-h-0 flex-col gap-2.5 overflow-y-auto pr-0.5 custom-scrollbar">
      {/* Compact Single-Line Portfolio Guard Status */}
      <div
        title={portfolioMeta.detail}
        className="flex items-center justify-between gap-2 rounded-xl border border-sky-200/70 bg-white/80 px-3 py-2 shadow-sm backdrop-blur-sm"
      >
        <span className="text-[11px] font-bold uppercase tracking-wider text-slate-500">
          Portfolio Guard
        </span>
        <div className="flex min-w-0 items-center gap-1.5">
          <span
            className={`truncate rounded-full border px-2 py-0.5 text-[11px] font-semibold ${portfolioMeta.badgeClass}`}
          >
            {portfolioMeta.label}
          </span>
          {onCollapseHud && (
            <button
              type="button"
              onClick={onCollapseHud}
              title="ย่อแผง Playbook ลงเป็นปุ่มมุมขวา"
              className="shrink-0 rounded-lg border border-slate-200 bg-white px-2 py-0.5 text-[11px] font-semibold text-slate-600 hover:bg-slate-50 hover:text-slate-900"
            >
              ▾ ย่อ
            </button>
          )}
        </div>
      </div>

      {/* Prerequisite Ticker & Context Drift Guard */}
      {guardDecisions.length > 0 && (
        <DecisionInbox
          decisions={guardDecisions}
          onDecide={(decision, answer) => {
            if (decision.id.startsWith('missing-')) {
              const ticker = decision.id.replace('missing-', '')
              if (answer === 'primary') {
                onDispatchTicker(ticker)
              } else {
                onDismissMissingPrerequisites?.()
              }
            } else if (decision.id === 'context-drift') {
              if (answer === 'primary') {
                onRefreshContextOnDrift()
              } else {
                onDismissDrift?.()
              }
            }
          }}
        />
      )}

      {/* Closed-Loop Settlement Scorecard */}
      {pastScorecard && (
        <div className="space-y-2 rounded-2xl border border-sky-200/80 bg-white/90 p-3 shadow-sm">
          {pastScorecard.alpha_pct && Number.isFinite(Number(pastScorecard.alpha_pct)) && (
            <div className="flex items-baseline justify-between px-1">
              <span className="text-xs font-semibold text-slate-600">
                Benchmark Alpha รอบก่อน ({pastScorecard.primary_ticker})
              </span>
              <span
                className={`text-lg font-extrabold ${
                  Number(pastScorecard.alpha_pct) >= 0 ? 'text-emerald-600' : 'text-rose-600'
                }`}
              >
                <NumberTicker
                  value={Number(pastScorecard.alpha_pct)}
                  format={{ maximumFractionDigits: 2, signDisplay: 'always' }}
                />
                %
              </span>
            </div>
          )}
          <ChangeSummary
            title={`Closed-Loop Scorecard · ${pastScorecard.primary_ticker}`}
            period={pastScorecard.outcome_label}
            changes={scorecardChanges}
          />
          {pastScorecard.reflection_lesson && (
            <div className="rounded-xl border border-amber-200/80 bg-amber-50/70 p-2.5 text-xs text-amber-950">
              <strong className="font-semibold">บทเรียนจากรอบก่อน:</strong>{' '}
              {pastScorecard.reflection_lesson}
            </div>
          )}
        </div>
      )}

      {/* Hermes Decision Playbook Distillation Card */}
      {decisionPlaybook ? (
        <motion.section
          initial={{ opacity: 0, y: 10, rotateX: 3 }}
          animate={{ opacity: 1, y: 0, rotateX: 0 }}
          transition={{ duration: 0.25 }}
          className="space-y-3 rounded-2xl border border-sky-200/85 bg-white/90 p-3.5 shadow-sm backdrop-blur-sm"
        >
          <div className="flex items-start justify-between gap-2">
            <div>
              <div className="text-[11px] font-bold uppercase tracking-wider text-sky-700">
                🪶 Hermes Decision Playbook
              </div>
              <div className="mt-1 flex items-center gap-2">
                <span
                  className={`rounded-lg border px-2.5 py-0.5 text-xs font-extrabold tracking-wide ${stanceBadgeClass(
                    decisionPlaybook.final_rating,
                  )}`}
                >
                  {decisionPlaybook.final_rating}
                </span>
                <span className="text-xs font-semibold text-slate-600">
                  Conviction {decisionPlaybook.core_conviction_score}/3
                </span>
              </div>
            </div>
          </div>

          <p className="rounded-xl border border-sky-100 bg-sky-50/50 p-2.5 text-xs leading-relaxed text-slate-800">
            {decisionPlaybook.executive_verdict}
          </p>

          {pillarCoverage && pillarCoverage.pillars.length > 0 && (
            <div
              aria-label="ความครอบคลุม 5 มิติการวิเคราะห์"
              className="rounded-xl border border-slate-200/80 bg-slate-50/70 p-2 space-y-1.5"
            >
              <div className="flex items-center justify-between gap-2 text-[11px]">
                <span className="font-bold text-slate-700">🧭 5 มิติความครบถ้วน</span>
                <span className="font-semibold text-slate-500">
                  เปิดประเด็น {pillarCoverage.raised_count}/{pillarCoverage.pillars.length} · ซักค้าน{' '}
                  {pillarCoverage.challenged_count}/{pillarCoverage.pillars.length}
                </span>
              </div>
              <div className="flex flex-wrap gap-1">
                {pillarCoverage.pillars.map((item) => {
                  const shortLabel = PILLAR_SHORT_LABELS[item.pillar] || item.pillar
                  const raised = item.raised_by.length > 0
                  const badgeTone = item.is_challenged
                    ? 'border-emerald-200 bg-emerald-50 text-emerald-800'
                    : raised
                      ? 'border-amber-200 bg-amber-50 text-amber-800'
                      : 'border-slate-200 bg-white text-slate-500'
                  const statusIcon = item.is_challenged ? '✅' : raised ? '💬' : '○'
                  return (
                    <span
                      key={item.pillar}
                      title={
                        item.is_challenged
                          ? `${item.pillar}: เปิดและซักค้านแล้ว`
                          : raised
                            ? `${item.pillar}: เปิดประเด็นแล้ว (ยังไม่ถูกซักค้าน)`
                            : `${item.pillar}: ยังไม่ถูกหยิบยกในรอบนี้`
                      }
                      className={`rounded-md border px-1.5 py-0.5 text-[11px] font-semibold ${badgeTone}`}
                    >
                      {statusIcon} {shortLabel}
                    </span>
                  )
                })}
              </div>
            </div>
          )}

          <SegmentedControl<PlaybookTab>
            size="sm"
            fullWidth
            kind="tabs"
            aria-label="มุมมอง Decision Playbook"
            value={activeTab}
            onChange={setActiveTab}
            options={[
              { value: 'playbook', label: 'แผนลงมือ' },
              { value: 'tensions', label: 'ข้อขัดแย้ง', count: decisionPlaybook.tension_axes.length },
              { value: 'alignment', label: 'แกนพอร์ต' },
              { value: 'risks', label: 'Kill-Switch', count: decisionPlaybook.kill_switches.length },
            ]}
          />

          {activeTab === 'playbook' && (
            <div className="space-y-2.5">
              {recommendationOptions.length > 0 && (
                <SmartRecommendation
                  title="Action Playbook"
                  options={recommendationOptions}
                  onAccept={(rec) => {
                    const matched = [
                      decisionPlaybook.action_playbook.primary_action,
                      ...decisionPlaybook.action_playbook.alternatives,
                    ].find((a) => a.option_id === rec.id)
                    if (matched) setSelectedAction(matched)
                  }}
                />
              )}

              {activePlaybookOption && activePlaybookOption.action_steps.length > 0 && (
                <div className="rounded-xl border border-slate-200/80 bg-slate-50/70 p-2.5 space-y-1.5">
                  <div className="text-[11px] font-bold uppercase tracking-wider text-slate-500">
                    ขั้นตอนปฏิบัติ ({activePlaybookOption.label})
                  </div>
                  <ul className="space-y-1">
                    {activePlaybookOption.action_steps.map((step, i) => (
                      <li key={i} className="flex items-start gap-1.5 text-xs text-slate-700">
                        <span className="font-bold text-sky-600">{i + 1}.</span>
                        <span>{step}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {decisionPlaybook.consensus_points.length > 0 && (
                <div className="rounded-xl border border-emerald-200/80 bg-emerald-50/50 p-2.5 space-y-1">
                  <div className="text-[11px] font-bold text-emerald-900">
                    ✅ จุดที่กูรูเห็นพ้องกัน
                  </div>
                  <ul className="space-y-0.5 text-xs text-emerald-950">
                    {decisionPlaybook.consensus_points.map((pt, i) => (
                      <li key={i}>• {pt}</li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          )}

          {activeTab === 'tensions' && (
            <div className="space-y-2">
              {decisionPlaybook.tension_axes.length === 0 ? (
                <p className="text-xs text-slate-500">ไม่มีประเด็นขัดแย้งหลักในรอบนี้</p>
              ) : (
                decisionPlaybook.tension_axes.map((ax, idx) => {
                  const bullMeta = getSpeakerMeta(ax.bull_side_speaker, personas)
                  const bearMeta = getSpeakerMeta(ax.bear_side_speaker, personas)
                  return (
                    <div
                      key={idx}
                      className="rounded-xl border border-slate-200 bg-slate-50/60 p-2.5 space-y-1.5"
                    >
                      <div className="text-xs font-bold text-slate-900">⚡ {ax.topic}</div>
                      <div className="rounded-lg border border-emerald-200 bg-emerald-50/70 p-2 text-xs text-emerald-950">
                        {onFocusSpeakerTurn ? (
                          <button
                            type="button"
                            onClick={() => onFocusSpeakerTurn(ax.bull_side_speaker)}
                            title="คลิกเพื่อเลื่อนไปอ่านเทิร์นล่าสุดของกูรูท่านนี้"
                            className="font-bold hover:underline"
                          >
                            มุมมองบวก ({bullMeta.emoji} {bullMeta.shortName}):{' '}
                          </button>
                        ) : (
                          <span className="font-bold">
                            มุมมองบวก ({bullMeta.emoji} {bullMeta.shortName}):{' '}
                          </span>
                        )}
                        {ax.bull_argument}
                      </div>
                      <div className="rounded-lg border border-rose-200 bg-rose-50/70 p-2 text-xs text-rose-950">
                        {onFocusSpeakerTurn ? (
                          <button
                            type="button"
                            onClick={() => onFocusSpeakerTurn(ax.bear_side_speaker)}
                            title="คลิกเพื่อเลื่อนไปอ่านเทิร์นล่าสุดของกูรูท่านนี้"
                            className="font-bold hover:underline"
                          >
                            มุมมองแย้ง ({bearMeta.emoji} {bearMeta.shortName}):{' '}
                          </button>
                        ) : (
                          <span className="font-bold">
                            มุมมองแย้ง ({bearMeta.emoji} {bearMeta.shortName}):{' '}
                          </span>
                        )}
                        {ax.bear_argument}
                      </div>
                    </div>
                  )
                })
              )}
            </div>
          )}

          {activeTab === 'alignment' && (
            <div className="space-y-2">
              {decisionPlaybook.portfolio_context_summary && (
                <p className="rounded-xl border border-sky-100 bg-sky-50/50 p-2.5 text-xs text-slate-700">
                  {decisionPlaybook.portfolio_context_summary}
                </p>
              )}
              {decisionPlaybook.personal_axis_alignment.length === 0 ? (
                <p className="text-xs text-slate-500">ยังไม่มีแกนกลยุทธ์ส่วนตัวในระบบ</p>
              ) : (
                decisionPlaybook.personal_axis_alignment.map((item, idx) => (
                  <div
                    key={idx}
                    className="flex items-start justify-between gap-2 rounded-xl border border-slate-200 bg-white p-2.5 text-xs"
                  >
                    <div>
                      <div className="font-bold text-slate-900">{item.dimension}</div>
                      <p className="mt-0.5 text-slate-600">{item.detail}</p>
                    </div>
                    <span
                      className={`shrink-0 rounded-full border px-2 py-0.5 text-[11px] font-bold uppercase ${
                        item.status === 'aligned'
                          ? 'border-emerald-200 bg-emerald-50 text-emerald-800'
                          : item.status === 'conflict'
                            ? 'border-rose-200 bg-rose-50 text-rose-800'
                            : item.status === 'caution'
                              ? 'border-amber-200 bg-amber-50 text-amber-800'
                              : 'border-slate-200 bg-slate-50 text-slate-600'
                      }`}
                    >
                      {item.status}
                    </span>
                  </div>
                ))
              )}
            </div>
          )}

          {activeTab === 'risks' && (
            <div className="space-y-2.5">
              <div className="rounded-xl border border-rose-200/80 bg-rose-50/60 p-2.5 space-y-1">
                <div className="text-[11px] font-bold uppercase tracking-wider text-rose-900">
                  🚨 Kill-Switches (จุดทบทวน/ตัดขาดทุน)
                </div>
                {decisionPlaybook.kill_switches.length === 0 ? (
                  <p className="text-xs text-rose-800">ไม่มีเงื่อนไขพิเศษเพิ่มเติม</p>
                ) : (
                  <ul className="space-y-1 text-xs text-rose-950">
                    {decisionPlaybook.kill_switches.map((k, i) => (
                      <li key={i}>• {k}</li>
                    ))}
                  </ul>
                )}
              </div>

              <div className="rounded-xl border border-slate-200 bg-white p-2.5">
                <div className="mb-1.5 text-[11px] font-bold uppercase tracking-wider text-slate-500">
                  สรุปจุดยืนกูรูรายคน
                </div>
                <div className="grid grid-cols-2 gap-1.5">
                  {Object.entries(decisionPlaybook.guru_stance_breakdown).map(([spk, stance]) => (
                    <div
                      key={spk}
                      className="flex items-center justify-between rounded-lg border border-slate-100 bg-slate-50/70 px-2 py-1 text-[11px]"
                    >
                      <span className="truncate font-medium text-slate-700">
                        {getSpeakerMeta(spk, personas).shortName}
                      </span>
                      <span
                        className={`rounded border px-1.5 py-0 text-[11px] font-bold ${stanceBadgeClass(
                          stance,
                        )}`}
                      >
                        {stance}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}

          {/* Integration Action Buttons */}
          <div className="flex flex-col gap-2 border-t border-sky-100 pt-2.5">
            <div className="flex flex-wrap gap-2">
              <button
                type="button"
                disabled={isSavingVault}
                onClick={onSaveToVault}
                className="flex-1 rounded-xl border border-sky-300 bg-sky-50 px-3 py-2 text-xs font-bold text-sky-900 transition-colors hover:bg-sky-100 disabled:opacity-50"
              >
                {isSavingVault ? '⏳ กำลังบันทึก...' : '📝 Save to Vault'}
              </button>
              <button
                type="button"
                disabled={!primaryTicker || isUpdatingThesis}
                onClick={onUpdateThesis}
                title={
                  primaryTicker
                    ? `อัปเดต Active Thesis ของ $${primaryTicker}`
                    : 'ต้องมีหุ้นหลัก ($TICKER) จึงจะอัปเดต Active Thesis ได้'
                }
                className="flex-1 rounded-xl border border-emerald-300 bg-emerald-50 px-3 py-2 text-xs font-bold text-emerald-900 transition-colors hover:bg-emerald-100 disabled:cursor-not-allowed disabled:opacity-50"
              >
                {isUpdatingThesis
                  ? '⏳ กำลังอัปเดต...'
                  : `🎯 Update Thesis${primaryTicker ? ` ($${primaryTicker})` : ''}`}
              </button>
            </div>
            {onExportYoutubePack && (
              <button
                type="button"
                disabled={!hasTurns || isExportingYoutube}
                onClick={onExportYoutubePack}
                className="w-full rounded-xl border border-rose-300 bg-gradient-to-r from-rose-50 to-amber-50 px-3 py-2 text-xs font-bold text-rose-900 transition-colors hover:from-rose-100 hover:to-amber-100 disabled:cursor-not-allowed disabled:opacity-50"
              >
                {isExportingYoutube
                  ? '⏳ กำลังรวมไฟล์เสียง + ซับไตเติล YouTube (.zip)...'
                  : '🎬 Export ชุดทำคลิป YouTube (.zip: WAV + SRT + Cue Sheet)'}
              </button>
            )}
          </div>
        </motion.section>
      ) : (
        <section className="space-y-3 rounded-2xl border border-dashed border-sky-200/85 bg-gradient-to-b from-white/85 to-sky-50/35 p-5 text-center backdrop-blur-sm">
          {/* 3D Levitating Hermes Crystallizer Core */}
          <div
            aria-hidden="true"
            className="rt-perspective relative mx-auto flex h-20 w-20 items-center justify-center"
          >
            <span className="rt-orbit-ring-primary absolute inset-0 rounded-full border border-sky-400/50" />
            <span className="rt-orbit-ring-secondary absolute inset-1.5 rounded-full border border-cyan-400/45" />
            <span className="rt-crystal-levitate flex h-11 w-11 items-center justify-center rounded-2xl border border-sky-200 bg-gradient-to-br from-white via-sky-50 to-cyan-100 text-2xl shadow-md">
              ⚖️
            </span>
          </div>
          <h3 className="text-xs font-bold text-slate-800">
            Hermes Decision Playbook จะปรากฏที่นี่
          </h3>
          <p className="text-[11.5px] leading-relaxed text-slate-500">
            เมื่อกูรูถกเถียงครบทุกยก Hermes จะตกผลึกข้อขัดแย้ง แผนลงมือ และ Kill-Switch ให้อัตโนมัติ
          </p>
          {onExportYoutubePack && hasTurns && (
            <button
              type="button"
              disabled={isExportingYoutube}
              onClick={onExportYoutubePack}
              className="w-full rounded-xl border border-rose-300 bg-rose-50 px-3 py-2 text-xs font-bold text-rose-900 transition-colors hover:bg-rose-100 disabled:opacity-50"
            >
              {isExportingYoutube
                ? '⏳ กำลังรวมชุด YouTube (.zip)...'
                : '🎬 Export ชุด YouTube จากเทิร์นที่มีอยู่'}
            </button>
          )}
        </section>
      )}
    </aside>
  )
}

