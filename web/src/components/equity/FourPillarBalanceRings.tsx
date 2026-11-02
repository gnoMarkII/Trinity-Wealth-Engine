import React from 'react'
import { ContextTooltip } from '../macro/explainer/ContextTooltip'

export interface PillarItem {
  id: string
  title: string
  score: number | null
  displayScore?: string
  maxScore?: number
  weightText: string
  weightBadgeType?: 'normal' | 'reweighted' | 'excluded' | 'timing'
  drivers: string
  statusColor?: 'emerald' | 'sky' | 'indigo' | 'amber' | 'rose' | 'zinc'
  isExcluded?: boolean
}


const PILLAR_EXPLAINERS: Record<string, {
  subtitle: string
  tooltip: {
    title: string
    what: string
    normalBenchmark: string
    decisionImpact: string
  }
}> = {
  'pillar-1-fundamental': {
    subtitle: '🏢 คุณภาพธุรกิจ & สุขภาพงบการเงิน',
    tooltip: {
      title: 'Pillar 1: Fundamental Quality (คุณภาพพื้นฐาน)',
      what: 'วัดความแข็งแกร่งของงบการเงิน ความสามารถในการทำกำไร (Operating Margin), คุณภาพกระแสเงินสด (FCF) และคะแนนตรวจสุขภาพการเงิน 9 ด่าน (Piotroski F-Score)',
      normalBenchmark: 'คะแนน > 70 ถือว่าพื้นฐานยอดเยี่ยม ป้อมปราการแกร่ง; ต่ำกว่า 40 ควรระวังความเสี่ยงด้านงบการเงิน',
      decisionImpact: 'หากพื้นฐานอ่อนแอ แม้ราคาจะลดลงมาเยอะก็ไม่ควรซื้อ เพราะอาจเป็นกับดักมูลค่า (Value Trap)',
    },
  },
  'pillar-2-guidance': {
    subtitle: '🎯 แนวโน้มกำไร & คำสัญญาผู้บริหาร',
    tooltip: {
      title: 'Pillar 2: Guidance & Expectations (ความคาดหวัง)',
      what: 'วัดความน่าเชื่อถือของผู้บริหารในการทำผลงานตามเป้า (Say-Do Ratio) และทิศทางการปรับประมาณการกำไรของนักวิเคราะห์สถาบัน (Consensus Revisions)',
      normalBenchmark: 'คะแนน > 60 แปลว่านักวิเคราะห์เริ่มปรับเป้ากำไรขึ้น; ถ้าคะแนนต่ำแปลว่าผลงานแย่กว่าเป้าต่อเนื่อง',
      decisionImpact: 'เป็นแรงขับเคลื่อนราคาหุ้นระยะสั้นถึงกลาง หุ้นที่ถูกปรับเป้ากำไรขึ้นต่อเนื่องมัก Outperform ตลาด',
    },
  },
  'pillar-3-valuation': {
    subtitle: '🏷️ ความถูกแพง & ส่วนเผื่อความปลอดภัย',
    tooltip: {
      title: 'Pillar 3: Valuation Margin (ความถูกแพง)',
      what: 'ประเมินมูลค่าแท้จริงผ่านแบบจำลองคิดลดกระแสเงินสด (DCF) และวิเคราะห์ย้อนกลับว่าราคาปัจจุบันตลาดคาดหวังการเติบโตเท่าใด (Reverse DCF Hurdle)',
      normalBenchmark: 'มี Margin of Safety อย่างน้อย 15-20% (ราคาตลาดต่ำกว่ามูลค่าแท้จริง)',
      decisionImpact: 'บอกแต้มต่อของการลงทุน อย่าซื้อหุ้นดีในราคาที่แพงเกินไปจนไม่มีส่วนเผื่อความปลอดภัย',
    },
  },
  'pillar-4-tactical': {
    subtitle: '⚡ จังหวะเวลา & รอยเท้าสถาบัน',
    tooltip: {
      title: 'Pillar 4: Tactical Execution (จังหวะเวลา)',
      what: 'วิเคราะห์ช่วงวงจรราคาหุ้น (Stage 1-4), แนวรับแนวต้าน, รายการซื้อขายของผู้บริหาร (Insider Form 4), และกระแสเงินสถาบัน (Smart Money)',
      normalBenchmark: 'Stage 2 ขาขึ้น พร้อมสัญญาณผู้บริหารซื้อสุทธิ = จังหวะเปิดสถานะที่ได้เปรียบสูงสุด',
      decisionImpact: 'ช่วยกำหนดจังหวะเข้าซื้อ (Timing) และจุดตัดขาดทุน (Stop Loss) เพื่อไม่ให้ติดดอยนาน',
    },
  },
}

export interface FourPillarBalanceRingsProps {
  fundamentalScore?: number | null
  guidanceScore?: number | null
  valuationScore?: number | null
  executionReadiness?: number | null
  isReweighted?: boolean
  reweightingMetadata?: {
    excluded_pillars?: string[]
    weights_used?: Record<string, number>
    reason?: string
  } | null
  hasAnalystRevisionOnly?: boolean
  executionBreakdown?: {
    stage?: number
    setup?: number | null
    insider?: number
    setup_reason?: string | null
  } | null
  className?: string
}

function getRingColors(score: number | null, isExcluded: boolean) {
  if (isExcluded) {
    return {
      stroke: '#a1a1aa', // zinc-400
      glow: 'rgba(161, 161, 170, 0.2)',
      text: 'text-zinc-600',
      badgeBg: 'bg-zinc-100 text-zinc-700 border-zinc-200',
    }
  }
  if (score == null) {
    return {
      stroke: '#94a3b8', // slate-400
      glow: 'rgba(148, 163, 184, 0.2)',
      text: 'text-zinc-500',
      badgeBg: 'bg-zinc-100 text-zinc-600 border-zinc-200',
    }
  }
  if (score >= 70) {
    return {
      stroke: '#10b981', // emerald-500
      glow: 'rgba(16, 185, 129, 0.25)',
      text: 'text-emerald-700',
      badgeBg: 'bg-emerald-50 text-emerald-800 border-emerald-200',
    }
  }
  if (score >= 50) {
    return {
      stroke: '#0284c7', // sky-600
      glow: 'rgba(2, 132, 199, 0.25)',
      text: 'text-sky-700',
      badgeBg: 'bg-sky-50 text-sky-800 border-sky-200',
    }
  }
  if (score >= 35) {
    return {
      stroke: '#f59e0b', // amber-500
      glow: 'rgba(245, 158, 11, 0.25)',
      text: 'text-amber-700',
      badgeBg: 'bg-amber-50 text-amber-800 border-amber-200',
    }
  }
  return {
    stroke: '#f43f5e', // rose-500
    glow: 'rgba(244, 63, 94, 0.25)',
    text: 'text-rose-700',
    badgeBg: 'bg-rose-50 text-rose-800 border-rose-200',
  }
}

export const FourPillarBalanceRings: React.FC<FourPillarBalanceRingsProps> = ({
  fundamentalScore,
  guidanceScore,
  valuationScore,
  executionReadiness,
  isReweighted,
  reweightingMetadata,
  hasAnalystRevisionOnly,
  executionBreakdown,
  className = '',
}) => {
  const isValuationExcluded = isReweighted || (reweightingMetadata?.excluded_pillars?.includes('valuation') ?? false)

  // Execution score scaled to 100 for consistent visualization (readiness is 0-10 in DTO)
  const executionScore100 = executionReadiness != null ? Math.round(executionReadiness * 10) : null

  const pillars: PillarItem[] = [
    {
      id: 'pillar-1-fundamental',
      title: 'Pillar 1: Fundamental',
      score: fundamentalScore ?? null,
      displayScore: fundamentalScore != null ? fundamentalScore.toFixed(1) : 'N/A',
      maxScore: 100,
      weightText: isReweighted ? '55% Wt (Reweighted)' : '40% Wt',
      weightBadgeType: isReweighted ? 'reweighted' : 'normal',
      drivers: 'งบการเงิน, FCF Quality, Piotroski F-Score',
      isExcluded: false,
    },
    {
      id: 'pillar-2-guidance',
      title: hasAnalystRevisionOnly ? 'Pillar 2: Analyst Revisions' : 'Pillar 2: Guidance & Revisions',
      score: guidanceScore ?? null,
      displayScore: guidanceScore != null ? guidanceScore.toFixed(1) : 'N/A',
      maxScore: 100,
      weightText: isReweighted ? '45% Wt (Reweighted)' : '30% Wt',
      weightBadgeType: isReweighted ? 'reweighted' : 'normal',
      drivers: hasAnalystRevisionOnly
        ? 'EPS Consensus Revisions'
        : 'Guidance Delivery & Institutional Revisions',
      isExcluded: false,
    },
    {
      id: 'pillar-3-valuation',
      title: 'Pillar 3: Valuation Margin',
      score: isValuationExcluded ? null : (valuationScore ?? null),
      displayScore: isValuationExcluded
        ? 'Info'
        : (valuationScore != null ? valuationScore.toFixed(1) : 'N/A'),
      maxScore: 100,
      weightText: isValuationExcluded ? '0% (Excluded)' : '30% Wt',
      weightBadgeType: isValuationExcluded ? 'excluded' : 'normal',
      drivers: isValuationExcluded
        ? 'Macro Anomaly (ERP ≤ 0 / Ke < Rf)'
        : '12M DCF Upside & Reverse DCF Hurdle',
      isExcluded: isValuationExcluded,
    },
    {
      id: 'pillar-4-tactical',
      title: 'Tactical Execution',
      score: executionScore100,
      displayScore: executionScore100 != null ? `${executionScore100}` : 'N/A',
      maxScore: 100,
      weightText: 'Timing & Risk Gate',
      weightBadgeType: 'timing',
      drivers: executionBreakdown?.stage
        ? `Stage ${executionBreakdown.stage} (${executionBreakdown.setup_reason || 'Timing Filter'})`
        : 'Stage Setup, S/R Levels, SEC Flow',
      isExcluded: false,
    },
  ]

  // SVG circular dimensions
  const radius = 34
  const strokeWidth = 5.5
  const circumference = 2 * Math.PI * radius

  return (
    <div className={`my-5 grid grid-cols-1 gap-3.5 sm:grid-cols-2 lg:grid-cols-4 ${className}`} data-testid="four-pillar-rings">
      {pillars.map((pillar) => {
        const ringColor = getRingColors(pillar.score, pillar.isExcluded ?? false)
        const normalizedPct = pillar.isExcluded
          ? 0
          : pillar.score != null
          ? Math.min(100, Math.max(0, pillar.score))
          : 0
        const strokeDashoffset = circumference - (normalizedPct / 100) * circumference

        return (
          <div
            key={pillar.id}
            data-testid={`pillar-card-${pillar.id}`}
            className="group relative flex flex-col justify-between rounded-xl border border-edge/70 bg-surface/80 p-4 transition-all duration-200 hover:border-sky-300 hover:shadow-md hover:bg-surface"
          >
            {/* Top row: Title & Weight Badge with ContextTooltip */}
            <div className="flex items-start justify-between gap-2 mb-1.5">
              <div className="flex items-center gap-1.5 min-w-0">
                <span className="text-xs font-semibold text-zinc-800 tracking-tight leading-snug truncate">
                  {pillar.title}
                </span>
                {PILLAR_EXPLAINERS[pillar.id] ? (
                  <ContextTooltip
                    title={PILLAR_EXPLAINERS[pillar.id]!.tooltip.title}
                    whatIsIt={PILLAR_EXPLAINERS[pillar.id]!.tooltip.what}
                    howToRead={PILLAR_EXPLAINERS[pillar.id]!.tooltip.normalBenchmark}
                    impactOnMoney={PILLAR_EXPLAINERS[pillar.id]!.tooltip.decisionImpact}
                  />
                ) : null}
              </div>
              <span
                className={`shrink-0 inline-flex items-center text-[10px] font-bold px-2 py-0.5 rounded-full border ${
                  pillar.weightBadgeType === 'reweighted'
                    ? 'bg-sky-50 text-sky-800 border-sky-300'
                    : pillar.weightBadgeType === 'excluded'
                    ? 'bg-amber-50 text-amber-800 border-amber-300'
                    : pillar.weightBadgeType === 'timing'
                    ? 'bg-purple-50 text-purple-800 border-purple-200'
                    : 'bg-zinc-100 text-zinc-700 border-zinc-200'
                }`}
              >
                {pillar.weightText}
              </span>
            </div>

            {/* Plain-Language Subtitle for Beginners */}
            {PILLAR_EXPLAINERS[pillar.id] ? (
              <div className="text-[11px] font-medium text-sky-800/80 mb-2 truncate">
                {PILLAR_EXPLAINERS[pillar.id]!.subtitle}
              </div>
            ) : null}

            {/* Middle: SVG Ring & Giant Score */}
            <div className="flex items-center gap-4 my-1">
              <div className="relative shrink-0 w-20 h-20 flex items-center justify-center">
                <svg className="w-20 h-20 -rotate-90 transform" viewBox="0 0 84 84">
                  {/* Background Track */}
                  <circle
                    cx="42"
                    cy="42"
                    r={radius}
                    stroke="currentColor"
                    strokeWidth={strokeWidth}
                    fill="transparent"
                    className="text-zinc-100"
                  />
                  {/* Animated Progress Ring */}
                  {!pillar.isExcluded && (
                    <circle
                      cx="42"
                      cy="42"
                      r={radius}
                      stroke={ringColor.stroke}
                      strokeWidth={strokeWidth}
                      strokeDasharray={circumference}
                      strokeDashoffset={strokeDashoffset}
                      strokeLinecap="round"
                      fill="transparent"
                      className="transition-all duration-700 ease-out"
                      style={{
                        filter: `drop-shadow(0 0 3px ${ringColor.glow})`,
                      }}
                    />
                  )}
                  {pillar.isExcluded && (
                    <circle
                      cx="42"
                      cy="42"
                      r={radius}
                      stroke="#d4d4d8"
                      strokeWidth={strokeWidth}
                      strokeDasharray="4 4"
                      fill="transparent"
                    />
                  )}
                </svg>

                {/* Score Number Centered inside Ring */}
                <div className="absolute inset-0 flex flex-col items-center justify-center pointer-events-none">
                  <span className={`text-lg font-black tracking-tight ${ringColor.text}`}>
                    {pillar.displayScore}
                  </span>
                  {!pillar.isExcluded && pillar.score != null && (
                    <span className="text-[9px] font-semibold text-zinc-400 -mt-0.5">/ 100</span>
                  )}
                </div>
              </div>

              {/* Score Assessment Summary */}
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-1.5 mb-1">
                  <span
                    className="w-2 h-2 rounded-full shrink-0"
                    style={{ backgroundColor: ringColor.stroke }}
                  />
                  <span className="text-xs font-bold text-zinc-900 truncate">
                    {pillar.isExcluded
                      ? 'Macro Excluded'
                      : pillar.score == null
                      ? 'No Data'
                      : pillar.score >= 70
                      ? 'Strong Conviction'
                      : pillar.score >= 50
                      ? 'Moderate'
                      : pillar.score >= 35
                      ? 'Cautious'
                      : 'High Risk'}
                  </span>
                </div>
                <div className="text-[11px] text-zinc-500 line-clamp-2 leading-relaxed">
                  {pillar.drivers}
                </div>
              </div>
            </div>

            {/* Bottom: Subtle Status Indicator Bar */}
            <div className="mt-2.5 pt-2 border-t border-edge/40 flex items-center justify-between text-[10px] text-zinc-400">
              <span>{pillar.isExcluded ? 'Status: Fenced' : 'Pillar Contribution'}</span>
              <span className="font-semibold text-zinc-600">
                {pillar.isExcluded
                  ? 'Non-binding'
                  : pillar.score != null
                  ? `${Math.round(normalizedPct)}% of max`
                  : 'N/A'}
              </span>
            </div>
          </div>
        )
      })}
    </div>
  )
}

export default FourPillarBalanceRings
