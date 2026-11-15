import type { GuruPersonaDTO, RoundtableStanceType } from '../../api/types'

export interface SpeakerVisualMeta {
  speakerId: string
  displayName: string
  shortName: string
  archetype: string
  voiceName: string
  accentColor: string
  emoji: string
  pitch: number
  rate: number
  enabled: boolean
  debateOrder: number
  presetId: string
  presetLabel: string
  speakingStyle: string
  customDirective: string
  roleLensLabel: string
  focusSummary: string
  primaryEvidenceLabels: string[]
  analyticalRubric: string
}

export const SPEAKER_META_MAP: Record<string, SpeakerVisualMeta> = {
  hermes_moderator: {
    speakerId: 'hermes_moderator',
    displayName: 'Hermes (Moderator)',
    shortName: 'Hermes',
    archetype: 'Moderator & Portfolio Distiller',
    voiceName: 'Kore',
    accentColor: '#0ea5e9',
    emoji: '🪶',
    pitch: 1.0,
    rate: 1.04,
    enabled: true,
    debateOrder: 0,
    presetId: 'default',
    presetLabel: 'Hermes Balanced CIO (ค่าเริ่มต้น)',
    speakingStyle: 'กลาง ชัดเจน ร้อยเรียงข้อถกเถียงของทุกฝ่ายให้ตกผลึกเทียบกับพอร์ตจริงของผู้ใช้',
    customDirective: '',
    roleLensLabel: 'ประธานโต๊ะ & พอร์ตจริง',
    focusSummary:
      'ร้อยเรียงเหตุผลของทุกสายเทียบ @Essence 8 หมวด, เงินสด/Bucket และเป้าหมายพอร์ต เพื่อสรุปแผนแบ่งไม้และ Kill-Switch (รับบท Devil\'s Advocate แทนเมื่อปิด Skeptic)',
    primaryEvidenceLabels: [
      '@Essence 8 หมวด & กฎเหล็ก',
      '@Portfolio สัดส่วน, เงินสด & เป้าหมาย',
      'มติ & หลักฐานทุกสายในวง',
    ],
    analyticalRubric:
      '1) Priority Metrics: action_stance_hint, DATA QUALITY FLAGS รายหุ้น, Goal Tension & Required CAGR, Cash% & Bucket Room.\n' +
      '2) Thresholds: ห้ามสรุป BUY หาก action_stance_hint เป็น WAIT/TRIM/REDUCE; ตรวจ exclude_targets & risk_limits ครบ.\n' +
      '3) Focus: ชั่งน้ำหนักหลักฐานเชิงตัวเลขของทุกสายเทียบพอร์ตและเป้าหมายผู้ใช้โดยไม่เอนเอียงตามลำดับการพูด',
  },
  value_guru: {
    speakerId: 'value_guru',
    displayName: 'Value & MoS Guru',
    shortName: 'Value',
    archetype: 'Buffett / Graham Value Investor',
    voiceName: 'Charon',
    accentColor: '#10b981',
    emoji: '🏛️',
    pitch: 0.9,
    rate: 0.96,
    enabled: true,
    debateOrder: 1,
    presetId: 'default',
    presetLabel: 'Buffett & Graham Value (ค่าเริ่มต้น)',
    speakingStyle: 'สุขุม รอบคอบ แบบนักลงทุนอาวุโส ชั่งน้ำหนักราคาเทียบมูลค่าที่แท้จริงเสมอ',
    customDirective: '',
    roleLensLabel: 'มูลค่า & Margin of Safety',
    focusSummary:
      'ชั่งราคาตลาดเทียบ Intrinsic Value (DCF / Reverse DCF), Margin of Safety และ FCF Yield เทียบ WACC โดยไม่ไล่ราคา (รองรับ Justified PBV/DDM สำหรับแบงก์)',
    primaryEvidenceLabels: [
      'DCF / Reverse DCF & MoS',
      'FCF Yield vs WACC & Credit',
      'Normalized Margin & Watchlist',
    ],
    analyticalRubric:
      '1) Priority Metrics: mos_entry_price_today, current_margin_of_safety_pct, fcf_yield_pct, pe_vs_peer_avg_pct, normalized_mid_cycle_ebit_margin_pct (ถ้าเป็น cyclical), Watchlist Target Price.\n' +
      '2) Thresholds: BUY เมื่อ MoS >= 15–20% และ TV% < 75%; ระวัง Peak Margin Normalization ในหุ้นวัฏจักร.\n' +
      '3) Round 2 Cross-Exam: ทดสอบว่า Growth ที่ตลาด Price-in (`market_implied_growth_pct`) คุ้มค่าส่วนเผื่อความปลอดภัยจริงหรือไม่',
  },
  quality_guru: {
    speakerId: 'quality_guru',
    displayName: 'Quality & Moat Guru',
    shortName: 'Quality',
    archetype: 'Munger / Terry Smith Compounder',
    voiceName: 'Fenrir',
    accentColor: '#6366f1',
    emoji: '💎',
    pitch: 0.95,
    rate: 1.0,
    enabled: true,
    debateOrder: 2,
    presetId: 'default',
    presetLabel: 'Munger & Terry Smith (ค่าเริ่มต้น)',
    speakingStyle: 'เฉียบคม มองทะลุโครงสร้างอุตสาหกรรม ROIC, Pricing Power และความยั่งยืนของคูเมือง',
    customDirective: '',
    roleLensLabel: 'คูเมือง, ROIC & คุณภาพงบ',
    focusSummary:
      'คัดความทนทานของ Economic Moat และคุณภาพกำไรผ่าน ROIC vs WACC, Piotroski F-Score, OCF/Net Income, Sloan Accrual และเทรนด์งบ 3 ปี',
    primaryEvidenceLabels: [
      'ROIC vs WACC & Moat',
      'Piotroski F-Score & งบ 3 ปี',
      'OCF/NI, Sloan & SBC Dilution',
    ],
    analyticalRubric:
      '1) Priority Metrics: roic_pct, piotroski_f_score (9 pts), ocf_to_net_income, 3Y Trajectory (Gross/EBIT Margin, CapEx/D&A), sector_health_score.\n' +
      '2) Thresholds: แยก Compounder (ROIC >= 15%, F-Score >= 7, OCF/NI >= 0.85x) ออกจาก Value Trap.\n' +
      '3) Round 2 Cross-Exam: ตรวจสอบว่า Growth ที่พูดถึงสร้าง ROIIC สูงจริงหรือเพียงทุ่ม CapEx แลกยอดขาย',
  },
  growth_guru: {
    speakerId: 'growth_guru',
    displayName: 'Growth & Expectations Guru',
    shortName: 'Growth',
    voiceName: 'Puck',
    archetype: 'Peter Lynch / Fisher Growth',
    accentColor: '#f59e0b',
    emoji: '🚀',
    pitch: 1.08,
    rate: 1.06,
    enabled: true,
    debateOrder: 3,
    presetId: 'default',
    presetLabel: 'Peter Lynch & Fisher Growth (ค่าเริ่มต้น)',
    speakingStyle: 'มีพลัง มองหาอัตราโตที่ตลาดยังประเมินไม่ครบ เทียบ Implied Growth กับศักยภาพจริง',
    customDirective: '',
    roleLensLabel: 'การเติบโต & Expectation Gap',
    focusSummary:
      'วัด Expectation Gap ระหว่างอัตราโตจริง/Guidance กับที่ตลาด Price-in แล้ว รวมถึง EPS Revision 30 วัน, Operating Leverage และ Catalyst จาก @Digest',
    primaryEvidenceLabels: [
      'Growth vs Implied Growth',
      'Guidance & EPS Revision 30d',
      'Operating Leverage & @Digest',
    ],
    analyticalRubric:
      '1) Priority Metrics: revenue_growth_yoy_pct, market_implied_growth_pct, eps_revision_net_30d, guidance_summary, 3Y Operating Leverage.\n' +
      '2) Thresholds: BUY เมื่อ Fundamental Growth เหนือ Implied Growth พร้อม EPS Revisions เป็นบวกและมี Catalyst 6–18 เดือน.\n' +
      '3) Round 2 Cross-Exam: พิสูจน์ความคุ้มค่าของ Multiple ด้วยฐานกำไรใหม่และ Operating Leverage จากข้อมูลจริง',
  },
  macro_tactical_guru: {
    speakerId: 'macro_tactical_guru',
    displayName: 'Macro & Tactical Guru',
    shortName: 'Macro',
    archetype: 'Howard Marks / Druckenmiller Cycle',
    voiceName: 'Aoede',
    accentColor: '#ec4899',
    emoji: '🌊',
    pitch: 1.04,
    rate: 1.02,
    enabled: true,
    debateOrder: 4,
    presetId: 'default',
    presetLabel: 'Marks & Druckenmiller (ค่าเริ่มต้น)',
    speakingStyle: 'อ่านเข็มนาฬิกาวัฏจักรตลาด สภาพคล่อง เงินเฟ้อ และจังหวะการจัดสรรเงินอย่างยืดหยุ่น',
    customDirective: '',
    roleLensLabel: 'มหภาค, Flow & จังหวะเข้าทำ',
    focusSummary:
      'อ่านระบอบ @Macro, ความอ่อนไหวดอกเบี้ย +50/100bps, กราฟเทคนิค (Price Stage, Buy Zone, R:R, Stop Loss) และ Smart Money Flow',
    primaryEvidenceLabels: [
      '@Macro Regime & Sector',
      'Tactical Stage, Buy Zone & Stop',
      'Rate Shock +100bps & Flow',
    ],
    analyticalRubric:
      '1) Priority Metrics: @Macro Regime/Allocation/Sector Rotation/Thai Stance, @Digest Transmission Chain, tactical_setup, macro_yield_shock_100bps_delta.\n' +
      '2) Thresholds: วาง Tranche & Stop Loss ให้สอดคล้อง Price Stage, R:R >= 2.0x, Bucket Room และ Cash Reserve.\n' +
      '3) Round 2 Cross-Exam: แปลงมุมมองพื้นฐานเป็นจังหวะเข้าซื้อ/ลดไม้พร้อมระดับราคาและ Risk Trigger ที่ชัดเจน',
  },
  skeptic_guru: {
    speakerId: 'skeptic_guru',
    displayName: "Skeptic / Devil's Advocate",
    shortName: 'Skeptic',
    archetype: 'Chanos / Burry Risk & Falsification',
    voiceName: 'Orus',
    accentColor: '#ef4444',
    emoji: '🛡️',
    pitch: 0.86,
    rate: 0.98,
    enabled: true,
    debateOrder: 5,
    presetId: 'default',
    presetLabel: 'Burry & Chanos Bear (ค่าเริ่มต้น)',
    speakingStyle: 'เข้ม ตรงไปตรงมา ตั้งคำถามทวนกระแสเพื่อตรวจหาจุดอ่อนและสมมติฐานที่เปราะบาง',
    customDirective: '',
    roleLensLabel: 'จับผิดงบ & Kill-Switch',
    focusSummary:
      'ทำ Pre-Mortem หาจุดแตกหักของ Thesis, Data Quality Flags, Dilution/ภาระหนี้, Insider Open-Market Sell และบังคับวาง Kill-Switch วัดผลได้ทุกเทิร์น',
    primaryEvidenceLabels: [
      'DATA QUALITY FLAGS & หนี้',
      'Thesis Falsifiers & Tail Risk',
      'Insider Sell & Kill-Switch',
    ],
    analyticalRubric:
      '1) Priority Metrics: DATA QUALITY FLAGS รายหุ้น, thesis_falsifiers, thesis_delta (breached_falsifiers), net_debt_ebitda, synthetic_credit_rating, interest_coverage_ratio, terminal_value_pct_of_ev.\n' +
      '2) Thresholds: บังคับวาง 1–2 Measurable Kill-Switches (Metric+Condition+Threshold) ทุกเทิร์น; เตือนเมื่อ TV% > 75% หรือติด Negative FCF.\n' +
      '3) Round 2 Cross-Exam: เจาะจุดอ่อนงบการเงิน/สมมติฐาน Margin ที่ฝ่าย Bull มองข้าม',
  },
  user: {
    speakerId: 'user',
    displayName: 'คุณ (Portfolio Manager)',
    shortName: 'You',
    archetype: 'Portfolio Owner',
    voiceName: 'User',
    accentColor: '#0284c7',
    emoji: '✋',
    pitch: 1.0,
    rate: 1.0,
    enabled: true,
    debateOrder: 99,
    presetId: 'default',
    presetLabel: '',
    speakingStyle: '',
    customDirective: '',
    roleLensLabel: 'เจ้าของพอร์ต',
    focusSummary: 'ซักถามแทรกและกำหนดกรอบความเสี่ยงจากเจ้าของพอร์ต',
    primaryEvidenceLabels: [],
    analyticalRubric: '',
  },
}

export const STAGE_SEAT_ORDER = [
  'hermes_moderator',
  'value_guru',
  'quality_guru',
  'growth_guru',
  'macro_tactical_guru',
  'skeptic_guru',
] as const

export function getSpeakerMeta(
  speakerId: string,
  personas?: GuruPersonaDTO[],
): SpeakerVisualMeta {
  const defaultIdx = STAGE_SEAT_ORDER.indexOf(speakerId as (typeof STAGE_SEAT_ORDER)[number])
  const base = SPEAKER_META_MAP[speakerId] ?? {
    speakerId,
    displayName: speakerId,
    shortName: speakerId,
    archetype: 'Roundtable Speaker',
    voiceName: 'Kore',
    accentColor: '#64748b',
    emoji: '🎙️',
    pitch: 1.0,
    rate: 1.0,
    enabled: true,
    debateOrder: defaultIdx >= 0 ? defaultIdx : 99,
    presetId: 'default',
    presetLabel: '',
    speakingStyle: '',
    customDirective: '',
    roleLensLabel: 'ร่วมเสวนา',
    focusSummary: '',
    primaryEvidenceLabels: [],
    analyticalRubric: '',
  }
  const serverOverride = personas?.find((p) => p.speaker_id === speakerId)
  if (!serverOverride) return base
  return {
    ...base,
    displayName: serverOverride.display_name || base.displayName,
    shortName: serverOverride.short_name || base.shortName,
    archetype: serverOverride.archetype || base.archetype,
    voiceName: serverOverride.voice_name || base.voiceName,
    accentColor: serverOverride.accent_color || base.accentColor,
    emoji: serverOverride.emoji || base.emoji,
    pitch: typeof serverOverride.pitch === 'number' ? serverOverride.pitch : base.pitch,
    rate: typeof serverOverride.rate === 'number' ? serverOverride.rate : base.rate,
    enabled:
      speakerId === 'hermes_moderator'
        ? true
        : typeof serverOverride.enabled === 'boolean'
          ? serverOverride.enabled
          : base.enabled,
    debateOrder:
      speakerId === 'hermes_moderator'
        ? 0
        : typeof serverOverride.debate_order === 'number'
          ? serverOverride.debate_order
          : base.debateOrder,
    presetId: serverOverride.preset_id || base.presetId,
    presetLabel: serverOverride.preset_label || base.presetLabel,
    speakingStyle: serverOverride.speaking_style ?? base.speakingStyle,
    customDirective: serverOverride.custom_directive ?? base.customDirective,
    roleLensLabel: serverOverride.role_lens_label || base.roleLensLabel,
    focusSummary: serverOverride.focus_summary || base.focusSummary,
    primaryEvidenceLabels:
      serverOverride.primary_evidence_labels && serverOverride.primary_evidence_labels.length > 0
        ? serverOverride.primary_evidence_labels
        : base.primaryEvidenceLabels,
    analyticalRubric: serverOverride.analytical_rubric || base.analyticalRubric,
  }
}

export interface ParsedRubricLine {
  title: string
  body: string
}

const RUBRIC_TERM_LABELS: ReadonlyArray<readonly [RegExp, string]> = [
  [/\baction_stance_hint\b/g, 'ท่าทีแนะนำจากโมเดลระบบ'],
  [/\bexclude_targets\b/g, 'หุ้นต้องห้าม'],
  [/\brisk_limits\b/g, 'เพดานเสี่ยง'],
  [/\bmos_entry_price_today\b/g, 'ราคาเข้าซื้อ MoS'],
  [/\bcurrent_margin_of_safety_pct\b/g, 'Margin of Safety (%)'],
  [/\bnormalized_mid_cycle_ebit_margin_pct\b/g, 'Normalized EBIT Margin กลางวัฏจักร'],
  [/\bfcf_yield_pct\b/g, 'FCF Yield (%)'],
  [/\bpe_vs_peer_avg_pct\b/g, 'P/E เทียบค่าเฉลี่ยกลุ่ม'],
  [/\broic_pct\b/g, 'ROIC (%)'],
  [/\bpiotroski_f_score\b/g, 'Piotroski F-Score'],
  [/\bocf_to_net_income\b/g, 'OCF/กำไรสุทธิ'],
  [/\bsloan_accrual\b/g, 'Sloan Accrual'],
  [/\bdupont_breakdown\b/g, 'โครงสร้าง DuPont ROE'],
  [/\bcapex_to_depreciation_ratio\b/g, 'อัตราส่วน CapEx/ค่าเสื่อม'],
  [/\breinvestment_lifecycle_phase\b/g, 'ระยะวงจรลงทุนซ้ำ'],
  [/\brevenue_growth_yoy_pct\b/g, 'อัตราโตรายได้ (YoY%)'],
  [/\bnet_income_growth_yoy_pct\b/g, 'อัตราโตกำไร (YoY%)'],
  [/\bmarket_implied_growth_pct\b/g, 'อัตราโตที่ตลาด Price-in'],
  [/\beps_revision_net_30d\b/g, 'การปรับประมาณการ EPS 30 วัน'],
  [/\beps_estimate_change_30d_pct\b/g, '% เปลี่ยนประมาณการ EPS 30 วัน'],
  [/\bearnings_momentum_score\b/g, 'คะแนนโมเมนตัมกำไร'],
  [/\bearnings_guidance_context\b/g, 'Guidance ผู้บริหาร'],
  [/\bguidance_summary\b/g, 'สรุป Guidance'],
  [/\btactical_setup\b/g, 'กรอบกราฟเทคนิค'],
  [/\bmacro_yield_shock_50bps\/100bps_delta\b/g, 'แรงกระแทกดอกเบี้ย +50/100bps'],
  [/\bmacro_yield_shock_100bps_delta\b/g, 'แรงกระแทกดอกเบี้ย +100bps'],
  [/\bsmart_money_flags\b/g, 'กระแสเงินสถาบัน'],
  [/\bbuy_zone_min\.\.max\b/g, 'โซนราคาเข้าซื้อ'],
  [/\binvalidation_stop_loss\b/g, 'จุด Stop Loss'],
  [/\bcurrent_rr_ratio\b/g, 'สัดส่วน R:R ปัจจุบัน'],
  [/\bnet_debt_ebitda\b/g, 'หนี้สุทธิ/EBITDA'],
  [/\bsynthetic_credit_rating\b/g, 'อันดับเครดิตจำลอง'],
  [/\binterest_coverage_ratio\b/g, 'ความสามารถจ่ายดอกเบี้ย (ICR)'],
  [/\bbenchmark_cost_of_debt_pct\b/g, 'ต้นทุนหนี้อ้างอิง (%)'],
  [/\bterminal_value_pct_of_ev\b/g, 'สัดส่วน Terminal Value (%)'],
  [/\bthesis_falsifiers\b/g, 'เงื่อนไขหักล้างสมมติฐาน'],
  [/\bbreached_falsifiers\b/g, 'เงื่อนไขที่ถูกทำลายแล้ว'],
  [/\binsider_conviction\b/g, 'พฤติกรรมซื้อขายของ Insider'],
  [/\bopen_market_s_value_usd_str\b/g, 'ยอดขาย Insider นอกตลาด'],
  [/\brule_10b5_1_s_value_usd_str\b/g, 'ยอดขายตามแผนล่วงหน้า 10b5-1'],
  [/\bamihud_illiquidity_20d\b/g, 'สภาพคล่อง Amihud 20 วัน'],
  [/\bmdd_pct\b/g, 'Max Drawdown อดีต (%)'],
  [/\bsector_health_score\b/g, 'สุขภาพอุตสาหกรรม'],
  [/\bsector KPI\b/gi, 'KPI เฉพาะอุตสาหกรรม'],
  [/\bkill_switches\b/g, 'Kill-Switch'],
  [/\btension_axes\b/g, 'แกนข้อขัดแย้ง'],
  [/\baction_steps\b/g, 'ขั้นตอนปฏิบัติ'],
]

export function formatRubricTextForUi(text: string): string {
  let out = text
  for (const [pattern, label] of RUBRIC_TERM_LABELS) {
    out = out.replace(pattern, label)
  }
  return out
}

export function parseAnalyticalRubricLines(rubric?: string): ParsedRubricLine[] {
  const raw = (rubric || '').trim()
  if (!raw) return []
  return raw
    .split('\n')
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => {
      const m = line.match(/^(\d+\)\s*[^:]+):\s*(.*)$/)
      if (!m) return { title: '', body: formatRubricTextForUi(line) }
      return { title: m[1]!.trim(), body: formatRubricTextForUi(m[2]!.trim()) }
    })
}

export function getOrderedSeatIds(
  personas?: GuruPersonaDTO[],
  includeDisabled: boolean = false,
): string[] {
  const ranked = STAGE_SEAT_ORDER.map((seatId, defaultIdx) => {
    const meta = getSpeakerMeta(seatId, personas)
    return { seatId, defaultIdx, order: meta.debateOrder, enabled: meta.enabled }
  })
  const filtered = includeDisabled ? ranked : ranked.filter((item) => item.enabled)
  filtered.sort((a, b) => a.order - b.order || a.defaultIdx - b.defaultIdx)
  return filtered.map((item) => item.seatId)
}

export function stanceBadgeClass(stance: RoundtableStanceType | string): string {
  switch (stance) {
    case 'BUY':
      return 'border-emerald-300 bg-emerald-50 text-emerald-800'
    case 'OVERWEIGHT':
      return 'border-teal-300 bg-teal-50 text-teal-800'
    case 'HOLD':
      return 'border-amber-300 bg-amber-50 text-amber-800'
    case 'UNDERWEIGHT':
      return 'border-orange-300 bg-orange-50 text-orange-800'
    case 'SELL':
      return 'border-rose-300 bg-rose-50 text-rose-800'
    default:
      return 'border-slate-300 bg-slate-50 text-slate-700'
  }
}

export function roundStageLabel(stage: string): { label: string; toneClass: string } {
  switch (stage) {
    case 'stage_intro':
      return {
        label: 'Opening Hook · Hermes เปิดประเด็น',
        toneClass: 'border-cyan-200 bg-cyan-50/90 text-cyan-800',
      }
    case 'opening':
      return {
        label: 'รอบที่ 1 · เปิดมุมมองหลัก',
        toneClass: 'border-sky-200 bg-sky-50/90 text-sky-800',
      }
    case 'rebuttal':
      return {
        label: 'รอบที่ 2 · ซักค้าน & สัดส่วนถือ',
        toneClass: 'border-blue-200 bg-blue-50/90 text-blue-800',
      }
    case 'crossfire':
      return {
        label: 'Crossfire · ปะทะเดือดตัวต่อตัว',
        toneClass: 'border-rose-200 bg-rose-50/90 text-rose-800',
      }
    case 'user_interjection':
      return {
        label: '✋ ยกมือแทรกถาม',
        toneClass: 'border-amber-200 bg-amber-50 text-amber-800',
      }
    case 'interjection_reply':
      return {
        label: '🎙️ ตอบคำถามแทรก',
        toneClass: 'border-violet-200 bg-violet-50 text-violet-800',
      }
    case 'synthesis':
      return {
        label: 'รอบที่ 3 · Hermes ตกผลึกตัดสินใจ',
        toneClass: 'border-emerald-200 bg-emerald-50 text-emerald-800',
      }
    default:
      return {
        label: stage,
        toneClass: 'border-slate-200 bg-slate-50 text-slate-700',
      }
  }
}

export interface ParsedEvidenceBullet {
  tag: 'FACT' | 'MATH' | 'INFERRED'
  text: string
  badgeClass: string
}

export function parseEvidenceBullet(raw: string): ParsedEvidenceBullet {
  const trimmed = raw.trim()
  const match = trimmed.match(/^\[(FACT|MATH|INFERRED)\]\s*(.*)$/i)
  if (!match) {
    return {
      tag: 'FACT',
      text: trimmed,
      badgeClass: 'border-emerald-200 bg-emerald-50 text-emerald-800',
    }
  }
  const tag = match[1]!.toUpperCase() as 'FACT' | 'MATH' | 'INFERRED'
  const text = match[2] || trimmed
  const badgeClass =
    tag === 'MATH'
      ? 'border-sky-200 bg-sky-50 text-sky-800'
      : tag === 'INFERRED'
        ? 'border-amber-200 bg-amber-50 text-amber-800'
        : 'border-emerald-200 bg-emerald-50 text-emerald-800'
  return { tag, text, badgeClass }
}

export function portfolioAttachmentLabel(state: string): {
  label: string
  detail: string
  badgeClass: string
} {
  switch (state) {
    case 'ATTACHED_HELD':
      return {
        label: 'แนบพอร์ตแล้ว · ถือครองอยู่',
        detail: 'กูรูเห็นสัดส่วนและต้นทุนจริงในพอร์ตของคุณแล้ว',
        badgeClass: 'border-emerald-200 bg-emerald-50 text-emerald-800',
      }
    case 'ATTACHED_FLAT_BOOK':
      return {
        label: 'แนบพอร์ตแล้ว · ยังไม่ได้ถือหุ้นนี้',
        detail: 'พอร์ตเชื่อมต่อแล้ว — หุ้นนี้ยังอยู่นอกพอร์ต (0 หุ้น)',
        badgeClass: 'border-sky-200 bg-sky-50 text-sky-800',
      }
    case 'ATTACHED_EMPTY':
      return {
        label: 'แนบพอร์ตแล้ว · พอร์ตว่าง',
        detail: 'ยังไม่พบรายการถือครองในพอร์ตที่เชื่อมต่อ',
        badgeClass: 'border-amber-200 bg-amber-50 text-amber-800',
      }
    default:
      return {
        label: 'ไม่แนบพอร์ต · วิเคราะห์เป็นกลาง',
        detail: 'ไม่ได้แนบ @Portfolio — กูรูจะไม่สมมติยอดถือครองของคุณ',
        badgeClass: 'border-slate-200 bg-slate-50 text-slate-700',
      }
  }
}

