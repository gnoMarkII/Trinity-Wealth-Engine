export type FlagCategory = 'critical_fallback' | 'model_methodology' | 'market_warning' | 'low_impact_fallback'

export interface FormattedFlag {
  rawCode: string
  label: string
  subtext?: string
  category: FlagCategory
}

export const CATEGORY_CONFIG: Record<
  FlagCategory,
  {
    title: string
    badgeText: string
    badgeBg: string
    badgeTextClass: string
    borderClass: string
    iconBgClass: string
    iconColorClass: string
    containerBgClass: string
  }
> = {
  critical_fallback: {
    title: 'ข้อจำกัดข้อมูลและค่าสำรอง (Data Limitations & Fallbacks)',
    badgeText: 'Critical Fallback',
    badgeBg: 'bg-rose-100',
    badgeTextClass: 'text-rose-800',
    borderClass: 'border-rose-200',
    iconBgClass: 'bg-rose-100',
    iconColorClass: 'text-rose-600',
    containerBgClass: 'bg-rose-50/50',
  },
  model_methodology: {
    title: 'สมมติฐานแบบจำลอง (Model Methodology)',
    badgeText: 'Methodology',
    badgeBg: 'bg-sky-100',
    badgeTextClass: 'text-sky-800',
    borderClass: 'border-sky-200',
    iconBgClass: 'bg-sky-100',
    iconColorClass: 'text-sky-600',
    containerBgClass: 'bg-sky-50/40',
  },
  market_warning: {
    title: 'สภาวะตลาด (Market Context Warning)',
    badgeText: 'Market Context',
    badgeBg: 'bg-amber-100',
    badgeTextClass: 'text-amber-800',
    borderClass: 'border-amber-200',
    iconBgClass: 'bg-amber-100',
    iconColorClass: 'text-amber-600',
    containerBgClass: 'bg-amber-50/40',
  },
  low_impact_fallback: {
    title: 'ข้อสังเกตและข้อมูลสำรองผลกระทบต่ำ (Low Impact / Notes)',
    badgeText: 'Informational',
    badgeBg: 'bg-zinc-100',
    badgeTextClass: 'text-zinc-700',
    borderClass: 'border-zinc-200',
    iconBgClass: 'bg-zinc-100',
    iconColorClass: 'text-zinc-600',
    containerBgClass: 'bg-zinc-50/50',
  },
}

const KNOWN_FLAGS: Record<string, { label: string; subtext?: string; category: FlagCategory }> = {
  // dcf_valuation.py
  'hardcoded_us_risk_free:dcf': {
    label: 'ใช้อัตราดอกเบี้ย Risk-Free อ้างอิง (4.25%)',
    subtext: 'ไม่พบข้อมูลอัตราดอกเบี้ยพันธบัตรรัฐบาลสหรัฐฯ 10 ปีในระบบ จึงใช้อัตราดอกเบี้ยสำรองในการคำนวณ DCF',
    category: 'critical_fallback',
  },
  'hardcoded_th_risk_free:dcf': {
    label: 'ใช้อัตราดอกเบี้ย Risk-Free ไทยอ้างอิง (2.75%)',
    subtext: 'ไม่พบข้อมูลพันธบัตรรัฐบาลไทย 10 ปีล่าสุด จึงใช้อัตราดอกเบี้ยสำรอง',
    category: 'critical_fallback',
  },
  'hardcoded_country_risk_premium:dcf': {
    label: 'ใช้ Country Risk Premium อ้างอิง (1.75%)',
    subtext: 'ใช้อัตราความเสี่ยงประเทศสำรองตามตาราง Damodaran',
    category: 'critical_fallback',
  },
  'hardcoded_cost_of_debt:dcf': {
    label: 'ใช้ Cost of Debt อ้างอิง (5.0%)',
    subtext: 'ไม่พบข้อมูลดอกเบี้ยจ่ายจริง จึงใช้ค่าประมาณการสำรอง (หากสัดส่วนหนี้ต่ำจะกระทบ WACC น้อย)',
    category: 'low_impact_fallback',
  },
  'kd_clamped:dcf': {
    label: 'จำกัดช่วง Cost of Debt (2%-15%)',
    subtext: 'ปรับอัตราดอกเบี้ยจ่ายให้อยู่ในกรอบมาตรฐานการคำนวณ',
    category: 'low_impact_fallback',
  },
  'wacc_below_terminal_growth:dcf': {
    label: 'WACC ต่ำกว่าอัตราเติบโต Terminal Growth',
    subtext: 'WACC มีค่าน้อยกว่าหรือเท่ากับอัตราการเติบโตระยะยาว ไม่สามารถใช้ Gordon Growth Model โดยตรง',
    category: 'critical_fallback',
  },
  'eps_proxy_base_growth:dcf': {
    label: 'ใช้ YoY EPS Growth เป็นตัวแทน FCF Base Growth',
    subtext: 'ประมาณการการเติบโตระยะยาว 5 ปีของ Free Cash Flow จากอัตราการเติบโต EPS',
    category: 'model_methodology',
  },
  'generic_base_growth_assumption:dcf': {
    label: 'ใช้ Base Growth อ้างอิง 5.0%',
    subtext: 'ไม่พบข้อมูล EPS Growth จึงใช้สมมติฐานการเติบโตมาตรฐาน 5%',
    category: 'model_methodology',
  },
  'rich_market_valuation_low_erp:dcf': {
    label: 'Equity Risk Premium (ERP) ของตลาดค่อนข้างต่ำ (<1.5%)',
    subtext: 'ผลตอบแทนชดเชยความเสี่ยงหุ้นเทียบกับพันธบัตรแคบลง สะท้อนสภาวะตลาด Valuation ตึงตัว',
    category: 'market_warning',
  },
  'negative_fcf_dcf_unavailable:dcf': {
    label: 'ไม่สามารถคำนวณ DCF ได้ (FCF ติดลบ)',
    subtext: 'กระแสเงินสดเสรีเป็นลบ จึงไม่สามารถประเมินมูลค่าด้วย DCF ได้',
    category: 'critical_fallback',
  },
  'beta_unavailable_dcf_unavailable:dcf': {
    label: 'ไม่สามารถคำนวณ DCF ได้ (ไม่พบค่า Beta)',
    subtext: 'ไม่มีข้อมูลความผันผวนเทียบกับตลาด (Beta) ทำให้คำนวณ Cost of Equity (Ke) ไม่ได้',
    category: 'critical_fallback',
  },

  // ownership.py
  '10b51_unfiltered:insider_signal': {
    label: 'ข้อมูล Insider ไม่ได้แยกแผน 10b5-1',
    subtext: 'รายการซื้อขายของผู้บริหารรวมทั้งการซื้อขายตามแผนล่วงหน้า (10b5-1) และแบบสมัครใจ',
    category: 'low_impact_fallback',
  },
  'insider_date_unavailable:insider_signal': {
    label: 'ไม่มีข้อมูลวันที่ในรายการ Insider',
    subtext: 'ไม่สามารถกรองรายการเฉพาะ 90 วันล่าสุดได้',
    category: 'critical_fallback',
  },

  // quant_scoring.py
  'negative_earnings:pe_undefined': {
    label: 'ไม่สามารถคำนวณ P/E ได้ (กำไรติดลบ)',
    subtext: 'บริษัทมีผลขาดทุนสุทธิ ทำให้ไม่สามารถประเมินมูลค่าผ่าน P/E Ratio ได้',
    category: 'critical_fallback',
  },
  'missing_growth_data:growth': {
    label: 'ข้อมูลการเติบโตไม่เพียงพอ',
    subtext: 'ขาดข้อมูลรายได้หรือกำไรย้อนหลังสำหรับคำนวณ Growth Score',
    category: 'critical_fallback',
  },
  'missing_dividend_data:dividend': {
    label: 'ข้อมูลเงินปันผลไม่เพียงพอ',
    subtext: 'ขาดข้อมูลการจ่ายเงินปันผลหรืออัตราตอบแทนปันผล',
    category: 'low_impact_fallback',
  },
  'unsustainable_payout:dividend': {
    label: 'อัตราจ่ายปันผลสูงเกินความยั่งยืน (>100%)',
    subtext: 'เงินปันผลที่จ่ายสูงกว่ากำไรสุทธิ มีความเสี่ยงที่จะลดการจ่ายปันผลในอนาคต',
    category: 'market_warning',
  },
  'missing_solvency_data:solvency': {
    label: 'ข้อมูลความมั่นคงทางการเงินไม่เพียงพอ',
    subtext: 'ขาดข้อมูลงบการเงินสำหรับคำนวณ Solvency Score',
    category: 'critical_fallback',
  },
  'high_leverage_risk:solvency': {
    label: 'ความเสี่ยงภาระหนี้สินสูง',
    subtext: 'สัดส่วนหนี้สินต่อทุน (D/E) หรือ Net Debt/EBITDA อยู่ในระดับสูงกว่ามาตรฐาน',
    category: 'market_warning',
  },
  'missing_liquidity_data:liquidity': {
    label: 'ข้อมูลสภาพคล่องการซื้อขายไม่เพียงพอ',
    subtext: 'ไม่พบข้อมูลมูลค่าการซื้อขายเฉลี่ยรายวัน (ADTV)',
    category: 'critical_fallback',
  },
  'low_liquidity:liquidity': {
    label: 'สภาพคล่องการซื้อขายค่อนข้างต่ำ',
    subtext: 'มูลค่าการซื้อขายเฉลี่ยรายวันต่ำกว่าเกณฑ์ อาจมีขีดจำกัดในการเข้าซื้อหรือขายออก',
    category: 'market_warning',
  },
  'insufficient_dimensions:composite': {
    label: 'มิติการประเมินไม่ครบถ้วน',
    subtext: 'มิติคำนวณ Score ไม่ครบ จึงมีการปรับ re-normalize น้ำหนักที่เหลือ',
    category: 'model_methodology',
  },
  'missing_fcf_quality_data:fcf_quality': {
    label: 'ข้อมูลคุณภาพกระแสเงินสดไม่เพียงพอ',
    subtext: 'ขาดข้อมูลงบกระแสเงินสดสำหรับประเมิน FCF Quality',
    category: 'critical_fallback',
  },
  'missing_debt_quality_data:debt_quality': {
    label: 'ข้อมูลคุณภาพหนี้สินไม่เพียงพอ',
    subtext: 'ขาดข้อมูลดอกเบี้ยจ่ายหรือภาระหนี้สำหรับประเมิน Debt Quality',
    category: 'critical_fallback',
  },
  'high_debt_risk:debt_quality': {
    label: 'ความเสี่ยงคุณภาพหนี้สินสูง',
    subtext: 'ความสามารถในการชำระดอกเบี้ย (Interest Coverage) ต่ำกว่าเกณฑ์ความปลอดภัย',
    category: 'market_warning',
  },

  // peer_valuation.py
  'missing_own_pe:peer_relative': {
    label: 'ไม่สามารถเปรียบเทียบ P/E กับกลุ่มได้ (ไม่มี P/E ตนเอง)',
    subtext: 'หุ้นมี P/E ติดลบหรือไม่พบค่า P/E จึงไม่สามารถเปรียบเทียบกับกลุ่มอุตสาหกรรมได้',
    category: 'critical_fallback',
  },
  'insufficient_peers:peer_relative': {
    label: 'จำนวนหุ้นเปรียบเทียบในกลุ่มไม่เพียงพอ',
    subtext: 'มีหุ้นคู่แข่งในกลุ่มเดียวกันน้อยกว่าเกณฑ์ที่ใช้ประเมิน Peer Relative Score',
    category: 'model_methodology',
  },

  // quant_engine.py
  'insufficient_periods:growth': {
    label: 'รอบข้อมูลงบการเงินย้อนหลังไม่เพียงพอ',
    subtext: 'ข้อมูลงบการเงินในอดีตมีน้อยกว่า 2 ช่วงเวลา ไม่สามารถคำนวณ YoY Growth ได้',
    category: 'critical_fallback',
  },
  'non_annual_period_gap:growth': {
    label: 'ระยะเวลาเปรียบเทียบงบการเงินไม่ใช่งวดปีเต็ม',
    subtext: 'งวดเวลาของข้อมูลเปรียบเทียบไม่อยู่ในรอบ 12 เดือนเต็ม',
    category: 'model_methodology',
  },

  // Institutional Upgrades (Market Data, Macro Linkage, DCF Bounds & Capital Allocation)
  'quote_ohlcv_mismatch:market_data': {
    label: 'ราคา Quote ล่าสุดไม่ตรงกับราคาปิด OHLCV',
    subtext: 'ราคาซื้อขายล่าสุดแตกต่างจากราคาปิดสิ้นวันของข้อมูลแท่งเทียน ระบบจึงตรึงการวิเคราะห์ไว้ที่ราคาปิดสิ้นวัน (EOD Close)',
    category: 'market_warning',
  },
  'share_dilution_risk:capital_allocation': {
    label: 'ความเสี่ยงการเกิด Share Dilution (การเจือจางหุ้น)',
    subtext: 'บริษัทมีการออกหุ้นเพิ่มทุนหรือหุ้นใหม่มากกว่าการซื้อหุ้นคืน ทำให้จำนวนหุ้นหมุนเวียนเพิ่มขึ้นต่อเนื่อง',
    category: 'market_warning',
  },
  'macro_regime_headwind:scorecard': {
    label: 'สภาวะมหภาคส่งผลต้าน (Macro Regime Headwind)',
    subtext: 'สภาวะมหภาค (อัตราดอกเบี้ย, ความผันผวน, หรือสภาพคล่อง) ไม่เอื้อต่อกลุ่มอุตสาหกรรมนี้ ทำให้มีการปรับลดคะแนนความเชื่อมั่น',
    category: 'market_warning',
  },
  'implied_growth_above_upper_bound:dcf': {
    label: 'อัตราเติบโตแฝง (Implied Growth) สูงกว่ากรอบบน',
    subtext: 'ราคาตลาดสะท้อนสมมติฐานการเติบโตที่สูงเกินกว่าช่วงปกติของการประเมินมูลค่า DCF',
    category: 'model_methodology',
  },
  'implied_growth_below_lower_bound:dcf': {
    label: 'อัตราเติบโตแฝง (Implied Growth) ต่ำกว่ากรอบล่าง',
    subtext: 'ราคาตลาดสะท้อนสมมติฐานการเติบโตที่ต่ำผิดปกติหรือมีส่วนลดมูลค่าสูงมาก',
    category: 'model_methodology',
  },
}

const STALE_REASON_LABELS: Record<string, string> = {
  insufficient_trading_history: 'ประวัติราคาซื้อขายไม่เพียงพอ',
  fetch_error: 'ดึงข้อมูลย้อนหลังไม่สำเร็จ',
  zero_benchmark_variance: 'ความผันผวนของดัชนีอ้างอิงเป็นศูนย์',
}

const METRIC_LABELS_TH: Record<string, string> = {
  beta: 'Beta',
  volatility: 'Volatility',
  mdd: 'Max Drawdown',
  technical_indicators: 'ตัวชี้วัดทางเทคนิค (Momentum)',
  price_percentile: 'Price Percentile',
}

export function parseDataQualityFlag(rawFlag: string): FormattedFlag {
  const known = KNOWN_FLAGS[rawFlag]
  if (known) {
    return {
      rawCode: rawFlag,
      label: known.label,
      subtext: known.subtext,
      category: known.category,
    }
  }

  // Fallback parser for dynamically formatted flags (e.g. "insufficient_trading_history:beta")
  const [code, domain] = rawFlag.split(':')
  const category: FlagCategory =
    rawFlag.includes('insufficient') || rawFlag.includes('unavailable') || rawFlag.includes('error')
      ? 'critical_fallback'
      : 'low_impact_fallback'

  if (code && domain && STALE_REASON_LABELS[code] && METRIC_LABELS_TH[domain]) {
    return {
      rawCode: rawFlag,
      label: `ไม่สามารถคำนวณ ${METRIC_LABELS_TH[domain]} ได้`,
      subtext: STALE_REASON_LABELS[code],
      category,
    }
  }

  const cleanCode = code ? code.replace(/_/g, ' ') : rawFlag
  return {
    rawCode: rawFlag,
    label: `${cleanCode}${domain ? ` (${domain.toUpperCase()})` : ''}`,
    category,
  }
}

