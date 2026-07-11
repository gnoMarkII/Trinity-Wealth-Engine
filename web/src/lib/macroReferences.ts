import type { MacroDashboardDTO } from '../api/types'

export type IndicatorCategory =
  | 'Volatility'
  | 'Credit'
  | 'Rates & Liquidity'
  | 'Commodities & Energy'
  | 'Macro & Equities'

export interface IndicatorMetadata {
  id: string
  name: string
  category: IndicatorCategory
  description: string
  sourceProvider: string
  extractedValue?: string | null
}

export interface SourceFileMetadata {
  filename: string
  title: string
  type: 'report' | 'quant_engine'
  description: string
  badgeText: string
}

const KNOWN_INDICATORS: Record<string, Omit<IndicatorMetadata, 'id' | 'extractedValue'>> = {
  obs_global_macro_snapshot_vix_20260709: {
    name: 'VIX Index (CBOE Volatility Index)',
    category: 'Volatility',
    description:
      'ดัชนีวัดความกลัวและความผันผวนคาดการณ์ในตลาดหุ้นสหรัฐฯ (S&P 500 Implied Volatility 30 วัน)',
    sourceProvider: 'CBOE / Market Feed',
  },
  obs_ratio_hyg_lqd: {
    name: 'HYG/LQD Ratio (Credit Risk Premium Proxy)',
    category: 'Credit',
    description:
      'อัตราส่วนเปรียบเทียบระหว่างตราสารหนี้ High Yield (HYG) กับ Investment Grade (LQD) บ่งบอกความเต็มใจรับความเสี่ยงเครดิตในตลาด',
    sourceProvider: 'iShares / Bond Market Proxy',
  },
  obs_global_macro_snapshot_cl_f_20260709: {
    name: 'WTI Crude Oil Futures (CL=F)',
    category: 'Commodities & Energy',
    description:
      'ราคาน้ำมันดิบ WTI อ้างอิงความเสี่ยงด้านอุปทานพลังงานและความตึงเครียดทางภูมิรัฐศาสตร์',
    sourceProvider: 'NYMEX / Commodities Feed',
  },
  obs_us10y: {
    name: 'US 10-Year Treasury Yield (US10Y)',
    category: 'Rates & Liquidity',
    description:
      'อัตราผลตอบแทนพันธบัตรรัฐบาลสหรัฐฯ อายุ 10 ปี เป็นเกณฑ์วัดต้นทุนทางการเงินและ Risk-free Rate หลักของโลก',
    sourceProvider: 'FRED / US Treasury',
  },
  obs_sp500_erp: {
    name: 'S&P 500 Equity Risk Premium (ERP)',
    category: 'Macro & Equities',
    description:
      'ส่วนต่างผลตอบแทนคาดหวังของตลาดหุ้นเมื่อเปรียบเทียบกับผลตอบแทนพันธบัตรรัฐบาล บ่งบอกถึงความคุ้มค่าด้าน Valuation',
    sourceProvider: 'Quant Engine / Bloomberg',
  },
}

export function enrichIndicator(id: string, data?: MacroDashboardDTO): IndicatorMetadata {
  const normalizedId = id.trim()

  // 0. ข้อมูลจริงจาก dashboard_indicators (DTO ที่ backend validate แล้ว) สำคัญกว่า
  // ตาราง hardcode/regex ทั้งหมดด้านล่าง — ใช้ label/ค่า/ที่มาจากตรงนั้นก่อนเสมอ
  // (ตาราง KNOWN_INDICATORS กับ pattern match เหลือไว้เป็น fallback สำหรับ id ที่
  // อ้างถึงใน observable_refs แต่ไม่ได้อยู่ในชุด indicator ของ dashboard)
  const dashboardIndicator = data?.dashboard_indicators?.find(
    (indicator) => indicator.indicator_id === normalizedId
  )

  // 1. Check known list or pattern match
  let meta = KNOWN_INDICATORS[normalizedId]

  if (!meta) {
    const lower = normalizedId.toLowerCase()
    if (lower.includes('vix')) {
      meta = {
        name: 'VIX Volatility Index',
        category: 'Volatility',
        description: 'ดัชนีวัดความผันผวนและระดับความกลัวของนักลงทุนในตลาดหุ้น',
        sourceProvider: 'CBOE Feed',
      }
    } else if (lower.includes('hyg') || lower.includes('lqd')) {
      meta = {
        name: 'Credit Risk Spread Proxy (HYG/LQD)',
        category: 'Credit',
        description: 'เครื่องชี้วัดความเต็มใจรับความเสี่ยงด้านเครดิตตราสารหนี้เอกชน',
        sourceProvider: 'Fixed Income Market',
      }
    } else if (lower.includes('cl_f') || lower.includes('oil') || lower.includes('crude')) {
      meta = {
        name: 'Crude Oil Price Benchmark',
        category: 'Commodities & Energy',
        description: 'ระดับราคาน้ำมันดิบโลก บ่งชี้ความเสี่ยงเงินเฟ้อและปัญหาอุปทานพลังงาน',
        sourceProvider: 'Commodities Exchange',
      }
    } else if (lower.includes('us10y') || lower.includes('treasury') || lower.includes('yield')) {
      meta = {
        name: 'US Treasury Yield Benchmark',
        category: 'Rates & Liquidity',
        description: 'ระดับอัตราดอกเบี้ยพันธบัตรรัฐบาลระยะยาว',
        sourceProvider: 'FRED Feed',
      }
    } else {
      meta = {
        name: normalizedId.replace(/^obs_/i, '').replace(/_/g, ' ').toUpperCase(),
        category: 'Macro & Equities',
        description: 'ตัวชี้วัดสภาวะเศรษฐกิจและตัวแปรแบบจำลองมหภาค',
        sourceProvider: 'Economic Data Feed',
      }
    }
  }

  const extractedValue =
    dashboardIndicator?.display_value || (data ? extractIndicatorValue(normalizedId, data) : null)

  return {
    id: normalizedId,
    name: dashboardIndicator?.label || meta.name,
    category: meta.category,
    description: meta.description,
    sourceProvider: dashboardIndicator?.provider || meta.sourceProvider,
    extractedValue,
  }
}

export function extractIndicatorValue(id: string, data: MacroDashboardDTO): string | null {
  const lower = id.toLowerCase()

  // Scan all regime evidence snippets
  const evidenceList = data.regime_evidence?.map((re) => re.evidence || '') || []
  const supportList: string[] = []

  data.asset_allocation?.forEach((a) => {
    if (a.supporting_data) supportList.push(...a.supporting_data)
  })
  data.pair_trades?.forEach((pt) => {
    if (pt.supporting_data) supportList.push(...pt.supporting_data)
  })

  const combinedTexts = [...evidenceList, ...supportList]

  for (const text of combinedTexts) {
    if (lower.includes('vix')) {
      const match = text.match(/VIX Index\s*=\s*([\d.]+)/i) || text.match(/VIX\s*=\s*([\d.]+)/i)
      if (match) return `${match[1]} pts`
    }
    if (lower.includes('hyg') || lower.includes('lqd')) {
      const match = text.match(/HYG\/LQD Ratio\s*=\s*([\d.]+)/i) || text.match(/HYG\/LQD\s*=\s*([\d.]+)/i)
      if (match?.[1]) return match[1]
    }
    if (lower.includes('cl_f') || lower.includes('oil') || lower.includes('crude')) {
      const match =
        text.match(/ราคาน้ำมันดิบ\s*=\s*([\d.]+\s*USD\/bbl)/i) ||
        text.match(/Oil\s*=\s*(\$?[\d.]+)/i)
      if (match?.[1]) return match[1]
    }
  }

  // Fallbacks if indicator matches asset supporting data
  if (lower.includes('sp500') || lower.includes('erp')) {
    for (const item of supportList) {
      if (item.includes('Forward Earnings Yield')) return item
      if (item.includes('Equity Risk Premium')) return item
    }
  }

  return null
}

export function enrichSourceFile(filename: string): SourceFileMetadata {
  const isPython = filename.endsWith('.py')

  if (isPython) {
    let title = filename
    let description = 'สคริปต์คำนวณและประมวลผลโมเดลเชิงปริมาณ (Quantitative Analytical Engine)'

    if (filename === 'valuation.py') {
      title = 'Quantitative Equity Valuation Engine (valuation.py)'
      description =
        'โมเดลคำนวณมูลค่าหุ้นเปรียบเทียบ (Forward Earnings Yield vs Bond Yield) และประเมินค่า Equity Risk Premium ของตลาด S&P 500'
    } else if (filename.includes('regime')) {
      title = 'Macro Regime Probability Engine'
      description = 'แบบจำลองคำนวณความน่าจะเป็นของวัฏจักรเศรษฐกิจ 4 สภาวะ (Recovery, Expansion, Stagflation, Recession)'
    }

    return {
      filename,
      title,
      type: 'quant_engine',
      description,
      badgeText: 'Quant Engine (.py)',
    }
  }

  // Report markdown
  let title = filename.replace(/\.md$/i, '').replace(/_/g, ' ')
  let description = 'รายงานวิเคราะห์และสแนปชอตสภาวะเศรษฐกิจในฐานข้อมูล Obsidian PKM'

  if (filename.includes('Global_Macro_Snapshot')) {
    title = 'รายงานสรุปภาพรวมเศรษฐกิจโลก (Global Macro Snapshot)'
    description =
      'บันทึกสแนปชอตตัวแปรมหภาคโลก ดัชนีความผันผวน สภาพคล่อง และราคาสินค้าโภคภัณฑ์ประจำงวด'
  } else if (filename.includes('Country_Macro_Snapshot')) {
    title = 'รายงานเศรษฐกิจรายประเทศและนโยบายการเงิน (Country Macro Snapshot)'
    description =
      'การวิเคราะห์นโยบายการเงินของธนาคารกลาง (Fed, ธปท.) และแนวโน้มเศรษฐกิจรายประเทศ'
  }

  return {
    filename,
    title,
    type: 'report',
    description,
    badgeText: 'Research Report (.md)',
  }
}
