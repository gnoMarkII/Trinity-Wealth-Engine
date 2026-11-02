export type FavoritePromptCategory = 'macro' | 'equity' | 'portfolio' | 'workflows'

export interface FavoritePromptItem {
  id: string
  title: string
  dimensionBadge: string
  prompt: string
  category: FavoritePromptCategory
  contextRefs: string[]
  isPinned: boolean
  isCustom?: boolean
  requiresTicker?: boolean
}

export interface CreateFavoritePromptInput {
  title: string
  dimensionBadge?: string
  prompt: string
  category: FavoritePromptCategory
  contextRefs: string[]
  requiresTicker?: boolean
}

export const STORAGE_KEY = 'trinity-favorite-prompts'
export const FAVORITES_UPDATED_EVENT = 'trinity-favorites-updated'

export const DEFAULT_FAVORITE_PROMPTS: FavoritePromptItem[] = [
  // ==========================================
  // 🌐 หมวด 1: เศรษฐกิจมหภาค (Macro — 8 มิติ)
  // ==========================================
  {
    id: 'macro-regime-dalio',
    title: 'วิเคราะห์ Macro Regime & Dalio Cycle ปัจจุบัน',
    dimensionBadge: 'Dalio Cycle & Regime',
    prompt:
      'ช่วยวิเคราะห์ภาวะเศรษฐกิจมหภาคปัจจุบันในทุกมิติ ทั้ง Macro Regime, Dalio Long-Term Debt Cycle, Tactical Cycle และประเมินว่าตอนนี้ตลาดอยู่ในโหมด Risk-On หรือ Risk-Off พร้อมคำแนะนำ Stance ของพอร์ต',
    category: 'macro',
    contextRefs: ['@Macro', '@Portfolio'],
    isPinned: true,
  },
  {
    id: 'macro-us-yield-fed',
    title: 'เจาะลึก US Treasury Yield Curve, Fed & สภาพคล่อง',
    dimensionBadge: 'US Yield Curve & Fed',
    prompt:
      'วิเคราะห์โครงสร้าง US Treasury Yield Curve (2Y/10Y/30Y), ผลการประมูลพันธบัตร (Auction Demand), ดัชนีความตึงเครียดทางการเงิน (OFR Financial Stress) และทิศทางดอกเบี้ยนโยบาย Fed ว่าส่งผลต่อสินทรัพย์เสี่ยงอย่างไร',
    category: 'macro',
    contextRefs: ['@Macro'],
    isPinned: false,
  },
  {
    id: 'macro-thailand-flow',
    title: 'วิเคราะห์เศรษฐกิจไทย, Fund Flow ต่างชาติ & SET Valuation',
    dimensionBadge: 'Thailand Fund Flow',
    prompt:
      'สรุปภาวะเศรษฐกิจและตลาดทุนไทย: ทิศทาง Fund Flow ต่างชาติ (หุ้นและพันธบัตร), Thai Yield Curve, หนี้สาธารณะไทย, Market Breadth และระดับ Valuation ของตลาดหุ้นไทยเทียบกับค่าเฉลี่ยในอดีต',
    category: 'macro',
    contextRefs: ['@Macro'],
    isPinned: false,
  },
  {
    id: 'macro-cross-border-gold',
    title: 'กระแสเงินทุนโลก (Cross-Border), ทองคำ & ค่าเงินบาท',
    dimensionBadge: 'Cross-Border & Gold',
    prompt:
      'วิเคราะห์การเคลื่อนย้ายเงินทุนระหว่างประเทศ (Cross-Border Flows), สถานะการถือครองโลหะมีค่า COT (Gold/Silver), ความผันผวนของสินค้าโภคภัณฑ์ (Commodity Volatility) และราคาทองคำในประเทศเทียบตลาดโลก',
    category: 'macro',
    contextRefs: ['@Macro'],
    isPinned: false,
  },
  {
    id: 'macro-global-liquidity-crypto',
    title: 'สภาพคล่องมหภาค (Global Liquidity) & Crypto Pulse',
    dimensionBadge: 'Global Liquidity & M2',
    prompt:
      'ตรวจสอบสภาพคล่องมหภาค (Macro Liquidity), มวลเงิน M2, Stablecoin Flow และความสัมพันธ์ระหว่างสภาพคล่องโลกกับสินทรัพย์เสี่ยงและกลุ่มเทคโนโลยี/คริปโต',
    category: 'macro',
    contextRefs: ['@Macro'],
    isPinned: false,
  },
  {
    id: 'macro-calendar-event-risk',
    title: 'ปฏิทินเศรษฐกิจสำคัญ & การเตรียมรับมือ Event Risk',
    dimensionBadge: 'Economic Calendar',
    prompt:
      'ตรวจสอบเหตุการณ์เศรษฐกิจสำคัญที่กำลังจะมาถึง (FOMC, CPI, Nonfarm Payrolls, กนง.) พร้อมประเมิน Scenario Analysis และวิธีป้องกันความเสี่ยง (Hedging) ของพอร์ต',
    category: 'macro',
    contextRefs: ['@Macro', '@Portfolio'],
    isPinned: false,
  },
  {
    id: 'macro-daily-digest-impact',
    title: 'สรุปข่าว Daily Digest วันนี้ & ผลกระทบต่อพอร์ต',
    dimensionBadge: 'Daily Digest & Impact',
    prompt:
      'ดึงข้อมูลสรุปข่าวการเงินรายวัน (Daily Digest) ล่าสุด วิเคราะห์ Market Sentiment และชี้เป้าว่าข่าวไหนส่งผลกระทบโดยตรงต่อสินทรัพย์ในพอร์ตของเราบ้าง',
    category: 'macro',
    contextRefs: ['@Digest', '@Portfolio'],
    isPinned: true,
  },
  {
    id: 'macro-run-research-job',
    title: '⚡ [สั่งรัน] อัปเดตรายงานวิจัย Macro Strategy ฉบับเต็ม',
    dimensionBadge: 'Run Macro Intel',
    prompt:
      'ช่วยสั่งรันงานวิจัยอัปเดตภาวะเศรษฐกิจมหภาคและการจัดสรรสินทรัพย์ (macro_intel) ฉบับล่าสุดเข้าคิวประมวลผลให้หน่อย',
    category: 'macro',
    contextRefs: ['@Macro'],
    isPinned: false,
  },

  // ==========================================
  // 🔬 หมวด 2: หุ้นรายตัว (Equity 5-Tier — 9 มิติ)
  // ==========================================
  {
    id: 'equity-5tier-full',
    title: 'เจาะลึกหุ้น [TICKER] ครบ 5 มิติ (5-Tier Institutional Thesis)',
    dimensionBadge: '5-Tier Full Thesis',
    prompt:
      'ดึงบทวิเคราะห์เชิงลึกของหุ้น [TICKER] ครบทั้ง 5 มิติ: (1) Executive Thesis & Moat, (2) Quality & 4-Pillar Balance, (3) Financial Health & Capital Allocation, (4) Valuation Spectrum & Margin of Safety, และ (5) Smart Money & Tactical Stage',
    category: 'equity',
    contextRefs: [],
    isPinned: true,
    requiresTicker: true,
  },
  {
    id: 'equity-valuation-dcf',
    title: 'ประเมิน Valuation, DCF Sensitivity & จุดเข้าซื้อหุ้น [TICKER]',
    dimensionBadge: 'Valuation & Margin of Safety',
    prompt:
      'วิเคราะห์ Valuation ของหุ้น [TICKER] เชิงลึก: เปรียบเทียบราคาปัจจุบันกับ Valuation Spectrum (Bear / Base / Bull Case), ประเมิน Margin of Safety, อัตราผลตอบแทนคาดหวัง (Price Return Decomposition) และระดับราคาที่เหมาะสมในการสะสมเพิ่ม',
    category: 'equity',
    contextRefs: [],
    isPinned: true,
    requiresTicker: true,
  },
  {
    id: 'equity-expectation-gap',
    title: 'วิเคราะห์ Expectation Gap ของหุ้น [TICKER] (ตลาดคาดหวังอะไร?)',
    dimensionBadge: 'Expectation Gap',
    prompt:
      'วิเคราะห์ Expectation Gap ของหุ้น [TICKER]: ราคาหุ้นปัจจุบันสะท้อนความคาดหวังการเติบโต (Implied Growth) ไว้สูงแค่ไหน มีโอกาสเกิด Positive หรือ Negative Surprise จากปัจจัยใดบ้าง',
    category: 'equity',
    contextRefs: [],
    isPinned: false,
    requiresTicker: true,
  },
  {
    id: 'equity-promise-earnings',
    title: 'ตรวจสอบ Promise Tracker & สรุป Earnings Call หุ้น [TICKER]',
    dimensionBadge: 'Promise & Earnings',
    prompt:
      'ตรวจสอบความน่าเชื่อถือของผู้บริหารหุ้น [TICKER] จาก Promise Tracker (สัญญาที่เคยให้ไว้ในอดีตทำได้จริงหรือไม่) พร้อมสรุปประเด็นสำคัญจาก Earnings Call และแนวโน้ม Guidance ล่าสุด',
    category: 'equity',
    contextRefs: [],
    isPinned: false,
    requiresTicker: true,
  },
  {
    id: 'equity-kill-switch-review',
    title: 'ตรวจเช็ก Kill-Switch & สุขภาพ Thesis ของหุ้น [TICKER]',
    dimensionBadge: 'Kill-Switch Proximity',
    prompt:
      'ตรวจสอบ Kill-Switch Proximity และ Data Quality Flags ของหุ้น [TICKER]: ปัจจัยพื้นฐานหรือคูเมือง (Moat) มีรอยร้าวหรือไม่ มีสัญญาณเตือนภัยใดที่เข้าใกล้จุดต้องทบทวนการถือครอง (Position Review)',
    category: 'equity',
    contextRefs: ['@Portfolio'],
    isPinned: false,
    requiresTicker: true,
  },
  {
    id: 'equity-smart-money-stage',
    title: 'แกะรอย Smart Money, Insider Trading & Tactical Stage หุ้น [TICKER]',
    dimensionBadge: 'Smart Money & Stage',
    prompt:
      'วิเคราะห์การเคลื่อนไหวของ Smart Money ในหุ้น [TICKER]: การซื้อขายของผู้บริหาร (Form 4 / Insider), การถือครองของสถาบัน, วัฏจักรราคา (Tactical Stage 1-4) และแนวรับ-แนวต้านสำคัญ',
    category: 'equity',
    contextRefs: [],
    isPinned: false,
    requiresTicker: true,
  },
  {
    id: 'equity-terminal-screener',
    title: 'เปรียบเทียบหุ้น [TICKER] กับคู่แข่งในอุตสาหกรรม (Screener)',
    dimensionBadge: 'Terminal V2 Screener',
    prompt:
      'ช่วยใช้เครื่องมือ Terminal V2 Screener เปรียบเทียบปัจจัยพื้นฐาน (Valuation, Quality, ROIC, Margin, Momentum) ของหุ้น [TICKER] และคู่แข่งสำคัญในกลุ่มเดียวกัน เพื่อหาตัวที่แข็งแกร่งและคุ้มค่าที่สุด',
    category: 'equity',
    contextRefs: [],
    isPinned: false,
    requiresTicker: true,
  },
  {
    id: 'equity-position-sizing',
    title: 'คำนวณขนาดไม้ (Position Sizing) หากจะซื้อหุ้น [TICKER] เข้าพอร์ต',
    dimensionBadge: 'Position Sizing & Sandbox',
    prompt:
      'หากต้องการลงทุนในหุ้น [TICKER] ช่วยตรวจสอบกับพอร์ตปัจจุบันและกฎ 3 ถังเงิน (@Essence) ว่าควรจัดอยู่ในถังใด ซื้อได้ในสัดส่วนกี่เปอร์เซ็นต์ของพอร์ต และแบ่งไม้ซื้ออย่างไรให้ปลอดภัยที่สุด (ใช้ Python Sandbox คำนวณตัวเลขเป๊ะๆ)',
    category: 'equity',
    contextRefs: ['@Portfolio', '@Essence'],
    isPinned: false,
    requiresTicker: true,
  },
  {
    id: 'equity-run-research-job',
    title: '⚡ [สั่งรัน] สั่งทีม AI วิเคราะห์หุ้น [TICKER] เชิงลึกใหม่ทั้งหมด',
    dimensionBadge: 'Run Equity Intel',
    prompt:
      'ช่วยสั่งรันงานวิจัยวิเคราะห์หุ้นรายตัวเชิงลึก (equity_intel) สำหรับหุ้น [TICKER] เข้าคิวประมวลผลให้หน่อย',
    category: 'equity',
    contextRefs: [],
    isPinned: false,
    requiresTicker: true,
  },

  // ==========================================
  // 💼 หมวด 3: พอร์ตโฟลิโอ & กลยุทธ์ (Portfolio — 6 มิติ)
  // ==========================================
  {
    id: 'portfolio-health-rebalance',
    title: 'ตรวจสุขภาพพอร์ต All-Asset & คำแนะนำ Rebalance',
    dimensionBadge: 'Health & Rebalance',
    prompt:
      'สรุปภาพรวมพอร์ตการลงทุนปัจจุบัน (NAV, P&L, สัดส่วนสินทรัพย์, เงินสด) พร้อมตรวจสอบ Drift Alerts ว่ามีสินทรัพย์ไหนน้ำหนักเกินหรือขาดจากเป้าหมาย และแนะนำแผนการ Rebalance ที่ชัดเจน',
    category: 'portfolio',
    contextRefs: ['@Portfolio', '@Macro'],
    isPinned: true,
  },
  {
    id: 'portfolio-look-through',
    title: 'วิเคราะห์ Look-Through พอร์ต (เช็กความซ้ำซ้อนของหุ้นและกองทุน)',
    dimensionBadge: 'Look-Through Holdings',
    prompt:
      'วิเคราะห์ Look-Through Holdings ของพอร์ตทั้งหมด เพื่อดูว่าเมื่อรวมกองทุนรวม, PVD, ETF และหุ้นรายตัวแล้ว พอร์ตของเรากระจุกตัวอยู่ในหุ้น กลุ่มอุตสาหกรรม หรือประเทศใดมากเกินไปหรือไม่',
    category: 'portfolio',
    contextRefs: ['@Portfolio'],
    isPinned: false,
  },
  {
    id: 'portfolio-3buckets-5pillars',
    title: 'ตรวจสอบพอร์ตเทียบกับกฎ 5 เสาหลัก & แผนเงิน 3 ถัง',
    dimensionBadge: '5-Pillars & 3-Buckets',
    prompt:
      'ดึงข้อมูลกฎการลงทุน 5 เสาหลักและแผนเงิน 3 ถัง (สภาพคล่อง / พอร์ตหลัก / เติบโต) มาตรวจสอบเทียบกับพอร์ตจริงในปัจจุบัน ว่าเรายังรักษาวินัยตามแผนหรือไม่ และควรปรับปรุงจุดใด',
    category: 'portfolio',
    contextRefs: ['@Portfolio', '@Essence'],
    isPinned: true,
  },
  {
    id: 'portfolio-opportunity-radar',
    title: 'สแกนหาโอกาสลงทุนที่ดีที่สุดตอนนี้ (Margin of Safety สูง)',
    dimensionBadge: 'Opportunity Radar',
    prompt:
      'ตรวจสอบโอกาสการลงทุน (Portfolio Opportunities) และหุ้นที่ติดตามอยู่ ว่าภายใต้ภาวะเศรษฐกิจมหภาคปัจจุบัน มีสินทรัพย์หรือหุ้นตัวไหนที่ราคาลงมาอยู่ในโซนปลอดภัย (Margin of Safety สูง) น่าทยอยสะสมบ้าง',
    category: 'portfolio',
    contextRefs: ['@Portfolio', '@Macro'],
    isPinned: false,
  },
  {
    id: 'portfolio-stress-test-sandbox',
    title: 'จำลอง Stress Test พอร์ตหากเกิดวิกฤตตลาดหรือดอกเบี้ยผันผวน',
    dimensionBadge: 'Python Stress Test',
    prompt:
      'ช่วยรันการคำนวณใน Python Sandbox เพื่อทำ Stress Test พอร์ตปัจจุบันใน 3 สถานการณ์: (1) ตลาดหุ้นโลกปรับฐาน -20%, (2) เงินบาทแข็งค่า/อ่อนค่า 10%, และ (3) อัตราดอกเบี้ยค้างสูงนาน เพื่อประเมิน Drawdown สูงสุดของพอร์ต',
    category: 'portfolio',
    contextRefs: ['@Portfolio', '@Macro'],
    isPinned: false,
  },
  {
    id: 'portfolio-dca-deployment',
    title: 'วางแผนจัดสรรเงินสดใหม่ (Cash Deployment / DCA) ประจำเดือน',
    dimensionBadge: 'DCA & Cash Deployment',
    prompt:
      'ช่วยวางแผนการจัดสรรเงินลงทุนใหม่ประจำเดือนนี้ โดยอ้างอิงจากสัดส่วนพอร์ตปัจจุบัน (@Portfolio) และภาวะเศรษฐกิจมหภาค (@Macro) ว่าควรเติมเงินเข้าถังไหนและสินทรัพย์ใดเป็นลำดับแรก',
    category: 'portfolio',
    contextRefs: ['@Portfolio', '@Macro', '@Essence'],
    isPinned: false,
  },

  // ==========================================
  // 🚀 หมวด 4: สั่งรันงานวิจัย & คลังความรู้ (Workflows — 5 มิติ)
  // ==========================================
  {
    id: 'workflow-news-youtube',
    title: '📺 [สั่งรัน] ดึงข่าวการเงินและสรุปคลิป YouTube การลงทุนล่าสุด',
    dimensionBadge: 'News & YouTube Sync',
    prompt:
      'ช่วยสั่งรันโฟลว์ดึงข้อมูลข่าวสารและสรุปคลิป YouTube ล่าสุด (news_youtube) เข้าบอร์ดงานให้หน่อย',
    category: 'workflows',
    contextRefs: ['@Digest'],
    isPinned: false,
  },
  {
    id: 'workflow-news-funnel',
    title: '🌪️ [สั่งรัน] คัดกรองเหตุการณ์สำคัญเข้าสู่ News Funnel',
    dimensionBadge: 'News Funnel Filter',
    prompt:
      'ช่วยสั่งรันงานคัดกรองข่าวและเหตุการณ์เศรษฐกิจสำคัญ (news_funnel) เพื่อเตรียมอนุมัติในบอร์ดงาน',
    category: 'workflows',
    contextRefs: ['@Digest'],
    isPinned: false,
  },
  {
    id: 'workflow-youtube-pitch',
    title: '🎬 [สั่งรัน] หาไอเดียและสร้างโครงเรื่องทำคลิป YouTube (Pitch)',
    dimensionBadge: 'YouTube Pitch Idea',
    prompt:
      'ช่วยสั่งรันงานวิเคราะห์และสร้างหัวข้อพิทช์เนื้อหา YouTube เชิงลึก (youtube_pitch) จากข่าวและภาวะตลาดล่าสุด',
    category: 'workflows',
    contextRefs: ['@Macro', '@Digest'],
    isPinned: false,
  },
  {
    id: 'workflow-search-vault',
    title: '📚 ค้นหาบันทึกความรู้และบทเรียนการลงทุนใน Obsidian Vault',
    dimensionBadge: 'Obsidian Vault Search',
    prompt:
      'ช่วยค้นหาบันทึกใน Knowledge Vault เกี่ยวกับหลักการประเมินมูลค่าหุ้นและการบริหารความเสี่ยงในช่วงตลาดผันผวน พร้อมสรุปประเด็นสำคัญให้หน่อย',
    category: 'workflows',
    contextRefs: ['@Essence'],
    isPinned: false,
  },
  {
    id: 'workflow-save-vault',
    title: '💾 สรุปประเด็นสำคัญจากการสนทนานี้บันทึกลง Obsidian Vault',
    dimensionBadge: 'Save Insight to Vault',
    prompt:
      'ช่วยสรุปข้อคิดเห็น กลยุทธ์ และข้อสรุปสำคัญจากการสนทนาของเราในรอบนี้ แล้วบันทึกเก็บเข้าสู่ Knowledge Vault (Obsidian) ให้เรียบร้อย',
    category: 'workflows',
    contextRefs: [],
    isPinned: false,
  },
]

/**
 * โหลดรายการ Favorite Prompts ทั้งหมด โดยผสาน Defaults เข้ากับค่า custom/pin ใน localStorage
 */
export function loadFavoritePrompts(): FavoritePromptItem[] {
  try {
    const raw = typeof window !== 'undefined' ? window.localStorage.getItem(STORAGE_KEY) : null
    if (!raw) return DEFAULT_FAVORITE_PROMPTS
    const parsed = JSON.parse(raw)
    if (!Array.isArray(parsed)) return DEFAULT_FAVORITE_PROMPTS

    // สร้าง Map จาก stored items เพื่อดูการแก้ไขหรือ custom items
    const storedMap = new Map<string, FavoritePromptItem>()
    const customItems: FavoritePromptItem[] = []

    for (const item of parsed) {
      if (!item || !item.id) continue
      if (item.isCustom) {
        customItems.push(item)
      } else {
        storedMap.set(item.id, item)
      }
    }

    // ผสาน default items กับ stored pin status
    const mergedDefaults = DEFAULT_FAVORITE_PROMPTS.map((def) => {
      const stored = storedMap.get(def.id)
      if (stored) {
        return {
          ...def,
          isPinned: typeof stored.isPinned === 'boolean' ? stored.isPinned : def.isPinned,
        }
      }
      return def
    })

    return [...mergedDefaults, ...customItems]
  } catch {
    return DEFAULT_FAVORITE_PROMPTS
  }
}

/**
 * บันทึกรายการ Favorite Prompts ลงใน localStorage และยิง CustomEvent แจ้งเตือนทุกคอมโพเนนต์
 */
export function saveFavoritePrompts(prompts: FavoritePromptItem[]): void {
  try {
    if (typeof window !== 'undefined') {
      window.localStorage.setItem(STORAGE_KEY, JSON.stringify(prompts))
      window.dispatchEvent(new CustomEvent(FAVORITES_UPDATED_EVENT, { detail: prompts }))
    }
  } catch {
    // Ignore storage quota or environment issues
  }
}

/**
 * สลับสถานะ Pin ของคำสั่งโปรด
 */
export function togglePinFavoritePrompt(id: string): FavoritePromptItem[] {
  const current = loadFavoritePrompts()
  const updated = current.map((item) => (item.id === id ? { ...item, isPinned: !item.isPinned } : item))
  saveFavoritePrompts(updated)
  return updated
}

/**
 * เพิ่มคำสั่งโปรดที่ผู้ใช้สร้างขึ้นเอง (Custom Prompt)
 */
export function addCustomFavoritePrompt(input: CreateFavoritePromptInput): FavoritePromptItem[] {
  const current = loadFavoritePrompts()
  const newItem: FavoritePromptItem = {
    id: `custom-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
    title: input.title.trim(),
    dimensionBadge: input.dimensionBadge?.trim() || 'Custom Prompt',
    prompt: input.prompt.trim(),
    category: input.category,
    contextRefs: input.contextRefs,
    isPinned: true,
    isCustom: true,
    requiresTicker: Boolean(input.requiresTicker || input.prompt.includes('[TICKER]')),
  }
  const updated = [newItem, ...current]
  saveFavoritePrompts(updated)
  return updated
}

/**
 * แก้ไขคำสั่งโปรดที่ผู้ใช้สร้างเอง
 */
export function updateFavoritePrompt(id: string, input: CreateFavoritePromptInput): FavoritePromptItem[] {
  const current = loadFavoritePrompts()
  const updated = current.map((item) => {
    if (item.id === id) {
      return {
        ...item,
        title: input.title.trim(),
        dimensionBadge: input.dimensionBadge?.trim() || item.dimensionBadge,
        prompt: input.prompt.trim(),
        category: input.category,
        contextRefs: input.contextRefs,
        requiresTicker: Boolean(input.requiresTicker || input.prompt.includes('[TICKER]')),
      }
    }
    return item
  })
  saveFavoritePrompts(updated)
  return updated
}

/**
 * ลบคำสั่งโปรด (เฉพาะที่ผู้ใช้สร้างขึ้นเอง หรือลบออกจากรายการ)
 */
export function deleteFavoritePrompt(id: string): FavoritePromptItem[] {
  const current = loadFavoritePrompts()
  const updated = current.filter((item) => item.id !== id)
  saveFavoritePrompts(updated)
  return updated
}

/**
 * รีเซ็ตกลับเป็นค่าเริ่มต้น 28 คำสั่งมาตรฐาน
 */
export function resetDefaultFavoritePrompts(): FavoritePromptItem[] {
  saveFavoritePrompts(DEFAULT_FAVORITE_PROMPTS)
  return DEFAULT_FAVORITE_PROMPTS
}

/**
 * แปลงข้อความคำสั่งและ Context Refs โดยแทนที่ [TICKER] ด้วยชื่อหุ้นจริง
 */
export function resolvePromptWithTicker(
  item: FavoritePromptItem,
  tickerInput: string = 'NVDA'
): { title: string; prompt: string; contextRefs: string[] } {
  const cleanTicker = tickerInput.trim().toUpperCase() || 'NVDA'
  const resolvedTitle = item.title.replace(/\[TICKER\]/g, cleanTicker)
  const resolvedPrompt = item.prompt.replace(/\[TICKER\]/g, cleanTicker)

  const refs = new Set<string>(item.contextRefs)
  if (item.requiresTicker || item.prompt.includes('[TICKER]') || item.category === 'equity') {
    refs.add(`$${cleanTicker}`)
  }

  return {
    title: resolvedTitle,
    prompt: resolvedPrompt,
    contextRefs: Array.from(refs),
  }
}

