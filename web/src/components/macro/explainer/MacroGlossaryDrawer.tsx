import { useState, useMemo } from 'react'
import { useFocusTrap } from '../../../hooks/useFocusTrap'

export interface GlossaryTerm {
  term: string
  thaiTerm: string
  category: 'Regime & Policy' | 'Bonds & Rates' | 'Equities & Valuation' | 'Portfolio & Risk' | 'Flows & Crypto'
  icon: string
  definition: string
  howToUse: string
}

export const GLOSSARY_TERMS: GlossaryTerm[] = [
  // 1. Regime & Policy
  {
    term: 'Goldilocks Economy',
    thaiTerm: 'ภาวะเศรษฐกิจโกลดิล็อกส์ ☀️',
    category: 'Regime & Policy',
    icon: '☀️',
    definition: 'เศรษฐกิจเติบโตได้ดีในอัตรากำลังพอดี ขณะที่เงินเฟ้ออยู่ในระดับต่ำและดอกเบี้ยนิ่ง ไม่ร้อนแรงหรือเย็นจนเกินไป',
    howToUse: 'เป็นสภาวะที่เป็นมิตรต่อตลาดหุ้นมากที่สุด ทยอยสะสมหุ้นเติบโตและสินทรัพย์เสี่ยงได้เต็มที่',
  },
  {
    term: 'Reflation',
    thaiTerm: 'ภาวะเศรษฐกิจฟื้นตัวพร้อมเงินเฟ้อ 🌤️',
    category: 'Regime & Policy',
    icon: '🌤️',
    definition: 'เศรษฐกิจเริ่มฟื้นตัวจากจุดต่ำสุด เม็ดเงินหมุนเวียนเพิ่มขึ้น แต่เริ่มมีเงินเฟ้อและแรงกดดันดอกเบี้ยตามมา',
    howToUse: 'หุ้นวัฏจักรและสินค้าโภคภัณฑ์มักได้ประโยชน์ ห้ามไล่ราคาหุ้นแพง ควรแบ่งเงินสดสำรอง 20% เพื่อรอจังหวะย่อตัว',
  },
  {
    term: 'Stagflation',
    thaiTerm: 'ภาวะเศรษฐกิจชะงักงันพร้อมของแพง ⛈️',
    category: 'Regime & Policy',
    icon: '⛈️',
    definition: 'สภาวะเลวร้ายที่เศรษฐกิจโตช้าหรือถดถอย คนว่างงานสูง แต่ราคาสินค้าและเงินเฟ้อยังพุ่งไม่หยุด',
    howToUse: 'ตลาดหุ้นมักถูกกดดันหนัก ห้ามทุ่มเงินในหุ้น ควรโยกเงินไปพักที่ทองคำ พันธบัตรระยะสั้น และถือเงินสดเป็นเกราะกำบัง',
  },
  {
    term: 'Recession',
    thaiTerm: 'ภาวะเศรษฐกิจถดถอย 🌧️',
    category: 'Regime & Policy',
    icon: '🌧️',
    definition: 'เศรษฐกิจหดตัวต่อเนื่อง รายได้และกำไรบริษัทลดลง การบริโภคซบเซา ธนาคารกลางมีแนวโน้มต้องลดดอกเบี้ยเพื่อกระตุ้น',
    howToUse: 'หุ้นมีความเสี่ยงสูง พันธบัตรรัฐบาลระยะยาวและเงินสดคือที่หลบภัยที่ดีที่สุดเพื่อรับกำไรจากดอกเบี้ยขาลง',
  },
  {
    term: 'Fed Funds Rate',
    thaiTerm: 'อัตราดอกเบี้ยนโยบายของสหรัฐฯ 🏛️',
    category: 'Regime & Policy',
    icon: '🏛️',
    definition: 'อัตราดอกเบี้ยอ้างอิงสูงสุดที่ธนาคารกลางสหรัฐฯ (Fed) ใช้ควบคุมปริมาณเงินและเงินเฟ้อในระบบเศรษฐกิจโลก',
    howToUse: 'เมื่อ Fed ขึ้นดอกเบี้ย เงินสดและพันธบัตรจะน่าดึงดูดขึ้น หุ้นมักเผชิญแรงขาย เมื่อ Fed ลดดอกเบี้ย สภาพคล่องจะไหลกลับเข้าสินทรัพย์เสี่ยง',
  },
  {
    term: 'Quantitative Tightening (QT)',
    thaiTerm: 'การดูดซับสภาพคล่องของธนาคารกลาง 📉',
    category: 'Regime & Policy',
    icon: '📉',
    definition: 'กระบวนการที่ Fed ปล่อยให้พันธบัตรในงบดุลหมดอายุโดยไม่ซื้อทดแทน เป็นการดึงเงินสดออกจากระบบการเงิน',
    howToUse: 'ทำให้สภาพคล่องในตลาดการเงินลดลง สถาบันการเงินจะระมัดระวังการลงทุนและปล่อยสินเชื่อมากขึ้น',
  },

  // 2. Bonds & Rates
  {
    term: 'Yield Curve',
    thaiTerm: 'เส้นอัตราผลตอบแทนพันธบัตร 📈',
    category: 'Bonds & Rates',
    icon: '📈',
    definition: 'กราฟเปรียบเทียบผลตอบแทนของพันธบัตรรัฐบาลสหรัฐฯ ตั้งแต่ระยะสั้น (1 เดือน) ถึงระยะยาว (30 ปี)',
    howToUse: 'ปกติเส้นจะชันขึ้น (ดอกเบี้ยยาวสูงกว่าสั้น) หากเส้นกลับหัว (Inverted: ดอกเบี้ยสั้นสูงกว่ายาว) มักเตือนว่าเศรษฐกิจอาจถดถอยใน 1–2 ปีข้างหน้า',
  },
  {
    term: '10-Year Treasury Yield',
    thaiTerm: 'ผลตอบแทนพันธบัตรรัฐบาลสหรัฐฯ 10 ปี 🇺🇸',
    category: 'Bonds & Rates',
    icon: '🇺🇸',
    definition: 'อัตราดอกเบี้ยอ้างอิงที่สำคัญที่สุดในโลก ใช้เป็นฐานคำนวณต้นทุนเงินกู้ ดอกเบี้ยบ้าน และการประเมินมูลค่าหุ้นทั่วโลก',
    howToUse: 'ถ้า Yield 10Y พุ่งขึ้นเร็ว หุ้นกลุ่มเทคโนโลยีและหุ้นเติบโตจะถูกกดดัน ถ้า Yield ลดลง ต้นทุนการเงินจะผ่อนคลาย',
  },
  {
    term: 'Real Yield',
    thaiTerm: 'อัตราผลตอบแทนที่แท้จริง (หักเงินเฟ้อ) 💎',
    category: 'Bonds & Rates',
    icon: '💎',
    definition: 'ผลตอบแทนพันธบัตรหลังหักการคาดการณ์เงินเฟ้อออกแล้ว (Nominal Yield - Inflation Expectation)',
    howToUse: 'ถ้า Real Yield สูง (เช่น เกิน 2.0%) การถือพันธบัตรหรือฝากเงินจะให้ผลตอบแทนชนะเงินเฟ้อชัดเจน สินทรัพย์ที่ไม่จ่ายดอกเบี้ยอย่างทองคำจะถูกแย่งเม็ดเงิน',
  },
  {
    term: 'Basis Points (bps)',
    thaiTerm: 'จุดพื้นฐาน (หน่วยวัดดอกเบี้ย) 🔢',
    category: 'Bonds & Rates',
    icon: '🔢',
    definition: 'หน่วยย่อยที่ใช้ในการวัดการเปลี่ยนแปลงอัตราดอกเบี้ย โดย 100 bps เท่ากับ 1.00% (25 bps = 0.25%)',
    howToUse: 'นักเศรษฐศาสตร์ใช้พูดถึงการขึ้นลงของดอกเบี้ยเพื่อความแม่นยำและป้องกันความสับสนกับเปอร์เซ็นต์ทั่วไป',
  },
  {
    term: 'Duration',
    thaiTerm: 'ดูเรชั่น (ความอ่อนไหวต่อดอกเบี้ยของตราสารหนี้) ⏱️',
    category: 'Bonds & Rates',
    icon: '⏱️',
    definition: 'ตัวเลขชี้วัดว่า ราคาของกองทุนตราสารหนี้หรือพันธบัตรจะขยับขึ้นลงกี่เปอร์เซ็นต์ หากอัตราดอกเบี้ยในตลาดเปลี่ยนไป 1.00%',
    howToUse: 'ถ้า Duration เท่ากับ 7 ปี เมื่อดอกเบี้ยขึ้น 1% ราคาพันธบัตรจะร่วงลงประมาณ 7% ในช่วงดอกเบี้ยขาขึ้นควรเลือกกองทุน Duration สั้น',
  },
  {
    term: 'Auction Bid-to-Cover Ratio',
    thaiTerm: 'อัตราส่วนความต้องการประมูลพันธบัตร 🤝',
    category: 'Bonds & Rates',
    icon: '🤝',
    definition: 'อัตราส่วนระหว่างมูลค่าคำสั่งเสนอซื้อพันธบัตรต่อนักลงทุน เทียบกับมูลค่าที่กระทรวงการคลังสหรัฐฯ ต้องการกู้',
    howToUse: 'ค่าเกิน 2.5 เท่า แปลว่านักลงทุนทั่วโลกแย่งกันซื้อ รัฐบาลกู้เงินสบาย ถ้าต่ำกว่า 2.2 เท่า สะท้อนว่าตลาดเริ่มอิ่มตัว อาจดันดอกเบี้ยให้สูงขึ้น',
  },

  // 3. Equities & Valuation
  {
    term: 'Equity Risk Premium (ERP)',
    thaiTerm: 'ส่วนชดเชยความเสี่ยงของการลงทุนในหุ้น ⚖️',
    category: 'Equities & Valuation',
    icon: '⚖️',
    definition: 'ผลตอบแทนส่วนเพิ่มที่นักลงทุนคาดหวังว่าจะได้จากการเสี่ยงถือหุ้น เทียบกับผลตอบแทนที่ได้ชัวร์ๆ จากการถือพันธบัตรรัฐบาล',
    howToUse: 'ถ้า ERP ติดลบหรือต่ำมาก แปลว่าหุ้นแพงมาก ซื้อตอนนี้ผลตอบแทนไม่คุ้มค่าความเสี่ยงเมื่อเทียบกับแค่ฝากเงินในพันธบัตร',
  },
  {
    term: 'Market Breadth (Advance/Decline)',
    thaiTerm: 'ความกว้างของตลาด (อัตราส่วนหุ้นขึ้นต่อหุ้นลง) 📊',
    category: 'Equities & Valuation',
    icon: '📊',
    definition: 'สัดส่วนจำนวนหุ้นในตลาดที่ราคาปรับตัวขึ้น เทียบกับจำนวนหุ้นที่ราคาปรับตัวลดลงในวันนั้นๆ',
    howToUse: 'ถ้าดัชนีรวมบวก แต่ A/D Ratio ต่ำกว่า 1.0 แปลว่าดัชนีขึ้นเพราะหุ้นตัวใหญ่ไม่กี่ตัว ตลาดยังไม่แข็งแกร่งจริง',
  },
  {
    term: 'Price-to-Earnings (P/E Ratio)',
    thaiTerm: 'อัตราส่วนราคาต่อกำไร 🏷️',
    category: 'Equities & Valuation',
    icon: '🏷️',
    definition: 'ราคาหุ้นปัจจุบันหารด้วยกำไรต่อหุ้น บอกว่าเรากำลังยอมจ่ายกี่เท่าของกำไรประจำปีของบริษัทนั้น',
    howToUse: 'P/E ต่ำเทียบกับค่าเฉลี่ยในอดีตมักหมายถึงราคาถูกหรือมี Margin of Safety ส่วน P/E สูงลิบสะท้อนว่าตลาดคาดหวังการเติบโตสูงมาก',
  },

  // 4. Portfolio & Risk
  {
    term: 'Tactical Asset Allocation (TAA)',
    thaiTerm: 'การปรับเอียงพอร์ตตามรอบเศรษฐกิจระยะสั้น 🎯',
    category: 'Portfolio & Risk',
    icon: '🎯',
    definition: 'กลยุทธ์การปรับเพิ่มหรือลดน้ำหนักสินทรัพย์ตามรอบวัฏจักรเศรษฐกิจ 3–6 เดือน สำหรับเงินก้อนใหม่หรือเงินส่วนยืดหยุ่น',
    howToUse: 'ใช้ฉวยโอกาสทำกำไรและลดความเสี่ยงตามฤดูกาล โดยไม่กระทบกระเทือนโครงสร้างเงินออมเกษียณหลัก',
  },
  {
    term: 'Strategic Asset Allocation (SAA)',
    thaiTerm: 'การจัดพอร์ตหลักระยะยาว 5–10 ปี 🏰',
    category: 'Portfolio & Risk',
    icon: '🏰',
    definition: 'สัดส่วนโครงสร้างพอร์ตหลักระยะยาวที่ออกแบบตามเป้าหมายชีวิตและระดับความเสี่ยง เช่น หุ้น 60% ตราสารหนี้ 40%',
    howToUse: 'เป็นเสาหลักของชีวิต ไม่ควรตื่นตระหนกเทขายทิ้งตามข่าวรายวัน แค่ทำการ Rebalance ตามรอบประจำปี',
  },
  {
    term: 'FX Hedging',
    thaiTerm: 'การป้องกันความเสี่ยงอัตราแลกเปลี่ยน 💵',
    category: 'Portfolio & Risk',
    icon: '💵',
    definition: 'การใช้สัญญาทางการเงินเพื่อล็อกอัตราแลกเปลี่ยนเมื่อคนไทยไปลงทุนในหุ้นหรือสินทรัพย์ต่างประเทศ',
    howToUse: 'เพื่อป้องกันไม่ให้กำไรจากการลงทุนถูกค่าเงินบาทแข็งค่ากลืนหายไป ควรเลือกลงทุนผ่านกองทุนที่มีนโยบาย Hedge ค่าเงิน',
  },
  {
    term: 'Invalidation Triggers',
    thaiTerm: 'เบรกเกอร์ตัดไฟ (จุดยอมรับว่าสมมติฐานผิด) 🚨',
    category: 'Portfolio & Risk',
    icon: '🚨',
    definition: 'ระดับตัวเลขตลาดที่กำหนดไว้ล่วงหน้าว่า หากหลุดกรอบนี้ แปลว่าสภาวะตลาดเปลี่ยนทิศทางกะทันหันและมุมมองเดิมใช้ไม่ได้แล้ว',
    howToUse: 'ช่วยตัดอารมณ์และป้องกันไม่ให้ดันทุรังถือสินทรัพย์เสี่ยง ให้ระงับการเข้าซื้อทันทีและเพิ่มเงินสดสำรอง',
  },
  {
    term: 'Stance: Overweight / Underweight / Neutral',
    thaiTerm: 'มุมมองน้ำหนักการลงทุน (เพิ่ม / ชะลอ / คงสัดส่วน) 🚦',
    category: 'Portfolio & Risk',
    icon: '🚦',
    definition: 'คำแนะนำเชิงเปรียบเทียบ Overweight = ทยอยสะสมเพิ่ม, Underweight = ลดสัดส่วน/ชะลอการซื้อ, Neutral = ถือตามสัดส่วนเดิม',
    howToUse: 'ใช้เป็นทิศทางในการหยอดเงินก้อนใหม่ ไม่จำเป็นต้องล้างพอร์ตเดิมทิ้งทั้งหมด',
  },

  // 5. Flows & Crypto
  {
    term: 'OFR Financial Stress Index',
    thaiTerm: 'ดัชนีความตึงตัวของระบบการเงินสหรัฐฯ 🩺',
    category: 'Flows & Crypto',
    icon: '🩺',
    definition: 'ดัชนีรวมที่สำนักงานวิจัยทางการเงินสหรัฐฯ ใช้ตรวจจับความเครียด ความเสี่ยงสภาพคล่อง และความผันผวนในระบบธนาคาร',
    howToUse: 'ค่าติดลบ (< 0) แปลว่าระบบการเงินผ่อนคลาย ปลอดภัย หากพุ่งเป็นบวกสูงผิดปกติ แปลว่าสถาบันการเงินเริ่มตึงตัว',
  },
  {
    term: 'CFTC Commitments of Traders (COT)',
    thaiTerm: 'เรดาร์ส่องไพ่กองทุนเฮดจ์ฟันด์ระดับโลก 👁️',
    category: 'Flows & Crypto',
    icon: '👁️',
    definition: 'รายงานของทางการสหรัฐฯ ที่เปิดเผยว่าผู้เล่นรายใหญ่และกองทุนเก็งกำไรระดับโลก ถือสถานะซื้อหรือขายสุทธิในสัญญา期貨สินค้าโภคภัณฑ์',
    howToUse: 'ช่วยให้เห็นว่าเงินฉลาด (Smart Money) กำลังวางเดิมพันฝั่งไหนในทองคำและน้ำมัน หากกองทุนเปิดสถานะซื้อสูงสุดในรอบหลายปี มักเตือนว่าความต้องการซื้ออาจใกล้สุดทาง',
  },
  {
    term: 'Crypto ETF Net Flows',
    thaiTerm: 'ปรอทวัดสภาพคล่องเงินสดใน Wall Street 🌊',
    category: 'Flows & Crypto',
    icon: '🌊',
    definition: 'ยอดรวมเม็ดเงินที่นักลงทุนสถาบันนำมาซื้อหรือไถ่ถอนกองทุน Bitcoin และ Ethereum Spot ETF ในตลาดหุ้นสหรัฐฯ',
    howToUse: 'เป็นตัวแทนชั้นดีของความกล้าเสี่ยง (Risk Appetite) หากเงินไหลเข้าต่อเนื่อง สะท้อนว่าสภาพคล่องส่วนเกินในระบบยังมีมาก',
  },
  {
    term: 'Thai Foreign Fund Flow',
    thaiTerm: 'กระแสเงินทุนนักลงทุนต่างชาติในตลาดไทย 🇹🇭',
    category: 'Flows & Crypto',
    icon: '🇹🇭',
    definition: 'ยอดรวมเม็ดเงินสุทธิที่นักลงทุนต่างชาติซื้อหรือขายในตลาดหุ้นไทย (SET) และตลาดตราสารหนี้ไทย',
    howToUse: 'หากต่างชาติขายสุทธิต่อเนื่อง ดัชนีหุ้นไทยมักปรับขึ้นได้ยากและค่าเงินบาทมีแนวโน้มอ่อนค่าลง',
  },
  {
    term: 'GTA Gold Premium / Discount',
    thaiTerm: 'ส่วนต่างราคาทองคำแท่งสมาคมค้าทองคำ 🪙',
    category: 'Flows & Crypto',
    icon: '🪙',
    definition: 'ส่วนต่างระหว่างราคาทองคำแท่งที่ประกาศโดยสมาคมค้าทองคำแห่งประเทศไทย เทียบกับราคาทองคำตลาดโลกแปลงเป็นเงินบาท',
    howToUse: 'สะท้อนความต้องการซื้อทองจริงในประเทศและต้นทุนค่าขนส่ง/ความตึงตัวของร้านทองในเยาวราช',
  },
]

const CATEGORIES: Array<GlossaryTerm['category'] | 'All'> = [
  'All',
  'Regime & Policy',
  'Bonds & Rates',
  'Equities & Valuation',
  'Portfolio & Risk',
  'Flows & Crypto',
]

const CATEGORY_NAMES: Record<string, string> = {
  All: 'ทั้งหมด',
  'Regime & Policy': 'สภาวะเศรษฐกิจ & นโยบาย',
  'Bonds & Rates': 'พันธบัตร & ดอกเบี้ย',
  'Equities & Valuation': 'ตลาดหุ้น & มูลค่า',
  'Portfolio & Risk': 'กลยุทธ์พอร์ต & ความเสี่ยง',
  'Flows & Crypto': 'กระแสเงิน & คริปโต',
}

interface MacroGlossaryDrawerProps {
  isOpen: boolean
  onClose: () => void
}

export const MacroGlossaryDrawer: React.FC<MacroGlossaryDrawerProps> = ({ isOpen, onClose }) => {
  const [search, setSearch] = useState('')
  const [selectedCategory, setSelectedCategory] = useState<GlossaryTerm['category'] | 'All'>('All')
  const panelRef = useFocusTrap<HTMLDivElement>(isOpen, onClose)

  const filteredTerms = useMemo(() => {
    return GLOSSARY_TERMS.filter((item) => {
      const matchCat = selectedCategory === 'All' || item.category === selectedCategory
      const query = search.trim().toLowerCase()
      if (!query) return matchCat

      const matchText =
        item.term.toLowerCase().includes(query) ||
        item.thaiTerm.toLowerCase().includes(query) ||
        item.definition.toLowerCase().includes(query) ||
        item.howToUse.toLowerCase().includes(query)

      return matchCat && matchText
    })
  }, [search, selectedCategory])

  if (!isOpen) return null

  return (
    <div className="fixed inset-0 z-50 overflow-hidden" role="dialog" aria-modal="true" aria-label="พจนานุกรม Macro ภาษาคน">
      {/* Backdrop */}
      <div
        className="fixed inset-0 bg-zinc-900/40 backdrop-blur-xs transition-opacity animate-in fade-in duration-200"
        onClick={onClose}
        aria-hidden="true"
      />

      <div className="fixed inset-y-0 right-0 max-w-full flex pl-10">
        <div
          ref={panelRef}
          className="w-screen max-w-md md:max-w-xl bg-white shadow-2xl flex flex-col border-l border-zinc-200 animate-in slide-in-from-right duration-250"
        >
          {/* Header */}
          <div className="p-4 sm:p-5 border-b border-zinc-200 bg-linear-to-r from-sky-50 to-indigo-50/50 flex items-center justify-between">
            <div className="flex items-center gap-2.5">
              <span className="text-2xl">📖</span>
              <div>
                <h2 className="text-base font-bold text-zinc-900">คู่มือคำศัพท์ Macro ภาษาคน</h2>
                <p className="text-xs text-zinc-600 mt-0.5">
                  รวม 25 ศัพท์เศรษฐกิจและการลงทุนสำคัญ เข้าใจได้ทันทีโดยไม่ต้องมีพื้นฐาน
                </p>
              </div>
            </div>
            <button
              type="button"
              onClick={onClose}
              className="rounded-lg p-2 text-zinc-500 hover:text-zinc-800 hover:bg-white/80 transition-colors focus:outline-hidden"
              aria-label="ปิดคู่มือศัพท์"
            >
              ✕
            </button>
          </div>

          {/* Search & Filter */}
          <div className="p-4 border-b border-zinc-100 bg-zinc-50/60 space-y-3">
            <div className="relative">
              <span className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none text-zinc-400 text-sm">
                🔍
              </span>
              <input
                type="text"
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                placeholder="ค้นหาคำศัพท์ เช่น Yield, ERP, TAA, บาทอ่อน..."
                className="w-full pl-9 pr-8 py-2 text-xs rounded-xl border border-zinc-300 bg-white placeholder-zinc-400 focus:outline-hidden focus:border-sky-500 focus:ring-2 focus:ring-sky-100 transition-all"
              />
              {search && (
                <button
                  type="button"
                  onClick={() => setSearch('')}
                  className="absolute inset-y-0 right-0 pr-3 flex items-center text-zinc-400 hover:text-zinc-600 text-xs"
                >
                  ✕
                </button>
              )}
            </div>

            {/* Category Pills */}
            <div className="flex items-center gap-1.5 overflow-x-auto pb-1 text-xs no-scrollbar">
              {CATEGORIES.map((cat) => (
                <button
                  key={cat}
                  type="button"
                  onClick={() => setSelectedCategory(cat)}
                  className={`px-2.5 py-1 rounded-lg font-medium whitespace-nowrap text-[11px] transition-colors ${
                    selectedCategory === cat
                      ? 'bg-zinc-900 text-white shadow-xs'
                      : 'bg-white text-zinc-600 border border-zinc-200 hover:bg-zinc-100'
                  }`}
                >
                  {CATEGORY_NAMES[cat]}
                </button>
              ))}
            </div>
          </div>

          {/* Terms List */}
          <div className="flex-1 overflow-y-auto p-4 sm:p-5 space-y-3.5">
            {filteredTerms.length === 0 ? (
              <div className="text-center py-12 text-zinc-500">
                <p className="text-2xl mb-1">🔍</p>
                <p className="text-xs font-semibold">ไม่พบคำศัพท์ที่ตรงกับการค้นหา</p>
                <p className="text-[11px] text-zinc-400 mt-1">ลองเปลี่ยนคำค้นหา หรือเลือกหมวดหมู่อื่น</p>
              </div>
            ) : (
              filteredTerms.map((item) => (
                <div
                  key={item.term}
                  className="rounded-xl border border-zinc-200 bg-white p-3.5 shadow-xs hover:border-sky-200 hover:shadow-sm transition-all"
                >
                  <div className="flex items-start justify-between gap-2">
                    <div className="flex items-center gap-2">
                      <span className="text-lg shrink-0">{item.icon}</span>
                      <div>
                        <h3 className="text-xs font-bold text-zinc-900 leading-snug">
                          {item.term}
                        </h3>
                        <span className="text-[11px] font-medium text-sky-800 block">
                          {item.thaiTerm}
                        </span>
                      </div>
                    </div>
                    <span className="text-[10px] font-semibold text-zinc-600 bg-zinc-100 px-2 py-0.5 rounded-full shrink-0">
                      {CATEGORY_NAMES[item.category]}
                    </span>
                  </div>

                  <div className="mt-2.5 space-y-2 text-xs leading-relaxed">
                    <div className="bg-sky-50/60 rounded-lg p-2.5 border border-sky-100">
                      <span className="font-bold text-sky-950 text-[11px] block mb-0.5">
                        📌 มันคืออะไร:
                      </span>
                      <p className="text-zinc-700 text-[11px] leading-relaxed">{item.definition}</p>
                    </div>

                    <div className="bg-emerald-50/60 rounded-lg p-2.5 border border-emerald-100">
                      <span className="font-bold text-emerald-950 text-[11px] block mb-0.5">
                        💰 วิธีนำไปใช้กับเงินเรา:
                      </span>
                      <p className="text-zinc-700 text-[11px] leading-relaxed">{item.howToUse}</p>
                    </div>
                  </div>
                </div>
              ))
            )}
          </div>

          {/* Footer */}
          <div className="p-3.5 border-t border-zinc-200 bg-zinc-50 flex items-center justify-between text-xs text-zinc-500">
            <span>แสดง {filteredTerms.length} จาก {GLOSSARY_TERMS.length} คำศัพท์</span>
            <button
              type="button"
              onClick={onClose}
              className="px-3 py-1.5 rounded-lg bg-zinc-900 text-white text-xs font-semibold hover:bg-zinc-800 transition-colors"
            >
              เข้าใจแล้ว ปิดหน้าต่าง
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}
