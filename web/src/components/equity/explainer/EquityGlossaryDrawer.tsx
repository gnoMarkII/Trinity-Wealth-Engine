import { useState, useMemo } from 'react'
import { useFocusTrap } from '../../../hooks/useFocusTrap'

export interface EquityGlossaryTerm {
  term: string
  thaiTerm: string
  category: 'Valuation & Price' | 'Business Quality' | 'Financial Health' | 'Tactical & Flow' | 'Risk & Derivatives'
  icon: string
  definition: string
  howToUse: string
}

export const EQUITY_GLOSSARY_TERMS: EquityGlossaryTerm[] = [
  // 1. Valuation & Price
  {
    term: 'P/E Ratio (Price-to-Earnings)',
    thaiTerm: 'ราคาต่อกำไรสุทธิ 🏷️',
    category: 'Valuation & Price',
    icon: '🏷️',
    definition: 'ราคาหุ้นปัจจุบันหารด้วยกำไรต่อหุ้น (EPS) บอกว่าถ้าเราซื้อหุ้นราคานี้ กิจการจะต้องทำกำไรเท่าเดิมอีกกี่ปีจึงจะคืนทุนเท่าราคาซื้อ',
    howToUse: 'P/E ต่ำ แปลว่าหุ้นถูกหรือตลาดกังวลว่ากำไรจะลดลงในอนาคต ส่วน P/E สูง แปลว่าหุ้นแพงหรือตลาดคาดหวังการเติบโตสูงมาก ควรเทียบกับค่าเฉลี่ยอดีตของหุ้นเองและกลุ่มอุตสาหกรรมเดียวกัน',
  },
  {
    term: 'P/BV Ratio (Price-to-Book)',
    thaiTerm: 'ราคาต่อมูลค่าทางบัญชี 📘',
    category: 'Valuation & Price',
    icon: '📘',
    definition: 'ราคาหุ้นเทียบกับส่วนของเจ้าของต่อหุ้น (มูลค่าทรัพย์สินสุทธิหลังหักหนี้สิน) บอกว่าเรากำลังซื้อหุ้นที่ราคาสูงหรือต่ำกว่ามูลค่าทางบัญชี',
    howToUse: 'P/BV ต่ำกว่า 1.0 เท่า มักพบในหุ้นกลุ่มธนาคารหรือพลังงานที่ถูกเทขายจนราคาต่ำกว่าสินทรัพย์ แต่ต้องระวังว่าสินทรัพย์นั้นด้อยค่าลงหรือไม่',
  },
  {
    term: 'DCF (Discounted Cash Flow)',
    thaiTerm: 'การประเมินมูลค่าด้วยการคิดลดกระแสเงินสด 🧮',
    category: 'Valuation & Price',
    icon: '🧮',
    definition: 'วิธีคำนวณมูลค่าที่แท้จริงของหุ้น โดยการคาดการณ์กระแสเงินสดอิสระ (FCF) ทั้งหมดที่บริษัทจะทำได้ในอนาคต แล้วคิดลดกลับมาเป็นมูลค่าเงินในวันนี้ตามต้นทุนเงินทุน (WACC)',
    howToUse: 'ถ้ามูลค่า DCF สูงกว่าราคาหุ้นในกระดานปัจจุบัน แปลว่าหุ้นมีส่วนลด (Undervalued) แต่ถ้าต่ำกว่า แปลว่าราคาหุ้นแพงเกินพื้นฐาน (Overvalued)',
  },
  {
    term: 'Reverse DCF',
    thaiTerm: 'การคิดย้อนกลับจากราคาหุ้นปัจจุบัน 🎯',
    category: 'Valuation & Price',
    icon: '🎯',
    definition: 'แทนที่จะคาดเดาอนาคต วิธีนี้จะเริ่มจาก "ราคาหุ้นวันนี้" แล้วคิดย้อนกลับไปว่า "บริษัทต้องทำรายได้และอัตรากำไรโตปีละกี่เปอร์เซ็นต์" ถึงจะสมเหตุสมผลกับราคานี้',
    howToUse: 'ช่วยตรวจจับฟองสบู่ ถ้าราคาหุ้นวันนี้ต้องการให้บริษัทโตปีละ 35% แต่บริษัทโตได้จริงแค่ 12% แปลว่าราคาหุ้นกำลังแบกความหวังสูงเกินจริงและเสี่ยงต่อการถูกเทขาย',
  },
  {
    term: 'Margin of Safety (MoS)',
    thaiTerm: 'ส่วนเผื่อความปลอดภัย 🛡️',
    category: 'Valuation & Price',
    icon: '🛡️',
    definition: 'ส่วนต่างระหว่าง "มูลค่าแท้จริงของหุ้น (Fair Value)" กับ "ราคาที่เราเข้าซื้อ" เปรียบเหมือนซื้อของราคา 75 บาทที่มูลค่าจริง 100 บาท เพื่อมีแต้มต่อป้องกันความผิดพลาด',
    howToUse: 'นักลงทุนสายคุณค่ามักต้องการ Margin of Safety อย่างน้อย 20–30% ก่อนตัดสินใจซื้อ เพื่อรองรับเหตุการณ์ไม่คาดฝันหรือสมมติฐานที่อาจผิดพลาด',
  },
  {
    term: 'EV/EBITDA',
    thaiTerm: 'มูลค่ากิจการเทียบกำไรจากการดำเนินงาน 🏢',
    category: 'Valuation & Price',
    icon: '🏢',
    definition: 'Enterprise Value (มูลค่าหุ้น + หนี้สินสุทธิ) หารด้วย EBITDA บอกว่าถ้ามีคนซื้อบริษัทนี้ทั้งหมดรวมภาระหนี้ ต้องใช้เวลาทำกำไรจากการดำเนินงานกี่ปีคืนทุน',
    howToUse: 'เหมาะอย่างยิ่งสำหรับเปรียบเทียบบริษัทที่มีโครงสร้างหนี้ไม่เท่ากัน หรือบริษัทที่อยู่ในกลุ่มอุตสาหกรรมหนักที่มีค่าเสื่อมราคาสูง',
  },
  {
    term: 'Terminal Value (TV)',
    thaiTerm: 'มูลค่าระยะยาวของกิจการ 🌐',
    category: 'Valuation & Price',
    icon: '🌐',
    definition: 'มูลค่าของกระแสเงินสดทั้งหมดที่คาดว่าจะเกิดขึ้นหลังจากพ้นช่วงประมาณการ 5-10 ปีแรกไปแล้วจนถึงตลอดกาล',
    howToUse: 'ในแบบจำลอง DCF มูลค่า Terminal Value มักกินสัดส่วนถึง 60–80% ของมูลค่าหุ้นทั้งหมด จึงต้องระวังสมมติฐานการเติบโตระยะยาว (Terminal Growth) ไม่ให้เกินการเติบโตของ GDP',
  },
  {
    term: 'WACC (Weighted Average Cost of Capital)',
    thaiTerm: 'ต้นทุนทางการเงินเฉลี่ยของบริษัท ⚖️',
    category: 'Valuation & Price',
    icon: '⚖️',
    definition: 'ต้นทุนรวมที่บริษัทต้องจ่ายเพื่อระดมทุน ทั้งจากดอกเบี้ยเงินกู้ยืมและผลตอบแทนที่ผู้ถือหุ้นคาดหวัง ใช้เป็นอัตราคิดลด (Discount Rate) ใน DCF',
    howToUse: 'ถ้าดอกเบี้ยในระบบเศรษฐกิจพุ่งขึ้น WACC จะสูงขึ้นตาม ส่งผลให้มูลค่าที่แท้จริงของหุ้นคำนวณออกมาได้ลดลงทันที',
  },

  // 2. Business Quality
  {
    term: 'Economic Moat',
    thaiTerm: 'ป้อมปราการทางธุรกิจ 🏰',
    category: 'Business Quality',
    icon: '🏰',
    definition: 'ความได้เปรียบทางการแข่งขันที่ยั่งยืน ซึ่งทำให้คู่แข่งรายใหม่เข้ามาแย่งส่วนแบ่งตลาดยาก (เช่น แบรนด์ Apple, ผลิตภัณฑ์ที่มีต้นทุนสลับค่ายสูง, สิทธิบัตร)',
    howToUse: 'บริษัทที่มี Moat กว้างจะสามารถรักษาระดับกำไรสูงและอัตราผลตอบแทนต่อเงินทุน (ROIC) ได้ยาวนาน แม้ต้องเผชิญกับคู่แข่งหรือเงินเฟ้อ',
  },
  {
    term: 'ROIC (Return on Invested Capital)',
    thaiTerm: 'ผลตอบแทนจากเงินลงทุนทั้งหมด 💎',
    category: 'Business Quality',
    icon: '💎',
    definition: 'กำไรจากการดำเนินงานหลังภาษีเทียบกับเงินทุนทั้งหมดที่ใช้ในธุรกิจ (ทั้งหนี้และทุน) วัดว่าผู้บริหารนำเงินไปต่อยอดสร้างผลตอบแทนได้เก่งแค่ไหน',
    howToUse: 'หุ้นชั้นยอดควรมี ROIC สูงกว่า WACC อย่างมีนัยสำคัญ (เช่น ROIC > 15% ขณะที่ WACC 8%) แสดงว่าทุก 100 บาทที่ลงทุนเพิ่ม สร้างมูลค่าส่วนเพิ่มให้ผู้ถือหุ้นจริง',
  },
  {
    term: 'ROE (Return on Equity)',
    thaiTerm: 'ผลตอบแทนต่อส่วนของผู้ถือหุ้น 📈',
    category: 'Business Quality',
    icon: '📈',
    definition: 'กำไรสุทธิเทียบกับเงินลงทุนของเจ้าของ สะท้อนประสิทธิภาพในการนำเงินของผู้ถือหุ้นไปสร้างกำไร',
    howToUse: 'ROE สูงมักเป็นสัญญาณดี (ปกติควร > 12–15%) แต่ต้องระวังว่า ROE สูงนั้นเกิดจากกำไรดีจริง หรือเกิดจากการกู้หนี้มาหมุนจนส่วนของผู้ถือหุ้นบางเฉียบ (วิเคราะห์ต่อด้วย Dupont)',
  },
  {
    term: 'Free Cash Flow (FCF)',
    thaiTerm: 'กระแสเงินสดอิสระ 💧',
    category: 'Business Quality',
    icon: '💧',
    definition: 'เงินสดเนื้อๆ ที่บริษัทผลิตได้จริงหลังหักค่าใช้จ่ายลงทุนซื้อเครื่องจักร เซิร์ฟเวอร์ หรือขยายสาขา (CapEx) แล้ว เป็นเงินสดที่พร้อมจ่ายปันผลหรือซื้อหุ้นคืน',
    howToUse: 'กำไรทางบัญชีอาจตกแต่งได้ แต่กระแสเงินสดอิสระโกหกยาก หุ้นที่กำไรโตแต่ FCF ติดลบต่อเนื่อง มักมีความเสี่ยงขาดสภาพคล่องในระยะยาว',
  },
  {
    term: 'Gross Margin & Operating Margin',
    thaiTerm: 'อัตรากำไรขั้นต้นและกำไรจากการดำเนินงาน 📊',
    category: 'Business Quality',
    icon: '📊',
    definition: 'Gross Margin วัดอำนาจการตั้งราคาเทียบต้นทุนสินค้า ส่วน Operating Margin วัดประสิทธิภาพในการควบคุมค่าใช้จ่ายในการขายและบริหาร',
    howToUse: 'ถ้า Operating Margin กว้างขึ้นเรื่อยๆ แสดงว่าบริษัทมี Operating Leverage (ยิ่งขายเยอะ กำไรยิ่งโตเร็วกว่ายอดขาย)',
  },
  {
    term: 'Shareholder Yield',
    thaiTerm: 'ผลตอบแทนรวมสู่ผู้ถือหุ้น 🎁',
    category: 'Business Quality',
    icon: '🎁',
    definition: 'ผลตอบแทนรวมที่บริษัทคืนเงินสดให้ผู้ถือหุ้น คำนวณจาก [เงินปันผล + ยอดซื้อหุ้นคืนสุทธิ + การลดหนี้] เทียบกับมูลค่าหลักทรัพย์ตามราคาตลาด',
    howToUse: 'ในตลาดหุ้นสหรัฐฯ บริษัทมักเน้นซื้อหุ้นคืนแทนการจ่ายปันผล Shareholder Yield จึงสะท้อนความใจป้ำในการคืนผลประโยชน์ให้ผู้ถือหุ้นได้ครบถ้วนกว่าดูแค่ Dividend Yield',
  },

  // 3. Financial Health
  {
    term: 'D/E Ratio (Debt-to-Equity)',
    thaiTerm: 'อัตราส่วนหนี้สินต่อทุน ⚖️',
    category: 'Financial Health',
    icon: '⚖️',
    definition: 'หนี้สินรวมหารด้วยส่วนของเจ้าของ บอกว่าบริษัทนี้ทำธุรกิจด้วยเงินกู้ยืมมากน้อยเพียงใดเมื่อเทียบกับเงินทุนของตัวเอง',
    howToUse: 'D/E ต่ำกว่า 1.0–1.5 เท่าถือว่าปลอดภัยและอนุรักษ์นิยม ถ้า D/E เกิน 2.5 เท่า แสดงว่าบริษัทพึ่งพาหนี้สูง มีความเสี่ยงต่อวิกฤตดอกเบี้ยขาขึ้น (ยกเว้นกลุ่มธนาคารและการเงิน)',
  },
  {
    term: 'Piotroski F-Score',
    thaiTerm: 'คะแนนตรวจสุขภาพการเงิน 9 ด่าน 🏥',
    category: 'Financial Health',
    icon: '🏥',
    definition: 'แบบทดสอบสุขภาพทางการเงิน 9 ข้อที่คิดค้นโดย ศ.โจเซฟ ปิโอโทรสกี้ แห่งมหาวิทยาลัยสแตนฟอร์ด ครอบคลุมทั้งการทำกำไร โครงสร้างหนี้ และประสิทธิภาพการทำงาน',
    howToUse: 'คะแนนเต็ม 9: คะแนน 7–9 คือสุขภาพแข็งแรงมาก, 4–6 ปานกลาง, และ 0–3 คือสัญญาณอันตราย มีโอกาสสูงที่ผลประกอบการจะทรุดตัว',
  },
  {
    term: 'Interest Coverage Ratio',
    thaiTerm: 'ความสามารถในการจ่ายดอกเบี้ย 🛡️',
    category: 'Financial Health',
    icon: '🛡️',
    definition: 'กำไรจากการดำเนินงาน (EBIT) หารด้วยค่าใช้จ่ายดอกเบี้ย บอกว่ากำไรที่ทำได้พอกับการจ่ายดอกเบี้ยเงินกู้กี่เท่า',
    howToUse: 'ควรมีค่ามากกว่า 3.0 เท่าขึ้นไป หากต่ำกว่า 1.5 เท่า แปลว่าบริษัทเริ่มมีปัญหาหาเงินมาจ่ายดอกเบี้ยไม่ทัน เสี่ยงต่อการผิดนัดชำระหนี้',
  },
  {
    term: 'Working Capital',
    thaiTerm: 'เงินทุนหมุนเวียน 🔄',
    category: 'Financial Health',
    icon: '🔄',
    definition: 'สินทรัพย์หมุนเวียน (เช่น เงินสด, ลูกหนี้, สต็อกสินค้า) หักลบด้วยหนี้สินหมุนเวียนที่ต้องจ่ายใน 1 ปี',
    howToUse: 'เงินทุนหมุนเวียนที่เป็นบวกและเพียงพอ ช่วยให้บริษัทไม่สะดุดเวลาต้องจ่ายค่าแรง ซื้อของเข้าร้าน หรือรับมือกับช่วงที่ลูกหนี้จ่ายเงินช้า',
  },
  {
    term: 'Dupont Analysis',
    thaiTerm: 'การชำแหละที่มาของกำไรดูปองท์ 🔍',
    category: 'Financial Health',
    icon: '🔍',
    definition: 'การแยกส่วนประกอบของ ROE ออกเป็น 3 ปัจจัย: 1) อัตรากำไรสุทธิ (Net Margin), 2) ประสิทธิภาพการใช้สินทรัพย์ (Asset Turnover), และ 3) การกู้หนี้ (Financial Leverage)',
    howToUse: 'ช่วยให้เห็นความจริงว่าบริษัททำ ROE ได้สูงเพราะขายของเก่งควบคุมต้นทุนดี หรือเป็นเพราะกู้หนี้มาท่วมหัว',
  },

  // 4. Tactical & Flow
  {
    term: 'Smart Money Flow',
    thaiTerm: 'รอยเท้าเงินสถาบันและกองทุน 🐋',
    category: 'Tactical & Flow',
    icon: '🐋',
    definition: 'การติดตามพฤติกรรมการซื้อขายของนักลงทุนสถาบัน กองทุนเฮดจ์ฟันด์ และผู้เล่นรายใหญ่ที่มีเครื่องมือและการวิเคราะห์ลึกกว่านักลงทุนรายย่อย',
    howToUse: 'หากพบสัญญาณ Smart Money เข้าสะสม (Accumulation) ในช่วงที่ราคาสร้างฐาน มักเป็นสัญญาณนำก่อนที่ราคาจะเริ่มวิ่งรอบใหม่',
  },
  {
    term: 'Insider Form 4 / แบบ 59-2',
    thaiTerm: 'รายงานการซื้อขายของผู้บริหาร 👔',
    category: 'Tactical & Flow',
    icon: '👔',
    definition: 'เอกสารรายงานทางการที่กฎหมาย กลต. บังคับให้กรรมการและผู้บริหารระดับสูงต้องเปิดเผยทุกครั้งที่มีการซื้อหรือขายหุ้นของบริษัทตนเอง',
    howToUse: 'ผู้บริหารขายหุ้นอาจมีร้อยเหตุผล (จ่ายภาษี, ซื้อบ้าน) แต่ผู้บริหารควักเงินส่วนตัวซื้อหุ้นมักมีเหตุผลเดียว คือ "มั่นใจว่าราคาหุ้นต่ำกว่ามูลค่าจริงและบริษัทกำลังจะดี"',
  },
  {
    term: 'Tactical Stage Cycle (4 Stages)',
    thaiTerm: 'วงจรพฤติกรรมราคา 4 สเตจ 🔄',
    category: 'Tactical & Flow',
    icon: '🔄',
    definition: 'ทฤษฎีวงจรราคาหุ้นแบ่งเป็น 4 สเตจ: 1. Accumulation (สะสมพลัง), 2. Markup (ขาขึ้นวิ่งแรง), 3. Distribution (กระจายของ), 4. Markdown (ขาลง)',
    howToUse: 'จุดทำกำไรที่ดีที่สุดและปลอดภัยที่สุดคือการเข้าซื้อในช่วงปลายสเตจ 1 ที่เริ่มเบรกเข้าสเตจ 2 และหลีกเลี่ยงการเข้าไปถือหุ้นในสเตจ 3 และ 4',
  },
  {
    term: 'Support & Resistance',
    thaiTerm: 'แนวรับและแนวต้าน 🧱',
    category: 'Tactical & Flow',
    icon: '🧱',
    definition: 'แนวรับคือระดับราคาที่มีแรงซื้อดักรอหนาแน่นจนราคาหยุดลง ส่วนแนวต้านคือระดับราคาที่มีแรงเทขายดักรอจนราคาผ่านได้ยาก',
    howToUse: 'ใช้เป็นจุดอ้างอิงในการวางจังหวะเข้าซื้อ (Buy on Dip ที่แนวรับ) และจุดตั้ง Stop Loss ใต้แนวรับ',
  },
  {
    term: 'Order Book Depth Ladder',
    thaiTerm: 'บันไดความลึกแรงซื้อแรงขาย 🪜',
    category: 'Tactical & Flow',
    icon: '🪜',
    definition: 'การวิเคราะห์ปริมาณคำสั่งซื้อ (Bid) และคำสั่งขาย (Ask) ในแต่ละช่วงราคาล่วงหน้า เพื่อตรวจหาแนวรับแนวต้านที่แท้จริงจากสภาพคล่อง',
    howToUse: 'ช่วยให้นักลงทุนรู้ว่าช่วงราคาไหนมีสภาพคล่องหนาแน่น สามารถซื้อหรือขายได้โดยไม่ทำให้ราคาหุ้นแกว่งตัวรูดลงไปแรง (Slippage)',
  },
  {
    term: 'Beta (ค่าเบต้า)',
    thaiTerm: 'ความอ่อนไหวเทียบกับตลาด ⚡',
    category: 'Tactical & Flow',
    icon: '⚡',
    definition: 'ค่าสถิติที่วัดว่าราคาหุ้นตัวนี้แกว่งตัวแรงกว่าหรือเบากว่าดัชนีตลาดภาพรวม (เช่น S&P 500 หรือ SET)',
    howToUse: 'Beta = 1.0 แปลว่าวิ่งเท่าตลาด, Beta > 1.5 แปลว่าผันผวนแรงกว่าตลาดมาก (เวลาตลาดขึ้นจะขึ้นแรง เวลาตลาดลงจะลงหนัก), Beta < 0.8 เหมาะกับผู้ต้องการความปลอดภัย',
  },

  // 5. Risk & Derivatives
  {
    term: 'Kill Switch / Stop Loss',
    thaiTerm: 'จุดตัดขาดทุน / ยอมแพ้เมื่อผิดทาง 🚨',
    category: 'Risk & Derivatives',
    icon: '🚨',
    definition: 'ระดับราคาหรือเงื่อนไขทางพื้นฐานที่ถูกกำหนดไว้ล่วงหน้าว่า "ถ้าหลุดระดับนี้ แปลว่าเหตุผลที่เราซื้อตั้งแต่แรกผิดไปแล้ว" ต้องขายทิ้งทันที',
    howToUse: 'กฎเหล็กในการป้องกันการติดดอยและรักษาเงินต้นไม่ให้เสียหายหนักเกินแก้ อย่าขยับจุด Stop Loss ลงไปเรื่อยๆ เพื่อหลอกตัวเอง',
  },
  {
    term: 'Thesis Falsifier',
    thaiTerm: 'ปัจจัยหักล้างสมมติฐานการลงทุน ❌',
    category: 'Risk & Derivatives',
    icon: '❌',
    definition: 'ตัวชี้วัดหรือเหตุการณ์เฉพาะเจาะจงที่หากเกิดขึ้นจริง จะพิสูจน์ว่ามุมมองเชิงบวกที่เรามีต่อหุ้นตัวนี้ไม่เป็นความจริงอีกต่อไป (เช่น อัตรากำไรหดตัวเกิน 5% หรือเสียลูกค้ารายใหญ่)',
    howToUse: 'นักลงทุนที่ดีต้องรู้ล่วงหน้าเสมอว่า "อะไรคือสิ่งที่จะทำให้เราเลิกลงทุนในหุ้นตัวนี้"',
  },
  {
    term: 'Options Call / Put Open Interest (OI)',
    thaiTerm: 'สัญญาคงค้างในตลาดออปชัน 📊',
    category: 'Risk & Derivatives',
    icon: '📊',
    definition: 'จำนวนสัญญาสิทธิซื้อ (Call) และสิทธิขาย (Put) ที่ยังไม่ได้ปิดสถานะในแต่ละระดับราคาใช้สิทธิ (Strike Price)',
    howToUse: 'ระดับ Strike ที่มี Call OI หรือ Put OI สูงมากๆ มักทำหน้าที่เป็นแนวต้านหรือแนวรับทางจิตวิทยาที่เจ้ามือหรือ Market Maker ต้องเข้ามาปรับพอร์ตเพื่อบริหารความเสี่ยง',
  },
  {
    term: 'Max Pain Theory',
    thaiTerm: 'ทฤษฎีแม็กซ์เพนในตลาดออปชัน 🎯',
    category: 'Risk & Derivatives',
    icon: '🎯',
    definition: 'ระดับราคาหุ้นที่จะทำให้ผู้ซื้อสัญญาสิทธิ (Options Buyers) ส่วนใหญ่ขาดทุนสูงสุด และทำให้ผู้ขายสิทธิ (Market Makers/สถาบัน) ได้กำไรจากค่าพรีเมียมสูงสุดในวันหมดอายุ',
    howToUse: 'ในสัปดาห์ที่มีการหมดอายุของสัญญาออปชัน ราคาหุ้นมักมีแรงดึงดูดให้เคลื่อนเข้าหาระดับราคา Max Pain',
  },
  {
    term: 'Short Interest & Days to Cover',
    thaiTerm: 'สัดส่วนการขายชอร์ตและเวลาที่ต้องใช้ซื้อคืน 📉',
    category: 'Risk & Derivatives',
    icon: '📉',
    definition: 'ปริมาณหุ้นที่ถูกยืมมาขายชอร์ตเพื่อเก็งกำไรขาลงเทียบกับจำนวนหุ้นทั้งหมด และจำนวนวันที่ต้องใช้ซื้อหุ้นคืนหมดตามปริมาณการซื้อขายเฉลี่ย',
    howToUse: 'ถ้า Short Interest สูงมาก (เช่น > 15-20%) แล้วบริษัทประกาศงบออกมาดีกว่าคาด อาจเกิดปรากฏการณ์ Short Squeeze (ผู้ชอร์ตแย่งกันซื้อคืนดันให้ราคาพุ่งระเบิด)',
  },
]

interface EquityGlossaryDrawerProps {
  isOpen: boolean
  onClose: () => void
}

export const EquityGlossaryDrawer: React.FC<EquityGlossaryDrawerProps> = ({ isOpen, onClose }) => {
  const [searchQuery, setSearchQuery] = useState('')
  const [selectedCategory, setSelectedCategory] = useState<string>('all')

  const containerRef = useFocusTrap<HTMLDivElement>(isOpen, onClose)

  const categories = useMemo(() => {
    return [
      { key: 'all', label: 'ทั้งหมด (30 คำ)' },
      { key: 'Valuation & Price', label: 'การประเมินมูลค่า (8)' },
      { key: 'Business Quality', label: 'คุณภาพธุรกิจ (6)' },
      { key: 'Financial Health', label: 'สุขภาพการเงิน (5)' },
      { key: 'Tactical & Flow', label: 'จังหวะ & กระแสเงิน (6)' },
      { key: 'Risk & Derivatives', label: 'ความเสี่ยง & ออปชัน (5)' },
    ]
  }, [])

  const filteredTerms = useMemo(() => {
    return EQUITY_GLOSSARY_TERMS.filter((t) => {
      const matchCategory = selectedCategory === 'all' || t.category === selectedCategory
      const query = searchQuery.trim().toLowerCase()
      const matchQuery =
        !query ||
        t.term.toLowerCase().includes(query) ||
        t.thaiTerm.toLowerCase().includes(query) ||
        t.definition.toLowerCase().includes(query) ||
        t.howToUse.toLowerCase().includes(query)
      return matchCategory && matchQuery
    })
  }, [selectedCategory, searchQuery])

  if (!isOpen) return null

  return (
    <div
      className="fixed inset-0 z-50 overflow-hidden bg-black/40 backdrop-blur-2xs transition-opacity animate-in fade-in duration-200"
      onClick={onClose}
      aria-modal="true"
      role="dialog"
      aria-label="คู่มือคำศัพท์วิเคราะห์หุ้นภาษาคน"
    >
      <div className="fixed inset-y-0 right-0 flex max-w-full pl-10">
        <div
          ref={containerRef}
          onClick={(e) => e.stopPropagation()}
          className="w-screen max-w-xl bg-white shadow-2xl flex flex-col border-l border-sky-100"
        >
          {/* Header */}
          <div className="p-5 border-b border-slate-100 bg-sky-50/40">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2.5">
                <span className="flex h-9 w-9 items-center justify-center rounded-xl bg-sky-100 text-lg">
                  📖
                </span>
                <div>
                  <h2 className="text-base font-bold text-zinc-900 tracking-tight">
                    คู่มือศัพท์วิเคราะห์หุ้นฉบับภาษาคน
                  </h2>
                  <p className="text-xs text-zinc-500">
                    คำศัพท์การเงินและตัวชี้วัดระดับสถาบัน 30 คำ แปลให้เข้าใจง่ายใน 3 บรรทัด
                  </p>
                </div>
              </div>
              <button
                type="button"
                onClick={onClose}
                className="rounded-lg p-2 text-zinc-400 hover:bg-slate-100 hover:text-zinc-600 transition-colors"
                aria-label="ปิดพจนานุกรม"
              >
                ✕
              </button>
            </div>

            {/* Search Input */}
            <div className="mt-4 relative">
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder="ค้นหาคำศัพท์ เช่น P/E, DCF, ROIC, Moat, Smart Money..."
                className="w-full rounded-xl border border-slate-200 bg-white px-3.5 py-2 pl-9 text-xs text-zinc-900 placeholder:text-zinc-400 focus:border-sky-500 focus:outline-hidden focus:ring-1 focus:ring-sky-500"
              />
              <span className="absolute left-3 top-2.5 text-xs text-zinc-400">🔍</span>
              {searchQuery && (
                <button
                  type="button"
                  onClick={() => setSearchQuery('')}
                  className="absolute right-3 top-2.5 text-xs text-zinc-400 hover:text-zinc-600"
                >
                  ✕
                </button>
              )}
            </div>

            {/* Category Filter Pills */}
            <div className="mt-3 flex gap-1.5 overflow-x-auto pb-1 text-xs scrollbar-thin">
              {categories.map((cat) => (
                <button
                  key={cat.key}
                  type="button"
                  onClick={() => setSelectedCategory(cat.key)}
                  className={`whitespace-nowrap px-2.5 py-1 rounded-lg text-[11px] font-medium transition-all ${
                    selectedCategory === cat.key
                      ? 'bg-sky-600 text-white shadow-2xs font-semibold'
                      : 'bg-white text-zinc-600 border border-slate-200 hover:bg-slate-50'
                  }`}
                >
                  {cat.label}
                </button>
              ))}
            </div>
          </div>

          {/* Terms List Body */}
          <div className="flex-1 overflow-y-auto p-5 space-y-3.5 divide-y divide-slate-100">
            {filteredTerms.length === 0 ? (
              <div className="py-12 text-center text-xs text-zinc-400">
                <span className="text-3xl block mb-2">🔍</span>
                ไม่พบคำศัพท์ที่ตรงกับ &ldquo;{searchQuery}&rdquo;
              </div>
            ) : (
              filteredTerms.map((item, idx) => (
                <div key={idx} className={idx === 0 ? '' : 'pt-3.5'}>
                  <div className="flex items-start justify-between gap-2">
                    <div>
                      <div className="flex items-center gap-2">
                        <span className="text-base">{item.icon}</span>
                        <h3 className="font-bold text-zinc-900 text-sm">{item.term}</h3>
                      </div>
                      <span className="text-xs font-semibold text-sky-700 block mt-0.5">
                        {item.thaiTerm}
                      </span>
                    </div>
                    <span className="rounded-md bg-slate-100 px-2 py-0.5 text-[10px] font-mono font-medium text-zinc-600 shrink-0">
                      {item.category}
                    </span>
                  </div>

                  <div className="mt-2 text-xs text-zinc-700 leading-relaxed bg-slate-50/70 rounded-xl p-3 border border-slate-100 space-y-2">
                    <div>
                      <span className="font-bold text-zinc-900 text-[11px] block text-sky-900">
                        📌 มันคืออะไร:
                      </span>
                      <p className="mt-0.5 text-zinc-700 text-[11px] leading-relaxed">{item.definition}</p>
                    </div>

                    <div className="border-t border-slate-200/60 pt-2">
                      <span className="font-bold text-emerald-900 text-[11px] block">
                        💡 วิธีดูและผลต่อการลงทุน:
                      </span>
                      <p className="mt-0.5 text-zinc-700 text-[11px] leading-relaxed">{item.howToUse}</p>
                    </div>
                  </div>
                </div>
              ))
            )}
          </div>

          {/* Footer note */}
          <div className="p-3.5 border-t border-slate-100 bg-slate-50/60 text-[11px] text-zinc-500 text-center flex items-center justify-center gap-2">
            <span>💡</span>
            <span>ศัพท์การเงินทั้งหมดคำนวณผ่านโมดูลคณิตศาสตร์ทางการ Python Deterministic ไร้การคาดเดา</span>
          </div>
        </div>
      </div>
    </div>
  )
}
