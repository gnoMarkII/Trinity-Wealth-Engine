import React, { useState } from 'react'
import type { TranscriptCallDTO } from '../../../api/types'

interface PromiseMethodologyAccordionProps {
  calls: TranscriptCallDTO[]
  frameworkVersion?: string
}

export const PromiseMethodologyAccordion: React.FC<PromiseMethodologyAccordionProps> = ({
  calls,
  frameworkVersion = '0.1',
}) => {
  const [isOpen, setIsOpen] = useState(false)

  const formatQuarterLabel = (q: string) => {
    const match = q.match(/^FY(\d{2})(\d{2})Q(\d)$/)
    if (match) {
      return `FY${match[2]} Q${match[3]}`
    }
    return q
  }

  return (
    <div className="rounded-2xl border border-edge bg-surface shadow-sm overflow-hidden transition-all">
      {/* Header / Clickable Toggle */}
      <button
        type="button"
        onClick={() => setIsOpen(!isOpen)}
        aria-expanded={isOpen}
        className="w-full p-5 text-left flex items-center justify-between gap-4 hover:bg-surface-muted/40 transition-colors focus:outline-none"
      >
        <div className="flex items-center gap-3">
          <span className="text-xl">📐</span>
          <div>
            <h4 className="font-bold text-zinc-900 dark:text-zinc-100 text-sm flex items-center gap-2">
              <span>วิธีการ เกณฑ์ให้คะแนน และบทถอดความที่ใช้</span>
              <span className="text-[11px] font-mono px-2 py-0.5 rounded bg-zinc-100 dark:bg-zinc-800 text-zinc-600 dark:text-zinc-400">
                Framework v{frameworkVersion}
              </span>
            </h4>
            <p className="text-xs text-zinc-500 mt-0.5">
              เกณฑ์ 4 ข้อในการคัดเลือกคำสัญญา, สูตรคะแนนถ่วงน้ำหนัก 80/20, เกณฑ์ความคลาดเคลื่อน (Tolerance) และรายการ Transcript 8 ไตรมาส
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2 flex-none">
          <span className="text-xs font-semibold text-zinc-500 hidden sm:inline">
            {isOpen ? 'ซ่อนรายละเอียด' : 'เปิดดูวิธีการและแหล่งข้อมูล'}
          </span>
          <span
            className={`w-6 h-6 rounded-lg flex items-center justify-center bg-surface-muted text-zinc-500 transition-transform duration-200 ${
              isOpen ? 'rotate-180' : ''
            }`}
          >
            ▼
          </span>
        </div>
      </button>

      {/* Expanded Accordion Body */}
      {isOpen && (
        <div className="p-5 pt-0 space-y-6 border-t border-edge/60 text-xs text-zinc-600 dark:text-zinc-400 animate-page-in">
          {/* Top 3 Methodology Cards */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 pt-4">
            {/* 1. 4-Test Promise Filter */}
            <div className="rounded-xl border border-edge/80 bg-surface-muted/30 p-4 space-y-2">
              <div className="font-bold text-zinc-900 dark:text-zinc-100 flex items-center gap-1.5 text-xs">
                <span>🎯</span>
                <span>เกณฑ์คัดเลือกคำสัญญา 4 ข้อ</span>
              </div>
              <ul className="space-y-1.5 list-disc list-inside leading-relaxed text-[11.5px]">
                <li>
                  <strong className="text-zinc-850 dark:text-zinc-200">Forward-looking:</strong> เป็นเป้าหมายอนาคต ไม่ใช่การสรุปผลย้อนหลัง
                </li>
                <li>
                  <strong className="text-zinc-850 dark:text-zinc-200">Management:</strong> กล่าวโดยผู้บริหาร (CEO, CFO, COO)
                </li>
                <li>
                  <strong className="text-zinc-850 dark:text-zinc-200">Own action:</strong> ผลงานของบริษัทเอง ไม่ใช่ปัจจัยภายนอก/คู่แข่ง
                </li>
                <li>
                  <strong className="text-zinc-850 dark:text-zinc-200">Checkable:</strong> ตรวจสอบได้ (ตัวเลขพร้อมช่วงเวลา หรือหมุดหมายพร้อมกำหนดส่งมอบ)
                </li>
              </ul>
              <div className="text-[10.5px] text-zinc-500 pt-1 border-t border-edge/40">
                *จำกัดสูงสุดไม่เกิน 12 ข้อต่อไตรมาส เพื่อความเป็นธรรมระหว่างบริษัทที่พูดมากและพูดน้อย
              </div>
            </div>

            {/* 2. Tolerances */}
            <div className="rounded-xl border border-edge/80 bg-surface-muted/30 p-4 space-y-2">
              <div className="font-bold text-zinc-900 dark:text-zinc-100 flex items-center gap-1.5 text-xs">
                <span>⚖️</span>
                <span>เกณฑ์ส่วนต่างที่ยอมรับได้ (Tolerance)</span>
              </div>
              <p className="leading-relaxed text-[11.5px]">
                ผลจริงต้องอยู่ในกรอบเป้าหมาย หรือไม่เกินส่วนต่างที่กำหนด:
              </p>
              <ul className="space-y-1 list-disc list-inside leading-relaxed text-[11.5px]">
                <li>ตัวเลขเงิน/รายได้ (Revenue): ยอมรับได้ <strong className="text-zinc-850 dark:text-zinc-200">±1%</strong></li>
                <li>อัตรากำไร (Operating/Gross Margin): ยอมรับได้ <strong className="text-zinc-850 dark:text-zinc-200">±0.5 จุด</strong></li>
                <li>กำไรต่อหุ้น (EPS) / จำนวนหน่วย: ยอมรับได้ <strong className="text-zinc-850 dark:text-zinc-200">±3%</strong></li>
                <li>ส่งมอบช้า ≤ 1 ไตรมาส หรือต่ำกว่าเป้าเล็กน้อย: นับเป็น <strong className="text-amber-600 dark:text-amber-400">สำเร็จบางส่วน (50 คะแนน)</strong></li>
              </ul>
            </div>

            {/* 3. Scoring Formula */}
            <div className="rounded-xl border border-edge/80 bg-surface-muted/30 p-4 space-y-2">
              <div className="font-bold text-zinc-900 dark:text-zinc-100 flex items-center gap-1.5 text-xs">
                <span>🏆</span>
                <span>สูตรคะแนน Credibility Score (80/20)</span>
              </div>
              <div className="space-y-1.5 text-[11.5px] leading-relaxed">
                <div>
                  <span className="font-semibold text-zinc-800 dark:text-zinc-200">Delivery Rate (80%):</span> คำนวณจากการทำได้จริง (ทำได้ 100, บางส่วน 50, ลดเป้า 40, ยอมรับพลาด 15, ทิ้งเงียบ 0) ถ่วงน้ำหนักตัวเลขบริษัท (2.0x) มากกว่าทิศทางทั่วไป (1.0x)
                </div>
                <div>
                  <span className="font-semibold text-zinc-800 dark:text-zinc-200">Transparency (20%):</span> วัดสัดส่วนที่ยอมรับความผิดพลาด เทียบกับกรณีที่เงียบหายไปเฉย ๆ
                </div>
                <div className="text-[10.5px] text-zinc-500 pt-1 border-t border-edge/40">
                  *สัญญาที่ยังไม่ถึงกำหนด (Pending) จะไม่ถูกนำมาหักคะแนน
                </div>
              </div>
            </div>
          </div>

          {/* Transcripts Used Section */}
          <div className="space-y-3 pt-2">
            <div className="flex items-center justify-between">
              <h5 className="font-bold text-zinc-800 dark:text-zinc-200 text-xs flex items-center gap-2">
                <span>🎙️</span>
                <span>บทถอดความที่ใช้ในการตรวจสอบ (Transcripts Used — 8 Quarters)</span>
              </h5>
              <span className="text-[11px] text-zinc-500">
                ถอดความฉบับเต็มทั้งช่วง Prepared Remarks และ Q&amp;A
              </span>
            </div>

            {calls && calls.length > 0 ? (
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
                {calls.map((call) => (
                  <div
                    key={call.quarter}
                    className="p-3 rounded-xl border border-edge/80 bg-surface-muted/20 flex flex-col justify-between gap-2"
                  >
                    <div>
                      <div className="flex items-center justify-between gap-1">
                        <span className="font-bold text-xs text-zinc-900 dark:text-zinc-100">
                          {formatQuarterLabel(call.quarter)}
                        </span>
                        {call.status === 'ok' ? (
                          <span className="text-[10px] px-1.5 py-0.2 rounded bg-emerald-100 text-emerald-800 dark:bg-emerald-950 dark:text-emerald-300 font-semibold">
                            สมบูรณ์
                          </span>
                        ) : (
                          <span className="text-[10px] px-1.5 py-0.2 rounded bg-rose-100 text-rose-800 dark:bg-rose-950 dark:text-rose-300 font-semibold">
                            ไม่มีข้อมูล
                          </span>
                        )}
                      </div>
                      <div className="text-[11px] text-zinc-500 mt-0.5">
                        {call.date || 'ไม่ระบุวันที่'}
                        {call.words ? ` · ${(call.words).toLocaleString()} คำ` : ''}
                      </div>
                    </div>

                    <div className="pt-1 border-t border-edge/40 flex items-center justify-between text-[11px]">
                      <span className="text-zinc-400">แหล่งข้อมูล: {call.source || 'StockAnalysis'}</span>
                      {call.source_url ? (
                        <a
                          href={call.source_url}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="font-semibold text-sky-600 dark:text-sky-400 hover:underline inline-flex items-center gap-0.5"
                          title="เปิดอ่านบทถอดความฉบับเต็มในแท็บใหม่"
                        >
                          <span>เปิดอ่าน</span>
                          <span>↗</span>
                        </a>
                      ) : (
                        <span className="text-zinc-400 italic">ไม่มีลิงก์</span>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-xs text-zinc-500 italic p-3 text-center border border-dashed border-edge/60 rounded-xl">
                ยังไม่มีข้อมูลบทถอดความที่บันทึกไว้
              </div>
            )}
          </div>

          {/* Verbatim Quote Discipline Notice */}
          <div className="p-3 rounded-xl bg-sky-500/5 border border-sky-500/20 text-[11px] leading-relaxed text-zinc-600 dark:text-zinc-400">
            <strong className="text-sky-700 dark:text-sky-300">📌 หลักการอ้างอิงคำพูดจริง (Verbatim Discipline):</strong>{' '}
            ข้อความอ้างอิงทุกข้อบนระบบเป็นคำพูดต้นฉบับภาษาอังกฤษของผู้บริหารคำต่อคำ ไม่มีการแปลหรือดัดแปลง เพื่อให้สามารถนำข้อความไปค้นหา (Search / Grep) ตรวจสอบกับบทถอดความต้นฉบับได้ 100%
          </div>
        </div>
      )}
    </div>
  )
}
