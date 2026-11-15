import React, { useState } from 'react'
import { BoltStrategyIcon, ChevronDownIcon, ChevronUpIcon } from '../../ui/AppIcons'

export interface StockWayfinderBannerProps {
  ticker: string
  companyName?: string | null
  market?: string
}

export const StockWayfinderBanner: React.FC<StockWayfinderBannerProps> = ({
  ticker,
  companyName,
  market = 'US',
}) => {
  const [isCollapsed, setIsCollapsed] = useState(false)

  return (
    <div className="rounded-2xl border border-sky-200/80 bg-gradient-to-r from-sky-50/90 via-white to-blue-50/60 p-4 shadow-sm">
      <div className="flex items-center justify-between border-b border-sky-100 pb-2.5">
        <div className="flex items-center gap-2 min-w-0">
          <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-lg bg-section-accent text-white shadow-sm">
            <BoltStrategyIcon className="h-3.5 w-3.5" />
          </span>
          <div className="min-w-0">
            <h3 className="text-xs font-bold text-sky-950 tracking-tight flex items-center gap-1.5 flex-wrap">
              <span>วิธีดูหุ้น {ticker} ใน 30 วินาที สำหรับมือใหม่ (3-Step Stock Wayfinder)</span>
              {companyName && (
                <span className="text-[11px] font-normal text-slate-500 hidden sm:inline">
                  — {companyName} ({market})
                </span>
              )}
            </h3>
          </div>
        </div>
        <button
          type="button"
          onClick={() => setIsCollapsed((prev) => !prev)}
          className="inline-flex items-center gap-1 text-[11px] font-semibold text-sky-700 hover:text-sky-900 transition-colors shrink-0"
        >
          <span>{isCollapsed ? 'แสดงคำแนะนำ' : 'ซ่อนคำแนะนำ'}</span>
          {isCollapsed ? <ChevronDownIcon className="h-3 w-3" /> : <ChevronUpIcon className="h-3 w-3" />}
        </button>
      </div>

      {!isCollapsed && (
        <div className="mt-3 grid grid-cols-1 md:grid-cols-3 gap-3 text-xs">
          {/* Step 1 */}
          <div className="rounded-xl bg-white/90 p-3 border border-sky-100 shadow-sm space-y-1">
            <div className="flex items-center gap-1.5 text-sky-900 font-bold text-xs">
              <span className="flex h-5 w-5 items-center justify-center rounded-full bg-sky-100 text-[11px] text-sky-700 font-mono">
                1
              </span>
              <span>เข้าใจธุรกิจ & คุณภาพ (10 วิ)</span>
            </div>
            <p className="text-[11px] text-slate-600 leading-relaxed">
              เช็ค <strong>Composite Score</strong> และ 4 เสาหลัก (Quality/Health) กิจการนี้กำไรสม่ำเสมอ มีป้อมปราการ (Moat) แข็งแกร่ง และหนี้ไม่ล้นใช่ไหม
            </p>
          </div>

          {/* Step 2 */}
          <div className="rounded-xl bg-white/90 p-3 border border-sky-100 shadow-sm space-y-1">
            <div className="flex items-center gap-1.5 text-amber-900 font-bold text-xs">
              <span className="flex h-5 w-5 items-center justify-center rounded-full bg-amber-100 text-[11px] text-amber-700 font-mono">
                2
              </span>
              <span>ประเมินความถูกแพง (10 วิ)</span>
            </div>
            <p className="text-[11px] text-slate-600 leading-relaxed">
              ดู <strong>Valuation Workbench</strong> และ <strong>Expectation Gap</strong> ราคาวันนี้ต่ำกว่ามูลค่าแท้จริง (มี Margin of Safety) หรือตลาดกำลังคาดหวังเวอร์เกินไป
            </p>
          </div>

          {/* Step 3 */}
          <div className="rounded-xl bg-white/90 p-3 border border-sky-100 shadow-sm space-y-1">
            <div className="flex items-center gap-1.5 text-emerald-900 font-bold text-xs">
              <span className="flex h-5 w-5 items-center justify-center rounded-full bg-emerald-100 text-[11px] text-emerald-700 font-mono">
                3
              </span>
              <span>เช็คจังหวะ & จุดตัดขาดทุน (10 วิ)</span>
            </div>
            <p className="text-[11px] text-slate-600 leading-relaxed">
              ดู <strong>Smart Money & Insider</strong> ว่ารายใหญ่ซื้อหรือขาย และสแกน <strong>Kill Switch</strong> เพื่อล็อกจุดตัดใจล่วงหน้าหากสมมติฐานผิดทาง
            </p>
          </div>
        </div>
      )}
    </div>
  )
}

