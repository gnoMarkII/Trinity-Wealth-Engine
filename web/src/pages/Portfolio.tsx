import { Link } from 'react-router-dom'

export default function Portfolio() {
  return (
    <div className="animate-page-in space-y-6 pb-12">
      {/* Banner แนะนำไปยังหน้า Macro Strategy Report */}
      <div className="rounded-2xl border border-sky-100 bg-gradient-to-r from-sky-50/90 via-white/85 to-orange-50/65 p-6 shadow-[0_10px_35px_rgba(14,165,233,0.06)] backdrop-blur-sm">
        <div className="flex flex-col items-start justify-between gap-4 sm:flex-row sm:items-center">
          <div className="space-y-1">
            <h2 className="text-base font-semibold text-zinc-900">
              มองหารายงานกลยุทธ์การจัดสรรสินทรัพย์ (Macro Strategy Direction)?
            </h2>
            <p className="text-sm text-zinc-600">
              รายงานคำแนะนำสัดส่วน Cross-Asset Allocation, Pair Trades และ Hedging Scenarios ถูกย้ายไปรวมไว้ที่หน้า
              Macro Strategy Report แล้ว เพื่อให้อ่านควบคู่กับสภาวะเศรษฐกิจได้อย่างต่อเนื่องในที่เดียว
            </p>
          </div>
          <Link
            to="/macro"
            className="shrink-0 rounded-xl bg-flow-blue px-4 py-2.5 text-xs font-semibold text-white shadow-[0_8px_22px_rgba(14,165,233,0.18)] transition-all hover:bg-sky-600 hover:shadow"
          >
            ไปที่รายงาน Macro Strategy →
          </Link>
        </div>
      </div>

      {/* Actual Portfolio Holdings & Bookkeeper Tracker Placeholder */}
      <div className="flow-panel rounded-2xl p-8 text-center">
        <div className="mx-auto max-w-md space-y-4">
          <div className="mx-auto flex h-12 w-12 items-center justify-center rounded-2xl bg-sky-50 text-sky-700">
            <svg
              className="h-6 w-6"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
              strokeWidth={1.75}
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                d="M3 13.125C3 12.504 3.504 12 4.125 12h2.25c.621 0 1.125.504 1.125 1.125v6.75C7.5 20.496 6.996 21 6.375 21h-2.25A1.125 1.125 0 013 19.875v-6.75zM9.75 8.625c0-.621.504-1.125 1.125-1.125h2.25c.621 0 1.125.504 1.125 1.125v11.25c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 01-1.125-1.125V8.625zM16.5 4.125c0-.621.504-1.125 1.125-1.125h2.25C19.379 3 19.875 3.504 19.875 4.125v15.75c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 01-1.125-1.125V4.125z"
              />
            </svg>
          </div>
          <h3 className="text-lg font-bold text-zinc-900">Actual Portfolio & Bookkeeper Tracker</h3>
          <p className="text-sm leading-relaxed text-zinc-500">
            พื้นที่สำหรับติดตามสถานะพอร์ตจริง รายการสินทรัพย์ที่ถือครอง (Holdings) และประวัติธุรกรรมจาก The Bookkeeper
            กำลังอยู่ระหว่างเตรียมความพร้อมสำหรับการเชื่อมต่อในเฟสถัดไป
          </p>
          <div className="pt-2">
            <span className="inline-block rounded-full border border-sky-100 bg-white/75 px-3 py-1 text-xs font-medium text-zinc-500">
              สถานะ: รอเชื่อมต่อข้อมูล Holdings จาก Obsidian Vault
            </span>
          </div>
        </div>
      </div>
    </div>
  )
}
