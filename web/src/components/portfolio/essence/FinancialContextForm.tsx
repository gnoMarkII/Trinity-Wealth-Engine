import { useState, useEffect } from 'react'
import type { FinancialContextDTO, UpdateFinancialContextRequestDTO } from '../../../api/types'

interface FinancialContextFormProps {
  financialContext: FinancialContextDTO | null
  portfolioId: string
  onSave: (updates: UpdateFinancialContextRequestDTO) => Promise<void>
  onProceedToDraft: () => Promise<void>
  loading: boolean
  statusText?: string | null
}

export default function FinancialContextForm({
  financialContext,
  portfolioId,
  onSave,
  onProceedToDraft,
  loading,
  statusText,
}: FinancialContextFormProps) {
  const [horizonYears, setHorizonYears] = useState(financialContext?.horizon_years || '')
  const [targetUseAmount, setTargetUseAmount] = useState(financialContext?.target_use_amount || '')
  const [targetUseTimeline, setTargetUseTimeline] = useState(financialContext?.target_use_timeline || '')
  const [emergencyReservesAmount, setEmergencyReservesAmount] = useState(
    financialContext?.emergency_reserves_amount || '',
  )
  const [emergencyReservesMonths, setEmergencyReservesMonths] = useState(
    financialContext?.emergency_reserves_months || '',
  )
  const [obligationsMonthly, setObligationsMonthly] = useState(
    financialContext?.obligations_monthly || '',
  )
  const [obligationsDescription, setObligationsDescription] = useState(
    financialContext?.obligations_description || '',
  )
  const [withdrawalFrequency, setWithdrawalFrequency] = useState(
    financialContext?.withdrawal_frequency || '',
  )
  const [experienceDescription, setExperienceDescription] = useState(
    financialContext?.experience_description || '',
  )
  const [unknownFields, setUnknownFields] = useState<string[]>(
    financialContext?.unknown_fields || [],
  )

  useEffect(() => {
    if (financialContext) {
      setHorizonYears(financialContext.horizon_years || '')
      setTargetUseAmount(financialContext.target_use_amount || '')
      setTargetUseTimeline(financialContext.target_use_timeline || '')
      setEmergencyReservesAmount(financialContext.emergency_reserves_amount || '')
      setEmergencyReservesMonths(financialContext.emergency_reserves_months || '')
      setObligationsMonthly(financialContext.obligations_monthly || '')
      setObligationsDescription(financialContext.obligations_description || '')
      setWithdrawalFrequency(financialContext.withdrawal_frequency || '')
      setExperienceDescription(financialContext.experience_description || '')
      setUnknownFields(financialContext.unknown_fields || [])
    }
  }, [financialContext])

  const toggleUnknown = (fieldKey: string) => {
    setUnknownFields((prev) =>
      prev.includes(fieldKey) ? prev.filter((f) => f !== fieldKey) : [...prev, fieldKey],
    )
  }

  const handleSave = async () => {
    await onSave({
      horizon_years: horizonYears || null,
      target_use_amount: targetUseAmount || null,
      target_use_timeline: targetUseTimeline || null,
      emergency_reserves_amount: emergencyReservesAmount || null,
      emergency_reserves_months: emergencyReservesMonths || null,
      obligations_monthly: obligationsMonthly || null,
      obligations_description: obligationsDescription || null,
      withdrawal_frequency: withdrawalFrequency || null,
      experience_description: experienceDescription || null,
      unknown_fields: unknownFields,
    })
  }

  const readinessIssues = financialContext?.readiness_issues ?? []
  const isReady = financialContext?.is_ready_for_numeric_policy ?? false

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      {/* Header */}
      <div className="rounded-2xl border border-sky-100 bg-white p-6 sm:p-8 shadow-sm">
        <div className="flex items-center gap-3 border-b border-sky-50 pb-4">
          <span className="flex h-10 w-10 items-center justify-center rounded-xl bg-sky-100 text-sky-700 font-bold text-lg">
            📊
          </span>
          <div>
            <h2 className="text-lg sm:text-xl font-bold text-zinc-900">
              บริบททางการเงินเฉพาะพอร์ตนี้ (Financial Context)
            </h2>
            <p className="text-xs text-zinc-500 mt-0.5">
              พอร์ต: <span className="font-semibold text-zinc-700">{portfolioId}</span> | ข้อมูลจำเป็นเพื่อใช้คำนวณ MDD, ขีดจำกัดผลขาดทุน และสัดส่วนจัดสรรที่ปลอดภัยต่อชีวิตจริง
            </p>
          </div>
        </div>

        {/* Readiness Status Banner */}
        <div className="mt-5">
          {isReady ? (
            <div className="rounded-xl border border-emerald-200 bg-emerald-50/70 p-4 flex items-center gap-3">
              <span className="text-lg">✅</span>
              <div>
                <span className="text-xs font-bold text-emerald-800">
                  ข้อมูลบริบทครบถ้วน พร้อมสำหรับการสร้างแกนหลักแบบมีตัวเลขรูปธรรม
                </span>
                <p className="text-[11px] text-emerald-700 mt-0.5">
                  ระบบมีกรอบเวลา เงินสำรอง และภาระใช้เงินเพียงพอในการเสนอ MDD และสัดส่วนสินทรัพย์
                </p>
              </div>
            </div>
          ) : (
            <div className="rounded-xl border border-amber-200 bg-amber-50/70 p-4 space-y-1.5">
              <div className="flex items-center gap-2 text-amber-900 font-bold text-xs">
                <span>⚠️</span>
                <span>ประเด็นที่ต้องระบุก่อนกำหนดตัวเลขความเสี่ยง:</span>
              </div>
              <ul className="text-xs text-amber-800 list-disc list-inside space-y-0.5">
                {readinessIssues.map((issue, idx) => (
                  <li key={idx}>
                    {issue.issue} ({issue.level === 'blocking' ? 'จำเป็น' : 'แนะนำ'})
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>

        {/* Form Fields */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mt-6 pt-2">
          {/* 1. Horizon & Target Use */}
          <div className="space-y-4 p-4 rounded-xl bg-sky-50/30 border border-sky-100/80">
            <h3 className="text-sm font-bold text-zinc-800 flex items-center justify-between">
              <span>⏳ 1. กรอบเวลาและแผนใช้เงิน</span>
              <button
                type="button"
                onClick={() => toggleUnknown('horizon_years')}
                className={`text-[11px] font-semibold px-2 py-0.5 rounded-md border ${
                  unknownFields.includes('horizon_years')
                    ? 'border-amber-300 bg-amber-100 text-amber-800'
                    : 'border-zinc-200 text-zinc-500 hover:bg-zinc-100'
                }`}
              >
                {unknownFields.includes('horizon_years') ? '✓ ยังไม่ทราบ' : 'ยังไม่ทราบ'}
              </button>
            </h3>

            <div>
              <label htmlFor="fc-horizon-years" className="block text-xs font-semibold text-zinc-700 mb-1">
                กรอบเวลาลงทุนของพอร์ตนี้ (ปี):
              </label>
              <input
                id="fc-horizon-years"
                type="text"
                disabled={unknownFields.includes('horizon_years')}
                value={horizonYears}
                onChange={(e) => setHorizonYears(e.target.value)}
                placeholder="เช่น 5-10 ปี หรือ มากกว่า 10 ปี"
                className="w-full rounded-lg border border-zinc-200 bg-white p-2.5 text-xs text-zinc-800 focus:outline-none focus:ring-2 focus:ring-sky-400 disabled:bg-zinc-100"
              />
            </div>

            <div>
              <label htmlFor="fc-target-amount" className="block text-xs font-semibold text-zinc-700 mb-1">
                ยอดเงินที่ต้องใช้ตามกำหนด (ถ้ามี):
              </label>
              <input
                id="fc-target-amount"
                type="text"
                value={targetUseAmount}
                onChange={(e) => setTargetUseAmount(e.target.value)}
                placeholder="เช่น 500,000 บาท หรือ ไม่มีกำหนดแน่นอน"
                className="w-full rounded-lg border border-zinc-200 bg-white p-2.5 text-xs text-zinc-800 focus:outline-none focus:ring-2 focus:ring-sky-400"
              />
            </div>

            <div>
              <label htmlFor="fc-target-timeline" className="block text-xs font-semibold text-zinc-700 mb-1">
                ช่วงเวลาที่จะต้องใช้เงินนี้:
              </label>
              <input
                id="fc-target-timeline"
                type="text"
                value={targetUseTimeline}
                onChange={(e) => setTargetUseTimeline(e.target.value)}
                placeholder="เช่น อีก 3 ปีข้างหน้า หรือ เพื่อการเกษียณ"
                className="w-full rounded-lg border border-zinc-200 bg-white p-2.5 text-xs text-zinc-800 focus:outline-none focus:ring-2 focus:ring-sky-400"
              />
            </div>
          </div>

          {/* 2. Reserves & Obligations */}
          <div className="space-y-4 p-4 rounded-xl bg-sky-50/30 border border-sky-100/80">
            <h3 className="text-sm font-bold text-zinc-800 flex items-center justify-between">
              <span>🛡️ 2. เงินสำรองและภาระผูกพัน</span>
              <button
                type="button"
                onClick={() => toggleUnknown('emergency_reserves_months')}
                className={`text-[11px] font-semibold px-2 py-0.5 rounded-md border ${
                  unknownFields.includes('emergency_reserves_months')
                    ? 'border-amber-300 bg-amber-100 text-amber-800'
                    : 'border-zinc-200 text-zinc-500 hover:bg-zinc-100'
                }`}
              >
                {unknownFields.includes('emergency_reserves_months') ? '✓ ยังไม่ทราบ' : 'ยังไม่ทราบ'}
              </button>
            </h3>

            <div>
              <label htmlFor="fc-reserves-months" className="block text-xs font-semibold text-zinc-700 mb-1">
                เงินสำรองฉุกเฉินที่มีอยู่ (เดือนของค่าใช้จ่าย):
              </label>
              <input
                id="fc-reserves-months"
                type="text"
                disabled={unknownFields.includes('emergency_reserves_months')}
                value={emergencyReservesMonths}
                onChange={(e) => setEmergencyReservesMonths(e.target.value)}
                placeholder="เช่น 6 เดือน หรือ 12 เดือน"
                className="w-full rounded-lg border border-zinc-200 bg-white p-2.5 text-xs text-zinc-800 focus:outline-none focus:ring-2 focus:ring-sky-400 disabled:bg-zinc-100"
              />
            </div>

            <div>
              <label htmlFor="fc-reserves-amount" className="block text-xs font-semibold text-zinc-700 mb-1">
                จำนวนเงินสำรองฉุกเฉิน (บาท):
              </label>
              <input
                id="fc-reserves-amount"
                type="text"
                value={emergencyReservesAmount}
                onChange={(e) => setEmergencyReservesAmount(e.target.value)}
                placeholder="เช่น 300,000 บาท"
                className="w-full rounded-lg border border-zinc-200 bg-white p-2.5 text-xs text-zinc-800 focus:outline-none focus:ring-2 focus:ring-sky-400"
              />
            </div>

            <div>
              <label htmlFor="fc-obligations" className="block text-xs font-semibold text-zinc-700 mb-1">
                ภาระผูกพันหรือค่าใช้จ่ายประจำเดือน (บาท/เดือน):
              </label>
              <input
                id="fc-obligations"
                type="text"
                value={obligationsMonthly}
                onChange={(e) => setObligationsMonthly(e.target.value)}
                placeholder="เช่น 20,000 บาท/เดือน"
                className="w-full rounded-lg border border-zinc-200 bg-white p-2.5 text-xs text-zinc-800 focus:outline-none focus:ring-2 focus:ring-sky-400"
              />
            </div>
          </div>

          {/* 3. Withdrawal & Experience */}
          <div className="space-y-4 p-4 rounded-xl bg-sky-50/30 border border-sky-100/80 md:col-span-2">
            <h3 className="text-sm font-bold text-zinc-800">
              💧 3. แผนการถอนเงินและประสบการณ์ลงทุน
            </h3>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <div>
                <label htmlFor="fc-withdrawal-freq" className="block text-xs font-semibold text-zinc-700 mb-1">
                  ความถี่หรือเงื่อนไขการถอนเงินจากพอร์ต:
                </label>
                <input
                  id="fc-withdrawal-freq"
                  type="text"
                  value={withdrawalFrequency}
                  onChange={(e) => setWithdrawalFrequency(e.target.value)}
                  placeholder="เช่น ไม่มีการถอนเงินระหว่างทาง หรือ ถอนเงินปันผลทุกไตรมาส"
                  className="w-full rounded-lg border border-zinc-200 bg-white p-2.5 text-xs text-zinc-800 focus:outline-none focus:ring-2 focus:ring-sky-400"
                />
              </div>

              <div>
                <label htmlFor="fc-experience" className="block text-xs font-semibold text-zinc-700 mb-1">
                  ประสบการณ์และสินทรัพย์ที่เคยลงทุน:
                </label>
                <input
                  id="fc-experience"
                  type="text"
                  value={experienceDescription}
                  onChange={(e) => setExperienceDescription(e.target.value)}
                  placeholder="เช่น กองทุนรวมดัชนี, หุ้นปันผลไทย, ETF สหรัฐฯ"
                  className="w-full rounded-lg border border-zinc-200 bg-white p-2.5 text-xs text-zinc-800 focus:outline-none focus:ring-2 focus:ring-sky-400"
                />
              </div>
            </div>
          </div>
        </div>

        {/* Action Buttons */}
        <div className="pt-6 border-t border-sky-100 flex flex-col sm:flex-row items-center justify-between gap-4 mt-6">
          <div className="text-xs text-zinc-500">
            {statusText ? (
              <span className="flex items-center gap-2 text-sky-700 font-semibold animate-pulse">
                <span className="h-2 w-2 rounded-full bg-sky-500" />
                {statusText}
              </span>
            ) : (
              <span>ข้อมูลจะถูกบันทึกเป็น Snapshot เฉพาะพอร์ตนี้เพื่อใช้เป็นหลักฐาน</span>
            )}
          </div>

          <div className="flex items-center gap-3 w-full sm:w-auto">
            <button
              type="button"
              onClick={handleSave}
              disabled={loading}
              className="rounded-xl border border-sky-300 bg-sky-50 px-4 py-2.5 text-xs sm:text-sm font-bold text-sky-700 hover:bg-sky-100 active:scale-98 transition-all"
            >
              💾 บันทึกบริบทไว้
            </button>

            <button
              type="button"
              onClick={async () => {
                await handleSave()
                await onProceedToDraft()
              }}
              disabled={loading}
              className="rounded-xl bg-gradient-to-r from-sky-600 to-blue-600 px-6 py-2.5 text-xs sm:text-sm font-bold text-white shadow-md shadow-sky-500/20 hover:from-sky-700 hover:to-blue-700 active:scale-98 transition-all"
            >
              {loading ? 'กำลังประมวลผล...' : 'ร่างแกนหลักครบ 8 หัวข้อ →'}
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}
