import { useState, useEffect } from 'react'
import AutoResizeTextarea from '../../ui/AutoResizeTextarea'
import type {
  ActualGoalItemDTO,
  FinancialContextDTO,
  UpdateFinancialContextRequestDTO,
} from '../../../api/types'
import { formatTHB } from '../../../utils/formatters'

interface FinancialContextFormProps {
  financialContext: FinancialContextDTO | null
  portfolioId: string
  portfolioName?: string
  portfolioGoals?: ActualGoalItemDTO[]
  onSave: (updates: UpdateFinancialContextRequestDTO) => Promise<any>
  onSyncGoals?: () => Promise<any>
  onProceedToDraft: () => Promise<any> | void
  loading: boolean
  statusText?: string | null
}

export default function FinancialContextForm({
  financialContext,
  portfolioId,
  portfolioName,
  portfolioGoals = [],
  onSave,
  onSyncGoals,
  onProceedToDraft,
  loading,
  statusText,
}: FinancialContextFormProps) {
  const displayName = portfolioName || (portfolioId === 'default' ? 'พอร์ตลงทุนหลัก' : portfolioId)
  const matchedGoal = portfolioGoals.find((g) => g.portfolio_id === portfolioId)

  // Field States
  const [horizonYears, setHorizonYears] = useState(
    financialContext?.horizon_years ||
      (matchedGoal?.deadline_days_left
        ? String(Math.max(1, Math.round(matchedGoal.deadline_days_left / 365.25)))
        : '')
  )
  const [targetUseAmount, setTargetUseAmount] = useState(
    financialContext?.target_use_amount ||
      (matchedGoal?.target_amount_thb ? String(matchedGoal.target_amount_thb) : '')
  )
  const [targetUseTimeline, setTargetUseTimeline] = useState(
    financialContext?.target_use_timeline || matchedGoal?.name || '',
  )
  const [emergencyReservesAmount, setEmergencyReservesAmount] = useState(
    financialContext?.emergency_reserves_amount || '0',
  )
  const [emergencyReservesMonths, setEmergencyReservesMonths] = useState(
    financialContext?.emergency_reserves_months || '0',
  )
  const [obligationsMonthly, setObligationsMonthly] = useState(
    financialContext?.obligations_monthly || '0',
  )
  const [obligationsDescription, setObligationsDescription] = useState(
    financialContext?.obligations_description || '',
  )
  const [withdrawalFrequency, setWithdrawalFrequency] = useState(
    financialContext?.withdrawal_frequency || '',
  )
  const [withdrawalAmount, setWithdrawalAmount] = useState(
    financialContext?.withdrawal_amount || '',
  )
  const [experienceDescription, setExperienceDescription] = useState(
    financialContext?.experience_description || '',
  )
  const [monthlyInvestableContribution, setMonthlyInvestableContribution] = useState(
    financialContext?.monthly_investable_contribution || '',
  )
  const [essentialMonthlyExpense, setEssentialMonthlyExpense] = useState(
    financialContext?.essential_monthly_expense || '',
  )
  const [reservesOutsidePortfolio, setReservesOutsidePortfolio] = useState(
    financialContext?.reserves_outside_portfolio || '',
  )
  const [unknownFields, setUnknownFields] = useState<string[]>(
    financialContext?.unknown_fields || [],
  )

  // UI state for Option 1: Collapsible drawer for fine-tuning
  const [isEditing, setIsEditing] = useState(false)
  const [syncing, setSyncing] = useState(false)
  const [savedSuccess, setSavedSuccess] = useState(false)

  useEffect(() => {
    if (financialContext) {
      setHorizonYears(
        financialContext.horizon_years ||
          (matchedGoal?.deadline_days_left
            ? String(Math.max(1, Math.round(matchedGoal.deadline_days_left / 365.25)))
            : '')
      )
      setTargetUseAmount(
        financialContext.target_use_amount ||
          (matchedGoal?.target_amount_thb ? String(matchedGoal.target_amount_thb) : '')
      )
      setTargetUseTimeline(financialContext.target_use_timeline || matchedGoal?.name || '')
      setEmergencyReservesAmount(financialContext.emergency_reserves_amount || '0')
      setEmergencyReservesMonths(financialContext.emergency_reserves_months || '0')
      setObligationsMonthly(financialContext.obligations_monthly || '0')
      setObligationsDescription(financialContext.obligations_description || '')
      setWithdrawalFrequency(financialContext.withdrawal_frequency || '')
      setWithdrawalAmount(financialContext.withdrawal_amount || '')
      setExperienceDescription(financialContext.experience_description || '')
      setMonthlyInvestableContribution(financialContext.monthly_investable_contribution || '')
      setEssentialMonthlyExpense(financialContext.essential_monthly_expense || '')
      setReservesOutsidePortfolio(financialContext.reserves_outside_portfolio || '')
      setUnknownFields(financialContext.unknown_fields || [])
    }
  }, [financialContext, matchedGoal])

  const handleSave = async () => {
    const cleanHorizon = (horizonYears || '').replace(/[^0-9.]/g, '') || (matchedGoal?.deadline_days_left ? String(Math.max(1, Math.round(matchedGoal.deadline_days_left / 365.25))) : '')
    const filteredUnknown = (unknownFields || []).filter(
      (f) => f !== 'horizon_years' && f !== 'emergency_reserves' && f !== 'obligations',
    )
    await onSave({
      horizon_years: cleanHorizon,
      target_use_amount: targetUseAmount || (matchedGoal?.target_amount_thb ? String(matchedGoal.target_amount_thb) : null),
      target_use_timeline: targetUseTimeline || matchedGoal?.name || null,
      emergency_reserves_amount: emergencyReservesAmount || '0',
      emergency_reserves_months: emergencyReservesMonths || '0',
      obligations_monthly: obligationsMonthly || '0',
      obligations_description: obligationsDescription || null,
      withdrawal_frequency: withdrawalFrequency || null,
      withdrawal_amount: withdrawalAmount || null,
      experience_description: experienceDescription || null,
      monthly_investable_contribution: monthlyInvestableContribution || null,
      essential_monthly_expense: essentialMonthlyExpense || null,
      reserves_outside_portfolio: reservesOutsidePortfolio || null,
      unknown_fields: filteredUnknown,
    })
    setSavedSuccess(true)
    setTimeout(() => setSavedSuccess(false), 2500)
  }

  const handleSyncGoals = async () => {
    if (!onSyncGoals) return
    setSyncing(true)
    try {
      await onSyncGoals()
    } catch {
      // Handled by hook error state
    } finally {
      setSyncing(false)
    }
  }

  const handleProceed = async () => {
    try {
      await handleSave()
      await onProceedToDraft()
    } catch {
      // Keep inputs if error occurs
    }
  }

  // Format display numbers cleanly
  const formatAmountDisplay = (val: string | null | undefined, fallback: string = '0 บาท') => {
    if (!val) return fallback
    const num = parseFloat(val.replace(/,/g, ''))
    if (isNaN(num)) return val
    return formatTHB(num)
  }

  // Linked goal display text
  const displayGoalName = targetUseTimeline || matchedGoal?.name || 'ยังไม่มีการผูกเป้าหมายเฉพาะพอร์ตนี้'

  const readinessIssues = financialContext?.readiness_issues ?? []
  const isReady = financialContext?.is_ready_for_numeric_policy ?? true

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      {/* Smart Context Header Card */}
      <div className="rounded-2xl border border-sky-100 bg-white p-6 sm:p-8 shadow-sm space-y-6">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-sky-50 pb-5">
          <div className="flex items-center gap-3.5">
            <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-gradient-to-br from-sky-500 to-blue-600 text-white font-extrabold text-xl shadow-sm shadow-sky-500/20">
              🎯
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h2 className="text-lg sm:text-xl font-extrabold text-zinc-900 tracking-tight">
                  บริบททางการเงินเฉพาะพอร์ตนี้ (Financial Context)
                </h2>
                <span className="rounded-full bg-sky-50 border border-sky-200/80 px-2.5 py-0.5 text-[11px] font-bold text-sky-700">
                  พอร์ต: {displayName}
                </span>
              </div>
              <p className="text-xs text-zinc-500 mt-1">
                สังเคราะห์จากเป้าหมายพอร์ต (Goals) และเงินสดคงเหลือเพื่อคำนวณ MDD
                และสัดส่วนที่ปลอดภัยต่อชีวิตจริง
              </p>
            </div>
          </div>

          {/* Sync Button */}
          {onSyncGoals && (
            <button
              type="button"
              onClick={handleSyncGoals}
              disabled={loading || syncing}
              className="inline-flex items-center gap-1.5 rounded-xl border border-sky-200 bg-sky-50/70 px-3.5 py-2 text-xs font-bold text-sky-700 hover:bg-sky-100 hover:text-sky-800 active:scale-95 transition-all cursor-pointer disabled:opacity-50 self-start sm:self-auto shrink-0 shadow-2xs"
              title={`ดึงเป้าหมายล่าสุดจากระบบ Portfolio Goals เฉพาะพอร์ต ${displayName}`}
            >
              <span className={syncing ? 'animate-spin' : ''}>🔄</span>
              <span>{syncing ? 'กำลังซิงค์...' : 'ดึงข้อมูลจาก Goals ใหม่'}</span>
            </button>
          )}
        </div>

        {/* Status / Insight Banner */}
        {isReady ? (
          <div className="rounded-xl border border-emerald-200/80 bg-emerald-50/70 p-4 flex items-center justify-between gap-3 shadow-2xs">
            <div className="flex items-center gap-3">
              <span className="flex h-7 w-7 items-center justify-center rounded-lg bg-emerald-100 text-emerald-700 text-sm">
                ✓
              </span>
              <div>
                <span className="text-xs font-bold text-emerald-900">
                  สังเคราะห์บริบทครบถ้วน พร้อมสำหรับการสร้างแกนหลัก (Investment Axis)
                </span>
                <p className="text-[11px] text-emerald-700 mt-0.5">
                  ระบบจัดเตรียมตัวเลขเป้าหมาย กรอบเวลา และเงินสำรองให้แล้ว
                  สามารถคลิกเพื่อร่างแกนหลักได้ทันที
                </p>
              </div>
            </div>
            <span className="hidden sm:inline-block rounded-full bg-emerald-100 text-emerald-800 text-[10px] font-extrabold px-2.5 py-0.5">
              พร้อม 100%
            </span>
          </div>
        ) : (
          <div className="rounded-xl border border-amber-200 bg-amber-50/70 p-4 space-y-1.5">
            <div className="flex items-center gap-2 text-amber-900 font-bold text-xs">
              <span>⚠️</span>
              <span>ประเด็นที่แนะนำให้ระบุก่อนกำหนดตัวเลขความเสี่ยง:</span>
            </div>
            <ul className="text-xs text-amber-800 list-disc list-inside space-y-0.5">
              {readinessIssues.map((issue, idx) => (
                <li key={idx}>
                  {issue.message || issue.issue} (
                  {(issue.severity || issue.level) === 'blocking' ? 'จำเป็น' : 'แนะนำ'})
                </li>
              ))}
            </ul>
          </div>
        )}

        {/* 4 Smart Summary Metric Cards */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3.5 pt-1">
          {/* Card 1: Target Wealth */}
          <div className="rounded-2xl border border-sky-100 bg-sky-50/30 p-4 space-y-2 hover:border-sky-200 transition-all">
            <div className="flex items-center justify-between">
              <span className="text-xs font-bold text-zinc-600 flex items-center gap-1.5">
                <span>🎯</span> เป้าหมายพอร์ต
              </span>
              <span className="text-[10px] font-bold px-1.5 py-0.5 rounded-md bg-sky-100 text-sky-800">
                Goals
              </span>
            </div>
            <div className="text-lg font-black text-zinc-900 tracking-tight">
              {targetUseAmount
                ? formatAmountDisplay(targetUseAmount)
                : matchedGoal?.target_amount_thb
                ? formatTHB(matchedGoal.target_amount_thb)
                : 'ยังไม่ระบุเป้าหมาย'}
            </div>
            <p className="text-[11px] text-zinc-500 truncate" title={displayGoalName}>
              {displayGoalName}
            </p>
          </div>

          {/* Card 2: Investment Horizon */}
          <div className="rounded-2xl border border-sky-100 bg-sky-50/30 p-4 space-y-2 hover:border-sky-200 transition-all">
            <div className="flex items-center justify-between">
              <span className="text-xs font-bold text-zinc-600 flex items-center gap-1.5">
                <span>⏳</span> กรอบเวลาลงทุน
              </span>
              <span className="text-[10px] font-bold px-1.5 py-0.5 rounded-md bg-indigo-100 text-indigo-800">
                Horizon
              </span>
            </div>
            <div className="text-lg font-black text-zinc-900 tracking-tight">
              {horizonYears ? `${horizonYears} ปี` : 'ยังไม่ระบุ'}
            </div>
            <p className="text-[11px] text-zinc-500">
              {horizonYears && parseInt(horizonYears) > 5
                ? 'เน้นพลังทบต้นระยะยาว'
                : horizonYears && parseInt(horizonYears) > 0
                ? 'กรอบเวลาปานกลาง'
                : 'รอระบุกรอบเวลา'}
            </p>
          </div>

          {/* Card 3: Emergency Reserves */}
          <div className="rounded-2xl border border-sky-100 bg-sky-50/30 p-4 space-y-2 hover:border-sky-200 transition-all">
            <div className="flex items-center justify-between">
              <span className="text-xs font-bold text-zinc-600 flex items-center gap-1.5">
                <span>🛡️</span> เงินสำรองฉุกเฉิน
              </span>
              <span className="text-[10px] font-bold px-1.5 py-0.5 rounded-md bg-emerald-100 text-emerald-800">
                Safety
              </span>
            </div>
            <div className="text-lg font-black text-zinc-900 tracking-tight">
              {emergencyReservesAmount && emergencyReservesAmount !== '0'
                ? formatAmountDisplay(emergencyReservesAmount)
                : reservesOutsidePortfolio
                ? 'สำรองนอกพอร์ต'
                : '0.00 ฿'}
            </div>
            <p className="text-[11px] text-zinc-500 truncate" title={reservesOutsidePortfolio || (emergencyReservesMonths && emergencyReservesMonths !== '0' ? `ครอบคลุม ~${emergencyReservesMonths} เดือนของค่าใช้จ่าย` : 'ไม่มีเงินสำรองในพอร์ตนี้')}>
              {reservesOutsidePortfolio || (emergencyReservesMonths && emergencyReservesMonths !== '0'
                ? `ครอบคลุม ~${emergencyReservesMonths} เดือนของค่าใช้จ่าย`
                : 'ไม่มีเงินสำรองในพอร์ตนี้')}
            </p>
          </div>

          {/* Card 4: Cash Flow Strategy */}
          <div className="rounded-2xl border border-sky-100 bg-sky-50/30 p-4 space-y-2 hover:border-sky-200 transition-all">
            <div className="flex items-center justify-between">
              <span className="text-xs font-bold text-zinc-600 flex items-center gap-1.5">
                <span>💧</span> แผนสภาพคล่อง
              </span>
              <span className="text-[10px] font-bold px-1.5 py-0.5 rounded-md bg-purple-100 text-purple-800">
                Cash Flow
              </span>
            </div>
            <div className="text-sm font-bold text-zinc-900 truncate">
              {withdrawalFrequency || 'ไม่มีการถอน'}
            </div>
            <p className="text-[11px] text-zinc-500">
              {monthlyInvestableContribution
                ? `DCA ${formatAmountDisplay(monthlyInvestableContribution)}/ด.`
                : 'สะสมความมั่งคั่งต่อเนื่อง'}
            </p>
          </div>
        </div>

        {/* Primary Action Button Bar */}
        <div className="pt-4 border-t border-sky-50 flex flex-col items-center justify-center gap-3">
          <button
            type="button"
            onClick={handleProceed}
            disabled={loading}
            className="w-full sm:w-auto min-w-[320px] rounded-2xl bg-gradient-to-r from-sky-600 via-blue-600 to-indigo-600 px-8 py-4 text-sm font-extrabold text-white shadow-md shadow-sky-500/25 hover:from-sky-700 hover:to-indigo-700 active:scale-98 transition-all cursor-pointer text-center inline-flex items-center justify-center gap-2"
          >
            {loading ? (
              <span>{statusText || 'กำลังประมวลผล...'}</span>
            ) : (
              <span>🚀 ข้อมูลครบถ้วน — ร่างแกนหลักครบ 8 หัวข้อทันที</span>
            )}
          </button>

          {/* Toggle fine-tuning accordion */}
          <button
            type="button"
            onClick={() => setIsEditing((prev) => !prev)}
            className="text-xs font-semibold text-zinc-500 hover:text-sky-600 transition-colors cursor-pointer py-1 flex items-center gap-1"
          >
            <span>{isEditing ? '▲ ปิดแผงปรับแต่งตัวเลข' : '✏️ ปรับแต่งตัวเลขเฉพาะจุด (หากต้องการแก้ไขเพิ่มเติม) ▼'}</span>
          </button>

          {statusText && (
            <p className="text-xs text-sky-700 font-semibold animate-pulse">{statusText}</p>
          )}
          {savedSuccess && (
            <p className="text-xs text-emerald-700 font-bold">✓ บันทึกการแก้ไขเรียบร้อย</p>
          )}
        </div>

        {/* Collapsible Fine-Tuning Drawer */}
        {isEditing && (
          <div className="rounded-2xl border border-sky-200 bg-sky-50/20 p-6 space-y-6 animate-fadeIn mt-4">
            <div className="flex items-center justify-between border-b border-sky-100 pb-3">
              <h3 className="text-xs font-bold text-zinc-800">
                ⚙️ ปรับแต่งตัวเลขบริบททางการเงินเฉพาะพอร์ตนี้
              </h3>
              <span className="text-[11px] text-zinc-500">
                แก้ไขเฉพาะฟิลด์ที่ต้องการ แล้วกดบันทึก
              </span>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-5">
              {/* Group 1: Horizon & Target */}
              <div className="space-y-3.5 bg-white p-4 rounded-xl border border-sky-100 shadow-2xs">
                <h4 className="text-xs font-bold text-zinc-800">⏳ 1. กรอบเวลาและยอดเงิน</h4>

                <div>
                  <label htmlFor="fc-horizon-years" className="block text-[11px] font-semibold text-zinc-700 mb-1">
                    กรอบเวลาลงทุนของพอร์ตนี้ (ปี):
                  </label>
                  <AutoResizeTextarea
                    id="fc-horizon-years"
                    minRows={1}
                    value={horizonYears}
                    onChange={(e) => setHorizonYears(e.target.value)}
                    placeholder="เช่น 10 ปี"
                    className="rounded-lg border border-zinc-200 bg-white p-2 text-xs text-zinc-800 focus:outline-none focus:ring-2 focus:ring-sky-400"
                  />
                </div>

                <div>
                  <label htmlFor="fc-target-amount" className="block text-[11px] font-semibold text-zinc-700 mb-1">
                    ยอดเงินเป้าหมาย (บาท):
                  </label>
                  <input
                    id="fc-target-amount"
                    type="text"
                    value={targetUseAmount}
                    onChange={(e) => setTargetUseAmount(e.target.value)}
                    placeholder="เช่น 10,000,000"
                    className="w-full rounded-lg border border-zinc-200 bg-white p-2 text-xs text-zinc-800 focus:outline-none focus:ring-2 focus:ring-sky-400"
                  />
                </div>

                <div>
                  <label htmlFor="fc-target-timeline" className="block text-[11px] font-semibold text-zinc-700 mb-1">
                    เป้าหมาย / วัตถุประสงค์การใช้เงิน:
                  </label>
                  <AutoResizeTextarea
                    id="fc-target-timeline"
                    minRows={2}
                    value={targetUseTimeline}
                    onChange={(e) => setTargetUseTimeline(e.target.value)}
                    placeholder="เช่น พอร์ต 10 ล้านบาท / เพื่อการเกษียณ"
                    className="rounded-lg border border-zinc-200 bg-white p-2 text-xs text-zinc-800 focus:outline-none focus:ring-2 focus:ring-sky-400"
                  />
                </div>
              </div>

              {/* Group 2: Reserves & Cashflow */}
              <div className="space-y-3.5 bg-white p-4 rounded-xl border border-sky-100 shadow-2xs">
                <h4 className="text-xs font-bold text-zinc-800">🛡️ 2. เงินสำรองและสภาพคล่อง</h4>

                <div>
                  <label htmlFor="fc-reserves-amount" className="block text-[11px] font-semibold text-zinc-700 mb-1">
                    จำนวนเงินสำรองฉุกเฉิน (บาท):
                  </label>
                  <input
                    id="fc-reserves-amount"
                    type="text"
                    value={emergencyReservesAmount}
                    onChange={(e) => setEmergencyReservesAmount(e.target.value)}
                    placeholder="เช่น 200,000"
                    className="w-full rounded-lg border border-zinc-200 bg-white p-2 text-xs text-zinc-800 focus:outline-none focus:ring-2 focus:ring-sky-400"
                  />
                </div>

                <div>
                  <label htmlFor="fc-reserves-months" className="block text-[11px] font-semibold text-zinc-700 mb-1">
                    เงินสำรองฉุกเฉิน (จำนวนเดือน):
                  </label>
                  <input
                    id="fc-reserves-months"
                    type="text"
                    value={emergencyReservesMonths}
                    onChange={(e) => setEmergencyReservesMonths(e.target.value)}
                    placeholder="เช่น 6"
                    className="w-full rounded-lg border border-zinc-200 bg-white p-2 text-xs text-zinc-800 focus:outline-none focus:ring-2 focus:ring-sky-400"
                  />
                </div>

                <div>
                  <label htmlFor="fc-monthly-contrib" className="block text-[11px] font-semibold text-zinc-700 mb-1">
                    เงินออม/สมทบลงทุนต่อเดือน (DCA/บาท):
                  </label>
                  <input
                    id="fc-monthly-contrib"
                    type="text"
                    value={monthlyInvestableContribution}
                    onChange={(e) => setMonthlyInvestableContribution(e.target.value)}
                    placeholder="เช่น 10,000 หรือ เว้นว่างถ้าไม่แน่นอน"
                    className="w-full rounded-lg border border-zinc-200 bg-white p-2 text-xs text-zinc-800 focus:outline-none focus:ring-2 focus:ring-sky-400"
                  />
                </div>
              </div>
            </div>

            {/* Save Button inside drawer */}
            <div className="flex justify-end gap-3 pt-2">
              <button
                type="button"
                onClick={() => setIsEditing(false)}
                className="rounded-xl border border-zinc-200 bg-white px-4 py-2 text-xs font-semibold text-zinc-600 hover:bg-zinc-50 cursor-pointer"
              >
                ปิดแผงแก้ไข
              </button>
              <button
                type="button"
                onClick={handleSave}
                disabled={loading}
                className="rounded-xl bg-sky-600 px-5 py-2 text-xs font-bold text-white hover:bg-sky-700 cursor-pointer shadow-2xs"
              >
                💾 บันทึกการแก้ไขตัวเลข
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
