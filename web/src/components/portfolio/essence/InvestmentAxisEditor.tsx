import { useState, useEffect, useMemo } from 'react'
import type {
  AxisDraftDTO,
  ConfirmedAxisDTO,
  UpdateAxisDraftRequestDTO,
  AllocationRowDTO,
} from '../../../api/types'

interface InvestmentAxisEditorProps {
  axisDraft: AxisDraftDTO | null
  confirmedAxis: ConfirmedAxisDTO | null
  portfolioId: string
  onUpdateDraft: (updates: UpdateAxisDraftRequestDTO) => Promise<void>
  onConfirmAxis: (proceedToBuckets: boolean) => Promise<void>
  onProceedToBuckets: () => void
  loading: boolean
  statusText?: string | null
}

function parseRiskLimitValue(val: unknown, fallback: string): string {
  if (val === null || val === undefined) return fallback
  if (typeof val === 'object' && val !== null && 'value' in val) {
    const inner = (val as { value?: unknown }).value
    return inner !== null && inner !== undefined && String(inner).trim() !== ''
      ? String(inner)
      : fallback
  }
  return String(val).trim() !== '' ? String(val) : fallback
}

export default function InvestmentAxisEditor({
  axisDraft,
  confirmedAxis,
  portfolioId,
  onUpdateDraft,
  onConfirmAxis,
  onProceedToBuckets,
  loading,
  statusText,
}: InvestmentAxisEditorProps) {
  const isConfirmed = !!confirmedAxis

  // Local editable state
  const [basicPolicy, setBasicPolicy] = useState(
    axisDraft?.basic_policy || confirmedAxis?.basic_policy || '',
  )
  const [mddPercent, setMddPercent] = useState(
    parseRiskLimitValue(
      axisDraft?.risk_limits?.mdd_max_annual ||
        axisDraft?.risk_limits?.mdd_percent ||
        confirmedAxis?.risk_limits?.mdd_max_annual ||
        confirmedAxis?.risk_limits?.mdd_percent,
      '15',
    ),
  )
  const [maxLossPerTrade, setMaxLossPerTrade] = useState(
    parseRiskLimitValue(
      axisDraft?.risk_limits?.max_loss_per_trade ||
        axisDraft?.risk_limits?.max_loss_per_trade_percent ||
        confirmedAxis?.risk_limits?.max_loss_per_trade ||
        confirmedAxis?.risk_limits?.max_loss_per_trade_percent,
      '2',
    ),
  )
  const [horizon, setHorizon] = useState(
    axisDraft?.investment_horizon || confirmedAxis?.investment_horizon || '',
  )
  const [rebalanceFreq, setRebalanceFreq] = useState(
    axisDraft?.rebalance_frequency || confirmedAxis?.rebalance_frequency || 'รายปี',
  )
  const [nonActions, setNonActions] = useState<string[]>(
    axisDraft?.non_actions || confirmedAxis?.non_actions || [],
  )
  const [newNonAction, setNewNonAction] = useState('')

  const [investTargets, setInvestTargets] = useState<string[]>(
    axisDraft?.invest_targets || confirmedAxis?.invest_targets || [],
  )
  const [excludeTargets, setExcludeTargets] = useState<string[]>(
    axisDraft?.exclude_targets || confirmedAxis?.exclude_targets || [],
  )
  const [primaryMethods, setPrimaryMethods] = useState<string[]>(
    axisDraft?.primary_methods || confirmedAxis?.primary_methods || [],
  )
  const [secondaryMethods, setSecondaryMethods] = useState<string[]>(
    axisDraft?.secondary_methods || confirmedAxis?.secondary_methods || [],
  )
  const [roleModels, setRoleModels] = useState<string[]>(
    axisDraft?.role_models || confirmedAxis?.role_models || [],
  )
  const [allocationRows, setAllocationRows] = useState<AllocationRowDTO[]>(
    axisDraft?.allocation_rows || confirmedAxis?.allocation_rows || [],
  )

  useEffect(() => {
    if (axisDraft) {
      setBasicPolicy(axisDraft.basic_policy || '')
      setMddPercent(
        parseRiskLimitValue(
          axisDraft.risk_limits?.mdd_max_annual || axisDraft.risk_limits?.mdd_percent,
          '15',
        ),
      )
      setMaxLossPerTrade(
        parseRiskLimitValue(
          axisDraft.risk_limits?.max_loss_per_trade ||
            axisDraft.risk_limits?.max_loss_per_trade_percent,
          '2',
        ),
      )
      setHorizon(axisDraft.investment_horizon || '')
      setRebalanceFreq(axisDraft.rebalance_frequency || 'รายปี')
      setNonActions(axisDraft.non_actions || [])
      setInvestTargets(axisDraft.invest_targets || [])
      setExcludeTargets(axisDraft.exclude_targets || [])
      setPrimaryMethods(axisDraft.primary_methods || [])
      setSecondaryMethods(axisDraft.secondary_methods || [])
      setRoleModels(axisDraft.role_models || [])
      setAllocationRows(axisDraft.allocation_rows || [])
    } else if (confirmedAxis) {
      setBasicPolicy(confirmedAxis.basic_policy || '')
      setMddPercent(
        parseRiskLimitValue(
          confirmedAxis.risk_limits?.mdd_max_annual || confirmedAxis.risk_limits?.mdd_percent,
          '15',
        ),
      )
      setMaxLossPerTrade(
        parseRiskLimitValue(
          confirmedAxis.risk_limits?.max_loss_per_trade ||
            confirmedAxis.risk_limits?.max_loss_per_trade_percent,
          '2',
        ),
      )
      setHorizon(confirmedAxis.investment_horizon || '')
      setRebalanceFreq(confirmedAxis.rebalance_frequency || 'รายปี')
      setNonActions(confirmedAxis.non_actions || [])
      setInvestTargets(confirmedAxis.invest_targets || [])
      setExcludeTargets(confirmedAxis.exclude_targets || [])
      setPrimaryMethods(confirmedAxis.primary_methods || [])
      setSecondaryMethods(confirmedAxis.secondary_methods || [])
      setRoleModels(confirmedAxis.role_models || [])
      setAllocationRows(confirmedAxis.allocation_rows || [])
    }
  }, [axisDraft, confirmedAxis])

  const handleAddNonAction = () => {
    if (newNonAction.trim() && !nonActions.includes(newNonAction.trim())) {
      setNonActions([...nonActions, newNonAction.trim()])
      setNewNonAction('')
    }
  }

  const handleRemoveNonAction = (idx: number) => {
    setNonActions(nonActions.filter((_, i) => i !== idx))
  }

  const handleSaveDraft = async () => {
    await onUpdateDraft({
      basic_policy: basicPolicy,
      risk_limits: {
        mdd_max_annual: { value: mddPercent, unit: 'percent', calculation_basis: 'annual_nav_drawdown', is_confirmed: true },
        max_loss_per_trade: { value: maxLossPerTrade, unit: 'percent', calculation_basis: 'portfolio_nav_at_entry', is_confirmed: true },
        mdd_percent: { value: mddPercent, unit: 'percent', calculation_basis: 'annual_nav_drawdown', is_confirmed: true },
        max_loss_per_trade_percent: { value: maxLossPerTrade, unit: 'percent', calculation_basis: 'portfolio_nav_at_entry', is_confirmed: true },
      },
      investment_horizon: horizon,
      rebalance_frequency: rebalanceFreq,
      non_actions: nonActions,
      invest_targets: investTargets,
      exclude_targets: excludeTargets,
      primary_methods: primaryMethods,
      secondary_methods: secondaryMethods,
      role_models: roleModels,
      allocation_rows: allocationRows,
    })
  }

  const totalAllocation = useMemo(() => {
    return allocationRows.reduce((sum, r) => sum + (parseFloat(r.target_percent) || 0), 0)
  }, [allocationRows])
  const isAllocation100 = Math.abs(totalAllocation - 100) <= 0.05

  const clientIssues = useMemo(() => {
    const issues: string[] = []
    if (!basicPolicy.trim()) issues.push('หัวข้อ 1: นโยบายพื้นฐาน')
    if (!mddPercent.trim()) issues.push('หัวข้อ 2: MDD สูงสุด')
    if (!maxLossPerTrade.trim()) issues.push('หัวข้อ 2: ขีดจำกัดขาดทุนต่อไม้')
    if (investTargets.length === 0) issues.push('หัวข้อ 3: สินทรัพย์ที่ลงทุน')
    if (excludeTargets.length === 0) issues.push('หัวข้อ 3: สินทรัพย์ที่ไม่ลงทุน')
    if (primaryMethods.length === 0) issues.push('หัวข้อ 4: วิธีการลงทุนหลัก')
    if (!horizon.trim()) issues.push('หัวข้อ 5: กรอบเวลา')
    if (allocationRows.length === 0) {
      issues.push('หัวข้อ 6: สัดส่วนจัดสรรสินทรัพย์')
    } else if (!isAllocation100) {
      issues.push(`หัวข้อ 6: สัดส่วนรวมต้องได้ 100% (ปัจจุบัน ${totalAllocation.toFixed(2)}%)`)
    }
    if (roleModels.length === 0) issues.push('หัวข้อ 7: นักลงทุนต้นแบบ')
    if (nonActions.length < 3) {
      issues.push(`หัวข้อ 8: สิ่งที่จะไม่ทำต้องมีอย่างน้อย 3 ข้อ (ปัจจุบัน ${nonActions.length}/3)`)
    }
    return issues
  }, [
    basicPolicy,
    mddPercent,
    maxLossPerTrade,
    investTargets,
    excludeTargets,
    primaryMethods,
    horizon,
    allocationRows,
    isAllocation100,
    totalAllocation,
    roleModels,
    nonActions,
  ])

  const isFormComplete = clientIssues.length === 0
  const canConfirm = !loading && (isConfirmed || (axisDraft?.is_complete ?? false) || isFormComplete)

  const handleConfirm = async (proceedToBuckets: boolean) => {
    try {
      await handleSaveDraft()
      await onConfirmAxis(proceedToBuckets)
    } catch {
      // Errors handled by parent hook
    }
  }

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      {/* Header Banner */}
      <div className="rounded-2xl border border-sky-100 bg-gradient-to-br from-sky-50/70 via-white to-blue-50/30 p-6 sm:p-8 shadow-sm">
        <div className="flex flex-wrap items-center justify-between gap-3 border-b border-sky-100 pb-4">
          <div className="flex items-center gap-2.5">
            <span className="flex h-10 w-10 items-center justify-center rounded-xl bg-gradient-to-tr from-sky-500 to-blue-600 text-white font-bold text-lg shadow-sm">
              🧭
            </span>
            <div>
              <h1 className="text-xl sm:text-2xl font-extrabold text-zinc-900 tracking-tight">
                แกนหลักการลงทุน (Investment Axis)
              </h1>
              <p className="text-xs text-zinc-500 mt-0.5">
                พอร์ต: <span className="font-semibold text-zinc-700">{portfolioId}</span> | แปลงแก่นแท้และบริบทการเงินเป็นนโยบาย 8 หัวข้อที่มีตัวเลขรูปธรรมและสิ่งที่จะไม่ทำ
              </p>
            </div>
          </div>

          <div>
            {isConfirmed ? (
              <span className="inline-flex items-center gap-1.5 rounded-full bg-emerald-50 px-3 py-1 text-xs font-bold text-emerald-700 border border-emerald-200">
                <span className="h-2 w-2 rounded-full bg-emerald-500" />
                ยืนยันแล้ว
              </span>
            ) : (
              <span className="inline-flex items-center gap-1.5 rounded-full bg-amber-50 px-3 py-1 text-xs font-bold text-amber-700 border border-amber-200">
                <span className="h-2 w-2 rounded-full bg-amber-500 animate-pulse" />
                ร่างนโยบายรอตรวจ
              </span>
            )}
          </div>
        </div>

        {/* Completeness Alert */}
        {!isConfirmed && clientIssues.length > 0 && (
          <div className="mt-4 rounded-xl border border-amber-200 bg-amber-50/80 p-3.5 text-xs text-amber-900 space-y-1">
            <span className="font-bold">⚠️ สิ่งที่ต้องระบุให้ครบก่อนยืนยันแกนหลัก:</span>
            <ul className="list-disc list-inside">
              {clientIssues.map((issue, idx) => (
                <li key={idx}>{issue}</li>
              ))}
            </ul>
          </div>
        )}
      </div>

      {/* 8 Sections Editor */}
      <div className="rounded-2xl border border-sky-100 bg-white p-6 sm:p-8 shadow-sm space-y-6">
        {/* Section 1: Basic Policy */}
        <div className="p-4 rounded-xl bg-sky-50/30 border border-sky-100 space-y-2">
          <label htmlFor="basic-policy-input" className="block text-sm font-bold text-zinc-900">
            📌 1. นโยบายพื้นฐาน (Basic Policy)
          </label>
          <p className="text-xs text-zinc-500">
            แนวทางหลักและสิ่งที่ให้ความสำคัญในการลงทุน เชื่อมโยงกับแก่นแท้ของคุณ
          </p>
          <textarea
            id="basic-policy-input"
            rows={2}
            disabled={isConfirmed}
            value={basicPolicy}
            onChange={(e) => setBasicPolicy(e.target.value)}
            className="w-full rounded-lg border border-zinc-200 bg-white p-3 text-sm text-zinc-800 focus:outline-none focus:ring-2 focus:ring-sky-400 disabled:bg-zinc-50"
          />
        </div>

        {/* Section 2: Risk Limits */}
        <div className="p-4 rounded-xl bg-sky-50/30 border border-sky-100 space-y-3">
          <h3 className="text-sm font-bold text-zinc-900">
            🛡️ 2. ระดับความเสี่ยงที่ยอมรับได้ (Risk Limits)
          </h3>
          <p className="text-xs text-zinc-500">
            MDD = การลดลงมากที่สุดจากจุดสูงสุดถึงจุดต่ำสุดในรอบปี | ขีดจำกัดต่อรายการ = การตัดขาดทุนสูงสุดต่อ 1 การซื้อขาย
          </p>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 pt-1">
            <div>
              <label htmlFor="mdd-input" className="block text-xs font-semibold text-zinc-700 mb-1">
                MDD สูงสุดต่อปี (%):
              </label>
              <div className="flex items-center gap-2">
                <input
                  id="mdd-input"
                  type="text"
                  disabled={isConfirmed}
                  value={mddPercent}
                  onChange={(e) => setMddPercent(e.target.value)}
                  className="w-full rounded-lg border border-zinc-200 bg-white p-2.5 text-xs text-zinc-800 focus:outline-none focus:ring-2 focus:ring-sky-400 disabled:bg-zinc-50 font-bold"
                />
                <span className="text-xs font-bold text-zinc-500">%</span>
              </div>
            </div>

            <div>
              <label htmlFor="max-loss-input" className="block text-xs font-semibold text-zinc-700 mb-1">
                ขีดจำกัดผลขาดทุนต่อ 1 การซื้อขาย (% ของพอร์ต):
              </label>
              <div className="flex items-center gap-2">
                <input
                  id="max-loss-input"
                  type="text"
                  disabled={isConfirmed}
                  value={maxLossPerTrade}
                  onChange={(e) => setMaxLossPerTrade(e.target.value)}
                  className="w-full rounded-lg border border-zinc-200 bg-white p-2.5 text-xs text-zinc-800 focus:outline-none focus:ring-2 focus:ring-sky-400 disabled:bg-zinc-50 font-bold"
                />
                <span className="text-xs font-bold text-zinc-500">%</span>
              </div>
            </div>
          </div>
        </div>

        {/* Section 3: Targets (Invest / Exclude) */}
        <div className="p-4 rounded-xl bg-sky-50/30 border border-sky-100 space-y-3">
          <h3 className="text-sm font-bold text-zinc-900">
            🎯 3. เป้าหมาย: ลงทุน / ไม่ลงทุน (Invest & Exclude Targets)
          </h3>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div>
              <span className="block text-xs font-semibold text-emerald-800 mb-1">
                สินทรัพย์หรือประเภทที่เลือกลงทุน:
              </span>
              <ul className="text-xs text-zinc-700 space-y-1 list-disc list-inside">
                {investTargets.map((t, idx) => (
                  <li key={idx}>{t}</li>
                ))}
              </ul>
            </div>
            <div>
              <span className="block text-xs font-semibold text-rose-800 mb-1">
                สินทรัพย์หรือหมวดที่ยกเว้น (ไม่ลงทุน):
              </span>
              <ul className="text-xs text-zinc-700 space-y-1 list-disc list-inside">
                {excludeTargets.map((t, idx) => (
                  <li key={idx}>{t}</li>
                ))}
              </ul>
            </div>
          </div>
        </div>

        {/* Section 4 & 5: Methods & Horizon */}
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <div className="p-4 rounded-xl bg-sky-50/30 border border-sky-100 space-y-2">
            <h3 className="text-sm font-bold text-zinc-900">
              ⚙️ 4. วิธีการลงทุน (Methods)
            </h3>
            <span className="block text-xs font-semibold text-zinc-600">วิธีการหลัก:</span>
            <ul className="text-xs text-zinc-700 space-y-1 list-disc list-inside">
              {primaryMethods.map((m, idx) => (
                <li key={idx}>{m}</li>
              ))}
            </ul>
            {secondaryMethods.length > 0 && (
              <>
                <span className="block text-xs font-semibold text-zinc-600 pt-1">วิธีการเสริม:</span>
                <ul className="text-xs text-zinc-700 space-y-1 list-disc list-inside">
                  {secondaryMethods.map((m, idx) => (
                    <li key={idx}>{m}</li>
                  ))}
                </ul>
              </>
            )}
          </div>

          <div className="p-4 rounded-xl bg-sky-50/30 border border-sky-100 space-y-2">
            <label htmlFor="horizon-input" className="block text-sm font-bold text-zinc-900">
              ⏳ 5. กรอบเวลาการลงทุน (Horizon)
            </label>
            <input
              id="horizon-input"
              type="text"
              disabled={isConfirmed}
              value={horizon}
              onChange={(e) => setHorizon(e.target.value)}
              placeholder="เช่น 5-10 ปี หรือ ระยะยาวต่อเนื่อง"
              className="w-full rounded-lg border border-zinc-200 bg-white p-2.5 text-xs text-zinc-800 focus:outline-none focus:ring-2 focus:ring-sky-400 disabled:bg-zinc-50"
            />
          </div>
        </div>

        {/* Section 6: Asset Allocation & Rebalance */}
        <div className="p-4 rounded-xl bg-sky-50/30 border border-sky-100 space-y-3">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-bold text-zinc-900">
              🥧 6. การจัดสรรสินทรัพย์และรอบ Rebalance
            </h3>
            <div className="flex items-center gap-1.5 text-xs">
              <label htmlFor="rebalance-input" className="text-zinc-500 font-medium">รอบปรับสมดุล:</label>
              <input
                id="rebalance-input"
                type="text"
                disabled={isConfirmed}
                value={rebalanceFreq}
                onChange={(e) => setRebalanceFreq(e.target.value)}
                className="w-24 rounded border border-zinc-200 px-2 py-0.5 text-xs font-bold text-zinc-800"
              />
            </div>
          </div>
          <p className="text-xs text-zinc-500">
            สัดส่วนเป้าหมายรวม 100% ซึ่งจะใช้เป็นฐานสร้าง Purpose Buckets ในขั้นตอนถัดไป
          </p>
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-2 pt-1">
            {allocationRows.map((row, idx) => (
              <div key={idx} className="p-2.5 rounded-lg bg-white border border-sky-200/70 text-xs">
                <span className="font-semibold text-zinc-800 block truncate">{row.category}</span>
                <span className="text-sm font-extrabold text-sky-700">{row.target_percent}%</span>
              </div>
            ))}
          </div>
        </div>

        {/* Section 7: Role Models */}
        <div className="p-4 rounded-xl bg-sky-50/30 border border-sky-100 space-y-2">
          <h3 className="text-sm font-bold text-zinc-900">
            👤 7. นักลงทุนที่ใช้เป็นต้นแบบ (Role Models)
          </h3>
          <ul className="text-xs text-zinc-700 space-y-1 list-disc list-inside">
            {roleModels.map((rm, idx) => (
              <li key={idx}>{rm}</li>
            ))}
          </ul>
        </div>

        {/* Section 8: Non-actions (>= 3) */}
        <div className="p-4 rounded-xl bg-rose-50/30 border border-rose-100 space-y-3">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-bold text-rose-900">
              🚫 8. สิ่งที่จะไม่ทำเด็ดขาด (Non-Actions อย่างน้อย 3 ข้อ)
            </h3>
            <span
              className={`text-xs font-bold px-2 py-0.5 rounded-full ${
                nonActions.length >= 3 ? 'bg-emerald-100 text-emerald-800' : 'bg-rose-100 text-rose-800'
              }`}
            >
              {nonActions.length}/3 ข้อ
            </span>
          </div>
          <p className="text-xs text-zinc-500">
            วินัยและขอบเขตการตัดสินใจที่ช่วยป้องกันความผิดพลาดร้ายแรงในชีวิตจริง
          </p>

          <div className="space-y-2">
            {nonActions.map((na, idx) => (
              <div
                key={idx}
                className="flex items-center justify-between gap-2 p-2.5 rounded-lg bg-white border border-rose-200 text-xs text-zinc-800"
              >
                <span>• {na}</span>
                {!isConfirmed && (
                  <button
                    type="button"
                    onClick={() => handleRemoveNonAction(idx)}
                    className="text-rose-500 hover:text-rose-700 font-bold px-1.5"
                  >
                    ✕
                  </button>
                )}
              </div>
            ))}
          </div>

          {!isConfirmed && (
            <div className="flex items-center gap-2 pt-1">
              <input
                type="text"
                value={newNonAction}
                onChange={(e) => setNewNonAction(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') {
                    e.preventDefault()
                    handleAddNonAction()
                  }
                }}
                placeholder="พิมพ์สิ่งที่จะไม่ทำ เช่น ไม่กู้เงินมาเทรด, ไม่ซื้อตามข่าวลือ..."
                className="flex-1 rounded-lg border border-zinc-200 bg-white p-2 text-xs text-zinc-800 focus:outline-none focus:ring-2 focus:ring-rose-400"
              />
              <button
                type="button"
                onClick={handleAddNonAction}
                className="rounded-lg bg-rose-600 px-3.5 py-2 text-xs font-bold text-white hover:bg-rose-700"
              >
                + เพิ่ม
              </button>
            </div>
          )}
        </div>

        {/* Action Bar */}
        <div className="pt-6 border-t border-sky-100 flex flex-col sm:flex-row items-center justify-between gap-4">
          <div className="text-xs text-zinc-500">
            {statusText ? (
              <span className="flex items-center gap-2 text-sky-700 font-semibold animate-pulse">
                <span className="h-2 w-2 rounded-full bg-sky-500" />
                {statusText}
              </span>
            ) : isConfirmed ? (
              <span className="text-emerald-700 font-semibold">
                ✓ ยืนยันแกนหลักเรียบร้อยแล้ว แผนพอร์ตยังไม่เปลี่ยนจนกว่าจะกดสร้างและใช้ Buckets
              </span>
            ) : clientIssues.length > 0 ? (
              <span className="text-amber-700 font-semibold">
                ⚠️ ยังไม่ครบ: {clientIssues.join(', ')}
              </span>
            ) : (
              <span className="text-emerald-700 font-semibold">
                ✓ ตรวจสอบครบ 8 หัวข้อแล้ว พร้อมยืนยันแกนหลักการลงทุน
              </span>
            )}
          </div>

          <div className="flex items-center gap-3 w-full sm:w-auto">
            {!isConfirmed && (
              <>
                <button
                  type="button"
                  onClick={handleSaveDraft}
                  disabled={loading}
                  className="rounded-xl border border-sky-300 bg-sky-50 px-4 py-2.5 text-xs sm:text-sm font-bold text-sky-700 hover:bg-sky-100 active:scale-98 transition-all"
                >
                  💾 บันทึกร่างแกนหลัก
                </button>

                <button
                  type="button"
                  onClick={() => handleConfirm(false)}
                  disabled={!canConfirm}
                  className="rounded-xl border border-emerald-300 bg-emerald-50 px-5 py-2.5 text-xs sm:text-sm font-bold text-emerald-800 hover:bg-emerald-100 active:scale-98 transition-all disabled:opacity-50"
                >
                  ✓ ยืนยันแกนหลักของพอร์ตนี้
                </button>
              </>
            )}

            <button
              type="button"
              onClick={() => {
                if (isConfirmed) {
                  onProceedToBuckets()
                } else {
                  handleConfirm(true)
                }
              }}
              disabled={loading || (!isConfirmed && !canConfirm)}
              className="rounded-xl bg-gradient-to-r from-sky-600 to-blue-600 px-6 py-2.5 text-xs sm:text-sm font-bold text-white shadow-md shadow-sky-500/20 hover:from-sky-700 hover:to-blue-700 active:scale-98 transition-all flex items-center gap-1.5 disabled:opacity-50"
            >
              <span>สร้าง Buckets จากแกนหลัก →</span>
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}
