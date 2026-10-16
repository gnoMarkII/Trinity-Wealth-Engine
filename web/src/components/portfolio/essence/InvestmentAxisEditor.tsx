import { useState, useEffect, useMemo } from 'react'
import AutoResizeTextarea from '../../ui/AutoResizeTextarea'
import type {
  AxisDraftDTO,
  ConfirmedAxisDTO,
  UpdateAxisDraftRequestDTO,
  AllocationRowDTO,
  ActualGoalItemDTO,
} from '../../../api/types'

interface InvestmentAxisEditorProps {
  axisDraft: AxisDraftDTO | null
  confirmedAxis: ConfirmedAxisDTO | null
  portfolioId: string
  portfolioName?: string
  portfolioGoals?: ActualGoalItemDTO[]
  onUpdateDraft: (updates: UpdateAxisDraftRequestDTO) => Promise<any>
  onConfirmAxis: (proceedToBuckets: boolean, targetDraft?: AxisDraftDTO) => Promise<any>
  onProceedToBuckets: () => void
  onRegenerateDraft?: () => Promise<any>
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

const ALLOCATION_PALETTE = [
  'bg-blue-500 text-white',
  'bg-emerald-500 text-white',
  'bg-violet-500 text-white',
  'bg-amber-500 text-white',
  'bg-rose-500 text-white',
  'bg-cyan-500 text-white',
  'bg-teal-500 text-white',
  'bg-indigo-500 text-white',
]

const ALLOCATION_BAR_COLORS = [
  'bg-blue-500',
  'bg-emerald-500',
  'bg-violet-500',
  'bg-amber-500',
  'bg-rose-500',
  'bg-cyan-500',
  'bg-teal-500',
  'bg-indigo-500',
]

export default function InvestmentAxisEditor({
  axisDraft,
  confirmedAxis,
  portfolioId,
  portfolioName,
  portfolioGoals,
  onUpdateDraft,
  onConfirmAxis,
  onProceedToBuckets,
  onRegenerateDraft,
  loading,
  statusText,
}: InvestmentAxisEditorProps) {
  const displayName = portfolioName || (portfolioId === 'default' ? 'พอร์ตลงทุนหลัก' : portfolioId)
  const isConfirmed = !!confirmedAxis
  const [isEditing, setIsEditing] = useState(false)
  const isReadOnly = isConfirmed && !isEditing

  const source = isReadOnly ? confirmedAxis : (axisDraft || confirmedAxis)

  // Local editable state
  const [basicPolicy, setBasicPolicy] = useState(source?.basic_policy || '')
  const [mddPercent, setMddPercent] = useState(
    parseRiskLimitValue(
      source?.risk_limits?.mdd_max_annual || source?.risk_limits?.mdd_percent,
      '15',
    ),
  )
  const [maxLossPerTrade, setMaxLossPerTrade] = useState(
    parseRiskLimitValue(
      source?.risk_limits?.max_loss_per_trade || source?.risk_limits?.max_loss_per_trade_percent,
      '2',
    ),
  )
  const [horizon, setHorizon] = useState(source?.investment_horizon || '')
  const [rebalanceFreq, setRebalanceFreq] = useState(source?.rebalance_frequency || 'ทุก 6 เดือน หรือเบี่ยงเบนเกิน 5%')
  const [nonActions, setNonActions] = useState<string[]>(source?.non_actions || [])
  const [investTargets, setInvestTargets] = useState<string[]>(source?.invest_targets || [])
  const [excludeTargets, setExcludeTargets] = useState<string[]>(source?.exclude_targets || [])
  const [primaryMethods, setPrimaryMethods] = useState<string[]>(source?.primary_methods || [])
  const [secondaryMethods, setSecondaryMethods] = useState<string[]>(source?.secondary_methods || [])
  const [roleModels, setRoleModels] = useState<string[]>(source?.role_models || [])
  const [allocationRows, setAllocationRows] = useState<AllocationRowDTO[]>(source?.allocation_rows || [])

  // Input states for adding new items
  const [newNonAction, setNewNonAction] = useState('')
  const [editingNonActionIdx, setEditingNonActionIdx] = useState<number | null>(null)
  const [editingNonActionText, setEditingNonActionText] = useState('')
  const [newInvestTarget, setNewInvestTarget] = useState('')
  const [newExcludeTarget, setNewExcludeTarget] = useState('')
  const [newPrimaryMethod, setNewPrimaryMethod] = useState('')
  const [newSecondaryMethod, setNewSecondaryMethod] = useState('')
  const [newRoleModel, setNewRoleModel] = useState('')

  useEffect(() => {
    const s = isReadOnly ? confirmedAxis : (axisDraft || confirmedAxis)
    if (s) {
      setBasicPolicy(s.basic_policy || '')
      setMddPercent(
        parseRiskLimitValue(
          s.risk_limits?.mdd_max_annual || s.risk_limits?.mdd_percent,
          '15',
        ),
      )
      setMaxLossPerTrade(
        parseRiskLimitValue(
          s.risk_limits?.max_loss_per_trade || s.risk_limits?.max_loss_per_trade_percent,
          '2',
        ),
      )
      setHorizon(s.investment_horizon || '')
      setRebalanceFreq(s.rebalance_frequency || 'ทุก 6 เดือน หรือเบี่ยงเบนเกิน 5%')
      setNonActions(s.non_actions || [])
      setInvestTargets(s.invest_targets || [])
      setExcludeTargets(s.exclude_targets || [])
      setPrimaryMethods(s.primary_methods || [])
      setSecondaryMethods(s.secondary_methods || [])
      setRoleModels(s.role_models || [])
      setAllocationRows(s.allocation_rows || [])
    }
  }, [axisDraft, confirmedAxis, isReadOnly])

  // Handlers for list management
  const handleAddInvestTarget = () => {
    if (newInvestTarget.trim() && !investTargets.includes(newInvestTarget.trim())) {
      setInvestTargets([...investTargets, newInvestTarget.trim()])
      setNewInvestTarget('')
    }
  }

  const handleAddExcludeTarget = () => {
    if (newExcludeTarget.trim() && !excludeTargets.includes(newExcludeTarget.trim())) {
      setExcludeTargets([...excludeTargets, newExcludeTarget.trim()])
      setNewExcludeTarget('')
    }
  }

  const handleAddPrimaryMethod = () => {
    if (newPrimaryMethod.trim() && !primaryMethods.includes(newPrimaryMethod.trim())) {
      setPrimaryMethods([...primaryMethods, newPrimaryMethod.trim()])
      setNewPrimaryMethod('')
    }
  }

  const handleAddSecondaryMethod = () => {
    if (newSecondaryMethod.trim() && !secondaryMethods.includes(newSecondaryMethod.trim())) {
      setSecondaryMethods([...secondaryMethods, newSecondaryMethod.trim()])
      setNewSecondaryMethod('')
    }
  }

  const handleAddRoleModel = () => {
    if (newRoleModel.trim() && !roleModels.includes(newRoleModel.trim())) {
      setRoleModels([...roleModels, newRoleModel.trim()])
      setNewRoleModel('')
    }
  }

  const handleAddNonAction = () => {
    if (newNonAction.trim() && !nonActions.includes(newNonAction.trim())) {
      setNonActions([...nonActions, newNonAction.trim()])
      setNewNonAction('')
    }
  }

  const handleRemoveNonAction = (index: number) => {
    setNonActions(nonActions.filter((_, i) => i !== index))
    if (editingNonActionIdx === index) {
      setEditingNonActionIdx(null)
      setEditingNonActionText('')
    }
  }

  const handleStartEditNonAction = (index: number, text: string) => {
    setEditingNonActionIdx(index)
    setEditingNonActionText(text)
  }

  const handleSaveEditNonAction = (index: number) => {
    if (editingNonActionText.trim()) {
      const updated = [...nonActions]
      updated[index] = editingNonActionText.trim()
      setNonActions(updated)
    }
    setEditingNonActionIdx(null)
    setEditingNonActionText('')
  }

  const handleCancelEditNonAction = () => {
    setEditingNonActionIdx(null)
    setEditingNonActionText('')
  }

  const handleUpdateAllocationRow = (index: number, field: string, value: string) => {
    const updated = [...allocationRows]
    const current = updated[index]
    if (!current) return

    const updatedRow: AllocationRowDTO = {
      ...current,
      [field]: value,
      category: field === 'category' ? value : current.category,
      target_percent: field === 'target_percent' ? value : current.target_percent,
    }
    if (field === 'category') {
      updatedRow.category_name = value
    }
    updated[index] = updatedRow
    setAllocationRows(updated)
  }

  const handleAddAllocationRow = () => {
    const newId = `alloc_${Date.now()}`
    setAllocationRows([
      ...allocationRows,
      {
        allocation_id: newId,
        category: 'สินทรัพย์ใหม่',
        category_name: 'สินทรัพย์ใหม่',
        target_percent: '0',
        role_description: 'ระบุบทบาทที่รองรับเป้าหมาย',
      },
    ])
  }

  const handleRemoveAllocationRow = (index: number) => {
    setAllocationRows(allocationRows.filter((_, i) => i !== index))
  }

  const handleSaveDraft = async () => {
    return await onUpdateDraft({
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
    if (secondaryMethods.length === 0) issues.push('หัวข้อ 4: วิธีการลงทุนเสริม')
    if (!horizon.trim()) issues.push('หัวข้อ 5: กรอบเวลา')
    if (allocationRows.length === 0) {
      issues.push('หัวข้อ 6: สัดส่วนจัดสรรสินทรัพย์')
    } else if (!isAllocation100) {
      issues.push(`หัวข้อ 6: สัดส่วนรวมต้องได้ 100% (ปัจจุบัน ${totalAllocation.toFixed(1)}%)`)
    }
    if (!rebalanceFreq.trim()) issues.push('หัวข้อ 6: รอบความถี่ Rebalance')
    if (roleModels.length === 0) issues.push('หัวข้อ 7: นักลงทุนต้นแบบ')
    const uniqueNonActions = new Set(nonActions.map((n) => n.trim()).filter(Boolean))
    if (uniqueNonActions.size < 3) {
      if (nonActions.length >= 3 && uniqueNonActions.size < nonActions.length) {
        issues.push(`หัวข้อ 8: สิ่งที่จะไม่ทำมีข้อความซ้ำกัน (ต้องไม่ซ้ำกันอย่างน้อย 3 ข้อ)`)
      } else {
        issues.push(
          `หัวข้อ 8: สิ่งที่จะไม่ทำต้องมีอย่างน้อย 3 ข้อ (ปัจจุบันมี ${uniqueNonActions.size} ข้อ ขาดอีก ${3 - uniqueNonActions.size} ข้อ)`
        )
      }
    }
    return issues
  }, [
    basicPolicy,
    mddPercent,
    maxLossPerTrade,
    investTargets,
    excludeTargets,
    primaryMethods,
    secondaryMethods,
    horizon,
    allocationRows,
    isAllocation100,
    totalAllocation,
    rebalanceFreq,
    roleModels,
    nonActions,
  ])

  const isFormComplete = clientIssues.length === 0
  const canConfirm = !loading && (isConfirmed || (axisDraft?.is_complete ?? false) || isFormComplete)

  const handleConfirm = async (proceedToBuckets: boolean) => {
    try {
      const updated = await handleSaveDraft()
      await onConfirmAxis(proceedToBuckets, updated)
      setIsEditing(false)
    } catch {
      // Errors handled by parent hook
    }
  }

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      {/* Header Banner */}
      <div className="rounded-2xl border border-sky-100 bg-gradient-to-br from-sky-50/80 via-white to-indigo-50/30 p-6 sm:p-7 shadow-sm">
        <div className="flex flex-wrap items-center justify-between gap-4 border-b border-sky-100/80 pb-5">
          <div className="flex items-center gap-3">
            <span className="flex h-12 w-12 items-center justify-center rounded-2xl bg-gradient-to-tr from-sky-500 to-indigo-600 text-white font-bold text-2xl shadow-md shadow-sky-500/20">
              🧭
            </span>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="text-xl sm:text-2xl font-black text-zinc-900 tracking-tight">
                  แกนหลักการลงทุน (Investment Axis)
                </h1>
                {isConfirmed && !isEditing ? (
                  <span className="inline-flex items-center gap-1.5 rounded-full bg-emerald-100 px-3 py-0.5 text-xs font-bold text-emerald-800 border border-emerald-300">
                    <span className="h-2 w-2 rounded-full bg-emerald-500 animate-pulse" />
                    ใช้งานจริง
                  </span>
                ) : (
                  <span className="inline-flex items-center gap-1 rounded-full bg-amber-100 px-2.5 py-0.5 text-xs font-bold text-amber-800 border border-amber-300">
                    ✏️ อยู่ในโหมดแก้ไข/ร่าง
                  </span>
                )}
              </div>
              <p className="text-xs text-zinc-600 mt-1">
                พอร์ต: <span className="font-bold text-zinc-800">{displayName}</span> | นโยบาย 8 หัวข้อที่แปลงมาจากแก่นแท้และเชื่อมโยงกับเป้าหมายทางการเงินของคุณ
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            {isConfirmed && !isEditing ? (
              <button
                type="button"
                onClick={() => setIsEditing(true)}
                className="inline-flex items-center gap-1.5 rounded-xl border border-sky-300 bg-white px-4 py-2 text-xs font-bold text-sky-700 shadow-xs hover:bg-sky-50 active:scale-98 transition-all"
              >
                ✏️ แก้ไขแกนหลัก
              </button>
            ) : isEditing ? (
              <button
                type="button"
                onClick={() => {
                  setIsEditing(false)
                  if (confirmedAxis) {
                    setBasicPolicy(confirmedAxis.basic_policy || '')
                    setMddPercent(parseRiskLimitValue(confirmedAxis.risk_limits?.mdd_max_annual, '15'))
                    setMaxLossPerTrade(parseRiskLimitValue(confirmedAxis.risk_limits?.max_loss_per_trade, '2'))
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
                }}
                className="rounded-xl border border-zinc-300 bg-white px-3.5 py-2 text-xs font-bold text-zinc-700 hover:bg-zinc-50"
              >
                ยกเลิกการแก้ไข
              </button>
            ) : null}

            {onRegenerateDraft && (!isConfirmed || isEditing) && (
              <button
                type="button"
                onClick={onRegenerateDraft}
                disabled={loading}
                className="inline-flex items-center gap-1.5 rounded-xl border border-indigo-200 bg-indigo-50 px-3.5 py-2 text-xs font-bold text-indigo-700 hover:bg-indigo-100 active:scale-98 transition-all disabled:opacity-50"
                title="สั่งให้ AI วิเคราะห์เป้าหมายล่าสุดและแก่นแท้เพื่อร่างแกนหลักชุดใหม่"
              >
                🔄 ร่างใหม่โดยอิงเป้าหมาย
              </button>
            )}
          </div>
        </div>

        {/* Completeness Checklist Alert */}
        {(!isConfirmed || isEditing) && clientIssues.length > 0 && (
          <div className="mt-4 rounded-xl border border-amber-200 bg-amber-50/90 p-4 text-xs text-amber-900 shadow-2xs space-y-1.5">
            <div className="flex items-center gap-1.5 font-bold text-amber-950">
              <span>⚠️</span>
              <span>สิ่งที่ต้องระบุให้ครบถ้วนก่อนยืนยันแกนหลัก ({clientIssues.length} รายการ):</span>
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-x-4 gap-y-1 pl-5 list-disc">
              {clientIssues.map((issue, idx) => (
                <div key={idx} className="text-amber-800">
                  • {issue}
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Linked Portfolio Goals Banner */}
      {portfolioGoals && portfolioGoals.length > 0 ? (
        <div className="rounded-2xl border border-indigo-100 bg-gradient-to-r from-blue-50/70 via-indigo-50/40 to-sky-50/60 p-5 shadow-xs space-y-3">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div className="flex items-center gap-2.5">
              <span className="flex h-8 w-8 items-center justify-center rounded-xl bg-indigo-600 text-white text-sm font-bold shadow-xs">
                🎯
              </span>
              <div>
                <h3 className="text-sm font-extrabold text-indigo-950">
                  เป้าหมายทางการเงินที่เชื่อมโยง ({portfolioGoals.length} เป้าหมาย)
                </h3>
                <p className="text-xs text-indigo-800/80">
                  นโยบาย 8 หัวข้อนี้ถูกคำนวณและออกแบบเพื่อขับเคลื่อนและปกป้องเป้าหมายเหล่านี้โดยตรง
                </p>
              </div>
            </div>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-3">
            {portfolioGoals.map((g, idx) => {
              const goalTypeLabels: Record<string, string> = {
                nav_target: 'มูลค่าพอร์ต',
                cash_target: 'เงินสด / สำรอง',
                passive_income_ytd: 'กระแสเงินสด',
                bucket_target: 'Bucket',
              }
              const typeLabel = goalTypeLabels[g.goal_type] || g.goal_type || 'เป้าหมาย'

              return (
                <div
                  key={g.name || idx}
                  className="rounded-xl border border-indigo-100/80 bg-white/95 p-4 shadow-xs space-y-2 hover:border-indigo-300 transition-colors"
                >
                  <div className="flex items-start justify-between gap-2">
                    <span className="font-bold text-xs text-zinc-900 line-clamp-1" title={g.name}>
                      {g.name}
                    </span>
                    <span className="rounded-md bg-indigo-50 px-2 py-0.5 text-[10px] font-bold text-indigo-700 shrink-0 border border-indigo-100">
                      {typeLabel}
                    </span>
                  </div>

                  <div className="flex items-baseline justify-between text-xs">
                    <span className="text-zinc-500 text-[11px]">เป้าหมาย:</span>
                    <span className="font-extrabold text-indigo-700 text-sm">
                      {Number(g.target_amount_thb || 0).toLocaleString('th-TH')} บาท
                    </span>
                  </div>

                  <div className="flex items-center justify-between text-[11px] text-zinc-500">
                    <span>กรอบเวลา:</span>
                    <span className="font-semibold text-zinc-700">
                      {g.deadline ? `${g.deadline} ${g.deadline_days_left ? `(เหลือ ${Math.round(g.deadline_days_left / 365)} ปี)` : ''}` : 'ไม่ระบุวัน'}
                    </span>
                  </div>

                  {typeof g.progress_pct === 'number' && (
                    <div className="space-y-1 pt-1 border-t border-zinc-100">
                      <div className="flex justify-between text-[10px]">
                        <span className="text-zinc-500">ความคืบหน้า</span>
                        <span className="font-bold text-zinc-800">{g.progress_pct.toFixed(1)}%</span>
                      </div>
                      <div className="h-2 w-full rounded-full bg-zinc-100 overflow-hidden">
                        <div
                          className="h-full rounded-full bg-gradient-to-r from-indigo-500 to-sky-500 transition-all"
                          style={{ width: `${Math.min(100, Math.max(0, g.progress_pct))}%` }}
                        />
                      </div>
                    </div>
                  )}
                </div>
              )
            })}
          </div>
        </div>
      ) : (
        <div className="rounded-xl border border-amber-200 bg-amber-50/60 p-4 text-xs text-amber-900 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span>ℹ️</span>
            <span>ยังไม่มีเป้าหมายพอร์ตที่ระบุไว้ในระบบ (สามารถตั้งค่าในหน้า Portfolio Goals เพื่อให้นโยบายผูกกับเป้าหมายจริงได้)</span>
          </div>
        </div>
      )}

      {/* 8 Sections Main Container */}
      <div className="space-y-6">

        {/* SECTION 1: Basic Policy */}
        <div className="rounded-2xl border border-sky-100 bg-white p-6 shadow-sm space-y-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <span className="flex h-7 w-7 items-center justify-center rounded-lg bg-sky-100 text-sky-700 text-sm font-bold">
                1
              </span>
              <h2 className="text-base font-extrabold text-zinc-900">
                นโยบายพื้นฐาน (Basic Policy)
              </h2>
            </div>
            <span className="text-xs text-zinc-400 font-medium hidden sm:inline">
              สูตร: [เป้าหมายตัวเลขจริง] + [กลยุทธ์ขับเคลื่อน] + [กรอบคุมความเสี่ยง]
            </span>
          </div>

          <p className="text-xs text-zinc-500">
            เจตนารมณ์หลักและหลักปรัชญาที่ยึดถือในการจัดพอร์ต เพื่อให้คุณมีทิศทางที่ชัดเจนในทุกสภาวะตลาด
          </p>

          {!isReadOnly ? (
            <AutoResizeTextarea
              id="basic-policy-input"
              minRows={3}
              value={basicPolicy}
              onChange={(e) => setBasicPolicy(e.target.value)}
              placeholder="ระบุนโยบายพื้นฐาน เช่น ลงทุนระยะยาว 10 ปีเพื่อบรรลุเป้าหมายพอร์ต 10 ล้านบาท..."
              className="rounded-xl border border-zinc-300 bg-white p-3.5 text-sm text-zinc-800 leading-relaxed focus:border-sky-500 focus:outline-none focus:ring-2 focus:ring-sky-200 shadow-2xs"
            />
          ) : (
            <div className="rounded-xl border border-sky-100 bg-sky-50/40 p-4 sm:p-5">
              <p className="text-sm sm:text-base font-semibold text-zinc-800 leading-relaxed">
                “{basicPolicy}”
              </p>
            </div>
          )}
        </div>

        {/* SECTION 2: Risk Limits */}
        <div className="rounded-2xl border border-sky-100 bg-white p-6 shadow-sm space-y-4">
          <div className="flex items-center gap-2">
            <span className="flex h-7 w-7 items-center justify-center rounded-lg bg-sky-100 text-sky-700 text-sm font-bold">
              2
            </span>
            <h2 className="text-base font-extrabold text-zinc-900">
              ระดับความเสี่ยงที่ยอมรับได้ (Risk Limits)
            </h2>
          </div>

          <p className="text-xs text-zinc-500">
            กรอบการคุมความเสี่ยงสูงสุด เพื่อให้คุณสามารถถือครองสินทรัพย์ได้อย่างสบายใจในวันที่ตลาดผันผวน
          </p>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div className="rounded-xl border border-sky-100 bg-sky-50/30 p-4 space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-xs font-bold text-zinc-700">MDD สูงสุดต่อปี (Maximum Drawdown)</span>
                <span className="text-[10px] font-semibold text-sky-700 bg-sky-100 px-2 py-0.5 rounded-full">
                  พอร์ตโดยรวม
                </span>
              </div>
              <p className="text-[11px] text-zinc-500 leading-tight">
                ผลขาดทุนสูงสุดจากจุดพีคในรอบปีที่ยอมรับได้ โดยไม่ตื่นตระหนกขายทิ้ง
              </p>
              {!isReadOnly ? (
                <div className="flex items-center gap-2 pt-1">
                  <input
                    id="mdd-input"
                    type="number"
                    step="0.5"
                    value={mddPercent}
                    onChange={(e) => setMddPercent(e.target.value)}
                    className="w-32 rounded-lg border border-zinc-300 bg-white px-3 py-2 text-base font-black text-sky-700 focus:outline-none focus:ring-2 focus:ring-sky-300"
                  />
                  <span className="text-sm font-bold text-zinc-600">%</span>
                </div>
              ) : (
                <div className="text-2xl font-black text-sky-700 pt-1">
                  {mddPercent}%
                </div>
              )}
            </div>

            <div className="rounded-xl border border-sky-100 bg-sky-50/30 p-4 space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-xs font-bold text-zinc-700">ขีดจำกัดผลขาดทุนต่อ 1 รายการ</span>
                <span className="text-[10px] font-semibold text-rose-700 bg-rose-100 px-2 py-0.5 rounded-full">
                  Max Loss / Trade
                </span>
              </div>
              <p className="text-[11px] text-zinc-500 leading-tight">
                ความเสียหายสูงสุดที่ยอมรับได้ต่อการซื้อขาย 1 ไม้ เพื่อป้องกันพอร์ตเสียหายหนัก
              </p>
              {!isReadOnly ? (
                <div className="flex items-center gap-2 pt-1">
                  <input
                    id="max-loss-input"
                    type="number"
                    step="0.1"
                    value={maxLossPerTrade}
                    onChange={(e) => setMaxLossPerTrade(e.target.value)}
                    className="w-32 rounded-lg border border-zinc-300 bg-white px-3 py-2 text-base font-black text-rose-700 focus:outline-none focus:ring-2 focus:ring-rose-300"
                  />
                  <span className="text-sm font-bold text-zinc-600">% ของพอร์ต</span>
                </div>
              ) : (
                <div className="text-2xl font-black text-rose-700 pt-1">
                  {maxLossPerTrade}% <span className="text-xs font-semibold text-zinc-500">ของพอร์ต</span>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* SECTION 3: Invest & Exclude Targets */}
        <div className="rounded-2xl border border-sky-100 bg-white p-6 shadow-sm space-y-4">
          <div className="flex items-center gap-2">
            <span className="flex h-7 w-7 items-center justify-center rounded-lg bg-sky-100 text-sky-700 text-sm font-bold">
              3
            </span>
            <h2 className="text-base font-extrabold text-zinc-900">
              เป้าหมายการลงทุน (Invest & Exclude Targets)
            </h2>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
            {/* Invest Targets */}
            <div className="rounded-xl border border-emerald-200/80 bg-emerald-50/30 p-4 space-y-3">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <span className="text-base">📈</span>
                  <span className="text-sm font-bold text-emerald-900">สินทรัพย์ที่เลือกลงทุน (Invest Targets)</span>
                </div>
                <span className="text-xs font-bold text-emerald-800 bg-emerald-100 px-2 py-0.5 rounded-full">
                  {investTargets.length} รายการ
                </span>
              </div>

              <div className="space-y-2">
                {investTargets.map((t, idx) => (
                  <div
                    key={idx}
                    className="flex items-center justify-between gap-2 rounded-lg bg-white border border-emerald-200/70 p-2.5 text-xs text-zinc-800 shadow-2xs"
                  >
                    <span className="font-medium">• {t}</span>
                    {!isReadOnly && (
                      <button
                        type="button"
                        onClick={() => setInvestTargets(investTargets.filter((_, i) => i !== idx))}
                        className="text-zinc-400 hover:text-rose-600 p-1"
                        title="ลบรายการนี้"
                      >
                        ✕
                      </button>
                    )}
                  </div>
                ))}
              </div>

              {!isReadOnly && (
                <div className="flex gap-2 pt-1">
                  <input
                    type="text"
                    value={newInvestTarget}
                    onChange={(e) => setNewInvestTarget(e.target.value)}
                    onKeyDown={(e) => e.key === 'Enter' && handleAddInvestTarget()}
                    placeholder="เพิ่มสินทรัพย์ที่ลงทุน..."
                    className="w-full rounded-lg border border-zinc-200 bg-white px-3 py-1.5 text-xs text-zinc-800 focus:outline-none focus:ring-2 focus:ring-emerald-300"
                  />
                  <button
                    type="button"
                    onClick={handleAddInvestTarget}
                    className="rounded-lg bg-emerald-600 px-3 py-1.5 text-xs font-bold text-white hover:bg-emerald-700 shrink-0"
                  >
                    + เพิ่ม
                  </button>
                </div>
              )}
            </div>

            {/* Exclude Targets */}
            <div className="rounded-xl border border-rose-200/80 bg-rose-50/30 p-4 space-y-3">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <span className="text-base">🚫</span>
                  <span className="text-sm font-bold text-rose-900">สินทรัพย์ที่ไม่แตะต้อง (Exclude Targets)</span>
                </div>
                <span className="text-xs font-bold text-rose-800 bg-rose-100 px-2 py-0.5 rounded-full">
                  {excludeTargets.length} รายการ
                </span>
              </div>

              <div className="space-y-2">
                {excludeTargets.map((t, idx) => (
                  <div
                    key={idx}
                    className="flex items-center justify-between gap-2 rounded-lg bg-white border border-rose-200/70 p-2.5 text-xs text-zinc-800 shadow-2xs"
                  >
                    <span className="font-medium">• {t}</span>
                    {!isReadOnly && (
                      <button
                        type="button"
                        onClick={() => setExcludeTargets(excludeTargets.filter((_, i) => i !== idx))}
                        className="text-zinc-400 hover:text-rose-600 p-1"
                        title="ลบรายการนี้"
                      >
                        ✕
                      </button>
                    )}
                  </div>
                ))}
              </div>

              {!isReadOnly && (
                <div className="flex gap-2 pt-1">
                  <input
                    type="text"
                    value={newExcludeTarget}
                    onChange={(e) => setNewExcludeTarget(e.target.value)}
                    onKeyDown={(e) => e.key === 'Enter' && handleAddExcludeTarget()}
                    placeholder="เพิ่มสินทรัพย์ที่ยกเว้น..."
                    className="w-full rounded-lg border border-zinc-200 bg-white px-3 py-1.5 text-xs text-zinc-800 focus:outline-none focus:ring-2 focus:ring-rose-300"
                  />
                  <button
                    type="button"
                    onClick={handleAddExcludeTarget}
                    className="rounded-lg bg-rose-600 px-3 py-1.5 text-xs font-bold text-white hover:bg-rose-700 shrink-0"
                  >
                    + เพิ่ม
                  </button>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* SECTION 4 & 5: Methods & Horizon */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
          {/* Section 4: Methods */}
          <div className="rounded-2xl border border-sky-100 bg-white p-6 shadow-sm space-y-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <span className="flex h-7 w-7 items-center justify-center rounded-lg bg-sky-100 text-sky-700 text-sm font-bold">
                  4
                </span>
                <h2 className="text-base font-extrabold text-zinc-900">
                  วิธีการลงทุน (Methods)
                </h2>
              </div>
              <span className="text-xs font-bold text-sky-800 bg-sky-100 px-2 py-0.5 rounded-full">
                {primaryMethods.length + secondaryMethods.length} กลยุทธ์
              </span>
            </div>

            {/* Primary Methods */}
            <div className="space-y-2">
              <span className="text-xs font-bold text-zinc-700 block">🔹 วิธีการหลัก (Primary Methods):</span>
              <div className="space-y-1.5">
                {primaryMethods.map((m, idx) => (
                  <div
                    key={idx}
                    className="flex items-center justify-between gap-2 rounded-lg bg-sky-50/50 border border-sky-100 p-2.5 text-xs text-zinc-800"
                  >
                    <span>• {m}</span>
                    {!isReadOnly && (
                      <button
                        type="button"
                        onClick={() => setPrimaryMethods(primaryMethods.filter((_, i) => i !== idx))}
                        className="text-zinc-400 hover:text-rose-600 p-1"
                      >
                        ✕
                      </button>
                    )}
                  </div>
                ))}
              </div>

              {!isReadOnly && (
                <div className="flex gap-2 pt-1">
                  <input
                    type="text"
                    value={newPrimaryMethod}
                    onChange={(e) => setNewPrimaryMethod(e.target.value)}
                    onKeyDown={(e) => e.key === 'Enter' && handleAddPrimaryMethod()}
                    placeholder="เพิ่มวิธีการหลัก..."
                    className="w-full rounded-lg border border-zinc-200 bg-white px-3 py-1.5 text-xs text-zinc-800 focus:outline-none focus:ring-2 focus:ring-sky-300"
                  />
                  <button
                    type="button"
                    onClick={handleAddPrimaryMethod}
                    className="rounded-lg bg-sky-600 px-3 py-1.5 text-xs font-bold text-white hover:bg-sky-700 shrink-0"
                  >
                    + เพิ่ม
                  </button>
                </div>
              )}
            </div>

            {/* Secondary Methods */}
            <div className="space-y-2 pt-2 border-t border-zinc-100">
              <span className="text-xs font-bold text-zinc-700 block">🔸 วิธีการเสริม (Secondary Methods):</span>
              <div className="space-y-1.5">
                {secondaryMethods.map((m, idx) => (
                  <div
                    key={idx}
                    className="flex items-center justify-between gap-2 rounded-lg bg-zinc-50 border border-zinc-200 p-2.5 text-xs text-zinc-800"
                  >
                    <span>• {m}</span>
                    {!isReadOnly && (
                      <button
                        type="button"
                        onClick={() => setSecondaryMethods(secondaryMethods.filter((_, i) => i !== idx))}
                        className="text-zinc-400 hover:text-rose-600 p-1"
                      >
                        ✕
                      </button>
                    )}
                  </div>
                ))}
              </div>

              {!isReadOnly && (
                <div className="flex gap-2 pt-1">
                  <input
                    type="text"
                    value={newSecondaryMethod}
                    onChange={(e) => setNewSecondaryMethod(e.target.value)}
                    onKeyDown={(e) => e.key === 'Enter' && handleAddSecondaryMethod()}
                    placeholder="เพิ่มวิธีการเสริม..."
                    className="w-full rounded-lg border border-zinc-200 bg-white px-3 py-1.5 text-xs text-zinc-800 focus:outline-none focus:ring-2 focus:ring-sky-300"
                  />
                  <button
                    type="button"
                    onClick={handleAddSecondaryMethod}
                    className="rounded-lg bg-zinc-700 px-3 py-1.5 text-xs font-bold text-white hover:bg-zinc-800 shrink-0"
                  >
                    + เพิ่ม
                  </button>
                </div>
              )}
            </div>
          </div>

          {/* Section 5: Horizon */}
          <div className="rounded-2xl border border-sky-100 bg-white p-6 shadow-sm space-y-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <span className="flex h-7 w-7 items-center justify-center rounded-lg bg-sky-100 text-sky-700 text-sm font-bold">
                  5
                </span>
                <h2 className="text-base font-extrabold text-zinc-900">
                  กรอบเวลาการลงทุน (Horizon)
                </h2>
              </div>
              {horizon ? (
                <span className="text-xs font-bold text-sky-800 bg-sky-100 px-2 py-0.5 rounded-full">
                  ✓ กำหนดแล้ว
                </span>
              ) : null}
            </div>

            <p className="text-xs text-zinc-500">
              ระยะเวลาที่ตั้งใจจะถือครองและสะสมสินทรัพย์เพื่อรองรับเป้าหมายชีวิต
            </p>

            {!isReadOnly ? (
              <div className="space-y-2">
                <AutoResizeTextarea
                  id="horizon-input"
                  minRows={2}
                  value={horizon}
                  onChange={(e) => setHorizon(e.target.value)}
                  placeholder="เช่น ระยะยาว 10 ปี (ถึงปี 2036 ตามเป้าหมายพอร์ต 10 ล้าน)"
                  className="rounded-xl border border-zinc-300 bg-white p-3.5 text-sm text-zinc-800 focus:outline-none focus:ring-2 focus:ring-sky-300 shadow-2xs leading-relaxed"
                />
                <div className="flex flex-wrap gap-1.5 pt-1">
                  {['ระยะยาว 10 ปีขึ้นไป', 'ระยะกลาง-ยาว 5-7 ปี', 'ระยะสั้น-กลาง 1-3 ปี'].map((preset) => (
                    <button
                      key={preset}
                      type="button"
                      onClick={() => setHorizon(preset)}
                      className="rounded-lg bg-zinc-100 px-2.5 py-1 text-[11px] font-semibold text-zinc-700 hover:bg-zinc-200"
                    >
                      {preset}
                    </button>
                  ))}
                </div>
              </div>
            ) : (
              <div className="rounded-xl border border-sky-100 bg-sky-50/40 p-4 flex items-center gap-3">
                <span className="text-2xl">⏳</span>
                <div>
                  <span className="text-xs text-zinc-500 font-semibold block">กรอบเวลาที่กำหนด:</span>
                  <span className="text-sm font-bold text-zinc-900">{horizon || 'ไม่ได้ระบุ'}</span>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* SECTION 6: Asset Allocation & Rebalance */}
        <div className="rounded-2xl border border-sky-100 bg-white p-6 shadow-sm space-y-4">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div className="flex items-center gap-2">
              <span className="flex h-7 w-7 items-center justify-center rounded-lg bg-sky-100 text-sky-700 text-sm font-bold">
                6
              </span>
              <h2 className="text-base font-extrabold text-zinc-900">
                การจัดสรรสินทรัพย์และรอบ Rebalance (Asset Allocation)
              </h2>
            </div>

            <div className="flex items-center gap-2">
              <span className="text-xs text-zinc-500 font-semibold">ผลรวมสัดส่วน:</span>
              <span
                className={`rounded-full px-3 py-0.5 text-xs font-black border ${
                  isAllocation100
                    ? 'bg-emerald-100 text-emerald-800 border-emerald-300'
                    : 'bg-rose-100 text-rose-800 border-rose-300 animate-pulse'
                }`}
              >
                {totalAllocation.toFixed(1)}% / 100%
              </span>
            </div>
          </div>

          <p className="text-xs text-zinc-500">
            สัดส่วนการแบ่งสรรเงินลงทุนตามหมวดสินทรัพย์ และระบุเป้าหมายที่แต่ละหมวดรองรับ เพื่อนำไปแปลงเป็น Purpose Buckets ต่อไป
          </p>

          {/* Visual Stacked Bar Chart */}
          <div className="h-4 w-full rounded-full bg-zinc-100 overflow-hidden flex shadow-inner">
            {allocationRows.map((row, idx) => {
              const pct = Math.max(0, parseFloat(row.target_percent) || 0)
              if (pct <= 0) return null
              const colorClass = ALLOCATION_BAR_COLORS[idx % ALLOCATION_BAR_COLORS.length]
              return (
                <div
                  key={idx}
                  className={`h-full ${colorClass} transition-all`}
                  style={{ width: `${pct}%` }}
                  title={`${row.category || (row as any).category_name}: ${pct}%`}
                />
              )
            })}
          </div>

          {/* Allocation Rows Cards/Editors */}
          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-3 pt-1">
            {allocationRows.map((row, idx) => {
              const badgeClass = ALLOCATION_PALETTE[idx % ALLOCATION_PALETTE.length]
              return (
                <div
                  key={row.allocation_id || idx}
                  className="rounded-xl border border-zinc-200/90 bg-white p-4 shadow-2xs space-y-2 relative group"
                >
                  <div className="flex items-center justify-between gap-2">
                    {!isReadOnly ? (
                      <input
                        type="text"
                        value={row.category || (row as any).category_name || ''}
                        onChange={(e) => handleUpdateAllocationRow(idx, 'category', e.target.value)}
                        placeholder="ชื่อหมวดสินทรัพย์"
                        className="text-xs font-bold text-zinc-800 rounded border border-zinc-200 px-2 py-1 w-full focus:outline-none focus:ring-1 focus:ring-sky-400"
                      />
                    ) : (
                      <span className="font-extrabold text-xs text-zinc-900 truncate">
                        {row.category || (row as any).category_name}
                      </span>
                    )}

                    {!isReadOnly ? (
                      <div className="flex items-center gap-1 shrink-0">
                        <input
                          type="number"
                          step="1"
                          value={row.target_percent}
                          onChange={(e) => handleUpdateAllocationRow(idx, 'target_percent', e.target.value)}
                          className="w-16 rounded border border-zinc-200 px-2 py-1 text-xs font-black text-sky-700 text-right focus:outline-none focus:ring-1 focus:ring-sky-400"
                        />
                        <span className="text-xs font-bold text-zinc-500">%</span>
                      </div>
                    ) : (
                      <span className={`px-2 py-0.5 rounded-md text-xs font-black shrink-0 ${badgeClass}`}>
                        {row.target_percent}%
                      </span>
                    )}
                  </div>

                  {!isReadOnly ? (
                    <AutoResizeTextarea
                      minRows={2}
                      value={row.role_description || ''}
                      onChange={(e) => handleUpdateAllocationRow(idx, 'role_description', e.target.value)}
                      placeholder="บทบาทในพอร์ต เช่น รองรับเป้าหมายพอร์ต 10 ล้าน..."
                      className="rounded border border-zinc-200 p-2 text-[11px] text-zinc-700 focus:outline-none focus:ring-1 focus:ring-sky-400"
                    />
                  ) : (
                    <p className="text-[11px] text-zinc-600 leading-tight">
                      {row.role_description || 'ไม่มีคำอธิบายบทบาท'}
                    </p>
                  )}

                  {!isReadOnly && (
                    <button
                      type="button"
                      onClick={() => handleRemoveAllocationRow(idx)}
                      className="absolute top-2 right-2 text-zinc-300 hover:text-rose-600 text-xs p-1"
                      title="ลบหมวดนี้"
                    >
                      ✕
                    </button>
                  )}
                </div>
              )
            })}
          </div>

          {!isReadOnly && (
            <button
              type="button"
              onClick={handleAddAllocationRow}
              className="rounded-xl border border-dashed border-sky-300 bg-sky-50/50 py-2.5 px-4 text-xs font-bold text-sky-700 w-full hover:bg-sky-100 transition-colors"
            >
              + เพิ่มหมวดสินทรัพย์
            </button>
          )}

          {/* Rebalance Frequency Setting */}
          <div className="pt-3 border-t border-zinc-100 flex flex-wrap items-center justify-between gap-3">
            <div className="flex items-center gap-2 flex-1 min-w-[280px]">
              <span className="text-xs font-bold text-zinc-700 shrink-0">รอบและความถี่ Rebalance:</span>
              {!isReadOnly ? (
                <input
                  type="text"
                  value={rebalanceFreq}
                  onChange={(e) => setRebalanceFreq(e.target.value)}
                  placeholder="เช่น ทุก 6 เดือน หรือเบี่ยงเบนเกิน 5%"
                  className="rounded-lg border border-zinc-300 px-3 py-1.5 text-xs text-zinc-800 font-semibold focus:outline-none focus:ring-1 focus:ring-sky-400 flex-1 min-w-[200px]"
                />
              ) : (
                <span className="rounded-lg bg-sky-50 px-3 py-1 text-xs font-bold text-sky-800 border border-sky-100">
                  {rebalanceFreq}
                </span>
              )}
            </div>

            {!isReadOnly && (
              <div className="flex items-center gap-1 text-[11px]">
                <span className="text-zinc-400">เลือกด่วน:</span>
                {['ทุก 6 เดือน หรือเบี่ยงเบน ±5%', 'รายปี', 'ทุกไตรมาส'].map((preset) => (
                  <button
                    key={preset}
                    type="button"
                    onClick={() => setRebalanceFreq(preset)}
                    className="rounded bg-zinc-100 px-2 py-0.5 font-medium text-zinc-600 hover:bg-zinc-200"
                  >
                    {preset}
                  </button>
                ))}
              </div>
            )}
          </div>
        </div>

        {/* SECTION 7: Role Models */}
        <div className="rounded-2xl border border-sky-100 bg-white p-6 shadow-sm space-y-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <span className="flex h-7 w-7 items-center justify-center rounded-lg bg-sky-100 text-sky-700 text-sm font-bold">
                7
              </span>
              <h2 className="text-base font-extrabold text-zinc-900">
                นักลงทุนต้นแบบและหลักการที่ยึดถือ (Role Models)
              </h2>
            </div>
            <span className="text-xs font-bold text-sky-800 bg-sky-100 px-2.5 py-0.5 rounded-full">
              {roleModels.length} ท่าน/แนวทาง
            </span>
          </div>

          <div className="space-y-2">
            {roleModels.map((rm, idx) => (
              <div
                key={idx}
                className="flex items-center justify-between gap-2 rounded-xl bg-gradient-to-r from-zinc-50 to-sky-50/30 border border-zinc-200 p-3 text-xs text-zinc-800 font-medium"
              >
                <div className="flex items-center gap-2">
                  <span className="text-base">👤</span>
                  <span>{rm}</span>
                </div>
                {!isReadOnly && (
                  <button
                    type="button"
                    onClick={() => setRoleModels(roleModels.filter((_, i) => i !== idx))}
                    className="text-zinc-400 hover:text-rose-600 p-1"
                  >
                    ✕
                  </button>
                )}
              </div>
            ))}
          </div>

          {!isReadOnly && (
            <div className="flex gap-2 pt-1">
              <AutoResizeTextarea
                minRows={1}
                value={newRoleModel}
                onChange={(e) => setNewRoleModel(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault()
                    handleAddRoleModel()
                  }
                }}
                placeholder="เพิ่มนักลงทุนต้นแบบ เช่น John Bogle (การลงทุนต้นทุนต่ำผ่านดัชนีระยะยาว)..."
                className="w-full rounded-lg border border-zinc-200 bg-white px-3 py-2 text-xs text-zinc-800 focus:outline-none focus:ring-2 focus:ring-sky-300"
              />
              <button
                type="button"
                onClick={handleAddRoleModel}
                className="rounded-lg bg-sky-600 px-4 py-2 text-xs font-bold text-white hover:bg-sky-700 shrink-0"
              >
                + เพิ่ม
              </button>
            </div>
          )}
        </div>

        {/* SECTION 8: Non-Actions (>= 3) */}
        <div className="rounded-2xl border border-rose-200 bg-white p-6 shadow-sm space-y-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <span className="flex h-7 w-7 items-center justify-center rounded-lg bg-rose-100 text-rose-700 text-sm font-bold">
                8
              </span>
              <h2 className="text-base font-extrabold text-rose-950">
                สิ่งที่จะไม่ทำเด็ดขาด (Non-Actions อย่างน้อย 3 ข้อ)
              </h2>
            </div>
            <span
              className={`text-xs font-bold px-3 py-1 rounded-full ${
                nonActions.length >= 3 ? 'bg-emerald-100 text-emerald-800' : 'bg-rose-100 text-rose-800 animate-pulse'
              }`}
            >
              {nonActions.length >= 3
                ? `✓ ${nonActions.length} ข้อ (ครบตามเกณฑ์)`
                : `${nonActions.length}/3 ข้อ (ขาดอีก ${3 - nonActions.length} ข้อ)`}
            </span>
          </div>

          <p className="text-xs text-zinc-500">
            กฎเหล็กเชิงพฤติกรรมเพื่อป้องกันความผิดพลาดและปกป้องเงินทุน ไม่ให้ทำลายเป้าหมายทางการเงินของคุณ
          </p>

          <div className="space-y-2.5">
            {nonActions.map((na, idx) => (
              <div
                key={idx}
                className="flex items-start justify-between gap-3 p-3.5 rounded-xl bg-rose-50/50 border border-rose-200 text-xs sm:text-sm text-rose-950 font-medium shadow-2xs"
              >
                {editingNonActionIdx === idx ? (
                  <div className="flex-1 space-y-2">
                    <AutoResizeTextarea
                      minRows={2}
                      value={editingNonActionText}
                      onChange={(e) => setEditingNonActionText(e.target.value)}
                      className="rounded-lg border border-rose-300 bg-white p-2.5 text-xs sm:text-sm text-zinc-900 focus:outline-none focus:ring-2 focus:ring-rose-400"
                    />
                    <div className="flex items-center gap-2">
                      <button
                        type="button"
                        onClick={() => handleSaveEditNonAction(idx)}
                        className="rounded-lg bg-rose-600 px-3 py-1 text-xs font-bold text-white hover:bg-rose-700"
                      >
                        ✓ บันทึก
                      </button>
                      <button
                        type="button"
                        onClick={handleCancelEditNonAction}
                        className="rounded-lg border border-zinc-200 px-3 py-1 text-xs font-semibold text-zinc-600 hover:bg-zinc-50"
                      >
                        ยกเลิก
                      </button>
                    </div>
                  </div>
                ) : (
                  <>
                    <div className="flex items-start gap-2.5 flex-1 min-w-0">
                      <span className="text-rose-600 font-bold shrink-0 mt-0.5">🚫</span>
                      <span className="leading-relaxed break-words">{na}</span>
                    </div>
                    {!isReadOnly && (
                      <div className="flex items-center gap-1 shrink-0">
                        <button
                          type="button"
                          onClick={() => handleStartEditNonAction(idx, na)}
                          className="text-zinc-400 hover:text-sky-600 p-1"
                          title="แก้ไขข้อนี้"
                        >
                          ✏️
                        </button>
                        <button
                          type="button"
                          onClick={() => handleRemoveNonAction(idx)}
                          className="text-zinc-400 hover:text-rose-600 p-1"
                          title="ลบข้อนี้"
                        >
                          ✕
                        </button>
                      </div>
                    )}
                  </>
                )}
              </div>
            ))}
          </div>

          {!isReadOnly && (
            <div className="flex items-start gap-2 pt-2">
              <AutoResizeTextarea
                id="non-action-input"
                minRows={2}
                value={newNonAction}
                onChange={(e) => setNewNonAction(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault()
                    handleAddNonAction()
                  }
                }}
                placeholder="เพิ่มกฎเหล็กข้อห้าม เช่น ห้ามดึงเงินจากเป้าหมายสำรองฉุกเฉินไปเสี่ยงเด็ดขาด..."
                className="rounded-xl border border-zinc-200 bg-white p-3 text-xs text-zinc-800 focus:outline-none focus:ring-2 focus:ring-rose-300 leading-relaxed shadow-2xs"
              />
              <button
                type="button"
                onClick={handleAddNonAction}
                className="rounded-xl bg-rose-600 px-5 py-3 text-xs font-bold text-white hover:bg-rose-700 shrink-0 shadow-2xs"
              >
                + เพิ่มกฎเหล็ก
              </button>
            </div>
          )}
        </div>

      </div>

      {/* Action Footer Buttons */}
      <div className="rounded-2xl border border-sky-100 bg-white p-6 shadow-sm flex flex-wrap items-center justify-between gap-4">
        <div>
          {statusText ? (
            <span className="text-xs font-semibold text-sky-700 animate-pulse">
              ⏳ {statusText}
            </span>
          ) : (
            <span className="text-xs text-zinc-500">
              {isConfirmed && !isEditing
                ? 'แกนหลักการลงทุนนี้ได้รับการยืนยันแล้ว สามารถดำเนินการจัด Purpose Buckets ได้'
                : 'เมื่อตรวจทานทั้ง 8 หัวข้อครบถ้วนแล้ว ให้กดยืนยันแกนหลักการลงทุน'}
            </span>
          )}
        </div>

        <div className="flex flex-wrap items-center gap-3">
          {(!isConfirmed || isEditing) && (
            <button
              type="button"
              onClick={handleSaveDraft}
              disabled={loading}
              className="rounded-xl border border-sky-300 bg-sky-50 px-5 py-2.5 text-xs font-bold text-sky-700 hover:bg-sky-100 active:scale-98 transition-all disabled:opacity-50"
            >
              💾 บันทึกแบบร่าง
            </button>
          )}

          {(!isConfirmed || isEditing) && (
            <button
              type="button"
              onClick={() => handleConfirm(false)}
              disabled={!canConfirm}
              className="rounded-xl border border-emerald-300 bg-emerald-50 px-5 py-2.5 text-xs font-bold text-emerald-800 shadow-sm hover:bg-emerald-100 active:scale-98 transition-all disabled:opacity-40 disabled:cursor-not-allowed"
            >
              ✓ ยืนยันแกนหลักการลงทุน
            </button>
          )}

          {(!isConfirmed || isEditing) && (
            <button
              type="button"
              onClick={() => handleConfirm(true)}
              disabled={!canConfirm}
              className="rounded-xl bg-emerald-600 px-6 py-2.5 text-xs font-bold text-white shadow-sm hover:bg-emerald-700 active:scale-98 transition-all disabled:opacity-40 disabled:cursor-not-allowed"
            >
              🚀 ยืนยันและไปขั้นจัด Buckets ทันที →
            </button>
          )}

          {isConfirmed && !isEditing && (
            <button
              type="button"
              onClick={onProceedToBuckets}
              className="rounded-xl bg-gradient-to-r from-purple-600 to-indigo-600 px-6 py-2.5 text-xs font-bold text-white shadow-md shadow-purple-500/20 hover:from-purple-700 hover:to-indigo-700 active:scale-98 transition-all"
            >
              🚀 ไปขั้นจัด Purpose Buckets →
            </button>
          )}
        </div>
      </div>
    </div>
  )
}
