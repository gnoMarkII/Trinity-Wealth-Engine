import { useState } from 'react'
import FormModal, { FormField, FormInput, FormSelect } from './FormModal'
import { GoalIcon } from '../icons/PortfolioIcons'
import { api } from '../../../api/client'
import type { ActualGoalItemDTO, ActualGoalsResponseDTO } from '../../../api/types'

interface Props {
  initialGoal?: ActualGoalItemDTO | null
  onClose: () => void
  onSuccess: (response: ActualGoalsResponseDTO) => void
}

export default function GoalModal({ initialGoal, onClose, onSuccess }: Props) {
  const isEdit = !!initialGoal
  const [name, setName] = useState(initialGoal?.name || '')
  const [goalType, setGoalType] = useState<'nav_target' | 'cash_target' | 'passive_income_ytd'>(
    (initialGoal?.goal_type as any) || 'nav_target'
  )
  const [targetAmountThb, setTargetAmountThb] = useState<string>(
    initialGoal?.target_amount_thb !== undefined ? initialGoal.target_amount_thb.toString() : ''
  )
  const [deadline, setDeadline] = useState<string>(initialGoal?.deadline || '')
  const [yearsFromNow, setYearsFromNow] = useState<string>('')
  const [notes, setNotes] = useState(initialGoal?.notes || '')

  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!name.trim() || !targetAmountThb || parseFloat(targetAmountThb) <= 0) {
      setError('กรุณาระบุชื่อเป้าหมาย และจำนวนเป้าหมาย THB ที่มากกว่า 0')
      return
    }
    setLoading(true)
    setError(null)
    try {
      const res = await api.upsertGoal(name.trim(), {
        goal_type: goalType,
        target_amount_thb: parseFloat(targetAmountThb),
        deadline: deadline || null,
        years_from_now: yearsFromNow ? parseInt(yearsFromNow, 10) : null,
        notes: notes.trim() || null,
      })
      onSuccess(res)
      onClose()
    } catch (err: any) {
      setError(err?.message || 'บันทึกเป้าหมายไม่สำเร็จ')
    } finally {
      setLoading(false)
    }
  }

  return (
    <FormModal
      titleId="goal-modal-title"
      title={isEdit ? `แก้ไขเป้าหมาย: ${initialGoal?.name}` : 'ตั้งเป้าหมายทางการเงินใหม่ (Add Goal)'}
      icon={<GoalIcon className="w-5 h-5 text-flow-blue" />}
      onClose={onClose}
      onSubmit={handleSubmit}
      error={error}
      loading={loading}
      submitText={isEdit ? '💾 บันทึกเป้าหมาย' : '+ ตั้งเป้าหมายใหม่'}
      submitClassName="bg-flow-blue hover:bg-sky-600"
      panelClassName="max-w-md rounded-2xl border border-sky-100 bg-white p-6 shadow-2xl"
    >
      <FormField label="ชื่อเป้าหมาย (Goal Name)" required>
        <FormInput
          type="text"
          value={name}
          onChange={(e) => setName(e.target.value)}
          disabled={isEdit}
          placeholder="e.g. พอร์ต 10 ล้านแรก, เงินปันผลเดือนละ 2 หมื่น"
          className="font-bold"
          required
        />
      </FormField>

      <div className="grid grid-cols-2 gap-3">
        <FormField label="ประเภทเป้าหมาย (Goal Type)">
          <FormSelect
            value={goalType}
            onChange={(e) => setGoalType(e.target.value as any)}
          >
            <option value="nav_target">มูลค่าพอร์ตรวม (Total NAV)</option>
            <option value="cash_target">เงินสำรองในพอร์ต (Cash Target)</option>
            <option value="passive_income_ytd">ปันผลสะสมปีนี้ (Passive Income YTD)</option>
          </FormSelect>
        </FormField>

        <FormField label="เป้าหมายที่ต้องการ (THB)" required>
          <FormInput
            type="number"
            step="any"
            min="0.01"
            value={targetAmountThb}
            onChange={(e) => setTargetAmountThb(e.target.value)}
            placeholder="e.g. 10000000"
            className="font-bold"
            required
          />
        </FormField>
      </div>

      <div className="grid grid-cols-2 gap-3">
        <FormField label="วันที่เป้าหมาย (Deadline - เลือกได้)">
          <FormInput
            type="date"
            value={deadline}
            onChange={(e) => {
              setDeadline(e.target.value)
              if (e.target.value) setYearsFromNow('')
            }}
            className="font-medium"
          />
        </FormField>

        <FormField label="หรืออีกกี่ปีข้างหน้า (Years)">
          <FormInput
            type="number"
            min="1"
            max="100"
            value={yearsFromNow}
            onChange={(e) => {
              setYearsFromNow(e.target.value)
              if (e.target.value) setDeadline('')
            }}
            placeholder="e.g. 5"
            className="font-mono"
          />
        </FormField>
      </div>

      <FormField label="รายละเอียดเพิ่มเติม / กลยุทธ์ (Notes)">
        <FormInput
          type="text"
          value={notes}
          onChange={(e) => setNotes(e.target.value)}
          placeholder="เช่น ทยอยออมเดือนละ 50,000 บาทใน Index ETF"
        />
      </FormField>
    </FormModal>
  )
}
