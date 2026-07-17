import { useState } from 'react'
import FormModal, { FormField, FormInput, FormSelect } from './FormModal'
import { IncomeIcon } from '../icons/PortfolioIcons'
import { api } from '../../../api/client'
import type { ActualPortfolioStateDTO } from '../../../api/types'

interface Props {
  holdingsSymbols: string[]
  onClose: () => void
  onSuccess: (state: ActualPortfolioStateDTO) => void
}

export default function IncomeModal({ holdingsSymbols, onClose, onSuccess }: Props) {
  const [incomeType, setIncomeType] = useState<'Dividend' | 'Interest' | 'Rental' | 'Other'>('Dividend')
  const [amountThb, setAmountThb] = useState<string>('')
  const [sourceSymbol, setSourceSymbol] = useState<string>('')
  const [date, setDate] = useState<string>(() => new Date().toISOString().split('T')[0] || '')
  const [notes, setNotes] = useState<string>('')

  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!amountThb || parseFloat(amountThb) <= 0) {
      setError('กรุณาระบุจำนวนเงินบาทที่ได้รับ (มากกว่า 0)')
      return
    }
    setLoading(true)
    setError(null)
    try {
      const state = await api.recordIncome({
        income_type: incomeType,
        amount_thb: parseFloat(amountThb),
        source_symbol: sourceSymbol || null,
        date: date || null,
        notes: notes.trim(),
      })
      onSuccess(state)
      onClose()
    } catch (err: any) {
      setError(err?.message || 'บันทึกรายรับไม่สำเร็จ')
    } finally {
      setLoading(false)
    }
  }

  return (
    <FormModal
      titleId="income-modal-title"
      title="บันทึกกระแสเงินสดรับ (Passive Income / Dividend)"
      icon={<IncomeIcon className="w-5 h-5 text-emerald-600" />}
      onClose={onClose}
      onSubmit={handleSubmit}
      error={error}
      loading={loading}
      submitText="💰 บันทึกรายรับเข้าพอร์ต"
      submitClassName="bg-emerald-600 hover:bg-emerald-700"
      panelClassName="max-w-md rounded-2xl border border-sky-100 bg-white p-6 shadow-2xl"
    >
      <div className="grid grid-cols-2 gap-3">
        <FormField label="ประเภทรายรับ (Income Type)">
          <FormSelect
            value={incomeType}
            onChange={(e) => setIncomeType(e.target.value as any)}
            className="font-bold"
          >
            <option value="Dividend">เงินปันผล (Dividend)</option>
            <option value="Interest">ดอกเบี้ย (Interest)</option>
            <option value="Rental">ค่าเช่า (Rental)</option>
            <option value="Other">อื่น ๆ (Other)</option>
          </FormSelect>
        </FormField>

        <FormField label="จำนวนเงินที่ได้รับ (THB)" required>
          <FormInput
            type="number"
            step="any"
            min="0.01"
            value={amountThb}
            onChange={(e) => setAmountThb(e.target.value)}
            placeholder="0.00 บาท"
            className="font-bold"
            required
          />
        </FormField>
      </div>

      {incomeType === 'Dividend' && (
        <FormField
          label="เชื่อมโยงจากสินทรัพย์ในพอร์ต (Source Ticker)"
          hint="* หากระบุ Ticker ระบบจะบวกเพิ่มเข้ายอด accumulated_dividend_thb ของสินทรัพย์ตัวนี้ให้อัตโนมัติ"
        >
          <FormSelect
            value={sourceSymbol}
            onChange={(e) => setSourceSymbol(e.target.value)}
            className="font-mono font-medium"
          >
            <option value="">-- ไม่ระบุ Ticker / รับจากกองกลาง --</option>
            {holdingsSymbols.map((s) => (
              <option key={s} value={s}>
                {s}
              </option>
            ))}
          </FormSelect>
        </FormField>
      )}

      <div className="grid grid-cols-2 gap-3">
        <FormField label="วันที่รับเงิน">
          <FormInput
            type="date"
            value={date}
            onChange={(e) => setDate(e.target.value)}
            className="font-medium"
          />
        </FormField>

        <FormField label="บันทึกเพิ่มเติม (Notes)">
          <FormInput
            type="text"
            value={notes}
            onChange={(e) => setNotes(e.target.value)}
            placeholder="เช่น ปันผล Q2/2569..."
          />
        </FormField>
      </div>
    </FormModal>
  )
}
