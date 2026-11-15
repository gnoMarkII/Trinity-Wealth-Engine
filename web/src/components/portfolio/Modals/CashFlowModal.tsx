import { useState } from 'react'
import FormModal, { FormField, FormInput, FormSelect } from './FormModal'
import { CashFlowIcon } from '../icons/PortfolioIcons'
import { api } from '../../../api/client'
import { todayLocalDateString } from '../../../utils/formatters'
import type { ActualPortfolioStateDTO } from '../../../api/types'

interface Props {
  selectedPortfolioId: string
  onClose: () => void
  onSuccess: (state: ActualPortfolioStateDTO) => void
}

export default function CashFlowModal({ selectedPortfolioId, onClose, onSuccess }: Props) {
  const [action, setAction] = useState<'deposit' | 'withdraw'>('deposit')
  const [amount, setAmount] = useState<string>('')
  const [currency, setCurrency] = useState<'THB' | 'USD'>('THB')
  const [exchangeRate, setExchangeRate] = useState<string>('')
  const [date, setDate] = useState<string>(() => todayLocalDateString())
  const [notes, setNotes] = useState<string>('')

  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!amount || parseFloat(amount) <= 0) {
      setError('กรุณาระบุจำนวนเงินให้ถูกต้อง')
      return
    }
    setLoading(true)
    setError(null)
    try {
      const state = await api.manageCashFlow(
        {
          action,
          amount: parseFloat(amount),
          currency,
          exchange_rate: exchangeRate ? parseFloat(exchangeRate) : null,
          date: date || null,
          notes: notes.trim(),
        },
        selectedPortfolioId
      )
      onSuccess(state)
      onClose()
    } catch (err: any) {
      setError(err?.message || 'บันทึกกระแสเงินสดไม่สำเร็จ')
    } finally {
      setLoading(false)
    }
  }

  return (
    <FormModal
      titleId="cashflow-modal-title"
      title="จัดการเงินสด (Deposit / Withdraw)"
      icon={<CashFlowIcon className="w-5 h-5 text-flow-blue" />}
      onClose={onClose}
      onSubmit={handleSubmit}
      error={error}
      loading={loading}
      submitText={action === 'deposit' ? '✅ ยืนยันการฝากเงิน (Deposit)' : '🚨 ยืนยันการถอนเงิน (Withdraw)'}
      submitClassName={action === 'deposit' ? 'bg-emerald-600 hover:bg-emerald-700' : 'bg-rose-600 hover:bg-rose-700'}
      panelClassName="max-w-md rounded-2xl border border-sky-100 bg-white p-6 shadow-2xl"
    >
      <FormField label="ประเภทการทำรายการ">
        <div className="flex rounded-xl bg-slate-100 p-1 font-bold">
          <button
            type="button"
            onClick={() => setAction('deposit')}
            className={`flex-1 rounded-lg py-2 text-center transition-all ${
              action === 'deposit' ? 'bg-emerald-600 text-white shadow-sm' : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            ➕ ฝากเงินเข้าพอร์ต (Deposit)
          </button>
          <button
            type="button"
            onClick={() => setAction('withdraw')}
            className={`flex-1 rounded-lg py-2 text-center transition-all ${
              action === 'withdraw' ? 'bg-rose-600 text-white shadow-sm' : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            ➖ ถอนเงินออก (Withdraw)
          </button>
        </div>
      </FormField>

      <div className="grid grid-cols-2 gap-3">
        <FormField label="จำนวนเงิน (Amount)" required>
          <FormInput
            type="number"
            step="any"
            min="0.01"
            value={amount}
            onChange={(e) => setAmount(e.target.value)}
            placeholder="0.00"
            className="font-mono font-bold"
            required
          />
        </FormField>

        <FormField label="สกุลเงิน (Currency)">
          <FormSelect
            value={currency}
            onChange={(e) => setCurrency(e.target.value as 'THB' | 'USD')}
            className="font-bold"
          >
            <option value="THB">THB (เงินบาท)</option>
            <option value="USD">USD (ดอลลาร์สหรัฐ)</option>
          </FormSelect>
        </FormField>
      </div>

      {currency === 'USD' && (
        <FormField
          label="อัตราแลกเปลี่ยน (USD/THB)"
          hint="* ว่างไว้เพื่อใช้เรทตลาดปัจจุบันอัตโนมัติจากระบบ"
        >
          <FormInput
            type="number"
            step="any"
            min="0"
            value={exchangeRate}
            onChange={(e) => setExchangeRate(e.target.value)}
            placeholder="e.g. 34.50"
            className="font-mono"
          />
        </FormField>
      )}

      <div className="grid grid-cols-2 gap-3">
        <FormField label="วันที่ทำรายการ">
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
            placeholder="เช่น โอนเงินเพิ่มพอร์ต..."
          />
        </FormField>
      </div>
    </FormModal>
  )
}
