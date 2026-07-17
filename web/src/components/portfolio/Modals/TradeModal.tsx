import { useState } from 'react'
import FormModal, { FormField, FormInput, FormSelect } from './FormModal'
import { TradeIcon } from '../icons/PortfolioIcons'
import { api } from '../../../api/client'
import type { AllocationTargetDTO, ActualPortfolioStateDTO } from '../../../api/types'

interface Props {
  targets: AllocationTargetDTO[]
  onClose: () => void
  onSuccess: (state: ActualPortfolioStateDTO) => void
}

export default function TradeModal({ targets, onClose, onSuccess }: Props) {
  const [symbol, setSymbol] = useState('')
  const [assetType, setAssetType] = useState('Stock')
  const [action, setAction] = useState<'buy' | 'sell'>('buy')
  const [units, setUnits] = useState<string>('')
  const [price, setPrice] = useState<string>('')
  const [currency, setCurrency] = useState<'THB' | 'USD'>('THB')
  const [exchangeRate, setExchangeRate] = useState<string>('')
  const [date, setDate] = useState<string>(() => new Date().toISOString().split('T')[0] || '')
  const [notes, setNotes] = useState<string>('')
  const [bucketId, setBucketId] = useState<string>('')

  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!symbol.trim() || !units || !price) {
      setError('กรุณาระบุ Symbol, จำนวนหน่วย และราคาต่อหน่วย')
      return
    }
    setLoading(true)
    setError(null)
    try {
      const state = await api.executeTrade({
        symbol: symbol.trim().toUpperCase(),
        asset_type: assetType,
        action,
        units: parseFloat(units),
        price: parseFloat(price),
        currency,
        exchange_rate: exchangeRate ? parseFloat(exchangeRate) : null,
        date: date || null,
        notes: notes.trim(),
        bucket_id: bucketId || null,
      })
      onSuccess(state)
      onClose()
    } catch (err: any) {
      setError(err?.message || 'บันทึกการเทรดไม่สำเร็จ')
    } finally {
      setLoading(false)
    }
  }

  return (
    <FormModal
      titleId="trade-modal-title"
      title="บันทึกคำสั่งซื้อ/ขายสินทรัพย์ (Trade Entry)"
      icon={<TradeIcon className="w-5 h-5 text-flow-blue" />}
      onClose={onClose}
      onSubmit={handleSubmit}
      error={error}
      loading={loading}
      submitText={action === 'buy' ? '✅ ยืนยันการซื้อ (BUY)' : '🚨 ยืนยันการขาย (SELL)'}
      submitClassName={action === 'buy' ? 'bg-emerald-600 hover:bg-emerald-700' : 'bg-rose-600 hover:bg-rose-700'}
    >
      <div className="grid grid-cols-2 gap-3">
        <FormField label="ประเภทรายการ">
          <div className="flex rounded-xl bg-zinc-100 p-1 font-bold">
            <button
              type="button"
              onClick={() => setAction('buy')}
              className={`flex-1 rounded-lg py-1.5 text-center transition-all ${
                action === 'buy' ? 'bg-emerald-600 text-white shadow-sm' : 'text-zinc-600 hover:text-zinc-900'
              }`}
            >
              ซื้อ (BUY)
            </button>
            <button
              type="button"
              onClick={() => setAction('sell')}
              className={`flex-1 rounded-lg py-1.5 text-center transition-all ${
                action === 'sell' ? 'bg-rose-600 text-white shadow-sm' : 'text-zinc-600 hover:text-zinc-900'
              }`}
            >
              ขาย (SELL)
            </button>
          </div>
        </FormField>

        <FormField label="สัญลักษณ์ / Ticker Symbol" required>
          <FormInput
            type="text"
            value={symbol}
            onChange={(e) => setSymbol(e.target.value)}
            placeholder="e.g. AAPL, PTT, NVDA"
            className="font-mono font-bold uppercase"
            required
          />
        </FormField>
      </div>

      <div className="grid grid-cols-2 gap-3">
        <FormField label="ประเภทสินทรัพย์ (Asset Type)">
          <FormSelect
            value={assetType}
            onChange={(e) => setAssetType(e.target.value)}
          >
            <option value="Stock">หุ้น (Stock)</option>
            <option value="ETF">กองทุนรวม / ETF</option>
            <option value="REIT">อสังหาฯ / REIT</option>
            <option value="Crypto">คริปโต (Crypto)</option>
            <option value="Bond">ตราสารหนี้ (Bond)</option>
          </FormSelect>
        </FormField>

        <FormField label="กลุ่มกลยุทธ์ (Bucket Assign)">
          <FormSelect
            value={bucketId}
            onChange={(e) => setBucketId(e.target.value)}
          >
            <option value="">-- ไม่ระบุกลุ่ม --</option>
            {targets.map((t) => (
              <option key={t.bucket_id} value={t.bucket_id}>
                {t.name} ({t.bucket_id})
              </option>
            ))}
          </FormSelect>
        </FormField>
      </div>

      <div className="grid grid-cols-3 gap-3">
        <FormField label="จำนวนหน่วย (Units)" required>
          <FormInput
            type="number"
            step="any"
            min="0"
            value={units}
            onChange={(e) => setUnits(e.target.value)}
            placeholder="0.00"
            className="font-mono font-bold"
            required
          />
        </FormField>

        <FormField label="ราคาต่อหน่วย (Price)" required>
          <FormInput
            type="number"
            step="any"
            min="0"
            value={price}
            onChange={(e) => setPrice(e.target.value)}
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
            <option value="THB">THB (฿)</option>
            <option value="USD">USD ($)</option>
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
            placeholder="เหตุผลในการเทรด..."
          />
        </FormField>
      </div>
    </FormModal>
  )
}
