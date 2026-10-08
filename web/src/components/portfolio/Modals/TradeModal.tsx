import { useState } from 'react'
import FormModal, { FormField, FormInput, FormSelect } from './FormModal'
import { TradeIcon } from '../icons/PortfolioIcons'
import { api } from '../../../api/client'
import { todayLocalDateString } from '../../../utils/formatters'
import type { AllocationTargetDTO, ActualPortfolioStateDTO, ActualHoldingDTO } from '../../../api/types'

interface Props {
  targets: AllocationTargetDTO[]
  holdings?: ActualHoldingDTO[]
  selectedPortfolioId: string
  onClose: () => void
  onSuccess: (state: ActualPortfolioStateDTO) => void
}

export default function TradeModal({ targets, holdings, selectedPortfolioId, onClose, onSuccess }: Props) {
  const [symbol, setSymbol] = useState('')
  const [assetType, setAssetType] = useState('Stock')
  const [action, setAction] = useState<'buy' | 'sell'>('buy')
  const [units, setUnits] = useState<string>('')
  const [price, setPrice] = useState<string>('')
  const [currency, setCurrency] = useState<'THB' | 'USD'>('THB')
  const [exchangeRate, setExchangeRate] = useState<string>('')
  const [date, setDate] = useState<string>(() => todayLocalDateString())
  const [notes, setNotes] = useState<string>('')
  const [bucketId, setBucketId] = useState<string>('')

  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [fxSource, setFxSource] = useState<string | null>(null)
  const [fetchingFx, setFetchingFx] = useState(false)

  const fetchFxForDate = async (targetDate: string) => {
    if (currency !== 'USD') return
    setFetchingFx(true)
    try {
      const res = await api.getFxRate(targetDate, selectedPortfolioId)
      if (res?.rate) {
        setExchangeRate(res.rate.toFixed(4))
        setFxSource(res.source)
      }
    } catch {
      // fallback smoothly
    } finally {
      setFetchingFx(false)
    }
  }

  const cashUsd = holdings?.find((h) => h.symbol === 'CASH_USD')?.units ?? 0
  const cashThb = holdings?.find((h) => h.symbol === 'CASH_THB')?.units ?? 0
  const availableCash = currency === 'USD' ? cashUsd : cashThb
  const totalRequired = (parseFloat(units) || 0) * (parseFloat(price) || 0)
  const missingAmount = Math.max(0, totalRequired - availableCash)

  const handleQuickTopupAndBuy = async () => {
    if (!symbol.trim() || !units || !price) return
    const topupAmount = missingAmount > 0 ? missingAmount : totalRequired
    setLoading(true)
    setError(null)
    try {
      // 1. Top-up cash
      await api.manageCashFlow({
        action: 'deposit',
        amount: topupAmount,
        currency,
        exchange_rate: exchangeRate ? parseFloat(exchangeRate) : null,
        date: date || null,
        notes: `Auto top-up for ${symbol.trim().toUpperCase()} buy trade`,
      }, selectedPortfolioId)

      // 2. Auto retry buy trade
      const state = await api.executeTrade({
        symbol: symbol.trim().toUpperCase(),
        asset_type: assetType,
        action: 'buy',
        units: parseFloat(units),
        price: parseFloat(price),
        currency,
        exchange_rate: exchangeRate ? parseFloat(exchangeRate) : null,
        date: date || null,
        notes: notes.trim(),
        bucket_id: bucketId || null,
      }, selectedPortfolioId)

      onSuccess(state)
      onClose()
    } catch (err: any) {
      setError(err?.message || 'เติมเงินสดและสั่งซื้อไม่สำเร็จ')
    } finally {
      setLoading(false)
    }
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!symbol.trim() || !units || !price) {
      setError('กรุณาระบุ Symbol, จำนวนหน่วย และราคาต่อหน่วย')
      return
    }
    setLoading(true)
    setError(null)
    try {
      const state = await api.executeTrade(
        {
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
        },
        selectedPortfolioId
      )
      onSuccess(state)
      onClose()
    } catch (err: any) {
      setError(err?.message || 'บันทึกการเทรดไม่สำเร็จ')
    } finally {
      setLoading(false)
    }
  }

  const renderErrorNode = () => {
    if (!error) return null
    const isInsufficientCash = /(?:Insufficient cash|เงินสดไม่พอ)/i.test(error) && action === 'buy'

    return (
      <div className="space-y-2">
        <div>{error}</div>
        {isInsufficientCash && (
          <button
            type="button"
            onClick={handleQuickTopupAndBuy}
            disabled={loading}
            className="mt-1 flex items-center justify-center gap-1.5 w-full rounded-xl bg-amber-600 px-3.5 py-2 text-xs font-bold text-white shadow-md hover:bg-amber-700 disabled:opacity-50 transition-colors cursor-pointer"
          >
            <span>💵</span>
            <span>
              เติมเงินสดเพิ่ม {missingAmount > 0 ? missingAmount.toFixed(2) : totalRequired.toFixed(2)} {currency} & ยืนยันการซื้อทันที
            </span>
          </button>
        )}
      </div>
    )
  }

  return (
    <FormModal
      titleId="trade-modal-title"
      title="บันทึกคำสั่งซื้อ/ขายสินทรัพย์ (Trade Entry)"
      icon={<TradeIcon className="w-5 h-5 text-flow-blue" />}
      onClose={onClose}
      onSubmit={handleSubmit}
      error={renderErrorNode()}
      loading={loading}
      submitText={action === 'buy' ? '✅ ยืนยันการซื้อ (BUY)' : '🚨 ยืนยันการขาย (SELL)'}
      submitClassName={action === 'buy' ? 'bg-emerald-600 hover:bg-emerald-700' : 'bg-rose-600 hover:bg-rose-700'}
    >
      <div className="flex items-center justify-between rounded-xl bg-zinc-50 border border-zinc-200/80 px-3 py-2 text-xs">
        <span className="font-semibold text-zinc-600 flex items-center gap-1.5">
          <span>💵</span> เงินสดคงเหลือในพอร์ต:
        </span>
        <div className="flex items-center gap-2 font-mono font-bold">
          <span className={cashUsd > 0 ? 'text-emerald-700' : 'text-zinc-500'}>
            ${cashUsd.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })} USD
          </span>
          <span className="text-zinc-300">|</span>
          <span className={cashThb > 0 ? 'text-emerald-700' : 'text-zinc-500'}>
            ฿{cashThb.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })} THB
          </span>
        </div>
      </div>
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
          hint={
            fetchingFx
              ? '⏳ กำลังดึงเรทอัตราแลกเปลี่ยน...'
              : fxSource
              ? `* เรท (${fxSource}): ${exchangeRate || '-'} THB/USD`
              : '* ว่างไว้เพื่อใช้เรทตลาดปัจจุบันอัตโนมัติจากระบบ'
          }
        >
          <div className="relative">
            <FormInput
              type="number"
              step="any"
              min="0"
              value={exchangeRate}
              onChange={(e) => {
                setExchangeRate(e.target.value)
                setFxSource('manual')
              }}
              placeholder="e.g. 34.50"
              className="font-mono pr-16"
            />
            <button
              type="button"
              onClick={() => fetchFxForDate(date)}
              disabled={fetchingFx}
              className="absolute right-1.5 top-1/2 -translate-y-1/2 rounded-lg bg-zinc-100 hover:bg-zinc-200 px-2 py-1 text-[11px] font-semibold text-zinc-600 disabled:opacity-50 transition-colors"
            >
              {fetchingFx ? '⏳' : '🔄 ดึงเรท'}
            </button>
          </div>
        </FormField>
      )}

      <div className="grid grid-cols-2 gap-3">
        <FormField label="วันที่ทำรายการ">
          <FormInput
            type="date"
            value={date}
            onChange={(e) => {
              const newDate = e.target.value
              setDate(newDate)
              if (currency === 'USD' && newDate) {
                fetchFxForDate(newDate)
              }
            }}
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
