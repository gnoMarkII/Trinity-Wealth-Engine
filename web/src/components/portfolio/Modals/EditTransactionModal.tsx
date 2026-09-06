import { useState } from 'react'
import FormModal, { FormField, FormInput } from './FormModal'
import { EditIcon } from '../icons/PortfolioIcons'
import { api } from '../../../api/client'
import type { TransactionItemDTO, ActualPortfolioStateDTO, ActualHoldingDTO } from '../../../api/types'

interface Props {
  transaction: TransactionItemDTO
  selectedPortfolioId: string
  holdings?: ActualHoldingDTO[]
  onClose: () => void
  onSuccess: (state: ActualPortfolioStateDTO) => void
}

export default function EditTransactionModal({
  transaction,
  selectedPortfolioId,
  holdings,
  onClose,
  onSuccess,
}: Props) {
  const parseTimestamp = (ts: string) => {
    if (!ts) return { date: '', time: '' }
    if (ts.includes('T')) {
      const [d, t] = ts.split('T')
      return { date: d || '', time: t ? t.slice(0, 8) : '' }
    }
    if (ts.includes(' ')) {
      const [d, t] = ts.split(' ')
      return { date: d || '', time: t ? t.slice(0, 8) : '' }
    }
    return { date: ts, time: '' }
  }

  const initial = parseTimestamp(transaction.timestamp || '')
  const [date, setDate] = useState<string>(initial.date)
  const [time, setTime] = useState<string>(initial.time)
  const [units, setUnits] = useState<string>(transaction.units ? String(transaction.units) : '')
  const [price, setPrice] = useState<string>(transaction.price ? String(transaction.price) : '')
  const [fxRate, setFxRate] = useState<string>(transaction.fx_rate ? String(transaction.fx_rate) : '')
  const [notes, setNotes] = useState<string>(transaction.notes || '')
  const [adjustCash, setAdjustCash] = useState<boolean>(true)

  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [fetchingFx, setFetchingFx] = useState(false)

  const isUSD = transaction.currency === 'USD'
  const isBuy = transaction.action.toUpperCase() === 'BUY'
  const isDime = (transaction.source || '').toUpperCase() === 'DIME'
  const holding = holdings?.find((h) => h.symbol.toUpperCase() === transaction.symbol.toUpperCase())
  const hasSyncedDividends = holding?.dividend_source === 'synced'

  const getLocalDateString = (d = new Date()) => {
    const year = d.getFullYear()
    const month = String(d.getMonth() + 1).padStart(2, '0')
    const day = String(d.getDate()).padStart(2, '0')
    return `${year}-${month}-${day}`
  }

  const getLocalTimeString = (d = new Date()) => {
    const hours = String(d.getHours()).padStart(2, '0')
    const mins = String(d.getMinutes()).padStart(2, '0')
    const secs = String(d.getSeconds()).padStart(2, '0')
    return `${hours}:${mins}:${secs}`
  }

  const handleSetToday = () => {
    if (isDime) return
    const today = getLocalDateString()
    setDate(today)
    if (isUSD) fetchFxForDate(today)
  }

  const handleSetYesterday = () => {
    if (isDime) return
    const yest = new Date()
    yest.setDate(yest.getDate() - 1)
    const yestStr = getLocalDateString(yest)
    setDate(yestStr)
    if (isUSD) fetchFxForDate(yestStr)
  }

  const handleSetCurrentTime = () => {
    if (isDime) return
    setTime(getLocalTimeString())
  }

  // Calculate live preview deltas
  const originalNativeAmount = transaction.units * transaction.price
  const newUnitsNum = parseFloat(units) || 0
  const newPriceNum = parseFloat(price) || 0
  const newNativeAmount = newUnitsNum * newPriceNum
  const deltaNative = newNativeAmount - originalNativeAmount

  const fetchFxForDate = async (targetDate: string) => {
    if (!targetDate || isDime) return
    setFetchingFx(true)
    try {
      const dateOnly = targetDate.includes('T') ? targetDate.split('T')[0] : targetDate.split(' ')[0]
      const res = await api.getFxRate(dateOnly, selectedPortfolioId)
      if (res?.rate) {
        setFxRate(String(res.rate))
      }
    } catch {
      // fallback smoothly
    } finally {
      setFetchingFx(false)
    }
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()

    setLoading(true)
    setError(null)

    try {
      let payload: any
      if (isDime) {
        payload = { notes: notes.trim() }
      } else {
        if (!date) {
          setError('กรุณาระบุวันที่ทำรายการ')
          setLoading(false)
          return
        }
        if (!units || newUnitsNum <= 0) {
          setError('กรุณาระบุจำนวนหน่วยที่มากกว่า 0')
          setLoading(false)
          return
        }
        if (!price || newPriceNum <= 0) {
          setError('กรุณาระบุราคาต่อหน่วยที่มากกว่า 0')
          setLoading(false)
          return
        }
        const formattedTimestamp = date ? (time ? `${date}T${time}` : date) : null
        payload = {
          timestamp: formattedTimestamp,
          units: newUnitsNum,
          price: newPriceNum,
          fx_rate: isUSD && fxRate ? parseFloat(fxRate) : null,
          notes: notes.trim(),
          adjust_cash: adjustCash,
        }
      }

      const updatedState = await api.editTransaction(
        transaction.transaction_id,
        payload,
        selectedPortfolioId
      )

      onSuccess(updatedState)
      onClose()
    } catch (err: any) {
      setError(err?.message || 'แก้ไขรายการ Transaction ไม่สำเร็จ')
    } finally {
      setLoading(false)
    }
  }

  return (
    <FormModal
      titleId="edit-transaction-title"
      title={`แก้ไขรายการ: ${transaction.symbol} (${transaction.action})`}
      icon={<EditIcon className="w-5 h-5" />}
      onClose={onClose}
      onSubmit={handleSubmit}
      loading={loading}
      error={error}
      submitText="บันทึกการแก้ไข"
      loadingText="กำลังคำนวณและ Replay..."
      submitClassName="bg-flow-blue hover:bg-sky-600"
    >
      {/* Warning for synced dividends */}
      {hasSyncedDividends && (
        <div className="rounded-xl border border-amber-200 bg-amber-50/90 p-3 text-xs text-amber-900">
          <div className="flex items-start gap-2">
            <span className="text-sm">⚠️</span>
            <div className="space-y-1">
              <p className="font-semibold">สินทรัพย์นี้มีการ Sync เงินปันผลอัตโนมัติ</p>
              <p className="text-[11px] text-amber-800">
                การแก้ไขจำนวนหุ้นหรือวันที่อาจส่งผลต่อการคำนวณปันผลย้อนหลัง ระบบจะรีเซ็ตสถานะเป็น Manual เพื่อให้คุณสามารถกด Sync ใหม่ได้เมื่อพร้อม
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Informative banner for Dime imports */}
      {isDime && (
        <div className="rounded-xl border border-sky-300 bg-sky-50/90 p-3 text-xs text-sky-950 dark:border-sky-800 dark:bg-sky-950/40 dark:text-sky-200">
          <div className="flex items-start gap-2">
            <span className="text-sm">ℹ️</span>
            <div className="space-y-1">
              <p className="font-semibold">รายการนำเข้าอัตโนมัติจาก Dime {transaction.confirmation_no ? `(${transaction.confirmation_no})` : ''}</p>
              <p className="text-[11px] leading-relaxed opacity-90">
                ตัวเลขเศรษฐศาสตร์ (จำนวนหุ้น, ราคา, วันที่, FX) ถูกล็อคเพื่อรักษาความถูกต้องทางบัญชี หากต้องการแก้ไขตัวเลข ให้ทำการ Void รายการนี้แล้วนำเข้าใหม่ (สามารถแก้ไข Notes ได้)
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Date & Time Selection with Quick Presets */}
      <div className="space-y-1.5">
        <div className="flex items-center justify-between">
          <span className="text-xs font-semibold text-zinc-700">
            วันและเวลาที่ทำรายการ (Date & Time) <span className="text-rose-500">*</span>
          </span>
          {!isDime && (
            <div className="flex items-center gap-1.5">
              <button
                type="button"
                onClick={handleSetToday}
                className="rounded-lg border border-sky-200 bg-sky-50 px-2.5 py-0.5 text-[11px] font-semibold text-flow-blue hover:bg-sky-100 transition-colors cursor-pointer"
              >
                📅 วันนี้
              </button>
              <button
                type="button"
                onClick={handleSetYesterday}
                className="rounded-lg border border-zinc-200 bg-zinc-50 px-2.5 py-0.5 text-[11px] font-semibold text-zinc-600 hover:bg-zinc-100 transition-colors cursor-pointer"
              >
                ⏮️ เมื่อวาน
              </button>
              <button
                type="button"
                onClick={handleSetCurrentTime}
                className="rounded-lg border border-zinc-200 bg-zinc-50 px-2 py-0.5 text-[11px] font-semibold text-zinc-500 hover:bg-zinc-100 transition-colors cursor-pointer"
                title="ตั้งเวลาเป็นเวลาปัจจุบัน"
              >
                🕒 ตอนนี้
              </button>
            </div>
          )}
        </div>

        <div className="grid grid-cols-12 gap-2">
          {/* Date Picker */}
          <div className={isUSD ? 'col-span-6 sm:col-span-7' : 'col-span-7 sm:col-span-8'}>
            <FormInput
              type="date"
              value={date}
              disabled={isDime}
              onChange={(e) => {
                const newDate = e.target.value
                setDate(newDate)
                if (isUSD && newDate) {
                  fetchFxForDate(newDate)
                }
              }}
              required={!isDime}
              className={`font-medium text-xs w-full ${isDime ? 'opacity-60 cursor-not-allowed bg-zinc-100' : 'cursor-pointer'}`}
            />
          </div>

          {/* Time Picker */}
          <div className={isUSD ? 'col-span-3' : 'col-span-5 sm:col-span-4'}>
            <FormInput
              type="time"
              step="1"
              value={time}
              disabled={isDime}
              onChange={(e) => setTime(e.target.value)}
              className={`font-mono text-xs w-full ${isDime ? 'opacity-60 cursor-not-allowed bg-zinc-100' : ''}`}
            />
          </div>

          {/* Fetch FX Button (for USD) */}
          {isUSD && !isDime && (
            <div className="col-span-3 sm:col-span-2 flex items-center">
              <button
                type="button"
                disabled={fetchingFx || !date}
                onClick={() => fetchFxForDate(date)}
                className="w-full h-full rounded-xl border border-sky-200 bg-sky-50 px-2 py-1.5 text-[11px] font-semibold text-flow-blue hover:bg-sky-100 transition-colors disabled:opacity-50 flex items-center justify-center cursor-pointer shadow-2xs"
                title="ดึงอัตราแลกเปลี่ยนย้อนหลังตามวันที่ที่เลือก"
              >
                {fetchingFx ? '⏳' : 'ดึง FX'}
              </button>
            </div>
          )}
        </div>
      </div>

      {/* Units & Price */}
      <div className="grid grid-cols-2 gap-3">
        <FormField label="จำนวนหุ้น (Units)" required={!isDime}>
          <FormInput
            type="number"
            step="any"
            min="0.000001"
            value={units}
            disabled={isDime}
            onChange={(e) => setUnits(e.target.value)}
            placeholder="0.00"
            className={`font-mono ${isDime ? 'opacity-60 cursor-not-allowed bg-zinc-100' : ''}`}
            required={!isDime}
          />
        </FormField>

        <FormField label={`ราคาต่อหน่วย (${transaction.currency})`} required={!isDime}>
          <FormInput
            type="number"
            step="any"
            min="0.0001"
            value={price}
            disabled={isDime}
            onChange={(e) => setPrice(e.target.value)}
            placeholder="0.00"
            className={`font-mono ${isDime ? 'opacity-60 cursor-not-allowed bg-zinc-100' : ''}`}
            required={!isDime}
          />
        </FormField>
      </div>

      {/* FX Rate for USD */}
      {isUSD && (
        <FormField label="อัตราแลกเปลี่ยน (USDTHB FX Rate)">
          <FormInput
            type="number"
            step="0.0001"
            value={fxRate}
            disabled={isDime}
            onChange={(e) => setFxRate(e.target.value)}
            placeholder="เช่น 36.5000"
            className={`font-mono ${isDime ? 'opacity-60 cursor-not-allowed bg-zinc-100' : ''}`}
          />
        </FormField>
      )}

      {/* Notes */}
      <FormField label="บันทึกช่วยจำ (Notes)">
        <FormInput
          type="text"
          value={notes}
          onChange={(e) => setNotes(e.target.value)}
          placeholder="ระบุเหตุผลหรือบันทึกเพิ่มเติม..."
        />
      </FormField>

      {/* Cash Adjustment Checkbox */}
      <div className="rounded-xl border border-sky-100 bg-sky-50/50 p-3.5 space-y-2">
        <div className="flex items-start gap-2.5 select-none">
          <input
            id="edit-tx-adjust-cash"
            type="checkbox"
            checked={adjustCash}
            disabled={isDime}
            onChange={(e) => setAdjustCash(e.target.checked)}
            className="mt-0.5 h-4 w-4 rounded border-sky-300 text-flow-blue focus:ring-flow-blue cursor-pointer"
          />
          <div className="space-y-0.5">
            <label htmlFor="edit-tx-adjust-cash" className="text-xs font-semibold text-zinc-900 cursor-pointer block">
              ปรับปรุงยอดเงินสด (CASH_{transaction.currency}) อัตโนมัติ
            </label>
            <p className="text-[11px] text-zinc-500">
              {isBuy
                ? 'หากซื้อเพิ่มขึ้นจะหักเงินสดเพิ่ม หากซื้อลดลงจะคืนเงินสดเข้าบัญชี'
                : 'หากขายเพิ่มขึ้นจะเพิ่มเงินสด หากขายลดลงจะหักเงินสดออกจากบัญชี'}
            </p>
          </div>
        </div>

        {/* Live Delta preview */}
        {Math.abs(deltaNative) > 0.0001 && (
          <div className="pt-2 border-t border-sky-100 text-[11px] flex items-center justify-between text-zinc-700">
            <span>ผลกระทบต่อเงินสด:</span>
            <span className={`font-mono font-bold ${
              (isBuy && deltaNative > 0) || (!isBuy && deltaNative < 0)
                ? 'text-rose-600'
                : 'text-emerald-600'
            }`}>
              {isBuy ? (deltaNative > 0 ? '-' : '+') : (deltaNative > 0 ? '+' : '-')}
              {Math.abs(deltaNative).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })} {transaction.currency}
            </span>
          </div>
        )}
      </div>
    </FormModal>
  )
}
