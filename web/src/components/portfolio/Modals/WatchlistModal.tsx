import { useState } from 'react'
import FormModal, { FormField, FormInput, FormSelect, FormTextarea } from './FormModal'
import { WatchlistIcon } from '../icons/PortfolioIcons'
import { api } from '../../../api/client'
import type { ActualWatchlistItemDTO, ActualWatchlistStateDTO } from '../../../api/types'

interface Props {
  initialItem?: ActualWatchlistItemDTO | null
  onClose: () => void
  onSuccess: (state: ActualWatchlistStateDTO) => void
}

export default function WatchlistModal({ initialItem, onClose, onSuccess }: Props) {
  const isEdit = !!initialItem
  const [symbol, setSymbol] = useState(initialItem?.symbol || '')
  const [assetType, setAssetType] = useState(initialItem?.asset_type || 'Stock')
  const [targetPrice, setTargetPrice] = useState<string>(
    initialItem?.target_price !== null && initialItem?.target_price !== undefined
      ? initialItem.target_price.toString()
      : ''
  )
  const [notes, setNotes] = useState(initialItem?.notes || '')

  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!symbol.trim()) {
      setError('กรุณาระบุ Symbol / Ticker')
      return
    }
    setLoading(true)
    setError(null)
    try {
      const state = await api.upsertWatchlistItem(symbol.trim().toUpperCase(), {
        asset_type: assetType,
        target_price: targetPrice !== '' ? parseFloat(targetPrice) : null,
        notes: notes.trim(),
      })
      onSuccess(state)
      onClose()
    } catch (err: any) {
      setError(err?.message || 'บันทึก Watchlist ไม่สำเร็จ')
    } finally {
      setLoading(false)
    }
  }

  return (
    <FormModal
      titleId="watchlist-modal-title"
      title={isEdit ? `แก้ไข Watchlist: ${initialItem?.symbol}` : 'เพิ่มสินทรัพย์เฝ้าระวัง (Add to Watchlist)'}
      icon={<WatchlistIcon className="w-5 h-5 text-flow-blue" />}
      onClose={onClose}
      onSubmit={handleSubmit}
      error={error}
      loading={loading}
      submitText={isEdit ? '💾 บันทึกการแก้ไข' : '+ เพิ่มเข้า Watchlist'}
      submitClassName="bg-flow-blue hover:bg-sky-600"
      panelClassName="max-w-md rounded-2xl border border-sky-100 bg-white p-6 shadow-2xl"
    >
      <div className="grid grid-cols-2 gap-3">
        <FormField label="สัญลักษณ์ (Ticker Symbol)" required>
          <FormInput
            type="text"
            value={symbol}
            onChange={(e) => setSymbol(e.target.value)}
            disabled={isEdit}
            placeholder="e.g. NVDA, PTT"
            className="font-mono font-bold uppercase"
            required
          />
        </FormField>

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
      </div>

      <FormField label="ราคาเป้าหมายที่อยากซื้อ (Target Price - ว่างไว้ได้)">
        <FormInput
          type="number"
          step="any"
          min="0"
          value={targetPrice}
          onChange={(e) => setTargetPrice(e.target.value)}
          placeholder="0.00"
          className="font-mono"
        />
      </FormField>

      <FormField label="หมายเหตุ / เหตุผลที่เฝ้ามอง (Notes)">
        <FormTextarea
          rows={2}
          value={notes}
          onChange={(e) => setNotes(e.target.value)}
          placeholder="เช่น รอพักตัวที่เส้น 200 วัน หรือรอดูงบ Q3..."
        />
      </FormField>
    </FormModal>
  )
}
