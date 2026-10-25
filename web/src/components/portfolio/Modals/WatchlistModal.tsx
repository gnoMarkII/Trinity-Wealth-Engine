import { useState } from 'react'
import FormModal, { FormField, FormInput, FormSelect, FormTextarea } from './FormModal'
import { WatchlistIcon } from '../icons/PortfolioIcons'
import { api } from '../../../api/client'
import type { ActualWatchlistItemDTO, ActualWatchlistStateDTO } from '../../../api/types'

export interface WatchlistDraft {
  symbol?: string
  asset_type?: string
  target_price?: number | null
  notes?: string | null
  suggested_bucket_id?: string | null
  applied_bucket_name?: string | null
  target_currency?: string | null
  evaluation_ref?: string | null
  source_refs?: string[] | null
  target_price_basis?: string | null
}

interface Props {
  portfolioId: string
  initialItem?: ActualWatchlistItemDTO | null
  initialDraft?: WatchlistDraft | null
  onClose: () => void
  onSuccess: (state: ActualWatchlistStateDTO) => void
}

export default function WatchlistModal({ portfolioId, initialItem, initialDraft, onClose, onSuccess }: Props) {
  const isEdit = !!initialItem
  const [symbol, setSymbol] = useState(initialItem?.symbol || initialDraft?.symbol || '')
  const [assetType, setAssetType] = useState(initialItem?.asset_type || initialDraft?.asset_type || 'Stock')
  const [targetPrice, setTargetPrice] = useState<string>(
    initialItem?.target_price !== null && initialItem?.target_price !== undefined
      ? initialItem.target_price.toString()
      : initialDraft?.target_price !== null && initialDraft?.target_price !== undefined
      ? initialDraft.target_price.toString()
      : ''
  )
  const [notes, setNotes] = useState(initialItem?.notes || initialDraft?.notes || '')
  const [suggestedBucketId, setSuggestedBucketId] = useState<string>(
    initialItem?.suggested_bucket_id || initialDraft?.suggested_bucket_id || ''
  )

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
        suggested_bucket_id: suggestedBucketId.trim() || null,
        applied_bucket_name: initialDraft?.applied_bucket_name || initialItem?.applied_bucket_name || undefined,
        target_currency: initialDraft?.target_currency || initialItem?.target_currency || undefined,
        evaluation_ref: initialDraft?.evaluation_ref || initialItem?.evaluation_ref || undefined,
        source_refs: initialDraft?.source_refs || initialItem?.source_refs || undefined,
        target_price_basis: initialDraft?.target_price_basis || initialItem?.target_price_basis || undefined,
      }, portfolioId)
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

      <div className="grid grid-cols-2 gap-3">
        <FormField label="ราคาเป้าหมาย (Target Price - ว่างได้)">
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

        <FormField label="Strategy Bucket แนะนำ / กำหนด">
          <FormInput
            type="text"
            value={suggestedBucketId}
            onChange={(e) => setSuggestedBucketId(e.target.value)}
            placeholder="e.g. CORE, DIV"
            className="font-mono uppercase text-flow-blue font-bold"
          />
        </FormField>
      </div>

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
