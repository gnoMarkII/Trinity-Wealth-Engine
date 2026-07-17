import { useState } from 'react'
import FormModal, { FormField, FormInput, FormSelect } from './FormModal'
import { EditIcon } from '../icons/PortfolioIcons'
import { api } from '../../../api/client'
import type { AllocationTargetDTO, ActualHoldingDTO, ActualPortfolioStateDTO } from '../../../api/types'

interface Props {
  holding: ActualHoldingDTO
  targets: AllocationTargetDTO[]
  onClose: () => void
  onSuccess: (state: ActualPortfolioStateDTO) => void
}

export default function HoldingCorrectionModal({ holding, targets, onClose, onSuccess }: Props) {
  const [units, setUnits] = useState<string>(holding.units.toString())
  const [avgCost, setAvgCost] = useState<string>((holding.avg_cost_thb ?? holding.avg_cost_usd ?? 0).toString())
  const [accumulatedDividendThb, setAccumulatedDividendThb] = useState<string>(
    holding.accumulated_dividend_thb !== null && holding.accumulated_dividend_thb !== undefined
      ? holding.accumulated_dividend_thb.toString()
      : ''
  )
  const [assetType, setAssetType] = useState<string>(holding.asset_type || 'Stock')
  const [bucketId, setBucketId] = useState<string>(holding.bucket_id || '')
  const [reason, setReason] = useState<string>('ปรับปรุงข้อมูลพอร์ต (Correction)')

  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setLoading(true)
    setError(null)
    try {
      const state = await api.editHolding(holding.symbol, {
        units: units !== '' ? parseFloat(units) : null,
        avg_cost: avgCost !== '' ? parseFloat(avgCost) : null,
        accumulated_dividend_thb: accumulatedDividendThb !== '' ? parseFloat(accumulatedDividendThb) : null,
        asset_type: assetType || null,
        bucket_id: bucketId || null,
        reason: reason.trim(),
      })
      onSuccess(state)
      onClose()
    } catch (err: any) {
      setError(err?.message || 'ปรับปรุงข้อมูล Holding ไม่สำเร็จ')
    } finally {
      setLoading(false)
    }
  }

  return (
    <FormModal
      titleId="correction-modal-title"
      title={
        <>
          <span>แก้ไขข้อมูลรายตัว: </span>
          <span className="text-flow-blue font-mono">{holding.symbol}</span>
        </>
      }
      icon={<EditIcon className="w-5 h-5 text-flow-blue" />}
      onClose={onClose}
      onSubmit={handleSubmit}
      error={error}
      loading={loading}
      submitText="💾 บันทึกการปรับปรุง"
      submitClassName="bg-flow-blue hover:bg-sky-600"
      panelClassName="max-w-md rounded-2xl border border-sky-100 bg-white p-6 shadow-2xl"
    >
      <div className="grid grid-cols-2 gap-3">
        <FormField label="จำนวนหน่วย (Units)" required>
          <FormInput
            type="number"
            step="any"
            min="0"
            value={units}
            onChange={(e) => setUnits(e.target.value)}
            className="font-bold"
            required
          />
        </FormField>

        <FormField
          label={`ต้นทุนเฉลี่ย (Avg Cost ${holding.asset_type === 'US_STOCK' ? 'USD' : 'THB'})`}
          required
        >
          <FormInput
            type="number"
            step="any"
            min="0"
            value={avgCost}
            onChange={(e) => setAvgCost(e.target.value)}
            className="font-bold"
            required
          />
        </FormField>
      </div>

      <div className="grid grid-cols-2 gap-3">
        <FormField label="ประเภทสินทรัพย์">
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

        <FormField label="กลุ่มกลยุทธ์ (Bucket)">
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

      <FormField label="เงินปันผลสะสม (Accumulated Dividend THB)">
        <FormInput
          type="number"
          step="any"
          value={accumulatedDividendThb}
          onChange={(e) => setAccumulatedDividendThb(e.target.value)}
          className="font-mono"
        />
      </FormField>

      <FormField label="เหตุผลในการแก้ไข (Audit Reason)" required>
        <FormInput
          type="text"
          value={reason}
          onChange={(e) => setReason(e.target.value)}
          required
        />
      </FormField>
    </FormModal>
  )
}
