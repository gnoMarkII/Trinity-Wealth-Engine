import { useState } from 'react'
import FormModal, { FormField, FormInput, FormSelect } from './FormModal'
import Modal from '../../ui/Modal'
import { EditIcon } from '../icons/PortfolioIcons'
import { api } from '../../../api/client'
import type { AllocationTargetDTO, ActualHoldingDTO, ActualPortfolioStateDTO } from '../../../api/types'
import { isPvdHolding, isValidIsoDate } from '../../../lib/pvdPresentation'

interface Props {
  portfolioId: string
  holding: ActualHoldingDTO
  targets: AllocationTargetDTO[]
  onClose: () => void
  onSuccess: (state: ActualPortfolioStateDTO) => void
}

export default function HoldingCorrectionModal({ portfolioId, holding, targets, onClose, onSuccess }: Props) {
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
    if (isPvdHolding(holding)) {
      setError('ยอด PVD อัปเดตจาก Statement จึงแก้ไขรายกองไม่ได้')
      return
    }
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
      }, portfolioId)
      onSuccess(state)
      onClose()
    } catch (err: any) {
      setError(err?.message || 'ปรับปรุงข้อมูล Holding ไม่สำเร็จ')
    } finally {
      setLoading(false)
    }
  }

  if (isPvdHolding(holding)) {
    const asOfDate = holding.pvd_valuation?.as_of_date
    return (
      <Modal titleId="correction-modal-title" onClose={onClose} panelClassName="max-w-md rounded-2xl border border-teal-100 bg-white p-6 shadow-2xl">
        <div className="flex items-start justify-between gap-4">
          <div>
            <h2 id="correction-modal-title" className="text-base font-bold text-zinc-900">ข้อมูล PVD แบบอ่านอย่างเดียว</h2>
            <p className="mt-1 font-mono text-sm text-teal-800">{holding.symbol}</p>
          </div>
          <button type="button" onClick={onClose} className="text-zinc-400 hover:text-zinc-700" aria-label="ปิดหน้าต่าง">✕</button>
        </div>
        <p className="mt-4 rounded-lg border border-teal-100 bg-teal-50 p-3 text-sm text-teal-900">
          ยอด PVD อัปเดตจาก Statement จึงแก้ไขรายกองไม่ได้
        </p>
        <dl className="mt-4 grid grid-cols-2 gap-3 text-sm">
          <dt className="text-zinc-500">หน่วย</dt><dd className="text-right font-mono">{holding.units.toLocaleString()}</dd>
          <dt className="text-zinc-500">มูลค่า</dt><dd className="text-right font-mono">{holding.market_value_thb.toLocaleString()}</dd>
          <dt className="text-zinc-500">วันที่ Statement</dt>
          <dd className="text-right font-mono">{isValidIsoDate(asOfDate) ? asOfDate : 'ไม่ทราบวันที่ Statement'}</dd>
        </dl>
        <div className="mt-5 flex justify-end">
          <button type="button" onClick={onClose} className="rounded-lg border border-zinc-300 px-4 py-2 text-sm font-semibold text-zinc-700 hover:bg-zinc-50">ปิด</button>
        </div>
      </Modal>
    )
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
            <option value="Fund">กองทุนรวม (Fund)</option>
            <option value="ETF">ETF</option>
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
