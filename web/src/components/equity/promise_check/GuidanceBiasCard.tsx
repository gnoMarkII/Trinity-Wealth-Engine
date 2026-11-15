import React from 'react'
import type { GuidanceBiasDTO } from '../../../api/types'

interface GuidanceBiasCardProps {
  bias: GuidanceBiasDTO
  milestoneSlipQuarters?: number | null
  milestoneSlipN: number
}

export const GuidanceBiasCard: React.FC<GuidanceBiasCardProps> = ({
  bias,
  milestoneSlipQuarters,
  milestoneSlipN,
}) => {
  const biasConfigs: Record<string, { icon: string; titleTh: string; color: string; bg: string; border: string }> = {
    Sandbagger: {
      icon: '🛡️',
      titleTh: 'Sandbagger (ให้ Guidance ต่ำและชนะเป้าสม่ำเสมอ)',
  color:'text-emerald-700',
      bg: 'bg-emerald-500/10',
      border: 'border-emerald-500/30',
    },
    'Straight shooter': {
      icon: '🎯',
      titleTh: 'Straight Shooter (ตัวเลขตรงไปตรงมา ใกล้เคียงเป้าหมาย)',
  color:'text-sky-700',
      bg: 'bg-sky-500/10',
      border: 'border-sky-500/30',
    },
    Overpromiser: {
      icon: '⚠️',
      titleTh: 'Overpromiser (มักให้เป้าหมายสูงเกินจริง พลาดเป้าบ่อย)',
  color:'text-rose-700',
      bg: 'bg-rose-500/10',
      border: 'border-rose-500/30',
    },
    Mixed: {
      icon: '⚖️',
      titleTh: 'Mixed (มีทั้งชนะและพลาดเป้าปะปนกัน)',
  color:'text-amber-700',
      bg: 'bg-amber-500/10',
      border: 'border-amber-500/30',
    },
  }

  const config = biasConfigs[bias.label] || {
    icon: '📊',
    titleTh: bias.label,
  color:'text-slate-700',
    bg: 'bg-slate-500/10',
    border: 'border-slate-500/30',
  }

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
      {/* Guidance Bias Analysis */}
      <div className={`rounded-2xl border ${config.border} ${config.bg} p-5 space-y-3`}>
        <div className="flex items-center gap-2">
          <span className="text-2xl">{config.icon}</span>
          <div>
  <span className="text-[11px] font-semibold text-slate-500 uppercase tracking-wider block">
              ลักษณะนิสัยการให้ Guidance (Management Bias)
            </span>
            <h4 className={`text-sm font-bold ${config.color}`}>
              {config.titleTh}
            </h4>
          </div>
        </div>

  <p className="text-xs text-slate-600 leading-relaxed">
          {bias.message}
        </p>

        {bias.n >= bias.min_n && (
          <div className="grid grid-cols-2 gap-2 pt-2 border-t border-edge/40">
            <div className="p-2.5 rounded-lg bg-surface/70 border border-edge/40 text-center">
              <span className="text-[11px] text-slate-500 block">Median Delta เทียบเป้า</span>
              <span className={`text-base font-bold ${
  (bias.median_delta_pct ?? 0) > 0 ?'text-emerald-600':
  (bias.median_delta_pct ?? 0) < 0 ?'text-rose-600':'text-slate-700'
              }`}>
                {(bias.median_delta_pct ?? 0) > 0 ? '+' : ''}
                {bias.median_delta_pct != null ? `${bias.median_delta_pct.toFixed(2)}%` : 'N/A'}
              </span>
            </div>

            <div className="p-2.5 rounded-lg bg-surface/70 border border-edge/40 text-center">
              <span className="text-[11px] text-slate-500 block">อัตราการชนะเป้า (Beat Rate)</span>
  <span className="text-base font-bold text-slate-900">
                {bias.beat_rate != null ? `${bias.beat_rate.toFixed(1)}%` : 'N/A'}
              </span>
            </div>
          </div>
        )}
      </div>

      {/* Milestone Delivery & Timeline Slippage */}
      <div className="rounded-2xl border border-edge bg-surface p-5 space-y-3 flex flex-col justify-between">
        <div>
          <div className="flex items-center gap-2 mb-2">
            <span className="text-xl">⏱️</span>
            <div>
  <span className="text-[11px] font-semibold text-slate-500 uppercase tracking-wider block">
                วินัยการส่งมอบตามกำหนดเวลา (Timeline Execution)
              </span>
  <h4 className="text-sm font-bold text-slate-900">
                Milestone Slippage & Delays
              </h4>
            </div>
          </div>
  <p className="text-xs text-slate-600 leading-relaxed">
            วัดอัตราความล่าช้าในการส่งมอบโปรเจกต์เชิงกลยุทธ์, การเปิดตัวสินค้า, การขยายกำลังการผลิต และการขออนุมัติตามกฎหมาย
          </p>
        </div>

        <div className="grid grid-cols-2 gap-2 pt-2 border-t border-edge/40">
          <div className="p-2.5 rounded-lg bg-surface-muted/40 border border-edge/40 text-center">
            <span className="text-[11px] text-slate-500 block">ความล่าช้าเฉลี่ย</span>
            <span className={`text-base font-bold ${
              milestoneSlipQuarters == null
                ? 'text-slate-500'
                : milestoneSlipQuarters > 0
  ?'text-amber-600'
  :'text-emerald-600'
            }`}>
              {milestoneSlipQuarters != null
                ? (milestoneSlipQuarters > 0 ? `+${milestoneSlipQuarters.toFixed(1)} ไตรมาส` : '0 ไตรมาส (ตรงเวลา)')
                : 'ยังไม่มีข้อมูล (< 3 โครงการ)'}
            </span>
          </div>

          <div className="p-2.5 rounded-lg bg-surface-muted/40 border border-edge/40 text-center">
            <span className="text-[11px] text-slate-500 block">Milestones ที่ติดตาม</span>
  <span className="text-base font-bold text-slate-900">
              {milestoneSlipN} โครงการ
            </span>
          </div>
        </div>
      </div>
    </div>
  )
}
