import React from 'react'

export interface GaugeZone {
  from: number
  to: number
  color: string
  label: string
}

export interface GaugeSubItem {
  label: string
  value: number
  unit?: string
}

export interface RadialGaugeProps {
  value: number
  min?: number
  max?: number
  title: string
  subtitle?: string
  unit?: string
  zones?: GaugeZone[]
  subItems?: GaugeSubItem[]
  asOfDate?: string
  dataLagNote?: string
  className?: string
}

const DEFAULT_ZONES: GaugeZone[] = [
  { from: -3.0, to: 0.0, color: '#10b981', label: 'Below Avg Stress' },
  { from: 0.0, to: 1.0, color: '#f59e0b', label: 'Moderate Stress' },
  { from: 1.0, to: 3.0, color: '#ef4444', label: 'High Stress' },
]

export const RadialGauge: React.FC<RadialGaugeProps> = ({
  value,
  min = -3.0,
  max = 3.0,
  title,
  subtitle,
  unit = '',
  zones = DEFAULT_ZONES,
  subItems,
  asOfDate,
  dataLagNote,
  className = '',
}) => {
  // Clamp value with fallback to 0 if undefined or NaN
  const numVal = typeof value === 'number' && !isNaN(value) ? value : 0
  const clampedVal = Math.max(min, Math.min(max, numVal))
  const range = max - min
  const pct = range > 0 ? (clampedVal - min) / range : 0.5

  // Semicircle arc angles: 180 deg (start) to 360 deg (end)
  // Angle for needle: -90 deg (min) to +90 deg (max)
  const needleAngle = -90 + pct * 180

  // Active zone
  const fallbackZone: GaugeZone = DEFAULT_ZONES[0] ?? { from: -3, to: 0, color: '#10b981', label: 'Normal' }
  const activeZone = zones.find((z) => clampedVal >= z.from && clampedVal <= z.to) || zones[zones.length - 1] || fallbackZone
  const zoneColor = activeZone.color || '#0ea5e9'

  // SVG Geometry for semi-circle
  const radius = 80
  const cx = 100
  const cy = 95
  const strokeWidth = 14
  // Circumference of semi-circle = PI * radius
  const circumference = Math.PI * radius
  const strokeDashoffset = circumference * (1 - pct)

  return (
    <div
      className={`rounded-2xl border border-sky-100 bg-white/80 p-5 shadow-[0_8px_25px_rgba(14,165,233,0.06)] backdrop-blur-md transition-all hover:shadow-[0_12px_32px_rgba(14,165,233,0.1)] ${className}`}
    >
      <div className="flex items-center justify-between border-b border-sky-100/60 pb-3">
        <div>
          <h3 className="font-semibold text-zinc-900 tracking-tight">{title}</h3>
          {subtitle && <p className="text-xs text-zinc-500">{subtitle}</p>}
        </div>
        {asOfDate && (
          <div className="text-right">
            <span className="inline-block rounded-full bg-sky-50 px-2.5 py-0.5 font-mono text-[11px] font-medium text-sky-700">
              {asOfDate}
            </span>
            {dataLagNote && (
              <span className="block text-[10px] text-zinc-400 mt-0.5">{dataLagNote}</span>
            )}
          </div>
        )}
      </div>

      <div className="relative mt-2 flex flex-col items-center">
        <svg viewBox="0 0 200 115" className="w-56 h-auto drop-shadow-sm">
          {/* Background Track */}
          <path
            d="M 20 95 A 80 80 0 0 1 180 95"
            fill="none"
            stroke="#f1f5f9"
            strokeWidth={strokeWidth}
            strokeLinecap="round"
          />

          {/* Active Colored Arc */}
          <path
            d="M 20 95 A 80 80 0 0 1 180 95"
            fill="none"
            stroke={zoneColor}
            strokeWidth={strokeWidth}
            strokeLinecap="round"
            strokeDasharray={circumference}
            strokeDashoffset={strokeDashoffset}
            className="transition-all duration-700 ease-out"
          />

          {/* Center Pivot & Needle */}
          <g transform={`translate(${cx}, ${cy}) rotate(${needleAngle})`}>
            <polygon points="-3,0 3,0 0,-70" fill="#334155" />
            <circle cx="0" cy="0" r="6" fill="#0f172a" />
            <circle cx="0" cy="0" r="2.5" fill="#ffffff" />
          </g>

          {/* Min and Max scale markers */}
          <text x="22" y="112" fill="#94a3b8" fontSize="10" fontWeight="500" textAnchor="start">
            {min}
          </text>
          <text x="178" y="112" fill="#94a3b8" fontSize="10" fontWeight="500" textAnchor="end">
            +{max}
          </text>
        </svg>

        {/* Central Metric Value */}
        <div className="mt-1 flex flex-col items-center">
          <div className="flex items-baseline gap-1">
            <span
              className="font-mono text-3xl font-extrabold tracking-tight"
              style={{ color: zoneColor }}
            >
              {typeof value === 'number' && !isNaN(value)
                ? (value > 0 ? `+${value.toFixed(2)}` : value.toFixed(2))
                : '—'}
            </span>
            {unit && <span className="font-mono text-xs text-zinc-500 font-semibold">{unit}</span>}
          </div>
          <span
            className="mt-1 rounded-full px-2.5 py-0.5 text-xs font-semibold"
            style={{
              backgroundColor: `${zoneColor}18`,
              color: zoneColor,
            }}
          >
            {activeZone.label}
          </span>
        </div>
      </div>

      {/* Sub-Metric Breakdown (e.g. OFR 5 categories) */}
      {subItems && subItems.length > 0 && (
        <div className="mt-4 border-t border-sky-100/60 pt-3">
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
            {subItems.map((item) => {
              const itemVal = typeof item.value === 'number' && !isNaN(item.value) ? item.value : 0
              const isPositive = itemVal > 0
              return (
                <div
                  key={item.label}
                  className="rounded-lg bg-slate-50/80 p-2 border border-slate-100/80"
                >
                  <p className="truncate text-[11px] font-medium text-zinc-500">{item.label}</p>
                  <p
                    className={`font-mono text-xs font-bold mt-0.5 ${
                      isPositive ? 'text-amber-600' : 'text-emerald-600'
                    }`}
                  >
                    {isPositive ? `+${itemVal.toFixed(2)}` : itemVal.toFixed(2)}
                    {item.unit || ''}
                  </p>
                </div>
              )
            })}
          </div>
        </div>
      )}
    </div>
  )
}
