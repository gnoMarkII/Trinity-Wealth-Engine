import React from 'react'

export type TrafficStatus = 'positive' | 'warning' | 'danger' | 'neutral' | 'info'

export interface TrafficLightSubtitleProps {
  status: TrafficStatus
  text: string
  icon?: string
  className?: string
}

const STATUS_CONFIG: Record<TrafficStatus, { bg: string; border: string; text: string; defaultIcon: string }> = {
  positive: {
    bg: 'bg-emerald-50/80',
    border: 'border-emerald-200/80',
    text: 'text-emerald-900',
    defaultIcon: '🟢',
  },
  warning: {
    bg: 'bg-amber-50/80',
    border: 'border-amber-200/80',
    text: 'text-amber-900',
    defaultIcon: '⚠️',
  },
  danger: {
    bg: 'bg-rose-50/80',
    border: 'border-rose-200/80',
    text: 'text-rose-900',
    defaultIcon: '🚨',
  },
  neutral: {
    bg: 'bg-zinc-50/80',
    border: 'border-zinc-200/80',
    text: 'text-zinc-800',
    defaultIcon: '⚪',
  },
  info: {
    bg: 'bg-sky-50/80',
    border: 'border-sky-200/80',
    text: 'text-sky-900',
    defaultIcon: '💡',
  },
}

export const TrafficLightSubtitle: React.FC<TrafficLightSubtitleProps> = ({
  status,
  text,
  icon,
  className = '',
}) => {
  const cfg = STATUS_CONFIG[status]
  const displayIcon = icon || cfg.defaultIcon

  return (
    <div
      className={`inline-flex items-center gap-1.5 rounded-md border px-2 py-0.5 text-[11px] font-medium leading-tight ${cfg.bg} ${cfg.border} ${cfg.text} ${className}`}
    >
      <span className="shrink-0 text-[10px]">{displayIcon}</span>
      <span>{text}</span>
    </div>
  )
}

