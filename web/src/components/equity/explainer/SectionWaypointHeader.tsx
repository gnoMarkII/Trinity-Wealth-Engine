import React from 'react'

export interface SectionWaypointHeaderProps {
  tierNumber: number
  icon: string
  title: string
  plainLanguageSummary: string
  frameworkLabel: string
  id?: string
}

export const SectionWaypointHeader: React.FC<SectionWaypointHeaderProps> = ({
  tierNumber,
  icon,
  title,
  plainLanguageSummary,
  frameworkLabel,
  id,
}) => {
  return (
    <div
      id={id}
      className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 pb-2.5 border-b border-edge/70 pt-4 first:pt-0"
    >
      <div className="flex items-start gap-3">
        <span className="flex items-center justify-center w-8 h-8 rounded-xl bg-sky-500/10 text-sky-800 border border-sky-300/60 text-base shrink-0 shadow-2xs mt-0.5">
          {icon}
        </span>
        <div>
          <div className="flex items-center gap-2">
            <span className="text-[10px] font-mono font-extrabold uppercase px-1.5 py-0.2 rounded bg-sky-100 text-sky-900 border border-sky-200">
              Tier {tierNumber}
            </span>
            <h3 className="text-sm font-extrabold text-zinc-900 tracking-tight">
              {title}
            </h3>
          </div>
          <p className="text-xs text-zinc-500 mt-0.5 leading-relaxed">
            {plainLanguageSummary}
          </p>
        </div>
      </div>
      <span className="self-start sm:self-center font-mono text-[10px] font-bold text-zinc-500 bg-surface px-2.5 py-1 rounded-full border border-edge uppercase tracking-wider shrink-0 shadow-3xs">
        {frameworkLabel}
      </span>
    </div>
  )
}

export default SectionWaypointHeader

