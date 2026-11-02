import React, { useState, useRef, useEffect } from 'react'

export interface ContextTooltipProps {
  title: string
  whatIsIt?: string
  howToRead?: string
  impactOnMoney?: string
  what?: string
  normalBenchmark?: string
  decisionImpact?: string
  children?: React.ReactNode
  align?: 'left' | 'right' | 'center'
}

export const ContextTooltip: React.FC<ContextTooltipProps> = ({
  title,
  whatIsIt,
  howToRead,
  impactOnMoney,
  what,
  normalBenchmark,
  decisionImpact,
  children,
  align = 'right',
}) => {
  const [isOpen, setIsOpen] = useState(false)
  const containerRef = useRef<HTMLDivElement>(null)

  const actualWhat = whatIsIt || what || ''
  const actualHow = howToRead || normalBenchmark || ''
  const actualImpact = impactOnMoney || decisionImpact || ''

  // Close on outside click
  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (containerRef.current && !containerRef.current.contains(event.target as Node)) {
        setIsOpen(false)
      }
    }
    if (isOpen) {
      document.addEventListener('mousedown', handleClickOutside)
    }
    return () => {
      document.removeEventListener('mousedown', handleClickOutside)
    }
  }, [isOpen])

  const alignmentClass =
    align === 'left'
      ? 'left-0'
      : align === 'center'
      ? 'left-1/2 -translate-x-1/2'
      : 'right-0'

  return (
    <div className="relative inline-flex items-center" ref={containerRef}>
      <button
        type="button"
        onClick={() => setIsOpen((prev) => !prev)}
        className="inline-flex items-center justify-center rounded-full p-0.5 text-zinc-400 hover:text-sky-600 hover:bg-sky-50 transition-colors focus:outline-hidden"
        title={`คลิกเพื่อดูคำอธิบาย: ${title}`}
        aria-label={`คำอธิบาย: ${title}`}
      >
        {children || (
          <span className="flex h-4 w-4 items-center justify-center rounded-full border border-zinc-300 text-[10px] font-bold text-zinc-500 hover:border-sky-400 hover:text-sky-600">
            ?
          </span>
        )}
      </button>

      {isOpen && (
        <div
          role="tooltip"
          className={`absolute z-50 top-full mt-2 w-72 sm:w-80 rounded-xl border border-sky-100 bg-white p-3.5 shadow-xl text-xs text-zinc-700 animate-in fade-in zoom-in-95 duration-150 ${alignmentClass}`}
        >
          <div className="flex items-center justify-between pb-2 border-b border-zinc-100">
            <span className="font-bold text-zinc-900 text-xs flex items-center gap-1.5">
              <span>💡</span>
              <span>{title}</span>
            </span>
            <button
              type="button"
              onClick={() => setIsOpen(false)}
              className="text-zinc-400 hover:text-zinc-600 text-xs px-1"
            >
              ✕
            </button>
          </div>

          <div className="mt-2.5 space-y-2 text-[11px] leading-relaxed">
            <div className="bg-sky-50/50 rounded-lg p-2 border border-sky-100/60">
              <span className="font-bold text-sky-900 block mb-0.5">📌 มันคืออะไร:</span>
              <span className="text-zinc-700">{actualWhat}</span>
            </div>

            <div className="bg-amber-50/50 rounded-lg p-2 border border-amber-100/60">
              <span className="font-bold text-amber-900 block mb-0.5">📊 วิธีดู (สูง/ต่ำ/ปกติ):</span>
              <span className="text-zinc-700">{actualHow}</span>
            </div>

            <div className="bg-emerald-50/50 rounded-lg p-2 border border-emerald-100/60">
              <span className="font-bold text-emerald-900 block mb-0.5">💰 กระทบเงินในกระเป๋าเรายังไง:</span>
              <span className="text-zinc-700">{actualImpact}</span>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
