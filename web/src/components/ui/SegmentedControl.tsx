import { useId, type ReactNode } from 'react'
import { motion } from 'motion/react'

interface Option {
  key: string
  label: ReactNode
}

interface Props {
  options: Option[]
  value: string
  onChange: (value: string) => void
  /** id ของ element ที่เป็นชื่อกลุ่ม (แทน <label> ซึ่งใช้ไม่ได้เพราะปุ่มไม่ใช่ form control) */
  ariaLabelledby?: string
  wrap?: boolean
}

export default function SegmentedControl({
  options,
  value,
  onChange,
  ariaLabelledby,
  wrap = true,
}: Props) {
  const layoutId = useId()

  return (
    <div
      role="group"
      aria-labelledby={ariaLabelledby}
      className={`flex ${
        wrap ? 'flex-wrap' : 'overflow-x-auto custom-scrollbar flex-nowrap'
      } gap-1 sm:gap-1.5 rounded-2xl border border-sky-200/75 bg-surface-1 p-1 sm:p-1.5 shadow-sm backdrop-blur-md max-w-full`}
    >
      {options.map((opt) => {
        const isActive = value === opt.key
        return (
          <button
            key={opt.key}
            type="button"
            onClick={() => onChange(opt.key)}
            aria-pressed={isActive}
            className={`relative rounded-xl px-3 sm:px-4 py-1.5 sm:py-2 text-sm font-semibold transition-colors duration-150 flex items-center gap-1.5 shrink-0 whitespace-nowrap focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-section-accent/45 ${
              isActive
                ? 'text-white font-bold'
                : 'text-slate-600 hover:bg-sky-50/75 hover:text-slate-900'
            }`}
          >
            {isActive && (
              <motion.span
                layoutId={`seg-pill-${layoutId}`}
                transition={{ type: 'spring', stiffness: 420, damping: 32 }}
                aria-hidden="true"
                className="absolute inset-0 rounded-xl bg-gradient-to-r from-section-accent to-section-accent-2 shadow-sm"
              />
            )}
            <span className="relative z-10 flex items-center gap-1.5">{opt.label}</span>
          </button>
        )
      })}
    </div>
  )
}
