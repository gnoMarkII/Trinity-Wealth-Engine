import { AnimatePresence, m, useReducedMotion } from 'motion/react'
import { useLoopingIndex } from './useLoopingIndex'

const SIMULATION_VALUES = [100_000, 102_400, 105_200, 108_700, 112_900, 116_300]
const CHART_POINTS = [
  [18, 160],
  [100, 151],
  [180, 139],
  [260, 121],
  [340, 92],
  [422, 52],
]

const CURVE_PATH = `M ${CHART_POINTS.map(([x, y]) => `${x} ${y}`).join(' L ')}`

function formatCurrency(value: number): string {
  return new Intl.NumberFormat('th-TH', { style: 'currency', currency: 'THB', maximumFractionDigits: 0 }).format(value)
}

export default function CompoundingSimulation() {
  const activeIndex = useLoopingIndex(SIMULATION_VALUES.length, 1_350)
  const shouldReduceMotion = useReducedMotion()
  const value = SIMULATION_VALUES[activeIndex]
  const point = CHART_POINTS[activeIndex]

  return (
    <section className="rounded-3xl border border-zinc-200 bg-white p-5 shadow-sm shadow-black/5 sm:p-7">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <p className="text-xs font-bold uppercase tracking-[0.2em] text-terra">Compounding discipline</p>
          <h2 className="mt-1 text-xl font-semibold text-zinc-900">Build, validate, repeat</h2>
        </div>
        <span className="rounded-full border border-amber-200 bg-amber-50 px-3 py-1 text-xs font-semibold text-amber-800">Illustrative simulation</span>
      </div>

      <div className="mt-5 grid gap-5 sm:grid-cols-[0.9fr_1.1fr] sm:items-center">
        <div>
          <p className="text-sm text-zinc-500">Simulated portfolio value</p>
          <AnimatePresence mode="wait">
            <m.p
              key={value}
              initial={shouldReduceMotion ? false : { opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={shouldReduceMotion ? undefined : { opacity: 0, y: -10 }}
              transition={{ duration: 0.28 }}
              className="mt-1 font-mono text-3xl font-bold tracking-tight text-zinc-900 sm:text-4xl"
            >
              {formatCurrency(value)}
            </m.p>
          </AnimatePresence>
          <p className="mt-3 text-xs leading-relaxed text-zinc-500">การจำลองแสดงผลของการทำตามกระบวนการอย่างต่อเนื่อง ไม่ใช่ผลตอบแทนจริงหรือการรับประกันผลกำไร</p>
        </div>

        <div className="rounded-2xl border border-zinc-200 bg-zinc-50 p-3">
          <svg viewBox="0 0 440 180" className="h-44 w-full" role="img" aria-label="กราฟจำลองการเติบโตแบบทบต้น">
            {[46, 92, 138].map((y) => <line key={y} x1="18" x2="422" y1={y} y2={y} stroke="#e4e4e7" strokeDasharray="4 6" />)}
            <m.path
              d={CURVE_PATH}
              fill="none"
              stroke="#924a2e"
              strokeWidth="4"
              strokeLinecap="round"
              strokeLinejoin="round"
              initial={shouldReduceMotion ? false : { pathLength: 0 }}
              animate={{ pathLength: 1 }}
              transition={shouldReduceMotion ? { duration: 0 } : { duration: 6.8, ease: 'easeInOut', repeat: Infinity, repeatDelay: 0.6 }}
            />
            <m.circle
              cx={point[0]}
              cy={point[1]}
              r="7"
              fill="#f59e0b"
              stroke="#ffffff"
              strokeWidth="3"
              animate={shouldReduceMotion ? { opacity: 1 } : { scale: [1, 1.35, 1] }}
              transition={{ duration: 1.1, repeat: Infinity, ease: 'easeInOut' }}
            />
          </svg>
          <div className="flex justify-between px-1 text-[10px] font-medium uppercase tracking-wider text-zinc-400">
            <span>Cycle start</span>
            <span>Review & repeat</span>
          </div>
        </div>
      </div>
    </section>
  )
}
