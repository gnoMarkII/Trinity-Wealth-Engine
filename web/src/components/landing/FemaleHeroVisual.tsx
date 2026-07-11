import { m, useReducedMotion } from 'motion/react'
import CandlestickStreamCanvas from './CandlestickStreamCanvas'

type FemaleHeroVisualProps = {
  fullBleed?: boolean
}

export default function FemaleHeroVisual({ fullBleed = false }: FemaleHeroVisualProps) {
  const shouldReduceMotion = useReducedMotion()
  const containerClassName = fullBleed
    ? 'absolute inset-0 h-full min-h-screen w-full max-w-none rounded-none border-0'
    : 'relative mx-auto aspect-[4/5] min-h-[31rem] w-full max-w-[48rem] overflow-hidden rounded-[2rem] border border-white/10 sm:aspect-[16/11] sm:min-h-[35rem] lg:aspect-[11/10]'

  return (
    <div className={`${containerClassName} isolate overflow-hidden bg-[#090b17] shadow-[0_42px_105px_rgba(0,0,0,0.52)]`}>
      <CandlestickStreamCanvas />
      <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(circle_at_75%_35%,rgba(245,158,11,0.18),transparent_23%),radial-gradient(circle_at_82%_74%,rgba(16,185,129,0.13),transparent_30%),linear-gradient(90deg,rgba(5,7,18,0.92)_0%,rgba(5,7,18,0.42)_47%,rgba(5,7,18,0.04)_100%)]" />
      <div className="pointer-events-none absolute inset-0 opacity-40 [background-image:linear-gradient(rgba(255,255,255,0.09)_1px,transparent_1px),linear-gradient(90deg,rgba(255,255,255,0.09)_1px,transparent_1px)] [background-size:3.5rem_3.5rem] [mask-image:linear-gradient(90deg,black,transparent_70%)]" />
      <m.img
        src="/landing/visor-analyst.png"
        alt=""
        className="absolute inset-0 h-full w-full object-cover object-[70%_center] opacity-90"
        animate={shouldReduceMotion ? { scale: 1 } : { scale: [1, 1.035, 1], x: [0, -5, 0] }}
        transition={{ duration: 12, repeat: Infinity, ease: 'easeInOut' }}
      />
      <div className="pointer-events-none absolute inset-0 bg-[linear-gradient(90deg,rgba(8,9,13,0.82),rgba(8,9,13,0.12)_58%,rgba(8,9,13,0.1)),linear-gradient(0deg,rgba(8,9,13,0.58),transparent_38%)]" />

      <m.div
        className="pointer-events-none absolute right-[17%] top-[18%] h-20 w-20 rounded-full border border-amber-100/30 bg-amber-200/10 blur-[1px]"
        animate={shouldReduceMotion ? { opacity: 0.65 } : { opacity: [0.22, 0.78, 0.22], scale: [0.8, 1.2, 0.8] }}
        transition={{ duration: 3.2, repeat: Infinity, ease: 'easeInOut' }}
      />
      <div className="absolute bottom-5 left-5 right-5 flex items-center justify-between border-t border-white/15 pt-3 text-[8px] font-semibold uppercase tracking-[0.2em] text-slate-300/70 sm:bottom-6 sm:left-7 sm:right-7">
        <span>Signal stream</span>
        <span className="text-emerald-300">Model active</span>
      </div>
    </div>
  )
}
