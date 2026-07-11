import { domAnimation, LazyMotion, m, MotionConfig, useReducedMotion } from 'motion/react'
import { useNavigate } from 'react-router-dom'
import RequireAuth from '../components/RequireAuth'
import FemaleHeroVisual from '../components/landing/FemaleHeroVisual'

function LandingContent() {
  const navigate = useNavigate()
  const shouldReduceMotion = useReducedMotion()

  return (
    <div className="relative min-h-screen overflow-hidden bg-[#08090d] text-white">
      <FemaleHeroVisual fullBleed />
      <div className="pointer-events-none absolute inset-0 bg-[linear-gradient(90deg,rgba(5,6,14,0.9)_0%,rgba(5,6,14,0.64)_35%,rgba(5,6,14,0.08)_75%),linear-gradient(0deg,rgba(5,6,14,0.72),transparent_44%)]" />
      <div className="pointer-events-none absolute inset-0 opacity-30 [background-image:linear-gradient(rgba(255,255,255,0.05)_1px,transparent_1px),linear-gradient(90deg,rgba(255,255,255,0.05)_1px,transparent_1px)] [background-size:5rem_5rem] [mask-image:radial-gradient(ellipse_at_70%_40%,black,transparent_74%)]" />

      <header className="relative mx-auto flex w-full max-w-7xl items-center px-6 py-6 lg:px-10">
        <div className="flex items-center gap-3">
          <span className="flex h-9 w-9 items-center justify-center rounded-xl border border-amber-200/40 bg-amber-300/10 text-sm font-bold text-amber-200">MR</span>
          <p className="text-sm font-semibold tracking-[0.18em] text-zinc-100">MONEY REROUTE</p>
        </div>
      </header>

      <main className="relative mx-auto flex min-h-[calc(100vh-5.25rem)] w-full max-w-7xl flex-col justify-center px-6 pb-12 lg:px-10">
        <section className="flex min-h-[calc(100vh-11rem)] items-center py-6">
          <m.div
            initial={shouldReduceMotion ? false : { opacity: 0, x: -24 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.7, ease: 'easeOut' }}
            className="relative z-10 max-w-xl"
          >
            <p className="text-xs font-bold uppercase tracking-[0.32em] text-amber-200">AI investment intelligence</p>
            <h1 className="mt-5 text-balance text-5xl font-semibold leading-[0.9] tracking-tight text-white sm:text-6xl lg:text-7xl">
              Know what<br />
              <span className="bg-gradient-to-r from-amber-200 via-orange-300 to-rose-300 bg-clip-text text-transparent">moves next.</span>
            </h1>
            <p className="mt-6 max-w-sm text-sm leading-relaxed text-zinc-400 sm:text-base">Signals distilled. Decisions retained.</p>
            <div className="mt-8 flex flex-wrap gap-3">
              <button
                onClick={() => navigate('/kanban')}
                className="rounded-xl bg-amber-300 px-5 py-3 text-sm font-bold text-zinc-950 shadow-[0_12px_36px_rgba(251,191,36,0.24)] transition-transform hover:-translate-y-0.5 hover:bg-amber-200 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-amber-200"
              >
                Agent Board
              </button>
              <button
                onClick={() => navigate('/macro')}
                className="rounded-xl border border-white/20 px-5 py-3 text-sm font-semibold text-zinc-100 transition-colors hover:border-white/50 hover:bg-white/10 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-amber-200"
              >
                Macro
              </button>
            </div>
          </m.div>

        </section>

        <div className="mt-4 flex flex-wrap items-center justify-between gap-3 border-t border-white/10 pt-5 text-[10px] font-semibold uppercase tracking-[0.2em] text-zinc-500 sm:text-xs">
          <span>Always-on research loop</span>
          <span>Human decision required</span>
          <span>Illustrative simulation only</span>
        </div>
      </main>
    </div>
  )
}

export default function Landing() {
  return (
    <RequireAuth>
      <LazyMotion features={domAnimation}>
        <MotionConfig reducedMotion="user">
          <LandingContent />
        </MotionConfig>
      </LazyMotion>
    </RequireAuth>
  )
}
