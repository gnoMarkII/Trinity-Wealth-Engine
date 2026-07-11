import { m, useReducedMotion } from 'motion/react'
import { usePageVisibility } from './usePageVisibility'

const ORBIT_AGENTS = [
  { x: -170, y: -96, z: -96, color: '#fbbf24', delay: 0.1, tilt: -7 },
  { x: 160, y: -118, z: -74, color: '#7dd3fc', delay: 0.6, tilt: 8 },
  { x: 184, y: 86, z: -104, color: '#f9a8d4', delay: 1.1, tilt: -5 },
  { x: -146, y: 136, z: -82, color: '#6ee7b7', delay: 1.6, tilt: 6 },
]

const COMPOUND_CANDLES = Array.from({ length: 12 }, (_, index) => ({
  x: -132 + ((index * 46) % 252),
  y: 104 - ((index * 32) % 164),
  delay: index * 0.38,
  height: 20 + (index % 4) * 8,
  width: 8 + (index % 2) * 2,
}))

export default function ThreeDIntelligence() {
  const shouldReduceMotion = useReducedMotion()
  const isPageVisible = usePageVisibility()
  const shouldAnimate = !shouldReduceMotion && isPageVisible

  return (
    <div className="relative mx-auto aspect-square w-full max-w-[38rem]" aria-label="ภาพจำลอง AI agents ทำงานร่วมกันอย่างต่อเนื่อง">
      <div className="absolute inset-[5%] rounded-full bg-amber-300/15 blur-3xl" aria-hidden="true" />
      <div className="absolute inset-0" style={{ perspective: '1100px' }} aria-hidden="true">
        <m.div
          className="absolute inset-0"
          style={{ transformStyle: 'preserve-3d' }}
          animate={!shouldAnimate ? { rotateX: -8, rotateY: 12 } : { rotateX: [-10, 9, -10], rotateY: [0, 360] }}
          transition={!shouldAnimate ? { duration: 0 } : { rotateX: { duration: 8, repeat: Infinity, ease: 'easeInOut' }, rotateY: { duration: 22, repeat: Infinity, ease: 'linear' } }}
        >
          <div className="absolute left-1/2 top-1/2 h-[82%] w-[82%] -translate-x-1/2 -translate-y-1/2 rounded-full border border-amber-200/30" style={{ transform: 'rotateX(68deg) translateZ(-20px)' }} />
          <div className="absolute left-1/2 top-1/2 h-[60%] w-[60%] -translate-x-1/2 -translate-y-1/2 rounded-full border border-sky-200/25" style={{ transform: 'rotateX(-66deg) rotateZ(28deg) translateZ(-4px)' }} />
          <div className="absolute left-1/2 top-1/2 h-[40%] w-[88%] -translate-x-1/2 -translate-y-1/2 rounded-full border border-rose-200/20" style={{ transform: 'rotateY(64deg) rotateZ(-14deg) translateZ(-12px)' }} />

          <div className="absolute left-1/2 top-1/2" style={{ transform: 'translate3d(-50%, -50%, 72px)' }}>
            <m.div
              className="relative flex h-44 w-44 items-center justify-center rounded-[2.25rem] border border-amber-100/70 bg-[radial-gradient(circle_at_30%_25%,#fff7d1,transparent_22%),linear-gradient(135deg,#f4b740,#924a2e_58%,#18181b)] shadow-[0_35px_70px_rgba(146,74,46,0.32)] sm:h-52 sm:w-52"
              animate={!shouldAnimate ? { scale: 1 } : { scale: [1, 1.06, 1], rotateZ: [-2, 2, -2] }}
              transition={{ duration: 4.2, repeat: Infinity, ease: 'easeInOut' }}
            >
              <div className="absolute inset-3 rounded-[1.7rem] border border-white/25" />
              <div className="absolute h-24 w-24 rounded-full border border-amber-100/50 bg-amber-100/10 blur-[1px]" />
              <span className="relative font-serif text-6xl text-amber-50 drop-shadow-[0_8px_12px_rgba(24,24,27,0.45)] sm:text-7xl">∞</span>
            </m.div>
          </div>

          {ORBIT_AGENTS.map((agent) => (
            <div
              key={`${agent.x}-${agent.y}`}
              className="absolute left-1/2 top-1/2"
              style={{ transform: `translate3d(${agent.x}px, ${agent.y}px, ${agent.z}px)` }}
            >
              <m.div
                className="relative h-16 w-16 [transform-style:preserve-3d] sm:h-[4.5rem] sm:w-[4.5rem]"
                style={{ filter: `drop-shadow(0 16px 20px ${agent.color}40)` }}
                animate={!shouldAnimate ? { opacity: 1 } : { y: [0, -16, 0], rotateZ: [agent.tilt, -agent.tilt, agent.tilt], rotateY: [-16, 18, -16] }}
                transition={{ duration: 3.6, delay: agent.delay, repeat: Infinity, ease: 'easeInOut' }}
              >
                <span className="absolute bottom-0 left-1/2 h-3 w-11 -translate-x-1/2 rounded-full bg-zinc-950/70 blur-md" />
                <span className="absolute left-[-4px] top-8 h-4 w-5 rotate-[-18deg] rounded-md border border-white/35 bg-zinc-300/85 shadow-md" style={{ transform: 'translateZ(2px)' }} />
                <span className="absolute right-[-4px] top-8 h-4 w-5 rotate-[18deg] rounded-md border border-white/35 bg-zinc-300/85 shadow-md" style={{ transform: 'translateZ(2px)' }} />
                <div className="absolute inset-x-2 bottom-1 h-8 rounded-[1.2rem] border border-white/55 bg-[linear-gradient(135deg,#e4e4e7,#52525b_52%,#18181b)] shadow-lg" style={{ transform: 'translateZ(8px)' }}>
                  <span className="absolute left-1/2 top-2 h-3.5 w-5 -translate-x-1/2 rounded-full border border-white/30" style={{ backgroundColor: agent.color }} />
                  <span className="absolute bottom-[-4px] left-1/2 h-2 w-6 -translate-x-1/2 rounded-full bg-sky-200/90 blur-[1px]" />
                </div>
                <div className="absolute inset-x-2 top-0 h-10 rounded-[48%] border border-white/80 bg-[radial-gradient(circle_at_32%_22%,#ffffff,transparent_18%),linear-gradient(145deg,#fafafa,#a1a1aa_54%,#3f3f46)] shadow-xl" style={{ transform: 'translateZ(14px)' }}>
                  <div className="absolute inset-x-2 top-4 h-3.5 rounded-full border border-cyan-100/50 bg-zinc-900 shadow-inner">
                    <span className="absolute left-2 top-1 h-1.5 w-1.5 rounded-full bg-cyan-200 shadow-[0_0_8px_rgba(165,243,252,1)]" />
                    <span className="absolute right-2 top-1 h-1.5 w-1.5 rounded-full bg-cyan-200 shadow-[0_0_8px_rgba(165,243,252,1)]" />
                  </div>
                  <span className="absolute left-1/2 top-[-6px] h-2.5 w-2.5 -translate-x-1/2 rounded-full border-2 border-zinc-100 bg-zinc-700" />
                </div>
              </m.div>
            </div>
          ))}
        </m.div>
      </div>

      <div className="absolute inset-0" aria-hidden="true">
        {COMPOUND_CANDLES.map((candle) => (
          <m.div
            key={`${candle.x}-${candle.y}`}
            className="absolute left-1/2 top-1/2 [transform-style:preserve-3d]"
            style={{ width: candle.width, height: candle.height + 16, marginLeft: candle.x, marginTop: candle.y }}
            animate={!shouldAnimate ? { opacity: 0.72 } : { opacity: [0, 1, 0], y: [48, -70, -150], x: [0, 18, -12], scale: [0.5, 1.16, 0.34], rotateY: [-18, 18, 34] }}
            transition={{ duration: 4.7, delay: candle.delay, repeat: Infinity, ease: 'easeOut' }}
          >
            <span className="absolute left-1/2 top-0 h-full w-px -translate-x-1/2 bg-emerald-100/90" />
            <span className="absolute bottom-2 left-0 right-0 rounded-sm border border-emerald-100/75 bg-[linear-gradient(90deg,#047857,#34d399_48%,#a7f3d0)] shadow-[0_0_15px_rgba(52,211,153,0.85)]" style={{ height: candle.height, transform: 'translateZ(8px)' }} />
          </m.div>
        ))}
      </div>

      <m.div
        className="absolute bottom-[9%] left-1/2 -translate-x-1/2 rounded-2xl border border-white/50 bg-white/75 px-4 py-2 text-center shadow-xl shadow-zinc-900/10 backdrop-blur"
        animate={!shouldAnimate ? { opacity: 1 } : { y: [0, -7, 0] }}
        transition={{ duration: 3.6, repeat: Infinity, ease: 'easeInOut' }}
      >
        <p className="text-[9px] font-bold uppercase tracking-[0.22em] text-zinc-400">Simulation</p>
        <p className="mt-0.5 font-mono text-sm font-bold text-zinc-900">100 → 116</p>
      </m.div>
    </div>
  )
}
