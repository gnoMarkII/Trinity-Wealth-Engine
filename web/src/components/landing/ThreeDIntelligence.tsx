import { m, useReducedMotion } from 'motion/react'
import { usePageVisibility } from './usePageVisibility'

const SATELLITES = [
  { mark: 'R', x: -154, y: -76, z: 70, color: '#f59e0b', delay: 0.1 },
  { mark: 'Q', x: 138, y: -104, z: 38, color: '#60a5fa', delay: 0.4 },
  { mark: 'S', x: 166, y: 82, z: 54, color: '#fb7185', delay: 0.7 },
  { mark: 'G', x: -126, y: 122, z: 24, color: '#34d399', delay: 1.0 },
]

const COIN_STREAM = Array.from({ length: 11 }, (_, index) => ({
  x: -110 + ((index * 47) % 225),
  y: 92 - ((index * 31) % 138),
  delay: index * 0.42,
  size: 6 + (index % 3) * 3,
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

          {SATELLITES.map((satellite) => (
            <div
              key={satellite.mark}
              className="absolute left-1/2 top-1/2"
              style={{ transform: `translate3d(${satellite.x}px, ${satellite.y}px, ${satellite.z}px)` }}
            >
              <m.div
                className="flex h-14 w-14 items-center justify-center rounded-2xl border border-white/35 bg-zinc-950/75 text-sm font-bold text-white shadow-2xl backdrop-blur sm:h-16 sm:w-16"
                style={{ boxShadow: `0 14px 36px ${satellite.color}38` }}
                animate={!shouldAnimate ? { opacity: 1 } : { y: [0, -12, 0], rotateZ: [-4, 5, -4] }}
                transition={{ duration: 3.2, delay: satellite.delay, repeat: Infinity, ease: 'easeInOut' }}
              >
                <span className="rounded-lg px-2 py-1" style={{ color: satellite.color, backgroundColor: `${satellite.color}1f` }}>{satellite.mark}</span>
              </m.div>
            </div>
          ))}
        </m.div>
      </div>

      <div className="absolute inset-0" aria-hidden="true">
        {COIN_STREAM.map((coin) => (
          <m.span
            key={`${coin.x}-${coin.y}`}
            className="absolute left-1/2 top-1/2 rounded-full bg-amber-300 shadow-[0_0_18px_rgba(251,191,36,0.9)]"
            style={{ width: coin.size, height: coin.size, marginLeft: coin.x, marginTop: coin.y }}
            animate={!shouldAnimate ? { opacity: 0.65 } : { opacity: [0, 1, 0], y: [26, -46, -100], x: [0, 12, -10], scale: [0.55, 1.15, 0.4] }}
            transition={{ duration: 3.8, delay: coin.delay, repeat: Infinity, ease: 'easeOut' }}
          />
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
