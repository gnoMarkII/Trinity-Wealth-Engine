import { m, useReducedMotion } from 'motion/react'
import { useLoopingIndex } from './useLoopingIndex'

const AGENTS = [
  { name: 'Research', detail: 'Scans signals', status: 'Researching', left: '4%', top: '42%', color: '#924a2e' },
  { name: 'Macro Quant', detail: 'Scores regimes', status: 'Quantifying', left: '25%', top: '12%', color: '#3987e5' },
  { name: 'Strategist', detail: 'Builds thesis', status: 'Synthesizing', left: '46%', top: '42%', color: '#c98500' },
  { name: 'Risk Guard', detail: 'Tests downside', status: 'Validating', left: '67%', top: '12%', color: '#e66767' },
  { name: 'Portfolio', detail: 'Monitors plan', status: 'Monitoring', left: '86%', top: '42%', color: '#199e70' },
]

const PACKET_PATH = {
  cx: [62, 184, 306, 430, 550],
  cy: [146, 70, 146, 70, 146],
}

export default function AgentWorkflowAnimation() {
  const activeIndex = useLoopingIndex(AGENTS.length, 1_450)
  const shouldReduceMotion = useReducedMotion()

  return (
    <section className="relative overflow-hidden rounded-3xl border border-zinc-200 bg-zinc-950 p-5 shadow-2xl shadow-zinc-950/10 sm:p-7">
      <div className="absolute inset-0 bg-[radial-gradient(circle_at_50%_0%,rgba(217,119,70,0.28),transparent_42%),linear-gradient(135deg,#18181b,#27272a)]" />
      <div className="relative">
        <div className="flex items-center justify-between gap-3">
          <div>
            <p className="text-xs font-bold uppercase tracking-[0.2em] text-amber-300">Live Agent Loop</p>
            <h2 className="mt-1 text-xl font-semibold text-white">AI agents collaborate continuously</h2>
          </div>
          <span className="inline-flex items-center gap-2 rounded-full border border-emerald-300/30 bg-emerald-400/10 px-3 py-1 text-xs font-semibold text-emerald-200">
            <span className="h-1.5 w-1.5 rounded-full bg-emerald-300" />
            System online
          </span>
        </div>

        <div className="relative mt-6 h-[18rem] min-w-0 sm:h-[19rem]">
          <svg viewBox="0 0 612 220" className="absolute inset-0 h-full w-full" aria-hidden="true">
            <m.path
              d="M62 146 C112 146 132 70 184 70 S254 146 306 146 S378 70 430 70 S500 146 550 146"
              fill="none"
              stroke="rgba(255,255,255,0.22)"
              strokeWidth="2"
              strokeDasharray="6 8"
              animate={shouldReduceMotion ? { pathLength: 1 } : { pathLength: [0.15, 1, 0.15] }}
              transition={{ duration: 6.4, ease: 'easeInOut', repeat: Infinity }}
            />
            {!shouldReduceMotion && (
              <m.circle
                r="7"
                fill="#fbbf24"
                animate={PACKET_PATH}
                transition={{ duration: 6.4, ease: 'linear', repeat: Infinity, repeatDelay: 0.5 }}
              />
            )}
          </svg>

          {AGENTS.map((agent, index) => {
            const isActive = activeIndex === index
            return (
              <m.article
                key={agent.name}
                className="absolute w-28 -translate-x-1/2 rounded-2xl border px-3 py-3 text-center shadow-xl sm:w-32"
                style={{ left: agent.left, top: agent.top, borderColor: `${agent.color}88`, backgroundColor: '#27272ae8' }}
                animate={shouldReduceMotion ? { opacity: 1 } : { scale: isActive ? 1.08 : 1, y: isActive ? -5 : 0 }}
                transition={{ type: 'spring', stiffness: 280, damping: 20 }}
              >
                <div className="mx-auto h-2 w-2 rounded-full" style={{ backgroundColor: agent.color, boxShadow: isActive ? `0 0 18px ${agent.color}` : 'none' }} />
                <h3 className="mt-2 text-xs font-bold text-white sm:text-sm">{agent.name}</h3>
                <p className="mt-1 text-[10px] leading-snug text-zinc-400">{agent.detail}</p>
                <p className="mt-2 text-[10px] font-semibold" style={{ color: isActive ? '#fcd34d' : '#a1a1aa' }}>
                  {isActive ? agent.status : 'Ready'}
                </p>
              </m.article>
            )
          })}
        </div>

        <div className="grid grid-cols-3 gap-2 border-t border-white/10 pt-4 text-center text-[10px] font-medium text-zinc-400 sm:text-xs">
          <span>Signal intake</span>
          <span>Guardrail check</span>
          <span>Portfolio action</span>
        </div>
      </div>
    </section>
  )
}
