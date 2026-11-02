import { useEffect, useState } from 'react'
import { api } from '../api/client'
import type { ActiveAgentStatusDTO } from '../api/types'
import { flowLabel } from '../lib/flows'
import { usePageVisibility } from '../hooks/usePageVisibility'

const POLL_INTERVAL_MS = 4000

export interface AgentRosterItem {
  label: string
  sublabel: string
  icon: string
  nodeKeys: string[]
}

// รายชื่อ 5 Specialist Agents ของ Trinity Swarm (ภายใต้การกำกับของ Hermes Manager)
// เชื่อมโยง nodeKeys ครบทุก specialist state (quant, narrative, synthesizer, allocator, archivist)
export const AGENT_ROSTER: AgentRosterItem[] = [
  {
    label: 'Macro Quant',
    sublabel: 'Observable Matrix',
    icon: '📊',
    nodeKeys: ['macro_quant', 'post_quant'],
  },
  {
    label: 'Macro Economist',
    sublabel: 'Global Macro & Policy',
    icon: '🌐',
    nodeKeys: ['macro_economist', 'post_economist'],
  },
  {
    label: 'Strategic Allocator',
    sublabel: 'Guardrails & Regime',
    icon: '⚖️',
    nodeKeys: ['strategic_allocator', 'post_macro_intel'],
  },
  {
    label: 'Equity Intel',
    sublabel: '3-Stage Stock Deep Dive',
    icon: '📈',
    nodeKeys: [
      'equity_quant',
      'equity_narrative',
      'equity_synthesizer',
      'post_equity_quant',
      'post_equity_narrative',
      'post_equity_intel',
    ],
  },
  {
    label: 'Archivist',
    sublabel: 'Obsidian Memory Vault',
    icon: '📚',
    nodeKeys: ['archivist', 'prepare_archivist', 'post_archivist'],
  },
]

export default function AgentStatusPanel() {
  const [status, setStatus] = useState<ActiveAgentStatusDTO | null>(null)
  const isPageVisible = usePageVisibility()

  useEffect(() => {
    // หยุด poll ตอน tab ถูกซ่อน เพื่อประหยัด bandwidth และ background request
    if (!isPageVisible) return
    let cancelled = false

    function poll() {
      api
        .getActiveAgentStatus()
        .then((s) => {
          if (!cancelled) setStatus(s)
        })
        .catch(() => {
          // เงียบไว้ — เดี๋ยว poll รอบถัดไปลองใหม่
        })
    }

    poll()
    const timer = window.setInterval(poll, POLL_INTERVAL_MS)
    return () => {
      cancelled = true
      window.clearInterval(timer)
    }
  }, [isPageVisible])

  const running = status?.running ?? false
  const isOtherFlow = running && status?.flow !== 'manager'

  return (
    <div className="space-y-2 border-t border-sky-100/80 pt-3">
      {/* Header with Title and Live State Pill */}
      <div className="flex items-center justify-between px-1">
        <h3 className="text-[10px] font-bold uppercase tracking-[0.16em] text-sky-800 flex items-center gap-1.5">
          <span>Trinity Swarm</span>
        </h3>
        <span
          className={`inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[10px] font-semibold transition-all ${
            running
              ? 'border border-emerald-200 bg-emerald-50 text-emerald-700 shadow-[0_0_8px_rgba(16,185,129,0.2)]'
              : 'border border-sky-100 bg-sky-50/60 text-sky-700'
          }`}
        >
          <span
            className={`h-1.5 w-1.5 rounded-full ${
              running ? 'animate-pulse bg-emerald-500' : 'bg-sky-400'
            }`}
          />
          {running ? 'Active Swarm' : 'Standby (5)'}
        </span>
      </div>

      {/* Main Glassmorphic Swarm Card */}
      <div className="rounded-2xl border border-sky-100/80 bg-white/60 p-1.5 shadow-[0_6px_24px_rgba(14,165,233,0.04)] backdrop-blur-md">
        {isOtherFlow && (
          <div className="mb-2 flex items-center gap-1.5 rounded-xl border border-amber-200 bg-amber-50/90 px-2.5 py-1.5 text-[11px] font-medium text-amber-800 shadow-sm">
            <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-amber-500" />
            <span>Workflow: {flowLabel(status?.flow ?? '')}</span>
          </div>
        )}

        <ul className="space-y-1">
          {AGENT_ROSTER.map((agent) => {
            const active = running && !isOtherFlow && agent.nodeKeys.includes(status?.node ?? '')
            return (
              <li
                key={agent.label}
                className={`group flex items-center justify-between rounded-xl px-2 py-1.5 transition-all text-xs ${
                  active
                    ? 'border border-emerald-200/80 bg-emerald-50/90 text-emerald-950 shadow-[0_2px_8px_rgba(16,185,129,0.12)]'
                    : 'text-zinc-600 hover:bg-sky-50/60 hover:text-zinc-900'
                }`}
              >
                <div className="flex items-center gap-2 min-w-0">
                  <span className="flex h-5 w-5 shrink-0 items-center justify-center rounded-lg bg-sky-100/60 text-[11px]">
                    {agent.icon}
                  </span>
                  <div className="min-w-0 flex flex-col">
                    <span className={`truncate text-xs font-medium leading-tight ${active ? 'text-emerald-950 font-semibold' : 'text-zinc-800'}`}>
                      {agent.label}
                    </span>
                    <span className="truncate text-[9px] text-zinc-400 group-hover:text-sky-700/70 transition-colors">
                      {agent.sublabel}
                    </span>
                  </div>
                </div>

                <div className="shrink-0 pl-1.5">
                  {active ? (
                    <span className="relative flex h-2 w-2">
                      <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-emerald-400 opacity-75" />
                      <span className="relative inline-flex h-2 w-2 rounded-full bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.7)]" />
                    </span>
                  ) : (
                    <span className="inline-block h-1.5 w-1.5 rounded-full bg-zinc-300 group-hover:bg-sky-400 transition-colors" />
                  )}
                </div>
              </li>
            )
          })}
        </ul>
      </div>
    </div>
  )
}
