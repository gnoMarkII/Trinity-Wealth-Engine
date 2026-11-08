import { AGENT_ROSTER, type AgentRosterItem } from '../AgentStatusPanel'
import type { ActiveAgentStatusDTO } from '../../api/types'
import { nodeDisplayName } from '../../lib/nodeDisplayNames'
import { flowLabel } from '../../lib/flows'

export interface ManagerSwarmViewProps {
  activeStatus?: ActiveAgentStatusDTO | null
  onTriggerResearch?: (prompt: string) => void
}

// ข้อมูลความสามารถและเครื่องมือเฉพาะทางของแต่ละ Specialist Agent
const AGENT_CAPABILITIES: Record<string, { roleTitle: string; tools: string[]; focusArea: string }> = {
  'Macro Quant': {
    roleTitle: 'Quantitative Observable Matrix',
    tools: ['Bond Yield Curve', 'FRED Data Matrix', 'Liquidity Proxy'],
    focusArea: 'คำนวณ Yield Spread, ค่าเงินดอลลาร์ และดัชนีชี้วัดเศรษฐกิจเชิงปริมาณ',
  },
  'Macro Economist': {
    roleTitle: 'Global Macro & Policy Specialist',
    tools: ['Fed/BoT Tracker', 'Policy Sentiment', 'Regime Synthesis'],
    focusArea: 'วิเคราะห์นโยบายการเงิน ธนาคารกลาง ทิศทางดอกเบี้ย และสภาวะเศรษฐกิจมหภาค',
  },
  'Strategic Allocator': {
    roleTitle: 'Guardrails & Portfolio Optimizer',
    tools: ['3-Bucket Allocation', 'Risk Parity', 'Regime Guardrails'],
    focusArea: 'กำหนดสัดส่วนสินทรัพย์ 3 ถัง (สภาพคล่อง / หลัก / เติบโต) ตามสภาวะตลาด',
  },
  'Equity Intel': {
    roleTitle: '3-Stage Stock Deep Dive',
    tools: ['5x5 DCF Sensitivity', 'Earnings Analyzer', 'Moat Scenarios'],
    focusArea: 'วิเคราะห์หุ้นรายตัวเชิงลึก 3 ขั้นตอน: งบการเงิน, Transcript และมูลค่าเหมาะสม',
  },
  Archivist: {
    roleTitle: 'Obsidian Memory & Knowledge Vault',
    tools: ['Obsidian Adapter', 'Vector Search', 'Audit Trail'],
    focusArea: 'บันทึกบทเรียน การตัดสินใจลงทุน และประวัติพอร์ตลงใน Obsidian Vault',
  },
}

export default function ManagerSwarmView({ activeStatus }: ManagerSwarmViewProps) {
  const isRunning = Boolean(activeStatus?.running)
  const isOtherFlow = isRunning && activeStatus?.flow !== 'manager'
  const activeNode = activeStatus?.node

  return (
    <div className="flex h-full flex-col min-h-0 overflow-y-auto custom-scrollbar p-4 space-y-4 animate-page-in">
      {/* Swarm Live Banner */}
      <div className="rounded-2xl border border-sky-100 bg-gradient-to-br from-white/90 via-sky-50/40 to-cyan-50/60 p-3.5 shadow-sm backdrop-blur-md">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2.5">
            <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-gradient-to-tr from-sky-400 via-cyan-400 to-sky-600 text-white shadow-[0_2px_10px_rgba(14,165,233,0.3)]">
              <span className="text-base">🐝</span>
            </div>
            <div>
              <h3 className="text-xs font-bold text-sky-950 flex items-center gap-1.5">
                <span>Trinity Swarm</span>
                <span className="text-[10px] text-sky-600 font-semibold">(5 Specialists)</span>
              </h3>
              <p className="text-[10px] text-zinc-500">
                Institutional Autonomous AI Team
              </p>
            </div>
          </div>

          <span
            className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-[10px] font-semibold transition-all ${
              isRunning
                ? 'border border-emerald-200 bg-emerald-50 text-emerald-700 shadow-[0_0_10px_rgba(16,185,129,0.25)]'
                : 'border border-sky-200 bg-sky-50/70 text-sky-700'
            }`}
          >
            <span
              className={`h-2 w-2 rounded-full ${
                isRunning ? 'animate-pulse bg-emerald-500' : 'bg-sky-400'
              }`}
            />
            {isRunning ? (
              <span>Active Swarm</span>
            ) : (
              <span>Standby (พร้อมทำงาน)</span>
            )}
          </span>
        </div>

        {/* Other Flow Indicator */}
        {isOtherFlow && (
          <div className="mt-2.5 flex items-center gap-2 rounded-xl border border-amber-200 bg-amber-50/90 px-3 py-1.5 text-xs font-medium text-amber-800 shadow-sm animate-notice-in">
            <span className="h-2 w-2 animate-pulse rounded-full bg-amber-500" />
            <span>กำลังรัน Workflow: {flowLabel(activeStatus?.flow ?? '')}</span>
          </div>
        )}
      </div>

      {/* Agents Roster Cards */}
      <div className="space-y-3">
        <div className="flex items-center justify-between px-1">
          <span className="text-xs font-bold uppercase tracking-wider text-zinc-400">
            รายชื่อ Institutional Agents
          </span>
          <span className="text-xs text-zinc-400 font-mono">
            Node: {activeNode ? nodeDisplayName(activeNode) : 'idle'}
          </span>
        </div>

        {AGENT_ROSTER.map((agent: AgentRosterItem) => {
          const isActive =
            isRunning && !isOtherFlow && agent.nodeKeys.includes(activeNode ?? '')
          const caps = AGENT_CAPABILITIES[agent.label] ?? {
            roleTitle: agent.sublabel,
            tools: ['Trinity Core'],
            focusArea: agent.sublabel,
          }

          return (
            <div
              key={agent.label}
              className={`relative overflow-hidden rounded-2xl border p-3.5 transition-all ${
                isActive
                  ? 'border-emerald-300 bg-gradient-to-tr from-emerald-50/95 via-sky-50/80 to-cyan-50/90 shadow-[0_4px_18px_rgba(16,185,129,0.18)] ring-2 ring-emerald-400/60 animate-card-in'
                  : 'border-sky-100/90 bg-white/70 hover:border-sky-200 hover:bg-sky-50/40 hover:shadow-sm'
              }`}
            >
              {/* Active Agent Ambient Pulse Beacon */}
              {isActive && (
                <div className="absolute top-0 right-0 -mr-6 -mt-6 h-20 w-20 rounded-full bg-emerald-400/20 blur-xl pointer-events-none" />
              )}

              <div className="flex items-start justify-between gap-2">
                <div className="flex items-center gap-3 min-w-0">
                  <div
                    className={`relative flex h-10 w-10 shrink-0 items-center justify-center rounded-xl text-lg shadow-sm transition-transform ${
                      isActive
                        ? 'bg-gradient-to-tr from-emerald-400 to-cyan-500 text-white scale-105'
                        : 'bg-sky-100/70 text-zinc-700'
                    }`}
                  >
                    <span>{agent.icon}</span>
                    {isActive && (
                      <span className="absolute -top-1 -right-1 flex h-3.5 w-3.5">
                        <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-emerald-400 opacity-75" />
                        <span className="relative inline-flex h-3.5 w-3.5 rounded-full bg-emerald-500 border-2 border-white" />
                      </span>
                    )}
                  </div>

                  <div className="min-w-0">
                    <h4
                      className={`text-sm font-bold leading-tight truncate ${
                        isActive ? 'text-emerald-950 font-bold' : 'text-sky-950'
                      }`}
                    >
                      {agent.label}
                    </h4>
                    <p className="text-xs text-zinc-500 font-medium truncate mt-0.5">
                      {caps.roleTitle}
                    </p>
                  </div>
                </div>

                <div className="shrink-0">
                  {isActive ? (
                    <span className="inline-flex items-center gap-1 rounded-full border border-emerald-300 bg-emerald-100/90 px-2.5 py-0.5 text-xs font-bold text-emerald-800 shadow-sm">
                      <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-emerald-600" />
                      กำลังทำงาน
                    </span>
                  ) : (
                    <span className="inline-flex items-center gap-1 rounded-full border border-zinc-200/80 bg-zinc-50 px-2 py-0.5 text-xs font-semibold text-zinc-500">
                      Standby
                    </span>
                  )}
                </div>
              </div>

              {/* Focus Area Description */}
              <p className="mt-2.5 text-xs sm:text-[13px] leading-relaxed text-zinc-600">
                {caps.focusArea}
              </p>

              {/* Tool Chips */}
              <div className="mt-2.5 flex flex-wrap gap-1.5">
                {caps.tools.map((tool) => (
                  <span
                    key={tool}
                    className={`rounded-md border px-2 py-0.5 text-xs font-medium font-mono ${
                      isActive
                        ? 'border-emerald-200 bg-emerald-50 text-emerald-800'
                        : 'border-sky-100 bg-sky-50/60 text-sky-700'
                    }`}
                  >
                    {tool}
                  </span>
                ))}
              </div>
            </div>
          )
        })}
      </div>

      {/* Collaboration Flow Diagram Card */}
      <div className="rounded-2xl border border-sky-100 bg-white/70 p-4 shadow-sm">
        <h4 className="text-xs font-bold uppercase tracking-wider text-zinc-500 mb-2.5">
          🏛️ Trinity Multi-Agent Architecture
        </h4>
        <div className="flex flex-col gap-2 text-xs font-medium text-zinc-700">
          <div className="flex items-center gap-2.5 p-2 rounded-xl bg-sky-50/70 border border-sky-100">
            <span className="font-bold text-sky-900 shrink-0">1. Hermes (Manager)</span>
            <span className="text-zinc-400">➔</span>
            <span className="text-zinc-600 truncate">วิเคราะห์เจตนา & สั่งรันสายพานวิจัยอัตโนมัติ</span>
          </div>
          <div className="flex items-center gap-2.5 p-2 rounded-xl bg-cyan-50/70 border border-cyan-100">
            <span className="font-bold text-cyan-900 shrink-0">2. Specialists</span>
            <span className="text-zinc-400">➔</span>
            <span className="text-zinc-600 truncate">Macro Quant, Economist, Equity Intel</span>
          </div>
          <div className="flex items-center gap-2.5 p-2 rounded-xl bg-indigo-50/70 border border-indigo-100">
            <span className="font-bold text-indigo-900 shrink-0">3. Allocator</span>
            <span className="text-zinc-400">➔</span>
            <span className="text-zinc-600 truncate">วาง Guardrails & สัดส่วนพอร์ต 3 ถัง</span>
          </div>
          <div className="flex items-center gap-2.5 p-2 rounded-xl bg-emerald-50/70 border border-emerald-100">
            <span className="font-bold text-emerald-900 shrink-0">4. Archivist</span>
            <span className="text-zinc-400">➔</span>
            <span className="text-zinc-600 truncate">จัดเก็บบันทึกลง Obsidian Memory Vault</span>
          </div>
        </div>
      </div>
    </div>
  )
}

