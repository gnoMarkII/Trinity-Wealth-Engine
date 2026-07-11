import { NavLink } from 'react-router-dom'
import { useAuth } from '../auth/useAuth'
import AgentStatusPanel from './AgentStatusPanel'

const links = [
  { to: '/', label: 'Home' },
  { to: '/kanban', label: 'Agent Kanban Board' },
  { to: '/portfolio', label: 'All-Asset Portfolio' },
  { to: '/macro', label: 'Macroeconomic Analysis' },
]

export default function Sidebar() {
  const { logout } = useAuth()

  return (
    <aside className="flex w-60 shrink-0 flex-col border-r border-sky-100 bg-panel p-4 shadow-[10px_0_35px_rgba(14,165,233,0.04)] backdrop-blur-xl">
      <h2 className="mb-6 flex items-center gap-2 px-1 text-sm font-semibold tracking-[0.08em] text-sky-950">
        <span className="h-2 w-2 rounded-full bg-flow-cyan shadow-[0_0_10px_rgba(6,182,212,0.5)]" />
        Money ReRoute
      </h2>
      <nav className="flex flex-col gap-1">
        {links.map((link) => (
          <NavLink
            key={link.to}
            to={link.to}
            className={({ isActive }) =>
              `rounded-xl px-3 py-2 text-sm transition-colors ${
                isActive
                  ? 'border border-sky-200 bg-panel font-medium text-sky-700 shadow-[0_8px_24px_rgba(14,165,233,0.08)]'
                  : 'text-zinc-500 hover:bg-surface-strong hover:text-zinc-800'
              }`
            }
          >
            {link.label}
          </NavLink>
        ))}
      </nav>

      <AgentStatusPanel />

      <div className="mt-auto border-t border-sky-100 pt-3">
        <button
          onClick={() => logout()}
          className="w-full rounded-xl px-3 py-2 text-left text-sm text-zinc-500 transition-colors hover:bg-surface-strong hover:text-zinc-800"
        >
          ออกจากระบบ
        </button>
      </div>
    </aside>
  )
}
