import { NavLink } from 'react-router-dom'
import { useAuth } from '../auth/AuthContext'
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
    <aside className="flex w-60 shrink-0 flex-col border-r border-zinc-200 bg-surface p-4">
      <h2 className="mb-6 flex items-center gap-2 px-1 text-sm font-semibold tracking-wide text-zinc-900">
        <span className="h-2 w-2 rounded-full bg-terra" />
        Money ReRoute
      </h2>
      <nav className="flex flex-col gap-1">
        {links.map((link) => (
          <NavLink
            key={link.to}
            to={link.to}
            className={({ isActive }) =>
              `rounded-lg px-3 py-2 text-sm transition-colors ${
                isActive
                  ? 'bg-white font-medium text-terra shadow-sm shadow-black/5'
                  : 'text-zinc-500 hover:bg-white/60 hover:text-zinc-800'
              }`
            }
          >
            {link.label}
          </NavLink>
        ))}
      </nav>

      <AgentStatusPanel />

      <div className="mt-auto border-t border-zinc-200 pt-3">
        <button
          onClick={() => logout()}
          className="w-full rounded-lg px-3 py-2 text-left text-sm text-zinc-500 transition-colors hover:bg-white/60 hover:text-zinc-800"
        >
          ออกจากระบบ
        </button>
      </div>
    </aside>
  )
}
