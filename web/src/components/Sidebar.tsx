import { NavLink } from 'react-router-dom'
import { useAuth } from '../auth/useAuth'
import HarnessTelemetryWidget from './HarnessTelemetryWidget'

interface NavItem {
  to: string
  label: string
  icon: string
  isFlagship?: boolean
  badge?: string
}

interface NavSection {
  title?: string
  items: NavItem[]
}

const navSections: NavSection[] = [
  {
    items: [
      {
        to: '/manager',
        label: 'Trinity AI Advisor',
        icon: '✨',
        isFlagship: true,
      },
      {
        to: '/digest',
        label: 'Daily Digest',
        icon: '📰',
      },
    ],
  },
  {
    title: 'Portfolio & Strategy',
    items: [
      {
        to: '/portfolio',
        label: 'All-Asset Portfolio',
        icon: '💼',
      },
      {
        to: '/advisory',
        label: 'Advisory Desk',
        icon: '⚡',
      },
    ],
  },
  {
    title: 'Market Research',
    items: [
      {
        to: '/macro',
        label: 'Macroeconomic',
        icon: '🌐',
      },
      {
        to: '/equity',
        label: 'Equity Analysis',
        icon: '📈',
      },
    ],
  },
]

export default function Sidebar() {
  const { logout } = useAuth()

  return (
    <aside className="flex w-60 shrink-0 flex-col border-r border-sky-100 bg-panel/95 p-3.5 shadow-[10px_0_35px_rgba(14,165,233,0.04)] backdrop-blur-xl h-full min-h-0 overflow-y-auto custom-scrollbar select-none">
      {/* Brand Header */}
      <NavLink
        to="/manager"
        className="group mb-5 flex items-center gap-2.5 rounded-2xl px-2 py-1.5 transition-colors hover:bg-sky-50/60"
        title="ไปยัง Trinity AI Advisor"
      >
        <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-xl bg-gradient-to-tr from-sky-400 via-cyan-400 to-sky-500 text-xs font-black text-white shadow-[0_2px_12px_rgba(14,165,233,0.35)] transition-transform group-hover:scale-105">
          MR
        </div>
        <div className="min-w-0">
          <h2 className="flex items-center gap-1.5 text-sm font-bold tracking-tight text-sky-950">
            <span>Money ReRoute</span>
          </h2>
          <p className="text-xs font-semibold tracking-wider uppercase text-sky-600/80">
            Trinity Wealth Engine
          </p>
        </div>
      </NavLink>

      {/* Navigation Sections */}
      <nav className="flex flex-col gap-4">
        {navSections.map((section, sIdx) => (
          <div key={sIdx} className="space-y-1">
            {section.title && (
              <h3 className="px-2 pb-0.5 text-[11px] font-bold uppercase tracking-wider text-zinc-400">
                {section.title}
              </h3>
            )}
            <div className="space-y-0.5">
              {section.items.map((item) => (
                <NavLink
                  key={item.to}
                  to={item.to}
                  className={({ isActive }) =>
                    `group flex items-center justify-between rounded-xl px-2.5 py-2 text-sm transition-all ${
                      isActive
                        ? item.isFlagship
                          ? 'border border-sky-300/90 bg-gradient-to-r from-sky-100/80 to-cyan-50/70 font-semibold text-sky-950 shadow-[0_2px_10px_rgba(14,165,233,0.12)]'
                          : 'border border-sky-200/90 bg-sky-50/80 font-semibold text-sky-900 shadow-[0_2px_8px_rgba(14,165,233,0.06)]'
                        : item.isFlagship
                          ? 'border border-sky-100 bg-white/60 font-medium text-sky-900 hover:border-sky-200 hover:bg-sky-50/50'
                          : 'text-zinc-600 hover:bg-sky-50/50 hover:text-zinc-900'
                    }`
                  }
                >
                  <div className="flex items-center gap-2 min-w-0">
                    <span className="flex h-5 w-5 shrink-0 items-center justify-center rounded-lg bg-sky-100/50 text-xs group-hover:scale-110 transition-transform">
                      {item.icon}
                    </span>
                    <span className="truncate">{item.label}</span>
                  </div>

                  {item.badge && (
                    <span className="rounded-md border border-sky-200 bg-sky-50 px-2 py-0.5 text-[11px] font-bold text-sky-700">
                      {item.badge}
                    </span>
                  )}
                </NavLink>
              ))}
            </div>
          </div>
        ))}
      </nav>

      {/* Institutional Harness Telemetry */}
      <HarnessTelemetryWidget />

      {/* Footer / Logout */}
      <div className="mt-auto border-t border-sky-100/80 pt-3">
        <button
          onClick={() => logout()}
          className="flex w-full items-center gap-2 rounded-xl px-2.5 py-2 text-left text-sm text-zinc-500 transition-colors hover:bg-red-50 hover:text-red-600"
        >
          <span>🚪</span>
          <span>ออกจากระบบ</span>
        </button>
      </div>
    </aside>
  )
}
