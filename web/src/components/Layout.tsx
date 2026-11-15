import { useEffect, useState } from 'react'
import { Link, Outlet, useLocation } from 'react-router-dom'
import RequireAuth from './RequireAuth'
import Sidebar from './Sidebar'
import { MenuIcon } from './ui/AppIcons'

function resolveSectionKey(pathname: string): string {
  if (pathname.startsWith('/manager') || pathname.startsWith('/advisor') || pathname.startsWith('/kanban')) {
    return 'manager'
  }
  if (pathname.startsWith('/roundtable')) return 'roundtable'
  if (pathname.startsWith('/digest')) return 'digest'
  if (pathname.startsWith('/portfolio')) return 'portfolio'
  if (pathname.startsWith('/advisory')) return 'advisory'
  if (pathname.startsWith('/macro')) return 'macro'
  if (pathname.startsWith('/equity')) return 'equity'
  return 'default'
}

const SECTION_LABELS: Record<string, string> = {
  manager: 'Trinity AI Advisor',
  roundtable: 'Guru Roundtable',
  digest: 'Daily Digest',
  portfolio: 'All-Asset Portfolio',
  advisory: 'Advisory Desk',
  macro: 'Macroeconomic',
  equity: 'Equity Analysis',
  default: 'Money ReRoute',
}

export default function Layout() {
  const location = useLocation()
  const [mobileNavOpen, setMobileNavOpen] = useState(false)
  const sectionKey = resolveSectionKey(location.pathname)
  const sectionLabel = SECTION_LABELS[sectionKey] || 'Money ReRoute'
  const isFullHeightPage =
    location.pathname.startsWith('/equity') ||
    location.pathname.startsWith('/manager') ||
    location.pathname.startsWith('/roundtable')

  useEffect(() => {
    setMobileNavOpen(false)
  }, [location.pathname])

  return (
    <RequireAuth>
      <div data-section={sectionKey} className="flow-theme flex flex-col md:flex-row h-full w-full overflow-hidden">
        <a
          href="#main-content"
          className="sr-only focus:not-sr-only focus:fixed focus:top-3 focus:left-3 focus:z-[100] focus:rounded-xl focus:border focus:border-sky-300 focus:bg-white focus:px-4 focus:py-2 focus:text-xs focus:font-bold focus:text-slate-900 focus:shadow-lg focus-visible:ring-2 focus-visible:ring-section-accent"
        >
          ข้ามไปเนื้อหาหลัก
        </a>
        {/* Mobile Navigation Header (< md) */}
        <header className="md:hidden shrink-0 h-14 px-3.5 border-b border-sky-100/90 bg-surface-1/95 backdrop-blur-xl flex items-center justify-between gap-2 z-30">
          <div className="flex items-center gap-2.5 min-w-0">
            <button
              type="button"
              onClick={() => setMobileNavOpen(true)}
              aria-label="เปิดเมนูนำทาง"
              aria-expanded={mobileNavOpen}
              className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl border border-sky-100 bg-white/85 text-slate-700 hover:bg-sky-50 hover:text-section-accent transition-colors shadow-sm"
            >
              <MenuIcon className="h-5 w-5" />
            </button>
            <Link to="/manager" className="flex items-center gap-2 min-w-0">
              <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-lg bg-gradient-to-tr from-section-accent via-cyan-500 to-section-accent-2 text-[11px] font-black text-white shadow-sm">
                MR
              </div>
              <span className="text-sm font-bold tracking-tight text-slate-900 truncate">
                {sectionLabel}
              </span>
            </Link>
          </div>
          <span className="shrink-0 rounded-full border border-sky-200 bg-sky-50/90 px-2.5 py-0.5 text-[11px] font-semibold text-sky-800">
            Trinity AI
          </span>
        </header>

        <Sidebar mobileOpen={mobileNavOpen} onCloseMobile={() => setMobileNavOpen(false)} />
        <main
          id="main-content"
          tabIndex={-1}
          className={`flow-main scroll-pt-16 min-w-0 min-h-0 flex-1 md:h-full focus:outline-none ${
            isFullHeightPage ? 'overflow-hidden p-0' : 'overflow-y-auto p-4 sm:p-5 lg:p-6 xl:p-8 custom-scrollbar'
          }`}
        >
          <Outlet />
        </main>
      </div>
    </RequireAuth>
  )
}
