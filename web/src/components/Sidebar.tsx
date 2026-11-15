import { useEffect, useState, type ComponentType } from 'react'
import { NavLink } from 'react-router-dom'
import { useAuth } from '../auth/useAuth'
import HarnessTelemetryWidget from './HarnessTelemetryWidget'
import {
  BoltStrategyIcon,
  BriefcasePortfolioIcon,
  ChartTrendIcon,
  CloseIcon,
  GlobeMacroIcon,
  LogoutIcon,
  MicStageIcon,
  NewspaperIcon,
  SidebarCollapseIcon,
  SidebarExpandIcon,
  SparklesIcon,
  type AppIconProps,
} from './ui/AppIcons'

interface NavItem {
  to: string
  label: string
  Icon: ComponentType<AppIconProps>
  isFlagship?: boolean
  badge?: string
}

interface NavSection {
  title?: string
  items: NavItem[]
}

const SIDEBAR_COLLAPSE_STORAGE_KEY = 'trinity_sidebar_collapsed'

function loadCollapsedState(): boolean {
  try {
    return localStorage.getItem(SIDEBAR_COLLAPSE_STORAGE_KEY) === 'true'
  } catch {
    return false
  }
}

const navSections: NavSection[] = [
  {
    items: [
      {
        to: '/manager',
        label: 'Trinity AI Advisor',
        Icon: SparklesIcon,
        isFlagship: true,
      },
      {
        to: '/roundtable',
        label: 'Guru Roundtable',
        Icon: MicStageIcon,
        badge: 'Voice',
      },
      {
        to: '/digest',
        label: 'Daily Digest',
        Icon: NewspaperIcon,
      },
    ],
  },
  {
    title: 'Portfolio & Strategy',
    items: [
      {
        to: '/portfolio',
        label: 'All-Asset Portfolio',
        Icon: BriefcasePortfolioIcon,
      },
      {
        to: '/advisory',
        label: 'Advisory Desk',
        Icon: BoltStrategyIcon,
      },
    ],
  },
  {
    title: 'Market Research',
    items: [
      {
        to: '/macro',
        label: 'Macroeconomic',
        Icon: GlobeMacroIcon,
      },
      {
        to: '/equity',
        label: 'Equity Analysis',
        Icon: ChartTrendIcon,
      },
    ],
  },
]

interface SidebarProps {
  mobileOpen?: boolean
  onCloseMobile?: () => void
}

interface SidebarContentProps {
  isCollapsed: boolean
  isMobile?: boolean
  onToggleCollapse?: () => void
  onCloseMobile?: () => void
  onLogout: () => void
}

function SidebarContent({
  isCollapsed,
  isMobile = false,
  onToggleCollapse,
  onCloseMobile,
  onLogout,
}: SidebarContentProps) {
  return (
    <>
      {/* Brand Header & Rail / Mobile Close Control */}
      <div
        className={`mb-5 flex items-center ${
          isCollapsed ? 'flex-col gap-2.5 w-full' : 'justify-between gap-1.5'
        }`}
      >
        <NavLink
          to="/manager"
          onClick={() => isMobile && onCloseMobile?.()}
          className={`group flex items-center gap-2.5 rounded-2xl transition-colors hover:bg-sky-50/70 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-section-accent/45 ${
            isCollapsed ? 'justify-center p-1.5' : 'px-2 py-1.5 min-w-0 flex-1'
          }`}
          title="ไปยัง Trinity AI Advisor"
        >
          <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-xl bg-gradient-to-tr from-section-accent via-cyan-500 to-section-accent-2 text-xs font-black text-white shadow-sm transition-transform group-hover:scale-105">
            MR
          </div>
          {!isCollapsed && (
            <div className="min-w-0">
              <h2 className="flex items-center gap-1.5 text-sm font-bold tracking-tight text-slate-900 truncate">
                <span>Money ReRoute</span>
              </h2>
              <p className="text-[11px] font-semibold tracking-wider uppercase text-sky-700/80 truncate">
                Trinity Wealth Engine
              </p>
            </div>
          )}
        </NavLink>

        {isMobile ? (
          <button
            type="button"
            onClick={() => onCloseMobile?.()}
            aria-label="ปิดเมนูนำทาง"
            title="ปิดเมนูนำทาง"
            className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl border border-sky-100 bg-white/85 text-slate-500 transition-colors hover:border-sky-200 hover:bg-sky-50 hover:text-slate-900 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-section-accent/45"
          >
            <CloseIcon className="h-4 w-4" />
          </button>
        ) : (
          <button
            type="button"
            onClick={onToggleCollapse}
            aria-label={isCollapsed ? 'ขยายแถบเมนูด้านข้าง' : 'ย่อแถบเมนูด้านข้าง'}
            title={isCollapsed ? 'ขยายแถบเมนูด้านข้าง' : 'ย่อแถบเมนูด้านข้าง (เพิ่มพื้นที่ทำงาน)'}
            className="flex h-7 w-7 shrink-0 items-center justify-center rounded-lg border border-sky-100 bg-white/75 text-slate-500 transition-colors hover:border-sky-200 hover:bg-sky-50 hover:text-sky-700 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-section-accent/45"
          >
            {isCollapsed ? (
              <SidebarExpandIcon className="h-4 w-4" />
            ) : (
              <SidebarCollapseIcon className="h-4 w-4" />
            )}
          </button>
        )}
      </div>

      {/* Navigation Sections */}
      <nav className={`flex flex-col gap-4 ${isCollapsed ? 'w-full items-center' : ''}`} aria-label="เมนูหลัก">
        {navSections.map((section, sIdx) => (
          <div key={sIdx} className={`space-y-1 ${isCollapsed ? 'w-full flex flex-col items-center' : ''}`}>
            {section.title && !isCollapsed && (
              <h3 className="px-2 pb-0.5 text-[11px] font-bold uppercase tracking-wider text-slate-500">
                {section.title}
              </h3>
            )}
            {section.title && isCollapsed && <div className="my-1 h-px w-6 bg-sky-100" aria-hidden="true" />}
            <div className={`space-y-0.5 ${isCollapsed ? 'w-full flex flex-col items-center gap-1' : ''}`}>
              {section.items.map((item) => {
                const { Icon } = item
                return (
                  <NavLink
                    key={item.to}
                    to={item.to}
                    title={item.label}
                    aria-label={item.label}
                    onClick={() => isMobile && onCloseMobile?.()}
                    className={({ isActive }) =>
                      `group relative flex items-center rounded-xl text-sm transition-all focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-section-accent/45 ${
                        isCollapsed
                          ? 'h-10 w-10 justify-center p-0'
                          : 'justify-between px-2.5 py-2 w-full'
                      } ${
                        isActive
                          ? item.isFlagship
                            ? 'border border-sky-300/80 bg-gradient-to-r from-sky-100/85 to-cyan-50/75 font-semibold text-slate-900 shadow-sm'
                            : 'border border-sky-200/85 bg-white/90 font-semibold text-sky-900 shadow-sm'
                          : item.isFlagship
                            ? 'border border-sky-100/80 bg-white/60 font-medium text-slate-800 hover:border-sky-200 hover:bg-sky-50/60'
                            : 'text-slate-600 hover:bg-sky-50/60 hover:text-slate-900'
                      }`
                    }
                  >
                    {({ isActive }) => (
                      <>
                        {!isCollapsed && isActive && (
                          <span
                            aria-hidden="true"
                            className="absolute left-0 top-2 bottom-2 w-1 rounded-r-full bg-section-accent"
                          />
                        )}
                        <div className={`flex items-center min-w-0 ${isCollapsed ? 'justify-center' : 'gap-2.5'}`}>
                          <span
                            className={`flex h-6 w-6 shrink-0 items-center justify-center rounded-lg transition-colors ${
                              isActive
                                ? 'bg-section-accent/15 text-section-accent'
                                : 'bg-sky-100/45 text-slate-500 group-hover:text-sky-700'
                            }`}
                          >
                            <Icon className="h-4 w-4" />
                          </span>
                          {!isCollapsed && <span className="truncate">{item.label}</span>}
                        </div>

                        {!isCollapsed && item.badge && (
                          <span className="rounded-md border border-sky-200 bg-sky-50/90 px-2 py-0.5 text-[11px] font-bold text-sky-700">
                            {item.badge}
                          </span>
                        )}
                      </>
                    )}
                  </NavLink>
                )
              })}
            </div>
          </div>
        ))}
      </nav>

      {/* Institutional Harness Telemetry */}
      <HarnessTelemetryWidget collapsed={isCollapsed} />

      {/* Footer / Logout */}
      <div className={`mt-auto border-t border-sky-100/80 pt-3 ${isCollapsed ? 'w-full flex justify-center' : ''}`}>
        <button
          onClick={() => {
            if (isMobile) onCloseMobile?.()
            onLogout()
          }}
          title="ออกจากระบบ"
          aria-label="ออกจากระบบ"
          className={`flex items-center rounded-xl text-sm text-slate-600 transition-colors hover:bg-rose-50 hover:text-rose-700 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-rose-500/40 ${
            isCollapsed ? 'h-10 w-10 justify-center' : 'w-full gap-2.5 px-2.5 py-2 text-left'
          }`}
        >
          <LogoutIcon className="h-4 w-4 shrink-0" />
          {!isCollapsed && <span>ออกจากระบบ</span>}
        </button>
      </div>
    </>
  )
}

export default function Sidebar({ mobileOpen = false, onCloseMobile }: SidebarProps) {
  const { logout } = useAuth()
  const [isCollapsed, setIsCollapsed] = useState<boolean>(loadCollapsedState)

  useEffect(() => {
    if (!mobileOpen) return
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        onCloseMobile?.()
      }
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [mobileOpen, onCloseMobile])

  const toggleCollapse = () => {
    setIsCollapsed((prev) => {
      const next = !prev
      try {
        localStorage.setItem(SIDEBAR_COLLAPSE_STORAGE_KEY, String(next))
      } catch {
        // ignore storage errors
      }
      return next
    })
  }

  return (
    <>
      {/* Desktop Collapsible Rail */}
      <aside
        className={`hidden md:flex shrink-0 flex-col border-r border-sky-100/90 bg-surface-1 shadow-[10px_0_35px_rgba(14,165,233,0.04)] backdrop-blur-xl h-full min-h-0 overflow-y-auto custom-scrollbar select-none transition-[width,padding] duration-200 ease-out ${
          isCollapsed ? 'w-[68px] p-2.5 items-center' : 'w-60 p-3.5'
        }`}
      >
        <SidebarContent
          isCollapsed={isCollapsed}
          onToggleCollapse={toggleCollapse}
          onLogout={() => logout()}
        />
      </aside>

      {/* Mobile Slide-Over Navigation Drawer (< md) */}
      {mobileOpen && (
        <div className="md:hidden fixed inset-0 z-50 flex" role="dialog" aria-modal="true" aria-label="เมนูนำทางมือถือ">
          <div
            aria-hidden="true"
            className="animate-fade-in fixed inset-0 bg-slate-900/35 backdrop-blur-sm"
            onClick={() => onCloseMobile?.()}
          />
          <aside className="animate-drawer-in relative z-10 flex w-72 max-w-[85vw] h-full flex-col border-r border-sky-100/90 bg-white/95 p-4 shadow-2xl backdrop-blur-xl overflow-y-auto custom-scrollbar select-none">
            <SidebarContent
              isCollapsed={false}
              isMobile
              onCloseMobile={onCloseMobile}
              onLogout={() => logout()}
            />
          </aside>
        </div>
      )}
    </>
  )
}
