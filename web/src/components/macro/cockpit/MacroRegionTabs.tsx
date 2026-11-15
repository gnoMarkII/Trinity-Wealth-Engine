import React from 'react'
import { CpuIcon, GlobeMacroIcon } from '../../ui/AppIcons'

export type MacroRegionTab = 'ai' | 'us' | 'th' | 'cross-border'

interface MacroRegionTabsProps {
  activeTab: MacroRegionTab
  onChange: (tab: MacroRegionTab) => void
  className?: string
}

const TABS: { key: MacroRegionTab; label: string; subtitle: string }[] = [
  {
    key: 'ai',
    label: 'บทวิเคราะห์ AI (AI Analysis)',
    subtitle: 'Regime Scenarios, 5D Evidence, Allocation & Trades',
  },
  {
    key: 'us',
    label: 'สหรัฐอเมริกา (US)',
    subtitle: 'Sector Rotation, Yield Curve, Financial Stress & Debt',
  },
  {
    key: 'th',
    label: 'ประเทศไทย (TH)',
    subtitle: 'SET Investor Flow, Breadth, Valuation & Retail Gold',
  },
  {
    key: 'cross-border',
    label: 'ความเชื่อมโยง US–TH',
    subtitle: 'Fed vs BoT Policy Rates, Spreads & Transmission',
  },
]

function renderTabMarker(key: MacroRegionTab): React.ReactNode {
  switch (key) {
    case 'ai':
      return (
        <span className="inline-flex h-5 w-5 items-center justify-center rounded-md bg-blue-50 text-section-accent border border-blue-200/80 shrink-0">
          <CpuIcon className="h-3.5 w-3.5" />
        </span>
      )
    case 'us':
      return (
        <span className="inline-flex h-5 items-center rounded-md border border-sky-200 bg-sky-50 px-1.5 font-mono text-[11px] font-bold text-sky-800 shrink-0">
          US
        </span>
      )
    case 'th':
      return (
        <span className="inline-flex h-5 items-center rounded-md border border-emerald-200 bg-emerald-50 px-1.5 font-mono text-[11px] font-bold text-emerald-800 shrink-0">
          TH
        </span>
      )
    case 'cross-border':
      return (
        <span className="inline-flex h-5 w-5 items-center justify-center rounded-md bg-sky-50 text-sky-700 border border-sky-200/80 shrink-0">
          <GlobeMacroIcon className="h-3.5 w-3.5" />
        </span>
      )
  }
}

export const MacroRegionTabs: React.FC<MacroRegionTabsProps> = ({
  activeTab,
  onChange,
  className = '',
}) => {
  return (
    <div className={`space-y-2 ${className}`}>
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-2 p-1.5 rounded-2xl bg-surface-1 border border-edge backdrop-blur-md shadow-inner">
        {TABS.map((tab) => {
          const isActive = activeTab === tab.key
          return (
            <button
              key={tab.key}
              type="button"
              onClick={() => onChange(tab.key)}
              aria-pressed={isActive}
              className={`relative overflow-hidden flex flex-col items-start justify-center rounded-xl px-4 py-3 text-left transition-all focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-section-accent/45 ${
                isActive
                  ? 'bg-white text-slate-900 shadow-sm ring-1 ring-blue-900/10'
                  : 'text-slate-600 hover:bg-white/70 hover:text-slate-900'
              }`}
            >
              {isActive && (
                <span className="absolute inset-x-0 top-0 h-0.5 bg-section-accent" aria-hidden="true" />
              )}
              <div className="flex items-center gap-2 font-semibold text-sm tracking-tight w-full">
                {renderTabMarker(tab.key)}
                <span className="truncate">{tab.label}</span>
                {isActive && (
                  <span className="ml-auto inline-block h-2 w-2 shrink-0 rounded-full bg-section-accent" />
                )}
              </div>
              <p className="mt-0.5 text-xs text-slate-500 line-clamp-1">{tab.subtitle}</p>
            </button>
          )
        })}
      </div>
    </div>
  )
}
