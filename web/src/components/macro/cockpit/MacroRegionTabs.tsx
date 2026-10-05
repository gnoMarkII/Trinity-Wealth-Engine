import React from 'react'

export type MacroRegionTab = 'ai' | 'us' | 'th' | 'cross-border'

interface MacroRegionTabsProps {
  activeTab: MacroRegionTab
  onChange: (tab: MacroRegionTab) => void
  className?: string
}

const TABS: { key: MacroRegionTab; label: string; flag: string; subtitle: string }[] = [
  {
    key: 'ai',
    label: 'บทวิเคราะห์ AI (AI Analysis)',
    flag: '🧠',
    subtitle: 'Regime Scenarios, 5D Evidence, Allocation & Trades',
  },
  {
    key: 'us',
    label: 'สหรัฐอเมริกา (US)',
    flag: '🇺🇸',
    subtitle: 'Sector Rotation, Yield Curve, Financial Stress & Debt',
  },
  {
    key: 'th',
    label: 'ประเทศไทย (TH)',
    flag: '🇹🇭',
    subtitle: 'SET Investor Flow, Breadth, Valuation & Retail Gold',
  },
  {
    key: 'cross-border',
    label: 'ความเชื่อมโยง US–TH',
    flag: '🌐',
    subtitle: 'Fed vs BoT Policy Rates, Spreads & Transmission',
  },
]

export const MacroRegionTabs: React.FC<MacroRegionTabsProps> = ({
  activeTab,
  onChange,
  className = '',
}) => {
  return (
    <div className={`space-y-2 ${className}`}>
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-2 p-1.5 rounded-2xl bg-slate-100/90 border border-slate-200/80 shadow-inner">
        {TABS.map((tab) => {
          const isActive = activeTab === tab.key
          return (
            <button
              key={tab.key}
              type="button"
              onClick={() => onChange(tab.key)}
              aria-pressed={isActive}
              className={`flex flex-col items-start justify-center rounded-xl px-4 py-3 text-left transition-all ${
                isActive
                  ? 'bg-white text-zinc-900 shadow-md ring-1 ring-black/5'
                  : 'text-zinc-600 hover:bg-white/60 hover:text-zinc-900'
              }`}
            >
              <div className="flex items-center gap-2 font-bold text-sm tracking-tight w-full">
                <span className="text-base">{tab.flag}</span>
                <span className="truncate">{tab.label}</span>
                {isActive && (
                  <span className="ml-auto inline-block h-2 w-2 shrink-0 rounded-full bg-sky-600" />
                )}
              </div>
              <p className="mt-0.5 text-[11px] text-zinc-400 line-clamp-1">{tab.subtitle}</p>
            </button>
          )
        })}
      </div>
    </div>
  )
}
