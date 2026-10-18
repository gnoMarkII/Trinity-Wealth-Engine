import React from 'react'

export const TopLoadingRibbon: React.FC<{ label?: string }> = ({ label }) => (
  <div className="w-full space-y-1.5" role="status" aria-live="polite">
    <div className="h-1 w-full overflow-hidden rounded-full bg-sky-100/80">
      <div className="h-full w-2/5 rounded-full bg-gradient-to-r from-sky-400 via-cyan-500 to-blue-600 animate-progress-ribbon" />
    </div>
    {label && (
      <div className="flex items-center gap-2 text-xs font-medium text-sky-700/90">
        <span className="inline-block h-2 w-2 rounded-full bg-sky-500 animate-ping" />
        <span>{label}</span>
      </div>
    )}
  </div>
)

export const EquityDetailSkeleton: React.FC<{ label?: string }> = ({
  label = 'กำลังโหลดบทวิเคราะห์หุ้น...',
}) => (
  <div className="space-y-6 animate-fade-in" aria-live="polite">
    <TopLoadingRibbon label={label} />

    {/* Tier 1 Executive Thesis Hero Skeleton */}
    <div className="flow-panel rounded-2xl border border-edge/80 p-6 space-y-4">
      <div className="flex items-center justify-between">
        <div className="h-5 w-48 rounded-lg animate-shimmer" />
        <div className="h-6 w-28 rounded-full animate-shimmer" />
      </div>
      <div className="space-y-2.5">
        <div className="h-4 w-full rounded animate-shimmer" />
        <div className="h-4 w-11/12 rounded animate-shimmer" />
        <div className="h-4 w-3/4 rounded animate-shimmer" />
      </div>
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 pt-2">
        {Array.from({ length: 4 }).map((_, idx) => (
          <div key={idx} className="h-16 rounded-xl border border-edge/60 animate-shimmer" />
        ))}
      </div>
    </div>

    {/* Tier 2 2-Column Analytical Grid Skeleton */}
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      <div className="flow-panel rounded-2xl border border-edge/80 p-6 space-y-4">
        <div className="h-5 w-44 rounded-lg animate-shimmer" />
        <div className="h-32 w-full rounded-xl animate-shimmer" />
        <div className="grid grid-cols-2 gap-3">
          <div className="h-14 rounded-xl animate-shimmer" />
          <div className="h-14 rounded-xl animate-shimmer" />
        </div>
      </div>
      <div className="flow-panel rounded-2xl border border-edge/80 p-6 space-y-4">
        <div className="h-5 w-44 rounded-lg animate-shimmer" />
        <div className="h-32 w-full rounded-xl animate-shimmer" />
        <div className="grid grid-cols-2 gap-3">
          <div className="h-14 rounded-xl animate-shimmer" />
          <div className="h-14 rounded-xl animate-shimmer" />
        </div>
      </div>
    </div>
  </div>
)

export const NewsCardListSkeleton: React.FC<{
  count?: number
  label?: string
}> = ({ count = 3, label }) => (
  <div className="space-y-4 animate-fade-in" aria-live="polite">
    {label && <TopLoadingRibbon label={label} />}
    <div className="grid grid-cols-1 gap-4">
      {Array.from({ length: count }).map((_, idx) => (
        <div
          key={idx}
          className="rounded-xl border border-edge bg-panel p-4 shadow-sm space-y-3"
        >
          <div className="flex items-start justify-between gap-3">
            <div className="h-5 w-3/4 rounded-lg animate-shimmer" />
            <div className="h-4 w-16 rounded animate-shimmer shrink-0" />
          </div>
          <div className="h-3.5 w-full rounded animate-shimmer" />
          <div className="h-3.5 w-2/3 rounded animate-shimmer" />
          <div className="flex items-center justify-between pt-2 border-t border-edge/60">
            <div className="h-3 w-32 rounded animate-shimmer" />
            <div className="h-3 w-24 rounded animate-shimmer" />
          </div>
        </div>
      ))}
    </div>
  </div>
)

export const PortfolioSummaryCardsSkeleton: React.FC = () => (
  <div className="grid grid-cols-1 gap-3.5 sm:grid-cols-2 xl:grid-cols-4 animate-fade-in" aria-live="polite">
    {Array.from({ length: 4 }).map((_, idx) => (
      <div
        key={idx}
        className="flow-panel relative overflow-hidden rounded-2xl border border-sky-100 bg-panel/75 p-5 shadow-xs space-y-3.5 animate-pulse"
      >
        <div className="flex items-center justify-between">
          <div className="h-4 w-28 rounded-md animate-shimmer" />
          <div className="h-4 w-4 rounded-full animate-shimmer" />
        </div>
        <div className="space-y-1.5">
          <div className="h-7 w-36 rounded-lg animate-shimmer" />
          <div className="h-3.5 w-24 rounded animate-shimmer" />
        </div>
        <div className="flex items-center justify-between pt-1 border-t border-sky-100/50">
          <div className="h-3 w-20 rounded animate-shimmer" />
          <div className="h-3 w-16 rounded animate-shimmer" />
        </div>
      </div>
    ))}
  </div>
)

export const PortfolioOverviewSkeleton: React.FC = () => (
  <div className="space-y-6 animate-fade-in animate-pulse" aria-live="polite">
    {/* Policy Lineage Banner Skeleton */}
    <div className="rounded-2xl border border-sky-100 bg-sky-50/30 p-5 shadow-xs flex items-center justify-between gap-4">
      <div className="flex items-center gap-3">
        <div className="h-10 w-10 rounded-xl animate-shimmer" />
        <div className="space-y-1.5">
          <div className="h-4 w-48 rounded animate-shimmer" />
          <div className="h-3 w-64 rounded animate-shimmer" />
        </div>
      </div>
      <div className="h-8 w-28 rounded-xl animate-shimmer hidden sm:block" />
    </div>

    {/* Donut Chart + Table 2-Col Grid Skeleton */}
    <div className="grid grid-cols-1 gap-6 2xl:grid-cols-12">
      {/* Concentric Double-Ring Donut Chart Placeholder */}
      <div className="rounded-2xl border border-sky-100 bg-panel p-6 shadow-sm 2xl:col-span-5 flex flex-col items-center justify-center min-h-[380px] space-y-4">
        <div className="flex items-center justify-between w-full">
          <div className="h-5 w-40 rounded animate-shimmer" />
          <div className="h-4 w-24 rounded animate-shimmer" />
        </div>
        <div className="relative flex items-center justify-center my-4">
          <div className="h-52 w-52 rounded-full border-[12px] border-sky-100/80 animate-shimmer" />
          <div className="absolute h-32 w-32 rounded-full border-[10px] border-blue-50/80 animate-shimmer" />
        </div>
        <div className="grid grid-cols-2 gap-2 w-full pt-2">
          {Array.from({ length: 4 }).map((_, i) => (
            <div key={i} className="h-7 rounded-lg animate-shimmer" />
          ))}
        </div>
      </div>

      {/* Target vs Actual Table Skeleton */}
      <div className="rounded-2xl border border-sky-100 bg-panel p-6 shadow-sm 2xl:col-span-7 space-y-4">
        <div className="flex items-center justify-between">
          <div className="h-5 w-48 rounded animate-shimmer" />
          <div className="h-8 w-32 rounded-xl animate-shimmer" />
        </div>
        <div className="space-y-2.5 pt-2">
          {Array.from({ length: 5 }).map((_, i) => (
            <div key={i} className="h-12 w-full rounded-xl border border-sky-50 animate-shimmer" />
          ))}
        </div>
      </div>
    </div>
  </div>
)

export const PortfolioHoldingsSkeleton: React.FC = () => (
  <div className="space-y-4 animate-fade-in animate-pulse" aria-live="polite">
    <div className="rounded-2xl border border-sky-100 bg-panel shadow-sm overflow-hidden">
      <div className="border-b border-sky-100 bg-sky-50/40 px-6 py-4 flex items-center justify-between">
        <div className="space-y-1">
          <div className="h-5 w-44 rounded animate-shimmer" />
          <div className="h-3 w-72 rounded animate-shimmer" />
        </div>
        <div className="h-8 w-24 rounded-xl animate-shimmer" />
      </div>
      <div className="p-4 space-y-3">
        {Array.from({ length: 6 }).map((_, idx) => (
          <div
            key={idx}
            className="flex items-center justify-between p-3.5 rounded-xl border border-sky-100/60 bg-white/60 space-x-4"
          >
            <div className="flex items-center gap-3 w-1/4">
              <div className="h-9 w-9 rounded-xl animate-shimmer shrink-0" />
              <div className="space-y-1 flex-1">
                <div className="h-4 w-16 rounded animate-shimmer" />
                <div className="h-3 w-28 rounded animate-shimmer" />
              </div>
            </div>
            <div className="h-6 w-24 rounded-full animate-shimmer hidden sm:block" />
            <div className="space-y-1 text-right w-1/5">
              <div className="h-4 w-20 rounded animate-shimmer ml-auto" />
              <div className="h-3 w-14 rounded animate-shimmer ml-auto" />
            </div>
            <div className="space-y-1 text-right w-1/5">
              <div className="h-4 w-24 rounded animate-shimmer ml-auto" />
              <div className="h-3 w-16 rounded animate-shimmer ml-auto" />
            </div>
            <div className="h-7 w-20 rounded-lg animate-shimmer shrink-0 hidden md:block" />
          </div>
        ))}
      </div>
    </div>
  </div>
)
