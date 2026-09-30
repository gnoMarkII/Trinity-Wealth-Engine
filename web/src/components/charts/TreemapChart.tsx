import React, { useState, useMemo } from 'react'

export interface TreemapItem {
  id?: string
  label: string
  value: number
  group?: string
  return_pct?: number
  color?: string
}

export interface TreemapChartProps {
  items: TreemapItem[]
  title?: string
  subtitle?: string
  unit?: string
  height?: number
  className?: string
  onItemClick?: (item: TreemapItem) => void
}

interface Rect {
  x: number
  y: number
  w: number
  h: number
  item: TreemapItem
}

// Slice-and-dice hierarchical tiling with alternating split direction
function computeTreemapLayout(
  items: TreemapItem[],
  x: number,
  y: number,
  w: number,
  h: number
): Rect[] {
  if (items.length === 0 || w <= 0 || h <= 0) return []

  const total = items.reduce((sum, it) => sum + Math.max(0, it.value), 0)
  if (total <= 0) return []

  if (items.length === 1 && items[0]) {
    return [{ x, y, w, h, item: items[0] }]
  }

  // Find partition closest to half total weight
  const half = total / 2
  let running = 0
  let splitIdx = 1

  for (let i = 0; i < items.length - 1; i++) {
    const it = items[i]
    if (it) {
      running += Math.max(0, it.value)
    }
    if (running >= half) {
      splitIdx = i + 1
      break
    }
  }

  const groupA = items.slice(0, splitIdx)
  const groupB = items.slice(splitIdx)
  const sumA = groupA.reduce((sum, it) => sum + Math.max(0, it.value), 0)
  const ratioA = sumA / total

  if (w >= h) {
    // Split vertically (columns)
    const wA = w * ratioA
    const wB = w - wA
    return [
      ...computeTreemapLayout(groupA, x, y, wA, h),
      ...computeTreemapLayout(groupB, x + wA, y, wB, h),
    ]
  } else {
    // Split horizontally (rows)
    const hA = h * ratioA
    const hB = h - hA
    return [
      ...computeTreemapLayout(groupA, x, y, w, hA),
      ...computeTreemapLayout(groupB, x, y + hA, w, hB),
    ]
  }
}

const DEFAULT_GROUP_COLORS: Record<string, string> = {
  Cash: '#3b82f6', // blue
  Equity: '#10b981', // emerald
  Stock: '#10b981',
  Bond: '#f59e0b', // amber
  Fixed_Income: '#f59e0b',
  Gold: '#eab308', // yellow
  Commodity: '#eab308',
  Alternative: '#8b5cf6', // purple
  Crypto: '#ec4899', // pink
}

export const TreemapChart: React.FC<TreemapChartProps> = ({
  items,
  title,
  subtitle,
  unit = 'THB',
  height = 360,
  className = '',
  onItemClick,
}) => {
  const [hoveredItem, setHoveredItem] = useState<{ item: TreemapItem; x: number; y: number } | null>(null)

  const validItems = useMemo(() => {
    return items
      .filter((it) => it.value > 0)
      .sort((a, b) => b.value - a.value)
  }, [items])

  const totalValue = useMemo(() => {
    return validItems.reduce((acc, it) => acc + it.value, 0)
  }, [validItems])

  const layout = useMemo(() => {
    // Fixed coordinate space for SVG viewbox
    const vWidth = 800
    const vHeight = height
    return computeTreemapLayout(validItems, 0, 0, vWidth, vHeight)
  }, [validItems, height])

  const getColor = (item: TreemapItem, index: number): string => {
    if (item.color) return item.color
    if (item.return_pct !== undefined && item.return_pct !== null) {
      if (item.return_pct > 0.05) return '#059669' // emerald-600
      if (item.return_pct > 0) return '#10b981' // emerald-500
      if (item.return_pct < -0.05) return '#dc2626' // red-600
      if (item.return_pct < 0) return '#ef4444' // red-500
      return '#475569' // slate-600
    }
    if (item.group && DEFAULT_GROUP_COLORS[item.group]) {
      return DEFAULT_GROUP_COLORS[item.group]!
    }
    const palette = ['#3b82f6', '#10b981', '#f59e0b', '#8b5cf6', '#06b6d4', '#ec4899', '#6366f1']
    return palette[index % palette.length] ?? '#3b82f6'
  }

  if (validItems.length === 0) {
    return (
      <div className={`rounded-xl border border-slate-800 bg-slate-900/80 p-6 text-center ${className}`}>
        {title && <h4 className="text-sm font-semibold text-slate-300">{title}</h4>}
        <p className="mt-4 text-xs text-slate-500">No holdings or asset-class breakdown available to render treemap.</p>
      </div>
    )
  }

  return (
    <div className={`relative rounded-xl border border-slate-800 bg-slate-900/90 p-5 shadow-lg backdrop-blur-sm ${className}`}>
      <div className="mb-4 flex flex-wrap items-baseline justify-between gap-2 border-b border-slate-800/80 pb-3">
        <div>
          {title && <h3 className="text-base font-semibold text-slate-100">{title}</h3>}
          {subtitle && <p className="text-xs text-slate-400">{subtitle}</p>}
        </div>
        <div className="text-right">
          <span className="text-xs text-slate-400">Total Value: </span>
          <span className="text-sm font-bold text-slate-100">
            {totalValue.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })} {unit}
          </span>
        </div>
      </div>

      <div className="relative w-full overflow-hidden rounded-lg border border-slate-800 bg-slate-950">
        <svg
          viewBox={`0 0 800 ${height}`}
          className="h-full w-full select-none"
          preserveAspectRatio="none"
          role="figure"
          aria-label={title || 'Asset Allocation Treemap'}
        >
          {layout.map((rect, idx) => {
            const pct = totalValue > 0 ? (rect.item.value / totalValue) * 100 : 0
            const fillColor = getColor(rect.item, idx)
            const showDetails = rect.w > 70 && rect.h > 45
            const showLabelOnly = rect.w > 40 && rect.h > 25

            return (
              <g
                key={rect.item.id || `${rect.item.label}-${idx}`}
                className="cursor-pointer transition-opacity duration-150 hover:opacity-90 focus:outline-none"
                tabIndex={0}
                role="button"
                aria-label={`${rect.item.label}: ${rect.item.value.toLocaleString()} ${unit} (${pct.toFixed(1)}%)`}
                onClick={() => onItemClick?.(rect.item)}
                onMouseEnter={(e) => {
                  const bounds = e.currentTarget.getBoundingClientRect()
                  setHoveredItem({ item: rect.item, x: bounds.left + bounds.width / 2, y: bounds.top })
                }}
                onMouseLeave={() => setHoveredItem(null)}
                onFocus={(e) => {
                  const bounds = e.currentTarget.getBoundingClientRect()
                  setHoveredItem({ item: rect.item, x: bounds.left + bounds.width / 2, y: bounds.top })
                }}
                onBlur={() => setHoveredItem(null)}
              >
                <rect
                  x={rect.x + 1}
                  y={rect.y + 1}
                  width={Math.max(0, rect.w - 2)}
                  height={Math.max(0, rect.h - 2)}
                  fill={fillColor}
                  fillOpacity={0.85}
                  stroke="#0f172a"
                  strokeWidth={2}
                  rx={4}
                />
                {showDetails ? (
                  <>
                    <text
                      x={rect.x + 8}
                      y={rect.y + 20}
                      fill="#ffffff"
                      fontSize="12"
                      fontWeight="600"
                      className="pointer-events-none drop-shadow-sm"
                    >
                      {rect.item.label}
                    </text>
                    <text
                      x={rect.x + 8}
                      y={rect.y + 36}
                      fill="#e2e8f0"
                      fontSize="11"
                      className="pointer-events-none opacity-90"
                    >
                      {pct.toFixed(1)}%
                    </text>
                  </>
                ) : showLabelOnly ? (
                  <text
                    x={rect.x + 4}
                    y={rect.y + 16}
                    fill="#ffffff"
                    fontSize="10"
                    fontWeight="500"
                    className="pointer-events-none"
                  >
                    {rect.item.label.slice(0, 8)}
                  </text>
                ) : null}
              </g>
            )
          })}
        </svg>

        {hoveredItem && (
          <div
            className="pointer-events-none fixed z-50 -translate-x-1/2 -translate-y-full transform rounded-lg border border-slate-700 bg-slate-900/95 px-3 py-2 text-xs shadow-xl backdrop-blur-md"
            style={{ left: hoveredItem.x, top: hoveredItem.y - 8 }}
          >
            <div className="font-semibold text-slate-100">{hoveredItem.item.label}</div>
            {hoveredItem.item.group && (
              <div className="text-[11px] text-slate-400">Class: {hoveredItem.item.group}</div>
            )}
            <div className="mt-1 font-mono text-emerald-400">
              {hoveredItem.item.value.toLocaleString(undefined, { minimumFractionDigits: 2 })} {unit}
            </div>
            <div className="text-[11px] text-slate-400">
              Share: {((hoveredItem.item.value / totalValue) * 100).toFixed(2)}%
            </div>
            {hoveredItem.item.return_pct !== undefined && hoveredItem.item.return_pct !== null && (
              <div
                className={`text-[11px] font-semibold ${
                  hoveredItem.item.return_pct >= 0 ? 'text-emerald-400' : 'text-red-400'
                }`}
              >
                Return: {hoveredItem.item.return_pct >= 0 ? '+' : ''}
                {(hoveredItem.item.return_pct * 100).toFixed(2)}%
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
