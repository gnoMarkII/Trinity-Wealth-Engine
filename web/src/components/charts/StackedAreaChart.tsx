import React, { useState, useMemo } from 'react'

export interface StackedAreaPoint {
  date: string
  values_by_category: Record<string, number | null | undefined>
}

export interface StackedAreaCategory {
  key: string
  label: string
  color: string
}

export interface StackedAreaChartProps {
  data: StackedAreaPoint[]
  categories: StackedAreaCategory[]
  title?: string
  subtitle?: string
  unit?: string
  height?: number
  defaultMode?: 'absolute' | 'percentage'
  className?: string
}

export const StackedAreaChart: React.FC<StackedAreaChartProps> = ({
  data,
  categories,
  title,
  subtitle,
  unit = 'THB',
  height = 320,
  defaultMode = 'absolute',
  className = '',
}) => {
  const [mode, setMode] = useState<'absolute' | 'percentage'>(defaultMode)
  const [hoverIndex, setHoverIndex] = useState<number | null>(null)

  // Chart dimensions inside SVG viewbox (0 0 800 height)
  const width = 800
  const padLeft = 70
  const padRight = 30
  const padTop = 25
  const padBottom = 35
  const plotWidth = width - padLeft - padRight
  const plotHeight = height - padTop - padBottom

  // Valid dates with data
  const validData = useMemo(() => {
    return data.filter((d) => d && d.date)
  }, [data])

  // Compute stacks per date
  const processedStacks = useMemo(() => {
    if (validData.length === 0 || categories.length === 0) return []

    return validData.map((d) => {
      let total = 0
      const vals: Record<string, number> = {}
      let hasAnyValue = false

      for (const cat of categories) {
        const v = d.values_by_category?.[cat.key]
        if (typeof v === 'number' && !isNaN(v) && v >= 0) {
          vals[cat.key] = v
          total += v
          hasAnyValue = true
        } else {
          vals[cat.key] = 0
        }
      }

      return {
        date: d.date,
        total,
        vals,
        hasData: hasAnyValue,
      }
    })
  }, [validData, categories])

  const maxTotal = useMemo(() => {
    if (mode === 'percentage') return 100
    const m = Math.max(1, ...processedStacks.map((p) => p.total))
    return m * 1.05
  }, [processedStacks, mode])

  // Coordinates mapping
  const points = useMemo(() => {
    if (processedStacks.length === 0) return []
    const n = processedStacks.length

    return processedStacks.map((p, idx) => {
      const x = padLeft + (n > 1 ? (idx / (n - 1)) * plotWidth : plotWidth / 2)
      let runningSum = 0
      const ys: Record<string, number> = {}

      for (const cat of categories) {
        const raw = p.vals[cat.key] || 0
        const valToStack = mode === 'percentage' ? (p.total > 0 ? (raw / p.total) * 100 : 0) : raw
        runningSum += valToStack
        const yFrac = Math.min(1, runningSum / maxTotal)
        ys[cat.key] = padTop + plotHeight - yFrac * plotHeight
      }

      return {
        date: p.date,
        x,
        ys,
        total: p.total,
        hasData: p.hasData,
        rawVals: p.vals,
      }
    })
  }, [processedStacks, categories, plotWidth, plotHeight, maxTotal, mode, padLeft, padTop])

  // Build SVG path per category layer
  const categoryPaths = useMemo(() => {
    if (points.length < 2) return []

    return categories.map((cat, catIdx) => {
      const prevCat = catIdx > 0 ? categories[catIdx - 1] : undefined
      const baselineKey = prevCat ? prevCat.key : null
      let topPath = ''
      let bottomPath = ''

      for (let i = 0; i < points.length; i++) {
        const pt = points[i]
        if (!pt) continue
        const yTop = pt.ys[cat.key] ?? (padTop + plotHeight)
        if (i === 0) {
          topPath = `M ${pt.x},${yTop}`
        } else {
          topPath += ` L ${pt.x},${yTop}`
        }
      }

      for (let i = points.length - 1; i >= 0; i--) {
        const pt = points[i]
        if (!pt) continue
        const yBottom = baselineKey ? (pt.ys[baselineKey] ?? (padTop + plotHeight)) : padTop + plotHeight
        bottomPath += ` L ${pt.x},${yBottom}`
      }

      return {
        key: cat.key,
        color: cat.color,
        label: cat.label,
        d: `${topPath} ${bottomPath} Z`,
      }
    })
  }, [points, categories, padTop, plotHeight])

  if (data.length === 0) {
    return (
      <div className={`rounded-xl border border-slate-800 bg-slate-900/80 p-6 text-center ${className}`}>
        {title && <h4 className="text-sm font-semibold text-slate-300">{title}</h4>}
        <p className="mt-4 text-xs text-slate-500">No historical series data available.</p>
      </div>
    )
  }

  const activePoint = hoverIndex !== null && points[hoverIndex] ? points[hoverIndex] : null

  return (
    <div className={`relative rounded-xl border border-slate-800 bg-slate-900/90 p-5 shadow-lg backdrop-blur-sm ${className}`}>
      <div className="mb-4 flex flex-wrap items-center justify-between gap-3 border-b border-slate-800/80 pb-3">
        <div>
          {title && <h3 className="text-base font-semibold text-slate-100">{title}</h3>}
          {subtitle && <p className="text-xs text-slate-400">{subtitle}</p>}
        </div>

        <div className="flex items-center gap-2">
          <div className="inline-flex rounded-lg border border-slate-800 bg-slate-950 p-1">
            <button
              type="button"
              className={`rounded px-2.5 py-1 text-xs font-medium transition-colors ${
                mode === 'absolute' ? 'bg-slate-800 text-slate-100' : 'text-slate-400 hover:text-slate-200'
              }`}
              onClick={() => setMode('absolute')}
            >
              {unit}
            </button>
            <button
              type="button"
              className={`rounded px-2.5 py-1 text-xs font-medium transition-colors ${
                mode === 'percentage' ? 'bg-slate-800 text-slate-100' : 'text-slate-400 hover:text-slate-200'
              }`}
              onClick={() => setMode('percentage')}
            >
              100% Share
            </button>
          </div>
        </div>
      </div>

      <div className="mb-3 flex flex-wrap items-center gap-4 text-xs">
        {categories.map((cat) => (
          <div key={cat.key} className="flex items-center gap-1.5">
            <span className="h-2.5 w-2.5 rounded-full" style={{ backgroundColor: cat.color }} />
            <span className="text-slate-300">{cat.label}</span>
          </div>
        ))}
      </div>

      <div className="relative w-full overflow-hidden rounded-lg border border-slate-800 bg-slate-950">
        <svg
          viewBox={`0 0 ${width} ${height}`}
          className="h-full w-full select-none"
          preserveAspectRatio="none"
          onMouseLeave={() => setHoverIndex(null)}
          role="figure"
          aria-label={title || 'Stacked Area Chart'}
        >
          {/* Y Axis Grid Lines */}
          {[0, 0.25, 0.5, 0.75, 1].map((frac, idx) => {
            const y = padTop + plotHeight * (1 - frac)
            const labelVal = mode === 'percentage' ? `${(frac * 100).toFixed(0)}%` : (maxTotal * frac).toLocaleString(undefined, { maximumFractionDigits: 0 })
            return (
              <g key={idx}>
                <line x1={padLeft} y1={y} x2={width - padRight} y2={y} stroke="#1e293b" strokeDasharray="3 3" />
                <text x={padLeft - 8} y={y + 4} textAnchor="end" fontSize="10" fill="#64748b">
                  {labelVal}
                </text>
              </g>
            )
          })}

          {/* Area Layers */}
          {categoryPaths.map((layer) => (
            <path
              key={layer.key}
              d={layer.d}
              fill={layer.color}
              fillOpacity={0.65}
              stroke={layer.color}
              strokeWidth={1.5}
            />
          ))}

          {/* X Axis Dates */}
          {points.length > 0 && points[0] && (
            <>
              <text x={points[0].x} y={height - 10} fontSize="10" fill="#64748b" textAnchor="start">
                {points[0].date}
              </text>
              {points.length > 1 && points[points.length - 1] && (
                <text
                  x={points[points.length - 1]!.x}
                  y={height - 10}
                  fontSize="10"
                  fill="#64748b"
                  textAnchor="end"
                >
                  {points[points.length - 1]!.date}
                </text>
              )}
            </>
          )}

          {/* Hover Crosshair */}
          {activePoint && (
            <line
              x1={activePoint.x}
              y1={padTop}
              x2={activePoint.x}
              y2={padTop + plotHeight}
              stroke="#cbd5e1"
              strokeDasharray="2 2"
              strokeWidth={1.5}
            />
          )}

          {/* Invisible interactive columns for tooltip tracking */}
          {points.map((pt, idx) => {
            const colWidth = plotWidth / points.length
            return (
              <rect
                key={idx}
                x={pt.x - colWidth / 2}
                y={padTop}
                width={colWidth}
                height={plotHeight}
                fill="transparent"
                onMouseEnter={() => setHoverIndex(idx)}
              />
            )
          })}
        </svg>

        {activePoint && (
          <div
            className="pointer-events-none absolute z-20 rounded-lg border border-slate-700 bg-slate-900/95 p-3 text-xs shadow-xl backdrop-blur-md"
            style={{
              left: Math.min(width - 180, Math.max(padLeft, activePoint.x)),
              top: 15,
            }}
          >
            <div className="border-b border-slate-800 pb-1.5 font-semibold text-slate-200">
              {activePoint.date}
            </div>
            <div className="mt-2 space-y-1">
              {categories.map((cat) => {
                const val = activePoint.rawVals[cat.key] || 0
                const share = activePoint.total > 0 ? (val / activePoint.total) * 100 : 0
                return (
                  <div key={cat.key} className="flex items-center justify-between gap-3 text-[11px]">
                    <span className="flex items-center gap-1.5 text-slate-300">
                      <span className="h-2 w-2 rounded-full" style={{ backgroundColor: cat.color }} />
                      {cat.label}:
                    </span>
                    <span className="font-mono text-slate-100">
                      {val.toLocaleString(undefined, { minimumFractionDigits: 1, maximumFractionDigits: 1 })}{' '}
                      <span className="text-slate-400">({share.toFixed(1)}%)</span>
                    </span>
                  </div>
                )
              })}
              <div className="mt-1.5 flex items-center justify-between border-t border-slate-800 pt-1 font-semibold text-slate-100">
                <span>Total:</span>
                <span className="font-mono text-emerald-400">
                  {activePoint.total.toLocaleString(undefined, { minimumFractionDigits: 1, maximumFractionDigits: 1 })} {unit}
                </span>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
