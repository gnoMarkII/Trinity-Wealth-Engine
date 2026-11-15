import React, { useState, useMemo } from 'react'

export interface BubbleItem {
  id: string
  label: string
  x: number
  y: number
  size: number
  group?: string
  color?: string
}

export interface BubbleChartProps {
  data: BubbleItem[]
  title?: string
  subtitle?: string
  xLabel: string
  yLabel: string
  sizeLabel: string
  xUnit?: string
  yUnit?: string
  sizeUnit?: string
  height?: number
  className?: string
  onBubbleClick?: (item: BubbleItem) => void
}

export const BubbleChart: React.FC<BubbleChartProps> = ({
  data,
  title,
  subtitle,
  xLabel,
  yLabel,
  sizeLabel,
  xUnit = '',
  yUnit = '',
  sizeUnit = '',
  height = 360,
  className = '',
  onBubbleClick,
}) => {
  const [hoveredBubble, setHoveredBubble] = useState<{ item: BubbleItem; cx: number; cy: number } | null>(null)

  const width = 800
  const padLeft = 70
  const padRight = 40
  const padTop = 30
  const padBottom = 45
  const plotWidth = width - padLeft - padRight
  const plotHeight = height - padTop - padBottom

  const validData = useMemo(() => {
    return data.filter((d) => typeof d.x === 'number' && typeof d.y === 'number' && d.size > 0)
  }, [data])

  const { xMin, xMax, yMin, yMax, sizeMin, sizeMax } = useMemo(() => {
    if (validData.length === 0) {
      return { xMin: 0, xMax: 100, yMin: 0, yMax: 100, sizeMin: 1, sizeMax: 10 }
    }
    const xs = validData.map((d) => d.x)
    const ys = validData.map((d) => d.y)
    const ss = validData.map((d) => d.size)

    const rawXMin = Math.min(...xs)
    const rawXMax = Math.max(...xs)
    const rawYMin = Math.min(...ys)
    const rawYMax = Math.max(...ys)

    const xSpan = rawXMax - rawXMin || 1
    const ySpan = rawYMax - rawYMin || 1

    return {
      xMin: rawXMin - xSpan * 0.1,
      xMax: rawXMax + xSpan * 0.1,
      yMin: rawYMin - ySpan * 0.1,
      yMax: rawYMax + ySpan * 0.1,
      sizeMin: Math.min(...ss),
      sizeMax: Math.max(...ss),
    }
  }, [validData])

  const bubbles = useMemo(() => {
    const minRadius = 8
    const maxRadius = 36

    return validData.map((d) => {
      const xFrac = (d.x - xMin) / (xMax - xMin || 1)
      const yFrac = (d.y - yMin) / (yMax - yMin || 1)

      const cx = padLeft + xFrac * plotWidth
      const cy = padTop + plotHeight - yFrac * plotHeight

      // Area-proportional radius scaling (radius proportional to sqrt(size))
      const sqrtMin = Math.sqrt(sizeMin)
      const sqrtMax = Math.sqrt(sizeMax)
      const sizeFrac = sqrtMax > sqrtMin ? (Math.sqrt(d.size) - sqrtMin) / (sqrtMax - sqrtMin) : 0.5
      const r = minRadius + sizeFrac * (maxRadius - minRadius)

      const color = d.color || '#38bdf8'

      return {
        item: d,
        cx,
        cy,
        r,
        color,
      }
    })
  }, [validData, xMin, xMax, yMin, yMax, sizeMin, sizeMax, plotWidth, plotHeight, padLeft, padTop])

  if (validData.length === 0) {
    return (
      <div className={`flow-panel rounded-2xl border border-edge bg-panel p-6 text-center shadow-sm ${className}`}>
        {title && <h4 className="text-sm font-semibold text-slate-800">{title}</h4>}
        <p className="mt-4 text-xs text-slate-500">No cross-sectional multi-dimensional data available.</p>
      </div>
    )
  }

  return (
    <div className={`relative flow-panel rounded-2xl border border-edge bg-panel p-5 shadow-sm backdrop-blur-md ${className}`}>
      <div className="mb-4 flex flex-wrap items-baseline justify-between gap-2 border-b border-edge pb-3">
        <div>
          {title && <h3 className="text-base font-semibold text-slate-900">{title}</h3>}
          {subtitle && <p className="text-xs text-slate-600">{subtitle}</p>}
        </div>
        <div className="text-xs text-slate-600">
          Axes: <span className="font-medium text-slate-900">{xLabel}</span> (X) vs <span className="font-medium text-slate-900">{yLabel}</span> (Y) • Size: <span className="font-medium text-slate-900">{sizeLabel}</span>
        </div>
      </div>

      <div className="relative w-full overflow-hidden rounded-xl border border-sky-100 bg-gradient-to-b from-slate-50/75 via-white to-sky-50/20 p-2 shadow-sm">
        <svg
          viewBox={`0 0 ${width} ${height}`}
          className="h-full w-full select-none"
          preserveAspectRatio="none"
          role="figure"
          aria-label={title || 'Cross-sectional Bubble Chart'}
        >
          {/* Horizontal Grid */}
          {[0, 0.25, 0.5, 0.75, 1].map((frac, idx) => {
            const y = padTop + plotHeight * (1 - frac)
            const val = yMin + frac * (yMax - yMin)
            return (
              <g key={`h-${idx}`}>
                <line x1={padLeft} y1={y} x2={width - padRight} y2={y} stroke="#e2e8f0" strokeDasharray="3 3" />
                <text x={padLeft - 8} y={y + 4} textAnchor="end" fontSize="10" fill="#475569">
                  {val.toFixed(1)} {yUnit}
                </text>
              </g>
            )
          })}

          {/* Vertical Grid */}
          {[0, 0.25, 0.5, 0.75, 1].map((frac, idx) => {
            const x = padLeft + frac * plotWidth
            const val = xMin + frac * (xMax - xMin)
            return (
              <g key={`v-${idx}`}>
                <line x1={x} y1={padTop} x2={x} y2={padTop + plotHeight} stroke="#e2e8f0" strokeDasharray="3 3" />
                <text x={x} y={height - padBottom + 16} textAnchor="middle" fontSize="10" fill="#475569">
                  {val.toFixed(1)} {xUnit}
                </text>
              </g>
            )
          })}

          {/* Bubbles */}
          {bubbles.map((b) => (
            <g
              key={b.item.id}
              className="cursor-pointer transition-transform duration-150 hover:scale-105"
              tabIndex={0}
              role="button"
              aria-label={`${b.item.label}: (${b.item.x} ${xUnit}, ${b.item.y} ${yUnit}) size: ${b.item.size} ${sizeUnit}`}
              onClick={() => onBubbleClick?.(b.item)}
              onMouseEnter={() => setHoveredBubble({ item: b.item, cx: b.cx, cy: b.cy })}
              onMouseLeave={() => setHoveredBubble(null)}
              onFocus={() => setHoveredBubble({ item: b.item, cx: b.cx, cy: b.cy })}
              onBlur={() => setHoveredBubble(null)}
            >
              <circle
                cx={b.cx}
                cy={b.cy}
                r={b.r}
                fill={b.color}
                fillOpacity={0.78}
                stroke={b.color}
                strokeWidth={2}
              />
              <text
                x={b.cx}
                y={b.cy + 4}
                textAnchor="middle"
                fontSize={b.r > 16 ? '11' : '9.5'}
                fill="#0f172a"
                fontWeight="700"
                stroke="#ffffff"
                strokeWidth="2.5"
                paintOrder="stroke"
                className="pointer-events-none"
              >
                {b.item.label}
              </text>
            </g>
          ))}
        </svg>

        {hoveredBubble && (
          <div
            className="pointer-events-none absolute z-20 -translate-x-1/2 -translate-y-full transform rounded-xl border border-sky-200 bg-white/95 p-3 text-xs shadow-lg backdrop-blur-md"
            style={{
              left: Math.min(width - 120, Math.max(padLeft + 60, hoveredBubble.cx)),
              top: Math.max(padTop + 40, hoveredBubble.cy - 12),
            }}
          >
            <div className="font-semibold text-slate-900">{hoveredBubble.item.label}</div>
            {hoveredBubble.item.group && (
              <div className="text-[11px] text-slate-500">{hoveredBubble.item.group}</div>
            )}
            <div className="mt-1.5 space-y-0.5 text-[11px]">
              <div className="text-slate-600">
                {xLabel}: <span className="font-mono font-semibold text-slate-900">{hoveredBubble.item.x} {xUnit}</span>
              </div>
              <div className="text-slate-600">
                {yLabel}: <span className="font-mono font-semibold text-slate-900">{hoveredBubble.item.y} {yUnit}</span>
              </div>
              <div className="text-slate-600">
                {sizeLabel}: <span className="font-mono font-semibold text-emerald-700">{hoveredBubble.item.size.toLocaleString()} {sizeUnit}</span>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
