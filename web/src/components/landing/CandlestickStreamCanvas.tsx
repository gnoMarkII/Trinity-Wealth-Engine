import { useEffect, useRef } from 'react'
import { useReducedMotion } from 'motion/react'
import { usePageVisibility } from '../../hooks/usePageVisibility'

type StreamCandle = {
  height: number
  lane: number
  offset: number
  rising: boolean
  speed: number
  width: number
}

const CANDLE_COUNT = 44
const MAX_PIXEL_RATIO = 1.5

function createCandles(): StreamCandle[] {
  let seed = 739391
  const random = () => {
    seed = (seed * 16807) % 2147483647
    return (seed - 1) / 2147483646
  }

  return Array.from({ length: CANDLE_COUNT }, (_, index) => ({
    height: 12 + random() * 38,
    lane: index % 4,
    offset: random(),
    rising: random() > 0.36,
    speed: 0.028 + random() * 0.036,
    width: 4 + random() * 7,
  }))
}

function drawStream(
  context: CanvasRenderingContext2D,
  candles: StreamCandle[],
  width: number,
  height: number,
  elapsed: number,
) {
  context.clearRect(0, 0, width, height)

  const wash = context.createRadialGradient(width * 0.82, height * 0.38, 12, width * 0.82, height * 0.38, width * 0.72)
  wash.addColorStop(0, 'rgba(245, 158, 11, 0.16)')
  wash.addColorStop(0.4, 'rgba(16, 185, 129, 0.08)')
  wash.addColorStop(1, 'rgba(8, 9, 13, 0)')
  context.fillStyle = wash
  context.fillRect(0, 0, width, height)

  candles.forEach((candle, index) => {
    const progress = (candle.offset + elapsed * candle.speed) % 1
    const direction = candle.lane % 2 === 0 ? 1 : -1
    const x = direction === 1 ? progress * (width + 130) - 65 : width - progress * (width + 130) + 65
    const arc = Math.sin(progress * Math.PI) * (42 + candle.lane * 9)
    const y = height * (0.18 + candle.lane * 0.18) + arc + Math.sin(index * 1.4 + elapsed * 0.8) * 8
    const scale = 0.54 + progress * 0.7
    const candleHeight = candle.height * scale
    const candleWidth = candle.width * scale
    const color = candle.rising ? '#34d399' : '#fb4f61'

    context.save()
    context.globalAlpha = 0.22 + scale * 0.44
    context.translate(x, y)
    context.rotate(direction === 1 ? -0.26 : 0.26)
    context.shadowBlur = 16 * scale
    context.shadowColor = color
    context.strokeStyle = color
    context.fillStyle = color
    context.lineWidth = Math.max(1, scale)
    context.beginPath()
    context.moveTo(0, -candleHeight * 0.78)
    context.lineTo(0, candleHeight * 0.78)
    context.stroke()
    context.fillRect(-candleWidth / 2, -candleHeight / 2, candleWidth, candleHeight)
    context.restore()
  })
}

export default function CandlestickStreamCanvas() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const shouldReduceMotion = useReducedMotion()
  const isPageVisible = usePageVisibility()
  const shouldAnimate = !shouldReduceMotion && isPageVisible

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const context = canvas.getContext('2d')
    if (!context) return

    const candles = createCandles()
    let animationFrame: number | undefined
    let width = 1
    let height = 1
    let pixelRatio = 1
    let startTime = performance.now()

    const paint = (timestamp: number) => {
      drawStream(context, candles, width, height, (timestamp - startTime) / 1000)
    }

    const animate = (timestamp: number) => {
      paint(timestamp)
      animationFrame = requestAnimationFrame(animate)
    }

    const resize = () => {
      const bounds = canvas.getBoundingClientRect()
      width = Math.max(1, bounds.width)
      height = Math.max(1, bounds.height)
      pixelRatio = Math.min(window.devicePixelRatio || 1, MAX_PIXEL_RATIO)
      canvas.width = Math.round(width * pixelRatio)
      canvas.height = Math.round(height * pixelRatio)
      context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0)
      startTime = performance.now()
      paint(startTime)
    }

    const observer = new ResizeObserver(resize)
    observer.observe(canvas)
    resize()

    if (shouldAnimate) animationFrame = requestAnimationFrame(animate)

    return () => {
      observer.disconnect()
      if (animationFrame) cancelAnimationFrame(animationFrame)
    }
  }, [shouldAnimate])

  return <canvas ref={canvasRef} className="pointer-events-none absolute inset-0 h-full w-full mix-blend-screen" aria-hidden="true" />
}
