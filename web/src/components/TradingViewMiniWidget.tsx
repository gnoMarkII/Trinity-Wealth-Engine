import { useEffect, useRef } from 'react'

interface Props {
  symbol: string // เช่น "AMEX:SPY", "NASDAQ:QQQ", "TVC:VIX", "TVC:US10Y"
  title?: string
}

/**
 * ฝัง TradingView Mini Symbol Overview widget — พึ่งพา CDN ภายนอก (s3.tradingview.com)
 * ต้องมีอินเทอร์เน็ตถึงจะขึ้นกราฟ ถ้าออฟไลน์จะเห็นแค่กรอบว่าง (known tradeoff ที่ยอมรับแล้ว)
 */
export default function TradingViewMiniWidget({ symbol, title }: Props) {
  const containerRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const container = containerRef.current
    if (!container) return

    // สร้าง wrapper ใหม่ทุกครั้งแทนการ innerHTML='' ของ container เดิม — กัน React 18
    // StrictMode dev-mode (mount→unmount→mount ซ้อนกัน) ไปลบ node ที่ script รุ่นก่อนหน้า
    // (โหลดจาก CDN แบบ async) ยังอ้างอิงอยู่ ทำให้ TradingView script เอง throw
    // "Cannot read properties of null" ตอนพยายาม querySelector ใส่ node ที่หายไปแล้ว
    const wrapper = document.createElement('div')
    container.appendChild(wrapper)

    const script = document.createElement('script')
    script.src = 'https://s3.tradingview.com/external-embedding/embed-widget-mini-symbol-overview.js'
    script.type = 'text/javascript'
    script.async = true
    script.text = JSON.stringify({
      symbol,
      width: '100%',
      height: 150,
      locale: 'en',
      dateRange: '1M',
      colorTheme: 'light',
      isTransparent: true,
      autosize: false,
    })
    wrapper.appendChild(script)

    return () => {
      wrapper.remove()
    }
  }, [symbol])

  return (
    <div className="rounded-xl border border-zinc-200 bg-white p-2 shadow-sm shadow-black/5">
      {title && <p className="mb-1 px-1 text-xs text-zinc-500">{title}</p>}
      <div ref={containerRef} style={{ height: 150 }} />
    </div>
  )
}
