import React from 'react'

export interface IllustrativeValueScaleProps {
  currentPrice: number | null | undefined
  intrinsicValue: number | null | undefined
  currencySymbol?: string
  className?: string
}

export const IllustrativeValueScale: React.FC<IllustrativeValueScaleProps> = ({
  currentPrice,
  intrinsicValue,
  currencySymbol = '$',
  className = '',
}) => {
  if (
    currentPrice == null ||
    intrinsicValue == null ||
    currentPrice <= 0 ||
    intrinsicValue <= 0 ||
    isNaN(currentPrice) ||
    isNaN(intrinsicValue)
  ) {
    return null
  }

  // Normalize: If Intrinsic Value = 100 units, how much is Current Price selling for?
  const priceRatio = currentPrice / intrinsicValue
  const normalizedPrice = Math.round(priceRatio * 100)
  const discountOrPremiumPct = Math.round(((currentPrice - intrinsicValue) / intrinsicValue) * 100)
  const isDiscount = discountOrPremiumPct < 0
  const isPremium = discountOrPremiumPct > 0

  return (
    <div className={`rounded-xl border border-sky-100 bg-sky-50/40 p-3.5 text-xs ${className}`}>
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-1 border-b border-sky-100 pb-2">
        <span className="font-bold text-zinc-900 flex items-center gap-1.5">
          <span>🏷️</span>
          <span>โมเดลเปรียบเทียบมูลค่า 100 บาท (Illustrative 100-Unit Scale)</span>
        </span>
        <span className="font-mono text-[11px] text-zinc-500">
          ราคาตลาด {currencySymbol}{currentPrice.toFixed(2)} vs มูลค่าแท้จริง {currencySymbol}{intrinsicValue.toFixed(2)}
        </span>
      </div>

      <div className="mt-2.5 space-y-2">
        <div className="flex items-center justify-between text-xs">
          <div>
            {isDiscount ? (
              <span className="inline-flex items-center gap-1 font-bold text-emerald-800">
                <span>🟢 มีส่วนลดราคา (Margin of Safety):</span>
                <span className="font-mono">{Math.abs(discountOrPremiumPct)}%</span>
              </span>
            ) : isPremium ? (
              <span className="inline-flex items-center gap-1 font-bold text-rose-800">
                <span>🔴 ขายแพงกว่าพื้นฐาน (Premium):</span>
                <span className="font-mono">+{discountOrPremiumPct}%</span>
              </span>
            ) : (
              <span className="font-bold text-zinc-800">⚪ ราคาเหมาะสมกับมูลค่าพื้นฐานพอดี</span>
            )}
          </div>
          <span className="font-mono font-bold text-zinc-900 text-sm">
            {normalizedPrice} / 100 บาท
          </span>
        </div>

        {/* Visual Progress Bar */}
        <div className="relative h-3 w-full rounded-full bg-slate-200 overflow-hidden">
          <div
            className={`h-full rounded-full transition-all duration-500 ${
              isDiscount ? 'bg-emerald-500' : isPremium ? 'bg-rose-500' : 'bg-sky-500'
            }`}
            style={{ width: `${Math.min(normalizedPrice, 150) / 1.5}%` }}
          />
          {/* Target Fair Value Marker (at 100 / 150 = 66.6%) */}
          <div
            className="absolute top-0 bottom-0 w-0.5 bg-zinc-800 shadow-sm"
            style={{ left: `${100 / 1.5}%` }}
            title="มูลค่าแท้จริง 100 บาท (Fair Value)"
          />
        </div>

        <p className="text-[11px] text-zinc-600 leading-relaxed">
          {isDiscount ? (
            <>
              💡 <strong>แปลภาษาคน:</strong> ถ้ามูลค่าแท้จริงของหุ้นตัวนี้คือ <strong>100 บาท</strong> วันนี้ตลาดกำลังขายลดราคาให้คุณที่ <strong>{normalizedPrice} บาท</strong> (ประหยัดไป {Math.abs(discountOrPremiumPct)} บาท เป็นแต้มต่อความปลอดภัย)
            </>
          ) : isPremium ? (
            <>
              ⚠️ <strong>แปลภาษาคน:</strong> หุ้นตัวนี้มูลค่าแท้จริงคือ <strong>100 บาท</strong> แต่วันนี้ตลาดขายที่ <strong>{normalizedPrice} บาท</strong> (คุณกำลังจ่ายแพงกว่าพื้นฐาน {discountOrPremiumPct} บาทจากกระแสเก็งกำไร)
            </>
          ) : (
            <>
              💡 <strong>แปลภาษาคน:</strong> ราคาหุ้นในตลาดสะท้อนมูลค่าแท้จริง 100 บาทพอดี ไม่ถูกและไม่แพง
            </>
          )}
        </p>
      </div>
    </div>
  )
}

