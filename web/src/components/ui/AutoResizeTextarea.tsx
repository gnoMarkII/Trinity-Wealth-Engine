import { useEffect, useRef, type TextareaHTMLAttributes } from 'react'

interface AutoResizeTextareaProps extends TextareaHTMLAttributes<HTMLTextAreaElement> {
  minRows?: number
  maxHeight?: number
}

export default function AutoResizeTextarea({
  value,
  onChange,
  className = '',
  minRows = 2,
  maxHeight,
  style,
  ...rest
}: AutoResizeTextareaProps) {
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  const adjustHeight = () => {
    const el = textareaRef.current
    if (!el) return
    el.style.height = 'auto'
    const scrollH = el.scrollHeight
    if (maxHeight && scrollH > maxHeight) {
      el.style.height = `${maxHeight}px`
      el.style.overflowY = 'auto'
    } else {
      el.style.height = `${scrollH}px`
      el.style.overflowY = 'hidden'
    }
  }

  useEffect(() => {
    adjustHeight()
  }, [value])

  return (
    <textarea
      ref={textareaRef}
      rows={minRows}
      value={value}
      onChange={(e) => {
        onChange?.(e)
        adjustHeight()
      }}
      style={{
        ...style,
        resize: 'none',
      }}
      className={`w-full transition-[height] duration-75 leading-relaxed ${className}`}
      {...rest}
    />
  )
}
