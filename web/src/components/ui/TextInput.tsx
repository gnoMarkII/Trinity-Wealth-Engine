import { forwardRef, type InputHTMLAttributes } from 'react'

interface Props extends InputHTMLAttributes<HTMLInputElement> {
  uiSize?: 'sm' | 'md'
}

const SIZE_CLASS: Record<string, string> = {
  sm: 'px-3 py-1.5 text-xs',
  md: 'px-3 py-2 text-sm',
}

const TextInput = forwardRef<HTMLInputElement, Props>(function TextInput(
  { uiSize = 'md', className = '', ...rest },
  ref
) {
  return (
    <input
      ref={ref}
      {...rest}
      className={`rounded-xl border border-sky-200/85 bg-surface-2 text-slate-900 outline-none shadow-sm transition-colors placeholder:text-slate-400 focus:border-section-accent focus-visible:ring-2 focus-visible:ring-section-accent/25 ${SIZE_CLASS[uiSize]} ${className}`}
    />
  )
})

export default TextInput
