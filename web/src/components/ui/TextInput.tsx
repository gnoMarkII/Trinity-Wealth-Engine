import type { InputHTMLAttributes } from 'react'

interface Props extends InputHTMLAttributes<HTMLInputElement> {
  uiSize?: 'sm' | 'md'
}

const SIZE_CLASS: Record<string, string> = {
  sm: 'px-3 py-1.5 text-xs',
  md: 'px-3 py-2 text-sm',
}

export default function TextInput({ uiSize = 'md', className = '', ...rest }: Props) {
  return (
    <input
      {...rest}
      className={`rounded-xl border border-sky-200 bg-panel text-zinc-900 outline-none shadow-sm shadow-sky-100/40 transition-colors placeholder-zinc-400 focus:border-flow-cyan focus:ring-2 focus:ring-flow-cyan/20 ${SIZE_CLASS[uiSize]} ${className}`}
    />
  )
}
