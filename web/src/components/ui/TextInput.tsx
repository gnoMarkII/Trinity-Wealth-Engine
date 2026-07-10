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
      className={`rounded-lg border border-zinc-200 bg-white text-zinc-900 outline-none transition-colors placeholder-zinc-400 focus:border-terra focus:ring-1 focus:ring-terra/30 ${SIZE_CLASS[uiSize]} ${className}`}
    />
  )
}
