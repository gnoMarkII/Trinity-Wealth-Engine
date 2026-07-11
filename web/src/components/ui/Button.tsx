import type { ButtonHTMLAttributes } from 'react'

interface Props extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'primary' | 'secondary'
  size?: 'sm' | 'md'
}

const VARIANT_CLASS: Record<string, string> = {
  primary: 'bg-flow-blue text-white shadow-[0_10px_24px_rgba(14,165,233,0.2)] hover:bg-sky-600',
  secondary: 'border border-sky-200 bg-white/70 text-sky-900 shadow-sm shadow-sky-100/60 hover:bg-sky-50',
}

const SIZE_CLASS: Record<string, string> = {
  sm: 'px-4 py-1.5 text-xs',
  md: 'px-4 py-2 text-sm',
}

export default function Button({ variant = 'primary', size = 'md', className = '', ...rest }: Props) {
  return (
    <button
      {...rest}
      className={`rounded-xl font-medium transition-all duration-150 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-flow-cyan active:scale-95 disabled:cursor-not-allowed disabled:opacity-50 ${VARIANT_CLASS[variant]} ${SIZE_CLASS[size]} ${className}`}
    />
  )
}
