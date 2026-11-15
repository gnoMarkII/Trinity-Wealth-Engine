import type { ButtonHTMLAttributes } from 'react'

interface Props extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'primary' | 'secondary' | 'ghost' | 'danger'
  size?: 'sm' | 'md'
}

const VARIANT_CLASS: Record<NonNullable<Props['variant']>, string> = {
  primary:
    'bg-section-accent text-white shadow-[0_8px_20px_rgb(var(--section-accent)/0.22)] hover:brightness-110',
  secondary:
    'border border-sky-200/90 bg-white/85 text-slate-700 shadow-sm hover:border-sky-300 hover:bg-sky-50/70 hover:text-slate-900',
  ghost:
    'text-slate-600 hover:bg-sky-50/70 hover:text-slate-900',
  danger:
    'border border-rose-200 bg-rose-50/80 text-rose-700 hover:bg-rose-600 hover:text-white',
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
