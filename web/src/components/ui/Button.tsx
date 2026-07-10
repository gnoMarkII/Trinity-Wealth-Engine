import type { ButtonHTMLAttributes } from 'react'

interface Props extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'primary' | 'secondary'
  size?: 'sm' | 'md'
}

const VARIANT_CLASS: Record<string, string> = {
  primary: 'bg-terra text-white shadow-sm hover:bg-terra-dark',
  secondary: 'border border-zinc-200 text-zinc-600 hover:bg-zinc-50',
}

const SIZE_CLASS: Record<string, string> = {
  sm: 'px-4 py-1.5 text-xs',
  md: 'px-4 py-2 text-sm',
}

export default function Button({ variant = 'primary', size = 'md', className = '', ...rest }: Props) {
  return (
    <button
      {...rest}
      className={`rounded-lg font-medium transition-all duration-150 active:scale-95 disabled:cursor-not-allowed disabled:opacity-50 ${VARIANT_CLASS[variant]} ${SIZE_CLASS[size]} ${className}`}
    />
  )
}
