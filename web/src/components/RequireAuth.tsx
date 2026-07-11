import { type ReactNode } from 'react'
import { Navigate } from 'react-router-dom'
import { useAuth } from '../auth/useAuth'

interface Props {
  children: ReactNode
}

export default function RequireAuth({ children }: Props) {
  const { status } = useAuth()

  if (status === 'loading') {
    return <div className="flex min-h-full items-center justify-center bg-white text-sm text-zinc-500">กำลังโหลด...</div>
  }
  if (status === 'unauthenticated') return <Navigate to="/login" replace />

  return <>{children}</>
}
