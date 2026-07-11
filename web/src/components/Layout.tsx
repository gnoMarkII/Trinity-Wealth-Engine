import { Navigate, Outlet } from 'react-router-dom'
import { useAuth } from '../auth/useAuth'
import Sidebar from './Sidebar'

export default function Layout() {
  const { status } = useAuth()

  if (status === 'loading') {
    return <div className="flex min-h-full items-center justify-center text-zinc-500">กำลังโหลด...</div>
  }
  if (status === 'unauthenticated') {
    return <Navigate to="/login" replace />
  }

  return (
    <div className="flow-theme flex h-full">
      <Sidebar />
      <main className="flow-main min-w-0 flex-1 overflow-y-auto p-5 sm:p-8">
        <Outlet />
      </main>
    </div>
  )
}
