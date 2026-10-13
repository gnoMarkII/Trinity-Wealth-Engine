import { Navigate, Outlet, useLocation } from 'react-router-dom'
import { useAuth } from '../auth/useAuth'
import Sidebar from './Sidebar'

export default function Layout() {
  const { status } = useAuth()
  const location = useLocation()
  const isEquityPage = location.pathname.startsWith('/equity')

  if (status === 'loading') {
    return <div className="flex min-h-full items-center justify-center text-zinc-500">กำลังโหลด...</div>
  }
  if (status === 'unauthenticated') {
    return <Navigate to="/login" replace />
  }

  return (
    <div className="flow-theme flex h-full w-full overflow-hidden">
      <Sidebar />
      <main
        className={`flow-main min-w-0 min-h-0 flex-1 h-full ${
          isEquityPage ? 'overflow-hidden p-0' : 'overflow-y-auto p-4 sm:p-5 lg:p-6 xl:p-8 custom-scrollbar'
        }`}
      >
        <Outlet />
      </main>
    </div>
  )
}
