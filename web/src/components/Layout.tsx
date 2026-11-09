import { Outlet, useLocation } from 'react-router-dom'
import RequireAuth from './RequireAuth'
import Sidebar from './Sidebar'

export default function Layout() {
  const location = useLocation()
  const isFullHeightPage = location.pathname.startsWith('/equity') || location.pathname.startsWith('/manager')

  return (
    <RequireAuth>
      <div className="flow-theme flex h-full w-full overflow-hidden">
        <Sidebar />
        <main
          className={`flow-main min-w-0 min-h-0 flex-1 h-full ${
            isFullHeightPage ? 'overflow-hidden p-0' : 'overflow-y-auto p-4 sm:p-5 lg:p-6 xl:p-8 custom-scrollbar'
          }`}
        >
          <Outlet />
        </main>
      </div>
    </RequireAuth>
  )
}
