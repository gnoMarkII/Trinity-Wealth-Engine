import { lazy, Suspense } from 'react'
import { BrowserRouter, Navigate, Route, Routes } from 'react-router-dom'
import { AuthProvider } from './auth/AuthContext'
import ErrorBoundary from './components/ErrorBoundary'
import Layout from './components/Layout'

// lazy ทุกหน้า — แยก chunk ต่อ route เพื่อไม่ให้โหลดโค้ดเกินจำเป็น
import Login from './pages/Login'
const Portfolio = lazy(() => import('./pages/Portfolio'))
const PortfolioAdvisory = lazy(() => import('./pages/PortfolioAdvisory'))
const Macro = lazy(() => import('./pages/Macro'))
const Equity = lazy(() => import('./pages/Equity'))
const DailyDigest = lazy(() => import('./pages/DailyDigest'))
const ManagerAdvisor = lazy(() => import('./pages/ManagerAdvisor'))
const RoundtableStudio = lazy(() => import('./pages/RoundtableStudio'))

function RouteFallback() {
  return (
    <div className="flex min-h-full items-center justify-center gap-2 text-sm font-medium text-slate-500">
      <span className="h-2 w-2 rounded-full bg-sky-500 animate-pulse" aria-hidden="true" />
      <span>กำลังโหลด...</span>
    </div>
  )
}

function App() {
  return (
    <ErrorBoundary>
      <BrowserRouter>
        <AuthProvider>
          <Suspense fallback={<RouteFallback />}>
            <Routes>
              <Route path="/login" element={<Login />} />
              <Route path="/" element={<Navigate to="/manager" replace />} />
              <Route element={<Layout />}>
                <Route path="/manager" element={<ManagerAdvisor />} />
                <Route path="/roundtable" element={<RoundtableStudio />} />
                <Route path="/advisor" element={<Navigate to="/manager" replace />} />
                <Route path="/kanban" element={<Navigate to="/manager?view=kanban" replace />} />
                <Route path="/digest" element={<DailyDigest />} />
                <Route path="/portfolio" element={<Portfolio />} />
                <Route path="/advisory" element={<PortfolioAdvisory />} />
                <Route path="/macro" element={<Macro />} />
                <Route path="/equity" element={<Equity />} />
                <Route path="/equity/:ticker" element={<Equity />} />
              </Route>
              <Route path="*" element={<Navigate to="/manager" replace />} />
            </Routes>
          </Suspense>
        </AuthProvider>
      </BrowserRouter>
    </ErrorBoundary>
  )
}

export default App
