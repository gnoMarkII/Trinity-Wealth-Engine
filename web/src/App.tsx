import { lazy, Suspense } from 'react'
import { BrowserRouter, Navigate, Route, Routes } from 'react-router-dom'
import { AuthProvider } from './auth/AuthContext'
import ErrorBoundary from './components/ErrorBoundary'
import Layout from './components/Layout'

// lazy ทุกหน้า — แยก chunk ต่อ route เพื่อไม่ให้ผู้ใช้ /kanban ต้องโหลดโค้ด landing
// (motion ~40KB) และกลับกัน; Login โหลด eager เพราะเป็นหน้าแรกที่ผู้ใช้ยังไม่ auth เจอเสมอ
import Login from './pages/Login'
const Kanban = lazy(() => import('./pages/Kanban'))
const Portfolio = lazy(() => import('./pages/Portfolio'))
const Macro = lazy(() => import('./pages/Macro'))
const Landing = lazy(() => import('./pages/Landing'))

function RouteFallback() {
  return <div className="flex min-h-full items-center justify-center text-sm text-zinc-400">กำลังโหลด...</div>
}

function App() {
  return (
    <ErrorBoundary>
      <BrowserRouter>
        <AuthProvider>
          <Suspense fallback={<RouteFallback />}>
            <Routes>
              <Route path="/login" element={<Login />} />
              <Route path="/" element={<Landing />} />
              <Route element={<Layout />}>
                <Route path="/kanban" element={<Kanban />} />
                <Route path="/portfolio" element={<Portfolio />} />
                <Route path="/macro" element={<Macro />} />
              </Route>
              <Route path="*" element={<Navigate to="/" replace />} />
            </Routes>
          </Suspense>
        </AuthProvider>
      </BrowserRouter>
    </ErrorBoundary>
  )
}

export default App
