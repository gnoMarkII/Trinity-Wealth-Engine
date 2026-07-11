import { BrowserRouter, Navigate, Route, Routes } from 'react-router-dom'
import { AuthProvider } from './auth/AuthContext'
import ErrorBoundary from './components/ErrorBoundary'
import Layout from './components/Layout'
import Login from './pages/Login'
import Kanban from './pages/Kanban'
import Portfolio from './pages/Portfolio'
import Macro from './pages/Macro'
import Landing from './pages/Landing'

function App() {
  return (
    <ErrorBoundary>
      <BrowserRouter>
        <AuthProvider>
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
        </AuthProvider>
      </BrowserRouter>
    </ErrorBoundary>
  )
}

export default App
