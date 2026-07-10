import { BrowserRouter, Navigate, Route, Routes } from 'react-router-dom'
import { AuthProvider } from './auth/AuthContext'
import ErrorBoundary from './components/ErrorBoundary'
import Layout from './components/Layout'
import Login from './pages/Login'
import Kanban from './pages/Kanban'
import Portfolio from './pages/Portfolio'
import Macro from './pages/Macro'

function App() {
  return (
    <ErrorBoundary>
      <BrowserRouter>
        <AuthProvider>
          <Routes>
            <Route path="/login" element={<Login />} />
            <Route element={<Layout />}>
              <Route path="/kanban" element={<Kanban />} />
              <Route path="/portfolio" element={<Portfolio />} />
              <Route path="/macro" element={<Macro />} />
              <Route path="/" element={<Navigate to="/kanban" replace />} />
            </Route>
          </Routes>
        </AuthProvider>
      </BrowserRouter>
    </ErrorBoundary>
  )
}

export default App
