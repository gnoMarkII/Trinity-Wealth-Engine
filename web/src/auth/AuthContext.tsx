import { useEffect, useState, type ReactNode } from 'react'
import { api, setUnauthorizedHandler } from '../api/client'
import { AuthContext, type AuthContextValue } from './useAuth'

export function AuthProvider({ children }: { children: ReactNode }) {
  const [status, setStatus] = useState<AuthContextValue['status']>('loading')

  useEffect(() => {
    fetch('/api/auth/me', { credentials: 'include' })
      .then((res) => res.json())
      .then((body) => setStatus(body.authenticated ? 'authenticated' : 'unauthenticated'))
      .catch(() => setStatus('unauthenticated'))
  }, [])

  useEffect(() => {
    setUnauthorizedHandler(() => setStatus('unauthenticated'))
    return () => setUnauthorizedHandler(null)
  }, [])

  async function login(password: string) {
    await api.login(password)
    setStatus('authenticated')
  }

  async function logout() {
    await api.logout()
    setStatus('unauthenticated')
  }

  return <AuthContext.Provider value={{ status, login, logout }}>{children}</AuthContext.Provider>
}
