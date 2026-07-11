import { createContext, useContext } from 'react'

// แยก context/hook ออกจาก AuthContext.tsx (component) — ไฟล์ที่ export ทั้ง component
// และ non-component ทำให้ React Fast Refresh ต้อง full-reload ทุกครั้งที่แก้ไฟล์
export interface AuthContextValue {
  status: 'loading' | 'authenticated' | 'unauthenticated'
  login: (password: string) => Promise<void>
  logout: () => Promise<void>
}

export const AuthContext = createContext<AuthContextValue | null>(null)

export function useAuth(): AuthContextValue {
  const ctx = useContext(AuthContext)
  if (!ctx) throw new Error('useAuth ต้องถูกเรียกภายใน <AuthProvider>')
  return ctx
}
