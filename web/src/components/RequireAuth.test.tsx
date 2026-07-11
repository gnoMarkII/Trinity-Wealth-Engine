import { render, screen } from '@testing-library/react'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import { describe, expect, it } from 'vitest'
import { AuthContext, type AuthContextValue } from '../auth/useAuth'
import RequireAuth from './RequireAuth'

function renderWithAuth(status: AuthContextValue['status']) {
  const value: AuthContextValue = {
    status,
    login: async () => {},
    logout: async () => {},
  }
  return render(
    <AuthContext.Provider value={value}>
      <MemoryRouter initialEntries={['/']}>
        <Routes>
          <Route
            path="/"
            element={
              <RequireAuth>
                <p>เนื้อหาลับ</p>
              </RequireAuth>
            }
          />
          <Route path="/login" element={<p>หน้า login</p>} />
        </Routes>
      </MemoryRouter>
    </AuthContext.Provider>
  )
}

describe('RequireAuth', () => {
  it('ระหว่างเช็ค session แสดงสถานะกำลังโหลด ไม่เผลอ redirect', () => {
    renderWithAuth('loading')
    expect(screen.getByText('กำลังโหลด...')).toBeInTheDocument()
    expect(screen.queryByText('เนื้อหาลับ')).not.toBeInTheDocument()
  })

  it('ยังไม่ auth → redirect ไปหน้า login', () => {
    renderWithAuth('unauthenticated')
    expect(screen.getByText('หน้า login')).toBeInTheDocument()
    expect(screen.queryByText('เนื้อหาลับ')).not.toBeInTheDocument()
  })

  it('auth แล้ว → แสดง children ปกติ', () => {
    renderWithAuth('authenticated')
    expect(screen.getByText('เนื้อหาลับ')).toBeInTheDocument()
  })
})
