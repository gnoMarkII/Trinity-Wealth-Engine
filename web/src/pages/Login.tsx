import { useState, type FormEvent } from 'react'
import { useNavigate } from 'react-router-dom'
import { useAuth } from '../auth/useAuth'
import { ApiError } from '../api/client'
import Button from '../components/ui/Button'
import TextInput from '../components/ui/TextInput'

export default function Login() {
  const { login } = useAuth()
  const navigate = useNavigate()
  const [password, setPassword] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [submitting, setSubmitting] = useState(false)

  async function handleSubmit(e: FormEvent) {
    e.preventDefault()
    setError(null)
    setSubmitting(true)
    try {
      await login(password)
      navigate('/', { replace: true })
    } catch (err) {
      setError(err instanceof ApiError ? err.message : 'เข้าสู่ระบบไม่สำเร็จ')
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <div className="flow-theme flex min-h-screen items-center justify-center px-6">
      <form
        onSubmit={handleSubmit}
        className="flow-panel animate-page-in w-full max-w-sm space-y-4 rounded-3xl p-8"
      >
        <p className="text-xs font-semibold uppercase tracking-[0.2em] text-sky-600">Investment intelligence</p>
        <h1 className="text-xl font-semibold text-sky-950">Money ReRoute</h1>
        <p className="text-sm text-zinc-500">กรอกรหัสผ่านเพื่อเข้าใช้งาน</p>
        <label htmlFor="login-password" className="sr-only">
          รหัสผ่าน
        </label>
        <TextInput
          id="login-password"
          type="password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          placeholder="รหัสผ่าน"
          autoFocus
          className="w-full"
        />
        {error && <p className="text-sm text-red-600">{error}</p>}
        <Button type="submit" disabled={submitting || !password} className="w-full">
          {submitting ? 'กำลังเข้าสู่ระบบ...' : 'เข้าสู่ระบบ'}
        </Button>
      </form>
    </div>
  )
}
