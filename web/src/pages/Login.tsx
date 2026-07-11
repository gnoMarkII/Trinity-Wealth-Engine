import { useState, type FormEvent } from 'react'
import { useNavigate } from 'react-router-dom'
import { useAuth } from '../auth/AuthContext'
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
    <div className="flex min-h-full items-center justify-center bg-white">
      <form
        onSubmit={handleSubmit}
        className="animate-page-in w-full max-w-sm space-y-4 rounded-2xl border border-zinc-200 bg-white p-8 shadow-xl shadow-black/5"
      >
        <h1 className="text-lg font-semibold text-zinc-900">Money ReRoute</h1>
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
