import '@testing-library/jest-dom/vitest'
import { cleanup } from '@testing-library/react'
import { afterEach } from 'vitest'

// ไม่ได้เปิด globals ของ Vitest — ต้อง cleanup DOM เองหลังทุก test
afterEach(() => {
  cleanup()
})
