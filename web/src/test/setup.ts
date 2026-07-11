import '@testing-library/jest-dom/vitest'
import { cleanup } from '@testing-library/react'
import { afterEach } from 'vitest'

// jsdom ไม่ implement Element.scrollTo — LiveTerminal เรียกตอน auto-scroll log
// (typeof guard เพราะ setup นี้รันกับ test ที่ประกาศ @vitest-environment node ด้วย)
if (typeof Element !== 'undefined' && !Element.prototype.scrollTo) {
  Element.prototype.scrollTo = () => {}
}

// ไม่ได้เปิด globals ของ Vitest — ต้อง cleanup DOM เองหลังทุก test
afterEach(() => {
  cleanup()
})
