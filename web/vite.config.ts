import { defineConfig } from 'vitest/config'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  test: {
    environment: 'jsdom',
    setupFiles: ['./src/test/setup.ts'],
    include: ['src/**/*.test.{ts,tsx}'],
  },
  server: {
    proxy: {
      // ให้ /api/* และ /health ไหลไปที่ FastAPI (uvicorn api.main:app) แบบ same-origin
      // จาก browser — เลี่ยงต้องตั้ง CORS เพราะ auth ใช้ cookie
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/health': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
})
