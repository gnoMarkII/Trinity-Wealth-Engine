/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        // Accent palette (sky blue ตามธีม Flow ปัจจุบัน) — ชื่อ "terra" เป็นมรดกจากธีม
        // terracotta เดิม คงชื่อไว้เพราะ class กระจายทั่ว component; ค่าสีจริงคือ sky/cyan
        terra: {
          DEFAULT: '#0EA5E9',
          dark: '#0284C7',
          light: '#38BDF8',
        },
        surface: '#F4F4F5',
        flow: {
          cyan: '#06B6D4',
          sky: '#38BDF8',
          blue: '#0EA5E9',
          coral: '#FB8C00',
          peach: '#FF8A65',
        },
      },
    },
  },
  plugins: [],
}
