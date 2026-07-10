/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        // Terracotta accent palette — single source of truth แทน arbitrary value
        // bg-[#924A2E] ที่เคยกระจายอยู่ทั่ว component (sync กับ CSS vars ใน index.css)
        terra: {
          DEFAULT: '#924A2E',
          dark: '#7D3E25',
          light: '#D97746',
        },
        surface: '#F4F4F5',
      },
    },
  },
  plugins: [],
}
