/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        // Terracotta accent palette — single source of truth แทน arbitrary value
        // bg-[#924A2E] ที่เคยกระจายอยู่ทั่ว component (sync กับ CSS vars ใน index.css)
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
