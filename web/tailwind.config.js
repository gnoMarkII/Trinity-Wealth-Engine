/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        // Semantic tokens ของธีม Flow — ค่าจริงประกาศเป็น CSS vars ใน index.css (:root)
        // ใช้แทนชั้น `.flow-theme .bg-white { !important }` เดิมที่ override รายคลาส
        // (สี accent ตระกูล terra เดิมถูกแทนด้วย sky-400/500/600 มาตรฐานซึ่งค่าตรงกันเป๊ะ)
        panel: 'var(--panel)',
        surface: 'var(--surface)',
        'surface-strong': 'var(--surface-strong)',
        edge: 'var(--edge)',
        flow: {
          cyan: '#06B6D4',
          sky: '#38BDF8',
          blue: '#0EA5E9',
        },
      },
    },
  },
  plugins: [],
}
