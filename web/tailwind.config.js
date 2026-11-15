/** @type {import('tailwindcss').Config} */
export default {
  darkMode: 'class',
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      fontFamily: {
        sans: [
          '"Plus Jakarta Sans"',
          '"Noto Sans Thai"',
          'system-ui',
          '-apple-system',
          'BlinkMacSystemFont',
          '"Segoe UI"',
          'Roboto',
          'sans-serif',
        ],
        mono: [
          '"JetBrains Mono"',
          'ui-monospace',
          'SFMono-Regular',
          'Menlo',
          'Monaco',
          'Consolas',
          'monospace',
        ],
      },
      colors: {
        // Semantic tokens ของธีม Flow — ค่าจริงประกาศเป็น CSS vars ใน index.css (:root)
        // ใช้แทนชั้น `.flow-theme .bg-white { !important }` เดิมที่ override รายคลาส
        // (สี accent ตระกูล terra เดิมถูกแทนด้วย sky-400/500/600 มาตรฐานซึ่งค่าตรงกันเป๊ะ)
        panel: 'var(--panel)',
        surface: 'var(--surface)',
        'surface-strong': 'var(--surface-strong)',
        'surface-1': 'var(--surface-1)',
        'surface-2': 'var(--surface-2)',
        'surface-3': 'var(--surface-3)',
        'surface-inset': 'var(--surface-inset)',
        edge: 'var(--edge)',
        primary: 'rgb(var(--primary-rgb) / <alpha-value>)',
        muted: 'rgb(var(--muted-rgb) / <alpha-value>)',
        accent: 'rgb(var(--accent-rgb) / <alpha-value>)',
        'section-accent': 'rgb(var(--section-accent) / <alpha-value>)',
        'section-accent-2': 'rgb(var(--section-accent-2) / <alpha-value>)',
        ink: {
          primary: 'var(--ink-primary)',
          secondary: 'var(--ink-secondary)',
          muted: 'var(--ink-muted)',
        },
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
