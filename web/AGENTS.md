# Frontend Web Guidelines (`web/`)

This directory contains the React 19 + TypeScript + Vite frontend.

---

## 1. 💻 Coding & Component Standards

1. **Framework:** React 19, TypeScript (strict mode), Vite.
2. **Design System:** Use design tokens defined in `src/index.css`. Maintain rich aesthetics with dark mode, rounded cards, clean typography, and responsive layouts.
3. **API Contracts:** Generated TypeScript types reside in `src/api/types.generated.ts`. Keep `client.ts` in sync with OpenAPI endpoints.
4. **State Management:** Use custom React hooks (e.g. `useInvestorPrinciples`, `usePortfolioState`) with optimistic updates and error handling.

---

## 2. 🧪 Quality Gates & Verification

Before concluding frontend tasks, execute:
- Type check: `npm --prefix web run typecheck`
- Test suite: `npm --prefix web test -- --run`
