# Backend API & Workers Guidelines (`api/`)

This directory contains FastAPI routers, background saga workers, dependency injection providers, and database adapters.

---

## 1. 🌐 HTTP Status Code Contracts

- **200 OK:** Synchronous operations returning immediate resources.
- **202 Accepted:** Asynchronous background processing (e.g. analysis jobs, queued sagas). Returns `job_id`.
- **404 Not Found:** Resource or job ID not found.
- **422 Unprocessable Content:** Schema or domain validation failure.
- **503 Service Unavailable:** Circuit breaker open or required third-party/LLM service unavailable.

---

## 2. 🔄 Saga State Machine & Transactional Outbox

1. **Transactional Outbox:** Background jobs and state mutations must be recorded within the same SQLite transaction via Unit of Work before notifying workers.
2. **Execution Lease Fencing:** Workers must check and respect `owns_execution` / lease locks to prevent split-brain processing.
3. **No Direct DAOs in Routers:** Routers in `api/routers/` must consume application services or dependency providers, never raw filesystem or database connections directly.
4. **Idempotency:** Mutations should support idempotency keys to ensure safe retries from client apps.
