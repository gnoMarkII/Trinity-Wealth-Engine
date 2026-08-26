# Architecture and Hexagonal Boundary

เอกสารนี้เป็น source of truth สำหรับขอบเขตของ bounded context และ dependency
direction ของ backend หลังการปรับ architecture รอบล่าสุด

## Dependency direction

```text
FastAPI routers / LangChain tools / workers
                 |
                 v
Application services (use cases + application DTOs)
                 |
                 v
Domain models/calculations  <---  outbound ports
                                      |
                                      v
                         driven adapters (SQLite, Markdown,
                         yfinance, SEC, NotebookLM CLI)
```

กฎสำคัญ:

1. Domain ไม่ import FastAPI, SQLite, yfinance, LangChain หรือ adapter
2. Application service รับ port ผ่าน constructor และคืน application DTO/plain data
3. Router แปลง request/response เท่านั้น ห้ามเรียก raw DAO, `state_db`, SQLite หรือ provider
4. Adapter เป็นจุดเดียวที่รู้จัก driver/external provider
5. Composition root (`api/dependencies.py`, `tools/**/bootstrap.py`) เป็นจุดสร้าง adapter
6. SQLite raw DAO ไม่ commit/rollback; `DbUnitOfWork` เป็น transaction owner
7. Compatibility facade มีไว้รักษา import/signature เดิม ไม่ใช่ runtime dependency ของ router

## Bounded contexts

| Context | Application boundary | Ports | Driven adapters |
|---|---|---|---|
| Portfolio | `tools/portfolio/application.py`, `services/*` | portfolio, price, dividend, journal, goals, performance | Markdown vault, SQLite mirror, yfinance |
| Equity research | `application/equity/service.py` | analyst cache/provider, valuation ledger/sidecar, insider ledger/sync | SQLite, yfinance/calendar/earnings, vault sidecar, SEC pipeline |
| OHLCV | `tools/market/ohlcv/application/query_service.py` | OHLCV, corporate action, asset resolver | yfinance adapter |
| Financials | `application/equity/service.py` → `tools/market/financials/service.py` | asset resolver + financial query ports | SQLite/EDGAR/yfinance adapters |
| Jobs | `application/jobs/service.py` | `JobRepositoryPort` | SQLite job adapter |
| Kanban | `application/kanban/service.py` | `KanbanRepositoryPort` | SQLite Kanban adapter |
| NotebookLM | `application/notebooklm/service.py` | job/card/dispatch/binary ports | SQLite, queue, CLI adapter |

## Transaction ownership

`api/db/repositories/*` execute SQL only. `api/db/adapters.py` supports both
standalone legacy calls and connection-bound adapters. A `DbUnitOfWork` exposes
bound `jobs`, `kanban`, `notebooklm`, and notification `outbox` repositories;
successful exit commits the whole unit and exceptions roll it back.
`notification_outbox` stores durable, idempotent external delivery events and
is marked sent only after Discord acknowledges delivery. `api/state_db.py` remains a
compatibility facade and is the only place where legacy connection functions
perform their historical commits.

Portfolio ledger mutations use the repository UoW and staged `LedgerChange`; a
filesystem/ledger side effect must not happen before UoW commit.

## Current implementation status

- Portfolio sub-services and bootstrap composition are implemented; the legacy
  `PortfolioService` keeps explicit signatures and delegates through one
  `PortfolioApplication`.
- Equity valuation, insider, and analyst routes now call application services;
  their SQLite/provider work is behind ports and driven adapters.
- Insider synchronization now uses a normalized history provider plus a
  connection-bound SQLite adapter; the provider fetch and ledger transaction
  are no longer coupled in the API route path.
- Agent and NotebookLM HTTP endpoints are inbound adapters in
  `api/routers/agents_router.py` and `api/routers/notebooklm_router.py`.
- Macro news-funnel card creation is an application use case; its SQLite
  `upsert_open_card` path reads and writes on one UoW connection, preventing
  duplicate open cards under concurrent runs.
- NotebookLM post-production/Discord delivery is an application use case with
  briefing-content, worker-state, notification, and durable outbox ports.
- JobQueue transitions (claim, completion, failure, restart recovery) and
  manager log writes use connection-bound repositories through
  `DbUnitOfWork`, keeping job + Kanban moves atomic. LangGraph workflow
  execution is delegated from the API compatibility entry point to
  `agents/job_runner.py`, which receives the repository port and does not
  construct SQLite connections.
- Equity financials asset classification, authoritative market checks, and
  provider-symbol resolution live in `EquityFinancialsApplicationService`; the
  router only maps request values and domain errors to HTTP responses.
- OHLCV request validation is owned by the query service and exposes a typed
  validation error so the router can preserve the historical 400 contract.
- OHLCV application query services require injected providers/resolver; the
  yfinance adapter is constructed by the bootstrap and the cache is process
  scoped so the old endpoint behavior remains intact.
- Architecture tests cover domain isolation, application/router infrastructure
  imports, raw DAO transaction rules, lower-layer API imports, and dynamic
  `sys.modules` lookups.

## Remaining migration work

The following are intentional compatibility/infrastructure items, not claims
of completion:

1. `api/db/legacy_adapter.py` is the remaining state-store compatibility
   bridge for a few worker-facing operations. It now exposes explicit methods
   (no dynamic `__getattr__`); replace it with fully typed lifecycle repository
   ports after downstream monkeypatch callers migrate.
2. Macro and older market utility modules still contain provider-specific
   logic. Classify each as a driven adapter or extract a port-backed service.
3. Route compatibility modules still re-export legacy provider symbols for
   direct imports. `api/dependencies.py` no longer reads those modules;
   production composition uses explicit provider adapters and test seams patch
   dependency factories/ports.
4. `api/routes_*.py` and `tools/portfolio/service.py` remain import/signature
  facades. They must not receive new business logic.

## Verification gates

```text
python -m pytest tests/architecture -q
python -m pytest tests/unit/db tests/unit/application tests/unit/market -q
python -m pytest tests/api/test_routes_valuation.py tests/api/test_routes_insider.py \
  tests/api/test_equity_analyst_context.py tests/api/test_routes_notebooklm.py -q
python -m pytest tests/ -q
```

The final gate is full test pass plus zero OpenAPI and LangChain manifest
drift. Any remaining compatibility shim must have an owner and a removal
condition in the migration issue before it can be deleted.
