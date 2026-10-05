# Sector Rotation EOD Task

Detailed acceptance, scheduler shadow, and rollback checks are in the [Sector Rotation verification plan](sector-rotation-verification-plan.md).

## Windows Task Scheduler

- Task name: `InvestAgents-SectorRotation-EOD`
- Action: `.venv\Scripts\python.exe scripts\refresh_sector_rotation.py`
- Trigger: Tuesday–Saturday, 10:15 AM machine-local time (`SE Asia Standard Time`). Bangkok is 11–12 hours ahead of New York, so these runs follow Monday–Friday US closes and leave time for Yahoo Finance to publish the daily bar.
- Principal: current Windows user, interactive logon, limited privilege. The user must be logged in; missed starts run when the machine/user becomes available.
- Overlap: ignored. Execution limit: 15 minutes. Failure retry: twice, 15 minutes apart.

Register or update the task from the repository root:

```powershell
.\scripts\register_sector_rotation_task.ps1
```

The registration script refuses to replace a same-named task that points to a different project/action.

## Provider lag and output

`SECTOR_ROTATION_EOD_MAX_STALE_SESSIONS` defaults to `1` (allowed range 0–5). EOD can commit a snapshot within that provider-lag limit and reports `status: stale` plus `missing_sessions`. Larger gaps return a non-zero exit code and trigger Task Scheduler retries. Macro analysis remains strict and will not consume stale snapshots.

Runs append JSONL records to `logs\sector_rotation_eod.jsonl`. Records include status/freshness, expected and observed dates, snapshot ID, coverage, formula/calendar versions, and a `scratch`/`configured` scope label. Vault paths and provider credentials are not logged.

Inspect the task and recent runs:

```powershell
Get-ScheduledTask -TaskName InvestAgents-SectorRotation-EOD
Get-ScheduledTaskInfo -TaskName InvestAgents-SectorRotation-EOD
Get-Content .\logs\sector_rotation_eod.jsonl -Tail 10
```

Keep `SECTOR_ROTATION_AI_ENABLED=false` until five completed US sessions, one weekly close, live report/UX checks, acceptance mapping, and a rollback exercise pass. The task only refreshes canonical data/evidence; it does not enable AI.
