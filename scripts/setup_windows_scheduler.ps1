# PowerShell Script สำหรับตั้งค่า Windows Task Scheduler ให้รัน News Funnel อัตโนมัติ
# ต้องรัน PowerShell ในฐานะ Administrator หรือสิทธิ์ผู้ใช้ของระบบ

$ProjectRoot = Resolve-Path "$PSScriptRoot\.."
$PythonExe = "uv"
$ScriptPath = "$ProjectRoot\scripts\run_news_funnel.py"

Write-Host "=========================================="
Write-Host " Setting up News Funnel Tasks on Windows "
Write-Host "=========================================="

# 1. Task สำหรับ Ingestion ทุก 1 ชั่วโมง
$ActionIngest = New-ScheduledTaskAction -Execute $PythonExe -Argument "run python scripts/run_news_funnel.py --mode ingest" -WorkingDirectory $ProjectRoot
$TriggerIngest = New-ScheduledTaskTrigger -Daily -At "00:00"
$TriggerIngest.Repetition = (New-ScheduledTaskTrigger -Once -At "00:00" -RepetitionInterval (New-TimeSpan -Hours 1)).Repetition
Register-ScheduledTask -TaskName "InvestAgents_NewsFunnel_Ingest" -Action $ActionIngest -Trigger $TriggerIngest -Description "Ingests financial & macro news candidates hourly" -Force

# 2. Task สำหรับ Morning Synthesis (07:30 น. ทุกวัน)
$ActionMorning = New-ScheduledTaskAction -Execute $PythonExe -Argument "run python scripts/run_news_funnel.py --mode synthesize --period morning" -WorkingDirectory $ProjectRoot
$TriggerMorning = New-ScheduledTaskTrigger -Daily -At "07:30"
Register-ScheduledTask -TaskName "InvestAgents_NewsFunnel_Morning" -Action $ActionMorning -Trigger $TriggerMorning -Description "Synthesizes morning macro news digest" -Force

# 3. Task สำหรับ Evening Synthesis (19:30 น. ทุกวัน)
$ActionEvening = New-ScheduledTaskAction -Execute $PythonExe -Argument "run python scripts/run_news_funnel.py --mode synthesize --period evening" -WorkingDirectory $ProjectRoot
$TriggerEvening = New-ScheduledTaskTrigger -Daily -At "19:30"
Register-ScheduledTask -TaskName "InvestAgents_NewsFunnel_Evening" -Action $ActionEvening -Trigger $TriggerEvening -Description "Synthesizes evening macro news digest" -Force

Write-Host "✓ Registered Scheduled Tasks successfully:"
Write-Host "  - InvestAgents_NewsFunnel_Ingest (Hourly)"
Write-Host "  - InvestAgents_NewsFunnel_Morning (Daily 07:30)"
Write-Host "  - InvestAgents_NewsFunnel_Evening (Daily 19:30)"
