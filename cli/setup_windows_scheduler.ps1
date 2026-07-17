# PowerShell Script for setting up Windows Task Scheduler to run News Funnel
# Must be run as Administrator or under user account with appropriate permissions

$ProjectRoot = Resolve-Path "$PSScriptRoot\.."
$UvCmd = Get-Command uv -ErrorAction SilentlyContinue
if ($UvCmd) {
    $PythonExe = $UvCmd.Source
} elseif (Test-Path "$HOME\.local\bin\uv.exe") {
    $PythonExe = "$HOME\.local\bin\uv.exe"
} else {
    $PythonExe = "uv"
}
$ScriptPath = "$ProjectRoot\cli\run_news_funnel.py"

Write-Host "=========================================="
Write-Host " Setting up News Funnel Tasks on Windows "
Write-Host "=========================================="

# Unregister old task name if present
Unregister-ScheduledTask -TaskName "InvestAgents_NewsFunnel_Ingest" -Confirm:$false -ErrorAction SilentlyContinue

# 1. Task for Hourly Fetch (accumulating raw candidates without LLM calls, offset at :15)
$ActionFetch = New-ScheduledTaskAction -Execute $PythonExe -Argument "run python cli/run_news_funnel.py --mode fetch" -WorkingDirectory $ProjectRoot
$TriggerFetch = New-ScheduledTaskTrigger -Daily -At "00:15"
$TriggerFetch.Repetition = (New-ScheduledTaskTrigger -Once -At "00:15" -RepetitionInterval (New-TimeSpan -Hours 1)).Repetition
Register-ScheduledTask -TaskName "InvestAgents_NewsFunnel_Fetch" -Action $ActionFetch -Trigger $TriggerFetch -Description "Fetches news candidates hourly without calling LLM" -Force

# 2. Task for Batch Triage (Daily at 07:00 and 19:00)
$ActionTriage = New-ScheduledTaskAction -Execute $PythonExe -Argument "run python cli/run_news_funnel.py --mode ingest" -WorkingDirectory $ProjectRoot
$TriggerTriageMorning = New-ScheduledTaskTrigger -Daily -At "07:00"
$TriggerTriageEvening = New-ScheduledTaskTrigger -Daily -At "19:00"
$TriggersTriage = @($TriggerTriageMorning, $TriggerTriageEvening)
Register-ScheduledTask -TaskName "InvestAgents_NewsFunnel_Triage" -Action $ActionTriage -Trigger $TriggersTriage -Description "Triages accumulated news candidates twice daily" -Force

# 3. Task for Morning Synthesis (Daily 07:30)
$ActionMorning = New-ScheduledTaskAction -Execute $PythonExe -Argument "run python cli/run_news_funnel.py --mode synthesize --period morning" -WorkingDirectory $ProjectRoot
$TriggerMorning = New-ScheduledTaskTrigger -Daily -At "07:30"
Register-ScheduledTask -TaskName "InvestAgents_NewsFunnel_Morning" -Action $ActionMorning -Trigger $TriggerMorning -Description "Synthesizes morning macro news digest" -Force

# 4. Task for Evening Synthesis (Daily 19:30)
$ActionEvening = New-ScheduledTaskAction -Execute $PythonExe -Argument "run python cli/run_news_funnel.py --mode synthesize --period evening" -WorkingDirectory $ProjectRoot
$TriggerEvening = New-ScheduledTaskTrigger -Daily -At "19:30"
Register-ScheduledTask -TaskName "InvestAgents_NewsFunnel_Evening" -Action $ActionEvening -Trigger $TriggerEvening -Description "Synthesizes evening macro news digest" -Force

Write-Host "[OK] Registered Scheduled Tasks successfully:"
Write-Host "  - InvestAgents_NewsFunnel_Fetch (Hourly at :15)"
Write-Host "  - InvestAgents_NewsFunnel_Triage (Daily 07:00 & 19:00)"
Write-Host "  - InvestAgents_NewsFunnel_Morning (Daily 07:30)"
Write-Host "  - InvestAgents_NewsFunnel_Evening (Daily 19:30)"
