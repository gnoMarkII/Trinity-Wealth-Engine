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

# 1. Task for Hourly Ingestion
$ActionIngest = New-ScheduledTaskAction -Execute $PythonExe -Argument "run python cli/run_news_funnel.py --mode ingest" -WorkingDirectory $ProjectRoot
$TriggerIngest = New-ScheduledTaskTrigger -Daily -At "00:00"
$TriggerIngest.Repetition = (New-ScheduledTaskTrigger -Once -At "00:00" -RepetitionInterval (New-TimeSpan -Hours 1)).Repetition
Register-ScheduledTask -TaskName "InvestAgents_NewsFunnel_Ingest" -Action $ActionIngest -Trigger $TriggerIngest -Description "Ingests financial & macro news candidates hourly" -Force

# 2. Task for Morning Synthesis (Daily 07:30)
$ActionMorning = New-ScheduledTaskAction -Execute $PythonExe -Argument "run python cli/run_news_funnel.py --mode synthesize --period morning" -WorkingDirectory $ProjectRoot
$TriggerMorning = New-ScheduledTaskTrigger -Daily -At "07:30"
Register-ScheduledTask -TaskName "InvestAgents_NewsFunnel_Morning" -Action $ActionMorning -Trigger $TriggerMorning -Description "Synthesizes morning macro news digest" -Force

# 3. Task for Evening Synthesis (Daily 19:30)
$ActionEvening = New-ScheduledTaskAction -Execute $PythonExe -Argument "run python cli/run_news_funnel.py --mode synthesize --period evening" -WorkingDirectory $ProjectRoot
$TriggerEvening = New-ScheduledTaskTrigger -Daily -At "19:30"
Register-ScheduledTask -TaskName "InvestAgents_NewsFunnel_Evening" -Action $ActionEvening -Trigger $TriggerEvening -Description "Synthesizes evening macro news digest" -Force

Write-Host "[OK] Registered Scheduled Tasks successfully:"
Write-Host "  - InvestAgents_NewsFunnel_Ingest (Hourly)"
Write-Host "  - InvestAgents_NewsFunnel_Morning (Daily 07:30)"
Write-Host "  - InvestAgents_NewsFunnel_Evening (Daily 19:30)"
