[CmdletBinding()]
param(
    [string]$TaskName = 'InvestAgents-SectorRotation-EOD'
)

$ErrorActionPreference = 'Stop'

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$PythonPath = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
$RefreshScript = Join-Path $ProjectRoot 'scripts\refresh_sector_rotation.py'

foreach ($RequiredPath in @($PythonPath, $RefreshScript, (Join-Path $ProjectRoot '.env'))) {
    if (-not (Test-Path -LiteralPath $RequiredPath -PathType Leaf)) {
        throw "Required project file is missing: $RequiredPath"
    }
}

$ExistingTask = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($ExistingTask) {
    $Actions = @($ExistingTask.Actions)
    $ExpectedArguments = '"{0}"' -f $RefreshScript
    if ($Actions.Count -ne 1 -or
        $Actions[0].Execute -ine $PythonPath -or
        $Actions[0].Arguments -ine $ExpectedArguments) {
        throw "Task '$TaskName' already exists with a different action; refusing to replace it."
    }
}

$Action = New-ScheduledTaskAction `
    -Execute $PythonPath `
    -Argument ('"{0}"' -f $RefreshScript) `
    -WorkingDirectory $ProjectRoot

# Thailand is 11–12 hours ahead of New York. Tuesday–Saturday local mornings
# therefore follow Monday–Friday US closes, with time for Yahoo EOD data to land.
$Trigger = New-ScheduledTaskTrigger `
    -Weekly `
    -DaysOfWeek Tuesday, Wednesday, Thursday, Friday, Saturday `
    -WeeksInterval 1 `
    -At '10:15AM'

$Identity = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
$Principal = New-ScheduledTaskPrincipal `
    -UserId $Identity `
    -LogonType Interactive `
    -RunLevel Limited

$Settings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -MultipleInstances IgnoreNew `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 15) `
    -RestartCount 2 `
    -RestartInterval (New-TimeSpan -Minutes 15) `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries

$RegisteredTask = Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $Action `
    -Trigger $Trigger `
    -Principal $Principal `
    -Settings $Settings `
    -Description 'Refresh and archive US Sector Rotation after the prior US regular session close.' `
    -Force

[pscustomobject]@{
    TaskName = $RegisteredTask.TaskName
    TaskPath = $RegisteredTask.TaskPath
    State = $RegisteredTask.State
    RunAs = $Identity
    LogonType = 'Interactive; runs while this user is logged on'
    LocalSchedule = 'Tuesday–Saturday 10:15 AM (machine local time)'
    WorkingDirectory = $ProjectRoot
    Action = ('{0} "{1}"' -f $PythonPath, $RefreshScript)
} | Format-List
