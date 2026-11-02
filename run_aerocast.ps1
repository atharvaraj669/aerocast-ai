# AeroCast-AI PowerShell Launcher
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$pythonExe = Join-Path $scriptDir ".venv\Scripts\python.exe"

if (-not (Test-Path $pythonExe)) {
    $pythonExe = "python"
}

& $pythonExe (Join-Path $scriptDir "train_and_predict.py") @args
