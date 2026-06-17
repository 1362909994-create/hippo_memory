param(
  [string]$ProjectRoot = "",
  [string]$ProjectName = "",
  [switch]$NoDeploy,
  [switch]$NoIndex,
  [switch]$Editable,
  [switch]$ForceProjectConfig,
  [switch]$InstallPythonWithWinget
)

$ErrorActionPreference = "Stop"

function Write-Step([string]$Message) {
  Write-Host "[hippo] $Message" -ForegroundColor Cyan
}

function Write-Warn([string]$Message) {
  Write-Host "[hippo] WARNING: $Message" -ForegroundColor Yellow
}

function Find-Python311 {
  $candidates = @(
    @{ File = "py"; PrefixArgs = @("-3.11") },
    @{ File = "python"; PrefixArgs = @() },
    @{ File = "python3"; PrefixArgs = @() }
  )
  foreach ($candidate in $candidates) {
    if (-not (Get-Command $candidate.File -ErrorAction SilentlyContinue)) {
      continue
    }
    $probeArgs = @($candidate.PrefixArgs) + @(
      "-c",
      "import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)"
    )
    try {
      & $candidate.File @probeArgs *> $null
      if ($LASTEXITCODE -eq 0) {
        return $candidate
      }
    } catch {
    }
  }
  return $null
}

function Invoke-SelectedPython([hashtable]$Python, [string[]]$Arguments) {
  $allArgs = @($Python.PrefixArgs) + $Arguments
  & $Python.File @allArgs
}

function Get-HippoCommand([hashtable]$Python) {
  $code = @"
import os
import shutil
import site
import sysconfig

paths = []
for base in (sysconfig.get_path("scripts"), os.path.join(site.USER_BASE, "Scripts")):
    if base:
        for name in ("hippo.exe", "hippo-script.py", "hippo"):
            paths.append(os.path.join(base, name))

for path in paths:
    if os.path.exists(path):
        print(path)
        raise SystemExit(0)

found = shutil.which("hippo")
if found:
    print(found)
    raise SystemExit(0)

raise SystemExit(1)
"@
  $allArgs = @($Python.PrefixArgs) + @("-c", $code)
  $output = & $Python.File @allArgs
  if ($LASTEXITCODE -ne 0 -or -not $output) {
    throw "hippo command was not found after install."
  }
  return ($output | Select-Object -First 1).ToString().Trim()
}

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
if (-not $ProjectRoot) {
  $ProjectRoot = (Get-Location).Path
}
$resolvedProjectRoot = (Resolve-Path -LiteralPath $ProjectRoot).Path

Write-Step "repository: $repoRoot"
Write-Step "target project: $resolvedProjectRoot"

$python = Find-Python311
if (-not $python -and $InstallPythonWithWinget) {
  Write-Step "Python 3.11+ not found; installing Python 3.11 with winget."
  winget install --id Python.Python.3.11 --exact --source winget
  $python = Find-Python311
}

if (-not $python) {
  throw "Python 3.11+ was not found. Install it first, or rerun with -InstallPythonWithWinget."
}

Write-Step "installing hippocampus-memory for current user."
$pipTarget = "$repoRoot[quality,tokens]"
$pipArgs = @("-m", "pip", "install", "--user")
if ($Editable) {
  $pipArgs += "-e"
}
$pipArgs += $pipTarget
Invoke-SelectedPython $python $pipArgs

$hippo = Get-HippoCommand $python
Write-Step "hippo command: $hippo"

if (-not (Get-Command reasonix -ErrorAction SilentlyContinue)) {
  Write-Warn "reasonix was not found on PATH. Install Reasonix before opening a memory-enabled CLI."
}

Write-Step "installing Reasonix global shim and status bar patch."
& $hippo reasonix-install-shim

if (-not $NoDeploy) {
  Write-Step "deploying project-local memory for Reasonix."
  $deployArgs = @("reasonix-deploy", "--root", $resolvedProjectRoot)
  if ($ProjectName) {
    $deployArgs += @("--project", $ProjectName)
  }
  if ($NoIndex) {
    $deployArgs += "--no-index"
  }
  if ($ForceProjectConfig) {
    $deployArgs += "--force-project-config"
  }
  & $hippo @deployArgs
}

Write-Step "done."
Write-Host ""
Write-Host "Open a new Reasonix session with:" -ForegroundColor Green
Write-Host "  reasonix code `"$resolvedProjectRoot`"" -ForegroundColor Green
Write-Host ""
Write-Host "If PowerShell cannot find hippo/reasonix in this window, close and reopen the terminal."
