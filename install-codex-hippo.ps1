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

function Get-HippoCommand {
  $cmd = Get-Command hippo -ErrorAction SilentlyContinue
  if ($cmd) {
    return $cmd.Source
  }
  $roots = @(
    (Join-Path $env:LOCALAPPDATA "Programs\Python"),
    (Join-Path $env:APPDATA "Python")
  )
  foreach ($root in $roots) {
    if (-not (Test-Path -LiteralPath $root)) {
      continue
    }
    $found = Get-ChildItem -LiteralPath $root -Recurse -Filter "hippo.exe" -ErrorAction SilentlyContinue |
      Where-Object { $_.FullName -match "\\Scripts\\hippo\.exe$" } |
      Select-Object -First 1
    if ($found) {
      return $found.FullName
    }
  }
  throw "hippo command was not found after install. Close and reopen the terminal, then retry."
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

$hippo = Get-HippoCommand
Write-Step "hippo command: $hippo"

if (-not $NoDeploy) {
  Write-Step "deploying project-local memory for Codex."
  $deployArgs = @("codex-deploy", "--root", $resolvedProjectRoot)
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
Write-Host "Open Codex in this project and keep AGENTS.md available to the session:" -ForegroundColor Green
Write-Host "  $resolvedProjectRoot" -ForegroundColor Green
