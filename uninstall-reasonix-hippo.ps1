param(
  [string]$ProjectRoot = "",
  [switch]$RemoveProjectData,
  [switch]$RemoveProjectMemory,
  [switch]$UninstallPackage
)

$ErrorActionPreference = "Stop"

function Write-Step([string]$Message) {
  Write-Host "[hippo] $Message" -ForegroundColor Cyan
}

function Find-Python {
  foreach ($name in @("py", "python", "python3")) {
    if (Get-Command $name -ErrorAction SilentlyContinue) {
      if ($name -eq "py") {
        return @{ File = "py"; PrefixArgs = @("-3") }
      }
      return @{ File = $name; PrefixArgs = @() }
    }
  }
  return $null
}

function Invoke-SelectedPython([hashtable]$Python, [string[]]$Arguments) {
  $allArgs = @($Python.PrefixArgs) + $Arguments
  & $Python.File @allArgs
}

function Get-HippoCommand([hashtable]$Python) {
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
  throw "hippo command was not found. If already uninstalled, there is nothing left for this script to remove."
}

if (-not $ProjectRoot) {
  $ProjectRoot = (Get-Location).Path
}
$resolvedProjectRoot = (Resolve-Path -LiteralPath $ProjectRoot).Path

$python = Find-Python
if (-not $python) {
  throw "Python was not found. Restore Reasonix manually from *.hippo-original files if needed."
}

$hippo = Get-HippoCommand $python
Write-Step "hippo command: $hippo"
Write-Step "removing Reasonix integration."

$argsList = @("reasonix-uninstall", "--root", $resolvedProjectRoot)
if ($RemoveProjectData) {
  $argsList += "--remove-project-data"
}
if ($RemoveProjectMemory) {
  $argsList += "--remove-project-memory"
}
& $hippo @argsList

if ($UninstallPackage) {
  Write-Step "uninstalling hippocampus-memory from this Python environment."
  Invoke-SelectedPython $python @("-m", "pip", "uninstall", "-y", "hippocampus-memory")
}

Write-Step "done."
