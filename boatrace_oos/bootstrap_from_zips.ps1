$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$files = @(
    (Join-Path $PSScriptRoot "data_zips\race_2023.zip"),
    (Join-Path $PSScriptRoot "data_zips\race_2024.zip"),
    (Join-Path $PSScriptRoot "data_zips\race.zip")
)

foreach ($f in $files) {
    if (-not (Test-Path $f)) { throw "Missing required zip: $f" }
}

$resolved = $files | ForEach-Object { (Resolve-Path $_).Path }
$env:LOCAL_ZIPS = ($resolved -join ";")

& .\.venv\Scripts\python.exe .\bootstrap_frozen.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& .\.venv\Scripts\python.exe .\strength_local_notifier.py --self-check
exit $LASTEXITCODE
