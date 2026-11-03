# Download nuScenes v1.0-trainval (official Motional CDN, same files as the nuscenes.org download page) and keep
# ONLY what the LiDAR-only lane needs: the metadata tables + samples/LIDAR_TOP + sweeps/LIDAR_TOP.
# (Camera/radar files are skipped; the full set is 293 GB compressed, the LiDAR subset is far smaller.)
# Per archive: resumable download -> filtered extract -> delete the archive.  Re-running skips finished archives.
# By downloading you accept the nuScenes terms of use (non-commercial research): https://www.nuscenes.org/terms-of-use
#
#   powershell -File lidar\scripts\fetch_nuscenes_trainval.ps1 -Dest C:\Users\Acer\datasets\nuscenes
param([string]$Dest = "C:\Users\Acer\datasets\nuscenes", [string]$Tmp = "C:\Users\Acer\datasets\_dl")
$base = "https://d36yt3mvayqw5m.cloudfront.net/public/v1.0"
New-Item -ItemType Directory -Force $Dest, $Tmp | Out-Null
$done = Join-Path $Dest ".fetched_trainval.txt"
if (-not (Test-Path $done)) { New-Item $done -ItemType File | Out-Null }

function Fetch($name, $patterns) {
    if (Select-String -Path $done -Pattern "^$name$" -Quiet) { Write-Host "skip $name (done)"; return }
    $f = Join-Path $Tmp "$name.tgz"
    Write-Host "$(Get-Date -Format s) download $name"
    for ($try = 0; $try -lt 20; $try++) {
        curl.exe -L -C - --retry 5 --retry-delay 10 -sS -o $f "$base/$name.tgz"
        if ($LASTEXITCODE -eq 0) { break }
        Start-Sleep 15
    }
    if ($LASTEXITCODE -ne 0) { Write-Host "FAILED download $name"; return }
    Write-Host "$(Get-Date -Format s) extract $name"
    Push-Location $Dest
    if ($patterns) { tar -xzf $f @patterns } else { tar -xzf $f }
    $rc = $LASTEXITCODE
    Pop-Location
    if ($rc -ne 0) { Write-Host "FAILED extract $name (tar exit $rc)"; return }
    Remove-Item $f -Force
    Add-Content $done $name
    Write-Host "$(Get-Date -Format s) done $name"
}

Fetch "v1.0-trainval_meta" $null
foreach ($i in 1..10) {
    Fetch ("v1.0-trainval{0:00}_blobs" -f $i) @("samples/LIDAR_TOP", "sweeps/LIDAR_TOP")
}
Write-Host "ALL DONE"
