# Live progress dashboard for the pipeline running on EC2.
#
#   powershell -ExecutionPolicy Bypass -File "<path>\dashboard.ps1"
#   powershell -ExecutionPolicy Bypass -File "<path>\dashboard.ps1" -Ec2Dns ec2-NEW.ap-south-1.compute.amazonaws.com
#
# Polls the instance every -Interval seconds over SSH (one short command, no load on the
# run) and draws per-notebook status, a progress bar for the active step with a live ETA,
# and machine health. Ctrl+C to quit — the run on EC2 is not affected.
param(
    [string]$Ec2Dns = "ec2-35-154-196-187.ap-south-1.compute.amazonaws.com",
    [int]$Interval = 20,
    [string]$KeyPath = "$env:USERPROFILE\.ssh\ber-key.pem"
)
$ErrorActionPreference = "Continue"
$EC2 = "ubuntu@$Ec2Dns"
$sshOpts = @("-i", $KeyPath, "-o", "ConnectTimeout=15", "-o", "StrictHostKeyChecking=accept-new", "-o", "BatchMode=yes")
[Console]::OutputEncoding = [Text.Encoding]::UTF8

$notebooks = [ordered]@{
    "00_preprocessing" = "Cleaning (train)"; "01_eda" = "EDA"; "02_blocking_candidate_generation" = "Candidates (train)";
    "03_feature_engineering" = "Features (train)"; "04_model_training" = "Train + validate"; "05_inference_submission" = "Test: clean/candidates/score/submit"
}
# Fallback size estimate for candidate files (~12.2 bytes/pair, 15 pairs/record) when no progress.json yet
$expectedPairs = @{ "train/india" = 4133346 * 15; "train/us" = 6186873 * 15 }
$history = @{}

function Bar([double]$frac, [int]$width = 40) {
    $frac = [math]::Max(0, [math]::Min(1, $frac))
    $full = [int][math]::Floor($frac * $width)
    return ("[" + ([string][char]0x2588) * $full + ([string][char]0x2591) * ($width - $full) + "]")
}
function Fmt([double]$sec) {
    if ($sec -lt 0 -or [double]::IsInfinity($sec) -or [double]::IsNaN($sec)) { return "--" }
    $ts = [TimeSpan]::FromSeconds($sec); return ("{0}h {1:D2}m" -f [int]$ts.TotalHours, $ts.Minutes)
}

while ($true) {
    $raw = & ssh @sshOpts $EC2 "bash ~/business_entity_resolution/scripts/ber_status.sh" 2>$null
    Clear-Host
    Write-Host "  Business Entity Resolution - EC2 pipeline dashboard" -ForegroundColor Cyan
    Write-Host ("  " + (Get-Date -Format "yyyy-MM-dd HH:mm:ss") + "   instance: $Ec2Dns   (refresh ${Interval}s, Ctrl+C to quit)") -ForegroundColor DarkGray
    Write-Host ""
    if (-not $raw) {
        Write-Host "  Cannot reach the instance (stopped? new Public DNS? IP changed in the security group?)" -ForegroundColor Red
        Start-Sleep -Seconds $Interval; continue
    }
    $kv = @{}; $logs = @(); $cands = @()
    foreach ($line in $raw) {
        $i = $line.IndexOf("="); if ($i -lt 0) { continue }
        $k = $line.Substring(0, $i); $v = $line.Substring($i + 1)
        switch ($k) { "log" { $logs += $v } "cand" { $cands += $v } default { $kv[$k] = $v } }
    }
    $now = [double]$kv["now"]

    # ---- notebook table ----
    Write-Host "  NOTEBOOKS" -ForegroundColor Yellow
    foreach ($nb in $notebooks.Keys) {
        $start = $logs | Where-Object { $_ -match "START $nb" } | Select-Object -Last 1
        $done = $logs | Where-Object { $_ -match "DONE  $nb" } | Select-Object -Last 1
        if ($done) {
            $secs = if ($done -match "in (\d+)s") { [int]$Matches[1] } else { 0 }
            Write-Host ("   [done]    {0,-36} {1}" -f $notebooks[$nb], (Fmt $secs)) -ForegroundColor Green
        } elseif ($start) {
            $t0 = [datetime]::Parse(($start -split " ")[0]).ToUniversalTime()
            $el = $now - ([DateTimeOffset]$t0).ToUnixTimeSeconds()
            Write-Host ("   [running] {0,-36} {1} elapsed" -f $notebooks[$nb], (Fmt $el)) -ForegroundColor White
        } else {
            Write-Host ("   [ ]       {0}" -f $notebooks[$nb]) -ForegroundColor DarkGray
        }
    }
    Write-Host ""

    # ---- active step progress ----
    Write-Host "  CURRENT STEP" -ForegroundColor Yellow
    $shown = $false
    if ($kv.ContainsKey("progress")) {
        try {
            $p = $kv["progress"] | ConvertFrom-Json
            $age = $now - $p.ts
            if ($age -lt 1800) {
                $frac = $p.done / [double]$p.total
                $key = "p:" + $p.stage
                if (-not $history.ContainsKey($key)) { $history[$key] = @($now, $frac) }
                $h = $history[$key]; $rate = ($frac - $h[1]) / [math]::Max(1, $now - $h[0])
                $eta = if ($rate -gt 0) { (1 - $frac) / $rate } else { -1 }
                Write-Host ("   {0}" -f $p.stage)
                Write-Host ("   {0} {1,6:P1}   {2:N0}/{3:N0}   ETA {4}   ({5})" -f (Bar $frac), $frac, $p.done, $p.total, (Fmt $eta), $p.detail)
                $shown = $true
            }
        } catch {}
    }
    if (-not $shown) {
        foreach ($c in $cands) {
            $name, $size = $c -split ":"
            $blk = $name -replace "\.(tmp|parquet)$", ""
            if ($name -like "*.parquet") { Write-Host ("   candidates {0,-14} {1} done" -f $blk, (Bar 1)) -ForegroundColor Green; continue }
            if ($expectedPairs.ContainsKey($blk)) {
                $frac = [double]$size / ($expectedPairs[$blk] * 12.2)
                $key = "c:" + $blk
                if (-not $history.ContainsKey($key)) { $history[$key] = @($now, $frac) }
                $h = $history[$key]; $rate = ($frac - $h[1]) / [math]::Max(1, $now - $h[0])
                $eta = if ($rate -gt 0) { (1 - $frac) / $rate } else { -1 }
                Write-Host ("   candidates {0,-14} {1} ~{2,6:P1}   ETA {3}   ({4:N0} MB, estimated from file size)" -f $blk, (Bar $frac), [math]::Min($frac, 0.99), (Fmt $eta), ([double]$size / 1MB))
            } else {
                Write-Host ("   candidates {0,-14} {1:N0} MB written" -f $blk, ([double]$size / 1MB))
            }
            $shown = $true
        }
    }
    if (-not $shown) { Write-Host "   (waiting for the next step to report progress)" -ForegroundColor DarkGray }
    Write-Host ""

    # ---- health ----
    Write-Host "  MACHINE" -ForegroundColor Yellow
    $mem = $kv["mem"] -split "/"
    if ($mem.Count -eq 2) { Write-Host ("   RAM   {0} {1:N1} / {2:N1} GB" -f (Bar ([double]$mem[0] / [double]$mem[1]) 30), ([double]$mem[0] / 1024), ([double]$mem[1] / 1024)) }
    $disk = ($kv["disk"] -replace "G", "") -split "/"
    if ($disk.Count -eq 2) { Write-Host ("   Disk  {0} {1} / {2} GB" -f (Bar ([double]$disk[0] / [double]$disk[1]) 30), $disk[0], $disk[1]) }
    Write-Host ("   CPU load {0} (2 vCPU)   runner: {1}" -f $kv["load"], $kv["tmux"])
    if ($kv.ContainsKey("error")) { Write-Host ("`n   ERROR: " + $kv["error"]) -ForegroundColor Red }
    if ($kv.ContainsKey("alldone")) { Write-Host "`n   ALL NOTEBOOKS DONE - submission files are in ~/business_entity_resolution/output/" -ForegroundColor Green }
    Start-Sleep -Seconds $Interval
}
