# Live progress dashboard for the pipeline running on EC2.
#
#   powershell -ExecutionPolicy Bypass -File "<path>\dashboard.ps1"
#   powershell -ExecutionPolicy Bypass -File "<path>\dashboard.ps1" -Ec2Dns ec2-NEW.ap-south-1.compute.amazonaws.com
#
# Polls the instance every -Interval seconds over SSH (one short command, no load on the run)
# and shows: per-notebook status (done / running / stopped), a progress bar with live ETA for
# the active step, candidate / pruning / scoring files, the submission files, machine health.
# Ctrl+C to quit — the run on EC2 is not affected.
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
    "00_preprocessing" = "Cleaning (train)"; "01_eda" = "EDA"; "02_blocking_candidate_generation" = "Candidates + pruning (train)";
    "03_feature_engineering" = "Features (train)"; "04_model_training" = "Train + validate"; "05_inference_submission" = "Test: clean / candidates / prune / score / submit"
}
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
function Utc([double]$epoch) { return ([DateTimeOffset]::FromUnixTimeSeconds([long]$epoch)).UtcDateTime.ToString("HH:mm") + " UTC" }

while ($true) {
    $raw = & ssh @sshOpts $EC2 "bash ~/business_entity_resolution/scripts/ber_status.sh" 2>$null
    Clear-Host
    Write-Host "  Business Entity Resolution - EC2 pipeline dashboard" -ForegroundColor Cyan
    Write-Host ("  " + (Get-Date -Format "yyyy-MM-dd HH:mm:ss") + " local   instance: $Ec2Dns   (refresh ${Interval}s, Ctrl+C to quit)") -ForegroundColor DarkGray
    Write-Host ""
    if (-not $raw) {
        Write-Host "  Cannot reach the instance (stopped? new Public DNS? IP changed in the security group?)" -ForegroundColor Red
        Start-Sleep -Seconds $Interval; continue
    }
    $kv = @{}; $logs = @(); $cands = @(); $outs = @()
    foreach ($line in $raw) {
        $i = $line.IndexOf("="); if ($i -lt 0) { continue }
        $k = $line.Substring(0, $i); $v = $line.Substring($i + 1)
        switch ($k) { "log" { $logs += $v } "cand" { $cands += $v } "out" { $outs += $v } default { $kv[$k] = $v } }
    }
    $now = [double]$kv["now"]
    $alive = ($kv["tmux"] -eq "running") -or ($kv["nbconvert"] -eq "alive")

    # ---------------- notebooks ----------------
    Write-Host "  NOTEBOOKS" -ForegroundColor Yellow
    foreach ($nb in $notebooks.Keys) {
        $start = $logs | Where-Object { $_ -match "START $nb" } | Select-Object -Last 1
        $done = $logs | Where-Object { $_ -match "DONE  $nb" } | Select-Object -Last 1
        $startIdx = [array]::LastIndexOf($logs, $start); $doneIdx = [array]::LastIndexOf($logs, $done)
        if ($done -and ($doneIdx -gt $startIdx)) {
            $secs = if ($done -match "in (\d+)s") { [int]$Matches[1] } else { 0 }
            Write-Host ("   [done]     {0,-48} {1}" -f $notebooks[$nb], (Fmt $secs)) -ForegroundColor Green
        } elseif ($start) {
            $t0 = [DateTimeOffset]::Parse(($start -split " ")[0]).ToUnixTimeSeconds()
            $el = $now - $t0
            if ($alive) {
                Write-Host ("   [running]  {0,-48} {1} elapsed" -f $notebooks[$nb], (Fmt $el)) -ForegroundColor White
            } else {
                Write-Host ("   [stopped]  {0,-48} runner exited without DONE (see error below)" -f $notebooks[$nb]) -ForegroundColor Red
            }
        } else {
            Write-Host ("   [ ]        {0}" -f $notebooks[$nb]) -ForegroundColor DarkGray
        }
    }
    Write-Host ""

    # ---------------- current step ----------------
    Write-Host "  CURRENT STEP" -ForegroundColor Yellow
    $shown = $false
    if ($alive -and $kv.ContainsKey("progress")) {
        try {
            $p = $kv["progress"] | ConvertFrom-Json
            if (($now - $p.ts) -lt 3600) {
                $frac = $p.done / [double]$p.total
                $key = "p:" + $p.stage
                if (-not $history.ContainsKey($key) -or $history[$key][1] -gt $frac) { $history[$key] = @($now, $frac) }
                $h = $history[$key]; $rate = ($frac - $h[1]) / [math]::Max(1, $now - $h[0])
                $eta = if ($rate -gt 0) { (1 - $frac) / $rate } else { -1 }
                Write-Host ("   {0}" -f $p.stage)
                Write-Host ("   {0} {1,6:P1}   {2:N0}/{3:N0}   ETA {4}   ({5})" -f (Bar $frac), $frac, $p.done, $p.total, (Fmt $eta), $p.detail)
                $shown = $true
            }
        } catch {}
    }
    if (-not $shown) {
        if ($alive) { Write-Host "   (running - waiting for the next progress report)" -ForegroundColor DarkGray }
        else { Write-Host "   idle - no pipeline is running on the instance" -ForegroundColor DarkGray }
    }
    Write-Host ""

    # ---------------- stage files ----------------
    Write-Host "  STAGE FILES" -ForegroundColor Yellow
    foreach ($c in $cands) {
        $name, $size = $c -split ":"
        $state = if ($name -like "*.tmp") { "writing" } else { "done" }
        $col = if ($state -eq "done") { "Gray" } else { "White" }
        Write-Host ("   {0,-40} {1,8:N0} MB  {2}" -f ($name -replace "\.(tmp|parquet)$", ""), ([double]$size / 1MB), $state) -ForegroundColor $col
    }
    if ($kv.ContainsKey("scoreparts")) { Write-Host ("   scores/test                              {0} parts scored" -f $kv["scoreparts"]) -ForegroundColor Gray }
    Write-Host ""

    # ---------------- submission ----------------
    Write-Host "  SUBMISSION (~/business_entity_resolution/output)" -ForegroundColor Yellow
    if ($outs.Count -eq 0) { Write-Host "   not written yet" -ForegroundColor DarkGray }
    foreach ($o in $outs) {
        $name, $size, $mt = $o -split ":"
        Write-Host ("   {0,-24} {1,9:N1} MB   written {2}" -f $name, ([double]$size / 1MB), (Utc $mt)) -ForegroundColor Green
    }
    Write-Host ""

    # ---------------- machine ----------------
    Write-Host "  MACHINE" -ForegroundColor Yellow
    $mem = $kv["mem"] -split "/"
    if ($mem.Count -eq 2) { Write-Host ("   RAM   {0} {1:N1} / {2:N1} GB" -f (Bar ([double]$mem[0] / [double]$mem[1]) 30), ([double]$mem[0] / 1024), ([double]$mem[1] / 1024)) }
    $disk = ($kv["disk"] -replace "G", "") -split "/"
    if ($disk.Count -eq 2) { Write-Host ("   Disk  {0} {1} / {2} GB" -f (Bar ([double]$disk[0] / [double]$disk[1]) 30), $disk[0], $disk[1]) }
    Write-Host ("   CPU load {0} (2 vCPU)   runner: {1}   notebook process: {2}" -f $kv["load"], $kv["tmux"], $kv["nbconvert"])
    if ($kv.ContainsKey("oom_ago")) {
        $ago = [double]$kv["oom_ago"]
        $col = if ($ago -lt 3 * 3600) { "Red" } else { "DarkGray" }
        Write-Host ("   last out-of-memory kill: {0} ago" -f (Fmt $ago)) -ForegroundColor $col
    }
    if ($kv.ContainsKey("error")) { Write-Host ("`n   ERROR: " + $kv["error"]) -ForegroundColor Red }
    if ($kv.ContainsKey("alldone")) { Write-Host "`n   ALL NOTEBOOKS DONE" -ForegroundColor Green }
    Start-Sleep -Seconds $Interval
}
