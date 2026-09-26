# Connect to the EC2 instance, make sure JupyterLab is running there, open the SSH
# tunnel and launch JupyterLab in the browser.
#
# Usage (from any PowerShell window):
#   powershell -ExecutionPolicy Bypass -File "<path>\connect_ec2.ps1"
#   powershell -ExecutionPolicy Bypass -File "<path>\connect_ec2.ps1" -Ec2Dns ec2-NEW-DNS.ap-south-1.compute.amazonaws.com
#
# The public DNS changes every time the instance is stopped/started: pass the new one
# with -Ec2Dns, or update the default below.
param(
    [string]$Ec2Dns = "ec2-35-154-196-187.ap-south-1.compute.amazonaws.com",
    [int]$Port = 8888,
    [string]$KeyPath = "$env:USERPROFILE\.ssh\ber-key.pem"
)

$ErrorActionPreference = "Stop"
$EC2 = "ubuntu@$Ec2Dns"

if (-not (Test-Path $KeyPath)) { throw "Key file not found: $KeyPath" }

# 1. Key permissions (OpenSSH refuses keys readable by other users). Safe to repeat.
icacls $KeyPath /inheritance:r | Out-Null
icacls $KeyPath /grant:r "$($env:USERNAME):(R)" | Out-Null

$sshOpts = @("-i", $KeyPath, "-o", "StrictHostKeyChecking=accept-new",
             "-o", "ServerAliveInterval=60", "-o", "ConnectTimeout=20")

# 2. On the instance: start JupyterLab inside tmux (only if not already running) and
#    print its URL with token.
$remote = @'
source ~/ber-venv/bin/activate
if ! tmux has-session -t jlab 2>/dev/null; then
  tmux new -d -s jlab "source ~/ber-venv/bin/activate && cd ~/business_entity_resolution && jupyter lab --no-browser --port __PORT__ --ip 127.0.0.1"
  sleep 8
fi
jupyter server list 2>/dev/null | grep -o "http://127.0.0.1:__PORT__/[^ ]*" | head -1
'@
$remote = ($remote -replace "`r", "") -replace "__PORT__", $Port
# Windows argument passing mangles quotes, so ship the script base64-encoded.
$b64 = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($remote))

Write-Host "Connecting to $EC2 and checking JupyterLab..." -ForegroundColor Cyan
$url = (& ssh @sshOpts $EC2 "echo $b64 | base64 -d | bash" | Select-Object -Last 1)
if (-not $url) {
    throw "Could not get the JupyterLab URL. SSH in and run: tmux attach -t jlab"
}
Write-Host "JupyterLab: $url" -ForegroundColor Green

# 3. SSH tunnel in its own window (keep that window open while you work).
#    Launched via WMI so it is NOT a child of this window and survives it closing.
$portBusy = (Test-NetConnection 127.0.0.1 -Port $Port -WarningAction SilentlyContinue).TcpTestSucceeded
if ($portBusy) {
    Write-Host "Port $Port already forwarded (existing tunnel window) - reusing it." -ForegroundColor Yellow
} else {
    $tunnel = "ssh -i '$KeyPath' -o ServerAliveInterval=60 -N -L ${Port}:localhost:${Port} $EC2"
    $cmd = "powershell.exe -NoExit -Command `"Write-Host 'SSH tunnel for JupyterLab - keep this window open' -ForegroundColor Yellow; $tunnel`""
    Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = $cmd } | Out-Null
    Start-Sleep -Seconds 5
}

# 4. Open JupyterLab in the default browser.
Start-Process $url

# 5. Interactive shell on the instance in this window (type 'exit' to leave;
#    Jupyter keeps running in tmux).
Write-Host "Opening an SSH shell on the instance (type 'exit' to leave)..." -ForegroundColor Cyan
& ssh @sshOpts $EC2
