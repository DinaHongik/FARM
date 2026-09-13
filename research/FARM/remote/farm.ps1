<#
.SYNOPSIS
  Drive FARM experiments on the V100 server from this laptop.

  The code and data must live on the server - CUDA needs them next to the GPU.
  This script is remote control only: it starts jobs there, streams logs back
  here, and pulls results down. Nothing heavy runs locally.

.EXAMPLE
  .\farm.ps1 doctor               # check connection, GPUs, python env, ollama, HF token
  .\farm.ps1 ls                   # list jobs and whether each is running
  .\farm.ps1 run stage1           # launch a job (returns immediately)
  .\farm.ps1 logs stage1 -Follow  # stream its output (Ctrl-C stops watching, not the job)
  .\farm.ps1 pull                 # copy results back to .\results_server\
  .\farm.ps1 stop trainB
  .\farm.ps1 token hf_xxxxxxxx    # store the HuggingFace token needed for training
#>
param(
  [Parameter(Position = 0)][string]$Command = "help",
  [Parameter(Position = 1)][string]$Job = "",
  [switch]$Follow
)

$ErrorActionPreference = "Stop"

# ---------- config ----------
# The .env is deliberately NOT in git. Search the places it plausibly lives so this
# works both from the original layout (FARM_VER_@/.env, i.e. the PARENT of the repo)
# and from a fresh `git clone` where it is easier to drop it inside the repo.
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$EnvCandidates = @(
    (Join-Path $ScriptDir "..\..\.env"),   # FARM_VER_@/.env  (original layout)
    (Join-Path $ScriptDir "..\.env"),      # FARM/.env        (after a bare clone)
    (Join-Path $ScriptDir ".env"),         # FARM/remote/.env
    (Join-Path (Get-Location) ".env")
)
$EnvFile = $EnvCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1

$Cfg = @{
    FARM_SSH_ALIAS      = "aicontents"
    FARM_SSH_HOST       = ""
    FARM_SSH_PORT       = "22"
    FARM_SSH_USER       = ""
    FARM_SSH_PASSWORD   = ""
    FARM_REMOTE_WORKDIR = "/raid/session/aicontents/farm"
}
if ($EnvFile) {
    foreach ($line in Get-Content $EnvFile) {
        if ($line -match '^\s*([A-Z0-9_]+)\s*=\s*(.*?)\s*$') {
            $k = $Matches[1]; $v = $Matches[2].Trim('"').Trim("'")
            if ($Cfg.ContainsKey($k)) { $Cfg[$k] = $v }
        }
    }
}

$ALIAS = $Cfg.FARM_SSH_ALIAS
$WD = $Cfg.FARM_REMOTE_WORKDIR
$RUNNER = "$WD/jobs/run_job.sh"

# Prefer an explicit host:port from .env over the ~/.ssh/config alias. The alias is a
# machine-local file that git cannot carry, so on a freshly cloned machine it does not
# exist - but .env does, because you paste it in. Falls back to the alias when the
# .env has no host (the original setup).
function Get-SshTarget {
    if ($Cfg.FARM_SSH_HOST -and $Cfg.FARM_SSH_USER) {
        return @("-p", $Cfg.FARM_SSH_PORT, "$($Cfg.FARM_SSH_USER)@$($Cfg.FARM_SSH_HOST)")
    }
    return @($ALIAS)
}

function Invoke-Remote {
    param([string]$Cmd)
    $t = Get-SshTarget
    ssh -o BatchMode=yes -o ConnectTimeout=15 -T @t $Cmd
}

# scp needs the port as a separate -P flag (capital P, unlike ssh's -p) and the host
# without it, so the ssh target cannot be reused verbatim.
function Get-ScpArgs {
    if ($Cfg.FARM_SSH_HOST -and $Cfg.FARM_SSH_USER) { return @("-P", $Cfg.FARM_SSH_PORT) }
    return @()
}
function Get-ScpHost {
    if ($Cfg.FARM_SSH_HOST -and $Cfg.FARM_SSH_USER) { return "$($Cfg.FARM_SSH_USER)@$($Cfg.FARM_SSH_HOST)" }
    return $ALIAS
}

function Show-Help {
    Write-Host ""
    Write-Host "  FARM remote control  ->  ${ALIAS}:${WD}" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "    setup                  ONE TIME on a new machine: install an SSH key from .env"
    Write-Host "    doctor                 check connection, GPUs, env, ollama, HF token"
    Write-Host "    progress               one-shot snapshot of every running job (safe to repeat)"
    Write-Host "    watch [sec]            refresh that snapshot until Ctrl-C"
    Write-Host "    gpu                    GPU utilisation right now"
    Write-Host "    ls                     list jobs + running state"
    Write-Host "    run <job>              launch a job (survives you closing this window)"
    Write-Host "    logs <job> [-Follow]   show or stream a job's output"
    Write-Host "    stop <job>             kill a running job"
    Write-Host "    pull                   copy remote results/ -> .\results_server\"
    Write-Host "    push <localfile>       upload a file into the remote workdir"
    Write-Host "    sync                   re-upload the job scripts in this folder"
    Write-Host "    token <hf_token>       store a HuggingFace token on the server"
    Write-Host "    shell                  interactive ssh session"
    Write-Host ""
    Write-Host "  Run '.\farm.ps1 ls' to see available jobs." -ForegroundColor DarkGray
    Write-Host ""
}

switch ($Command.ToLower()) {

    "doctor" { Invoke-Remote "bash $WD/jobs/doctor.sh" }

    "gpu" { Invoke-Remote "nvidia-smi" }

    # One-shot snapshot: safe to run repeatedly, returns immediately, no Ctrl-C needed.
    "progress" { Invoke-Remote "bash $WD/jobs/progress.sh" }

    # Refresh the snapshot every N seconds until you press Ctrl-C.
    # Ctrl-C here only stops the display - the jobs on the server keep running.
    "watch" {
        $sec = 30
        if ($Job -and ($Job -as [int])) { $sec = [int]$Job }
        Write-Host "refreshing every ${sec}s - Ctrl-C stops the DISPLAY only, jobs keep running" -ForegroundColor DarkGray
        while ($true) {
            Clear-Host
            Invoke-Remote "bash $WD/jobs/progress.sh"
            Start-Sleep -Seconds $sec
        }
    }

    "ls" { Invoke-Remote "bash $RUNNER status" }

    "run" {
        if (-not $Job) { Write-Host "usage: .\farm.ps1 run <job>   (see: .\farm.ps1 ls)" -ForegroundColor Red; exit 1 }
        Invoke-Remote "bash $RUNNER start $Job"
        Write-Host "stream it with:  .\farm.ps1 logs $Job -Follow" -ForegroundColor DarkGray
    }

    "logs" {
        if (-not $Job) { Write-Host "usage: .\farm.ps1 logs <job> [-Follow]" -ForegroundColor Red; exit 1 }
        if ($Follow) {
            Write-Host "streaming '$Job'  (Ctrl-C stops watching; the job keeps running)" -ForegroundColor Cyan
            Invoke-Remote "bash $RUNNER follow $Job"
        }
        else {
            Invoke-Remote "bash $RUNNER log $Job 80"
        }
    }

    "stop" {
        if (-not $Job) { Write-Host "usage: .\farm.ps1 stop <job>" -ForegroundColor Red; exit 1 }
        Invoke-Remote "bash $RUNNER stop $Job"
    }

    "pull" {
        $dest = Join-Path $ScriptDir "..\results_server"
        New-Item -ItemType Directory -Force -Path $dest | Out-Null
        Write-Host "pulling ${WD}/results/ -> $dest" -ForegroundColor Cyan
        $s = Get-ScpArgs
        scp -o BatchMode=yes @s -r "$(Get-ScpHost):${WD}/results/." $dest
        Write-Host "done." -ForegroundColor Green
        $fc = Join-Path $dest "FINAL_COMPARISON.md"
        if (Test-Path $fc) {
            Write-Host ""
            Write-Host "  FINAL_COMPARISON.md is here:" -ForegroundColor Green
            Write-Host "  $fc" -ForegroundColor Green
        }
        Get-ChildItem $dest -File | Sort-Object LastWriteTime -Descending |
            Select-Object -First 15 Name, Length, LastWriteTime | Format-Table -AutoSize
    }

    "push" {
        if (-not $Job) { Write-Host "usage: .\farm.ps1 push <localfile>" -ForegroundColor Red; exit 1 }
        if (-not (Test-Path $Job)) { Write-Host "no such file: $Job" -ForegroundColor Red; exit 1 }
        $s = Get-ScpArgs
        scp -o BatchMode=yes @s $Job "$(Get-ScpHost):${WD}/"
        Write-Host "uploaded $(Split-Path -Leaf $Job)" -ForegroundColor Green
    }

    "sync" {
        Invoke-Remote "mkdir -p $WD/jobs"
        $s = Get-ScpArgs
        $h = Get-ScpHost
        Get-ChildItem $ScriptDir -Filter "*.sh"  | ForEach-Object { scp -o BatchMode=yes @s $_.FullName "${h}:${WD}/jobs/" }
        Get-ChildItem $ScriptDir -Filter "*.py"  | ForEach-Object { scp -o BatchMode=yes @s $_.FullName "${h}:${WD}/jobs/" }
        Invoke-Remote "chmod +x $WD/jobs/*.sh; ls -la $WD/jobs/"
        Write-Host "job scripts synced." -ForegroundColor Green
    }

    "token" {
        if (-not $Job) { Write-Host "usage: .\farm.ps1 token hf_xxxxxxxx" -ForegroundColor Red; exit 1 }
        Invoke-Remote "bash $WD/jobs/set_token.sh $Job"
    }

    # One-time bootstrap on a machine that has never talked to the server.
    # Creates an ed25519 key if there is none and installs the PUBLIC half into the
    # server's authorized_keys using the password from .env, so every later command
    # can run with BatchMode=yes (no prompts, works unattended).
    "setup" {
        if (-not $EnvFile) {
            Write-Host "no .env found. Looked in:" -ForegroundColor Red
            $EnvCandidates | ForEach-Object { Write-Host "  $_" -ForegroundColor DarkGray }
            Write-Host "Create one (see FARM/.env.example) and re-run." -ForegroundColor Red
            exit 1
        }
        Write-Host "using .env: $EnvFile" -ForegroundColor DarkGray
        foreach ($k in @("FARM_SSH_HOST", "FARM_SSH_USER", "FARM_SSH_PASSWORD")) {
            if (-not $Cfg[$k]) { Write-Host "$k is not set in $EnvFile" -ForegroundColor Red; exit 1 }
        }

        $key = Join-Path $HOME ".ssh\id_ed25519"
        if (-not (Test-Path $key)) {
            Write-Host "generating $key" -ForegroundColor Cyan
            New-Item -ItemType Directory -Force -Path (Split-Path $key) | Out-Null
            ssh-keygen -t ed25519 -N '""' -f $key -C "farm-remote"
        }
        $pub = (Get-Content "$key.pub" -Raw).Trim()

        # plink carries the password non-interactively; OpenSSH on Windows cannot.
        $plink = Get-Command plink -ErrorAction SilentlyContinue
        if (-not $plink) {
            Write-Host ""
            Write-Host "plink not found - install PuTTY, or run this once by hand and type the password:" -ForegroundColor Yellow
            Write-Host "  type `"$key.pub`" | ssh -p $($Cfg.FARM_SSH_PORT) $($Cfg.FARM_SSH_USER)@$($Cfg.FARM_SSH_HOST) `"mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys`"" -ForegroundColor Yellow
            exit 1
        }
        $inner = "mkdir -p ~/.ssh && chmod 700 ~/.ssh && grep -qxF '$pub' ~/.ssh/authorized_keys 2>/dev/null || echo '$pub' >> ~/.ssh/authorized_keys; chmod 600 ~/.ssh/authorized_keys; echo KEY_OK"
        plink -batch -ssh -P $Cfg.FARM_SSH_PORT -l $Cfg.FARM_SSH_USER -pw $Cfg.FARM_SSH_PASSWORD $Cfg.FARM_SSH_HOST $inner

        Write-Host ""
        Write-Host "verifying key auth..." -ForegroundColor Cyan
        Invoke-Remote "echo CONNECTED as \$(whoami) on \$(hostname)"
        Write-Host ""
        Write-Host "setup done. Now run:  .\farm.ps1 progress" -ForegroundColor Green
    }

    "shell" {
        $t = Get-SshTarget
        ssh @t
    }

    default { Show-Help }
}
