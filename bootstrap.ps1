function Show-Help {
    Write-Host "Usage: .\bootstrap.ps1 [command]"
    Write-Host ""
    Write-Host "Commands:"
    Write-Host "  init    Initialize local directories (logs, redis_data)"
    Write-Host "  build   Build the containers using $env:COMPOSE_ENGINE"
    Write-Host "  start   Start the containers using $env:COMPOSE_ENGINE (attached for CLI interaction)"
    Write-Host "  stop    Stop and remove containers using $env:COMPOSE_ENGINE"
}

# Determine the compose command
$ComposeEngine = $env:COMPOSE_ENGINE
if ([string]::IsNullOrEmpty($ComposeEngine)) {
    $ComposeEngine = "docker" # Default to docker if not set
}

$ComposeCommand = ""
if ($ComposeEngine -eq "podman") {
    $ComposeCommand = "podman compose"
} elseif ($ComposeEngine -eq "docker") {
    $ComposeCommand = "docker compose"
} else {
    Write-Error "Invalid COMPOSE_ENGINE value '$ComposeEngine'. Must be 'docker' or 'podman'."
    exit 1
}

$action = $args[0]

switch ($action) {
    "init" {
        Write-Host "Initializing directories..."
        if (!(Test-Path -Path "logs")) { New-Item -ItemType Directory -Path "logs" }
        if (!(Test-Path -Path "redis_data")) { New-Item -ItemType Directory -Path "redis_data" }
        Write-Host "Done."
    }
    "build" {
        Write-Host "Building containers with $ComposeEngine..."
        Invoke-Expression "$ComposeCommand build"
    }
    "start" {
        Write-Host "Starting containers with $ComposeEngine..."
        Invoke-Expression "$ComposeCommand up"
    }
    "stop" {
        Write-Host "Stopping containers with $ComposeEngine..."
        Invoke-Expression "$ComposeCommand down"
    }
    "shell" {
        Write-Host "Entering container shell with $ComposeEngine..."
        Invoke-Expression "$ComposeCommand exec cknot-app /bin/bash"
    }
    Default {
        Show-Help
    }
}