$workspaceRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$agentApp = Get-Command agent-app -ErrorAction SilentlyContinue

if (-not $agentApp) {
    Write-Error "agent-app command not found. Run 'python -m pip install -e .[dev]' first."
    exit 1
}

$arguments = @($args)
if ($arguments -notcontains "--workspace-root") {
    $arguments += @("--workspace-root", $workspaceRoot)
}

& $agentApp.Source @arguments
exit $LASTEXITCODE
