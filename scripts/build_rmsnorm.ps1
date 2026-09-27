[CmdletBinding()]
param(
    [string]$Python = (
        Join-Path (Split-Path -Parent $PSScriptRoot) '.venv\Scripts\python.exe'
    )
)

$ErrorActionPreference = 'Stop'

$Root = Split-Path -Parent $PSScriptRoot
$VsDevCmd = 'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat'

if (!(Test-Path $VsDevCmd)) {
    throw "VS developer environment not found: $VsDevCmd"
}

if (!(Test-Path $Python)) {
    throw "64-bit venv Python not found: $Python"
}

$ExtensionDir = Join-Path $Root 'cuda_extensions\rmsnorm'

$ExtensionOutput = Join-Path $Root 'build\python'
New-Item -ItemType Directory -Force -Path $ExtensionOutput | Out-Null
$env:TORCH_CUDA_ARCH_LIST = '7.5'
$env:DISTUTILS_USE_SDK = '1'
$command = 'call "{0}" -arch=x64 -host_arch=x64 && "{1}" setup.py build_ext --inplace' -f $VsDevCmd, $Python

Push-Location $ExtensionDir

try {
    & cmd.exe /d /c $command
    if ($LASTEXITCODE -ne 0) {
        throw "rmsnorm extension build failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}

Copy-Item `
    (Join-Path $ExtensionDir 'rmsnorm_cuda*.pyd') `
    $ExtensionOutput `
    -Force

Write-Output 'Built and copied rmsnorm_cuda into build\python\'
